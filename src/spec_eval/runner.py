"""Eval orchestrator: server lifecycle + per-cell loop.

A "cell" is one ``(task, spec_config, seed)`` triple. Cells live under
``<run_dir>/<cell_id>/`` so multi-config sweeps don't collide. Cells are
skip-if-done by default; ``--force`` re-runs them.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from spec_eval.algo_detect import defaults_for, detect_algorithm
from spec_eval.client import AsyncSGLangClient, SGLangClient
from spec_eval.guards import (
    GuardResult,
    is_baseline_tuple,
    parse_config_tuple,
    vocab_guard,
)
from spec_eval.metrics import BenchmarkMetrics, print_results
from spec_eval.registry import BENCHMARKS
from spec_eval.server import ServerConfig, SpecConfig, flush_cache, launch, stop, wait_for_server
from spec_eval.tasks import Benchmarker  # noqa: F401 — registers tasks

logger = logging.getLogger(__name__)

# ─── Presets ───────────────────────────────────────────────────────────────

ENGLISH_PRESET = ["humaneval", "simpleqa", "gsm8k", "mtbench"]
REASONING_PRESET = ["arc_challenge", "hellaswag", "mtbench"]
ALL_EN = [
    "humaneval", "livecodebench",
    "simpleqa", "financeqa",
    "mmlu", "gpqa",
    "gsm8k", "math500", "aime",
    "arc_challenge", "hellaswag",
    "mtbench",
]
ALL = ALL_EN + ["ceval", "mmstar"]


def expand_task_spec(tasks_arg: str) -> List[str]:
    t = tasks_arg.strip().lower()
    if t == "english":
        return list(ENGLISH_PRESET)
    if t == "reasoning":
        return list(REASONING_PRESET)
    if t == "all-en":
        return list(ALL_EN)
    if t == "all":
        return list(ALL)
    return [x.strip() for x in tasks_arg.split(",") if x.strip()]


def parse_task_spec(item: str) -> Tuple[str, Optional[int], Optional[List[str]]]:
    """``humaneval`` | ``humaneval:50`` | ``mmlu:50:physics,chem``"""
    parts = item.split(":")
    name = parts[0]
    n = int(parts[1]) if len(parts) >= 2 and parts[1] else None
    subset: Optional[List[str]] = None
    if len(parts) >= 3 and parts[2]:
        subset = [s for s in parts[2].split(",") if s]
    return name, n, subset


# ─── Run config ────────────────────────────────────────────────────────────


@dataclass
class CellConfig:
    """One (spec_config, seed) cell. A run iterates the cartesian product of
    these against the chosen task list."""

    bs: int = 1
    num_steps: int = 5
    eagle_topk: int = 8
    draft_tokens: int = 64
    seed: int = 42

    @property
    def is_baseline(self) -> bool:
        return self.num_steps == 0

    @property
    def short_id(self) -> str:
        if self.is_baseline:
            return f"baseline_bs{self.bs}_s{self.seed}"
        return (
            f"bs{self.bs}_steps{self.num_steps}_topk{self.eagle_topk}"
            f"_dt{self.draft_tokens}_s{self.seed}"
        )

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int, int], seed: int = 42) -> "CellConfig":
        return cls(bs=t[0], num_steps=t[1], eagle_topk=t[2], draft_tokens=t[3], seed=seed)


@dataclass
class EvalConfig:
    """Top-level options for one CLI invocation."""

    target: str
    draft: Optional[str] = None
    algorithm: str = "EAGLE"  # "EAGLE" | "EAGLE3" — auto-detected if not pinned
    task_specs: List[str] = field(default_factory=list)
    num_samples_default: Optional[int] = 50
    output_dir: Path = Path("results")
    run_name: Optional[str] = None
    cell_configs: List[CellConfig] = field(default_factory=list)
    seeds: List[int] = field(default_factory=lambda: [42])
    concurrency: int = 1
    force: bool = False
    skip_launch_server: bool = False
    skip_if_exists: bool = False
    """If a previous run under ``output_dir`` has the same ``run_signature``
    AND a complete ``summary.json``, return that run_dir instead of executing."""
    # full server overrides — see ServerConfig fields
    server_overrides: Dict[str, Any] = field(default_factory=dict)


# ─── Driver ────────────────────────────────────────────────────────────────


class EvalRun:
    def __init__(self, ec: EvalConfig):
        self.ec = ec
        self.run_dir = self._build_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _build_run_dir(self) -> Path:
        if self.ec.run_name:
            return self.ec.output_dir / self.ec.run_name
        ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        target_short = self.ec.target.split("/")[-1]
        draft_short = self.ec.draft.split("/")[-1] if self.ec.draft else "baseline"
        return self.ec.output_dir / f"{target_short}__{draft_short}__{ts}"

    @staticmethod
    def _git_sha() -> str:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2
            )
            return out.stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _drafter_sha256(draft: Optional[str]) -> Optional[str]:
        """SHA256 of ``<draft>/model.safetensors`` if the drafter is a local
        directory containing that file. Hugging Face IDs return ``None``.

        Lifted from the old eval.py (lines 559-569). Pins the actual weights
        — useful when a checkpoint is silently overwritten between runs."""
        if not draft:
            return None
        path = Path(draft).expanduser()
        sft = path / "model.safetensors"
        if not sft.is_file():
            return None
        h = hashlib.sha256()
        with open(sft, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _log_tail(log_path: str, n_lines: int = 40) -> str:
        """Return the last ``n_lines`` of ``log_path``. Best-effort; ``""`` on
        any failure (e.g. log not flushed)."""
        try:
            with open(log_path, "rb") as f:
                # Cheap tail: read whole file (server logs are small enough).
                data = f.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
        lines = data.splitlines()
        return "\n".join(lines[-n_lines:])

    def _write_config(self) -> None:
        from spec_eval.stats import run_signature  # noqa: PLC0415

        drafter_sha = self._drafter_sha256(self.ec.draft)
        cfg = {
            "target": self.ec.target,
            "draft": self.ec.draft,
            "drafter_sha256": drafter_sha,
            "algorithm": self.ec.algorithm,
            "tasks": self.ec.task_specs,
            "num_samples_default": self.ec.num_samples_default,
            "cell_configs": [asdict(c) for c in self.ec.cell_configs],
            "seeds": self.ec.seeds,
            "concurrency": self.ec.concurrency,
            "server_overrides": self.ec.server_overrides,
            "started_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "git_sha": self._git_sha(),
        }
        cfg["run_signature"] = run_signature(cfg)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        if drafter_sha:
            logger.info("drafter SHA256: %s (%s/model.safetensors)",
                        drafter_sha[:16] + "…", self.ec.draft)
        logger.info("run signature: %s", cfg["run_signature"])

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        kwargs = {}
        if self.ec.server_overrides.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True
        return AutoTokenizer.from_pretrained(self.ec.target, **kwargs)

    def _server_for(self, cell: CellConfig) -> ServerConfig:
        """Build a ServerConfig for one cell. Baseline cells drop the drafter."""
        ov = self.ec.server_overrides
        # Always enable step-time recording so /server_info.step_time_dict is
        # populated — required for the SpecForge-style p20 step-time metric.
        env = dict(ov.get("extra_env") or {})
        env.setdefault("SGLANG_RECORD_STEP_TIME", "1")
        sc = ServerConfig(
            target=self.ec.target,
            draft=None if cell.is_baseline else self.ec.draft,
            host=ov.get("host", "127.0.0.1"),
            port=ov.get("port", 0),
            dtype=ov.get("dtype", "bfloat16"),
            mem_fraction_static=ov.get("mem_fraction_static", 0.85),
            tp_size=ov.get("tp_size", 1),
            context_length=ov.get("context_length", 4096),
            trust_remote_code=ov.get("trust_remote_code", False),
            attention_backend=ov.get("attention_backend"),
            cuda_graph_max_bs=ov.get("cuda_graph_max_bs", cell.bs if cell.bs > 1 else None),
            enable_metrics=ov.get("enable_metrics", True),
            extra_env=env,
            extra_args=list(ov.get("extra_args", [])),
            spec=SpecConfig(
                algorithm=self.ec.algorithm,
                num_steps=cell.num_steps,
                eagle_topk=cell.eagle_topk,
                draft_tokens=cell.draft_tokens,
            ),
        )
        return sc

    # ─── Entry point ──────────────────────────────────────────────────────

    def execute(self) -> Dict[str, Any]:
        self._write_config()

        # Dedup: if a prior completed run under output_dir matches our
        # signature, hand the caller that run instead of repeating work.
        if self.ec.skip_if_exists:
            prior = self._find_matching_prior_run()
            if prior is not None:
                logger.warning(
                    "skip-if-exists: matching completed run at %s — not re-running",
                    prior,
                )
                self.run_dir = prior
                with open(prior / "summary.json") as f:
                    return json.load(f)

        # Vocab guard once, before booting anything.
        if self.ec.draft:
            r: GuardResult = vocab_guard(self.ec.target, self.ec.draft)
            if not r.ok:
                raise SystemExit(f"PRE-FLIGHT FAIL: {r.message}")
            logger.info("pre-flight: %s", r.message)

        results: Dict[str, Any] = {
            "run_name": self.run_dir.name,
            "target": self.ec.target,
            "draft": self.ec.draft,
            "algorithm": self.ec.algorithm,
            "cells": {},
        }

        # If the caller asked us to talk to an already-running server, we
        # iterate cells without re-booting — but spec params are fixed by the
        # running server, so we ignore cell tuples beyond the first.
        if self.ec.skip_launch_server:
            cells = self.ec.cell_configs[:1] or [CellConfig()]
            for cell in cells:
                rc = self._run_cell(cell, server_proc=None, server_log=None)
                results["cells"][cell.short_id] = rc
        else:
            for cell in self.ec.cell_configs:
                rc = self._boot_and_run_cell(cell)
                results["cells"][cell.short_id] = rc

        self._write_summary(results)
        return results

    def _boot_and_run_cell(self, cell: CellConfig) -> Dict[str, Any]:
        scfg = self._server_for(cell)
        log_path = str(self.run_dir / f"server_{cell.short_id}.log")
        proc, log_f = launch(scfg, log_path)
        try:
            if not wait_for_server(scfg.host, scfg.port, timeout_s=600):
                # Dump the last 40 lines of the server log so the user doesn't
                # have to chase the file. Carried over from old eval.py.
                tail = self._log_tail(log_path, n_lines=40)
                msg_lines = [
                    f"server never reached /health (full log: {log_path})",
                    "─── last 40 lines of server log ───────────────────────",
                    tail or "  (log empty or unreadable)",
                    "───────────────────────────────────────────────────────",
                ]
                print("\n".join(msg_lines), file=sys.stderr)
                raise SystemExit(1)
            # mutate the EvalConfig's server_overrides so subsequent loops
            # know which port was picked
            self.ec.server_overrides["host"] = scfg.host
            self.ec.server_overrides["port"] = scfg.port
            return self._run_cell(cell, server_proc=proc, server_log=log_f, server_cfg=scfg)
        finally:
            stop(proc, log_f)

    def _run_cell(
        self,
        cell: CellConfig,
        server_proc=None,
        server_log=None,
        server_cfg: Optional[ServerConfig] = None,
    ) -> Dict[str, Any]:
        cell_dir = self.run_dir / cell.short_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        # Stash cell config for provenance
        with open(cell_dir / "cell.json", "w") as f:
            json.dump({**asdict(cell), "algorithm": self.ec.algorithm}, f, indent=2)

        tokenizer = self._load_tokenizer()
        client = self._build_client()

        cell_result: Dict[str, Any] = {"tasks": {}}

        # Use a try/finally to ensure sync client closes; async client closes itself.
        sync_client_to_close = client if isinstance(client, SGLangClient) else None
        try:
            for spec in self.ec.task_specs:
                task_name, n_override, subset = parse_task_spec(spec)
                task_dir = cell_dir / task_name
                metrics_path = task_dir / "metrics.json"

                if metrics_path.is_file() and not self.ec.force:
                    logger.info(
                        "[skip] %s/%s (metrics.json exists; --force to overwrite)",
                        cell.short_id, task_name,
                    )
                    with open(metrics_path) as f:
                        cell_result["tasks"][task_name] = json.load(f)
                    continue

                try:
                    cls = BENCHMARKS.get(task_name)
                except KeyError as e:
                    logger.error(str(e))
                    continue

                n_samples = (
                    n_override if n_override is not None else self.ec.num_samples_default
                )
                logger.info(
                    "→ cell=%s task=%s  n=%s  subset=%s",
                    cell.short_id, task_name, n_samples, subset,
                )

                if subset is not None:
                    bench = cls(num_samples=n_samples, subset=subset, seed=cell.seed)
                else:
                    bench = cls(num_samples=n_samples, seed=cell.seed)

                metrics, rows = bench.run(
                    client,
                    tokenizer,
                    num_steps=(cell.num_steps if not cell.is_baseline else None),
                    batch_size=cell.bs,
                )
                self._write_task(task_dir, metrics, rows)
                print_results([metrics], task_name, show_accuracy=bench.SHOW_ACCURACY)
                cell_result["tasks"][task_name] = metrics.to_dict()

                # Inter-task flush so the radix cache doesn't leak prompts.
                flush_cache(
                    self.ec.server_overrides.get("host", "127.0.0.1"),
                    self.ec.server_overrides.get("port", 30000),
                )
        finally:
            if sync_client_to_close is not None:
                sync_client_to_close.close()
        return cell_result

    def _build_client(self) -> Union[SGLangClient, AsyncSGLangClient]:
        host = self.ec.server_overrides.get("host", "127.0.0.1")
        port = self.ec.server_overrides.get("port", 30000)
        if self.ec.concurrency > 1:
            return AsyncSGLangClient(host=host, port=port, concurrency=self.ec.concurrency)
        return SGLangClient(host=host, port=port)

    def _write_task(self, task_dir: Path, metrics: BenchmarkMetrics, rows: List[Dict]) -> None:
        task_dir.mkdir(parents=True, exist_ok=True)
        with open(task_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        with open(task_dir / "requests.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    def _write_summary(self, results: Dict[str, Any]) -> None:
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)

    def _find_matching_prior_run(self) -> Optional[Path]:
        """Walk ``output_dir`` looking for a finished run with the same
        ``run_signature`` as ours. Returns the first match (oldest by mtime)
        or ``None``."""
        from spec_eval.stats import run_signature  # noqa: PLC0415

        my_sig: Optional[str] = None
        cfg_path = self.run_dir / "config.json"
        if cfg_path.is_file():
            with open(cfg_path) as f:
                my_sig = json.load(f).get("run_signature")
        if not my_sig:
            return None

        out_root = self.ec.output_dir
        if not out_root.is_dir():
            return None
        candidates: List[Path] = []
        for child in out_root.iterdir():
            if not child.is_dir() or child == self.run_dir:
                continue
            cf = child / "config.json"
            sf = child / "summary.json"
            if not (cf.is_file() and sf.is_file()):
                continue
            try:
                with open(cf) as f:
                    sig = json.load(f).get("run_signature")
            except Exception:  # noqa: BLE001
                continue
            if sig == my_sig:
                candidates.append(child)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[0]


# ─── Helpers used by the CLI ───────────────────────────────────────────────


def resolve_cell_configs(
    config_list: Optional[List[str]],
    seeds: List[int],
    auto_defaults: bool,
    draft: Optional[str],
    algorithm: str,
    explicit: Optional[Tuple[int, int, int, int]] = None,
) -> List[CellConfig]:
    """Materialise the cartesian product of (config tuples × seeds).

    Precedence:
      1. ``config_list`` (SpecForge format) if provided
      2. ``explicit`` 4-tuple if provided
      3. ``auto_defaults=True`` ⇒ EAGLE-2 (5,8,64) or EAGLE-3 (3,1,4)
    """
    tuples: List[Tuple[int, int, int, int]] = []
    if config_list:
        tuples = [parse_config_tuple(s) for s in config_list]
    elif explicit is not None:
        tuples = [explicit]
    elif auto_defaults:
        d = defaults_for(algorithm)
        tuples = [(1, d.num_steps, d.eagle_topk, d.draft_tokens)]
    else:
        tuples = [(1, 5, 8, 64)]  # EAGLE-2 defaults

    # Baseline-only requests still need a drafter on disk to populate the
    # server's algo arg path for the spec cells; but a pure baseline run
    # (draft is None) keeps the (1,0,0,0) tuple and skips spec args at boot.
    cells: List[CellConfig] = []
    for t in tuples:
        for seed in seeds:
            cells.append(CellConfig.from_tuple(t, seed=seed))
    return cells


def resolve_algorithm(draft: Optional[str], override: Optional[str]) -> str:
    """Algorithm override wins; otherwise auto-detect from the drafter."""
    if override:
        return detect_algorithm(draft, override=override)
    if not draft:
        return "EAGLE"
    return detect_algorithm(draft, override=None)
