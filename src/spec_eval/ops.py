"""Operational subcommands: doctor (env health check) + status (run progress).

Both are designed to be fast (no GPU import, no model load) and to give
operators the "is my box still healthy?" / "where is my long-running
sweep at?" answer in one shot.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── Doctor ────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: Optional[str] = None  # remediation hint when not ok

    def render(self) -> str:
        mark = "✓" if self.ok else "✗"
        line = f"  [{mark}] {self.name:32s}  {self.detail}"
        if not self.ok and self.hint:
            line += f"\n         → {self.hint}"
        return line


def _check_python() -> CheckResult:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return CheckResult(
        name="python >= 3.10",
        ok=ok,
        detail=f"{sys.version.split()[0]} ({sys.executable})",
        hint="install Python 3.10+ in this venv (uv python install 3.10)" if not ok else None,
    )


def _check_module(modname: str, extra_hint: str = "") -> CheckResult:
    try:
        mod = importlib.import_module(modname)
        version = getattr(mod, "__version__", "?")
        return CheckResult(modname, True, f"version={version}")
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name=modname,
            ok=False,
            detail=f"import failed: {type(e).__name__}: {e}",
            hint=extra_hint or f"uv pip install {modname}",
        )


def _check_torch_cuda() -> CheckResult:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return CheckResult(
                "torch.cuda available",
                False,
                "torch installed but cuda.is_available() == False",
                hint="install a CUDA build: see scripts/setup_cuda_env.sh",
            )
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        return CheckResult(
            "torch.cuda available",
            True,
            f"{n} device(s); 0={name!r}; torch={torch.__version__}",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "torch.cuda available",
            False,
            f"torch import failed: {type(e).__name__}: {e}",
            hint="rebuild venv: rm -rf .venv uv.lock && uv sync && uv pip install 'sglang[all]>=0.5'",
        )


def _check_cuda_home() -> CheckResult:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not home:
        return CheckResult(
            "CUDA_HOME set",
            False,
            "CUDA_HOME / CUDA_PATH not set in environment",
            hint="source $HOME/.spec-eval-cuda.env (or run scripts/setup_cuda_env.sh)",
        )
    if not Path(home).is_dir():
        return CheckResult(
            "CUDA_HOME set",
            False,
            f"CUDA_HOME={home} but the directory does not exist",
            hint="re-run scripts/setup_cuda_env.sh",
        )
    return CheckResult("CUDA_HOME set", True, home)


def _check_nvcc() -> CheckResult:
    p = shutil.which("nvcc")
    if not p:
        return CheckResult(
            "nvcc on PATH",
            False,
            "nvcc not found",
            hint="source $HOME/.spec-eval-cuda.env",
        )
    return CheckResult("nvcc on PATH", True, p)


def _check_curand_header() -> CheckResult:
    """FlashInfer JIT needs curand_kernel.h; we've seen this slip on conda
    installs. Cheaper to check now than wait for a 60s JIT failure later."""
    home = os.environ.get("CUDA_HOME")
    if not home:
        return CheckResult("curand_kernel.h", False, "CUDA_HOME unset; skipped")
    candidates = [
        Path(home) / "include" / "curand_kernel.h",
        Path(home) / "targets" / "x86_64-linux" / "include" / "curand_kernel.h",
        Path(home) / "targets" / "aarch64-linux" / "include" / "curand_kernel.h",
    ]
    for c in candidates:
        if c.is_file():
            return CheckResult("curand_kernel.h", True, str(c))
    return CheckResult(
        "curand_kernel.h",
        False,
        f"not found under {home}/include or targets/*/include",
        hint="bash scripts/setup_cuda_env.sh --symlinks-only",
    )


def _check_sglang() -> CheckResult:
    return _check_module(
        "sglang",
        extra_hint="uv pip install 'sglang[all]>=0.5' (needs CUDA env sourced first)",
    )


def _check_hf_token() -> CheckResult:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return CheckResult("HF_TOKEN set", True, f"len={len(tok)} (env)")
    # Some datasets are public — this is informational only.
    return CheckResult(
        "HF_TOKEN set",
        True,
        "(not set — fine for public datasets/models; gated repos will 401)",
    )


def _check_port_free(port: int) -> CheckResult:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return CheckResult(
                f"port {port} free",
                False,
                "already bound (another sglang already running?)",
                hint=f"lsof -iTCP:{port}  # find the offender; or use a different --port",
            )
    finally:
        s.close()
    return CheckResult(f"port {port} free", True, "OK")


def _check_disk_free(path: Path, min_gb: float = 5.0) -> CheckResult:
    try:
        stat = shutil.disk_usage(str(path))
    except Exception as e:  # noqa: BLE001
        return CheckResult(f"disk free under {path}", False, f"{type(e).__name__}: {e}")
    free_gb = stat.free / (1 << 30)
    ok = free_gb >= min_gb
    return CheckResult(
        f"disk free under {path}",
        ok,
        f"{free_gb:.1f} GiB free (warn < {min_gb:.0f} GiB)",
        hint="clear ~/.cache/huggingface and old results/" if not ok else None,
    )


def run_doctor(port: int = 30000, output_dir: Path = Path("results")) -> Tuple[List[CheckResult], int]:
    """Run all preflight checks. Returns (results, exit_code)."""
    checks = [
        _check_python(),
        _check_module("numpy"),
        _check_module("httpx"),
        _check_module("transformers"),
        _check_module("datasets"),
        _check_cuda_home(),
        _check_nvcc(),
        _check_curand_header(),
        _check_torch_cuda(),
        _check_sglang(),
        _check_hf_token(),
        _check_port_free(port),
        _check_disk_free(output_dir),
    ]
    fails = [c for c in checks if not c.ok]
    return checks, (1 if fails else 0)


# ─── Status ────────────────────────────────────────────────────────────────


def _human_dt(ts_iso: Optional[str]) -> str:
    if not ts_iso:
        return "?"
    try:
        dt = _dt.datetime.fromisoformat(ts_iso.rstrip("Z"))
    except ValueError:
        return ts_iso
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _scan_run_dir(run_dir: Path) -> Dict[str, Any]:
    """Return a structured snapshot of a run's progress."""
    cfg: Dict[str, Any] = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            cfg = json.load(f)

    tasks = cfg.get("tasks") or []
    cells = cfg.get("cell_configs") or []
    expected_tasks = len(tasks)
    expected_cells = len(cells)
    expected_total = max(expected_tasks * expected_cells, 1)

    # Walk cell dirs and tally task metrics.json presence
    completed: List[Tuple[str, str]] = []
    in_flight: List[Tuple[str, str]] = []
    if run_dir.is_dir():
        for cell_dir in sorted(run_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            cell_json = cell_dir / "cell.json"
            if not cell_json.is_file():
                continue  # not a task cell
            for task_dir in sorted(cell_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                metrics = task_dir / "metrics.json"
                requests = task_dir / "requests.jsonl"
                if metrics.is_file():
                    completed.append((cell_dir.name, task_dir.name))
                elif requests.is_file():
                    in_flight.append((cell_dir.name, task_dir.name))

    summary_path = run_dir / "summary.json"
    state = "finished" if summary_path.is_file() else (
        "running" if in_flight or completed else "started"
    )

    # Server logs (one per cell)
    server_logs = sorted(run_dir.glob("server_*.log"))

    return {
        "run_dir": str(run_dir),
        "state": state,
        "started_utc": cfg.get("started_utc"),
        "target": cfg.get("target"),
        "draft": cfg.get("draft"),
        "algorithm": cfg.get("algorithm"),
        "run_signature": cfg.get("run_signature"),
        "expected_tasks": expected_tasks,
        "expected_cells": expected_cells,
        "expected_total": expected_total,
        "completed": completed,
        "in_flight": in_flight,
        "server_logs": [str(p.relative_to(run_dir)) for p in server_logs],
    }


def render_status(run_dir: Path) -> str:
    snap = _scan_run_dir(run_dir)
    expected = snap["expected_total"]
    done = len(snap["completed"])
    flight = len(snap["in_flight"])
    pct = (100.0 * done / expected) if expected else 0.0

    lines = [
        f"# spec-eval status — `{Path(snap['run_dir']).name}`",
        "",
        f"- **State**       : `{snap['state']}`",
        f"- **Started**     : {_human_dt(snap['started_utc'])}",
        f"- **Target**      : `{snap['target']}`",
        f"- **Draft**       : `{snap['draft'] or '(baseline)'}`",
        f"- **Algorithm**   : `{snap['algorithm']}`",
        f"- **Signature**   : `{snap['run_signature'] or 'n/a'}`",
        f"- **Progress**    : {done}/{expected} task cells "
        f"(in-flight={flight}) — {pct:.0f}%",
        "",
        "## Completed",
    ]
    if snap["completed"]:
        for cell, task in snap["completed"]:
            lines.append(f"- `{cell}` / `{task}`")
    else:
        lines.append("(none)")
    lines.append("")
    if snap["in_flight"]:
        lines.append("## In-flight")
        for cell, task in snap["in_flight"]:
            lines.append(f"- `{cell}` / `{task}`  (requests.jsonl present, metrics.json missing)")
        lines.append("")
    if snap["server_logs"]:
        lines.append("## Server logs")
        for p in snap["server_logs"]:
            lines.append(f"- `{p}`")
    return "\n".join(lines)
