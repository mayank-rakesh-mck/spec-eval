"""CLI: ``uv run spec-eval <subcommand> ...``

Subcommands
-----------
  run         Boot SGLang and run benchmarks (the main path).
  report      Render a markdown report from a finished run dir.
  compare     Side-by-side diff of two runs (baseline vs spec-decode).
  audit       Inspect a draft model (vocab, algorithm, defaults).
  doctor      Preflight environment health check (Python, CUDA, sglang, ports).
  status      Snapshot a run-dir's progress (in-flight cells, ETA hints).
  list-tasks  Print known benchmark names.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from spec_eval.algo_detect import defaults_for, detect_algorithm
from spec_eval.guards import vocab_guard
from spec_eval.ops import render_status, run_doctor
from spec_eval.registry import BENCHMARKS
from spec_eval.report import (
    render_compare,
    render_run,
    write_compare,
    write_pareto_csv,
    write_report,
)
from spec_eval.runner import (
    EvalConfig,
    EvalRun,
    expand_task_spec,
    resolve_algorithm,
    resolve_cell_configs,
)
from spec_eval.tasks import Benchmarker  # noqa: F401 — registers tasks


def _add_common_logging(p: argparse.ArgumentParser) -> None:
    p.add_argument("--verbose", "-v", action="count", default=0)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    model = p.add_argument_group("model")
    model.add_argument("--target", required=True, help="target HF id or local path")
    model.add_argument(
        "--draft",
        default=None,
        help="draft (EAGLE) HF id or local path. Omit for baseline (no spec decoding).",
    )

    spec = p.add_argument_group("speculative decoding")
    spec.add_argument(
        "--algorithm",
        default="auto",
        choices=["auto", "EAGLE", "EAGLE3"],
        help="auto detects from draft's config.json / name (EAGLE-3 hints).",
    )
    spec.add_argument(
        "--auto-defaults",
        action="store_true",
        default=True,
        help="apply algo-aware spec defaults (EAGLE: 5/8/64, EAGLE3: 3/1/4)",
    )
    spec.add_argument("--num-steps", type=int, default=None)
    spec.add_argument("--eagle-topk", type=int, default=None)
    spec.add_argument("--draft-tokens", type=int, default=None)
    spec.add_argument(
        "--config-list",
        nargs="+",
        default=None,
        help=(
            "SpecForge-style sweep: 'batch_size,num_steps,topk,draft_tokens' "
            "tuples (space-separated). Special: 'bs,0,0,0' = baseline cell. "
            "Example: --config-list 1,0,0,0 1,5,8,64 (baseline + EAGLE-2)"
        ),
    )
    spec.add_argument("--seeds", default="42", help="comma-separated seeds")

    server = p.add_argument_group("server")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=0)
    server.add_argument("--dtype", default="bfloat16")
    server.add_argument("--mem-fraction-static", type=float, default=0.85)
    server.add_argument("--tp-size", type=int, default=1)
    server.add_argument("--context-length", type=int, default=4096)
    server.add_argument("--trust-remote-code", action="store_true")
    server.add_argument("--attention-backend", default=None, help="e.g. fa3")
    server.add_argument("--cuda-graph-max-bs", type=int, default=None)
    server.add_argument(
        "--no-enable-metrics",
        dest="enable_metrics",
        action="store_false",
        default=True,
        help=(
            "skip the SGLang --enable-metrics flag. ONLY use for A/B perf "
            "measurement of the metrics-collector overhead itself; without it "
            "you lose inference_time / queue_time / decode_throughput / ITL."
        ),
    )
    server.add_argument("--skip-launch-server", action="store_true")
    server.add_argument("--extra-arg", action="append", default=[])

    bench = p.add_argument_group("benchmarks")
    bench.add_argument(
        "--tasks",
        default="english",
        help=(
            "presets: english | reasoning | all-en | all  | "
            "or comma list: humaneval,gsm8k  | "
            "per-task overrides: humaneval:50, mmlu:50:physics"
        ),
    )
    bench.add_argument("--num-samples", type=int, default=50)
    bench.add_argument("--output-dir", default="results")
    bench.add_argument("--run-name", default=None)
    bench.add_argument("--force", action="store_true", help="overwrite finished cells")
    bench.add_argument(
        "--skip-if-exists",
        action="store_true",
        help=(
            "if a finished run with the same (target, draft, tasks, cells, seeds, "
            "num_samples) exists under --output-dir, reuse it instead of re-running"
        ),
    )
    bench.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="client-side concurrent /generate requests (uses async client)",
    )
    bench.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan and exit without booting the server",
    )

    _add_common_logging(p)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spec-eval",
        description="EAGLE-2 / EAGLE-3 speculative-decoding eval pipeline for SGLang.",
    )
    sub = p.add_subparsers(dest="cmd")

    # run (default)
    run_p = sub.add_parser(
        "run",
        help="boot SGLang + run benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_run_args(run_p)

    # report
    rep_p = sub.add_parser("report", help="render markdown report for a run dir")
    rep_p.add_argument("run_dir", type=Path)
    rep_p.add_argument("--output", type=Path, default=None)
    rep_p.add_argument("--print", action="store_true", help="also print to stdout")
    rep_p.add_argument(
        "--pareto-csv",
        type=Path,
        default=None,
        help="also write a flat (run, cell, task, accept_length, throughput, …) CSV",
    )
    _add_common_logging(rep_p)

    # compare
    cmp_p = sub.add_parser("compare", help="diff a baseline run against a spec-decode run")
    cmp_p.add_argument("baseline_dir", type=Path)
    cmp_p.add_argument("spec_dir", type=Path)
    cmp_p.add_argument("--output", type=Path, default=None)
    cmp_p.add_argument("--print", action="store_true")
    cmp_p.add_argument(
        "--pareto-csv",
        type=Path,
        default=None,
        help="write a flat CSV with rows from BOTH runs for Pareto plotting",
    )
    _add_common_logging(cmp_p)

    # audit
    aud_p = sub.add_parser("audit", help="inspect a draft model (vocab, algorithm)")
    aud_p.add_argument("--target", required=True)
    aud_p.add_argument("--draft", default=None)
    _add_common_logging(aud_p)

    # doctor — environment health check
    doc_p = sub.add_parser("doctor", help="preflight environment health check (no GPU work)")
    doc_p.add_argument("--port", type=int, default=30000, help="port we'd boot sglang on")
    doc_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="dir to disk-free-check",
    )
    doc_p.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of human text",
    )
    _add_common_logging(doc_p)

    # status — snapshot of a run dir
    sta_p = sub.add_parser("status", help="snapshot a run-dir's progress")
    sta_p.add_argument("run_dir", type=Path)
    sta_p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    _add_common_logging(sta_p)

    # list-tasks
    sub.add_parser("list-tasks", help="print known benchmark names")
    return p


def _set_log_level(verbose: int) -> None:
    level = logging.WARNING - 10 * min(verbose + 1, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: List[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    # Default subcommand: "run". This keeps the v0 invocation pattern working:
    #   spec-eval --target ... --draft ...
    if args.cmd is None:
        # Re-parse with "run" prepended.
        return main(["run", *(sys.argv[1:] if argv is None else argv)])

    _set_log_level(getattr(args, "verbose", 0))

    if args.cmd == "list-tasks":
        print("Known benchmarks:")
        for n in BENCHMARKS.names():
            print(f"  - {n}")
        return 0

    if args.cmd == "audit":
        return _cmd_audit(args)

    if args.cmd == "report":
        return _cmd_report(args)

    if args.cmd == "compare":
        return _cmd_compare(args)

    if args.cmd == "doctor":
        return _cmd_doctor(args)

    if args.cmd == "status":
        return _cmd_status(args)

    if args.cmd == "run":
        return _cmd_run(args)

    p.error(f"unknown subcommand {args.cmd!r}")
    return 2


# ─── Subcommand impls ──────────────────────────────────────────────────────


def _cmd_audit(args) -> int:
    print(f"target : {args.target}")
    if args.draft:
        algo = detect_algorithm(args.draft)
        defs = defaults_for(algo)
        print(f"draft  : {args.draft}")
        print(f"algo   : {algo}  (auto-detected)")
        print(f"defaults: num_steps={defs.num_steps}  topk={defs.eagle_topk}  draft_tokens={defs.draft_tokens}")
    g = vocab_guard(args.target, args.draft)
    print(f"vocab  : {g.message}")
    return 0 if g.ok else 1


def _cmd_report(args) -> int:
    if args.print:
        print(render_run(args.run_dir))
    out = write_report(args.run_dir, args.output)
    print(f"wrote {out}")
    if args.pareto_csv:
        csv_out = write_pareto_csv([args.run_dir], args.pareto_csv)
        print(f"wrote {csv_out}")
    return 0


def _cmd_doctor(args) -> int:
    results, rc = run_doctor(port=args.port, output_dir=args.output_dir)
    if args.json:
        print(json.dumps(
            [{"name": r.name, "ok": r.ok, "detail": r.detail, "hint": r.hint}
             for r in results],
            indent=2,
        ))
        return rc
    print("spec-eval doctor")
    print()
    for r in results:
        print(r.render())
    print()
    fails = [r for r in results if not r.ok]
    if fails:
        print(f"  {len(fails)} check(s) failed.")
    else:
        print("  all checks passed.")
    return rc


def _cmd_status(args) -> int:
    if not args.run_dir.is_dir():
        print(f"error: {args.run_dir} is not a directory", file=sys.stderr)
        return 2
    if args.json:
        from spec_eval.ops import _scan_run_dir  # noqa: PLC0415
        print(json.dumps(_scan_run_dir(args.run_dir), indent=2))
        return 0
    print(render_status(args.run_dir))
    return 0


def _cmd_compare(args) -> int:
    if args.print:
        print(render_compare(args.baseline_dir, args.spec_dir))
    out = write_compare(args.baseline_dir, args.spec_dir, args.output)
    print(f"wrote {out}")
    if args.pareto_csv:
        csv_out = write_pareto_csv([args.baseline_dir, args.spec_dir], args.pareto_csv)
        print(f"wrote {csv_out}")
    return 0


def _cmd_run(args) -> int:
    task_specs = expand_task_spec(args.tasks)
    if not task_specs:
        print("error: no tasks selected", file=sys.stderr)
        return 2

    # 1. Algorithm: auto-detect from draft if asked.
    algo = resolve_algorithm(args.draft, None if args.algorithm == "auto" else args.algorithm)
    algo_defs = defaults_for(algo)

    # 2. If user passed --config-list, that wins. Otherwise build a single
    #    cell from --num-steps/--eagle-topk/--draft-tokens (or defaults).
    explicit_tuple = None
    if args.num_steps is not None or args.eagle_topk is not None or args.draft_tokens is not None:
        explicit_tuple = (
            1,
            args.num_steps if args.num_steps is not None else algo_defs.num_steps,
            args.eagle_topk if args.eagle_topk is not None else algo_defs.eagle_topk,
            args.draft_tokens if args.draft_tokens is not None else algo_defs.draft_tokens,
        )

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    cells = resolve_cell_configs(
        config_list=args.config_list,
        seeds=seeds,
        auto_defaults=True,
        draft=args.draft,
        algorithm=algo,
        explicit=explicit_tuple,
    )

    # 3. Plan + dry-run output.
    plan = {
        "target": args.target,
        "draft": args.draft,
        "algorithm": algo,
        "tasks": task_specs,
        "cells": [c.short_id for c in cells],
        "concurrency": args.concurrency,
        "num_samples_default": args.num_samples,
    }
    print("plan:")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    ec = EvalConfig(
        target=args.target,
        draft=args.draft,
        algorithm=algo,
        task_specs=task_specs,
        num_samples_default=args.num_samples,
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        cell_configs=cells,
        seeds=seeds,
        concurrency=args.concurrency,
        force=args.force,
        skip_launch_server=args.skip_launch_server,
        skip_if_exists=args.skip_if_exists,
        server_overrides={
            "host": args.host,
            "port": args.port,
            "dtype": args.dtype,
            "mem_fraction_static": args.mem_fraction_static,
            "tp_size": args.tp_size,
            "context_length": args.context_length,
            "trust_remote_code": args.trust_remote_code,
            "attention_backend": args.attention_backend,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "enable_metrics": args.enable_metrics,
            "extra_args": list(args.extra_arg),
        },
    )
    run = EvalRun(ec)
    run.execute()

    # 4. Auto-generate a report so the user gets something readable immediately.
    try:
        write_report(run.run_dir)
        print(f"\nrun complete; results + report.md in: {run.run_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"\nrun complete; results in: {run.run_dir} (report failed: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
