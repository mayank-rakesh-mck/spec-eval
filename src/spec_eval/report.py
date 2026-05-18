"""Generate a human-readable markdown report from a run directory.

Walks ``<run_dir>/<cell_id>/<task>/metrics.json`` and emits a table per cell
plus a roll-up. Used by both ``spec-eval report`` and ``spec-eval compare``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "summary.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def _load_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "config.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def _fmt(v, spec: str = ".3f", none: str = "—") -> str:
    if isinstance(v, (int, float)):
        return format(v, spec)
    return none


def _ci_str(ci: Optional[Dict[str, Any]], spec: str = ".3f") -> str:
    """Render a bootstrap CI dict as ``[lo, hi]`` or ``—`` if absent."""
    if not isinstance(ci, dict):
        return "—"
    lo, hi = ci.get("lo"), ci.get("hi")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return "—"
    return f"[{format(lo, spec)}, {format(hi, spec)}]"


def _cell_table(cell_name: str, tasks: Dict[str, Dict[str, Any]]) -> str:
    """Two tables per cell: a headline summary + a spec-decode detail block."""
    lines = [f"### Cell: `{cell_name}`", ""]
    lines.append("**Headline** (95% CIs are percentile-bootstrap, n=2000 resamples)")
    lines.append("")
    lines.append(
        "| Task | N | Accuracy | Latency (s) | Throughput (tok/s) | "
        "Accept length | AL 95% CI | Spec |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|:-:|:-:|")
    for task, m in sorted(tasks.items()):
        active = "✓" if m.get("spec_decode_active") else "—"
        lines.append(
            f"| `{task}` | {m.get('num_questions', 0)} | "
            f"{_fmt(m.get('accuracy'), '.4f')} | "
            f"{_fmt(m.get('latency'), '.2f')} | "
            f"{_fmt(m.get('output_throughput'), '.1f')} | "
            f"{_fmt(m.get('accept_length'), '.3f')} | "
            f"{_ci_str(m.get('accept_length_ci'))} | {active} |"
        )

    # Spec-decode detail (only emit if at least one task has spec_decode_active)
    if any(t.get("spec_decode_active") for t in tasks.values()):
        lines.append("")
        lines.append("**Speculative-decoding detail** (per cell, summed over the task's prompts)")
        lines.append("")
        lines.append(
            "| Task | α_per_token (RoS / MoR) | α_norm (RoS / MoR) | "
            "Accept len (RoS / MoR) | Tree accept | Accept hist (drafts→steps) | "
            "Drafts mean / step | Step p20 (ms) | Eff. speed (tps) |"
        )
        lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|")
        for task, m in sorted(tasks.items()):
            if not m.get("spec_decode_active"):
                continue
            hist = m.get("spec_accept_histogram_total")
            hist_s = (
                ", ".join(f"{i}→{c}" for i, c in enumerate(hist[:6]))
                + ("…" if hist and len(hist) > 6 else "")
                if hist
                else "—"
            )
            lines.append(
                f"| `{task}` | "
                f"{_fmt(m.get('alpha_per_token'), '.3f')} / "
                f"{_fmt(m.get('alpha_per_token_mor'), '.3f')} | "
                f"{_fmt(m.get('alpha_normalized'), '.3f')} / "
                f"{_fmt(m.get('alpha_normalized_mor'), '.3f')} | "
                f"{_fmt(m.get('accept_length'), '.3f')} / "
                f"{_fmt(m.get('accept_length_mor'), '.3f')} | "
                f"{_fmt(m.get('accept_rate_overall'), '.3f')} | {hist_s} | "
                f"{_fmt(m.get('spec_accepted_drafts_mean'), '.3f')} | "
                f"{_fmt(m.get('step_time_p20_ms'), '.2f')} | "
                f"{_fmt(m.get('effective_speed_tps'), '.1f')} |"
            )

        # Per-prompt distribution (variance behind the cell-level ratio-of-sums).
        # Skip if no task carries a distribution (e.g. tiny n with no spec activity).
        has_dist = any(
            t.get("accept_length_stats") or t.get("accept_rate_stats")
            for t in tasks.values()
            if t.get("spec_decode_active")
        )
        if has_dist:
            lines.append("")
            lines.append("**Per-prompt distribution** (behind the ratio-of-sums headline)")
            lines.append("")
            lines.append(
                "| Task | accept_length min | p50 | p90 | max | "
                "accept_rate min | p50 | p90 | max |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for task, m in sorted(tasks.items()):
                if not m.get("spec_decode_active"):
                    continue
                al = m.get("accept_length_stats") or {}
                ar = m.get("accept_rate_stats") or {}
                lines.append(
                    f"| `{task}` | "
                    f"{_fmt(al.get('min'), '.3f')} | "
                    f"{_fmt(al.get('p50'), '.3f')} | "
                    f"{_fmt(al.get('p90'), '.3f')} | "
                    f"{_fmt(al.get('max'), '.3f')} | "
                    f"{_fmt(ar.get('min'), '.3f')} | "
                    f"{_fmt(ar.get('p50'), '.3f')} | "
                    f"{_fmt(ar.get('p90'), '.3f')} | "
                    f"{_fmt(ar.get('max'), '.3f')} |"
                )

    # Latency + cache detail (only show if at least one task has the data)
    has_latency = any(
        t.get("e2e_latency_p50") is not None for t in tasks.values()
    )
    has_cache = any(t.get("cache_hit_rate") is not None for t in tasks.values())
    has_itl = any(t.get("itl_ms_p50") is not None for t in tasks.values())
    if has_latency or has_cache:
        lines.append("")
        lines.append("**Latency & cache detail**")
        lines.append("")
        lines.append(
            "| Task | e2e p50 (s) | p90 | p99 | queue p50 (ms) | "
            "decode tps p50 | Cache hit % | Prompt tok | Cached tok | Retractions |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for task, m in sorted(tasks.items()):
            qt_ms = (
                m["queue_time_p50"] * 1000 if isinstance(m.get("queue_time_p50"), (int, float)) else None
            )
            hit = (
                m["cache_hit_rate"] * 100
                if isinstance(m.get("cache_hit_rate"), (int, float))
                else None
            )
            lines.append(
                f"| `{task}` | "
                f"{_fmt(m.get('e2e_latency_p50'), '.3f')} | "
                f"{_fmt(m.get('e2e_latency_p90'), '.3f')} | "
                f"{_fmt(m.get('e2e_latency_p99'), '.3f')} | "
                f"{_fmt(qt_ms, '.1f')} | "
                f"{_fmt(m.get('decode_throughput_p50'), '.1f')} | "
                f"{_fmt(hit, '.1f')} | "
                f"{m.get('prompt_tokens_sum', 0)} | "
                f"{m.get('cached_tokens_sum', 0)} | "
                f"{m.get('total_retractions', 0)} |"
            )

    # Inter-token latency — separate block because the headline metric is
    # ms/tok, not tok/s, and spec-decode can hurt this while helping tok/s.
    if has_itl:
        lines.append("")
        lines.append(
            "**Inter-token latency** (ms / generated token, derived from "
            "`inference_time / completion_tokens`)"
        )
        lines.append("")
        lines.append("| Task | ITL p50 (ms) | p90 | p99 |")
        lines.append("|---|---:|---:|---:|")
        for task, m in sorted(tasks.items()):
            if m.get("itl_ms_p50") is None:
                continue
            lines.append(
                f"| `{task}` | "
                f"{_fmt(m.get('itl_ms_p50'), '.2f')} | "
                f"{_fmt(m.get('itl_ms_p90'), '.2f')} | "
                f"{_fmt(m.get('itl_ms_p99'), '.2f')} |"
            )

    # Sanity flags (only emit if any flag fired)
    insane_total = sum(
        (t.get("sanity") or {}).get("n_insane", 0) for t in tasks.values()
    )
    if insane_total > 0:
        lines.append("")
        lines.append("**Sanity flags**")
        lines.append("")
        lines.append("| Task | Insane / N | Flags |")
        lines.append("|---|---:|---|")
        for task, m in sorted(tasks.items()):
            s = m.get("sanity") or {}
            n_in = s.get("n_insane", 0)
            if not n_in:
                continue
            flags = ", ".join(f"{k}={v}" for k, v in (s.get("flags_total") or {}).items())
            lines.append(f"| `{task}` | {n_in} / {s.get('n', 0)} | {flags} |")

    lines.append("")
    return "\n".join(lines)


def render_run(run_dir: Path) -> str:
    """Render one run directory to markdown."""
    summary = _load_summary(run_dir)
    config = _load_config(run_dir)
    if summary is None:
        return f"# {run_dir.name}\n\n(no summary.json yet)\n"

    out: List[str] = [f"# spec-eval report — `{run_dir.name}`", ""]
    if config:
        out.append("## Config")
        out.append("")
        out.append(f"- **Target**: `{config.get('target')}`")
        out.append(f"- **Draft**: `{config.get('draft') or '(baseline, no spec decoding)'}`")
        sha = config.get("drafter_sha256")
        if sha:
            out.append(f"- **Drafter SHA256**: `{sha}`")
        out.append(f"- **Algorithm**: `{config.get('algorithm')}`")
        out.append(f"- **Tasks**: `{', '.join(config.get('tasks', []))}`")
        out.append(f"- **Concurrency**: {config.get('concurrency')}")
        out.append(f"- **Started**: {config.get('started_utc')}")
        out.append(f"- **Git SHA**: `{config.get('git_sha') or 'n/a'}`")
        out.append("")

    cells: Dict[str, Dict[str, Any]] = summary.get("cells", {})
    if not cells:
        out.append("(no cells in summary)")
        return "\n".join(out)

    out.append("## Per-cell results")
    out.append("")
    for cell_name in sorted(cells):
        out.append(_cell_table(cell_name, cells[cell_name].get("tasks", {})))

    # Roll-up across all cells, per task — useful when sweeping seeds.
    out.append("## Roll-up (averaged across cells)")
    out.append("")
    rollup = _rollup_tasks(cells)
    out.append("| Task | Cells | Accuracy (avg) | Throughput (avg, tok/s) | Accept length (avg) |")
    out.append("|---|---:|---:|---:|---:|")
    for task, agg in sorted(rollup.items()):
        acc_s = (
            f"{agg['accuracy']:.4f}" if agg["accuracy"] is not None else "—"
        )
        out.append(
            f"| `{task}` | {agg['n_cells']} | {acc_s} | "
            f"{agg['throughput']:.1f} | {agg['accept_length']:.3f} |"
        )

    # Cross-cell + cross-task aggregation. Throughput is a ratio so geomean
    # is the right cross-task summary; accept_length is a token count so we
    # report arithmetic mean.  (Harmonic mean is appropriate for ITL — we
    # surface that too for the ms/tok scale.)
    cross = _cross_task_aggregate(cells)
    if cross:
        out.append("")
        out.append("## Cross-task aggregation (one cell × many tasks)")
        out.append("")
        out.append(
            "| Cell | Tasks | Throughput geomean (tok/s) | "
            "Accept length mean | ITL harmean (ms) |"
        )
        out.append("|---|---:|---:|---:|---:|")
        for cell_name, row in cross.items():
            out.append(
                f"| `{cell_name}` | {row['n_tasks']} | "
                f"{_fmt(row['throughput_geomean'], '.1f')} | "
                f"{_fmt(row['accept_length_mean'], '.3f')} | "
                f"{_fmt(row['itl_ms_harmean'], '.2f')} |"
            )
    out.append("")
    return "\n".join(out)


def _cross_task_aggregate(cells: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """For each cell, summarise across its tasks.

    - throughput → geometric mean (correct average for a ratio)
    - accept_length → arithmetic mean (correct for a token-count)
    - ITL ms → harmonic mean (correct for ms/tok, which is 1/rate)
    """
    from spec_eval.stats import geomean, harmean  # noqa: PLC0415

    out: Dict[str, Dict[str, Any]] = {}
    for cell_name, cell in cells.items():
        tasks = cell.get("tasks", {})
        if not tasks:
            continue
        tputs = [t.get("output_throughput") for t in tasks.values()]
        als = [t.get("accept_length") for t in tasks.values()]
        itls = [t.get("itl_ms_p50") for t in tasks.values()]
        out[cell_name] = {
            "n_tasks": len(tasks),
            "throughput_geomean": geomean([x for x in tputs if isinstance(x, (int, float))]),
            "accept_length_mean": (
                float(sum(x for x in als if isinstance(x, (int, float)))) /
                max(1, sum(1 for x in als if isinstance(x, (int, float))))
                if any(isinstance(x, (int, float)) for x in als) else None
            ),
            "itl_ms_harmean": harmean([x for x in itls if isinstance(x, (int, float))]),
        }
    return out


def write_pareto_csv(run_dirs: List[Path], output_path: Path) -> Path:
    """Emit a flat CSV: one row per (run, cell, task) with the columns needed
    to draw the Pareto frontier (accept_length × throughput).

    Designed for ``pandas.read_csv`` + matplotlib plotting; kept dependency-
    free here.

    Output columns: run, cell, task, accept_length, output_throughput,
    accuracy, itl_ms_p50, alpha_per_token, spec_decode_active.
    """
    fields = [
        "run", "cell", "task",
        "accept_length", "output_throughput",
        "accuracy", "itl_ms_p50",
        "alpha_per_token", "spec_decode_active",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for run_dir in run_dirs:
            summary = _load_summary(run_dir)
            if summary is None:
                continue
            for cell_name, cell in summary.get("cells", {}).items():
                for task, m in cell.get("tasks", {}).items():
                    w.writerow({
                        "run": run_dir.name,
                        "cell": cell_name,
                        "task": task,
                        "accept_length": m.get("accept_length"),
                        "output_throughput": m.get("output_throughput"),
                        "accuracy": m.get("accuracy"),
                        "itl_ms_p50": m.get("itl_ms_p50"),
                        "alpha_per_token": m.get("alpha_per_token"),
                        "spec_decode_active": m.get("spec_decode_active"),
                    })
    return output_path


def _rollup_tasks(cells: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Average across cells, per task. Skips ``None`` per metric independently."""
    keys = (
        "output_throughput",
        "accept_length",
        "alpha_per_token",
        "alpha_normalized",
        "accept_rate_overall",
        "cache_hit_rate",
        "e2e_latency_p50",
        "step_time_p20_ms",
    )
    agg: Dict[str, Dict[str, Any]] = {}
    for cell in cells.values():
        for task, m in cell.get("tasks", {}).items():
            slot = agg.setdefault(
                task,
                {"n_cells": 0, "accuracy_sum": 0.0, "accuracy_count": 0,
                 **{k: {"sum": 0.0, "n": 0} for k in keys}},
            )
            slot["n_cells"] += 1
            if m.get("accuracy") is not None:
                slot["accuracy_sum"] += m["accuracy"]
                slot["accuracy_count"] += 1
            for k in keys:
                v = m.get(k)
                if isinstance(v, (int, float)):
                    slot[k]["sum"] += v
                    slot[k]["n"] += 1
    out: Dict[str, Dict[str, Any]] = {}
    for task, slot in agg.items():
        row = {
            "n_cells": slot["n_cells"],
            "accuracy": (
                slot["accuracy_sum"] / slot["accuracy_count"]
                if slot["accuracy_count"] > 0
                else None
            ),
        }
        for k in keys:
            row[k] = slot[k]["sum"] / slot[k]["n"] if slot[k]["n"] > 0 else None
        # backward-compat aliases
        row["throughput"] = row["output_throughput"] if row["output_throughput"] is not None else 0.0
        row["accept_length"] = row["accept_length"] if row["accept_length"] is not None else 1.0
        out[task] = row
    return out


def _load_request_rows(run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Read per-task ``requests.jsonl`` files for every cell, returning
    ``{task: [row, ...]}`` aggregated across cells. Per-prompt token counts
    are what we need to run paired tests.

    Aggregating across cells is reasonable when there's one cell per side
    (the common case). If callers care about multi-cell paired tests they
    should compare cell-by-cell separately.
    """
    rows: Dict[str, List[Dict[str, Any]]] = {}
    if not run_dir.is_dir():
        return rows
    for cell_dir in run_dir.iterdir():
        if not cell_dir.is_dir():
            continue
        for task_dir in cell_dir.iterdir():
            if not task_dir.is_dir():
                continue
            rq = task_dir / "requests.jsonl"
            if not rq.is_file():
                continue
            with open(rq) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.setdefault(task_dir.name, []).append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def _per_prompt_throughput(rows: List[Dict[str, Any]]) -> List[float]:
    """Recover per-prompt tok/s from ``requests.jsonl`` rows.

    Prefer ``inference_time`` (server's pure-compute number, excludes queue);
    fall back to ``e2e_latency`` when SGLang was launched without
    ``--enable-metrics``. The fallback over-counts queue time but is
    monotone in the same direction, so paired tests are still meaningful.
    """
    out: List[float] = []
    for r in rows:
        ct = r.get("completion_tokens") or 0
        if ct <= 0:
            continue
        denom = r.get("inference_time")
        if not isinstance(denom, (int, float)) or denom <= 0:
            denom = r.get("e2e_latency")
        if isinstance(denom, (int, float)) and denom > 0:
            out.append(ct / denom)
    return out


def render_compare(baseline_dir: Path, spec_dir: Path) -> str:
    """Side-by-side comparison of two runs (usually baseline vs spec-decode)."""
    from spec_eval.stats import paired_wilcoxon  # noqa: PLC0415

    base = _load_summary(baseline_dir)
    spec = _load_summary(spec_dir)
    if base is None or spec is None:
        return "# compare: one or both runs missing summary.json"

    base_tasks = _rollup_tasks(base.get("cells", {}))
    spec_tasks = _rollup_tasks(spec.get("cells", {}))

    out: List[str] = [
        f"# compare — `{baseline_dir.name}` vs `{spec_dir.name}`",
        "",
        "## Throughput speedup",
        "",
        "| Task | Baseline tok/s | Spec tok/s | Speedup × | Accept len | α_per_token | Δ accuracy | Δ cache hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_tasks = sorted(set(base_tasks) | set(spec_tasks))
    for task in all_tasks:
        b = base_tasks.get(task)
        s = spec_tasks.get(task)
        if not b or not s:
            continue
        speedup = s["throughput"] / b["throughput"] if b["throughput"] > 0 else float("nan")
        delta_acc = "—"
        if b["accuracy"] is not None and s["accuracy"] is not None:
            delta_acc = f"{(s['accuracy'] - b['accuracy']) * 100:+.2f}pp"
        delta_hit = "—"
        if (
            isinstance(b.get("cache_hit_rate"), (int, float))
            and isinstance(s.get("cache_hit_rate"), (int, float))
        ):
            delta_hit = f"{(s['cache_hit_rate'] - b['cache_hit_rate']) * 100:+.2f}pp"
        out.append(
            f"| `{task}` | {b['throughput']:.1f} | {s['throughput']:.1f} | "
            f"**{speedup:.2f}×** | {s['accept_length']:.3f} | "
            f"{_fmt(s.get('alpha_per_token'), '.3f')} | {delta_acc} | {delta_hit} |"
        )

    # ─── Paired Wilcoxon on per-prompt throughput ─────────────────────────
    # Requires identical prompts in both runs (typically true when both runs
    # used the same task, seed and N). We pair by *position* (the
    # benchmarker emits rows in deterministic order under a fixed seed).
    base_rows = _load_request_rows(baseline_dir)
    spec_rows = _load_request_rows(spec_dir)
    sig_lines: List[str] = []
    for task in sorted(set(base_rows) & set(spec_rows)):
        b_tps = _per_prompt_throughput(base_rows[task])
        s_tps = _per_prompt_throughput(spec_rows[task])
        n = min(len(b_tps), len(s_tps))
        if n < 5:
            continue
        wlx = paired_wilcoxon(s_tps[:n], b_tps[:n])  # spec - baseline
        if wlx is None:
            continue
        sig = "✓" if wlx["p_value"] < 0.05 else "—"
        sig_lines.append(
            f"| `{task}` | {n} | {wlx['median_delta']:+.2f} | "
            f"{wlx['z']:+.2f} | {wlx['p_value']:.4f} | {wlx['effect_direction']} | {sig} |"
        )
    if sig_lines:
        out.append("")
        out.append(
            "## Paired Wilcoxon — per-prompt throughput (spec − baseline, tok/s)"
        )
        out.append("")
        out.append(
            "Pairs by *prompt position* under a fixed seed. ✓ = `p < 0.05` "
            "two-sided; pairs with zero diff are dropped (Wilcoxon convention)."
        )
        out.append("")
        out.append(
            "| Task | n | median Δ tok/s | z | p | direction | sig |"
        )
        out.append("|---|---:|---:|---:|---:|:-:|:-:|")
        out.extend(sig_lines)
    out.append("")
    return "\n".join(out)


def write_report(run_dir: Path, output_path: Optional[Path] = None) -> Path:
    md = render_run(run_dir)
    out = output_path or (run_dir / "report.md")
    out.write_text(md)
    return out


def write_compare(
    baseline_dir: Path, spec_dir: Path, output_path: Optional[Path] = None
) -> Path:
    md = render_compare(baseline_dir, spec_dir)
    out = output_path or (spec_dir / f"compare_vs_{baseline_dir.name}.md")
    out.write_text(md)
    return out
