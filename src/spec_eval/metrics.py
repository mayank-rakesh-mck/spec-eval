"""Metrics aggregation. Mirrors SpecForge's BenchmarkMetrics surface but adds
every spec-decode / latency / cache field SGLang exposes in ``meta_info``.

Field reference (see also https://github.com/sgl-project/sglang/pull/18332):

  meta_info:
    spec_verify_ct, spec_accept_length, spec_accept_rate,
    spec_accept_token_num, spec_draft_token_num, spec_accept_histogram,
    completion_tokens, prompt_tokens, cached_tokens,
    e2e_latency, inference_time, queue_time, decode_throughput,
    total_retractions, finish_reason, id

  /server_info -> internal_states[0].step_time_dict[<bs>]
    array of per-step decode times in seconds; 20th-pct is SpecForge's
    preferred speed-from-server number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkMetrics:
    """One row per task run. SpecForge-compatible names plus our extras."""

    # ─── SpecForge-compatible top-level fields ─────────────────────────────
    latency: float = 0.0
    output_throughput: float = 0.0
    accept_length: float = 1.0  # ratio-of-sums across the cell (headline)
    accuracy: Optional[float] = None
    num_questions: int = 0
    num_valid_predictions: int = 0

    # ─── Per-task generation config (provenance) ───────────────────────────
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None

    # ─── Token-count sums (raw inputs for derivations) ─────────────────────
    completion_tokens_sum: int = 0
    verify_tokens_sum: int = 0
    prompt_tokens_sum: int = 0
    cached_tokens_sum: int = 0
    spec_accept_token_num_sum: int = 0
    spec_draft_token_num_sum: int = 0
    spec_decode_active: bool = False

    # ─── Derived: cache hit rate ───────────────────────────────────────────
    cache_hit_rate: Optional[float] = None

    # ─── Spec-decode derived: Leviathan-style α + tree acceptance ──────────
    accept_rate_overall: Optional[float] = None
    """Tree-fraction acceptance: spec_accept_token_num / spec_draft_token_num.
    NOT Leviathan's α — this is "fraction of all tree-proposed tokens accepted",
    naturally low because most tree branches get culled."""

    alpha_per_token: Optional[float] = None
    """Leviathan-style HEADLINE (ratio-of-sums): ``(accept_length - 1) / num_steps``."""

    alpha_normalized: Optional[float] = None
    """Leviathan-style HEADLINE (ratio-of-sums): ``accept_length / (num_steps + 1)``."""

    # ─── Mean-of-ratios diagnostics ────────────────────────────────────────
    # Old eval.py reports both estimators of E[accept_length]:
    #   (A) ratio-of-sums  — preferred, headline (set above)
    #   (B) mean-of-ratios — biased toward short prompts, kept as diagnostic
    # A large gap (A) - (B) ⇒ response-length variance within the cell.
    accept_length_mor: Optional[float] = None
    """Mean-of-ratios E[accept_length]: ``mean_p(completion_p / verify_p)``."""

    alpha_per_token_mor: Optional[float] = None
    """Mean-of-ratios α: ``(accept_length_mor - 1) / num_steps``."""

    alpha_normalized_mor: Optional[float] = None
    """Mean-of-ratios α_normalized: ``accept_length_mor / (num_steps + 1)``."""

    # ─── Per-prompt distribution stats (n/mean/p50/p90/min/max) ────────────
    accept_length_stats: Optional[Dict[str, float]] = None
    """Distribution of per-prompt ``completion_tokens / spec_verify_ct``.
    Headline ``accept_length`` is the ratio-of-sums; this is the per-prompt
    variance behind it (papers usually report just the mean — we expose both)."""

    accept_rate_stats: Optional[Dict[str, float]] = None
    """Distribution of per-prompt SGLang ``spec_accept_rate`` (server-side tree
    acceptance fraction). Headline ``accept_rate_overall`` is the ratio-of-sums;
    this is the per-prompt spread."""

    # ─── Histogram (PMF over #accepted-drafts-per-step) ────────────────────
    spec_accept_histogram_total: Optional[List[int]] = None  # NEW — summed
    spec_accept_histogram_pmf: Optional[List[float]] = None  # NEW — normalised
    spec_accepted_drafts_mean: Optional[float] = None
    """Mean number of accepted drafts per step (= accept_length - 1)."""

    # ─── Latency percentiles (from meta_info) ──────────────────────────────
    e2e_latency_p50: Optional[float] = None
    e2e_latency_p90: Optional[float] = None
    inference_time_p50: Optional[float] = None
    queue_time_p50: Optional[float] = None
    decode_throughput_p50: Optional[float] = None
    decode_throughput_p90: Optional[float] = None

    # ─── /server_info ──────────────────────────────────────────────────────
    step_time_p20_ms: Optional[float] = None
    effective_speed_tps: Optional[float] = None
    """SpecForge convention: ``1 / step_time_p20 * accept_length``."""

    # ─── Memory pressure ───────────────────────────────────────────────────
    total_retractions: int = 0

    # ─── Sanity rollup (per-row flag totals) ───────────────────────────────
    sanity: Optional[Dict[str, Any]] = None

    # ─── Optional categorical breakdown ────────────────────────────────────
    categorical_performance: Optional[Dict[str, "BenchmarkMetrics"]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _percentile(xs: List[float], pct: float) -> Optional[float]:
    if not xs:
        return None
    return float(np.percentile(xs, pct))


def _stats(xs: List[float]) -> Optional[Dict[str, float]]:
    """``{n, mean, p50, p90, min, max}`` over a list of finite numbers.

    Matches the shape the old eval.py used (lines 356-367) so downstream
    comparison code can read either summary."""
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    arr = np.asarray(xs, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _sum_histograms(hists: List[List[int]]) -> Optional[List[int]]:
    """Element-wise sum of variable-length histograms (pad to the longest)."""
    hists = [h for h in hists if h]
    if not hists:
        return None
    max_len = max(len(h) for h in hists)
    out = [0] * max_len
    for h in hists:
        for i, v in enumerate(h):
            out[i] += int(v)
    return out


def compute_metrics(
    rows: List[Dict[str, Any]],
    latency: float,
    answer_keys: Optional[List[str]] = None,  # noqa: ARG001 — multi-turn already flattened
    num_steps: Optional[int] = None,
    step_time_p20_ms: Optional[float] = None,
) -> BenchmarkMetrics:
    """Aggregate per-request rows into a BenchmarkMetrics.

    Each row carries the per-request meta_info fields captured by
    :func:`spec_eval.tasks.base.Benchmarker._row_from_result`.

    Args:
        rows: per-request dicts.
        latency: cell wall-clock (seconds).
        num_steps: ``--speculative-num-steps`` for the current cell. Required to
            derive Leviathan-style α. ``None`` ⇒ skip α derivation.
        step_time_p20_ms: 20th-percentile decode step time in milliseconds,
            pulled from ``/server_info``. ``None`` if SGLang didn't record it.
    """
    if not rows:
        return BenchmarkMetrics(latency=latency)

    # Sums
    completion_sum = int(sum(r.get("completion_tokens", 0) or 0 for r in rows))
    verify_sum = int(sum(r.get("spec_verify_ct", 0) or 0 for r in rows))
    prompt_sum = int(sum(r.get("prompt_tokens", 0) or 0 for r in rows))
    cached_sum = int(sum(r.get("cached_tokens", 0) or 0 for r in rows))
    spec_acc_sum = int(sum(r.get("spec_accept_token_num", 0) or 0 for r in rows))
    spec_drft_sum = int(sum(r.get("spec_draft_token_num", 0) or 0 for r in rows))
    retractions = int(sum(r.get("total_retractions", 0) or 0 for r in rows))

    spec_active = verify_sum > 0
    accept_length = completion_sum / verify_sum if spec_active else 1.0
    output_throughput = completion_sum / latency if latency > 0 else 0.0
    cache_hit_rate = (cached_sum / prompt_sum) if prompt_sum > 0 else None
    accept_rate_overall = (
        (spec_acc_sum / spec_drft_sum) if spec_drft_sum > 0 else None
    )

    # Per-prompt distributions (MoR inputs + variance for the report)
    per_prompt_accept_length: List[float] = []
    for r in rows:
        ct = r.get("completion_tokens") or 0
        vc = r.get("spec_verify_ct") or 0
        if ct > 0 and vc > 0:
            per_prompt_accept_length.append(ct / vc)
    per_prompt_accept_rate = [
        float(r["spec_accept_rate"])
        for r in rows
        if isinstance(r.get("spec_accept_rate"), (int, float))
    ]
    accept_length_stats = _stats(per_prompt_accept_length)
    accept_rate_stats = _stats(per_prompt_accept_rate)

    # Leviathan-style derivations (need num_steps).
    # Headline = ratio-of-sums; MoR = mean of per-prompt ratios (diagnostic).
    # A large gap between the two ⇒ response-length variance within the cell.
    alpha_per_token: Optional[float] = None
    alpha_normalized: Optional[float] = None
    accept_length_mor: Optional[float] = None
    alpha_per_token_mor: Optional[float] = None
    alpha_normalized_mor: Optional[float] = None
    if spec_active and num_steps and num_steps > 0:
        alpha_per_token = (accept_length - 1.0) / num_steps
        alpha_normalized = accept_length / (num_steps + 1.0)
        if accept_length_stats:
            accept_length_mor = accept_length_stats["mean"]
            alpha_per_token_mor = (accept_length_mor - 1.0) / num_steps
            alpha_normalized_mor = accept_length_mor / (num_steps + 1.0)

    # Histogram aggregation
    histograms = [r.get("spec_accept_histogram") for r in rows]
    hist_total = _sum_histograms([h for h in histograms if isinstance(h, list)])
    hist_pmf: Optional[List[float]] = None
    drafts_mean: Optional[float] = None
    if hist_total:
        total = sum(hist_total)
        if total > 0:
            hist_pmf = [c / total for c in hist_total]
            drafts_mean = sum(i * c for i, c in enumerate(hist_total)) / total

    # Latency percentiles — only when meta_info exposes them
    e2e = [r["e2e_latency"] for r in rows if isinstance(r.get("e2e_latency"), (int, float))]
    inf = [r["inference_time"] for r in rows if isinstance(r.get("inference_time"), (int, float))]
    qtime = [r["queue_time"] for r in rows if isinstance(r.get("queue_time"), (int, float))]
    dec_tp = [r["decode_throughput"] for r in rows if isinstance(r.get("decode_throughput"), (int, float))]

    effective_speed = None
    if step_time_p20_ms and step_time_p20_ms > 0:
        effective_speed = (1000.0 / step_time_p20_ms) * accept_length

    return BenchmarkMetrics(
        latency=latency,
        output_throughput=output_throughput,
        accept_length=accept_length,
        num_questions=len(rows),
        completion_tokens_sum=completion_sum,
        verify_tokens_sum=verify_sum,
        prompt_tokens_sum=prompt_sum,
        cached_tokens_sum=cached_sum,
        spec_accept_token_num_sum=spec_acc_sum,
        spec_draft_token_num_sum=spec_drft_sum,
        spec_decode_active=spec_active,
        cache_hit_rate=cache_hit_rate,
        accept_rate_overall=accept_rate_overall,
        alpha_per_token=alpha_per_token,
        alpha_normalized=alpha_normalized,
        accept_length_mor=accept_length_mor,
        alpha_per_token_mor=alpha_per_token_mor,
        alpha_normalized_mor=alpha_normalized_mor,
        accept_length_stats=accept_length_stats,
        accept_rate_stats=accept_rate_stats,
        spec_accept_histogram_total=hist_total,
        spec_accept_histogram_pmf=hist_pmf,
        spec_accepted_drafts_mean=drafts_mean,
        e2e_latency_p50=_percentile(e2e, 50),
        e2e_latency_p90=_percentile(e2e, 90),
        inference_time_p50=_percentile(inf, 50),
        queue_time_p50=_percentile(qtime, 50),
        decode_throughput_p50=_percentile(dec_tp, 50),
        decode_throughput_p90=_percentile(dec_tp, 90),
        step_time_p20_ms=step_time_p20_ms,
        effective_speed_tps=effective_speed,
        total_retractions=retractions,
    )


def print_results(
    metrics_list: List[BenchmarkMetrics],
    benchmark_name: str,
    show_accuracy: bool = False,
) -> None:
    """Console summary. Same overall shape as SpecForge so reports stay diff-able."""
    if not metrics_list:
        print(f"[{benchmark_name}] no metrics to print")
        return
    m = metrics_list[0]
    avg_lat = float(np.mean([x.latency for x in metrics_list]))
    avg_tps = float(np.mean([x.output_throughput for x in metrics_list]))
    avg_al = float(np.mean([x.accept_length for x in metrics_list]))

    print()
    print("=" * 64)
    print(f"  {benchmark_name} — Evaluation Results")
    print("=" * 64)
    print(f"  Questions               : {m.num_questions}")
    if show_accuracy:
        accs = [x.accuracy for x in metrics_list if x.accuracy is not None]
        if accs:
            avg_acc = float(np.mean(accs))
            print(f"  Accuracy                : {avg_acc:.4f}  ({avg_acc * 100:.2f}%)")
        else:
            print(f"  Accuracy                : n/a")
    print(f"  Latency (s, avg)        : {avg_lat:.3f}")
    print(f"  Output throughput (tps) : {avg_tps:.2f}")
    print(f"  Accept length (avg)     : {avg_al:.3f}")
    if m.spec_decode_active:
        def _fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"
        print(f"  Accept length (overall) : {_fmt(m.accept_length)}    "
              f"(MoR mean: {_fmt(m.accept_length_mor)})")
        if m.accept_length_stats:
            s = m.accept_length_stats
            print(f"    per-prompt           : "
                  f"min={_fmt(s.get('min'))}  p50={_fmt(s.get('p50'))}  "
                  f"p90={_fmt(s.get('p90'))}  max={_fmt(s.get('max'))}  (n={s.get('n')})")
        print(f"  α_per_token  (Leviathan): {_fmt(m.alpha_per_token)}    "
              f"(MoR: {_fmt(m.alpha_per_token_mor)})")
        print(f"  α_normalized (Leviathan): {_fmt(m.alpha_normalized)}    "
              f"(MoR: {_fmt(m.alpha_normalized_mor)})")
        print(f"  Tree accept rate        : {_fmt(m.accept_rate_overall)}  "
              f"(= spec_accept_token_num / spec_draft_token_num)")
        if m.accept_rate_stats:
            s = m.accept_rate_stats
            print(f"    per-prompt           : "
                  f"min={_fmt(s.get('min'))}  p50={_fmt(s.get('p50'))}  "
                  f"p90={_fmt(s.get('p90'))}  max={_fmt(s.get('max'))}  (n={s.get('n')})")
        if m.spec_accept_histogram_total:
            print(f"  Accept histogram (sum)  : {m.spec_accept_histogram_total}")
            print(f"  Mean #accepted drafts/step: {_fmt(m.spec_accepted_drafts_mean)}")
    if m.cache_hit_rate is not None:
        print(f"  Cache hit rate          : {m.cache_hit_rate * 100:.1f}%  "
              f"({m.cached_tokens_sum} / {m.prompt_tokens_sum} prompt tokens)")
    if m.step_time_p20_ms is not None:
        print(f"  Step time (p20, ms)     : {m.step_time_p20_ms:.2f}")
        print(f"  Effective speed (tps)   : {m.effective_speed_tps:.2f}")
    if m.e2e_latency_p50 is not None:
        print(f"  e2e latency p50 / p90   : {m.e2e_latency_p50:.3f}s / "
              f"{m.e2e_latency_p90:.3f}s")
    if m.queue_time_p50 is not None and m.queue_time_p50 > 0:
        print(f"  queue time p50          : {m.queue_time_p50 * 1000:.1f} ms")
    if m.total_retractions:
        print(f"  ⚠ retractions           : {m.total_retractions}")
    if m.sanity and m.sanity.get("n_insane"):
        print(f"  ⚠ sanity flags          : {m.sanity['n_insane']}/{m.sanity['n']} "
              f"({m.sanity.get('flags_total')})")
    print("=" * 64)
    print()
