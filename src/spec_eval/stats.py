"""Statistical helpers: bootstrap CIs, paired tests, percentiles, hashing.

Kept zero-deps (numpy only) so smoke_test.py can exercise everything without
hitting GPU or network. The runner+report+compare wire these in.

Three entry points:
    bootstrap_ci(xs, stat="mean", confidence=0.95, n_resamples=2000)
        Percentile bootstrap CI for a 1-D sample.
    paired_wilcoxon(xs, ys)
        Two-sided paired Wilcoxon signed-rank test (no scipy dependency).
    run_signature(config_dict)
        Deterministic short hash of (target, draft, tasks, cells, seeds, N)
        for run-deduplication / cross-run join keys.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np


# ─── Bootstrap CIs ──────────────────────────────────────────────────────────


_STAT_FNS: Dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "p50": lambda x: float(np.percentile(x, 50)),
    "p90": lambda x: float(np.percentile(x, 90)),
    "p99": lambda x: float(np.percentile(x, 99)),
}


def bootstrap_ci(
    xs: Sequence[float],
    stat: str = "mean",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    """Percentile bootstrap confidence interval for a 1-D sample.

    Returns ``{"point": float, "lo": float, "hi": float, "n": int}`` or
    ``None`` if the sample is too small.

    Why percentile bootstrap (not BCa): it's nonparametric, has zero
    distributional assumptions, and ~2000 resamples is enough for stable
    95% CIs at typical eval N (50-500). BCa is mildly tighter but
    requires jackknife + skewness correction — not worth the extra code
    for our use case where 50-prompt-per-cell samples already carry
    chunky CIs.

    Sample size guard: ``n >= 5`` (anything smaller is statistical theater).
    """
    arr = np.asarray([float(x) for x in xs if isinstance(x, (int, float))], dtype=float)
    if arr.size < 5:
        return None
    if stat not in _STAT_FNS:
        raise ValueError(f"unknown stat={stat!r}; choose from {list(_STAT_FNS)}")
    fn = _STAT_FNS[stat]

    rng = np.random.default_rng(seed)
    n = arr.size
    # vectorised resample: (n_resamples, n) draws with replacement
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = arr[idx]
    if stat == "mean":
        reps = samples.mean(axis=1)
    elif stat == "median":
        reps = np.median(samples, axis=1)
    elif stat == "p50":
        reps = np.percentile(samples, 50, axis=1)
    elif stat == "p90":
        reps = np.percentile(samples, 90, axis=1)
    else:  # p99
        reps = np.percentile(samples, 99, axis=1)
    alpha = 1.0 - confidence
    lo = float(np.percentile(reps, 100 * alpha / 2.0))
    hi = float(np.percentile(reps, 100 * (1 - alpha / 2.0)))
    return {
        "point": fn(arr),
        "lo": lo,
        "hi": hi,
        "n": int(n),
        "stat": stat,
        "confidence": confidence,
    }


# ─── Paired Wilcoxon (no scipy) ─────────────────────────────────────────────


def paired_wilcoxon(
    xs: Sequence[float],
    ys: Sequence[float],
) -> Optional[Dict[str, Any]]:
    """Two-sided paired Wilcoxon signed-rank test.

    Tests H0: median(x - y) == 0. Returns ``{"statistic", "p_value",
    "n_nonzero", "effect_direction"}`` or ``None`` if there aren't
    enough non-zero pairs.

    We use the normal approximation with tie/zero corrections — fine for
    n >= 20 (Mann's continuity-corrected formula). For smaller n the
    p-value is approximate; we surface ``n_nonzero`` so the caller can
    apply judgement.

    No scipy dependency: a paired Wilcoxon is ~30 lines.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    diffs = np.asarray(
        [float(a) - float(b) for a, b in zip(xs, ys)
         if isinstance(a, (int, float)) and isinstance(b, (int, float))],
        dtype=float,
    )
    diffs = diffs[~np.isnan(diffs)]
    # Drop zeros (Wilcoxon convention: discard ties on the difference)
    nz = diffs[diffs != 0]
    n = nz.size
    if n < 5:
        return None

    abs_d = np.abs(nz)
    # Average-rank handling for ties
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    signed = ranks * np.sign(nz)
    W_plus = float(signed[signed > 0].sum())
    W_minus = float(-signed[signed < 0].sum())
    W = min(W_plus, W_minus)

    # Normal approximation (works well for n >= 20)
    mean_W = n * (n + 1) / 4.0
    # tie correction
    _, tie_counts = np.unique(abs_d, return_counts=True)
    tie_term = sum(t * (t * t - 1) for t in tie_counts) / 48.0
    var_W = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if var_W <= 0:
        return None
    z = (W - mean_W) / math.sqrt(var_W)
    # two-sided p-value via standard normal CDF (use erfc, no scipy)
    p_value = math.erfc(abs(z) / math.sqrt(2.0))

    direction = "x > y" if W_plus > W_minus else ("x < y" if W_plus < W_minus else "tie")
    return {
        "statistic": W,
        "z": z,
        "p_value": p_value,
        "n_nonzero": int(n),
        "effect_direction": direction,
        "median_delta": float(np.median(nz)),
    }


# ─── Geometric / harmonic means (for cross-task aggregation) ────────────────


def geomean(xs: Iterable[float]) -> Optional[float]:
    """Geometric mean — the right average for ratios / speedups across tasks.

    Skips non-positive entries (geomean is undefined for zero/negative)."""
    pos = [float(x) for x in xs if isinstance(x, (int, float)) and x > 0]
    if not pos:
        return None
    return float(np.exp(np.mean(np.log(pos))))


def harmean(xs: Iterable[float]) -> Optional[float]:
    """Harmonic mean — the right average for rates expressed as 1/x."""
    pos = [float(x) for x in xs if isinstance(x, (int, float)) and x > 0]
    if not pos:
        return None
    return float(len(pos) / np.sum(1.0 / np.asarray(pos)))


# ─── Run signature (hash of "what was actually run") ────────────────────────


def run_signature(cfg: Dict[str, Any], *, length: int = 12) -> str:
    """Short, deterministic hash of the inputs that define "the same run".

    Used by ``--skip-if-exists`` to dedup identical re-invocations and by
    ``compare`` as a cross-run join key.

    Hashes (target, draft, drafter_sha256, algorithm, tasks list,
    cell_configs list, seeds, num_samples_default). Pointedly EXCLUDES:
        - run_name / output_dir (path is cosmetic)
        - started_utc / git_sha (we want re-runs at the same code to match)
        - server_overrides (host/port/dtype don't change what got measured;
          dtype is a real concern but it's stamped elsewhere)

    Use ``length=full`` (64) if you want SHA-256 in full.
    """
    keys = (
        "target", "draft", "drafter_sha256", "algorithm",
        "tasks", "cell_configs", "seeds", "num_samples_default",
    )
    blob = {k: cfg.get(k) for k in keys}
    # Sort-keys+separators for canonical JSON so {} order doesn't change the hash.
    s = json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    if length == "full" or length >= 64:
        return digest
    return digest[: int(length)]
