"""Offline smoke test for spec-eval's pure-Python machinery.

Exercises:
  1. ``compute_metrics``: ratio-of-sums and mean-of-ratios derivations,
     per-prompt distribution stats, baseline path (no spec), JSON round-trip,
     new ITL p99 + bootstrap CIs.
  2. ``_stats`` helper shape matches the old eval.py.
  3. ``EvalRun._drafter_sha256`` and ``EvalRun._log_tail`` on real files.
  4. ``sanity_flags`` flag taxonomy.
  5. Report rendering of a fake summary (markdown well-formed, ITL block
     present, bootstrap CI column present, Pareto CSV writer works).
  6. ``stats``: bootstrap CI shape, paired Wilcoxon direction, geomean /
     harmean correctness, run_signature determinism.
  7. ``ops``: doctor returns structured results, status renderer detects
     state, run signature comparison works.

Runs in <1 s, no GPU, no SGLang. Exit non-zero on any assertion failure.

Invoked via ``make smoke-test`` or ``uv run python scripts/smoke_test.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here / "src"))


def test_compute_metrics() -> None:
    from spec_eval.metrics import compute_metrics

    # Two rows engineered so RoS ≠ MoR (length variance).
    #   Row A: 5/1   ⇒ per-prompt accept_length = 5.0
    #   Row B: 100/25 ⇒ per-prompt accept_length = 4.0
    # RoS = 105/26 ≈ 4.0385; MoR = (5+4)/2 = 4.5.
    row_a = {
        "completion_tokens": 5, "spec_verify_ct": 1, "prompt_tokens": 10,
        "cached_tokens": 0, "spec_accept_token_num": 4, "spec_draft_token_num": 8,
        "spec_accept_rate": 0.50,
        "spec_accept_histogram": [0, 0, 0, 0, 0, 1],
        "e2e_latency": 0.20, "inference_time": 0.18,
        "queue_time": 0.001, "decode_throughput": 25.0, "total_retractions": 0,
    }
    row_b = {
        "completion_tokens": 100, "spec_verify_ct": 25, "prompt_tokens": 50,
        "cached_tokens": 10, "spec_accept_token_num": 75, "spec_draft_token_num": 200,
        "spec_accept_rate": 0.375,
        "spec_accept_histogram": [0, 0, 0, 5, 10, 10],
        "e2e_latency": 1.0, "inference_time": 0.95,
        "queue_time": 0.01, "decode_throughput": 100.0, "total_retractions": 0,
    }

    m = compute_metrics([row_a, row_b], latency=2.0, num_steps=5, step_time_p20_ms=10.0)

    assert abs(m.accept_length - 105 / 26) < 1e-9, m.accept_length
    assert abs(m.accept_length_mor - 4.5) < 1e-9, m.accept_length_mor
    assert abs(m.alpha_per_token - (m.accept_length - 1) / 5) < 1e-12
    assert abs(m.alpha_per_token_mor - (m.accept_length_mor - 1) / 5) < 1e-12
    assert abs(m.alpha_normalized - m.accept_length / 6) < 1e-12
    assert abs(m.alpha_normalized_mor - m.accept_length_mor / 6) < 1e-12
    assert abs(m.accept_length - m.accept_length_mor) > 0.4

    assert m.accept_length_stats == {
        "n": 2, "mean": 4.5, "p50": 4.5, "p90": 4.9, "min": 4.0, "max": 5.0,
    }
    assert m.accept_rate_stats["min"] == 0.375
    assert m.accept_rate_stats["max"] == 0.50
    assert abs(m.accept_rate_overall - 79 / 208) < 1e-12

    # JSON round-trip
    again = json.loads(json.dumps(m.to_dict()))
    assert again["accept_length_mor"] == m.accept_length_mor
    assert again["accept_length_stats"]["p50"] == 4.5

    # Baseline path: no verify ⇒ no α, no MoR
    baseline = compute_metrics(
        [{**row_a, "spec_verify_ct": 0, "completion_tokens": 50}],
        latency=1.0, num_steps=None,
    )
    assert baseline.spec_decode_active is False
    assert baseline.alpha_per_token is None
    assert baseline.alpha_per_token_mor is None
    assert baseline.accept_length_mor is None

    # Empty rows
    empty = compute_metrics([], latency=0.0, num_steps=5)
    assert empty.accept_length_stats is None
    print("  ✓ compute_metrics: RoS + MoR + distribution + JSON round-trip")


def test_stats_helper() -> None:
    from spec_eval.metrics import _stats

    s = _stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert set(s.keys()) == {"n", "mean", "p50", "p90", "min", "max"}
    assert s["n"] == 5
    assert s["mean"] == 3.0
    assert s["min"] == 1.0
    assert s["max"] == 5.0
    assert _stats([]) is None
    assert _stats([None, "x", float("nan")]) is not None  # NaN survives sort
    print("  ✓ _stats helper shape matches old eval.py")


def test_drafter_sha256_and_log_tail() -> None:
    from spec_eval.runner import EvalRun

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert EvalRun._drafter_sha256(str(tmp)) is None
        assert EvalRun._drafter_sha256(None) is None
        assert EvalRun._drafter_sha256("hf-org/some-model") is None

        sft = tmp / "model.safetensors"
        sft.write_bytes(b"hello-eagle-weights")
        sha = EvalRun._drafter_sha256(str(tmp))
        assert sha is not None and len(sha) == 64
        assert EvalRun._drafter_sha256(str(tmp)) == sha  # deterministic

        log = tmp / "server.log"
        log.write_text("\n".join(f"line {i}" for i in range(100)))
        tail = EvalRun._log_tail(str(log), n_lines=5).splitlines()
        assert tail == [f"line {i}" for i in range(95, 100)]
        assert EvalRun._log_tail("/nonexistent/path") == ""
    print("  ✓ drafter_sha256 + log_tail")


def test_sanity_flags() -> None:
    from spec_eval.sanity import aggregate_flags, sanity_flags

    f = sanity_flags("hello world", "what time is it?", finish_reason="stop", completion_tokens=2)
    assert all(v is False for v in f.values()), f

    f2 = sanity_flags(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "tell me something", finish_reason="stop", completion_tokens=10,
    )
    assert f2["repetition"] is True

    f3 = sanity_flags(
        "SYSTEM: you are a helpful assistant\nUSER: ...",
        "question", finish_reason="stop", completion_tokens=10,
    )
    assert f3["system_leak"] is True

    f4 = sanity_flags("", "question", finish_reason="stop", completion_tokens=0)
    assert f4["too_short"] is True and f4["generation_empty"] is True

    rolled = aggregate_flags([{"sanity": f}, {"sanity": f2}, {"sanity": f3}])
    assert rolled["n"] == 3
    assert rolled["n_insane"] == 2
    assert rolled["flags_total"]["repetition"] == 1
    assert rolled["flags_total"]["system_leak"] == 1
    print("  ✓ sanity_flags + aggregate")


def test_render_run() -> None:
    from spec_eval.report import render_run

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({
            "target": "Qwen/Qwen2.5-7B",
            "draft": "./drafters/eagle3-qwen",
            "drafter_sha256": "0123abcd" * 8,
            "algorithm": "EAGLE3",
            "tasks": ["humaneval", "gsm8k"],
            "concurrency": 4,
            "started_utc": "2026-05-16T12:00:00Z",
            "git_sha": "deadbeef",
        }))
        fake = {
            "num_questions": 50, "accuracy": 0.74,
            "latency": 12.5, "output_throughput": 234.0,
            "accept_length": 4.04, "accept_length_mor": 4.5,
            "spec_decode_active": True,
            "alpha_per_token": 0.608, "alpha_per_token_mor": 0.70,
            "alpha_normalized": 0.673, "alpha_normalized_mor": 0.75,
            "accept_rate_overall": 0.38,
            "spec_accept_histogram_total": [10, 20, 15, 5, 0, 1],
            "spec_accepted_drafts_mean": 1.6,
            "step_time_p20_ms": 4.2, "effective_speed_tps": 962.0,
            "accept_length_stats": {"n": 50, "mean": 4.5, "p50": 4.4,
                                    "p90": 5.2, "min": 2.0, "max": 6.1},
            "accept_rate_stats":   {"n": 50, "mean": 0.42, "p50": 0.40,
                                    "p90": 0.55, "min": 0.20, "max": 0.70},
            "e2e_latency_p50": 0.45, "e2e_latency_p90": 0.80,
            "queue_time_p50": 0.005, "decode_throughput_p50": 250.0,
            "cache_hit_rate": 0.15, "prompt_tokens_sum": 12000,
            "cached_tokens_sum": 1800, "total_retractions": 0,
            "sanity": {"n": 50, "n_insane": 0, "flags_total": {}},
        }
        (run / "summary.json").write_text(json.dumps({
            "cells": {"bs1_steps3_topk1_dt4_s42": {"tasks": {"humaneval": fake}}}
        }))
        md = render_run(run)
        for needle in (
            "Drafter SHA256",
            "α_per_token (RoS / MoR)",
            "Per-prompt distribution",
            "accept_length min",
            "Speculative-decoding detail",
        ):
            assert needle in md, f"missing section: {needle!r}"
    print("  ✓ render_run includes all new sections")


def test_metrics_new_fields() -> None:
    """ITL p99 + bootstrap CI fields populate when there's enough signal."""
    from spec_eval.metrics import compute_metrics

    rows = []
    for i in range(50):
        rows.append({
            "completion_tokens": 100 + i,
            "spec_verify_ct": 30 + (i % 5),
            "prompt_tokens": 50,
            "cached_tokens": 10,
            "spec_accept_token_num": 80 + i,
            "spec_draft_token_num": 200,
            "spec_accept_rate": 0.6,
            "spec_accept_histogram": [10, 5, 3, 2, 1],
            "e2e_latency": 1.0 + i * 0.02,
            "inference_time": 0.9 + i * 0.018,
            "queue_time": 0.05,
            "decode_throughput": 200 - i,
            "total_retractions": 0,
        })
    m = compute_metrics(rows, latency=10.0, num_steps=5, step_time_p20_ms=4.5)
    assert m.e2e_latency_p99 is not None
    assert m.itl_ms_p50 is not None and m.itl_ms_p99 >= m.itl_ms_p50
    assert m.accept_length_ci is not None
    assert m.accept_length_ci["lo"] <= m.accept_length_ci["point"] <= m.accept_length_ci["hi"]
    assert m.output_throughput_ci is not None
    assert m.itl_used_e2e_fallback == 0  # inference_time was present
    print("  ✓ ITL p50/p90/p99 + bootstrap CIs populate on N=50")


def test_metrics_itl_fallback() -> None:
    """When SGLang is booted without --enable-metrics, only e2e_latency is in
    meta_info. ITL must still populate (falling back to e2e_latency) and the
    fallback counter must record it so reports can flag the over-count."""
    from spec_eval.metrics import compute_metrics

    rows = [{
        "completion_tokens": 100, "spec_verify_ct": 30, "prompt_tokens": 50,
        "cached_tokens": 0, "spec_accept_token_num": 80,
        "spec_draft_token_num": 200, "spec_accept_rate": 0.6,
        "spec_accept_histogram": [5, 2, 1, 1],
        "e2e_latency": 1.0,                # only this — like a no-flag run
        "inference_time": None, "queue_time": None, "decode_throughput": None,
        "total_retractions": 0,
    } for _ in range(20)]
    m = compute_metrics(rows, latency=20.0, num_steps=5)
    assert m.itl_ms_p50 is not None, "ITL fallback should produce a value"
    assert m.itl_used_e2e_fallback == 20, \
        f"all 20 rows should have hit the fallback, got {m.itl_used_e2e_fallback}"
    assert m.output_throughput_ci is not None, \
        "throughput CI should still populate from the fallback denom"
    print("  ✓ ITL falls back to e2e_latency when inference_time is absent")


def test_stats_module() -> None:
    """The new spec_eval.stats helpers: bootstrap_ci, paired_wilcoxon, etc."""
    import random
    from spec_eval.stats import (
        bootstrap_ci, geomean, harmean, paired_wilcoxon, run_signature,
    )

    random.seed(0)
    xs = [random.gauss(0.0, 1.0) for _ in range(200)]
    ci = bootstrap_ci(xs, stat="mean")
    assert set(ci) == {"point", "lo", "hi", "n", "stat", "confidence"}
    assert ci["lo"] < ci["point"] < ci["hi"]
    assert ci["n"] == 200
    # tiny samples ⇒ None
    assert bootstrap_ci([1.0, 2.0]) is None

    # paired Wilcoxon: x clearly > y
    fast = [random.gauss(10.0, 2.0) for _ in range(40)]
    slow = [random.gauss(6.0, 2.0) for _ in range(40)]
    w = paired_wilcoxon(fast, slow)
    assert w["p_value"] < 1e-5
    assert w["effect_direction"] == "x > y"
    assert w["median_delta"] > 0
    # tie path: both samples identical ⇒ all diffs zero ⇒ None
    assert paired_wilcoxon([1.0] * 30, [1.0] * 30) is None

    # geomean / harmean exact answers
    assert abs(geomean([1, 2, 4, 8]) - (1 * 2 * 4 * 8) ** 0.25) < 1e-9
    assert abs(harmean([1, 2, 4, 8]) - 4 / (1 + 0.5 + 0.25 + 0.125)) < 1e-9
    assert geomean([]) is None and harmean([]) is None
    assert geomean([0, -1]) is None  # filters non-positive

    # run_signature is deterministic and exclusively depends on the keys we care about
    cfg = {"target": "a", "draft": "b", "drafter_sha256": "x" * 16,
           "algorithm": "EAGLE", "tasks": ["t1", "t2"],
           "cell_configs": [{"bs": 1, "num_steps": 5}], "seeds": [42],
           "num_samples_default": 50}
    s1 = run_signature(cfg)
    s2 = run_signature({**cfg, "started_utc": "irrelevant", "git_sha": "noise"})
    s3 = run_signature({**cfg, "seeds": [43]})  # actual change
    assert s1 == s2, "ignored fields must not change the signature"
    assert s1 != s3, "seeds must affect signature"
    assert len(s1) == 12
    print("  ✓ stats: bootstrap CI, paired Wilcoxon, geomean/harmean, run_signature")


def test_ops_status() -> None:
    """`spec-eval status` snapshot detects completed / in-flight / finished."""
    from spec_eval.ops import _scan_run_dir, render_status

    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "demo"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({
            "target": "t", "draft": "d", "algorithm": "EAGLE",
            "tasks": ["humaneval", "gsm8k", "simpleqa", "mtbench"],
            "cell_configs": [{"bs": 1}],
            "seeds": [42], "num_samples_default": 50, "concurrency": 1,
            "server_overrides": {}, "drafter_sha256": None,
            "started_utc": "2026-05-18T10:00:00Z", "run_signature": "x" * 12,
        }))
        cell = run / "bs1_cellA"
        cell.mkdir()
        (cell / "cell.json").write_text("{}")
        for done in ("humaneval", "gsm8k"):
            (cell / done).mkdir()
            (cell / done / "metrics.json").write_text("{}")
        (cell / "simpleqa").mkdir()
        (cell / "simpleqa" / "requests.jsonl").write_text("{}\n")

        snap = _scan_run_dir(run)
        assert snap["state"] == "running"
        assert len(snap["completed"]) == 2
        assert len(snap["in_flight"]) == 1
        md = render_status(run)
        assert "running" in md and "In-flight" in md

        # Mark finished — now state should flip.
        (run / "summary.json").write_text("{}")
        snap = _scan_run_dir(run)
        assert snap["state"] == "finished"
    print("  ✓ ops.status detects running/finished + counts task cells")


def test_ops_doctor() -> None:
    """Doctor returns structured CheckResults; ports & disk should pass locally."""
    from spec_eval.ops import run_doctor

    with tempfile.TemporaryDirectory() as td:
        results, _ = run_doctor(port=0, output_dir=Path(td))
        names = {r.name for r in results}
        for required in ("python >= 3.10", "numpy", "httpx", "transformers",
                         "datasets", "port 0 free"):
            assert required in names, f"missing doctor check {required!r}"
        # disk check on tempdir should pass
        disk = next(r for r in results if r.name.startswith("disk free under"))
        assert disk.ok, f"disk check unexpectedly failed: {disk.detail}"
    print("  ✓ ops.doctor returns structured CheckResults")


def test_pareto_csv() -> None:
    """write_pareto_csv emits one row per (run, cell, task)."""
    import csv as _csv
    from spec_eval.report import write_pareto_csv

    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "r1"
        run.mkdir()
        (run / "summary.json").write_text(json.dumps({
            "cells": {
                "cellA": {"tasks": {
                    "humaneval": {"accept_length": 4.1, "output_throughput": 250.0,
                                  "accuracy": 0.7, "itl_ms_p50": 4.0,
                                  "alpha_per_token": 0.62, "spec_decode_active": True},
                    "gsm8k":     {"accept_length": 3.8, "output_throughput": 200.0,
                                  "accuracy": 0.55, "itl_ms_p50": 5.0,
                                  "alpha_per_token": 0.56, "spec_decode_active": True},
                }},
            }
        }))
        out_csv = Path(td) / "pareto.csv"
        write_pareto_csv([run], out_csv)
        with open(out_csv) as f:
            reader = list(_csv.DictReader(f))
        assert len(reader) == 2
        assert {r["task"] for r in reader} == {"humaneval", "gsm8k"}
        assert all(r["run"] == "r1" for r in reader)
    print("  ✓ write_pareto_csv emits one row per (run, cell, task)")


def test_compare_paired_wilcoxon() -> None:
    """`render_compare` finds requests.jsonl rows and emits the Wilcoxon block."""
    from spec_eval.report import render_compare

    def _mk_run(td: Path, name: str, mean_itime: float) -> Path:
        run = td / name
        run.mkdir()
        (run / "config.json").write_text("{}")
        (run / "summary.json").write_text(json.dumps({
            "cells": {"cellA": {"tasks": {"humaneval": {
                "output_throughput": 100.0 / mean_itime,
                "accept_length": 1.0, "spec_decode_active": False,
                "num_questions": 30,
            }}}}
        }))
        cell = run / "cellA"; cell.mkdir()
        task = cell / "humaneval"; task.mkdir()
        rows = []
        for i in range(30):
            rows.append({"completion_tokens": 100,
                         "inference_time": mean_itime + 0.01 * i})
        (task / "requests.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        return run

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = _mk_run(td, "baseline", mean_itime=1.0)
        spec = _mk_run(td, "spec", mean_itime=0.5)  # 2× faster ⇒ spec > baseline
        md = render_compare(base, spec)
        assert "Paired Wilcoxon" in md
        # spec is faster ⇒ median delta positive ⇒ direction "x > y"
        assert "x > y" in md
    print("  ✓ render_compare emits paired Wilcoxon block on requests.jsonl pairs")


def main() -> int:
    _add_src_to_path()
    print("spec-eval offline smoke test")
    print("─" * 50)
    test_compute_metrics()
    test_stats_helper()
    test_drafter_sha256_and_log_tail()
    test_sanity_flags()
    test_render_run()
    test_metrics_new_fields()
    test_metrics_itl_fallback()
    test_stats_module()
    test_ops_status()
    test_ops_doctor()
    test_pareto_csv()
    test_compare_paired_wilcoxon()
    print("─" * 50)
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
