# spec-eval

EAGLE-2 / **EAGLE-3** speculative-decoding eval pipeline for **SGLang**, managed with **UV**.

Mirrors the [SpecForge benchmarker](https://github.com/sgl-project/SpecForge/tree/main/benchmarks/benchmarker)
layout (base class + registry + per-dataset modules), but talks to SGLang over
HTTP `/generate` so the eval venv stays lean — no `import sglang` in the eval
code, just a subprocess that runs `python -m sglang.launch_server`.

For each (task × spec-config × seed) cell the pipeline reports:

- **Task accuracy** — per-benchmark quality metric (where gradable).
- **Acceptance metrics** — `accept_length`, Leviathan α, tree accept rate, accept histogram, per-prompt distribution.
- **Throughput / latency** — wall-clock TPS, e2e/inference/queue percentiles, decode TPS, p20 step time from `/server_info`.
- **Cache & memory pressure** — radix-cache hit rate, cached prompt tokens, retraction count.
- **Sanity flags** — cheap rule-based response checks (repetition, prompt-echo, system-leak, etc.).
- **Provenance** — drafter SHA256, git SHA, per-task sampling config, started UTC.

Full glossary in [§ Metrics reference](#metrics-reference).

---

## Table of contents

- [What's new vs SpecForge / the old `eval.py`](#whats-new-vs-specforge--the-old-evalpy)
- [Quick start](#quick-start)
- [Makefile](#makefile)
- [EAGLE-3 support](#eagle-3-support)
- [Benchmarks](#benchmarks)
- [CLI reference](#cli-reference)
- [Metrics reference](#metrics-reference)
- [Output layout](#output-layout)
- [Concurrency](#concurrency)
- [Architecture](#architecture)
- [Notes](#notes)

---

## What's new vs SpecForge / the old `eval.py`

| Feature | spec-eval | SpecForge `bench_eagle3.py` | old `eval.py` |
| --- | :-: | :-: | :-: |
| EAGLE-2 + EAGLE-3 in one CLI | ✓ | EAGLE-3 only | EAGLE-2 only |
| Auto-detect algorithm from drafter | ✓ | — | — |
| Algo-aware spec defaults (5/8/64 vs 3/1/4) | ✓ | — | — |
| `--config-list` multi-cell sweep | ✓ | ✓ | — |
| Async client (`--concurrency N`) | ✓ | — | — |
| Resume / skip-if-done + `--force` | ✓ | — | ✓ |
| Vocab guard pre-flight | ✓ | — | ✓ |
| Sanity checks (repetition, prompt-echo, system-leak) | ✓ | — | ✓ |
| Task accuracy reporting | ✓ | ✓ | — |
| Per-prompt distribution stats (p50/p90 of accept_length) | ✓ | — | ✓ |
| Both RoS + MoR α estimators reported | ✓ | — | ✓ |
| Cache hit-rate, retractions, queue-time percentiles | ✓ | — | — |
| `/server_info` step_time p20 + effective speed | ✓ | ✓ | — |
| Drafter SHA256 in provenance | ✓ | — | ✓ |
| Server-log tail on boot failure | ✓ | — | ✓ |
| `spec-eval report` (markdown) | ✓ | — | — |
| `spec-eval compare` (speedup table) | ✓ | — | — |
| Lean eval venv (no `import sglang`) | ✓ | — | ✓ |

---

## Quick start

```bash
# 1. Create the eval venv
make setup                          # = uv sync

# 2. On a GPU box, install sglang in the SAME venv
make install-sglang                 # = uv pip install "sglang[all]>=0.5"

# 3. EAGLE-2 against Llama-3.1-8B (algorithm auto-detected from drafter)
make run-eagle2 \
    TARGET=meta-llama/Llama-3.1-8B-Instruct \
    DRAFT=yuhuili/EAGLE-LLaMA3.1-Instruct-8B

# 4. EAGLE-3 against Llama-3.1-8B
make run-eagle3 \
    TARGET=meta-llama/Llama-3.1-8B-Instruct \
    DRAFT_EAGLE3=lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B

# 5. Sweep baseline + EAGLE-2 + EAGLE-3 trees in one boot
make run-sweep \
    TARGET=meta-llama/Llama-3.1-8B-Instruct \
    DRAFT_EAGLE3=lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B

# 6. Compare baseline vs spec
make compare BASELINE_DIR=results/baseline SPEC_DIR=results/eagle2
```

Or call the CLI directly:

```bash
uv run spec-eval run \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --draft  lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --tasks english --num-samples 50
```

---

## Makefile

```bash
make help            # list every target + variable
make setup           # uv sync (no sglang)
make install-sglang  # uv pip install "sglang[all]>=0.5"

# Inspection (no GPU)
make list-tasks
make audit TARGET=... DRAFT=...
make smoke-test                              # offline metric/helper checks

# Eval runs (variables: TARGET, DRAFT[_EAGLE3], TASKS, N, SEEDS, CONCURRENCY, RUN_NAME)
make run-baseline TARGET=... TASKS=english
make run-eagle2   TARGET=... DRAFT=...
make run-eagle3   TARGET=... DRAFT_EAGLE3=...
make run-sweep    TARGET=... DRAFT_EAGLE3=...

# Reports
make report  RUN_DIR=results/<dir>
make compare BASELINE_DIR=results/baseline SPEC_DIR=results/eagle3

# Hygiene
make lint            # best-effort if ruff installed
make typecheck       # best-effort if mypy installed
make clean           # remove __pycache__
make clean-results   # rm -rf results/
```

Append `EXTRA_ARGS="--force --verbose"` to forward flags to `spec-eval run`.
Override any variable inline: `make run-eagle2 TARGET=meta-llama/Llama-3.1-70B-Instruct N=200 CONCURRENCY=8`.

---

## EAGLE-3 support

EAGLE-3 is treated as a first-class algorithm:

- `--algorithm auto` (default) reads the draft's `config.json` (locally or
  from HF Hub) and looks for v3 fingerprints (`architectures` ending in
  `Eagle3`, `eagle_config.use_aux_hidden_state`, or `eagle3` in the name).
- Hyperparameter defaults follow SGLang's conventions:
  - **EAGLE-2**: `num_steps=5, eagle_topk=8, draft_tokens=64`
  - **EAGLE-3**: `num_steps=3, eagle_topk=1, draft_tokens=4`
- `make audit TARGET=<hf> DRAFT=<hf>` prints the detected algorithm + vocab
  guard + defaults without booting a server.

---

## Benchmarks

| Category | Benchmark | Accuracy? | Notes |
| --- | --- | :-: | --- |
| Coding | `humaneval` | ✓ | exec-based pass@1 |
| Coding | `livecodebench` | — | contamination-free; LCB grader external |
| QA | `simpleqa` | — | LLM-judge required for grading |
| QA | `financeqa` | — | context-grounded financial QA |
| QA | `mmlu` | ✓ | 4-way MCQ, CoT → `Answer: X` |
| QA / Science | `gpqa` | ✓ | grad-level MCQ (gated on HF) |
| Math | `gsm8k` | ✓ | 5-shot CoT, last-int extraction |
| Math | `math500` | ✓ | `\boxed{}` extraction |
| Math | `aime` | ✓ | competition math, 0-999 integer answer |
| Reasoning | `arc_challenge` | ✓ | grade-school science MCQ |
| Reasoning | `hellaswag` | ✓ | common-sense continuation MCQ |
| Reasoning / Chat | `mtbench` | — | 80 × 2-turn chat; no auto-judge |
| Chinese | `ceval` | ✓ | not in english/all-en presets |
| Multimodal | `mmstar` | (req VLM) | scaffold — needs a VLM client |

Presets (`--tasks <preset>`):

- `english` (default): `humaneval`, `simpleqa`, `gsm8k`, `mtbench` — one per category
- `reasoning`: `arc_challenge`, `hellaswag`, `mtbench`
- `all-en`: every English-text benchmark
- `all`: `all-en` + `ceval`, `mmstar`

Per-task overrides: `humaneval:100`, `mmlu:50:high_school_physics,abstract_algebra`

---

## CLI reference

```bash
uv run spec-eval --help
# Subcommands:
#   run         Boot SGLang + run benchmarks
#   report      Render markdown report from a run dir
#   compare     Side-by-side diff of two runs
#   audit       Inspect a draft (algo + vocab + defaults)
#   list-tasks  Print known benchmark names
```

### `run`

```bash
uv run spec-eval run \
  --target <hf> \
  --draft  <hf> \              # omit for baseline
  --algorithm auto \           # auto | EAGLE | EAGLE3
  --tasks english \            # english | reasoning | all-en | all | comma-list
  --num-samples 50 \
  --seeds 42,123 \             # stability replicates
  --concurrency 8 \            # async /generate pool
  --config-list 1,0,0,0 1,5,8,64 1,3,1,4 \   # baseline + EAGLE-2 + EAGLE-3
  --force \                    # overwrite finished cells
  --dry-run                    # print plan, don't execute
```

### `report` / `compare`

```bash
uv run spec-eval report results/<run-dir>          # writes results/<run-dir>/report.md
uv run spec-eval report results/<run-dir> --print  # also print to stdout
uv run spec-eval compare results/<baseline> results/<spec> --print
```

### `audit`

```bash
uv run spec-eval audit \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --draft  lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
# target  : meta-llama/Llama-3.1-8B-Instruct
# draft   : lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
# algo    : EAGLE3  (auto-detected)
# defaults: num_steps=3  topk=1  draft_tokens=4
# vocab   : vocab OK: 128256 (target == draft)
```

---

## Metrics reference

Every metric in `metrics.json` is documented here. They come from one of three
sources — SGLang's `/generate` `meta_info` block, the `/server_info`
endpoint, or per-cell client-side aggregation.

### Top-level — task & throughput

| Field | Unit | Where from | Definition |
| --- | --- | --- | --- |
| `accuracy` | fraction (0-1) | client | Per-task quality metric. `None` for ungradable tasks (mtbench, simpleqa, livecodebench). |
| `num_questions` | int | client | Number of prompts actually evaluated. |
| `num_valid_predictions` | int | client | Predictions that parsed successfully (denominator-safe for accuracy). |
| `latency` | seconds | client | Wall-clock for the task (start → all rows returned). With `--concurrency > 1` this is the gather time, not the sum of per-prompt latencies. |
| `output_throughput` | tokens/sec | client | `completion_tokens_sum / latency`. End-to-end TPS including queueing. |

### Acceptance — speculative decoding

The headline number is **`accept_length`** = `E[completed_tokens / verify_step]`.
We report it two ways because the literature is inconsistent:

| Estimator | Formula | Field(s) | Bias |
| --- | --- | --- | --- |
| **Ratio-of-sums (RoS)** | `sum(completion) / sum(verify_ct)` | `accept_length`, `alpha_per_token`, `alpha_normalized` | None — each verify step weighted equally (Horvitz–Thompson). What EAGLE/Leviathan papers report. **This is the headline.** |
| **Mean-of-ratios (MoR)** | `mean_p(completion_p / verify_p)` | `accept_length_mor`, `alpha_per_token_mor`, `alpha_normalized_mor` | Biased toward short prompts (1-2 lucky accepts dominate a per-prompt ratio). Kept as a **diagnostic** — the gap `RoS - MoR` is itself a signal that response lengths vary in the cell. |

From `accept_length`, the Leviathan-style α derivations:

| Field | Formula | Range | What it tells you |
| --- | --- | --- | --- |
| `alpha_per_token` | `(accept_length - 1) / num_steps` | [0, 1] | Per-drafter-token acceptance probability. Directly comparable across drafters at fixed `num_steps`. |
| `alpha_normalized` | `accept_length / (num_steps + 1)` | [0, 1] | Fraction of ideal speedup realised. `1.0` = every drafted token accepted. |
| `alpha_per_token_mor` / `alpha_normalized_mor` | same formulas, MoR input | [0, 1] | Diagnostic counterparts. |

Tree-level acceptance (the other "accept rate" SGLang exposes):

| Field | Formula | Range | What it tells you |
| --- | --- | --- | --- |
| `accept_rate_overall` | `sum(spec_accept_token_num) / sum(spec_draft_token_num)` | [0, 1] | "Fraction of all tree-proposed tokens accepted". **NOT Leviathan's α** — naturally low because most tree branches get culled. Compare across tree topologies (`topk`, `draft_tokens`). |

Token counts (sums across the task's prompts):

| Field | Definition |
| --- | --- |
| `completion_tokens_sum` | Total generated tokens (numerator of throughput). |
| `verify_tokens_sum` | Total `spec_verify_ct` (denominator of RoS `accept_length`). |
| `prompt_tokens_sum` | Total prompt tokens served. |
| `spec_accept_token_num_sum` / `spec_draft_token_num_sum` | Numerator / denominator of `accept_rate_overall`. |
| `spec_decode_active` | `verify_sum > 0`. `False` on baseline cells. |

Per-prompt distributions (the variance the cell-level mean hides):

| Field | Shape | Definition |
| --- | --- | --- |
| `accept_length_stats` | `{n, mean, p50, p90, min, max}` | Distribution of per-prompt `completion / verify_ct`. |
| `accept_rate_stats` | `{n, mean, p50, p90, min, max}` | Distribution of per-prompt server-side `spec_accept_rate`. |

Tree shape:

| Field | Definition |
| --- | --- |
| `spec_accept_histogram_total` | Element-wise sum of per-request `spec_accept_histogram`. `hist[k]` = how many verify steps accepted `k` drafted tokens. |
| `spec_accept_histogram_pmf` | Same vector, normalised to a probability mass function. |
| `spec_accepted_drafts_mean` | `Σ k·hist[k] / Σ hist[k]` — mean #accepted drafts per step. Equals `accept_length - 1`. |

### Throughput, latency & step-time

Wall-clock and percentiles from `meta_info`:

| Field | Unit | Definition |
| --- | --- | --- |
| `e2e_latency_p50` / `e2e_latency_p90` | seconds | SGLang's end-to-end request latency (queue + prefill + decode). |
| `inference_time_p50` | seconds | Compute-only time, excludes queueing. |
| `queue_time_p50` | seconds | Time spent waiting for the GPU. With `--concurrency > 1` this is your batching headroom. |
| `decode_throughput_p50` / `decode_throughput_p90` | tokens/sec | Per-request decode-phase TPS (`completion_tokens / inference_time`). |

Server-side step time, from `/server_info.step_time_dict`:

| Field | Unit | Definition |
| --- | --- | --- |
| `step_time_p20_ms` | milliseconds | 20th-percentile per-step decode time at the current batch size. SpecForge's preferred speed-from-server number — robust to first-step warm-up and tail outliers. **Populated only when `SGLANG_RECORD_STEP_TIME=1`** (the runner sets this automatically). |
| `effective_speed_tps` | tokens/sec | `(1000 / step_time_p20_ms) × accept_length` — server-side speed × acceptance, the apples-to-apples speed-up number. |

### Cache & memory pressure

| Field | Unit | Definition |
| --- | --- | --- |
| `cached_tokens_sum` | int | Total prompt tokens served from radix cache. |
| `cache_hit_rate` | fraction | `cached_tokens_sum / prompt_tokens_sum`. The runner calls `/flush_cache` between tasks so this doesn't leak across cells. |
| `total_retractions` | int | SGLang request-retraction count (KV-cache pressure / OOM-avoidance). A non-zero value flags that memory was tight; try `--mem-fraction-static 0.80`. |

### Sanity flags

Cheap rule-based response checks aggregated into `metrics.sanity = {n, n_insane, insane_fraction, flags_total}`:

| Flag | Trigger |
| --- | --- |
| `too_short` | Response < 5 chars. |
| `repetition` | A single char dominates the response (>85%). |
| `prompt_echo` | Response is >90% the user prompt verbatim. |
| `system_leak` | Response starts with `SYSTEM:` / `USER:` / `ASSISTANT:`, or quotes the system message. Catches SGLang frontend "default" template bugs. |
| `generation_empty` | `completion_tokens == 0` or `finish_reason == "error"`. |

A high `insane_fraction` usually means the chat template was applied wrong or
the drafter generated for a different model family — investigate before
trusting the throughput numbers.

### Provenance

| File | Field | Definition |
| --- | --- | --- |
| `config.json` | `target`, `draft` | HF id or local path. |
| `config.json` | `drafter_sha256` | SHA256 of `<draft>/model.safetensors`. `None` for HF ids that aren't downloaded locally. Pins the actual weight bytes, catches silent overwrites. |
| `config.json` | `algorithm` | Resolved algorithm — `EAGLE` or `EAGLE3`. |
| `config.json` | `cell_configs` | List of `(bs, num_steps, eagle_topk, draft_tokens, seed)` tuples actually executed. |
| `config.json` | `git_sha`, `started_utc` | Repo state + run timestamp. |
| `<cell>/cell.json` | full cell config | Echo of the cell's hyperparameters. |
| `<cell>/<task>/metrics.json` | `max_new_tokens`, `temperature`, `stop` | Per-task sampling config actually used (each `Benchmarker` declares its own defaults). |

### Console & report rendering

`print_results` is called at the end of every task and dumps the headline +
spec-decode + cache + sanity blocks to stdout. The same data is rendered as
markdown into `<run-dir>/report.md` with four tables per cell:

1. **Headline** — accuracy, latency, throughput, accept_length per task.
2. **Speculative-decoding detail** — α (RoS / MoR), accept_length (RoS / MoR), tree accept, histogram, drafts/step, step_time p20, effective speed.
3. **Per-prompt distribution** — accept_length and accept_rate `{min, p50, p90, max}`.
4. **Latency & cache detail** — e2e p50/p90, queue p50, decode TPS p50, cache hit %, prompt/cached tokens, retractions.
5. **Sanity flags** — only emitted when any flag fires.

`spec-eval compare <baseline> <spec>` adds a top-level speedup table:
`speedup = spec.throughput / baseline.throughput`, plus `Δ accuracy` and
`Δ cache_hit` in percentage points.

---

## Output layout

```
results/<target>__<draft>__<ts>/
├── config.json                # target/draft/algorithm/drafter_sha256/git_sha/...
├── summary.json               # all cells, all tasks, all metrics
├── report.md                  # auto-generated markdown
├── server_<cell>.log          # full SGLang server log per cell
└── <cell_id>/                 # e.g. bs1_steps5_topk8_dt64_s42  OR  baseline_bs1_s42
    ├── cell.json              # this cell's hyperparams
    └── <task>/
        ├── metrics.json       # everything in § Metrics reference
        └── requests.jsonl     # per-prompt: response, all meta_info fields, sanity flags
```

---

## Concurrency

`--concurrency N` switches the client to an async httpx pool that lets SGLang
batch on the server side.

- Defaults to **1** (sequential). Use this for clean per-prompt latency.
- `--concurrency 8` / `--concurrency 16` for headline throughput numbers.
- Caveat: `latency` becomes wall-clock-of-gather and `queue_time_p50` rises —
  that's the point.

---

## Architecture

```
spec_eval/
├── cli.py            # subcommands: run / report / compare / audit / list-tasks
├── runner.py         # EvalRun, CellConfig, EvalConfig — boots server, iterates cells
├── server.py         # SGLang lifecycle (subprocess, EAGLE/EAGLE3 args)
├── client.py         # SGLangClient (sync) + AsyncSGLangClient (concurrency)
├── algo_detect.py    # EAGLE vs EAGLE3 from config.json or filename
├── guards.py         # vocab_guard + config-tuple parsing
├── sanity.py         # rule-based response QC flags
├── metrics.py        # BenchmarkMetrics + compute_metrics + print_results
├── report.py         # markdown rendering for report / compare subcommands
├── registry.py       # @BENCHMARKS.register decorator
├── utils.py          # chat-template + cached downloads
└── tasks/
    ├── base.py       # Benchmarker ABC (sync + async run loops)
    ├── aime.py  arc_challenge.py  ceval.py  financeqa.py  gpqa.py
    ├── gsm8k.py  hellaswag.py  humaneval.py  livecodebench.py
    └── math500.py  mmlu.py  mmstar.py  mtbench.py  simpleqa.py

scripts/
└── smoke_test.py     # offline assertions over metrics/sanity/report (no GPU)

archive/
└── eval.py           # the original SpecForge-style script (kept for reference)
```

The eval venv has **no `import sglang`** anywhere — sglang lives behind the
`sglang.launch_server` subprocess and is reached over HTTP. That keeps
`uv sync` working on a dev box without CUDA/FlashInfer wheels, and lets you
upgrade sglang independently of the eval pipeline.

---

## Notes

- **HumanEval / LiveCodeBench** run untrusted model output in a subprocess
  with a 10s timeout. Don't run on a shared host without a sandbox.
- **MMStar** needs a VLM-aware client (image content); the scaffold is here
  for SpecForge parity but raises at request time on a text-only target.
- **MT-Bench** has no automatic judge; we only report spec-decode metrics.
  Add an LLM-judge step downstream if you need quality scores.
- **GPQA** and **LiveCodeBench** are gated on HF — `huggingface-cli login` first.
- **Step-time metric** requires `SGLANG_RECORD_STEP_TIME=1`; the runner sets
  this automatically. If `step_time_p20_ms` is `None` in `metrics.json`, your
  SGLang build is too old or the env var was overridden.
