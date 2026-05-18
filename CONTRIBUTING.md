# Contributing to spec-eval

Thanks for the interest. This file covers:

- [Adding a new benchmark](#adding-a-new-benchmark) — the most common task.
- [Project layout](#project-layout) — where things live.
- [Coding conventions](#coding-conventions) — style + invariants.
- [Testing](#testing) — what `make smoke-test` covers and how to extend it.
- [Submitting changes](#submitting-changes) — commits + PRs.

---

## Adding a new benchmark

The pipeline is built around one ABC: `spec_eval.tasks.base.Benchmarker`.
A new benchmark is one ~30-line file under `src/spec_eval/tasks/`. The
CLI picks it up automatically via the `@BENCHMARKS.register` decorator.

### 1. Scaffold

Create `src/spec_eval/tasks/<name>.py`:

```python
"""<NAME> — <one-sentence description>.

Dataset source: <HF id or URL>
Gradeable?     : <yes/no>
Why include it : <one-sentence rationale>
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from spec_eval.registry import BENCHMARKS
from spec_eval.tasks.base import Benchmarker


@BENCHMARKS.register("<name>")
class <Name>Benchmarker(Benchmarker):
    NAME = "<name>"
    SHOW_ACCURACY = True   # False for tasks we can't grade (mtbench, simpleqa)

    def __init__(self, num_samples: Optional[int] = None, subset=None, seed: int = 0):
        # The runner passes `seed` through — forward it to the base so
        # data sub-sampling is deterministic across re-runs.
        super().__init__(num_samples, subset, seed)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """Return (questions, labels). Same length; labels=None entries are OK.

        Honour ``self.num_samples`` and ``self.subset``. Use
        ``spec_eval.utils.download_and_cache`` for raw URLs or HF ``datasets``
        for everything else.
        """
        ...

    def build_messages(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """OpenAI-style chat messages for a single prompt.

        Set ``{"raw": True}`` on the first message to skip chat-templating
        (few-shot GSM8K shape — sends the bare content to /generate).
        """
        ...

    # Optional overrides — defaults are usually fine

    def get_max_new_tokens(self) -> int:
        return 2048

    def get_temperature(self) -> float:
        return 0.0  # greedy by default; spec-decode is provably equivalent

    def get_stop(self) -> Optional[List[str]]:
        return None  # e.g. ["Question:", "<|eot_id|>"]

    def get_system_prompt(self) -> Optional[str]:
        return None  # set if you use one in build_messages; sanity check uses it

    # Only override these if the task is gradable

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Any:
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        return None
```

### 2. Wire it into presets (optional)

If the task should show up in `--tasks english` / `reasoning` / `all-en`,
add it to the corresponding list in `src/spec_eval/runner.py`:

```python
ENGLISH_PRESET = ["humaneval", "simpleqa", "gsm8k", "mtbench", "<name>"]
```

The CLI picks individual tasks up automatically by registry name, so you
can always run `--tasks <name>` without touching presets.

### 3. Smoke-test

```bash
make smoke-test          # the offline assertions still pass?
make list-tasks          # is the new task listed?
make audit TARGET=... DRAFT=...   # does the vocab guard still pass?
```

If you can run on GPU, do a real 5-sample run:

```bash
uv run spec-eval run \
  --target meta-llama/Llama-3.1-8B-Instruct \
  --tasks <name> --num-samples 5 --dry-run    # plan only
```

### Worked example: GSM8K

The shortest fully-featured benchmarker is `src/spec_eval/tasks/gsm8k.py`
(94 lines). It demonstrates the three patterns most tasks need:

- **Raw few-shot prompting** (`raw: True` on the message)
- **Regex-extracted gold answer** (`extract_answer` + helper functions)
- **Per-task `max_new_tokens` + `stop`** (avoids drift past `Question:`)

Skim that before writing your own and you'll typically be done in 30
minutes.

### Pitfalls

- **Different vocab between target and drafter** ⇒ silent garbage at
  inference. The `vocab_guard` catches this for EAGLE but not all
  derivative algorithms — `audit` your pair first.
- **Forgot to forward `seed`** ⇒ `TypeError: __init__() got an
  unexpected keyword argument 'seed'`. Inherit the signature exactly.
- **HF gated dataset** (GPQA, LiveCodeBench) ⇒ `401 Unauthorized`.
  Run `huggingface-cli login` once on the box.
- **`build_messages` returns a system message** ⇒ remember to override
  `get_system_prompt()` so the sanity-flag `system_leak` check can
  detect template echo.

---

## Project layout

```
src/spec_eval/
├── cli.py            ← CLI subcommands (run / report / compare / audit / doctor / status / list-tasks)
├── runner.py         ← EvalRun + CellConfig + the per-cell loop
├── server.py         ← SGLang subprocess (launch / health / stop / flush_cache)
├── client.py         ← Sync + async HTTP client for /generate and /server_info
├── algo_detect.py    ← EAGLE vs EAGLE-3 detection
├── guards.py         ← vocab_guard + config-tuple parsing
├── sanity.py         ← rule-based response QC flags
├── metrics.py        ← BenchmarkMetrics dataclass + compute_metrics
├── stats.py          ← bootstrap CIs, paired Wilcoxon, run_signature
├── ops.py            ← `doctor` and `status` subcommands
├── report.py         ← markdown rendering + Pareto CSV export
├── registry.py       ← @BENCHMARKS.register
├── utils.py          ← chat-template helpers + cached downloads
└── tasks/            ← one file per benchmark + the Benchmarker ABC
```

Three boundaries you must preserve:

1. **No `import sglang` in the eval venv.** All sglang access is via the
   subprocess + HTTP client. This keeps `uv sync` working on CPU dev
   boxes and lets us upgrade sglang independently.
2. **Per-cell server reboot.** A cell is `(spec_config, seed)`; spec
   params change ⇒ server reboots. The runner owns this — don't
   shortcut it from a Benchmarker.
3. **Client-side chat templating.** We send pre-templated strings to
   `/generate`, never OpenAI-style messages. Avoids SGLang frontend
   template surprises.

---

## Coding conventions

- **Python 3.10+**, type hints required on public APIs.
- **No emojis in code** (only in print/markdown output when explicitly
  matched to existing UI like `✓`/`✗` in the doctor).
- **Comments explain *why*, not *what*.** No comments narrating what a
  line does — only intent, trade-offs, or constraints the code can't
  convey.
- **Lazy imports** for heavyweight modules (`torch`, `transformers`)
  inside function bodies — keeps `spec-eval doctor` and `--help`
  responsive on machines without CUDA.
- **`from __future__ import annotations`** in every module so type
  hints stay cheap.
- **Run `make lint typecheck`** before pushing (ruff + mypy, both
  best-effort if installed).

---

## Testing

`make smoke-test` runs the offline test suite (no GPU, no SGLang).
It currently covers:

| Test | Asserts |
| --- | --- |
| `test_compute_metrics` | RoS + MoR + per-prompt distributions + JSON round-trip + baseline + empty path |
| `test_stats_helper` | `_stats` shape matches the old eval.py |
| `test_drafter_sha256_and_log_tail` | SHA256 deterministic, log tail truncation |
| `test_sanity_flags` | All 5 sanity flag triggers + aggregate roll-up |
| `test_render_run` | Report markdown includes every section |
| `test_metrics_new_fields` | ITL p50/p90/p99 + bootstrap CIs populate |
| `test_stats_module` | bootstrap_ci, paired_wilcoxon, geomean, harmean, run_signature |
| `test_ops_status` | Status snapshot detects completed / in-flight / finished |
| `test_ops_doctor` | Doctor returns structured CheckResults |
| `test_pareto_csv` | One CSV row per (run, cell, task) |
| `test_compare_paired_wilcoxon` | Compare emits Wilcoxon block when requests.jsonl pairs exist |

When adding a new metric or report section, add a corresponding test —
the suite runs in <2 seconds and catches regressions that the GPU CI
loop can't.

---

## Submitting changes

1. **One concern per commit.** The history is mined for migration
   notes; a commit that touches docs + metrics + Makefile + ops is a
   commit that can't be cherry-picked.
2. **Commit message format:**
   ```
   <type>(<scope>): <one-line summary>

   <why this change exists — not what changed>
   <what surface area moved>
   <anything reviewers should specifically look at>
   ```
   Types we use: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
3. **Run before pushing:**
   ```
   make smoke-test
   make lint        # if you have ruff
   make typecheck   # if you have mypy
   ```
4. **PRs:**
   - Title in the same `<type>(<scope>): <summary>` shape.
   - Body covers (a) the user-visible change, (b) how to verify, (c) any
     follow-ups intentionally left out of scope.
   - Add a `Test plan` checklist with the exact commands you ran.

If you're touching the SGLang subprocess lifecycle, the chat-template
boundary, or the metrics dataclass, please ping a maintainer for a
design pass before writing code — those three surfaces are load-bearing
across the rest of the pipeline.
