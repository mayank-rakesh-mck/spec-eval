# spec-eval — convenience targets for the common workflows.
#
# Overridable variables (pass on the command line: `make run-eagle2 TARGET=...`)
TARGET       ?= meta-llama/Llama-3.1-8B-Instruct
DRAFT        ?= yuhuili/EAGLE-LLaMA3.1-Instruct-8B
DRAFT_EAGLE3 ?= lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
TASKS        ?= english
N            ?= 50
SEEDS        ?= 42
CONCURRENCY  ?= 1
OUTPUT_DIR   ?= results
RUN_NAME     ?=
BASELINE_DIR ?=
SPEC_DIR     ?=
RUN_DIR      ?=

# Forward extra args after `--` (e.g. `make run-eagle2 -- --force --verbose`)
EXTRA_ARGS   ?=

UV           ?= uv

# Per-cell EAGLE-2 / EAGLE-3 hyperparam tuples used by the sweep target.
EAGLE2_TUPLE ?= 1,5,8,64
EAGLE3_TUPLE ?= 1,3,1,4
BASELINE_TUPLE ?= 1,0,0,0

.DEFAULT_GOAL := help
.PHONY: help setup install-sglang list-tasks audit \
        run-baseline run-eagle2 run-eagle3 run-sweep \
        report compare smoke-test lint typecheck clean clean-results

help:  ## Show this help.
	@printf "spec-eval — common workflows\n\n"
	@printf "Usage:\n  make <target> [VAR=value ...]\n\n"
	@printf "Targets:\n"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / \
	     {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables (defaults shown):\n"
	@printf "  TARGET       = $(TARGET)\n"
	@printf "  DRAFT        = $(DRAFT)\n"
	@printf "  DRAFT_EAGLE3 = $(DRAFT_EAGLE3)\n"
	@printf "  TASKS        = $(TASKS)         # english|reasoning|all-en|all|comma-list\n"
	@printf "  N            = $(N)             # per-task sample count\n"
	@printf "  SEEDS        = $(SEEDS)         # comma list of seeds\n"
	@printf "  CONCURRENCY  = $(CONCURRENCY)             # async pool size\n"
	@printf "  OUTPUT_DIR   = $(OUTPUT_DIR)\n"
	@printf "  RUN_NAME     = $(RUN_NAME)              # optional run-dir name\n"
	@printf "  EXTRA_ARGS   = $(EXTRA_ARGS)              # forwarded to spec-eval\n"

# ─── Setup ─────────────────────────────────────────────────────────────────

setup:  ## Install eval deps into the UV venv (no sglang).
	$(UV) sync

install-sglang:  ## Install sglang[all] into the same venv (GPU box only).
	$(UV) pip install "sglang[all]>=0.5"

# ─── Inspection (no GPU needed) ────────────────────────────────────────────

list-tasks:  ## List known benchmarks.
	$(UV) run spec-eval list-tasks

audit:  ## Audit a (target, draft) pair: vocab + algorithm + defaults. Vars: TARGET, DRAFT
	$(UV) run spec-eval audit --target "$(TARGET)" --draft "$(DRAFT)"

# ─── Eval runs ─────────────────────────────────────────────────────────────

run-baseline:  ## Baseline (no spec-decode). Vars: TARGET, TASKS, N, RUN_NAME
	$(UV) run spec-eval run \
	  --target "$(TARGET)" \
	  --tasks "$(TASKS)" --num-samples "$(N)" \
	  --seeds "$(SEEDS)" --concurrency "$(CONCURRENCY)" \
	  --output-dir "$(OUTPUT_DIR)" $(if $(RUN_NAME),--run-name "$(RUN_NAME)",) \
	  $(EXTRA_ARGS)

run-eagle2:  ## EAGLE-2 spec-decode run. Vars: TARGET, DRAFT, TASKS, N
	$(UV) run spec-eval run \
	  --target "$(TARGET)" --draft "$(DRAFT)" \
	  --algorithm EAGLE \
	  --tasks "$(TASKS)" --num-samples "$(N)" \
	  --seeds "$(SEEDS)" --concurrency "$(CONCURRENCY)" \
	  --output-dir "$(OUTPUT_DIR)" $(if $(RUN_NAME),--run-name "$(RUN_NAME)",) \
	  $(EXTRA_ARGS)

run-eagle3:  ## EAGLE-3 spec-decode run. Vars: TARGET, DRAFT_EAGLE3, TASKS, N
	$(UV) run spec-eval run \
	  --target "$(TARGET)" --draft "$(DRAFT_EAGLE3)" \
	  --algorithm EAGLE3 \
	  --tasks "$(TASKS)" --num-samples "$(N)" \
	  --seeds "$(SEEDS)" --concurrency "$(CONCURRENCY)" \
	  --output-dir "$(OUTPUT_DIR)" $(if $(RUN_NAME),--run-name "$(RUN_NAME)",) \
	  $(EXTRA_ARGS)

run-sweep:  ## Sweep baseline + EAGLE-2 + EAGLE-3 trees in one boot. Vars: TARGET, DRAFT_EAGLE3
	$(UV) run spec-eval run \
	  --target "$(TARGET)" --draft "$(DRAFT_EAGLE3)" \
	  --algorithm auto \
	  --config-list $(BASELINE_TUPLE) $(EAGLE2_TUPLE) $(EAGLE3_TUPLE) \
	  --tasks "$(TASKS)" --num-samples "$(N)" \
	  --seeds "$(SEEDS)" --concurrency "$(CONCURRENCY)" \
	  --output-dir "$(OUTPUT_DIR)" $(if $(RUN_NAME),--run-name "$(RUN_NAME)",) \
	  $(EXTRA_ARGS)

# ─── Reports ───────────────────────────────────────────────────────────────

report:  ## Render markdown for a finished run. Vars: RUN_DIR
	@if [ -z "$(RUN_DIR)" ]; then echo "RUN_DIR is required"; exit 2; fi
	$(UV) run spec-eval report "$(RUN_DIR)" --print

compare:  ## Diff baseline vs spec run. Vars: BASELINE_DIR, SPEC_DIR
	@if [ -z "$(BASELINE_DIR)" ] || [ -z "$(SPEC_DIR)" ]; then \
	    echo "BASELINE_DIR and SPEC_DIR are required"; exit 2; fi
	$(UV) run spec-eval compare "$(BASELINE_DIR)" "$(SPEC_DIR)" --print

# ─── Dev: tests + hygiene ──────────────────────────────────────────────────

smoke-test:  ## Offline smoke test for metrics + helpers (no GPU, no server).
	$(UV) run python scripts/smoke_test.py

lint:  ## Best-effort lint with ruff (if installed in the venv).
	@$(UV) run --no-sync python -c "import ruff" 2>/dev/null && \
	    $(UV) run ruff check src/ || \
	    echo "ruff not installed; skipping (uv pip install ruff)"

typecheck:  ## Best-effort type-check with mypy (if installed).
	@$(UV) run --no-sync python -c "import mypy" 2>/dev/null && \
	    $(UV) run mypy src/spec_eval || \
	    echo "mypy not installed; skipping (uv pip install mypy)"

clean:  ## Remove __pycache__ and *.egg-info.
	find . -type d \( -name __pycache__ -o -name "*.egg-info" \) \
	    -not -path "./.venv/*" -prune -exec rm -rf {} +

clean-results:  ## Remove everything under $(OUTPUT_DIR)/.
	rm -rf "$(OUTPUT_DIR)"
