#!/usr/bin/env bash
# SpecForge-style sweep: one server boot per (batch_size, num_steps, topk, draft_tokens)
# tuple, all sharing the same drafter. Use this to compare EAGLE-2 vs EAGLE-3 trees
# vs a tight linear-chain config on the same target.
set -euo pipefail

TARGET="${TARGET:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT="${DRAFT:-lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B}"
OUT_DIR="${OUT_DIR:-results}"
N="${N:-50}"

uv run spec-eval run \
  --target "$TARGET" --draft "$DRAFT" \
  --tasks english --num-samples "$N" \
  --config-list 1,0,0,0  1,3,1,4  1,5,8,64 \
  --seeds 42,123 \
  --concurrency 4 \
  --output-dir "$OUT_DIR" --run-name sweep \
  -vv

uv run spec-eval report "$OUT_DIR/sweep" --print
