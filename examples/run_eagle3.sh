#!/usr/bin/env bash
# EAGLE-3 eval. Algorithm is auto-detected from the draft model's name/config.
set -euo pipefail

TARGET="${TARGET:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT="${DRAFT:-lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B}"
OUT_DIR="${OUT_DIR:-results}"
N="${N:-50}"

uv run spec-eval audit --target "$TARGET" --draft "$DRAFT"

echo "==> EAGLE-3 spec-decode run (algo auto-detected, defaults = 3/1/4)"
uv run spec-eval run \
  --target "$TARGET" --draft "$DRAFT" \
  --tasks english --num-samples "$N" \
  --output-dir "$OUT_DIR" --run-name eagle3 \
  --concurrency 4 \
  -vv

echo "==> Baseline (no spec decode)"
uv run spec-eval run \
  --target "$TARGET" \
  --tasks english --num-samples "$N" \
  --output-dir "$OUT_DIR" --run-name baseline \
  --concurrency 4 \
  -vv

echo "==> compare baseline vs eagle3"
uv run spec-eval compare "$OUT_DIR/baseline" "$OUT_DIR/eagle3" --print
