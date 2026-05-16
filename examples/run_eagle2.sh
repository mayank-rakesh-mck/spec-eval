#!/usr/bin/env bash
# EAGLE-2 eval: Llama-3.1-8B-Instruct + the lmsys EAGLE-2 drafter, then a
# baseline run for speedup comparison, then auto-generate the compare report.
set -euo pipefail

TARGET="${TARGET:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT="${DRAFT:-yuhuili/EAGLE-LLaMA3.1-Instruct-8B}"
OUT_DIR="${OUT_DIR:-results}"
N="${N:-50}"

echo "==> EAGLE-2 spec-decode run"
uv run spec-eval run \
  --target "$TARGET" --draft "$DRAFT" \
  --algorithm EAGLE \
  --tasks english --num-samples "$N" \
  --output-dir "$OUT_DIR" --run-name eagle2 \
  -vv

echo "==> Baseline (no spec decode)"
uv run spec-eval run \
  --target "$TARGET" \
  --tasks english --num-samples "$N" \
  --output-dir "$OUT_DIR" --run-name baseline \
  -vv

echo "==> compare baseline vs eagle2"
uv run spec-eval compare "$OUT_DIR/baseline" "$OUT_DIR/eagle2" --print
