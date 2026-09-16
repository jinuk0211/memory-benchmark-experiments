#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/gemma3_locomo_51099149
BUNDLE="$ROOT/bundle"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/results/split"
DATA="$BUNDLE/MemoryData/datasets/LoCoMo/locomo10.json"
export BASELINE_STRICT_COMPARISON=1 OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
until [[ -f "$OUT/SPLIT_COMPLETE" ]]; do sleep 30; done
if [[ -f "$OUT/phases/higmem.complete" ]]; then exit 0; fi
printf '%s\n' higmem > "$OUT/CURRENT_PHASE"
cd "$BUNDLE/HiGMem"
"$PY" run_higmem_gemma.py --model gemma-3-12b-it --api-base "$OPENAI_BASE_URL" --dataset "$DATA" --output "$OUT/higmem" --sample-workers 1 --qa-workers 1
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/phases/higmem.complete"
rm -f "$OUT/CURRENT_PHASE"