#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/gemma3_locomo_51017220
BUNDLE="$ROOT/bundle"
PY="$ROOT/.venv/bin/python"
DATA="$BUNDLE/MemoryData/datasets/LoCoMo/locomo10.json"
SCORER="$BUNDLE/HiGMem/official_locomo_evaluation.py"
OUT="$ROOT/results/full"

while [[ ! -f "$OUT/phases/ours_no_binding.complete" ]]; do
  if ! supervisorctl status gemma-locomo-full | grep -q RUNNING; then
    echo "WAIT_ABORT main benchmark stopped before ours completed"
    exit 1
  fi
  sleep 15
done

echo "LANGMEM_CORRECTION_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
supervisorctl stop gemma-locomo-full || true
restart_main=1
restart_full() {
  if [[ "$restart_main" == 1 ]]; then
    supervisorctl start gemma-locomo-full || true
  fi
}
trap restart_full EXIT

archive="$OUT/invalid/langmem_deletes_true"
mkdir -p "$archive"
if [[ -d "$OUT/common/langmem" ]]; then mv "$OUT/common/langmem" "$archive/common_langmem"; fi
if [[ -d "$OUT/scored/langmem" ]]; then mv "$OUT/scored/langmem" "$archive/scored_langmem"; fi
if [[ -f "$OUT/phases/common_langmem.complete" ]]; then mv "$OUT/phases/common_langmem.complete" "$archive/common_langmem.complete"; fi

export BASELINE_STRICT_COMPARISON=1
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export LOCOMO_MODEL=gemma-3-12b-it
export LOCOMO_MODEL_SHA256=$(cut -d " " -f 1 "$ROOT/model_fp16.sha256")
export LOCOMO_CHECKPOINT=ggml-org/gemma-3-12b-it-GGUF/gemma-3-12b-it-f16.gguf
export LOCOMO_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export NLTK_DATA="$ROOT/nltk_data"
export HF_HOME="$ROOT/hf_home"
export FASTEMBED_CACHE_PATH="$ROOT/fastembed_cache"

cd "$BUNDLE/MemoryData"
"$PY" scripts/run_six_baselines.py \
  --methods langmem \
  --datasets locomo \
  --model gemma-3-12b-it \
  --base-url "$OPENAI_BASE_URL" \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-base-url "$EMBEDDING_BASE_URL" \
  --embedding-dim 384 \
  --locomo-data "$DATA" \
  --max-queries 0 \
  --artifact-root "$OUT/common"
"$PY" "$BUNDLE/finalize_results.py" common \
  --raw "$OUT/common/langmem/locomo" \
  --dataset "$DATA" \
  --scorer "$SCORER" \
  --output "$OUT/scored/langmem" \
  --expected-count 1540
"$PY" -c "import json; p='$OUT/scored/langmem/report.json'; d=json.load(open(p)); assert d['complete'] and d['n']==1540 and d['empty_predictions']==0, d"

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/phases/common_langmem.complete"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/phases/common_langmem_corrected.complete"
echo "LANGMEM_CORRECTION_COMPLETE $(cat "$OUT/phases/common_langmem_corrected.complete")"

restart_main=0
supervisorctl start gemma-locomo-full
trap - EXIT