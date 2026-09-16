#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/gemma3_locomo_51099149
BUNDLE="$ROOT/bundle"
PY="$ROOT/.venv/bin/python"
DATA="$BUNDLE/MemoryData/datasets/LoCoMo/locomo10.json"
SCORER="$BUNDLE/HiGMem/official_locomo_evaluation.py"
OUT="$ROOT/results/split"
MODEL_SHA=$(cut -d " " -f 1 "$ROOT/model_fp16.sha256")
export BASELINE_STRICT_COMPARISON=1 OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export LOCOMO_MODEL=gemma-3-12b-it LOCOMO_MODEL_SHA256="$MODEL_SHA"
export LOCOMO_CHECKPOINT=ggml-org/gemma-3-12b-it-GGUF/gemma-3-12b-it-f16.gguf
export LOCOMO_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export NLTK_DATA=/root/nltk_data_51099149 HF_HOME="$ROOT/hf_home" FASTEMBED_CACHE_PATH="$ROOT/fastembed_cache"
mkdir -p "$OUT/phases" "$OUT/scored"
run_phase() {
  local name="$1"; shift
  if [[ -f "$OUT/phases/$name.complete" ]]; then echo "PHASE_SKIP $name"; return; fi
  printf '%s\n' "$name" > "$OUT/CURRENT_PHASE"
  echo "PHASE_START $name $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$@"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/phases/$name.complete"
  echo "PHASE_COMPLETE $name $(cat "$OUT/phases/$name.complete")"
}
run_common() {
  local method="$1"; cd "$BUNDLE/MemoryData"
  "$PY" scripts/run_six_baselines.py --methods "$method" --datasets locomo --model gemma-3-12b-it --base-url "$OPENAI_BASE_URL" --embedding-model sentence-transformers/all-MiniLM-L6-v2 --embedding-base-url "$EMBEDDING_BASE_URL" --embedding-dim 384 --locomo-data "$DATA" --max-queries 0 --artifact-root "$OUT/common"
  "$PY" "$BUNDLE/finalize_results.py" common --raw "$OUT/common/$method/locomo" --dataset "$DATA" --scorer "$SCORER" --output "$OUT/scored/$method" --expected-count 1540
}
run_higmem() {
  cd "$BUNDLE/HiGMem"
  "$PY" run_higmem_gemma.py     --model gemma-3-12b-it     --api-base "$OPENAI_BASE_URL"     --dataset "$DATA"     --output "$OUT/higmem"     --sample-workers 1     --qa-workers 1
}
run_ours() {
  cd "$BUNDLE"
  "$PY" prepare_ours_input.py
  "$PY" ours_no_binding_runner.py freeze --inputs "$BUNDLE/ours_input" --out "$OUT/ours" --shards 1 --shard 0
  "$PY" ours_no_binding_runner.py prepare --inputs "$BUNDLE/ours_input" --out "$OUT/ours" --shards 1 --shard 0
  "$PY" ours_no_binding_runner.py evaluate --inputs "$BUNDLE/ours_input" --out "$OUT/ours" --shards 1 --shard 0
  "$PY" finalize_results.py ours --raw "$OUT/ours/predictions/no_binding" --dataset "$DATA" --scorer "$SCORER" --output "$OUT/scored/ours_no_binding" --expected-count 1540
}
for method in lightmem simplemem; do run_phase "common_$method" run_common "$method"; done
run_phase ours_no_binding run_ours
run_phase higmem run_higmem
rm -f "$OUT/CURRENT_PHASE"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/SPLIT_COMPLETE"
