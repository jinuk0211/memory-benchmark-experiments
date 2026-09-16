#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/gemma3_locomo_51017220
BUNDLE="$ROOT/bundle"
PY="$ROOT/.venv/bin/python"
DATA="$BUNDLE/MemoryData/datasets/LoCoMo/locomo10.json"
HIGMEM_SMOKE_DATA="$BUNDLE/higmem_smoke_locomo.json"
SCORER="$BUNDLE/HiGMem/official_locomo_evaluation.py"
OUT="$ROOT/results/smoke"
MODEL_SHA=$(cut -d " " -f 1 "$ROOT/model_fp16.sha256")

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export LOCOMO_MODEL=gemma-3-12b-it
export LOCOMO_MODEL_SHA256="$MODEL_SHA"
export LOCOMO_CHECKPOINT=ggml-org/gemma-3-12b-it-GGUF/gemma-3-12b-it-f16.gguf
export LOCOMO_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export NLTK_DATA="$ROOT/nltk_data"
export HF_HOME="$ROOT/hf_home"
export FASTEMBED_CACHE_PATH="$ROOT/fastembed_cache"

rm -rf "$OUT"
mkdir -p "$OUT"

"$PY" -c 'from openai import OpenAI; c=OpenAI(base_url="http://127.0.0.1:8000/v1",api_key="EMPTY",timeout=600); r=c.chat.completions.create(model="gemma-3-12b-it",messages=[{"role":"user","content":"Reply with exactly READY"}],temperature=0,max_tokens=8); assert r.choices[0].message.content; print(r.choices[0].message.content)'
"$PY" -c 'from openai import OpenAI; c=OpenAI(base_url="http://127.0.0.1:8001/v1",api_key="EMPTY"); r=c.embeddings.create(model="sentence-transformers/all-MiniLM-L6-v2",input=["smoke"],encoding_format="float"); assert len(r.data[0].embedding)==384; print("embedding_384_ok")'

"$PY" "$BUNDLE/full_context_runner.py" --dataset "$DATA" --output "$OUT/full_context" --model gemma-3-12b-it --base-url "$OPENAI_BASE_URL" --max-questions 1

cd "$BUNDLE/MemoryData"
"$PY" scripts/run_six_baselines.py   --methods langmem mem0 a_mem lightmem simplemem   --datasets locomo   --model gemma-3-12b-it   --base-url "$OPENAI_BASE_URL"   --embedding-model sentence-transformers/all-MiniLM-L6-v2   --embedding-base-url "$EMBEDDING_BASE_URL"   --embedding-dim 384   --locomo-data "$DATA"   --max-samples 1   --max-queries 1   --max-context-chunks 1   --artifact-root "$OUT/common"

for method in langmem mem0 a_mem lightmem simplemem; do
  "$PY" "$BUNDLE/finalize_results.py" common     --raw "$OUT/common/$method/locomo"     --dataset "$DATA"     --scorer "$SCORER"     --output "$OUT/scored/$method"     --expected-count 1
done

"$PY" "$BUNDLE/prepare_higmem_smoke.py"
cd "$BUNDLE/HiGMem"
"$PY" run_higmem_gemma.py   --model gemma-3-12b-it   --api-base "$OPENAI_BASE_URL"   --dataset "$HIGMEM_SMOKE_DATA"   --output "$OUT/higmem"   --sample conv-26   --pilot-questions 1   --sample-workers 1   --qa-workers 1

"$PY" "$BUNDLE/prepare_ours_smoke.py"
"$PY" "$BUNDLE/ours_no_binding_runner.py" freeze   --inputs "$BUNDLE/ours_input_smoke" --out "$OUT/ours" --shards 1 --shard 0
"$PY" "$BUNDLE/ours_no_binding_runner.py" prepare   --inputs "$BUNDLE/ours_input_smoke" --out "$OUT/ours" --shards 1 --shard 0
"$PY" "$BUNDLE/ours_no_binding_runner.py" evaluate   --inputs "$BUNDLE/ours_input_smoke" --out "$OUT/ours" --shards 1 --shard 0
"$PY" "$BUNDLE/finalize_results.py" ours   --raw "$OUT/ours/predictions/no_binding"   --dataset "$DATA"   --scorer "$SCORER"   --output "$OUT/scored/ours_no_binding"   --expected-count 1

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/SMOKE_COMPLETE"