#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/workspace/longmemeval_s_native3_5090_20260910
cd "$TASK_ROOT"
exec 9>queue/native3_execution.lock
flock -n 9 || { echo "A native3 queue or manual method is already running" >&2; exit 1; }
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENAI_API_KEY=EMPTY MEM0_TELEMETRY=false PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false SKIP_SBERT_SIM=1
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
unset OPENROUTER_API_KEY
METHOD=${1:?Usage: run_one.sh langmem|mem0|a_mem smoke|full}
PHASE=${2:?Usage: run_one.sh langmem|mem0|a_mem smoke|full}
case "$PHASE" in
  smoke) EXTRA=(--ids-file "$TASK_ROOT/queue/smoke_ids.json") ;;
  full) EXTRA=() ;;
  *) echo "Expected smoke or full" >&2; exit 2 ;;
esac
MODEL=Qwen/Qwen3.5-9B
EMBEDDING=sentence-transformers/all-MiniLM-L6-v2
TOKENIZER=/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
MINILM=/workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
COMMON=(--dataset "$TASK_ROOT/longmemeval_s_cleaned.json" --api-base http://127.0.0.1:18083/v1 --model "$MODEL")
case "$METHOD" in
  langmem)
    exec "$TASK_ROOT/.venv-client/bin/python" "$TASK_ROOT/official_recovery/langmem_native_sessions_v1/runner.py" \
      --method langmem --run-dir "$TASK_ROOT/runs/langmem_native_sessions_v1" "${COMMON[@]}" \
      --embedding-api-base http://127.0.0.1:18084/v1 --embedding-model "$EMBEDDING" \
      --embedding-dims 384 --tokenizer "$TOKENIZER" "${EXTRA[@]}" ;;
  mem0)
    exec "$TASK_ROOT/.venv-mem0-oss0194/bin/python" "$TASK_ROOT/official_recovery/mem0_native_pairs_v1/runner.py" \
      --run-dir "$TASK_ROOT/runs/mem0_native_pairs_v1" --source-root "$TASK_ROOT/source/MemoryData" "${COMMON[@]}" \
      --embedding-api-base http://127.0.0.1:18084/v1 --embedding-model "$EMBEDDING" \
      --embedding-dims 384 --tokenizer "$TOKENIZER" "${EXTRA[@]}" ;;
  a_mem)
    exec "$TASK_ROOT/.venv-client/bin/python" "$TASK_ROOT/official_recovery/a_mem_paper_v1/runner.py" \
      --method a_mem --run-dir "$TASK_ROOT/runs/a_mem_paper_v1" "${COMMON[@]}" \
      --embedding-model "$MINILM" "${EXTRA[@]}" ;;
  *) echo "Expected langmem, mem0, or a_mem" >&2; exit 2 ;;
esac
