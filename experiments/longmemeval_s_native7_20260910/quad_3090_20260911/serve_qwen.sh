#!/usr/bin/env bash
set -euo pipefail
unset CONTAINER_API_KEY OPENROUTER_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
test "${CONTAINER_ID:-}" = 50577514
GPU_UUID="${1:?GPU UUID required}"
GPU_INDEX="$(nvidia-smi --id="$GPU_UUID" --query-gpu=index --format=csv,noheader,nounits)"
[[ "$GPU_INDEX" =~ ^[0-3]$ ]] || { printf 'Unexpected GPU mapping\n' >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX"
printf 'Verified GPU binding %s -> CUDA index %s\n' "$GPU_UUID" "$GPU_INDEX"
PORT="${2:?Port required}"
MAX_SEQS=1
if [[ "$PORT" = 18081 || "$PORT" = 18091 || "$PORT" = 18101 || "$PORT" = 18111 ]]; then
  SEQS_FILE="/workspace/quad_3090_20260911/seqs_${PORT}.txt"
  if [[ -f "$SEQS_FILE" ]]; then MAX_SEQS="$(cat "$SEQS_FILE")"; fi
fi
[[ "$MAX_SEQS" = 1 || "$MAX_SEQS" = 2 ]] || exit 2
TASK_ROOT=/workspace/longmemeval_s_native7_20260910
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
MODEL_PATH="$HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
cd "$TASK_ROOT"
test -x "$TASK_ROOT/.venv-inference/bin/python"
test -f "$MODEL_PATH/config.json"
exec "$TASK_ROOT/.venv-inference/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.86 \
  --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enforce-eager \
  --language-model-only \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --seed 20260909
