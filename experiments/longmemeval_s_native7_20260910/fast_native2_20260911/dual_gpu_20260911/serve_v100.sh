#!/usr/bin/env bash
set -euo pipefail
unset CONTAINER_API_KEY OPENROUTER_API_KEY
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
  --port 18087 \
  --dtype float16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enforce-eager \
  --language-model-only \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --seed 20260909
