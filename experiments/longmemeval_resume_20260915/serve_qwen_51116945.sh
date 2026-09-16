#!/usr/bin/env bash
set -euo pipefail

test "${CONTAINER_ID:-}" = 51116945
unset CONTAINER_API_KEY OPENROUTER_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY
R=/workspace/longmemeval_s_native7_20260910
MODEL=/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
until [[ -s "$R/models_downloaded.txt" && -s /workspace/quad_3090_20260911/inference_env_verified.txt ]]; do
  sleep 10
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec "$R/.venv-inference/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 18081 \
  --dtype float16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.86 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enforce-eager \
  --language-model-only \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --seed 20260907
