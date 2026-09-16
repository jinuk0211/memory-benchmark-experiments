#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/gemma3_locomo_51017220
exec "$ROOT/src/llama.cpp/build/bin/llama-server" \
  --model "$ROOT/models/gemma-3-12b-it-f16.gguf" \
  --alias gemma-3-12b-it \
  --host 127.0.0.1 \
  --port 8000 \
  --ctx-size 49152 \
  --parallel 1 \
  --n-gpu-layers 99 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --flash-attn on \
  --no-mmproj \
  --jinja \
  --metrics \
  --threads 16 \
  --threads-batch 32 \
  --seed 20260907
