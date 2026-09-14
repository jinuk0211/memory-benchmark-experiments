#!/bin/bash
set -euo pipefail
test "${CONTAINER_ID:-}" = 50577514
unset CONTAINER_API_KEY OPENROUTER_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
R=/workspace/longmemeval_s_native7_20260910
export CUDA_VISIBLE_DEVICES='' HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$R"
exec "$R/.venv-inference/bin/python" "$R/serve_minilm.py" --model-path /workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --journal "$R/logs/embedding_usage.jsonl" --port 18084
