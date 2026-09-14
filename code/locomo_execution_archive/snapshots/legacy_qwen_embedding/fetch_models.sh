#!/bin/bash
set -euo pipefail
cd /workspace/generalization_20260908/modern
export HF_HOME=/workspace/.hf_home
export PYTHONUNBUFFERED=1
for model in Qwen/Qwen3.5-9B google/gemma-4-E4B-it; do
  /workspace/locomo-refinement/.venv/bin/python fetch_models.py --model "$model"
done