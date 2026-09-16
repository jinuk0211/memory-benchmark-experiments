#!/bin/bash
set -euo pipefail
TASK=/workspace/locomo_ablation1540_revised_20260914
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/workspace/.hf_home
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$TASK"
exec "$TASK/.venv/bin/python" "$TASK/code/reader_pilot.py" --inputs "$TASK/pilot_input" --out "$TASK/reader_pilot"
