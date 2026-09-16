#!/bin/bash
set -euo pipefail
TASK=/workspace/locomo_ablation1540_revised_20260914
READER=${1:?Pass the frozen selected reader: legacy or grounded}
case "$READER" in legacy|grounded) ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/workspace/.hf_home
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$TASK"
"$TASK/.venv/bin/python" "$TASK/code/run_revised.py" freeze --out "$TASK/full_$READER" --memory-root "$TASK/run" --reader "$READER" --population full1540
exec "$TASK/.venv/bin/python" "$TASK/code/run_revised.py" evaluate --out "$TASK/full_$READER" --memory-root "$TASK/run" --reader "$READER" --population full1540
