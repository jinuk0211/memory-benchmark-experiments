#!/bin/bash
set -euo pipefail
cd /workspace/generalization_20260908
export HF_HOME=/workspace/.hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
for stage in prepare score construct evaluate; do
  /workspace/locomo-refinement/.venv/bin/python run_transfer.py "$stage" --dataset longmemeval --data /workspace/generalization_20260908/data/longmemeval_s_cleaned.json --out /workspace/generalization_20260908/runs/qwen3_lme12 --environment /workspace/generalization_20260908/source/environment.json --limit 12
done

