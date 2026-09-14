#!/bin/bash
set -euo pipefail
cd /workspace/recursive_minilm_20260909
export HF_HOME=/workspace/.hf_home
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
py=/workspace/generalization_20260908/modern/.venv/bin/python
sha256sum -c launch_source.sha256
CUDA_VISIBLE_DEVICES='' "$py" prepare_protocol.py
for stage in plan prepare score construct evaluate; do
  "$py" -c 'import shutil; assert shutil.disk_usage(".").free >= 8*1024**3, "Less than8GiB disk remains"'
  if [ "$stage" = plan ] || [ "$stage" = construct ]; then
    CUDA_VISIBLE_DEVICES='' "$py" run_transfer.py "$stage" --dataset locomo --data data/locomo10.json --out runs/qwen35_baseline --environment environment.json
  else
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if [ -n "$active" ]; then
      echo "GPU_OCCUPIED_WAITING $active"
      sleep 20
      active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
      if [ -n "$active" ]; then echo "GPU_OCCUPIED_ABORT $active"; exit 1; fi
    fi
    "$py" run_transfer.py "$stage" --dataset locomo --data data/locomo10.json --out runs/qwen35_baseline --environment environment.json
  fi
done
CUDA_VISIBLE_DEVICES='' "$py" score_refined_full.py --dataset data/locomo10.json --predictions runs/qwen35_baseline/s_parent_single_2000.jsonl --scorer source/official_evaluation.py --output runs/qwen35_baseline/OFFICIAL_REFINED.json
echo INITIAL_MINILM_BASELINE_COMPLETE_RECURSIVE_WORK_REMAINS
