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
sha256sum -c recursive_source.sha256
while [ ! -f runs/qwen35_baseline/OFFICIAL_REFINED.json ]; do
  status=$(supervisorctl status recursive-minilm-baseline || true)
  case "$status" in
    *RUNNING*|*STARTING*) echo "WAITING_FOR_BASELINE $status"; sleep 30;;
    *) echo "BASELINE_NOT_RUNNING_OR_COMPLETE $status"; exit 1;;
  esac
done
for stage in construct evaluate; do
  "$py" -c 'import shutil; assert shutil.disk_usage(".").free >= 8*1024**3, "Less than8GiB disk remains"'
  idle=false
  for attempt in $(seq 1 12); do
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if [ -z "$active" ]; then idle=true; break; fi
    echo "WAITING_FOR_GPU_RELEASE $active"; sleep 10
  done
  if [ "$idle" != true ]; then echo GPU_REMAINS_OCCUPIED; exit 1; fi
  "$py" run_recursive.py "$stage"
done
CUDA_VISIBLE_DEVICES='' "$py" summarize_recursive.py
echo RECURSIVE_LOCOMO_COMPLETE_GEMMA_TRANSFER_PENDING
