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
sha256sum -c launch_source.sha256 >/dev/null
sha256sum -c recursive_source.sha256 >/dev/null
sha256sum -c recursive_v2_source.sha256
while [ ! -f gemma_transfer/CANDIDATE_DECISION.json ]; do
  status=$(supervisorctl status recursive-minilm-gemma18 || true)
  case "$status" in
    *RUNNING*|*STARTING*) echo "WAITING_FOR_V1_DECISION $status"; sleep 30;;
    *) echo "V1_DECISION_NOT_AVAILABLE $status"; exit 1;;
  esac
done
set +e
CUDA_VISIBLE_DEVICES='' "$py" queue_entry_v2.py
result=$?
set -e
if [ "$result" -eq 3 ]; then exit 0; fi
if [ "$result" -ne 0 ]; then exit "$result"; fi
for stage in construct evaluate; do
  "$py" -c 'import shutil; assert shutil.disk_usage(".").free >= 8*1024**3, "Less than 8 GiB disk remains"'
  idle=false
  for attempt in $(seq 1 18); do
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if [ -z "$active" ]; then idle=true; break; fi
    echo "WAITING_FOR_GPU_RELEASE $active"; sleep 10
  done
  if [ "$idle" != true ]; then echo GPU_REMAINS_OCCUPIED; exit 1; fi
  sha256sum -c recursive_v2_source.sha256 >/dev/null
  "$py" run_recursive_v2.py "$stage"
done
CUDA_VISIBLE_DEVICES='' "$py" summarize_recursive_v2.py
echo RECURSIVE_V2_LOCOMO_COMPLETE_REQUIRES_FROZEN_GEMMA_TRANSFER
