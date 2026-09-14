#!/bin/bash
set -euo pipefail
cd /workspace/locomo_ablation300_20260913
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/workspace/longmemeval_s_native7_20260910/.venv-inference/bin/python
mkdir -p run
"$PY" code/run_ablation.py freeze --shard 0
p0= p1=
trap 'if [[ -n "$p0" ]]; then kill "$p0" 2>/dev/null || true; fi; if [[ -n "$p1" ]]; then kill "$p1" 2>/dev/null || true; fi' EXIT
for stage in prepare evaluate; do
  CUDA_VISIBLE_DEVICES=0,1 "$PY" code/run_ablation.py "$stage" --shard 0 > run/"$stage"_0.log 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES=2,3 "$PY" code/run_ablation.py "$stage" --shard 1 > run/"$stage"_1.log 2>&1 &
  p1=$!
  wait "$p0"
  p0=
  wait "$p1"
  p1=
done
printf '%s\n' ALL_ABLATION300_PREDICTIONS_COMPLETE
