#!/bin/bash
set -euo pipefail
cd /workspace/locomo_ablation300_20260913
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/workspace/longmemeval_s_native7_20260910/.venv-inference/bin/python
p0= p1=
cleanup() {
  if [[ -n "$p0" ]]; then kill -TERM -- -"$p0" 2>/dev/null || true; fi
  if [[ -n "$p1" ]]; then kill -TERM -- -"$p1" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
setsid env CUDA_VISIBLE_DEVICES=0,1 "$PY" code/run_ablation.py evaluate --shard 0 > run/evaluate_0.retry1.log 2>&1 &
p0=$!
setsid env CUDA_VISIBLE_DEVICES=2,3 "$PY" code/run_ablation.py evaluate --shard 1 > run/evaluate_1.retry1.log 2>&1 &
p1=$!
wait "$p0"
kill -TERM -- -"$p0" 2>/dev/null || true
p0=
wait "$p1"
kill -TERM -- -"$p1" 2>/dev/null || true
p1=
printf '%s\n' ALL_ABLATION300_PREDICTIONS_COMPLETE
