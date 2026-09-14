#!/bin/bash
set -euo pipefail
cd /workspace/recursive_minilm_20260909/gemma_transfer
export HF_HOME=/workspace/.hf_home
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
py=/workspace/generalization_20260908/modern/.venv/bin/python
sha256sum -c gemma_source.sha256
while [ ! -f ../runs/recursive_v1/OFFICIAL_COMPARISON.json ]; do
  status=$(supervisorctl status recursive-minilm-refinement || true)
  case "$status" in
    *RUNNING*|*STARTING*) echo "WAITING_FOR_RECURSIVE_LOCOMO $status"; sleep 30;;
    *) echo "LOCOMO_NOT_RUNNING_OR_COMPLETE $status"; exit 1;;
  esac
done
(cd .. && sha256sum -c launch_source.sha256 && sha256sum -c recursive_source.sha256)
CUDA_VISIBLE_DEVICES='' "$py" candidate_gate.py
wait_idle() {
  "$py" -c 'import shutil; assert shutil.disk_usage(".").free >= 8*1024**3, "Less than8GiB disk remains"'
  for attempt in $(seq 1 18); do
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if [ -z "$active" ]; then return 0; fi
    echo "WAITING_FOR_GPU_RELEASE $active"; sleep 10
  done
  echo GPU_REMAINS_OCCUPIED; return 1
}
for stage in plan prepare score construct evaluate; do
  wait_idle
  sha256sum -c gemma_source.sha256 >/dev/null
  "$py" run_transfer.py "$stage" --dataset longmemeval --data data/longmemeval18.json --out runs/gemma_baseline --environment environment.json
done
for stage in construct evaluate; do
  wait_idle
  sha256sum -c gemma_source.sha256 >/dev/null
  "$py" run_recursive.py "$stage"
done
CUDA_VISIBLE_DEVICES='' "$py" summarize_transfer.py
if [ -n "${OPENAI_API_KEY:-}" ] && [ -n "${OPENAI_BASE_URL:-}" ]; then
  for method in seed r40_fused_four_turn s_parent_single_2000 recursive_source_rehearsal_v1; do
    folder=gemma_baseline
    if [ "$method" = recursive_source_rehearsal_v1 ]; then folder=gemma_recursive_v1; fi
    CUDA_VISIBLE_DEVICES='' "$py" judge_longmemeval.py --dataset data/longmemeval18.json --predictions "runs/$folder/$method.jsonl" --output "judgments/$method.jsonl" --cache-dir judgments/cache
  done
  CUDA_VISIBLE_DEVICES='' "$py" summarize_transfer.py
else
  echo GEMMA_ANSWERS_COMPLETE_OFFICIAL_JUDGE_CONFIGURATION_PENDING
fi
