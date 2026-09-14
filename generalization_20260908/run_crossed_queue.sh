#!/bin/bash
# Sequential GPU scheduling; all benchmark settings remain in locked Python protocols.
set -euo pipefail
cd /workspace/generalization_20260908
export HF_HOME=/workspace/.hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
py=/workspace/locomo-refinement/.venv/bin/python
resume_source=0
cleanup() {
  if [ "$resume_source" -eq 1 ]; then
    supervisorctl start memory-transfer-qwen-lme12 || true
  fi
}
trap cleanup EXIT
while [ ! -f environment.json ]; do
  sleep 30
done
# Readiness is established by the SHA-verified downloader before this file is written.
"$py" -m unittest test_transfer_data test_run_transfer test_run_reader_transfer -q
"$py" run_reader_transfer.py --stage plan --environment environment.json --out runs/mistral_reader100 --source-run /workspace/locomo-refinement/runs/portable_parent_audit_v1
source_status=$(supervisorctl status memory-transfer-qwen-lme12 || true)
if [[ "$source_status" == *RUNNING* ]]; then
  supervisorctl stop memory-transfer-qwen-lme12
  resume_source=1
fi
"$py" run_reader_transfer.py --stage evaluate --environment environment.json --out runs/mistral_reader100 --source-run /workspace/locomo-refinement/runs/portable_parent_audit_v1
if [ "$resume_source" -eq 1 ]; then
  supervisorctl start memory-transfer-qwen-lme12
  resume_source=0
fi
while [[ "$(supervisorctl status memory-transfer-qwen-lme12 || true)" == *RUNNING* ]]; do
  sleep 30
done
"$py" -c 'import json; s=json.load(open("runs/qwen3_lme12/status.json")); assert s["phase"] == "generation_complete", s'
for stage in plan prepare score construct evaluate; do
  "$py" run_transfer.py "$stage" --dataset longmemeval --data /workspace/generalization_20260908/data/longmemeval_s_cleaned.json --out /workspace/generalization_20260908/runs/mistral_lme12 --environment /workspace/generalization_20260908/environment.json --model mistralai/Mistral-7B-Instruct-v0.3 --limit 12
done
echo TRANSFER_QUEUE_GENERATION_COMPLETE


