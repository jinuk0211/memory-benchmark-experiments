#!/bin/bash
# Resume only the failed Gemma feasibility pilot using a separately pinned writer capacity.
set -euo pipefail
cd /workspace/generalization_20260908/modern
export HF_HOME=/workspace/.hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
py=/workspace/generalization_20260908/modern/.venv/bin/python
lineage=recovery_lineage/gemma8k_failure_20260908_0936
[ -d "$lineage" ]
(cd "$lineage" && sha256sum -c failure_artifacts.sha256)
sha256sum -c runtime_code_reviewed.sha256
sha256sum -c reporting_code_reviewed.sha256
sha256sum -c recovery_queue_code_reviewed.sha256
sha256sum -c capacity_recovery_prerequisites.sha256
cd /workspace/generalization_20260908/full_transfer
sha256sum -c runtime_code_reviewed.sha256
sha256sum -c full_launch_code_reviewed.sha256
sha256sum -c capacity_pilot_code_reviewed.sha256
CUDA_VISIBLE_DEVICES='' "$py" -m unittest test_transfer_data test_run_transfer test_run_reader_transfer test_full_capacity test_parent_single_fast test_chat_tokenizer_compat test_audit_target_capacity test_full_launch test_capacity_pilot -q
check_space() {
  "$py" -c 'import shutil,sys; free=shutil.disk_usage(".").free; print("CAPACITY_PILOT_FREE_SPACE",free,flush=True); sys.exit(0 if free>=8*1024**3 else "Insufficient free disk space; capacity pilot stopped without deleting files")'
}
wait_gpu_idle() {
  for attempt in $(seq 1 10); do
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
    if [ -z "$active" ]; then return 0; fi
    echo "WAITING_FOR_GPU_TEARDOWN attempt=$attempt active_pids=$active"
    sleep 30
  done
  echo 'GPU remains occupied; capacity pilot stopped without terminating any process' >&2
  return 1
}
model=google/gemma-4-E4B-it
environment=/workspace/generalization_20260908/modern/environment--google--gemma-4-E4B-it.json
out=/workspace/generalization_20260908/full_transfer/runs/gemma4_lme12_writer65k
reference=/workspace/generalization_20260908/modern/runs/gemma4_lme12
check_space
wait_gpu_idle
"$py" smoke_writer65k.py --model "$model" --environment "$environment" --out "/workspace/generalization_20260908/full_transfer/runs/smoke65k_gemma4_capacity_pilot_$(date +%s%N)"
for stage in plan prepare score construct evaluate; do
  args=(run_capacity_pilot.py "$stage" --dataset longmemeval --data /workspace/generalization_20260908/data/longmemeval_s_cleaned.json --out "$out" --environment "$environment" --model "$model" --limit 12 --seed 20260907 --embed-model Qwen/Qwen3-Embedding-0.6B --embed-batch-size 4 --reference-run "$reference")
  if [ "$stage" = plan ] || [ "$stage" = construct ]; then
    CUDA_VISIBLE_DEVICES='' "$py" "${args[@]}"
  else
    check_space
    wait_gpu_idle
    "$py" "${args[@]}"
  fi
done
# The existing full500 gate requires this service to exit zero after this marker.
echo MODERN_LOCOMO_AND_PILOT_GENERATION_COMPLETE
