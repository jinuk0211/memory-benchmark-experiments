#!/bin/bash
# Qwen3.5 is intended primary; Gemma tests generalization. No accidental8B inference.
set -euo pipefail
cd /workspace/generalization_20260908/modern
export HF_HOME=/workspace/.hf_home
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
py=/workspace/generalization_20260908/modern/.venv/bin/python
while [ ! -f runtime_installed.txt ] || [ ! -f runtime_code_reviewed.sha256 ] || [ ! -f reporting_code_reviewed.sha256 ]; do
  sleep 30
done
sha256sum -c runtime_code_reviewed.sha256
sha256sum -c reporting_code_reviewed.sha256
"$py" -m unittest test_transfer_data test_run_transfer test_run_reader_transfer test_modern_runtime test_chat_tokenizer_compat -q
for target in qwen35 gemma4; do
  if [ "$target" = qwen35 ]; then
    model=Qwen/Qwen3.5-9B
    environment=environment--Qwen--Qwen3.5-9B.json
  else
    model=google/gemma-4-E4B-it
    environment=environment--google--gemma-4-E4B-it.json
  fi
  while [ ! -f "$environment" ]; do sleep 30; done
  "$py" smoke_runtime.py --model "$model" --environment "$environment" --out "runs/smoke_${target}_$(date +%s%N)"
  for stage in plan prepare score construct evaluate; do
    "$py" run_transfer.py "$stage" --dataset locomo --data /workspace/generalization_20260908/locomo_transfer/locomo3_unchanged_samples.json --out "/workspace/generalization_20260908/modern/runs/${target}_locomo3_fullrole" --environment "$environment" --model "$model" --seed 20260907
  done
  "$py" report_locomo.py --run "/workspace/generalization_20260908/modern/runs/${target}_locomo3_fullrole"
done
# Both model LoCoMo validations precede feasibility pilots; pilot capacity failures
# must not prevent the independent-family LoCoMo validation already completed.
for target in qwen35 gemma4; do
  if [ "$target" = qwen35 ]; then
    model=Qwen/Qwen3.5-9B
    environment=environment--Qwen--Qwen3.5-9B.json
  else
    model=google/gemma-4-E4B-it
    environment=environment--google--gemma-4-E4B-it.json
  fi
  for stage in plan prepare score construct evaluate; do
    "$py" run_transfer.py "$stage" --dataset longmemeval --data /workspace/generalization_20260908/data/longmemeval_s_cleaned.json --out "/workspace/generalization_20260908/modern/runs/${target}_lme12" --environment "$environment" --model "$model" --seed 20260907 --limit 12
  done
done
echo MODERN_LOCOMO_AND_PILOT_GENERATION_COMPLETE
# Full500 remains required; separately capacity-gated full_transfer runner follows.
