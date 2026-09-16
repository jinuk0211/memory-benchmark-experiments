#!/bin/bash
set -euo pipefail
TASK=/workspace/locomo_ablation1540_revised_20260914
export HF_HOME=/workspace/.hf_home
exec > >(tee -a "$TASK/runtime/download-models.log") 2>&1
"$TASK/.venv/bin/hf" download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
"$TASK/.venv/bin/hf" download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
MODEL=/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
test -s "$MODEL/config.json"
test -s "$MODEL/model.safetensors.index.json"
find -L "$MODEL" -maxdepth 1 -name '*.safetensors' -printf '%f %s\n'
touch "$TASK/runtime/MODELS_DOWNLOADED"
