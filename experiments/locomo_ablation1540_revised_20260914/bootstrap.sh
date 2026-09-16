#!/bin/bash
set -euo pipefail
TASK=/workspace/locomo_ablation1540_revised_20260914
export HF_HOME=/workspace/.hf_home
export UV_NO_CACHE=1
export PIP_NO_CACHE_DIR=1
mkdir -p "$TASK/runtime"
exec > >(tee -a "$TASK/runtime/bootstrap.log") 2>&1
printf 'BOOTSTRAP_START '; date -u -Iseconds
uv venv --python /venv/main/bin/python "$TASK/.venv"
uv pip install --python "$TASK/.venv/bin/python" --no-cache -r "$TASK/requirements-runtime.txt"
"$TASK/.venv/bin/pip" freeze > "$TASK/runtime/pip-freeze.txt" 2>/dev/null || uv pip freeze --python "$TASK/.venv/bin/python" > "$TASK/runtime/pip-freeze.txt"
"$TASK/.venv/bin/hf" download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --exclude '*.gguf' '*.pt' '*.bin'
"$TASK/.venv/bin/hf" download sentence-transformers/all-MiniLM-L6-v2 --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --exclude 'onnx/*' 'openvino/*' 'rust_model.ot' 'tf_model.h5' 'pytorch_model.bin'
printf 'BOOTSTRAP_COMPLETE '; date -u -Iseconds
touch "$TASK/runtime/BOOTSTRAP_COMPLETE"
