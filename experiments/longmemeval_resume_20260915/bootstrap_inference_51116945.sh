#!/usr/bin/env bash
set -euo pipefail

test "${CONTAINER_ID:-}" = 51116945
unset CONTAINER_API_KEY OPENROUTER_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY
export PATH=/venv/main/bin:$PATH
export UV_CACHE_DIR=/workspace/quad_3090_20260911/uv-cache
export UV_LINK_MODE=hardlink
export UV_CONCURRENT_DOWNLOADS=4
export UV_CONCURRENT_INSTALLS=2
export PIP_NO_CACHE_DIR=1
R=/workspace/longmemeval_s_native7_20260910
M=/workspace/quad_3090_20260911
PY=/.uv/python_install/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
mkdir -p "$R" "$M" "$UV_CACHE_DIR"
uv python install 3.12.14
"$PY" -c 'import sys; assert sys.version_info[:3] == (3, 12, 14), sys.version'
if [[ ! -x "$R/.venv-inference/bin/python" ]]; then
  uv venv --python "$PY" "$R/.venv-inference"
fi
uv pip install --python "$R/.venv-inference/bin/python" --no-deps --only-binary :all: -r "$M/bootstrap_inference_requirements.txt"
uv pip check --python "$R/.venv-inference/bin/python"
"$R/.venv-inference/bin/python" -c 'import torch, transformers, vllm; assert torch.__version__ == "2.10.0+cu129"; assert transformers.__version__ == "5.5.3"; assert vllm.__version__ == "0.19.1"'
uv cache clean --cache-dir "$UV_CACHE_DIR"
printf '%s\n' PINNED_INFERENCE_ENV_VERIFIED > "$M/inference_env_verified.txt"
