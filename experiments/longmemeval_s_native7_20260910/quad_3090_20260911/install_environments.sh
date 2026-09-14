#!/usr/bin/env bash
# Runtime preparation only. Never creates recovery_verified.json or resets the guard.
set -euo pipefail
if [[ "${1:-}" != --bounded ]]; then
  exec timeout --signal=TERM --kill-after=15s 3600 bash "$0" --bounded
fi
test "${CONTAINER_ID:-}" = 50577514
unset CONTAINER_API_KEY OPENROUTER_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
R=/workspace/longmemeval_s_native7_20260910
M=/workspace/quad_3090_20260911
P311=/.uv/python_install/cpython-3.11-linux-x86_64-gnu/bin/python3.11
export PATH=/venv/main/bin:$PATH
export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
unset UV_NO_CACHE
export UV_CACHE_DIR=/workspace/quad_3090_20260911/uv-cache UV_LINK_MODE=hardlink UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=2
export PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
command -v uv >/dev/null
python3 -c 'import sys; assert sys.version_info[:3] == (3,12,14), sys.version'
"$P311" -c 'import sys; assert sys.version_info[:3] == (3,11,16), sys.version'
test -f "$M/model_integrity_expected.json"
test -f "$R/source/LightMem/src/lightmem/memory/lightmem.py"
mkdir -p "$R/logs" "$HF_HOME/hub"

disk_ok() {
  python3 - <<'PY'
import shutil
assert shutil.disk_usage('/workspace').free >= 3 * 1024**3, 'Less than 3 GiB free; preserve results and report setup failure'
PY
}

install_env() {
  local name="$1" interpreter="$2" requirements="$3"
  disk_ok
  if [[ -x "$R/$name/bin/python" && -s "$M/bootstrap_$name.freeze.txt" ]] &&
      uv pip freeze --python "$R/$name/bin/python" | cmp -s "$M/bootstrap_$name.freeze.txt" - &&
      uv pip check --python "$R/$name/bin/python"; then
    printf '%s already verified; preserving the shared download cache\n' "$name"
    return
  fi
  if [[ ! -x "$R/$name/bin/python" ]]; then
    uv venv --python "$interpreter" "$R/$name"
  fi
  uv pip install --python "$R/$name/bin/python" --no-deps --only-binary :all: -r "$requirements"
  uv pip check --python "$R/$name/bin/python"
  uv pip freeze --python "$R/$name/bin/python" > "$M/bootstrap_$name.freeze.txt"
  test "$UV_CACHE_DIR" = /workspace/quad_3090_20260911/uv-cache
  uv cache clean --cache-dir "$UV_CACHE_DIR"
  disk_ok
}

install_environments() {
  install_env .venv-inference "$(command -v python3)" "$M/bootstrap_inference_requirements.txt"
  install_env .venv-lightmem "$P311" "$M/bootstrap_lightmem_requirements.txt"
  # Same pinned source tree is loaded directly by native_lightmem.py. Reproduce
  # editable importability without resolving unpinned build-system dependencies.
  "$R/.venv-lightmem/bin/python" - <<'PY'
import sysconfig
from pathlib import Path
(Path(sysconfig.get_path('purelib')) / 'lightmem_pinned_source.pth').write_text(
    '/workspace/longmemeval_s_native7_20260910/source/LightMem/src\n')
PY
  install_env .venv-simplemem-native0253 "$P311" "$M/bootstrap_simplemem_requirements.txt"
  "$R/.venv-inference/bin/python" -c 'import torch,transformers,vllm; assert torch.__version__=="2.10.0+cu129"; assert transformers.__version__=="5.5.3"; assert vllm.__version__=="0.19.1"'
  "$R/.venv-lightmem/bin/python" -c 'import torch,transformers; from lightmem.memory.lightmem import LightMemory; assert torch.__version__.split("+")[0]=="2.8.0"; assert transformers.__version__=="4.57.0"'
  "$R/.venv-simplemem-native0253/bin/python" -c 'import torch,lancedb,tantivy; from importlib.metadata import version; assert torch.__version__=="2.8.0+cpu"; assert lancedb.__version__=="0.25.3"; assert version("tantivy")=="0.25.0"'
}


install_environments
disk_ok
printf 'Pinned environments verified.\\n'
