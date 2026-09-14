#!/bin/bash
set -euo pipefail
cd /workspace/generalization_20260908/modern
unset UV_NO_CACHE
export UV_CACHE_DIR=/workspace/generalization_20260908/modern/uv_cache
export UV_HTTP_RETRIES=5
export UV_CONCURRENT_DOWNLOADS=4
export UV_HTTP_TIMEOUT=120
retry_install() {
  for attempt in 1 2 3; do
    if "$@"; then return 0; fi
    echo "INSTALL_RETRY attempt=$attempt" >&2
    sleep 10
  done
  return 1
}
if [ ! -x .venv/bin/python ]; then
  uv venv --python /venv/main/bin/python .venv
fi
/workspace/locomo-refinement/.venv/bin/python fetch_runtime_wheels.py
retry_install /venv/main/bin/python -m pip --python .venv/bin/python install --no-deps /workspace/generalization_20260908/modern/wheelhouse/*.whl
retry_install /venv/main/bin/python -m pip --python .venv/bin/python install --cache-dir /workspace/generalization_20260908/modern/pip_cache --retries 5 --resume-retries 10 --timeout 120 --find-links /workspace/generalization_20260908/modern/wheelhouse 'vllm==0.19.1' 'torch==2.10.0+cu129' 'torchvision==0.25.0+cu129' 'torchaudio==2.10.0+cu129' 'transformers==5.5.3' 'tokenizers==0.22.2' 'huggingface-hub==1.6.0' 'sentence-transformers==5.2.0' 'numpy==2.2.6' 'nltk==3.9.1' 'rank-bm25==0.2.2' 'pytest==8.3.5'
uv pip check --python .venv/bin/python
uv pip freeze --python .venv/bin/python > installed_packages.txt
.venv/bin/python -c 'import vllm, torch, transformers; from vllm.engine.arg_utils import EngineArgs; assert "language_model_only" in EngineArgs.__dataclass_fields__; print("IMPORTS_OK", vllm.__version__, torch.__version__, transformers.__version__, flush=True)'
printf '%s\n' 'Dependency installation and CPU imports passed; GPU compatibility is not yet established.' > runtime_installed.txt