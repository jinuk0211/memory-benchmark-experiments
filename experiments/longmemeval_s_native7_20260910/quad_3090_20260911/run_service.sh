#!/bin/bash
set -euo pipefail
test "${CONTAINER_ID:-}" = 50577514
unset CONTAINER_API_KEY OPENROUTER_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN
R=/workspace/longmemeval_s_native7_20260910
Q=/workspace/quad_3090_20260911
export PYTHONUNBUFFERED=1 HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$R:$R/fast_native2_20260911" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NLTK_DATA=/root/.cache/longmemeval_s_native7_20260910/nltk_data
cd "$R"
case "${1:?Service required}" in
  pipeline) exec /usr/bin/python3 "$Q/pipeline.py" ;;
  meter) exec "$R/.venv-inference/bin/python" "$Q/quad_metered_proxy.py" ;;
  simplemem) export CUDA_VISIBLE_DEVICES=''; exec "$R/.venv-simplemem-native0253/bin/python" "$Q/quad_simplemem_queue.py" ;;
  lightmem) exec "$R/.venv-lightmem/bin/python" "$Q/quad_lightmem_queue.py" ;;
  *) printf 'Unknown service\n' >&2; exit 2 ;;
esac