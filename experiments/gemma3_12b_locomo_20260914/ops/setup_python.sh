#!/usr/bin/env bash
set -euo pipefail

root=/workspace/gemma3_locomo_51017220
bundle="$root/bundle"
venv="$root/.venv"
export HF_HOME="$root/hf_home"
export FASTEMBED_CACHE_PATH="$root/fastembed_cache"

if [ ! -x "$venv/bin/python" ]; then
  uv venv "$venv" --python 3.12
fi
uv pip install --python "$venv/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
(
  cd "$bundle/MemoryData"
  uv pip install --python "$venv/bin/python" -r requirements-six-baselines.txt
)
uv pip install --python "$venv/bin/python" fastapi uvicorn fastembed jsonschema pytest editdistance bert-score

"$venv/bin/python" -m nltk.downloader -d "$root/nltk_data" punkt punkt_tab stopwords wordnet
"$venv/bin/python" -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup']))"
"$venv/bin/python" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu').encode(['warmup'])"
uv pip freeze --python "$venv/bin/python" > "$root/python_freeze.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/python_setup_complete.txt"

