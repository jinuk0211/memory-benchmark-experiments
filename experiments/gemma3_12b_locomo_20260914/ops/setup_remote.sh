#!/usr/bin/env bash
set -euo pipefail

root=/workspace/gemma3_locomo_51017220
mkdir -p "$root/models" "$root/logs" "$root/src"

if [[ ! -d "$root/src/llama.cpp/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$root/src/llama.cpp"
else
  git -C "$root/src/llama.cpp" pull --ff-only
fi

cmake -S "$root/src/llama.cpp" -B "$root/src/llama.cpp/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF
cmake --build "$root/src/llama.cpp/build" --target llama-server llama-cli -j 16

model="$root/models/gemma-3-12b-it-Q4_K_M.gguf"
url='https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf?download=true'
curl -fL --retry 8 --retry-all-errors --continue-at - --output "$model.part" "$url"
mv "$model.part" "$model"

sha256sum "$model" > "$model.sha256"
git -C "$root/src/llama.cpp" rev-parse HEAD > "$root/llama_cpp_revision.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/setup_complete.txt"