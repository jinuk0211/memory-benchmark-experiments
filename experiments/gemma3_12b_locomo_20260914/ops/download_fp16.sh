#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/gemma3_locomo_51017220
MODELS="$ROOT/models"
URL=https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-f16.gguf
TARGET="$MODELS/gemma-3-12b-it-f16.gguf"
EXPECTED_SIZE=23539658496
EXPECTED_SHA=78cd8535609c7b1c517c2b6a2f2935486e915e116f9d74863a8e787e1c08ef44

mkdir -p "$MODELS" "$ROOT/logs"
rm -f "$MODELS/gemma-3-12b-it-Q4_K_M.gguf"
curl -fL --retry 10 --retry-delay 5 --continue-at - "$URL" --output "$TARGET"
actual_size=$(stat -c %s "$TARGET")
test "$actual_size" = "$EXPECTED_SIZE"
echo "$EXPECTED_SHA  $TARGET" | sha256sum --check -
sha256sum "$TARGET" > "$ROOT/model_fp16.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/fp16_download_complete.txt"
