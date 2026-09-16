#!/bin/bash
set -euo pipefail

manifest=/workspace/quad_3090_20260911/model_integrity_expected.json
marker=/workspace/longmemeval_s_native7_20260910/models_downloaded.txt
max_jobs=6

download_file() {
  local key="$1"
  local repo="$2"
  local destination="$3"
  local relative="$4"
  local expected="$5"
  local target="$destination/$relative"
  local partial="$target.part"
  local actual=""
  mkdir -p "$(dirname "$target")"
  if test -f "$target"; then
    actual=$(sha256sum "$target" | cut -d' ' -f1)
  fi
  if test "$actual" = "$expected"; then
    echo "VERIFIED_EXISTING $key $relative"
    return 0
  fi
  test ! -e "$target" || mv "$target" "$target.invalid.$(date +%s)"
  local encoded
  encoded=$(jq -rn --arg value "$relative" '$value|@uri')
  local url="https://modelscope.cn/api/v1/models/$repo/repo?Revision=master&FilePath=$encoded"
  echo "DOWNLOADING $key $relative"
  aria2c --continue=true --max-connection-per-server=8 --split=8 --min-split-size=8M \
    --file-allocation=none --max-tries=0 --retry-wait=3 --timeout=30 \
    --auto-file-renaming=false --allow-overwrite=false --summary-interval=30 \
    --console-log-level=warn --dir="$(dirname "$partial")" --out="$(basename "$partial")" "$url"
  actual=$(sha256sum "$partial" | cut -d' ' -f1)
  if test "$actual" != "$expected"; then
    echo "SHA256_MISMATCH $key $relative expected=$expected actual=$actual" >&2
    return 1
  fi
  mv "$partial" "$target"
  echo "VERIFIED_DOWNLOADED $key $relative"
}

queue_model() {
  local key="$1"
  local repo="$2"
  local destination
  destination=$(jq -r ".models.${key}.path" "$manifest")
  mkdir -p "$destination"
  while IFS=$'\t' read -r relative expected; do
    while test "$(jobs -rp | wc -l)" -ge "$max_jobs"; do
      wait -n
    done
    download_file "$key" "$repo" "$destination" "$relative" "$expected" &
  done < <(jq -r ".models.${key}.files|to_entries[]|[.key,.value]|@tsv" "$manifest")
}

queue_model qwen Qwen/Qwen3.5-9B
queue_model minilm sentence-transformers/all-MiniLM-L6-v2
queue_model compressor microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
wait
printf '%s\n' QWEN35_MINILM_LLMINGUA_SHA256_VERIFIED > "$marker"
echo ALL_MODELS_SHA256_VERIFIED
