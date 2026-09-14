#!/bin/bash
set -euo pipefail
R=/workspace/longmemeval_s_native7_20260910
S=$R/fast_native2_20260911
P=$R/.venv-simplemem-native0253/bin/python
C=$R/official_recovery/simplemem_native_dialogues_v3
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 SKIP_SBERT_SIM=1 PYTHONUNBUFFERED=1 OPENAI_API_KEY=EMPTY SIMPLEMEM_API_KEY=EMPTY SIMPLEMEM_EMBEDDING_API_KEY=EMPTY NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 NLTK_DATA=/root/.cache/longmemeval_s_native7_20260910/nltk_data
unset OPENROUTER_API_KEY CONTAINER_API_KEY
cd "$R"
if [ ! -f "$S/simplemem_v3_runtime.json" ]; then
  echo '["118b2229"]' > "$S/simplemem_v3_probe_ids.json"
  "$P" "$C/runner.py" --dataset "$R/longmemeval_s_cleaned.json" --run-dir "$R/runs/simplemem_native_dialogues_v3" --api-base http://127.0.0.1:18083/v1 --model Qwen/Qwen3.5-9B --embedding-model /workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --ids-file "$S/simplemem_v3_probe_ids.json"
  "$P" - <<'PY'
import importlib.util, hashlib, json, time
from pathlib import Path
root=Path('/workspace/longmemeval_s_native7_20260910')
candidate=root/'official_recovery/simplemem_native_dialogues_v3'
spec=importlib.util.spec_from_file_location('proof',candidate/'runner.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
run=root/'runs/simplemem_native_dialogues_v3'
protocol=native.read_json(run/'protocol.json')
row=next(x for x in native.read_json(root/'longmemeval_s_cleaned.json') if x['question_id']=='118b2229')
identity={'protocol_sha256':native.digest(protocol),'source_sha256':native.digest(native.source_only(row)), 'query_sha256':native.digest({k:row[k] for k in ('question_id','question','question_date')})}
assert native.verified(run/'histories'/hashlib.sha256(b'118b2229').hexdigest()[:24],identity)
assert native.source_hashes()==protocol['source_files_sha256']
native.save_json(root/'fast_native2_20260911/simplemem_v3_runtime.json', {'status':'runtime_verified','candidate':str(candidate),'verified_at':time.time(),'probe_question_id':'118b2229','protocol_sha256':native.digest(protocol),'integrity':{'runtime_files':{'official_recovery/simplemem_native_dialogues_v3/runner.py':native.sha(candidate/'runner.py')}}})
print('SimpleMem v3 failed-case probe verified; starting all 500 histories', flush=True)
PY
fi
exec "$P" "$S/simplemem_queue.py" --runtime-receipt "$S/simplemem_v3_runtime.json"
