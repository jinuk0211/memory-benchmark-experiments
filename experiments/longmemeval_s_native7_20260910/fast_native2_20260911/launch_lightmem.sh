#!/bin/bash
set -euo pipefail
R=/workspace/longmemeval_s_native7_20260910
S=$R/fast_native2_20260911
cd "$R"
test "${CONTAINER_ID:-}" = 50468468
echo '3bc3fa4afa36b22031ee85e1f08f72489893ac4152da4239fc2aa696417bdda6  fast_native2_20260911/lightmem_launch_20260911.tgz' | sha256sum -c -
tar -xzf "$S/lightmem_launch_20260911.tgz" -C "$R"
python3 -m pytest -q "$S/test_lightmem_queue.py"
cat > /etc/supervisor/conf.d/native2-lightmem.conf <<EOF
[program:native2-lightmem]
command=/usr/bin/env -u OPENROUTER_API_KEY -u CONTAINER_API_KEY HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 SKIP_SBERT_SIM=1 PYTHONUNBUFFERED=1 OPENAI_API_KEY=EMPTY NO_PROXY=localhost,127.0.0.1,::1 NLTK_DATA=/root/.cache/longmemeval_s_native7_20260910/nltk_data $R/.venv-lightmem/bin/python $S/lightmem_queue.py --plan $S/lightmem_plan.json --state-dir $S --protocol-sha256 0ce3945e48896edd5b7cf8d5f0c09e79c0047951a0208472e50740e2d1f3b326 --vendor $R/official_longmemeval --workers 1 --retry-failed
directory=$R
autostart=true
autorestart=false
startsecs=3
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=60
redirect_stderr=true
stdout_logfile=$R/queue/native2-lightmem.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
EOF
supervisorctl reread
supervisorctl update native2-lightmem
supervisorctl status native2-lightmem
