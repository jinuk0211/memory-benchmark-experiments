#!/bin/bash
set -euo pipefail
R=/workspace/longmemeval_s_native7_20260910
S=$R/fast_native2_20260911
cd "$R"
test "${CONTAINER_ID:-}" = 50468468
(cd "$R/official_recovery/simplemem_native_dialogues_v3" && "$R/.venv-simplemem-native0253/bin/python" -m unittest -q test_location_compat)
python3 -m pytest -q "$S/test_cost_guard.py" "$S/test_lightmem_queue.py" "$S/test_finish_and_stop.py"
python3 "$S/cost_guard.py" --check-only
python3 "$S/finish_and_stop.py" --expected-instance-id 50468468 --check-only
cat > /etc/supervisor/conf.d/native2-simplemem.conf <<EOF
[program:native2-simplemem]
command=/bin/bash $S/run_simplemem_v3.sh
directory=$R
autostart=true
autorestart=false
startsecs=3
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=60
redirect_stderr=true
stdout_logfile=$R/queue/native2-simplemem.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
EOF
cat > /etc/supervisor/conf.d/native2-finish.conf <<EOF
[program:native2-finish]
command=/usr/bin/python3 $S/finish_and_stop.py --expected-instance-id 50468468 --activate
directory=$R
autostart=true
autorestart=unexpected
startsecs=3
startretries=3
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile=$R/queue/native2-finish.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
EOF
cat > /etc/supervisor/conf.d/native2-cost-guard.conf <<EOF
[program:native2-cost-guard]
command=/usr/bin/python3 $S/cost_guard.py
directory=$R
autostart=true
autorestart=unexpected
startsecs=3
startretries=3
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile=$R/queue/native2-cost-guard.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
EOF
supervisorctl reread
supervisorctl update native2-simplemem native2-finish native2-cost-guard
supervisorctl status native2-simplemem native2-lightmem native2-finish native2-cost-guard
