#!/usr/bin/env bash
set -euo pipefail
unset CONTAINER_API_KEY OPENROUTER_API_KEY DUAL_GPU_ROUTES
TASK_ROOT=/workspace/longmemeval_s_native7_20260910
STATE="$TASK_ROOT/queue/dual_gpu_20260911"
export METER_INSTANCE_ID=50558359
export METER_ACTIVITY_PATH="$STATE/inference_activity.json"
export PYTHONUNBUFFERED=1
cd "$TASK_ROOT"
exec "$TASK_ROOT/.venv-inference/bin/python" "$STATE/dual_metered_proxy.py" --upstream-base-url http://127.0.0.1:18087/v1 --host 127.0.0.1 --port 18081 --journal "$STATE/v100_usage.jsonl" --run-id dual_gpu_20260911 --method unattributed --timeout 600
