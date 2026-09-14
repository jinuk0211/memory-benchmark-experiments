#!/usr/bin/env bash
set -uo pipefail
cd /workspace/MemoryData || exit 2
retry_plan=/workspace/MemoryData/comparison_config/plan-emem-tools.json
retry_output=/workspace/locomo-comparison-qwen35-fp16-emem-tools
final_output=/workspace/locomo-emem-tools-finalized-20260907
if [[ -e "$retry_output" || -e "$final_output" ]]; then
    printf '%s\n' 'Refusing to reuse retry artifacts.' >&2
    exit 2
fi
/venv/main/bin/python scripts/locomo_server_queue.py --plan "$retry_plan"
queue_exit=$?
if [[ "$queue_exit" -ne 0 && "$queue_exit" -ne 1 ]]; then
    exit "$queue_exit"
fi
/venv/main/bin/python scripts/finalize_locomo_comparison.py --plan "$retry_plan" --output "$final_output" || exit $?
jq -e '.finalization_complete == true and any(.methods[]; .method == "e_mem" and .state == "complete")' "$final_output/summary.json" >/dev/null
