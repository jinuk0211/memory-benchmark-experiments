"""Run the live A-MEM smoke gate, then the unchanged full 1540-QA plan."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import locomo_server_queue as queue
from scripts.run_locomo_live_probes import run_probe
from scripts.retry_simplemem_after_queue import require_free_ports
from scripts.score_locomo_comparison import write_report


def main():
    full_path = Path('/workspace/finish-amem-validated-plan-r12.json')
    full = json.loads(full_path.read_text())
    queue.validate_plan(full)
    if Path(full['output']).exists():
        raise FileExistsError(full['output'])
    plan = dict(full, output='/workspace/locomo-amem-live-probe-r12',
                run_id='locomo-amem-live-probe-20260908-r12')
    out = Path(plan['output'])
    out.mkdir(exist_ok=False)
    write_report(out / 'probe_plan.json', plan)
    write_report(out / 'probe_status.json', {'state': 'waiting_for_emem', 'passed': False})
    while queue.supervisor_state('locomo-emem-saved-recovery-r11') in {'RUNNING', 'STARTING', 'STOPPING'}:
        time.sleep(15)
    require_free_ports()
    queue.validate_plan(plan)
    queue.wait_vllm(plan)
    try:
        with queue.service([plan['server_python'], str(ROOT / 'scripts/serve_minilm_embeddings.py'),
                            '--model-path', plan['embedding_path']], out / 'embedding_server.log') as encoder:
            queue.wait_health('http://127.0.0.1:18081', encoder)
            write_report(out / 'probe_status.json', {'state': 'testing', 'passed': False})
            report = run_probe(plan, 'a_mem', 40)
    except Exception as error:
        write_report(out / 'probe_status.json', {'state': 'failed', 'passed': False,
                     'error_type': type(error).__name__, 'error': str(error)})
        raise
    write_report(out / 'probe_status.json', {'state': 'passed_starting_full', 'passed': True,
                 'qa_count': report['qa_count'], 'memory_turns': report['memory_turns'],
                 'response_delivery': report['response_delivery']})
    require_free_ports()
    print('LIVE_PROBE_PASSED_STARTING_FULL_1540', flush=True)
    return subprocess.run([full['server_python'], str(ROOT / 'scripts/run_remaining.py'),
                           '--plan', str(full_path), '--methods', 'a_mem'], cwd=ROOT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
