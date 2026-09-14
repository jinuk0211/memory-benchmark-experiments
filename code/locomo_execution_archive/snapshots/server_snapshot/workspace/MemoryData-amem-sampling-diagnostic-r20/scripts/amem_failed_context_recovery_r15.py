"""After existing A-MEM and SimpleMem exit, retry only failed original A-MEM shards."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import locomo_server_queue as queue
from scripts.finalize_locomo_comparison import running_writers
from scripts.run_locomo_live_probes import run_probe
from scripts.retry_simplemem_after_queue import require_free_ports
from scripts.score_locomo_comparison import write_report

PLAN_PATH = Path('/workspace/finish-amem-failed-contexts-plan-r15.json')
PRIOR_OUTPUT = Path('/workspace/locomo-finish-amem-validated-r12/a_mem')
PROBE_OUTPUT = Path('/workspace/locomo-amem-recovery-probe-r15')
PREDECESSORS = ('locomo-amem-probe-then-full-r12', 'locomo-simplemem-probe-then-full-r13')
OTHER_PRIOR_OUTPUTS = (Path('/workspace/locomo-amem-live-probe-r12'),
                      Path('/workspace/locomo-simplemem-live-probe-r13'),
                      Path('/workspace/locomo-finish-simplemem-validated-r13'))


def failed_contexts(status: dict) -> list[int]:
    """Only a complete, unambiguous terminal event set permits retry selection."""
    if status.get('active_contexts') != []:
        raise ValueError('Original context workers are still active or unobserved')
    starts = [row for row in status['events'] if row.get('event') == 'start']
    ends = [row for row in status['events'] if row.get('event') == 'end']
    for rows in (starts, ends):
        if (len(rows) != 10 or any(type(row.get('context')) is not int for row in rows)
                or sorted(row['context'] for row in rows) != list(range(10))):
            raise ValueError('All ten original contexts must have one start and one terminal event')
    if any(type(row.get('exit_code')) is not int for row in ends):
        raise ValueError('Original exit codes must be known integers')
    failed = sorted(row['context'] for row in ends if row['exit_code'] != 0)
    if status.get('failed') is not bool(failed):
        raise ValueError('Original failure marker disagrees with terminal events')
    return failed


def main() -> int:
    plan = json.loads(PLAN_PATH.read_text())
    queue.validate_plan(plan)
    output = Path(plan['output'])
    if output.exists():
        raise FileExistsError(output)
    PROBE_OUTPUT.mkdir(exist_ok=False)
    state_path = PROBE_OUTPUT / 'recovery_status.json'
    write_report(state_path, {'state': 'waiting_for_predecessors', 'complete': False})
    try:
        while True:
            states = {name: queue.supervisor_state(name) for name in PREDECESSORS}
            if any(value not in {'RUNNING', 'STARTING', 'STOPPING', 'EXITED'} for value in states.values()):
                raise RuntimeError(f'Unexpected predecessor state: {states}')
            if all(value == 'EXITED' for value in states.values()):
                break
            time.sleep(15)
        if any(running_writers(path) for path in (PRIOR_OUTPUT.parent, *OTHER_PRIOR_OUTPUTS)):
            raise RuntimeError('Prior experiment artifact writers are still live')
        indices = failed_contexts(json.loads((PRIOR_OUTPUT / 'parallel_status.json').read_text()))
        if not indices:
            write_report(state_path, {'state': 'no_failed_contexts', 'complete': False})
            return 0
        preserved = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in PRIOR_OUTPUT.rglob('*') if path.is_file()}
        write_report(PROBE_OUTPUT / 'prior_artifact_sha256.json', preserved)
        write_report(PROBE_OUTPUT / 'selection.json', {'context_indices': indices,
                     'prior_output': str(PRIOR_OUTPUT), 'successful_contexts_reexecuted': False})
        require_free_ports()
        queue.validate_plan(plan)
        queue.wait_vllm(plan)
        probe = dict(plan, output=str(PROBE_OUTPUT), run_id='locomo-amem-recovery-probe-20260908-r15')
        encoder = [plan['server_python'], str(ROOT / 'scripts/serve_minilm_embeddings.py'),
                   '--model-path', plan['embedding_path']]
        write_report(state_path, {'state': 'testing', 'complete': False, 'context_indices': indices})
        with queue.service(encoder, PROBE_OUTPUT / 'embedding_server.log') as process:
            queue.wait_health('http://127.0.0.1:18081', process)
            report = run_probe(probe, 'a_mem', 80)
        if report.get('passed') is not True:
            raise RuntimeError('A-MEM recovery smoke did not pass')
        require_free_ports()
        queue.validate_plan(plan)
        output.mkdir(exist_ok=False)
        write_report(state_path, {'state': 'running_failed_contexts', 'complete': False,
                     'context_indices': indices, 'smoke_passed': True})
        expected = queue.expected_questions(plan['dataset'], 1540)
        item = next(item for item in plan['methods'] if item['method'] == 'a_mem')
        with queue.service(encoder, output / 'embedding_server.log') as process:
            queue.wait_health('http://127.0.0.1:18081', process)
            result = queue.run_method(plan, item, expected, context_indices=indices)
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
               for path, digest in preserved.items()):
            raise RuntimeError('Original A-MEM artifacts changed during recovery')
        write_report(state_path, {'state': 'awaiting_composite_audit', 'complete': False,
                     'context_indices': indices, 'original_artifact_hashes_preserved': True,
                     'selected_attempt': result})
        return 0
    except Exception as error:
        write_report(state_path, {'state': 'failed', 'complete': False,
                     'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
