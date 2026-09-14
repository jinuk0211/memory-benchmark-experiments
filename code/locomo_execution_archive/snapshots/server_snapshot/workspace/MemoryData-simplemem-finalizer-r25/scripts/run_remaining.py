"""Versioned full LoCoMo runs, preserving every failed and completed context."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import locomo_server_queue as queue
from scripts.finalize_locomo_comparison import finalize_method
from scripts.score_locomo_comparison import expected_questions, write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--methods', nargs='+', required=True, choices=['a_mem', 'mem0', 'e_mem', 'simplemem'])
    parser.add_argument('--after-service')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    queue.validate_plan(plan)
    if args.after_service:
        while queue.supervisor_state(args.after_service) in {'RUNNING', 'STARTING', 'STOPPING'}:
            time.sleep(30)
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    expected = expected_questions(plan['dataset'], 1540)
    queue.wait_vllm(plan)
    outcomes = []
    with queue.service([plan['server_python'], str(ROOT / 'scripts/serve_minilm_embeddings.py'),
                        '--model-path', plan['embedding_path']], output / 'embedding_server.log') as encoder:
        queue.wait_health('http://127.0.0.1:18081', encoder)
        for method in args.methods:
            queue.validate_plan(plan)
            item = next(item for item in plan['methods'] if item['method'] == method)
            write_report(output / 'status.json', {'state': 'running', 'method': method, 'complete': False, 'outcomes': outcomes})
            try:
                report = queue.run_method(plan, item, expected)
                outcome = {'method': method, 'complete': report['complete'], 'report': report}
            except Exception as exc:
                outcome = {'method': method, 'complete': False, 'error_type': type(exc).__name__, 'error': str(exc)}
                marker = output / method / 'parallel_status.json'
                if marker.exists() and json.loads(marker.read_text()).get('complete'):
                    try:
                        report = finalize_method(plan, item, expected, output / 'finalized' / method)
                        outcome.update(complete=report['complete'], report=report)
                    except Exception as audit_error:
                        outcome['audit_error'] = str(audit_error)
            outcomes.append(outcome)
            print(json.dumps({k: v for k, v in outcome.items() if k != 'report'}), flush=True)
    complete = all(item['complete'] for item in outcomes)
    write_report(output / 'status.json', {'state': 'complete' if complete else 'needs_repair', 'complete': complete, 'outcomes': outcomes})
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
