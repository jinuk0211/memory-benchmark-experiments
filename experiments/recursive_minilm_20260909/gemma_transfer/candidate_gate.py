"""Freeze a measured LoCoMo candidate before opening the transfer evaluation."""
import hashlib
import json
import math
from pathlib import Path

from guard_inputs import save_once, validate

ROOT = Path(__file__).resolve().parent
METHOD = 'recursive_source_rehearsal_v1'


def decision(report: dict, reference: dict) -> dict:
    recursive = report['methods'][METHOD]
    original = report['methods']['s_parent_single_2000']
    values = [recursive['official_f1'], original['official_f1'], reference['official_f1'],
              report['paired']['development7']['effect_pp']]
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values):
        raise ValueError('Nonfinite candidate score')
    if (recursive['n'] != 1540 or original['n'] != 1540 or reference['n'] != 1540
            or not reference['predictions_complete'] or not reference['coverage_complete']):
        raise ValueError('Incomplete comparison population')
    return {'ready_for_transfer': values[0] > values[1] and values[0] > values[2] and values[3] > 0,
            'recursive_f1': values[0], 'original_f1': values[1],
            'best_completed_baseline_f1': values[2], 'development7_effect_pp': values[3],
            'rule': 'Full1540 beats original and best completed baseline; development7 improves.'}


def main() -> int:
    source = ROOT.parent
    report_path = source / 'runs/recursive_v1/OFFICIAL_COMPARISON.json'
    raw = report_path.read_bytes()
    report = json.loads(raw)
    reference = json.loads((ROOT / 'BASELINE_REFERENCE.json').read_text())
    result = decision(report, reference)
    result['locomo_report_sha256'] = hashlib.sha256(raw).hexdigest()
    if report['method'] != METHOD or report['dataset_sha256'] != 'cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993':
        raise ValueError('Unexpected LoCoMo method or corpus')
    for method, expected in report['prediction_hashes'].items():
        folder = 'recursive_v1' if method == METHOD else 'qwen35_baseline'
        if hashlib.sha256((source / 'runs' / folder / (method + '.jsonl')).read_bytes()).hexdigest() != expected:
            raise ValueError('LoCoMo predictions changed')
    files = ['recursive_memory.py'] + [str(path.relative_to(ROOT)) for path in sorted((ROOT / 'source').glob('*.py'))]
    code = {}
    for name in files:
        transferred = (ROOT / name).read_bytes()
        if transferred != (source / name).read_bytes():
            raise ValueError('Transfer algorithm differs from measured Qwen: ' + name)
        code[name] = hashlib.sha256(transferred).hexdigest()
    result['algorithm_sha256'] = code
    result['transfer_input'] = validate(ROOT / 'data/longmemeval18.json', 'longmemeval', None)
    save_once(ROOT / 'CANDIDATE_DECISION.json', json.dumps(result, indent=2).encode())
    print(json.dumps({key: value for key, value in result.items() if key not in ('algorithm_sha256', 'transfer_input')}))
    if not result['ready_for_transfer']:
        return 3
    save_once(ROOT / 'CANDIDATE_FREEZE.json', json.dumps(result, indent=2).encode())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
