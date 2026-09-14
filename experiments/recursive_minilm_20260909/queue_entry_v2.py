"""Continue v2 only after the frozen v1 fails its existing transfer gate."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'gemma_transfer'))
from candidate_gate import decision
from guard_inputs import save_once


def main() -> int:
    report_path = ROOT / 'runs/recursive_v1/OFFICIAL_COMPARISON.json'
    decision_path = ROOT / 'gemma_transfer/CANDIDATE_DECISION.json'
    report = json.loads(report_path.read_text(encoding='utf8'))
    recorded = json.loads(decision_path.read_text(encoding='utf8'))
    reference = json.loads((ROOT / 'gemma_transfer/BASELINE_REFERENCE.json').read_text(encoding='utf8'))
    expected = decision(report, reference)
    if (recorded['locomo_report_sha256'] != hashlib.sha256(report_path.read_bytes()).hexdigest()
            or any(recorded.get(k) != v for k, v in expected.items())
            or report['method'] != 'recursive_source_rehearsal_v1'
            or report['dataset_sha256'] != 'cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993'):
        raise ValueError('Existing v1 transfer decision or comparison changed')
    for method, digest in report['prediction_hashes'].items():
        folder = 'recursive_v1' if method == report['method'] else 'qwen35_baseline'
        if hashlib.sha256((ROOT / 'runs' / folder / (method + '.jsonl')).read_bytes()).hexdigest() != digest:
            raise ValueError('Measured v1 predictions changed')
    if expected['ready_for_transfer']:
        print('V1_QUALIFIED_LEAVE_ITS_GEMMA_QUEUE_RUNNING')
        return 3
    receipt = {**expected, 'decision_sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest(),
               'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
               'v2_reason': 'v1 misses predefined baseline-win/source-development gate'}
    save_once(ROOT / 'runs/recursive_v2/PREDECESSOR_DECISION.json', json.dumps(receipt, indent=2).encode())
    print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
