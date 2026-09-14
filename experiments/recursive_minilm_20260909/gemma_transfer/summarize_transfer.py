"""Exact18 paired transfer report; token F1 never substitutes for official accuracy."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
from guard_inputs import validate
from transfer_data import paired_summary
from judge_longmemeval import JUDGE_MODEL, TYPES

METHODS = ('seed', 'r40_fused_four_turn', 's_parent_single_2000', 'recursive_source_rehearsal_v1')


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def checked_rows(path: Path, references: dict, method: str) -> list[dict]:
    rows = read_rows(path)
    if len(rows) != 18 or {row['question_id'] for row in rows} != set(references):
        raise ValueError('Expected exact18 unique predictions')
    for row in rows:
        ref = references[row['question_id']]
        if (row['method'] != method or row['question'] != ref['question']
                or row['gold'] != str(ref['answer']) or row['question_type'] != ref['question_type']
                or not isinstance(row['hypothesis'], str) or row['hypothesis'] != row['prediction']):
            raise ValueError('Transfer prediction metadata mismatch')
        row['diagnostic_token_f1'] = core.generic_f1(row['hypothesis'], ref['answer'])
    return rows


def attach_official(rows: list[dict], path: Path) -> bool:
    if not path.exists():
        return False
    labels = read_rows(path)
    if len(labels) != len(rows) or {r['question_id'] for r in labels} != {r['question_id'] for r in rows}:
        return False
    by_id = {r['question_id']: r for r in labels}
    for row in rows:
        label = by_id[row['question_id']]
        verdict = label['autoeval_label']
        if (label['hypothesis'] != row['hypothesis'] or verdict['model'] != JUDGE_MODEL
                or type(verdict['label']) is not bool):
            raise ValueError('Official judgment mismatch')
        row['official_accuracy'] = int(verdict['label'])
    return True


def main() -> None:
    data = ROOT / 'data/longmemeval18.json'
    lock = validate(data, 'longmemeval', None)
    freeze_raw = (ROOT / 'CANDIDATE_FREEZE.json').read_bytes()
    if not json.loads(freeze_raw)['ready_for_transfer']:
        raise ValueError('Candidate was not frozen for transfer')
    references = {r['question_id']: r for r in json.loads(data.read_bytes())}
    inputs = {method: ROOT / 'runs' / ('gemma_recursive_v1' if method == METHODS[-1] else 'gemma_baseline')
              / (method + '.jsonl') for method in METHODS}
    hashes = {method: hashlib.sha256(path.read_bytes()).hexdigest() for method, path in inputs.items()}
    rows = {method: checked_rows(path, references, method) for method, path in inputs.items()}
    official = {method: attach_official(items, ROOT / 'judgments' / (method + '.jsonl')) for method, items in rows.items()}
    metrics = ['diagnostic_token_f1'] + (['official_accuracy'] if all(official.values()) else [])
    report = {'n': 18, 'input_lock': lock, 'candidate_freeze_sha256': hashlib.sha256(freeze_raw).hexdigest(),
              'prediction_hashes': hashes, 'official_complete': all(official.values()),
              'official_complete_by_method': official, 'judge_model': JUDGE_MODEL,
              'metrics': {}, 'scope': 'Precommitted18 cross-model/dataset pilot, not full500 or proof of broad generalization.'}
    lines = ['# Gemma4-E4B + MiniLM LongMemEval18', '',
             'Fixed complete histories:3questions per type,4abstentions. Paired comparison;0–100 scores.', '']
    for metric in metrics:
        comparisons = {base: paired_summary(rows[base], rows[METHODS[-1]], manifest=lock['selected_ids'],
                        score_key=metric) for base in METHODS[:-1]}
        summaries = {method: {'overall': statistics.mean(r[metric] for r in items),
                    'by_type': {kind: statistics.mean(r[metric] for r in items if r['question_type'] == kind) for kind in TYPES},
                    'abstention': statistics.mean(r[metric] for r in items if r['question_id'].endswith('_abs'))}
                    for method, items in rows.items()}
        report['metrics'][metric] = {'methods': summaries, 'paired_recursive_minus': comparisons}
        lines += [metric + (' (diagnostic only)' if metric.startswith('diagnostic') else ' (pinned official judge)'), '',
                  '| Method | All18 | User | Assistant | Preference | Multi-session | Temporal | Update | Abstention |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for method, item in summaries.items():
            values = [item['overall']] + [item['by_type'][kind] for kind in TYPES] + [item['abstention']]
            lines.append('| ' + method + ' | ' + ' | '.join(f'{v*100:.3f}' for v in values) + ' |')
        lines.append('')
    if not report['official_complete']:
        lines.append('Official accuracy remains pending. Token F1 cannot establish the official improvement claim.')
    if hashes != {method: hashlib.sha256(path.read_bytes()).hexdigest() for method, path in inputs.items()}:
        raise ValueError('Predictions changed during summary')
    (ROOT / 'TRANSFER_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'n': 18, 'official_complete': report['official_complete'], 'metrics': metrics}))


if __name__ == '__main__':
    main()
