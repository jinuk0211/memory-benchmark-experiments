"""Canonical full1540 scoring and paired recursive-improvement comparison."""
from collections import defaultdict
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
from transfer_data import paired_summary
from recursive_memory import RECIPE

DATA_HASH = 'cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993'
SCORER_HASH = '8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd'
REPORTING_CONVS = {'conv-42', 'conv-47', 'conv-50'}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def validate_rows(dataset: list[dict], rows: list[dict], method: str) -> list[dict]:
    expected = {f"{sample['sample_id']}:{i}": (sample['sample_id'], i, qa)
                for sample in dataset for i, qa in enumerate(sample['qa']) if qa['category'] in (1, 2, 3, 4)}
    if len(expected) != 1540 or len(rows) != 1540 or len({row['question_id'] for row in rows}) != 1540:
        raise ValueError('Incomplete/duplicate canonical1540 population')
    by_id = {row['question_id']: row for row in rows}
    if set(by_id) != set(expected):
        raise ValueError('Unexpected question IDs')
    result = []
    for identity, (cid, index, qa) in expected.items():
        row = by_id[identity]
        if (row['conversation_id'] != cid or row['method'] != method or row['question'] != qa['question']
                or row['gold'] != str(qa['answer']) or row['category'] != qa['category']
                or not isinstance(row['prediction'], str) or row['hypothesis'] != row['prediction']):
            raise ValueError('Canonical QA or prediction metadata changed')
        result.append({**qa, 'sample': cid, 'index': index, 'prediction': row['prediction']})
    return result


def main() -> None:
    data = ROOT / 'data/locomo10.json'
    scorer = ROOT / 'source/official_evaluation.py'
    if hashlib.sha256(data.read_bytes()).hexdigest() != DATA_HASH or hashlib.sha256(scorer.read_bytes()).hexdigest() != SCORER_HASH:
        raise ValueError('Official scoring source differs')
    dataset = json.loads(data.read_text())
    spec = importlib.util.spec_from_file_location('canonical_official', scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = {'seed': ROOT / 'runs/qwen35_baseline/seed.jsonl',
              'r40_fused_four_turn': ROOT / 'runs/qwen35_baseline/r40_fused_four_turn.jsonl',
              's_parent_single_2000': ROOT / 'runs/qwen35_baseline/s_parent_single_2000.jsonl',
              RECIPE['name']: ROOT / 'runs/recursive_v1' / (RECIPE['name'] + '.jsonl')}
    hashes = {method: hashlib.sha256(path.read_bytes()).hexdigest() for method, path in inputs.items()}
    rows_by_method, summaries = {}, {}
    for method, path in inputs.items():
        rows = read_jsonl(path)
        records = validate_rows(dataset, rows, method)
        with contextlib.redirect_stdout(io.StringIO()):
            scores, _, _ = module.eval_question_answering(records, eval_key='prediction', metric='f1')
        by_id = {row['question_id']: row for row in rows}
        ordered = [by_id[f"{row['sample']}:{row['index']}"] for row in records]
        if len(scores) != 1540:
            raise ValueError('Official scorer count mismatch')
        buckets = defaultdict(list)
        for row, score in zip(ordered, scores):
            if not 0 <= score <= 1:
                raise ValueError('Invalid official score')
            row['official_f1'] = float(score)
            buckets[str(row['category'])].append(score)
        rows_by_method[method] = ordered
        summaries[method] = {'n': 1540, 'official_f1': statistics.mean(scores),
                            'by_category': {kind: {'n': len(values), 'f1': statistics.mean(values)}
                                            for kind, values in buckets.items()},
                            'empty_answers': sum(not row['prediction'].strip() for row in ordered)}
    base, improved = rows_by_method['s_parent_single_2000'], rows_by_method[RECIPE['name']]
    comparisons = {}
    for population in ('all1540', 'development7', 'reporting3_previously_exposed'):
        keep = lambda row: (population == 'all1540' or
                            (row['conversation_id'] in REPORTING_CONVS) == population.startswith('reporting3'))
        before, after = [row for row in base if keep(row)], [row for row in improved if keep(row)]
        comparisons[population] = paired_summary(before, after, manifest=[row['question_id'] for row in before],
                                                 dataset='locomo', score_key='official_f1')
    report = {'method': RECIPE['name'], 'recipe': RECIPE, 'dataset_sha256': DATA_HASH,
              'scorer_sha256': SCORER_HASH, 'prediction_hashes': hashes, 'methods': summaries,
              'paired': comparisons, 'transfer_complete': False,
              'comparison_caveat': 'LoCoMo has prior exposure; baseline reader flows/implementation variants differ.'}
    if hashes != {method: hashlib.sha256(path.read_bytes()).hexdigest() for method, path in inputs.items()}:
        raise ValueError('Predictions changed during scoring')
    target = ROOT / 'runs/recursive_v1/OFFICIAL_COMPARISON.json'
    if target.exists() and json.loads(target.read_text()) != report:
        raise ValueError('Refusing to overwrite a different result')
    core.save(target, report)
    lines = ['# Qwen3.5-9B + MiniLM LoCoMo recursive comparison', '',
             'Official category-specific F1,0–100. Same1540 questions, canonical scorer.', '',
             '| Method | All1540 | Multi-hop | Temporal | Open-domain | Single-hop |',
             '|---|---:|---:|---:|---:|---:|']
    for method, summary in summaries.items():
        values = [summary['official_f1']] + [summary['by_category'][str(kind)]['f1'] for kind in (1, 2, 3, 4)]
        lines.append('| ' + method + ' | ' + ' | '.join(f'{value*100:.3f}' for value in values) + ' |')
    lines += ['', 'Gemma LongMemEval transfer remains pending. Prior LoCoMo history exposure is disclosed in PROTOCOL.md.',
              'Recorded external baseline references: E-Mem56.96, LightMem46.60, HiGMem40.33, SimpleMem38.19, Mem036.50, LangMem29.93. Their native reader flows and implementation variants differ.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'methods': summaries, 'transfer_complete': False}))


if __name__ == '__main__':
    main()
