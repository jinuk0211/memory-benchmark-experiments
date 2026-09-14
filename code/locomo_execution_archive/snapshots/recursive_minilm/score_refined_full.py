"""Offline canonical 1540-QA official F1 for the refined-only MiniLM run."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import statistics
from pathlib import Path

DATASET_SHA256 = 'cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993'
SCORER_SHA256 = '8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd'
METHOD = 's_parent_single_2000'


def build_report(dataset, rows, score_fn):
    """Keep original identities/gold and reject partial or substituted results."""
    if len(dataset) != 10 or len({s['sample_id'] for s in dataset}) != 10:
        raise ValueError('Expected the original ten conversations')
    expected = {f'{sample["sample_id"]}:{index}': (sample['sample_id'], index, qa)
                for sample in dataset for index, qa in enumerate(sample['qa'])
                if qa['category'] in (1, 2, 3, 4)}
    if len(expected) != 1540:
        raise ValueError('Expected the original 1540 category1-4 QA')
    found = {}
    for row in rows:
        identity = row['question_id']
        if identity not in expected or identity in found or row['method'] != METHOD:
            raise ValueError('Unexpected, duplicate, or non-refined prediction')
        cid, index, qa = expected[identity]
        prediction = row['prediction']
        if (row['conversation_id'] != cid or row['question'] != qa['question']
                or row['gold'] != str(qa['answer']) or type(row['category']) is not int
                or row['category'] != qa['category'] or not isinstance(prediction, str)
                or row['hypothesis'] != prediction):
            raise ValueError('Prediction metadata differs from original QA')
        found[identity] = {**qa, 'sample': cid, 'index': index, 'prediction': prediction}
    if set(found) != set(expected):
        raise ValueError('Incomplete original QA coverage')
    records = [found[identity] for identity in expected]
    scores = [float(value) for value in score_fn(records)]
    if len(scores) != 1540 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError('Invalid official scorer output')
    return {'schema_version': 1, 'method': METHOD, 'predictions_complete': True,
            'qa_count': 1540, 'official_f1': statistics.mean(scores),
            'empty_predictions': sum(not row['prediction'].strip() for row in records),
            'note': 'Unchanged canonical QA metadata and official category-specific F1. '
                    'Native empty or capped answers remain in the denominator. '
                    'This proves prediction coverage/scoring only; usage and closed backup are separate.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'predictions', 'scorer', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('Refusing to overwrite a score report')
    paths = {'dataset': args.dataset, 'predictions': args.predictions, 'scorer': args.scorer}
    contents = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    if hashes['dataset'] != DATASET_SHA256 or hashes['scorer'] != SCORER_SHA256:
        raise ValueError('Canonical dataset or official scorer hash mismatch')
    spec = importlib.util.spec_from_file_location('original_locomo_refined', args.scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def score(records):
        with contextlib.redirect_stdout(io.StringIO()):
            values, _, _ = module.eval_question_answering(records, eval_key='prediction', metric='f1')
        return values

    rows = [json.loads(line) for line in contents['predictions'].decode('utf-8').split('\n') if line.strip()]
    report = build_report(json.loads(contents['dataset']), rows, score)
    if hashes != {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}:
        raise ValueError('Scoring inputs changed during evaluation')
    report['sources_sha256'] = hashes
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'predictions_complete': True, 'qa_count': 1540, 'official_f1': report['official_f1']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
