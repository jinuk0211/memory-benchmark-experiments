"""Report full-role LoCoMo results on populations fixed before target inference."""
import argparse
import hashlib
import json
from pathlib import Path

from transfer_data import paired_summary

METHODS = ('seed', 'r40_fused_four_turn', 's_parent_single_2000')
POPULATIONS = ('generated_population', 'primary_question_population', 'secondary_question_population')


def summarize(run, population_file):
    populations = json.loads(population_file.read_text(encoding='utf-8'))
    if populations['population_selection_used_target_outcomes'] is not False:
        raise ValueError('Population was selected with target outcomes')
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['config']['dataset'] != 'locomo':
        raise ValueError('Expected LoCoMo run')
    data_path = population_file.with_name('locomo3_unchanged_samples.json')
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if protocol['dataset_sha256'] != data_sha256:
        raise ValueError('Run dataset differs from the frozen evaluation input')
    samples = json.loads(data_path.read_text(encoding='utf-8'))
    actual_records = {str(sample['sample_id']) + ':' + str(index): (str(sample['sample_id']), int(qa['category']))
                      for sample in samples for index, qa in enumerate(sample['qa']) if int(qa['category']) in (1, 2, 3, 4)}
    declared_records = {record['id']: (record['conv_id'], int(record['category']))
                        for record in populations['generated_population']}
    if actual_records != declared_records:
        raise ValueError('Declared population differs from dataset questions')
    rows = {method: [json.loads(line) for line in (run / (method + '.jsonl')).read_text(encoding='utf-8').splitlines() if line]
            for method in METHODS}
    generated_ids = [record['id'] for record in populations['generated_population']]
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    if status['phase'] != 'generation_complete' or status['questions'] != len(generated_ids):
        raise ValueError('Run has not completed the declared population')
    for method in METHODS:
        if any(row['method'] != method for row in rows[method]):
            raise ValueError('Method label mismatch')
        if any((row['conversation_id'], int(row['category'])) != declared_records.get(row['question_id']) for row in rows[method]):
            raise ValueError('Row conversation/category differs from frozen manifest')
        paired_summary(rows[method], rows[method], manifest=generated_ids, dataset='locomo', score_key='official_f1')
    results = {}
    for name in POPULATIONS:
        ids = [record['id'] for record in populations[name]]
        if len(ids) != len(set(ids)) or not set(ids) <= set(generated_ids):
            raise ValueError('Invalid declared population')
        selected = {method: [row for row in rows[method] if row['question_id'] in set(ids)] for method in METHODS}
        results[name] = {base: paired_summary(selected[base], selected[METHODS[2]], manifest=ids,
                                             dataset='locomo', score_key='official_f1') for base in METHODS[:2]}
    return {'evaluation_data_sha256': data_sha256, 'model': protocol['config']['model'], 'model_roles': protocol['model_transfer'],
            'population_sha256': hashlib.sha256(population_file.read_bytes()).hexdigest(),
            'protocol_sha256': hashlib.sha256((run / 'protocol.json').read_bytes()).hexdigest(),
            'history_exposure_note': 'All three conversation histories were exposed during accidental 8B development; fresh questions are not unseen histories.',
            'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--populations', type=Path, default=Path(__file__).resolve().parent.parent / 'locomo_transfer/evaluation_populations.json')
    args = parser.parse_args()
    result = summarize(args.run, args.populations)
    destination = args.run / 'predeclared_locomo_results.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
