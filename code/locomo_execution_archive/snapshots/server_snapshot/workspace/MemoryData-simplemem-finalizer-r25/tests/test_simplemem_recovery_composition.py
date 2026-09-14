"""Old-first QA preservation and whole-attempt accounting, without scoring."""
import copy
import importlib
from collections import Counter

import pytest


@pytest.fixture
def data():
    expected = {(f'conv-{n // 154}', n % 154): {
        'question': f'Question {n}?', 'answer': f'Answer {n}',
        'category': n % 4 + 1, 'evidence': [f'D{n}:1'],
    } for n in range(1540)}
    rows = [{'sample': sample, 'index': index, **qa,
             'prediction': f' old wrong {n} \n', 'seconds': n / 10}
            for n, ((sample, index), qa) in enumerate(expected.items())]
    old = copy.deepcopy(rows[:848])
    new = [dict(r, prediction=r['answer']) for r in copy.deepcopy(rows[123:])]
    return expected, old, new


def compose(data, old_run='old-run', new_run='new-run'):
    target = importlib.import_module('scripts.compose_simplemem_recovery')
    expected, old, new = data
    return target.compose_predictions(old, new, expected, old_run, new_run)


def partition(data, usage, provenance=None):
    target = importlib.import_module('scripts.compose_simplemem_recovery')
    result = compose(data)
    return target.partition_usage(usage, result['provenance'] if provenance is None
                                  else provenance, data[0], 'old-run', 'new-run')


def usage(request_id, run='old-run', sample='conv-0', phase='qa', question='0',
          kind='chat_completion', **extra):
    return {'request_id': request_id, 'run_id': run, 'method': 'simplemem',
            'sample_id': sample, 'phase': phase, 'question_id': question,
            'request_kind': kind, 'usage': {'total_tokens': 30}, **extra}


def test_exact_1540_union_selects_old848_new692_before_scoring(data, monkeypatch):
    scorer = importlib.import_module('scripts.score_locomo_comparison')
    monkeypatch.setattr(scorer, 'official_scores', lambda *_: pytest.fail('No scoring'))
    original = copy.deepcopy(data)
    result = compose(data)
    assert len(result['predictions']) == len(result['provenance']) == 1540
    assert Counter(p['source'] for p in result['provenance']) == {'old': 848, 'new': 692}
    assert result['predictions'][:848] == data[1]
    assert result['predictions'][848:] == data[2][725:]
    assert all(result['predictions'][n]['prediction'] != data[2][n - 123]['prediction']
               for n in range(123, 848))  # All 725 improved new duplicates lose.
    assert data == original
    result['predictions'][0]['evidence'].append('mutation')
    assert data == original


def test_selection_follows_dataset_insertion_order_and_preserves_source_indices(data):
    expected, old, new = data
    result = compose((expected, list(reversed(old)), list(reversed(new))))
    assert [(r['sample'], r['index']) for r in result['predictions']] == list(expected)
    assert result['provenance'][0] == {'sample': 'conv-0', 'index': 0, 'source': 'old',
                                     'run_id': 'old-run', 'source_row_index': 847}


@pytest.mark.parametrize('case', ['unknown', 'duplicate_old', 'duplicate_new',
                                 'metadata', 'nonstr_prediction', 'empty_old',
                                 'empty_excluded_new', 'missing_fill', 'nonobject'])
def test_invalid_predictions_fail_closed_including_excluded_candidates(data, case):
    _, old, new = data
    if case == 'unknown':
        new[0]['sample'] = 'foreign'
    elif case == 'duplicate_old':
        old.append(copy.deepcopy(old[0]))
    elif case == 'duplicate_new':
        new.append(copy.deepcopy(new[0]))
    elif case == 'metadata':
        new[0]['answer'] = 'changed answer'
    elif case == 'nonstr_prediction':
        new[0]['prediction'] = None
    elif case == 'empty_old':
        old[0]['prediction'] = ''
    elif case == 'empty_excluded_new':
        new[0]['prediction'] = ' \n '
    elif case == 'missing_fill':
        new.pop()
    else:
        new[0] = None
    with pytest.raises(ValueError):
        compose(data)


@pytest.mark.parametrize('old_run,new_run', [('same', 'same'), ('', 'new'),
                                          ('old', ' '), (None, 'new')])
def test_run_ids_must_be_explicit_nonempty_and_disjoint(data, old_run, new_run):
    with pytest.raises(ValueError):
        compose(data, old_run, new_run)


def test_expected_scope_cannot_be_reduced_below_original1540(data):
    data[0].popitem()
    with pytest.raises(ValueError):
        compose(data)


def test_partition_keeps_chosen_qa_attempts_embeddings_and_both_mixed_constructions(data):
    rows = [
        usage('old-qa'), usage('new-duplicate', 'new-run', question='123'),
        usage('new-fill', 'new-run', 'conv-5', question='100'),
        usage('new-embed', 'new-run', 'conv-5', question=100, kind='embedding'),
        usage('new-repair', 'new-run', 'conv-5', question='100', success=False,
              repair_group_id='group', repair_attempt=0, delivered_to_client=False),
        usage('old-excluded-failure', sample='conv-5', question='100', success=False),
        usage('old-only-construction', phase='initialize', question=None),
        usage('new-unused-construction', 'new-run', phase='initialize', question=None),
        usage('old-mixed-construction', sample='conv-5', phase='memory_add',
              question=None, kind='embedding'),
        usage('new-mixed-construction', 'new-run', 'conv-5', 'memory_finalize', None),
    ]
    original = copy.deepcopy(rows)
    result = partition(data, rows)
    selected = {'old-qa', 'new-fill', 'new-embed', 'new-repair',
                'old-only-construction', 'old-mixed-construction', 'new-mixed-construction'}
    assert result['selected'] == [r for r in rows if r['request_id'] in selected]
    assert result['excluded'] == [r for r in rows if r['request_id'] not in selected]
    assert len(result['selected']) + len(result['excluded']) == len(rows)
    assert sum(r['usage']['total_tokens'] for r in rows) == 300
    assert rows == original
    result['selected'][0]['usage']['total_tokens'] = 999
    result['excluded'][0]['usage']['total_tokens'] = 999
    assert rows == original


@pytest.mark.parametrize('changes', [
    {'method': 'mem0'}, {'run_id': 'foreign'}, {'sample_id': 'foreign'},
    {'phase': 'unknown'}, {'request_kind': 'completion'}, {'question_id': 'missing'},
    {'question_id': True}, {'question_id': 0.0}, {'question_id': '1.0'},
    {'question_id': '00'}, {'question_id': '154'}, {'question_id': None},
    {'request_id': ''}, {'request_id': ' '}, {'request_id': None},
])
def test_even_excluded_usage_requires_valid_attribution(data, changes):
    excluded = usage('excluded', 'new-run', question='123')
    excluded.update(changes)
    with pytest.raises(ValueError):
        partition(data, [usage('valid'), excluded])


@pytest.mark.parametrize('case', ['duplicate_id', 'nondict', 'missing_provenance',
                                 'duplicate_provenance', 'bad_source', 'bad_run'])
def test_usage_or_provenance_ambiguities_fail_closed(data, case):
    rows = [usage('valid')]
    provenance = compose(data)['provenance']
    if case == 'duplicate_id':
        rows.append(usage('valid', 'new-run', question='123'))
    elif case == 'nondict':
        rows.append(None)
    elif case == 'missing_provenance':
        provenance.pop()
    elif case == 'duplicate_provenance':
        provenance.append(copy.deepcopy(provenance[0]))
    elif case == 'bad_source':
        provenance[0]['source'] = 'other'
    else:
        provenance[0]['run_id'] = 'new-run'
    with pytest.raises(ValueError):
        partition(data, rows, provenance)


def test_partition_does_not_fake_usage_or_delivery_validation(data):
    rows = [usage('unknown', usage=None, delivered_to_client=None)]
    result = partition(data, rows)
    assert result['selected'] == rows
    assert 'outer finalizer' in result['note']
    assert 'complete' not in result


def test_explicit_third_run_fills_only_three_missing_keys_without_score_selection(data):
    target = importlib.import_module('scripts.compose_simplemem_recovery')
    expected, old, new = data
    repair = copy.deepcopy(new[-3:])
    recovered = new[:-3]
    before = copy.deepcopy((old, recovered, repair))
    result = target.compose_predictions(old, recovered, expected, 'old-run', 'new-run',
        repair_rows=repair, repair_run_id='repair-run')
    assert Counter(row['source'] for row in result['provenance']) == {'old': 848, 'new': 689, 'repair': 3}
    assert result['predictions'][:848] == old
    assert (old, recovered, repair) == before
    rows = [usage('old-build', sample='conv-9', phase='memory_add'),
            usage('r17-build', 'new-run', 'conv-9', 'memory_add'),
            usage('r21-init', 'repair-run', 'conv-9', 'initialize', None),
            usage('r17-failure', 'new-run', 'conv-9', question='153'),
            usage('r21-qa', 'repair-run', 'conv-9', question='153'),
            usage('r21-embed', 'repair-run', 'conv-9', question='153', kind='embedding')]
    partitioned = target.partition_usage(rows, result['provenance'], expected, 'old-run', 'new-run', repair_run_id='repair-run')
    assert {r['request_id'] for r in partitioned['selected']} == {'r17-build', 'r21-init', 'r21-qa', 'r21-embed'}
    assert len(partitioned['selected']) + len(partitioned['excluded']) == len(rows)


@pytest.mark.parametrize('fault', ['overlap', 'missing', 'extra', 'run_collision', 'half_explicit', 'new_memory', 'foreign_repair_qa'])
def test_third_run_never_remaps_or_silently_discards_foreign_repairs(data, fault):
    target = importlib.import_module('scripts.compose_simplemem_recovery')
    expected, old, new = data
    repair, current, run_id = copy.deepcopy(new[-3:]), new[:-3], 'repair-run'
    if fault == 'overlap':
        repair[0] = copy.deepcopy(old[0])
    elif fault == 'missing':
        repair.pop()
    elif fault == 'extra':
        repair.append(copy.deepcopy(old[0]))
    elif fault == 'run_collision':
        run_id = 'new-run'
    elif fault == 'half_explicit':
        run_id = None
    if fault in ('new_memory', 'foreign_repair_qa'):
        result = target.compose_predictions(old, current, expected, 'old-run', 'new-run', repair_rows=repair, repair_run_id=run_id)
        row = usage('bad', run_id, 'conv-9', phase='memory_add' if fault == 'new_memory' else 'qa', question='0')
        with pytest.raises(ValueError):
            target.partition_usage([row], result['provenance'], expected, 'old-run', 'new-run', repair_run_id=run_id)
    else:
        with pytest.raises(ValueError):
            target.compose_predictions(old, current, expected, 'old-run', 'new-run', repair_rows=repair, repair_run_id=run_id)
