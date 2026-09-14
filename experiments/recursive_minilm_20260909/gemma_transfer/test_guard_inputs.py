"""Population drift and history truncation must fail before model loading."""
import json
from pathlib import Path

import pytest
import guard_inputs as guard


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    receipt = (Path(__file__).parent / 'TRANSFER_SELECTION.json').read_bytes()
    (tmp_path / 'TRANSFER_SELECTION.json').write_bytes(receipt)
    chosen = json.loads(receipt)
    ids = chosen['selected_ids'] + [f'fixture_{i}' for i in range(482)]
    rows = [{'question_id': qid, 'question': 'evaluation question',
             'answer': 'evaluation answer', 'haystack_sessions': [['original', 'complete']]}
            for qid in ids]
    source = tmp_path / 'full500.json'
    source.write_text(json.dumps(rows), encoding='utf8')
    monkeypatch.setattr(guard, 'FULL_SHA256', guard.sha(source.read_bytes()))
    guard.prepare(source, tmp_path)
    return tmp_path, source


def test_complete_original18_and_idempotent_prepare(frozen):
    root, source = frozen
    lock = guard.prepare(source, root)
    data = root / 'data/longmemeval18.json'
    assert guard.validate(data, 'longmemeval', None, root) == lock
    assert json.loads(data.read_text()) == json.loads(source.read_text())[:18]


@pytest.mark.parametrize('dataset,limit', [('locomo', None), ('longmemeval', 1), ('longmemeval', 18)])
def test_reject_other_population_options(frozen, dataset, limit):
    root, _ = frozen
    with pytest.raises(ValueError, match='all18'):
        guard.validate(root / 'data/longmemeval18.json', dataset, limit, root)


def test_reject_history_truncation_even_with_same18ids(frozen):
    root, _ = frozen
    data = root / 'data/longmemeval18.json'
    rows = json.loads(data.read_text())
    rows[0]['haystack_sessions'][0].pop()
    data.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='histories changed'):
        guard.validate(data, 'longmemeval', None, root)


def test_reject_selection_change_and_alternative_path(frozen):
    root, _ = frozen
    alternate = root / 'other.json'
    alternate.write_bytes((root / 'data/longmemeval18.json').read_bytes())
    with pytest.raises(ValueError, match='path'):
        guard.validate(alternate, 'longmemeval', None, root)
    path = root / 'TRANSFER_SELECTION.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='selection changed'):
        guard.input_lock(root / 'data/longmemeval18.json', root)


def test_reject_changed_canonical_source(frozen):
    root, source = frozen
    source.write_bytes(source.read_bytes() + b' ')
    with pytest.raises(ValueError, match='full500 source changed'):
        guard.prepare(source, root)
