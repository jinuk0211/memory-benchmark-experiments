import json
import pytest
from summarize_transfer import attach_official, checked_rows, JUDGE_MODEL


def fixture():
    references = {str(i): {'question': 'q', 'answer': 'a', 'question_type': 'multi-session'} for i in range(18)}
    rows = [{'question_id': qid, 'method': 'method', 'question': 'q', 'gold': 'a',
             'question_type': 'multi-session', 'hypothesis': '', 'prediction': ''} for qid in references]
    return references, rows


def write(path, rows):
    path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf8')


def test_empty_predictions_keep_full_denominator_and_duplicates_fail(tmp_path):
    refs, rows = fixture()
    path = tmp_path / 'pred.jsonl'
    write(path, rows)
    checked = checked_rows(path, refs, 'method')
    assert len(checked) == 18 and sum(r['diagnostic_token_f1'] for r in checked) == 0
    rows[-1] = rows[0]
    write(path, rows)
    with pytest.raises(ValueError, match='exact18'):
        checked_rows(path, refs, 'method')


def test_partial_or_wrong_judge_cannot_be_official_complete(tmp_path):
    _, rows = fixture()
    labels = [{'question_id': r['question_id'], 'hypothesis': '',
               'autoeval_label': {'model': JUDGE_MODEL, 'label': False}} for r in rows]
    path = tmp_path / 'judged.jsonl'
    write(path, labels[:-1])
    assert not attach_official(rows, path)
    write(path, labels)
    assert attach_official(rows, path) and sum(r['official_accuracy'] for r in rows) == 0
    labels[0]['hypothesis'] = 'different'
    write(path, labels)
    with pytest.raises(ValueError, match='mismatch'):
        attach_official(rows, path)
