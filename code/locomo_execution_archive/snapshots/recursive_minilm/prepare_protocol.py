"""Freeze the transfer subset before development outcomes are inspected."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 'recursive-minilm-transfer-20260909-v1'
DATA_HASH = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
TYPES = ('single-session-user', 'single-session-assistant', 'single-session-preference',
         'multi-session', 'temporal-reasoning', 'knowledge-update')


def select_ids(metadata: list[dict], excluded: list[str]) -> list[str]:
    """Use IDs/type/abstention only, with disjoint original-question clusters."""
    if len({row['question_id'] for row in metadata}) != len(metadata):
        raise ValueError('Duplicate question IDs')
    used = {qid.removesuffix('_abs') for qid in excluded}
    selected = []
    def order(row: dict) -> tuple[str, str]:
        qid = row['question_id']
        return hashlib.sha256((SEED + '\0' + qid).encode()).hexdigest(), qid
    for kind in TYPES:
        eligible = sorted((row for row in metadata if row['question_type'] == kind
                           and row['question_id'].removesuffix('_abs') not in used), key=order)
        abstentions = [row for row in eligible if row['question_id'].endswith('_abs')]
        chosen = abstentions[:1]
        used.update(row['question_id'].removesuffix('_abs') for row in chosen)
        regular = [row for row in eligible if not row['question_id'].endswith('_abs')
                   and row['question_id'].removesuffix('_abs') not in used]
        chosen.extend(regular[:3 - len(chosen)])
        if len(chosen) != 3:
            raise ValueError('Insufficient distinct clusters: ' + kind)
        selected.extend(row['question_id'] for row in chosen)
        used.update(row['question_id'].removesuffix('_abs') for row in chosen)
    return selected


def main() -> None:
    data = Path('/workspace/generalization_20260908/data/longmemeval_s_cleaned.json')
    raw = data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_HASH:
        raise ValueError('Full LongMemEval source hash differs')
    rows = json.loads(raw)
    if len(rows) != 500:
        raise ValueError('Expected all500 before subset selection')
    excluded = json.loads((ROOT / 'previous_lme12_ids.json').read_text())
    metadata = [{key: row[key] for key in ('question_id', 'question_type')} for row in rows]
    ids = select_ids(metadata, excluded)
    by_id = {row['question_id']: row for row in rows}
    subset = [by_id[qid] for qid in ids]
    receipt = {'seed': SEED, 'full_data_sha256': DATA_HASH, 'selected_ids': ids,
               'selection_fields': ['question_id', 'question_type', 'ID suffix _abs'],
               'per_type': dict(Counter(row['question_type'] for row in subset)),
               'abstention_count': sum(qid.endswith('_abs') for qid in ids),
               'excluded_prior_ids': excluded, 'benchmark_outcomes_used': False,
               'complete_histories': True, 'subset_size': 18}
    for path, value in ((ROOT / 'data/longmemeval18.json', subset),
                        (ROOT / 'TRANSFER_SELECTION.json', receipt)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Refusing a changed transfer selection')
        if not path.exists():
            with path.open('x', encoding='utf8') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
