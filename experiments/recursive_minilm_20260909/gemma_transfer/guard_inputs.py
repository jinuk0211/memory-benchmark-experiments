"""Enforce the precommitted18 complete histories before any Gemma GPU work."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FULL_SHA256 = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
SELECTION_SHA256 = 'd5474e9cca307d6517530c1f04224eb1d76472dd0277529f912cd207c6ae22b8'


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def selection(root: Path = ROOT) -> dict:
    raw = (root / 'TRANSFER_SELECTION.json').read_bytes()
    if sha(raw) != SELECTION_SHA256:
        raise ValueError('Precommitted selection changed')
    return json.loads(raw)


def save_once(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Frozen input changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def prepare(source: Path, root: Path = ROOT) -> dict:
    chosen = selection(root)
    raw = source.read_bytes()
    if sha(raw) != FULL_SHA256:
        raise ValueError('Canonical full500 source changed')
    rows = json.loads(raw)
    by_id = {row['question_id']: row for row in rows}
    if len(rows) != 500 or len(by_id) != 500:
        raise ValueError('Expected500 distinct original questions')
    # Copy complete original records, without cutting sessions or modifying QA.
    subset = [by_id[qid] for qid in chosen['selected_ids']]
    content = json.dumps(subset, ensure_ascii=False, indent=2).encode('utf8')
    lock = {'full_source_sha256': FULL_SHA256, 'selection_sha256': SELECTION_SHA256,
            'subset_sha256': sha(content), 'selected_ids': chosen['selected_ids'],
            'complete_original_records': True}
    save_once(root / 'data/longmemeval18.json', content)
    save_once(root / 'INPUT_LOCK.json', json.dumps(lock, indent=2).encode('utf8'))
    return lock


def input_lock(data: Path, root: Path = ROOT) -> dict:
    chosen = selection(root)
    if data.resolve() != (root / 'data/longmemeval18.json').resolve():
        raise ValueError('Only the frozen18-question path is allowed')
    lock = json.loads((root / 'INPUT_LOCK.json').read_text(encoding='utf8'))
    if (lock['full_source_sha256'] != FULL_SHA256
            or lock['selection_sha256'] != SELECTION_SHA256
            or lock['selected_ids'] != chosen['selected_ids']
            or lock['complete_original_records'] is not True):
        raise ValueError('Input lock differs from precommitment')
    return lock


def validate(data: Path, dataset: str, limit: int | None, root: Path = ROOT) -> dict:
    if dataset != 'longmemeval' or limit is not None:
        raise ValueError('Gemma requires all18 precommitted LongMemEval questions')
    lock = input_lock(data, root)
    raw = data.read_bytes()
    if sha(raw) != lock['subset_sha256']:
        raise ValueError('Frozen complete histories changed')
    if [row['question_id'] for row in json.loads(raw)] != lock['selected_ids']:
        raise ValueError('Frozen question identities changed')
    return lock


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source)))
