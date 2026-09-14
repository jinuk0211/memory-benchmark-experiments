import argparse
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import simplemem_queue as queue


def test_queue_preserves_successes_continues_failure_and_bounds_retries(tmp_path):
    root = tmp_path
    state = root / 'fast_native2_20260911'
    state.mkdir()
    run = root / 'runs/simplemem_native_dialogues_v4'
    run.mkdir(parents=True)
    runner = root / 'official_recovery/simplemem_native_dialogues_v4/runner.py'
    runner.parent.mkdir(parents=True)
    runner.write_text('synthetic native source')
    rows = [{'question_id': f'q{i}', 'question': f'question {i}', 'question_date': '2026-01-01'} for i in range(500)]
    raw = json.dumps(rows).encode()
    (root / 'longmemeval_s_cleaned.json').write_bytes(raw)
    hashes = {'runner.py': 'frozen'}
    protocol = {'source_files_sha256': hashes, 'population_ids': [row['question_id'] for row in rows],
                'runtime': {'embedding_model': '/pinned/minilm', 'api_base': 'http://127.0.0.1:18083/v1'}}
    (run / 'protocol.json').write_text(json.dumps(protocol))
    runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
    receipt = root / 'receipt.json'
    receipt.write_text(json.dumps({'status': 'runtime_verified', 'candidate': str(runner.parent),
        'integrity': {'runtime_files': {'official_recovery/simplemem_native_dialogues_v4/runner.py': runner_hash}}}))
    saved, counts, events = {}, {}, []
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    def native_run(args):
        qid = json.loads(args.ids_file.read_text())[0]
        events.append(qid)
        counts[qid] = counts.get(qid, 0) + 1
        history = run / 'histories' / hashlib.sha256(qid.encode()).hexdigest()[:24]
        attempt = history / f'attempt_{counts[qid]:04d}'
        attempt.mkdir(parents=True)
        if qid == 'q2' or (qid == 'q1' and counts[qid] == 1):
            (attempt / 'failure.json').write_text('{}')
            return 1
        saved[str(history)] = {'question_id': qid, 'status': 'generated', 'hypothesis': 'native answer'}
        return 0
    native = SimpleNamespace(read_json=lambda p: json.loads(p.read_text()), source_hashes=lambda: hashes,
        source_only=lambda row: {'source': row['question_id']}, digest=digest,
        verified=lambda history, identity: saved.get(str(history)), run=native_run)
    loader = SimpleNamespace(exec_module=lambda module: None)
    from io import BytesIO
    def model_response(*args, **kwargs):
        return BytesIO(b'{"data":[{"id":"Qwen/Qwen3.5-9B"}]}')
    fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)
    with patch.object(queue, 'ROOT', root), patch.object(queue, 'RUNNER_SHA', runner_hash), patch.object(queue, 'DATA_SHA', hashlib.sha256(raw).hexdigest()), patch.object(queue, 'live_native_workers', return_value=[]), patch.object(queue, 'urlopen', side_effect=model_response), patch.object(queue.shutil, 'disk_usage', return_value=SimpleNamespace(free=3 * 1024**3)), patch.object(queue.importlib.util, 'spec_from_file_location', return_value=SimpleNamespace(loader=loader)), patch.object(queue.importlib.util, 'module_from_spec', return_value=native), patch.dict('sys.modules', {'fcntl': fake_fcntl}), patch.object(argparse.ArgumentParser, 'parse_args', return_value=SimpleNamespace(runtime_receipt=receipt)):
        assert queue.main() == 1
        assert events.index('q499') < len(events) - 2
        assert counts['q1'] == counts['q2'] == 2
        assert all(count <= 2 for count in counts.values())
        status = json.loads((run / 'status.json').read_text())
        assert status['generated'] == 499 and status['failed'] == 1
        before = events.copy()
        assert queue.main() == 1
        assert events == before
        assert len(json.loads((run / 'predictions.json').read_text())) == 499


def test_complete_requires_exact_nonempty_canonical_predictions(tmp_path):
    run, state = tmp_path / 'run', tmp_path / 'state'
    run.mkdir()
    state.mkdir()
    ids = [f'q{i}' for i in range(500)]
    predictions = {qid: {'question_id': qid, 'status': 'generated', 'hypothesis': 'answer'} for qid in ids}
    assert not queue.publish(run, state, ids, predictions, {}, active='q499')
    assert not queue.publish(run, state, ids, predictions, {'q1': {'error': 'failure'}})
    assert queue.publish(run, state, ids, predictions, {}, phase='finished')
    predictions['q7']['hypothesis'] = ' '
    with pytest.raises(ValueError):
        queue.publish(run, state, ids, predictions, {}, phase='finished')
