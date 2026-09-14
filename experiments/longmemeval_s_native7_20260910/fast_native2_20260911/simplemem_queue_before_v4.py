"""Schedule canonical histories through the unchanged native SimpleMem runner."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.request import urlopen

ROOT = Path('/workspace/longmemeval_s_native7_20260910')
RUNNER_SHA = 'a0ea36852659ee86e37b7fb2779c81da9bba0a6aef9717f62b4e2e5e936d8a98'
DATA_SHA = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
MAX_ATTEMPTS = 2


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def publish(run, state_dir, ids, predictions, failures, active=None, phase='running'):
    if set(predictions) - set(ids) or len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError('Invalid canonical coverage')
    for qid, row in predictions.items():
        if row.get('question_id') != qid or row.get('status') != 'generated' or not isinstance(row.get('hypothesis'), str) or not row['hypothesis'].strip():
            raise ValueError('Invalid native prediction')
    complete = len(predictions) == 500 and not failures and active is None
    status = {'method': 'simplemem', 'planned': 500, 'generated': len(predictions),
              'failed': len(failures), 'population': 500, 'officially_judged': 0,
              'status': 'generation_complete' if complete else 'generation_incomplete' if phase == 'finished' else phase,
              'active_question_id': active, 'updated_at': time.time()}
    atomic(run / 'predictions.json', [predictions[qid] for qid in ids if qid in predictions])
    atomic(run / 'failures.json', [failures[qid] for qid in ids if qid in failures])
    atomic(run / 'status.json', status)
    atomic(state_dir / 'simplemem_status.json', status)
    return complete


def live_native_workers(runner, run):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            argv = (path / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if str(runner) in argv and (str(run) in argv or any(arg.startswith(str(run) + '/') for arg in argv)):
                found.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-receipt', type=Path, required=True)
    args = parser.parse_args()
    root = ROOT.resolve()
    run = root / 'runs/simplemem_native_dialogues_v3'
    state_dir = root / 'fast_native2_20260911'
    runner = root / 'official_recovery/simplemem_native_dialogues_v3/runner.py'
    if hashlib.sha256(runner.read_bytes()).hexdigest() != RUNNER_SHA:
        raise ValueError('Native SimpleMem runner changed')
    receipt = json.loads(args.runtime_receipt.read_text())
    if receipt.get('status') != 'runtime_verified' or receipt.get('candidate') != str(runner.parent):
        raise ValueError('Expected verified v2 runtime')
    if receipt['integrity']['runtime_files']['official_recovery/simplemem_native_dialogues_v3/runner.py'] != RUNNER_SHA:
        raise ValueError('Receipt does not bind native runner')
    state_dir.mkdir(exist_ok=True)
    lock = (state_dir / 'simplemem_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if live_native_workers(runner, run):
        raise RuntimeError('Existing native process must exit before queue starts')
    raw = (root / 'longmemeval_s_cleaned.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA:
        raise ValueError('Canonical dataset changed')
    data = json.loads(raw)
    ids = [row['question_id'] for row in data]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError('Expected canonical 500')
    spec = importlib.util.spec_from_file_location('native_simplemem_v2', runner)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    protocol = native.read_json(run / 'protocol.json')
    if protocol['source_files_sha256'] != native.source_hashes() or protocol['population_ids'] != ids:
        raise ValueError('Native protocol/source mismatch')
    protocol_hash = native.digest(protocol)
    identities, histories, predictions, failures = {}, {}, {}, {}
    for row in data:
        qid = row['question_id']
        identities[qid] = {'protocol_sha256': protocol_hash, 'source_sha256': native.digest(native.source_only(row)),
                           'query_sha256': native.digest({key: row[key] for key in ('question_id', 'question', 'question_date')})}
        histories[qid] = run / 'histories' / hashlib.sha256(qid.encode()).hexdigest()[:24]
        saved = native.verified(histories[qid], identities[qid])
        if saved:
            predictions[qid] = saved
    runtime = protocol['runtime']
    ids_file = state_dir / 'simplemem_active_ids.json'
    dispatch = argparse.Namespace(dataset=root / 'longmemeval_s_cleaned.json', run_dir=run,
        embedding_model=Path(runtime['embedding_model']), api_base=runtime['api_base'], ids_file=ids_file)
    for attempt_pass in range(MAX_ATTEMPTS):
        for qid in ids:
            if qid in predictions:
                continue
            attempts = list(histories[qid].glob('attempt_*'))
            if len(attempts) >= MAX_ATTEMPTS:
                failures[qid] = {'question_id': qid, 'attempts': len(attempts), 'reason': 'Native attempts exhausted; artifacts preserved'}
                continue
            while True:
                if shutil.disk_usage(root).free < 2 * 1024**3:
                    publish(run, state_dir, ids, predictions, failures, phase='waiting_for_disk')
                    time.sleep(30)
                    continue
                try:
                    with urlopen(runtime['api_base'] + '/models', timeout=10) as response:
                        models = json.load(response)
                    if not any(item['id'] == 'Qwen/Qwen3.5-9B' for item in models['data']):
                        raise ValueError('Wrong endpoint model')
                    break
                except (OSError, ValueError, KeyError):
                    publish(run, state_dir, ids, predictions, failures, phase='waiting_for_model')
                    time.sleep(30)
            atomic(ids_file, [qid])
            publish(run, state_dir, ids, predictions, failures, active=qid)
            code = native.run(dispatch)
            saved = native.verified(histories[qid], identities[qid])
            if code == 0 and saved is not None:
                predictions[qid] = saved
                failures.pop(qid, None)
            else:
                failures[qid] = {'question_id': qid, 'attempts': len(list(histories[qid].glob('attempt_*'))), 'exit_code': code}
            publish(run, state_dir, ids, predictions, failures)
    complete = publish(run, state_dir, ids, predictions, failures, phase='finished')
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
