"""Initialize a fresh canonical run without importing any past result bytes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

ROOT = Path('/workspace/longmemeval_s_native7_20260910')
STATE = Path('/workspace/quad_3090_20260911')
INSTANCE = '50577514'
DATA_SHA = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
CODE_SHA = 'a03f4b8d8c3b510712c3e7568fe5d016fc16e87ed208a94eed2644adecf02b11'
SIMPLE_PROTOCOL = '099254c45e1077d44c851907f5f37bd1e0a36e7127e9eef87e406181d479977b'
LIGHT_PROTOCOL = '0ce3945e48896edd5b7cf8d5f0c09e79c0047951a0208472e50740e2d1f3b326'
ARCHIVES = {
    'sources_data.tgz': ('c330cbddaf19019424a831fef5f00eaaeb23239880e9bd4e9519f42bb466903d', {
        'longmemeval_s_cleaned.json': (277383467, DATA_SHA)}),
    'v4_live_probe_backup_20260911.tgz': ('0165034ab11708d111d9fc84250fa975e5da7bbf64eee8f7dd47293d294c057b', {
        'runs/simplemem_native_dialogues_v4/protocol.json': (13192, '041977dd93d1daa171de421a254614cd3755451266aa6356b4ad618bf6b52727'),
        'fast_native2_20260911/simplemem_v4_runtime.json': (515, '4dcaf91e9fd84668a02d725e489036837c5f80c23fe840d63a16c92bbed393a3')}),
    'smoke_lightmem_verified_20260910T103053Z.tgz': ('1f56ededfeebfd144bc80ce4b1755e66a8f9301c8918bb82184335001bfbb05e', {
        'runs/lightmem/protocol.json': (772, LIGHT_PROTOCOL)}),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def authorization(state):
    path = state / 'FRESH_RUN.json'
    value = read(path)
    if (value.get('status') != 'fresh_run_authorized' or value.get('instance_id') != INSTANCE
            or value.get('dataset_sha256') != DATA_SHA
            or value.get('methods') != {'simplemem': 500, 'lightmem': 500}
            or value.get('past_results_reused') is not False
            or value.get('fresh_generation') is not True
            or value.get('canonical_population') != 500
            or value.get('automatic_instance_stop') is not False
            or value.get('run_id') != 'quad_3090_50577514_fresh_20260911'):
        raise ValueError('Fresh run authorization differs')
    return value, sha(path)


def ensure_empty_results(root, state):
    runs = root / 'runs'
    allowed = {'simplemem_native_dialogues_v4/protocol.json', 'lightmem/protocol.json'}
    if runs.exists():
        for path in runs.rglob('*'):
            if path.is_symlink() or (path.is_file() and path.relative_to(runs).as_posix() not in allowed):
                raise ValueError('Past or unowned run artifacts must not enter the fresh run')
    fast = root / 'fast_native2_20260911'
    for path in (runs / 'simplemem_native_dialogues_v4/histories', fast / 'lightmem_lanes',
                 runs / 'lightmem_fast_native2'):
        if path.exists() and any(path.iterdir()):
            raise ValueError('Existing history directories block fresh initialization')
    for path in (fast / 'lightmem_attempts.json', fast / 'lightmem_status.json',
                 fast / 'simplemem_status.json', state / 'READY.json'):
        if path.exists():
            raise ValueError('Existing queue state blocks fresh initialization')
    if any(state.glob('routes_*.json')):
        raise ValueError('Existing dispatch state blocks fresh initialization')


def expected_members():
    return {name: expected for _, entries in ARCHIVES.values() for name, expected in entries.items()}


def verify_members(root):
    proof = {}
    for name, (size, wanted) in expected_members().items():
        path = root / name
        if path.stat().st_size != size or sha(path) != wanted:
            raise ValueError('Fresh input file differs: ' + name)
        proof[name] = wanted
    return proof


def extract_selected(archive_path, expected_archive, entries, root):
    if sha(archive_path) != expected_archive:
        raise ValueError('Input archive hash differs: ' + archive_path.name)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for name, (size, wanted) in entries.items():
            selected = [member for member in members if member.name == name]
            if len(selected) != 1 or not selected[0].isfile() or selected[0].size != size:
                raise ValueError('Expected one regular pinned archive member: ' + name)
            target = root / name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Selected input destination escapes experiment root')
            if target.exists():
                if target.stat().st_size != size or sha(target) != wanted:
                    raise ValueError('Existing input differs; preserving it: ' + name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.fresh.tmp')
            with archive.extractfile(selected[0]) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, 8 * 1024**2)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != size or sha(temporary) != wanted:
                raise ValueError('Extracted input hash differs: ' + name)
            temporary.replace(target)


def prepare(root=ROOT, state=STATE):
    auth, auth_sha = authorization(state)
    initialized = state / 'fresh_initialized.json'
    if initialized.exists():
        previous = read(initialized)
        if (previous.get('status') != 'fresh_initialized' or previous.get('instance_id') != INSTANCE
                or previous.get('fresh_run_sha256') != auth_sha
                or previous.get('initial_counts') != {'simplemem': 0, 'lightmem': 0}
                or previous.get('past_results_reused') is not False
                or previous.get('input_files_sha256') != {name: value[1] for name, value in expected_members().items()}):
            raise ValueError('Fresh initialization identity differs')
        verify_members(root)
        return previous
    ensure_empty_results(root, state)
    for filename, (archive_sha, entries) in ARCHIVES.items():
        extract_selected(state / filename, archive_sha, entries, root)
    proof = verify_members(root)
    # Recheck before committing initialization; no worker should run before READY.
    ensure_empty_results(root, state)
    if authorization(state)[1] != auth_sha:
        raise ValueError('Fresh authorization changed during initialization')
    result = {'status': 'fresh_initialized', 'instance_id': INSTANCE, 'run_id': auth['run_id'],
              'fresh_run_sha256': auth_sha, 'initialized_at': time.time(),
              'initial_counts': {'simplemem': 0, 'lightmem': 0}, 'past_results_reused': False,
              'input_files_sha256': proof,
              'archives_sha256': {name: value[0] for name, value in ARCHIVES.items()},
              'old_runtime_receipt_use': 'Immutable protocol lineage only; current four-GPU runtime proof is required separately'}
    atomic(initialized, result)
    return result


def main():
    if os.environ.get('CONTAINER_ID') != INSTANCE:
        raise ValueError('Wrong fresh-run instance')
    os.environ.pop('CONTAINER_API_KEY', None)
    import fcntl
    with (STATE / 'fresh_initialize.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = prepare()
    print(json.dumps({'status': receipt['status'], 'initial_counts': receipt['initial_counts'],
                      'past_results_reused': False}), flush=True)


if __name__ == '__main__':
    main()
