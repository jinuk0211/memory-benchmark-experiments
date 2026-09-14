"""Server-local full-500 verification, durable backup, then self-instance STOP."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_official
from run_all import check_complete
import score_diagnostic_f1

ROOT = Path('/workspace/longmemeval_s_native7_20260910')
INSTANCE_ID = '50468468'
RUNS = {'simplemem': 'runs/simplemem_native_dialogues_v3',
        'lightmem': 'runs/lightmem_fast_native2'}
STATE = 'queue/fast_native2_finish'
ARCHIVE = 'archives/simplemem_lightmem_full500_native2.tgz'


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def digest_stream(stream, forbidden: tuple[bytes, ...] = ()) -> str:
    digest = hashlib.sha256()
    overlap = max((len(value) for value in forbidden), default=1) - 1
    tail = b''
    while chunk := stream.read(1024 * 1024):
        if any(value in tail + chunk for value in forbidden):
            raise ValueError('CredentialPresentInArchiveInput')
        digest.update(chunk)
        tail = (tail + chunk)[-overlap:] if overlap else b''
    return digest.hexdigest()


def sha(path: Path, forbidden: tuple[bytes, ...] = ()) -> str:
    with path.open('rb') as stream:
        return digest_stream(stream, forbidden)


def owned_processes(root: Path, proc: Path = Path('/proc')) -> list[int]:
    """Read argv only; never read or persist process environments."""
    if not proc.is_dir():
        raise RuntimeError('LinuxProcessInventoryUnavailable')
    found = []
    scripts = {'native_five.py', 'native_lightmem.py', 'lightmem_queue.py',
               'simplemem_queue.py', 'simplemem_recovery.py'}
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = entry.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            runner = any(Path(arg).name in scripts or
                         '/simplemem_native_dialogues_v3/runner.py' in arg for arg in args)
            if not runner:
                continue
            # Other-user services can hide cwd; only inspect it for a possible runner.
            under_root = any(arg.startswith(str(root) + '/') for arg in args)
            if not under_root:
                under_root = entry.joinpath('cwd').resolve(strict=True).is_relative_to(root.resolve())
            if under_root:
                found.append(int(entry.name))
        except FileNotFoundError:
            continue
    return found


def complete_statuses(root: Path) -> None:
    for method, relative in RUNS.items():
        run = root / relative
        check_complete(method, read(run / 'status.json'), 500)
        failure = run / 'failures.json'
        if failure.exists() and read(failure) != []:
            raise ValueError('OutstandingFailures')

@contextmanager
def idle_queues(root: Path):
    """Hold both advisory locks until the final STOP decision is persisted."""
    import fcntl
    with ExitStack() as stack:
        for name in ('simplemem_queue.lock', 'lightmem_queue.lock'):
            lock = stack.enter_context((root / 'fast_native2_20260911' / name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_simplemem(root: Path):
    path = root / 'official_recovery/simplemem_native_dialogues_v3/runner.py'
    spec = importlib.util.spec_from_file_location('finish_guard_simplemem', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_simplemem_sources(root: Path) -> None:
    native = load_simplemem(root)
    run = root / RUNS['simplemem']
    protocol = read(run / 'protocol.json')
    if native.source_hashes() != protocol['source_files_sha256']:
        raise ValueError('SimplememSourceCodeChanged')
    predictions = {row['question_id']: row for row in read(run / 'predictions.json')}
    for item in read(root / 'longmemeval_s_cleaned.json'):
        qid = item['question_id']
        identity = {'protocol_sha256': native.digest(protocol),
                    'source_sha256': native.digest(native.source_only(item)),
                    'query_sha256': native.digest({key: item[key] for key in
                                                  ('question_id', 'question', 'question_date')})}
        history = run / 'histories' / hashlib.sha256(qid.encode()).hexdigest()[:24]
        verified = native.verified(history, identity)
        if verified is None or verified != predictions[qid]:
            raise ValueError('SimplememSealedPredictionMismatch')
        receipts = [path for path in history.glob('attempt_*/completion.json')
                    if read(path).get('identity') == identity]
        if not receipts or (receipts[0].parent / 'failure.json').exists():
            raise ValueError('SimplememCompletedAttemptFailed')


def validate_lightmem_sources(root: Path) -> list[Path]:
    run = root / RUNS['lightmem']
    receipt = read(run / 'aggregation_receipt.json')
    sources = receipt['sources']
    ids = [row['question_id'] for row in read(root / 'longmemeval_s_cleaned.json')]
    if set(sources) != set(ids):
        raise ValueError('LightmemAggregationCoverageMismatch')
    from lightmem_queue import validate_prediction
    items = {row['question_id']: row for row in read(root / 'longmemeval_s_cleaned.json')}
    protocol = read(run / 'protocol.json')
    directories = []
    for qid, source in sources.items():
        directory = Path(source['source_dir'])
        allowed = (root / 'fast_native2_20260911/lightmem_lanes', root / 'runs/lightmem/e47becba')
        if not any(directory.resolve().is_relative_to(base.resolve()) for base in allowed):
            raise ValueError('LightmemSourceOutsideOwnedRoots')
        if (sha(directory / 'prediction.json') != source['prediction_sha256']
                or sha(run / qid / 'prediction.json') != source['prediction_sha256']):
            raise ValueError('LightmemAggregationPredictionMismatch')
        validate_prediction(directory, items[qid], protocol)
        prediction = read(directory / 'prediction.json')
        attempt = directory / prediction['attempt']
        if (not attempt.resolve().is_relative_to(directory.resolve())
                or not (attempt / 'source.json').is_file()
                or not (attempt / 'construction.json').is_file()
                or not (attempt / 'qdrant').is_dir()
                or not any(path.is_file() and not excluded(path.relative_to(attempt))
                           for path in (attempt / 'qdrant').rglob('*'))
                or (attempt / 'failure.json').exists()):
            raise ValueError('LightmemSourceEvidenceMissing')
        directories.append(directory)
    return directories


def export_and_score(root: Path) -> None:
    """Revalidate with the original exporter even when exports already exist."""
    destination = root / STATE / 'exports'
    destination.mkdir(parents=True, exist_ok=True)
    for method, relative in RUNS.items():
        protocol = read(root / relative / 'protocol.json')
        model = protocol.get('model', protocol.get('runtime', {}).get('model'))
        if model != 'Qwen/Qwen3.5-9B':
            raise ValueError('BackboneProtocolMismatch')
        output = destination / (method + '.full500.jsonl')
        receipt = output.with_name(output.name + '.receipt.json')
        with tempfile.TemporaryDirectory(prefix='validate-', dir=destination.parent) as temporary:
            fresh = Path(temporary) / output.name
            export_official.export(argparse.Namespace(
                method=method, run_dir=root / relative,
                dataset=root / 'longmemeval_s_cleaned.json',
                vendor=root / 'official_longmemeval', output=fresh))
            for source, target in ((fresh, output),
                                   (fresh.with_name(fresh.name + '.receipt.json'), receipt)):
                if target.exists():
                    if target.read_bytes() != source.read_bytes():
                        raise ValueError('ExistingExportDoesNotMatchVerifiedInputs')
                else:
                    source.replace(target)
            fresh_score = Path(temporary) / (method + '.diagnostic_f1.json')
            score_diagnostic_f1.score(root / 'longmemeval_s_cleaned.json', output, fresh_score)
            score = destination / fresh_score.name
            if score.exists():
                if read(score) != read(fresh_score):
                    raise ValueError('ExistingScoreDoesNotMatchVerifiedExport')
            else:
                fresh_score.replace(score)
    validate_simplemem_sources(root)
    validate_lightmem_sources(root)


def excluded(path: Path) -> bool:
    names = [part.lower() for part in path.parts]
    return (any(name in {'.git', '__pycache__', '.venv', '.ssh', '.env', 'environ',
                         'environment.json', 'credentials.json', 'credentials', 'secrets.json',
                         'id_rsa', 'id_ed25519'} or name.startswith('.env.') for name in names)
            or path.name.lower() in {'lock', '.lock'}
            or path.suffix.lower() in {'.lock', '.tmp', '.pyc', '.pem', '.key'})


def archive_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}

    def visit(path: Path, ancestors: frozenset[Path] = frozenset(), source_only=False):
        if excluded(path.relative_to(root)):
            return
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError('ArchiveSymlinkEscapesRoot')
        if resolved in ancestors:
            raise ValueError('ArchiveSymlinkCycle')
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child, ancestors | {resolved}, source_only)
        elif path.is_file():
            if not source_only or path.suffix.lower() in {'.py', '.json', '.yaml', '.yml', '.toml', '.md', '.txt', '.sh', '.conf', '.example'}:
                files[path.relative_to(root).as_posix()] = path
        else:
            raise ValueError('UnsupportedArchiveInput')

    for relative in (*RUNS.values(), STATE + '/exports', 'longmemeval_s_cleaned.json',
                     'native_five.py', 'native_lightmem.py', 'run_all.py',
                     'export_official.py', 'score_diagnostic_f1.py',
                     'source/MemoryData/utils/request_metering.py'):
        visit(root / relative)
    for relative in ('official_recovery/simplemem_native_dialogues_v3', 'source/LightMem'):
        visit(root / relative, source_only=True)
    # Queue contains complete lane memories, copied predictions, and launch evidence.
    visit(root / 'fast_native2_20260911')
    for directory in validate_lightmem_sources(root):
        visit(directory)
    for relative in ('source/source_snapshot.json', 'model_integrity.json'):
        if (root / relative).exists():
            visit(root / relative)
    for name in export_official.UPSTREAM_HASHES:
        visit(root / 'official_longmemeval' / name)
    return files


def verify_archive(path: Path, expected: dict) -> None:
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if (len(names) != len(set(names)) or set(names) != set(expected) | {'archive_manifest.json'}
                or any(not member.isfile() or PurePosixPath(member.name).is_absolute()
                       or '..' in PurePosixPath(member.name).parts for member in members)):
            raise ValueError('ArchiveMemberMismatch')
        if json.load(archive.extractfile('archive_manifest.json'))['files'] != expected:
            raise ValueError('ArchiveManifestMismatch')
        for name, item in expected.items():
            member = archive.getmember(name)
            with archive.extractfile(member) as stream:
                if member.size != item['bytes'] or digest_stream(stream) != item['sha256']:
                    raise ValueError('ArchiveFileHashMismatch')


def create_archive(root: Path) -> dict:
    files = archive_files(root)
    forbidden = tuple(value.encode() for key, value in os.environ.items()
                      if len(value) >= 8 and any(word in key.upper() for word in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET')))
    manifest = {name: {'bytes': path.stat().st_size, 'sha256': sha(path, forbidden)}
                for name, path in sorted(files.items())}
    target = root / ARCHIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        required = int(sum(item['bytes'] for item in manifest.values()) * 1.02) + 512 * 1024 ** 2
        if shutil.disk_usage(target.parent).free < required:
            raise OSError('InsufficientDiskForVerifiedArchive')
        partial = target.with_name(target.name + '.partial')
        with partial.open('wb') as output:
            with tarfile.open(fileobj=output, mode='w:gz', compresslevel=1, dereference=True) as archive:
                for name, path in sorted(files.items()):
                    archive.add(path, arcname=name, recursive=False)
                raw = json.dumps({'schema': 'native2-complete-backup-v1', 'files': manifest}, sort_keys=True).encode()
                info = tarfile.TarInfo('archive_manifest.json')
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
            output.flush()
            os.fsync(output.fileno())
        verify_archive(partial, manifest)
        partial.replace(target)
    verify_archive(target, manifest)
    if any(path.stat().st_size != manifest[name]['bytes'] or sha(path) != manifest[name]['sha256']
           for name, path in files.items()):
        raise ValueError('ArchiveInputsChanged')
    receipt = {'archive': str(target), 'archive_sha256': sha(target), 'files': manifest,
               'verified': True, 'officialjudge_pending': True}
    receipt_path = target.with_name(target.name + '.receipt.json')
    if receipt_path.exists() and read(receipt_path) != receipt:
        raise ValueError('ArchiveReceiptMismatch')
    save(receipt_path, receipt)
    return receipt


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('RedirectRefused')


class Vast:
    def __init__(self, expected_id: str, environ=None, opener=None):
        environment = os.environ if environ is None else environ
        self.identifier = environment.get('CONTAINER_ID', '')
        self.key = environment.get('CONTAINER_API_KEY', '')
        if expected_id != INSTANCE_ID or self.identifier != expected_id or not self.key:
            raise ValueError('MissingCredentialOrSelfInstanceMismatch')
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, stop=False) -> dict:
        url = 'https://console.vast.ai/api/v0/instances/' + self.identifier + '/'
        request = urllib.request.Request(url if stop else url + '?owner=me',
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'},
            data=b'{"state":"stopped"}' if stop else None, method='PUT' if stop else 'GET')
        with self.opener.open(request, timeout=20) as response:
            if response.status != 200:
                raise RuntimeError('UnexpectedVastHTTPStatus')
            result = json.load(response)
        if stop:
            if result.get('success') is not True:
                raise RuntimeError('StopNotExplicitlyAccepted')
        else:
            row = result.get('instances')
            if not isinstance(row, dict) or str(row.get('id')) != self.identifier:
                raise ValueError('SelfInstanceMismatch')
            if row.get('actual_status') != 'running' or row.get('cur_state') != 'running':
                raise ValueError('SelfInstanceNotRunning')
        return {'ok': True, 'event': 'stop_accepted' if stop else 'check_passed'}


def stop_once(root: Path, api: Vast) -> str:
    path = root / STATE / 'stop_state.json'
    if path.exists():
        return 'stop_accepted' if read(path).get('accepted') is True else 'needs_attention'
    for attempt in range(1, 4):
        api.request()
        save(path, {'accepted': False, 'attempt': attempt, 'instance_id': INSTANCE_ID})
        try:
            api.request(stop=True)
            save(path, {'accepted': True, 'attempt': attempt, 'instance_id': INSTANCE_ID})
            return 'stop_accepted'
        except urllib.error.HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise
        except (TimeoutError, ConnectionError, urllib.error.URLError):
            pass
        if attempt < 3:
            time.sleep(5)
    return 'needs_attention'


def tick(root: Path, api: Vast, activate: bool) -> dict:
    stopped = root / STATE / 'stop_state.json'
    if stopped.exists():
        return {'status': 'stop_accepted' if read(stopped).get('accepted') is True else 'needs_attention',
                'reason': 'persisted_stop_request'}
    active = owned_processes(root)
    if active:
        return {'status': 'waiting', 'active_owned_pids': active}
    try:
        complete_statuses(root)
    except (FileNotFoundError, RuntimeError):
        return {'status': 'waiting', 'reason': 'both_full500_required'}
    with idle_queues(root):
        if owned_processes(root):
            return {'status': 'waiting', 'reason': 'owned_runner_started'}
        export_and_score(root)
        archive = create_archive(root)
        complete_statuses(root)
        if owned_processes(root):
            raise RuntimeError('OwnedRunnerRestartedDuringBackup')
        result = {'status': 'verified_ready', 'archive': archive['archive'],
                  'archive_sha256': archive['archive_sha256'], 'generated': 1000,
                  'officialjudge_pending': True}
        save(root / STATE / 'verified_ready.json', result)
        if activate:
            result['status'] = stop_once(root, api)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-instance-id', required=True, choices=[INSTANCE_ID])
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    try:
        api = Vast(args.expected_instance_id)
        if args.check_only:
            print(json.dumps(api.request()), flush=True)
            return 0
        import fcntl
        directory = ROOT / STATE
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'guard.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                try:
                    result = tick(ROOT, api, args.activate)
                except Exception as error:
                    result = {'status': 'needs_attention', 'error_type': type(error).__name__}
                result['at_unix'] = time.time()
                save(directory / 'status.json', result)
                print(json.dumps(result), flush=True)
                if result['status'] == 'stop_accepted':
                    return 0
                time.sleep(60)
    except Exception as error:
        print(json.dumps({'status': 'needs_attention', 'error_type': type(error).__name__}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())



