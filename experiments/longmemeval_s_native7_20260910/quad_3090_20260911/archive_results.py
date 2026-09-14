"""One-shot full-500 validation and archive. No instance-control capability."""
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

sys.path.insert(0, '/workspace/longmemeval_s_native7_20260910')
sys.path.insert(0, '/workspace/longmemeval_s_native7_20260910/fast_native2_20260911')
import export_official
from run_all import check_complete
import score_diagnostic_f1

ROOT = Path('/workspace/longmemeval_s_native7_20260910')
INSTANCE_ID = '50577514'
QUAD = Path('/workspace/quad_3090_20260911')
RUNS = {'simplemem': 'runs/simplemem_native_dialogues_v4',
        'lightmem': 'runs/lightmem_fast_native2'}
STATE = 'queue/quad_3090_finish'
ARCHIVE = 'archives/simplemem_lightmem_full500_quad_3090_50577514.tgz'


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
               'simplemem_queue.py', 'simplemem_recovery.py',
               'simplemem_quad_queue.py', 'lightmem_quad_queue.py', 'quad_simplemem_queue.py', 'quad_lightmem_queue.py'}
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = entry.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            runner = any(Path(arg).name in scripts or
                         '/simplemem_native_dialogues_v4/runner.py' in arg for arg in args)
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
    """Hold both canonical queue locks throughout result and archive validation."""
    import fcntl
    with ExitStack() as stack:
        for name in ('simplemem_queue.lock', 'lightmem_queue.lock'):
            lock = stack.enter_context((root / 'fast_native2_20260911' / name).open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_simplemem(root: Path):
    path = root / 'official_recovery/simplemem_native_dialogues_v4/runner.py'
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
        allowed = (root / 'fast_native2_20260911/lightmem_lanes',)
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
                         'id_rsa', 'id_ed25519', '.hf-token', 'hf-token', 'hf-download.dpapi'} or name.startswith('.env.') for name in names)
            or path.name.lower() in {'lock', '.lock'}
            or path.suffix.lower() in {'.lock', '.tmp', '.pyc', '.pem', '.key', '.dpapi'})


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
    for relative in ('official_recovery/simplemem_native_dialogues_v4', 'source/LightMem'):
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


def create_archive(root: Path, evidence: dict[str, Path]) -> dict:
    files = archive_files(root)
    if files.keys() & evidence.keys():
        raise ValueError('ArchiveEvidenceNameCollision')
    files.update(evidence)
    manifest = {name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
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
                raw = json.dumps({'schema': 'quad-complete-backup-v1', 'files': manifest}, sort_keys=True).encode()
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
               'verified': True, 'officialjudge_pending': True, 'instance_id': INSTANCE_ID}
    receipt_path = target.with_name(target.name + '.receipt.json')
    if receipt_path.exists() and read(receipt_path) != receipt:
        raise ValueError('ArchiveReceiptMismatch')
    save(receipt_path, receipt)
    return receipt


PUBLIC_CODE = {
    'serve_qwen.after_simplemem_tuning.sh',
    'tune_lightmem.py', 'test_tune_lightmem.py', 'test_lightmem_integration.py', 'quad-tune-lightmem.conf',
    'tune_simplemem.py', 'test_tune_simplemem.py', 'serve_qwen.before_tuning.sh', 'quad-tune.conf', 'archive_results.py', 'test_archive_results.py', 'pipeline.py', 'runtime_probe.py', 'prepare_fresh.py', 'verify_fresh.py', 'run_service.sh',
    'runtime_stream_probe.py', 'stream_probe.py', 'probe_replicas.py', 'verify_restored.py', 'download_models.py',
    'install_environments.sh', 'prepare_nltk.sh', 'serve_qwen.sh', 'serve_minilm.sh',
    'serve_meter.sh', 'quad_config.py', 'quad_scheduler.py', 'simplemem_queue.py',
    'lightmem_queue.py', 'simplemem_quad_queue.py', 'lightmem_quad_queue.py', 'quad_simplemem_queue.py', 'quad_lightmem_queue.py',
    'quad_metered_proxy.py', 'metered_proxy.py', 'services.conf', 'queues.conf',
    'run_pipeline.sh', 'run_simplemem.sh', 'run_lightmem.sh', 'run_archive.sh',
    'bootstrap_inference_requirements.txt', 'bootstrap_lightmem_requirements.txt',
    'bootstrap_simplemem_requirements.txt',
}
PUBLIC_RECEIPTS = {
    'lightmem_tuning.json', 'seqs_18101.txt', 'seqs_18111.txt',
    'deployment.json', 'READY.json', 'runtime_verified.json',
    'FRESH_RUN.json', 'fresh_initialized.json', 'fresh_artifacts_verified.json',
    'local_restore_extracted.json', 'model_integrity_expected.json',
    'model_download_receipt.json', 'model_download_verified.json', 'model_files_verified.json',
    'nltk_verified.json', 'environment_verified.json', 'install_verified.json',
    'inference_freeze.txt', 'lightmem_freeze.txt', 'simplemem_freeze.txt', 'nltk_recovery_files.json', 'nltk_recovery_probe.json',
    'bootstrap_.venv-inference.freeze.txt', 'bootstrap_.venv-lightmem.freeze.txt',
    'bootstrap_.venv-simplemem-native0253.freeze.txt',
    'routes_simplemem.json', 'routes_lightmem.json',
    'routes_simplemem.initialized.json', 'routes_lightmem.initialized.json',
    'simplemem_quad_status.json', 'lightmem_quad_status.json',
    'simplemem_tuning.json', 'seqs_18081.txt', 'seqs_18091.txt', 'request_usage.jsonl', 'embedding_usage.jsonl', 'service_restarts.json',
}
PUBLIC_CODE.update({'quad-' + name + '.conf' for name in
                    ('sm0', 'sm1', 'lm0', 'lm1', 'meter', 'minilm', 'pipeline', 'simplemem', 'lightmem', 'install', 'models')})
LANES = {'sm0', 'sm1', 'lm0', 'lm1'}
DEPLOYMENT_SHA = '46f9747b2ab2ab5d780f46485c02c0a37041af36edad0f1c065b9c47fc42e7ac'


def owned_file(directory: Path, relative: str) -> Path:
    path = directory / relative
    if (PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts
            or excluded(Path(relative)) or not path.is_file()
            or not path.resolve().is_relative_to(directory.resolve())):
        raise ValueError('EvidenceFileOutsideOwnedRootOrMissing')
    return path


def validate_fresh_run(quad: Path) -> dict:
    receipt = read(owned_file(quad, 'FRESH_RUN.json'))
    if (receipt.get('status') != 'fresh_run_authorized' or receipt.get('instance_id') != INSTANCE_ID
            or receipt.get('dataset_sha256') != 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
            or receipt.get('methods') != {'simplemem': 500, 'lightmem': 500}
            or receipt.get('past_results_reused') is not False
            or receipt.get('fresh_generation') is not True or receipt.get('canonical_population') != 500
            or not isinstance(receipt.get('run_id'), str) or not receipt['run_id'].strip()):
        raise ValueError('FreshRunAuthorizationRequired')
    initialized = read(owned_file(quad, 'fresh_initialized.json'))
    if (initialized.get('status') != 'fresh_initialized' or initialized.get('instance_id') != INSTANCE_ID
            or initialized.get('fresh_run_sha256') != sha(quad / 'FRESH_RUN.json')
            or initialized.get('initial_counts') != {'simplemem': 0, 'lightmem': 0}
            or initialized.get('past_results_reused') is not False):
        raise ValueError('FreshInitializationProofRequired')
    return receipt


def validate_runtime(quad: Path) -> list[Path]:
    deployment = read(owned_file(quad, 'deployment.json'))
    if sha(quad / 'deployment.json') != DEPLOYMENT_SHA or deployment.get('instance_id') != INSTANCE_ID:
        raise ValueError('DeploymentChanged')
    proof = read(owned_file(quad, 'runtime_verified.json'))
    if (proof.get('status') != 'runtime_verified' or proof.get('instance_id') != INSTANCE_ID
            or proof.get('deployment_sha256') != DEPLOYMENT_SHA
            or proof.get('protocol_proof') != deployment['protocols']
            or not isinstance(proof.get('model_proof'), dict) or not proof['model_proof']
            or proof.get('fresh_run_sha256') != sha(quad / 'FRESH_RUN.json')
            or set(proof.get('lanes', {})) != LANES
            or set(proof.get('replicas', {})) != LANES):
        raise ValueError('FourLaneRuntimeProofRequired')
    ready = read(owned_file(quad, 'READY.json'))
    if any(ready.get(key) != proof[key] for key in
           ('status', 'instance_id', 'deployment_sha256', 'protocol_proof', 'model_proof', 'lanes', 'replicas', 'fresh_run_sha256')):
        raise ValueError('ReadyDiffersFromRuntimeProof')
    scripts = {sha(quad / name) for name in
               ('runtime_probe.py', 'runtime_stream_probe.py', 'stream_probe.py', 'probe_replicas.py', 'pipeline.py')
               if (quad / name).is_file()}
    files = []
    for lane in sorted(LANES):
        binding, entry = deployment['lanes'][lane], proof['lanes'][lane]
        path = owned_file(quad, entry['receipt'])
        lane_proof, replica = read(path), proof['replicas'][lane]
        if (sha(path) != entry['sha256'] or lane_proof.get('status') != 'runtime_probe_verified'
                or lane_proof.get('instance_id') != INSTANCE_ID or lane_proof.get('lane') != lane
                or lane_proof.get('gpu_uuid') != binding['gpu_uuid']
                or lane_proof.get('benchmark_population_member') is not False
                or lane_proof.get('script_sha256') not in scripts
                or replica.get('status') != 'inference_verified'
                or replica.get('model') != deployment['model'] or replica.get('max_model_len') != 65536
                or any(replica.get(key) != binding[key] for key in ('gpu_uuid', 'api_base'))):
            raise ValueError('LaneRuntimeIdentityMismatch')
        stream = lane_proof.get('streamed_chat', {})
        if (stream.get('done_received') is not True or stream.get('finish_reason') != 'stop'
                or stream.get('content_characters', 0) < 1
                or lane_proof.get('qwen_models', {}).get('max_model_len') != 65536):
            raise ValueError('LaneStreamProofMissing')
        if binding['method'] == 'lightmem':
            native = lane_proof.get('native_lightmem', {})
            forwards = native.get('cuda_forward_calls', {})
            devices = lane_proof.get('native_model_devices', {})
            if (native.get('synthetic_fact_recovered') is not True or native.get('stored_points', 0) < 1
                    or forwards.get('compressor', 0) < 1 or forwards.get('internal_minilm', 0) < 2
                    or not str(devices.get('compressor', '')).startswith('cuda')
                    or not str(devices.get('internal_minilm', '')).startswith('cuda')
                    or not lane_proof.get('concurrent_vllm_pids')):
                raise ValueError('NativeLightMemCudaProofMissing')
        files.append(path)
    return files


def quad_evidence(quad: Path) -> dict[str, Path]:
    validate_fresh_run(quad)
    proofs = validate_runtime(quad)
    for method in ('simplemem', 'lightmem'):
        routes = read(owned_file(quad, 'routes_' + method + '.json'))
        if (routes.get('instance_id') != INSTANCE_ID or routes.get('method') != method
                or routes.get('deployment_sha256') != DEPLOYMENT_SHA):
            raise ValueError('RoutingEvidenceIdentityMismatch')
    usage = owned_file(quad, 'request_usage.jsonl')
    if usage.stat().st_size == 0:
        raise ValueError('ActualUsageJournalMissing')
    files = {owned_file(quad, name) for name in PUBLIC_CODE | PUBLIC_RECEIPTS
             if (quad / name).exists()}
    files.update(proofs)
    # Include only named public evidence, never the Q directory recursively.
    for path in proofs:
        for name in ('native.log', 'synthetic_inputs.json'):
            sibling = path.with_name(name)
            if sibling.is_file():
                files.add(owned_file(quad, sibling.relative_to(quad).as_posix()))
    return {'quad_3090_20260911/' + path.relative_to(quad).as_posix(): path
            for path in sorted(files)}


def validate_fresh_results(root: Path, quad: Path) -> None:
    ids = {row['question_id'] for row in read(root / 'longmemeval_s_cleaned.json')}
    config = read(quad / 'deployment.json')
    fresh = validate_fresh_run(quad)
    identity = {'fresh_run_sha256': sha(quad / 'FRESH_RUN.json'),
                'execution_run_id': fresh['run_id'],
                'fresh_initialized_sha256': sha(quad / 'fresh_initialized.json')}
    for method in RUNS:
        routes = read(quad / ('routes_' + method + '.json'))
        if any(routes.get(key) != value for key, value in identity.items()):
            raise ValueError('FreshRoutingIdentityMismatch')
        protocol_path = root / RUNS[method] / 'protocol.json'
        actual_protocol = (load_simplemem(root).digest(read(protocol_path))
                           if method == 'simplemem' else sha(protocol_path))
        if (actual_protocol != config['protocols'][method]
                or routes.get('protocol_sha256') != actual_protocol):
            raise ValueError('FrozenProtocolChanged')
        if set(routes.get('histories', {})) != ids or set(routes.get('dispatches', {})) != ids:
            raise ValueError('FreshRunMustDispatchEveryCanonicalHistory')
        for qid, route in routes['histories'].items():
            binding = config['lanes'].get(route.get('lane'), {})
            if (binding.get('method') != method or route.get('instance_id') != INSTANCE_ID
                    or any(route.get(key) != binding.get(key) for key in ('gpu_uuid', 'gpu_index', 'api_base'))):
                raise ValueError('FreshRouteGpuBindingMismatch')
            if method == 'simplemem':
                history = root / RUNS[method] / 'histories' / hashlib.sha256(qid.encode()).hexdigest()[:24]
                paths = [p.parent / 'quad_route.json' for p in history.glob('attempt_*/completion.json')
                         if not (p.parent / 'failure.json').exists()]
            else:
                source = read(root / RUNS[method] / 'aggregation_receipt.json')['sources'][qid]
                directory = Path(source['source_dir'])
                prediction = read(directory / 'prediction.json')
                paths = [directory / prediction['attempt'] / 'quad_route.json']
            if not paths:
                raise ValueError('FreshAttemptDispatchEvidenceMissing')
            for path in paths:
                receipt = read(path)
                if (receipt.get('deployment_sha256') != DEPLOYMENT_SHA
                        or receipt.get('question_id') != qid or receipt.get('method') != method
                        or receipt.get('route') != route
                        or receipt.get('dispatch') not in routes['dispatches'][qid]
                        or any(receipt.get(key) != value for key, value in identity.items())):
                    raise ValueError('FreshAttemptRouteMismatch')


def validate_usage(root: Path, quad: Path) -> dict:
    config = read(quad / 'deployment.json')
    fresh = validate_fresh_run(quad)
    identity = {'fresh_run_sha256': sha(quad / 'FRESH_RUN.json'),
                'execution_run_id': fresh['run_id'],
                'fresh_initialized_sha256': sha(quad / 'fresh_initialized.json')}
    ids = {row['question_id'] for row in read(root / 'longmemeval_s_cleaned.json')}
    routes = {method: read(quad / ('routes_' + method + '.json'))['histories'] for method in RUNS}
    covered = {method: set() for method in RUNS}
    observed_lanes, probes, records = set(), {}, 0
    with owned_file(quad, 'request_usage.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            records += 1
            if row.get('success') is not True:
                continue
            lane = row.get('inference_lane')
            binding = config['lanes'].get(lane, {})
            if (lane not in LANES or row.get('inference_instance_id') != INSTANCE_ID
                    or row.get('inference_gpu_uuid') != binding.get('gpu_uuid')
                    or row.get('deployment_sha256') != DEPLOYMENT_SHA
                    or any(row.get(key) != value for key, value in identity.items())):
                raise ValueError('SuccessfulUsageRoutingIdentityMismatch')
            observed_lanes.add(lane)
            if row.get('canonical_benchmark') is True:
                method, qid = row.get('method'), row.get('sample_id')
                if (method not in RUNS or qid not in ids or routes[method][qid].get('lane') != lane
                        or row.get('question_id') not in (None, '', qid)):
                    raise ValueError('CanonicalUsageHistoryMismatch')
                covered[method].add(qid)
            elif row.get('canonical_benchmark') is False and row.get('method') == 'runtime_probe':
                probes.setdefault(row.get('run_id'), []).append(row)
            else:
                raise ValueError('SuccessfulUsageScopeMissing')
    if observed_lanes != LANES or any(done != ids for done in covered.values()):
        raise ValueError('FourGpuFullPopulationUsageEvidenceMissing')
    gate = read(quad / 'runtime_verified.json')
    for lane, entry in gate['lanes'].items():
        proof = read(quad / entry['receipt'])
        route = proof.get('central_route', {})
        count = route.get('successful_requests', 0)
        matched = [row for row in probes.get(proof.get('probe_id'), []) if row['inference_lane'] == lane]
        # A final response can reach its client before the meter fsyncs its row.
        # The proof binds the observed prefix; retain all later physical rows too.
        prefix = matched[:count] if isinstance(count, int) and count > 0 else []
        digest = hashlib.sha256(json.dumps(prefix, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if (not prefix or len(prefix) != count or digest != route.get('records_sha256')
                or route.get('inference_lane') != lane
                or route.get('inference_gpu_uuid') != config['lanes'][lane]['gpu_uuid']):
            raise ValueError('SyntheticRouteProofDoesNotMatchUsageJournal')
    return {'journal_records': records, 'gpu_lanes': sorted(observed_lanes),
            'canonical_histories_with_success': {method: len(done) for method, done in covered.items()}}


def finish(root: Path, quad: Path) -> dict:
    if owned_processes(root):
        raise RuntimeError('OwnedWorkersStillActive')
    with idle_queues(root):
        if owned_processes(root):
            raise RuntimeError('OwnedWorkerStarted')
        complete_statuses(root)
        evidence = quad_evidence(quad)
        validate_fresh_results(root, quad)
        usage = validate_usage(root, quad)
        export_and_score(root)
        # Preserve all physical calls, including synthetic probes and failed attempts.
        # Missing historical journals are not treated as zero cost.
        for name in ('chat_usage.jsonl', 'embedding_usage.jsonl'):
            path = root / 'logs' / name
            if path.is_file():
                evidence['historical_usage/' + name] = path
        archive = create_archive(root, evidence)
        complete_statuses(root)
        if owned_processes(root):
            raise RuntimeError('OwnedWorkerRestartedDuringArchive')
        result = {'status': 'archive_verified', 'verified': True, 'instance_id': INSTANCE_ID,
                  'fresh_run': True, 'past_results_reused': False,
                  'archive': archive['archive'], 'archive_sha256': archive['archive_sha256'],
                  'generated': {'simplemem': 500, 'lightmem': 500},
                  'fresh_run_sha256': sha(quad / 'FRESH_RUN.json'),
                  'runtime_receipt_sha256': sha(quad / 'runtime_verified.json'), 'usage_proof': usage,
                  'cost_scope': 'This fresh run only; raw usage includes retries and synthetic probes, previous experiments are not merged',
                  'officialjudge_pending': True, 'instance_control_performed': False}
        save(quad / 'archive_verified.json', result)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-instance-id', choices=[INSTANCE_ID], default=INSTANCE_ID)
    parser.parse_args()
    try:
        if os.environ.get('CONTAINER_ID') != INSTANCE_ID:
            raise ValueError('WrongArchiveInstance')
        import fcntl
        QUAD.mkdir(parents=True, exist_ok=True)
        with (QUAD / 'archive.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = QUAD / 'archive_verified.json'
            if previous.exists():
                previous.replace(QUAD / ('archive_verified.previous.' + str(time.time_ns()) + '.json'))
            result = finish(ROOT, QUAD)
        print(json.dumps({key: result[key] for key in
                          ('status', 'archive', 'archive_sha256', 'generated', 'instance_control_performed')}), flush=True)
        return 0
    except Exception as error:
        result = {'status': 'needs_attention', 'error_type': type(error).__name__,
                  'instance_id': INSTANCE_ID, 'at_unix': time.time(), 'instance_control_performed': False}
        try:
            save(QUAD / 'archive_status.json', result)
        except OSError:
            result['status_persistence'] = 'unavailable'
        try:
            print(json.dumps(result), flush=True)
        except OSError:
            pass
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
