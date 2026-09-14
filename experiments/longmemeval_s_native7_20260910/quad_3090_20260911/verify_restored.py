"""Verify restored histories and pinned models without starting experiments."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path('/workspace/longmemeval_s_native7_20260910')
STATE = Path('/workspace/quad_3090_20260911')
SIMPLE_PROTOCOL = '099254c45e1077d44c851907f5f37bd1e0a36e7127e9eef87e406181d479977b'
LIGHT_PROTOCOL = '0ce3945e48896edd5b7cf8d5f0c09e79c0047951a0208472e50740e2d1f3b326'
RUNNER_SHA = '733fb9a3f9ae33d0c62ca0099ee2e4e7f1630e6906bd216018040ab654eff582'
DATA_SHA = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-backups', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CONTAINER_ID') != '50577514':
        raise ValueError('WrongRecoveryInstance')
    dataset = ROOT/'longmemeval_s_cleaned.json'
    if sha(dataset) != DATA_SHA:
        raise ValueError('DatasetChanged')
    rows = read(dataset)
    ids = [row['question_id'] for row in rows]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError('CanonicalCoverageChanged')
    runner = ROOT/'official_recovery/simplemem_native_dialogues_v4/runner.py'
    if sha(runner) != RUNNER_SHA:
        raise ValueError('SimpleMemRunnerChanged')
    native = load('restored_simplemem', runner)
    run = ROOT/'runs/simplemem_native_dialogues_v4'
    protocol = read(run/'protocol.json')
    if native.digest(protocol) != SIMPLE_PROTOCOL or protocol['source_files_sha256'] != native.source_hashes() or protocol['population_ids'] != ids:
        raise ValueError('SimpleMemProtocolChanged')
    simple_ids = []
    for row in rows:
        qid = row['question_id']
        identity = {'protocol_sha256':SIMPLE_PROTOCOL,
                    'source_sha256':native.digest(native.source_only(row)),
                    'query_sha256':native.digest({key:row[key] for key in ('question_id','question','question_date')})}
        history = run/'histories'/hashlib.sha256(qid.encode()).hexdigest()[:24]
        if native.verified(history, identity):
            receipts = [p for p in history.glob('attempt_*/completion.json')
                        if read(p).get('identity') == identity]
            if not receipts or (receipts[0].parent/'failure.json').exists():
                raise ValueError('SimpleMemCompletedAttemptFailed:'+qid)
            simple_ids.append(qid)
    fast = ROOT/'fast_native2_20260911'
    light = load('restored_lightmem_queue', fast/'lightmem_queue.py')
    light_args = argparse.Namespace(plan=fast/'lightmem_plan.json', protocol_sha256=LIGHT_PROTOCOL)
    light_ids, light_rows, light_protocol, _, original = light.preflight(light_args)
    if light_ids != ids:
        raise ValueError('LightMemPopulationChanged')
    light_done = []
    for qid in ids:
        targets = ([original/qid] if qid == light.REUSE_ID else [])
        targets += list((fast/'lightmem_lanes').glob('lane_*/'+qid))
        candidates = [p for p in targets if (p/'prediction.json').exists()]
        if len(candidates)>1:
            raise ValueError('DuplicateLightMemCompletion')
        if candidates:
            light.validate_prediction(candidates[0], light_rows[qid], light_protocol)
            prediction = read(candidates[0]/'prediction.json')
            attempt = candidates[0]/prediction['attempt']
            if (not attempt.resolve().is_relative_to(candidates[0].resolve())
                    or (attempt/'failure.json').exists()
                    or not (attempt/'qdrant').is_dir()
                    or not any(p.is_file() and p.name.lower() not in {'lock', '.lock'} and p.suffix.lower() not in {'.lock', '.tmp', '.pyc', '.pem', '.key'} and p.stat().st_size > 0
                               for p in (attempt/'qdrant').rglob('*'))):
                raise ValueError('LightMemDatabaseEvidenceMissing:'+qid)
            light_done.append(qid)
    if light.REUSE_ID not in light_done:
        raise ValueError('OriginalLightMemReuseEvidenceMissing')
    restore_mode = 'provider_copy'
    archive_proof = {}
    if args.local_backups:
        restore_mode = 'local_verified_backups'
        expected_archives = {
            'sources_data.tgz':'c330cbddaf19019424a831fef5f00eaaeb23239880e9bd4e9519f42bb466903d',
            'local_recovery_code.tgz':'a03f4b8d8c3b510712c3e7568fe5d016fc16e87ed208a94eed2644adecf02b11',
            'v4_live_probe_backup_20260911.tgz':'0165034ab11708d111d9fc84250fa975e5da7bbf64eee8f7dd47293d294c057b',
            'smoke_lightmem_verified_20260910T103053Z.tgz':'1f56ededfeebfd144bc80ce4b1755e66a8f9301c8918bb82184335001bfbb05e',
        }
        extraction = read(STATE/'local_restore_extracted.json')
        if (extraction.get('mode') != restore_mode or extraction.get('instance_id') != '50577514'
                or extraction.get('archives_sha256') != expected_archives):
            raise ValueError('LocalBackupLineageMismatch')
        for name, expected in expected_archives.items():
            archive_proof[name] = sha(STATE/name)
            if archive_proof[name] != expected:
                raise ValueError('LocalBackupHashMismatch:'+name)
        for name in ('v4_live_probe_backup_20260911.tgz', 'smoke_lightmem_verified_20260910T103053Z.tgz'):
            with tarfile.open(STATE/name) as archive:
                for member in archive.getmembers():
                    if not member.isfile() or not (member.name.startswith('runs/')
                            or member.name == 'fast_native2_20260911/simplemem_v4_runtime.json'):
                        continue
                    if member.name.endswith('/status.json'):
                        continue
                    destination = (ROOT/member.name).resolve()
                    if not destination.is_relative_to(ROOT.resolve()):
                        raise ValueError('BackupMemberEscapesProject')
                    with archive.extractfile(member) as stream:
                        expected = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if sha(destination) != expected:
                        raise ValueError('ExtractedBackupMemberChanged:'+member.name)
        if not {'6ade9755','5d3d2817'}.issubset(simple_ids) or 'e47becba' not in light_done:
            raise ValueError('LocalBackupNativeEvidenceMissing')
    else:
        for method, verified, known in [('simplemem',len(simple_ids),53), ('lightmem',len(light_done),46)]:
            saved = read(fast/(method+'_status.json'))
            if verified < max(known, saved.get('generated',0)):
                raise ValueError('RestoredCoverageBelowKnownCompleted:'+method)
    model_proof = {}
    for name, model in read(STATE/'model_integrity_expected.json')['models'].items():
        actual = {}
        for relative, expected in model['files'].items():
            actual[relative] = sha(Path(model['path'])/relative)
            if actual[relative] != expected:
                raise ValueError('ModelHashMismatch:'+name+'/'+relative)
        model_proof[name] = {'path':model['path'], 'files':actual}
    receipt = {'status':'restored_artifacts_verified', 'source_instance_id':'50468468',
               'instance_id':'50577514', 'verified_at':time.time(), 'model_proof':model_proof,
               'restore_mode':restore_mode, 'local_archives_sha256':archive_proof,
               'previous_reported_counts':{'simplemem':53,'lightmem':46},
               'previous_full_run_restored':not args.local_backups,
               'protocol_proof':{'simplemem':SIMPLE_PROTOCOL,'lightmem':LIGHT_PROTOCOL},
               'simplemem_verified_ids':simple_ids, 'lightmem_verified_ids':light_done,
               'runtime_inference_verified':False}
    destination = STATE/'restored_artifacts_verified.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps({'status':receipt['status'], 'simplemem':len(simple_ids), 'lightmem':len(light_done), 'models_verified':list(model_proof), 'experiments_started':False}),flush=True)


if __name__ == '__main__':
    main()