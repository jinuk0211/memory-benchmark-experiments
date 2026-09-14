"""Verify fresh-run inputs, frozen source, models and three installed environments."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time
from urllib.parse import unquote, urlsplit

import prepare_fresh as fresh

RUNNER_SHA = '733fb9a3f9ae33d0c62ca0099ee2e4e7f1630e6906bd216018040ab654eff582'
UPSTREAM_SHA = '32912049141844f3d174a731093d87fbf8b086984e53cdbca245a6099e807f07'
ENVIRONMENTS = {
    '.venv-inference': ('inference', '3.12.14', '17a9b7b6d55306a5ded3921e1eb02f60ad41cfc2c60f696b68379db1569df2b7'),
    '.venv-lightmem': ('lightmem', '3.11.16', '07a382574228c76b901a3574c252b8646b49ac6e7387acce1909e31176e4fbf8'),
    '.venv-simplemem-native0253': ('simplemem', '3.11.16', '4b9d7900e6e158e98f3b31e919692d765cedcecc549020ffc54fabc96f0def50'),
}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify_code(root, state):
    path = state / 'local_recovery_code.tgz'
    if fresh.sha(path) != fresh.CODE_SHA:
        raise ValueError('Frozen source archive differs')
    count = 0
    names = set()
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isdir():
                continue
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or member.name in names):
                raise ValueError('Invalid frozen source archive member')
            names.add(member.name)
            target = root / member.name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Frozen source destination escaped root')
            with archive.extractfile(member) as stream:
                wanted = hashlib.file_digest(stream, 'sha256').hexdigest()
            if fresh.sha(target) != wanted:
                raise ValueError('Frozen source file differs: ' + member.name)
            count += 1
    if count != 558:
        raise ValueError('Frozen source inventory differs')
    return {'archive_sha256': fresh.CODE_SHA, 'verified_files': count}


def verify_protocols(root):
    dataset = root / 'longmemeval_s_cleaned.json'
    if fresh.sha(dataset) != fresh.DATA_SHA:
        raise ValueError('Canonical dataset differs')
    rows = fresh.read(dataset)
    ids = [row['question_id'] for row in rows]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError('Expected exact canonical population500')
    simple_path = root / 'runs/simplemem_native_dialogues_v4/protocol.json'
    simple = fresh.read(simple_path)
    runner_root = root / 'official_recovery/simplemem_native_dialogues_v4'
    if fresh.sha(runner_root / 'runner.py') != RUNNER_SHA:
        raise ValueError('SimpleMem runner differs')
    if (canonical(simple) != fresh.SIMPLE_PROTOCOL or simple['dataset_sha256'] != fresh.DATA_SHA
            or simple['population_ids'] != ids
            or simple['policy']['official_commit'] != 'db80b6a7c591e0ea730a058e9f5fc4eb06572299'):
        raise ValueError('SimpleMem protocol differs')
    for name, expected in simple['source_files_sha256'].items():
        target = (root / 'native_five.py' if name == 'harness/native_five.py' else
                  root / 'source/MemoryData/utils/request_metering.py' if name == 'harness/request_metering.py' else
                  runner_root / name)
        if fresh.sha(target) != expected:
            raise ValueError('SimpleMem protocol source differs: ' + name)
    for filename, expected in simple['embedding_config_sha256'].items():
        if fresh.sha(Path(simple['runtime']['embedding_model']) / filename) != expected:
            raise ValueError('SimpleMem embedding configuration differs')
    light_path = root / 'runs/lightmem/protocol.json'
    if fresh.sha(light_path) != fresh.LIGHT_PROTOCOL:
        raise ValueError('LightMem protocol differs')
    light = fresh.read(light_path)
    if (fresh.sha(root / 'native_lightmem.py') != light['runner_sha256']
            or light['upstream_sha256'] != UPSTREAM_SHA
            or fresh.sha(root / 'source/LightMem/experiments/longmemeval/run_lightmem_qwen.py') != UPSTREAM_SHA
            or light['dataset_sha256'] != fresh.DATA_SHA):
        raise ValueError('LightMem source differs')
    command = fresh.read(root / 'fast_native2_20260911/lightmem_plan.json')['runner_commands']['lightmem']
    flags = {'--dataset', '--run-dir', '--source-root', '--api-base', '--model',
             '--embedding-model', '--compressor-model'}
    if (len(command) != 2 + 2 * len(flags) or set(command[2::2]) != flags
            or command[1] != str(root / 'native_lightmem.py')):
        raise ValueError('LightMem native command changed')
    arguments = dict(zip(command[2::2], command[3::2]))
    for key in ('model', 'embedding_model', 'compressor_model'):
        if arguments['--' + key.replace('_', '-')] != light[key]:
            raise ValueError('LightMem model argument differs')
    if (arguments['--dataset'] != str(dataset) or arguments['--run-dir'] != str(light_path.parent)
            or arguments['--source-root'] != str(root / 'source/LightMem')):
        raise ValueError('LightMem frozen input paths differ')
    return {'simplemem': fresh.SIMPLE_PROTOCOL, 'lightmem': fresh.LIGHT_PROTOCOL}


def verify_models(state):
    # Reuse inventory validation, not the downloader receipt, as byte verification is independent.
    import download_models
    manifest_path = state / 'model_integrity_expected.json'
    if fresh.sha(manifest_path) != '18e3df8b2c2b6b9873619ad31ce8b2f55fde93c987a1ea3b48586b1bd237c02d':
        raise ValueError('Pinned model manifest differs')
    manifest = fresh.read(manifest_path)
    files = download_models.inventory(manifest)
    proof = {}
    for item in files:
        target = item['target']
        if target.stat().st_size != item['bytes'] or fresh.sha(target) != item['sha256']:
            raise ValueError('Pinned model file differs: ' + item['model'] + '/' + item['name'])
        model = proof.setdefault(item['model'], {'path': manifest['models'][item['model']]['path'], 'files': {}})
        model['files'][item['name']] = item['sha256']
    if len(files) != 30:
        raise ValueError('Pinned model inventory differs')
    return proof


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def expected_packages(requirements):
    packages = {}
    for line in requirements.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ' @ ' in line:
            name, url = line.split(' @ ', 1)
            wheel = unquote(Path(urlsplit(url).path).name)
            parts = wheel.split('-')
            if not wheel.endswith('.whl') or len(parts) < 5 or normalized(parts[0]) != normalized(name):
                raise ValueError('Invalid frozen wheel requirement')
            version = parts[1]
        elif line.count('==') == 1:
            name, version = line.split('==')
        else:
            raise ValueError('Unpinned environment requirement')
        key = normalized(name)
        if key in packages:
            raise ValueError('Duplicate environment package')
        packages[key] = version
    return packages


def verify_environment(root, state, env_name, details, uv):
    label, python_version, required_sha = details
    requirements = state / ('bootstrap_' + label + '_requirements.txt')
    if fresh.sha(requirements) != required_sha:
        raise ValueError('Frozen environment requirements differ: ' + label)
    interpreter = root / env_name / 'bin/python'
    environment = {key: value for key, value in os.environ.items()
                   if not any(word in key.upper() for word in ('TOKEN', 'API_KEY', 'SECRET', 'PASSWORD'))}
    def run(command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, env=environment)
        if result.returncode:
            raise ValueError('Environment verification command failed: ' + label)
        return result.stdout
    current_freeze = run([uv, 'pip', 'freeze', '--python', str(interpreter)])
    freeze_path = state / ('bootstrap_' + env_name + '.freeze.txt')
    if current_freeze != freeze_path.read_text(encoding='utf-8'):
        raise ValueError('Installed environment differs from bootstrap freeze: ' + label)
    run([uv, 'pip', 'check', '--python', str(interpreter)])
    inventory = json.loads(run([str(interpreter), '-c',
        'import importlib.metadata as m,json,sys;print(json.dumps({"python":sys.version.split()[0],'
        '"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))']))
    actual = {normalized(name): version for name, version in inventory['packages'].items()}
    expected = expected_packages(requirements.read_text(encoding='utf-8'))
    if inventory['python'] != python_version or actual != expected:
        raise ValueError('Installed package versions differ from pinned requirements: ' + label)
    return {'python': inventory['python'], 'package_count': len(actual),
            'requirements_sha256': required_sha, 'freeze_sha256': fresh.sha(freeze_path),
            'pip_check_returncode': 0, 'verified_at': time.time()}


def verify(root=fresh.ROOT, state=fresh.STATE):
    auth, auth_sha = fresh.authorization(state)
    init = fresh.read(state / 'fresh_initialized.json')
    if (init.get('status') != 'fresh_initialized' or init.get('instance_id') != fresh.INSTANCE
            or init.get('fresh_run_sha256') != auth_sha
            or init.get('initial_counts') != {'simplemem': 0, 'lightmem': 0}
            or init.get('past_results_reused') is not False):
        raise ValueError('Fresh initialization receipt differs')
    fresh.verify_members(root)
    source_proof = verify_code(root, state)
    model_proof = verify_models(state)
    protocol_proof = verify_protocols(root)
    uv = shutil.which('uv')
    if not uv:
        raise ValueError('The bootstrap uv executable is unavailable')
    environment_proof = {name: verify_environment(root, state, name, details, uv)
                         for name, details in ENVIRONMENTS.items()}
    if fresh.authorization(state)[1] != auth_sha:
        raise ValueError('Fresh authorization changed during verification')
    result = {'status': 'fresh_artifacts_verified', 'instance_id': fresh.INSTANCE,
              'run_id': auth['run_id'], 'fresh_run_sha256': auth_sha,
              'fresh_initialized_sha256': fresh.sha(state / 'fresh_initialized.json'),
              'past_results_reused': False, 'canonical_population': 500,
              'methods': {'simplemem': 500, 'lightmem': 500}, 'verified_at': time.time(),
              'model_proof': model_proof, 'protocol_proof': protocol_proof,
              'source_proof': source_proof, 'environment_proof': environment_proof,
              'runtime_inference_verified': False}
    fresh.atomic(state / 'fresh_artifacts_verified.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if os.environ.get('CONTAINER_ID') != fresh.INSTANCE:
        raise ValueError('Wrong fresh-run instance')
    os.environ.pop('CONTAINER_API_KEY', None)
    receipt = verify()
    print(json.dumps({'status': receipt['status'], 'past_results_reused': False,
                      'models_verified': list(receipt['model_proof']),
                      'environments_verified': list(receipt['environment_proof'])}), flush=True)


if __name__ == '__main__':
    main()
