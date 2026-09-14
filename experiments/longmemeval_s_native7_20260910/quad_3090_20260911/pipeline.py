"""Server-local setup, proof, fresh queues and verified archive. No provider API."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import xmlrpc.client

import quad_config as cfg


def rpc():
    from supervisor.xmlrpc import SupervisorTransport
    return xmlrpc.client.ServerProxy('http://127.0.0.1', transport=SupervisorTransport(
        None, None, 'unix:///var/run/supervisor.sock')).supervisor


def report(phase, **extra):
    item = {'phase': phase, 'instance_id': cfg.INSTANCE, 'time': time.time(), **extra}
    cfg.atomic(cfg.STATE / 'pipeline_status.json', item)
    print(json.dumps(item), flush=True)


def wait_service_finished(name, maximum):
    until = time.monotonic() + maximum
    while time.monotonic() < until:
        info = rpc().getProcessInfo(name)
        if info['statename'] == 'EXITED':
            if info['exitstatus'] == 0:
                return
            raise RuntimeError(name + ' failed; inspect its preserved log')
        if info['statename'] in {'FATAL', 'STOPPED', 'UNKNOWN'}:
            raise RuntimeError(name + ' is not progressing: ' + info['statename'])
        time.sleep(10)
    raise TimeoutError(name + ' did not finish within its preparation budget')


def run(script, maximum, python=None):
    command = [python or '/venv/main/bin/python3', str(cfg.STATE / script)]
    with (cfg.STATE / (script + '.log')).open('a') as stream:
        subprocess.run(['timeout', '--signal=TERM', '--kill-after=20s', str(maximum), *command],
                       cwd=cfg.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       timeout=maximum + 40, check=True)


def start(name, retry=False):
    info = rpc().getProcessInfo(name)
    if info['statename'] in {'RUNNING', 'STARTING'}:
        return
    if info['statename'] not in {'STOPPED', 'EXITED'}:
        raise RuntimeError('Unexpected service state: ' + name + ':' + info['statename'])
    if info['statename'] == 'EXITED' and name not in ('quad-simplemem', 'quad-lightmem') and not retry:
        return
    rpc().startProcess(name, False)


def recover_inference():
    path = cfg.STATE / 'service_restarts.json'
    counts = cfg.read(path) if path.exists() else {}
    for name in ('quad-sm0', 'quad-sm1', 'quad-lm0', 'quad-lm1', 'quad-meter', 'quad-minilm'):
        info = rpc().getProcessInfo(name)
        if info['statename'] != 'EXITED':
            continue
        used = counts.get(name, 0)
        if used >= 2:
            raise RuntimeError('Inference restart budget exhausted: ' + name)
        counts[name] = used + 1
        cfg.atomic(path, counts)
        start(name, retry=True)
        print(json.dumps({'event': 'service_retry', 'service': name, 'attempt': used + 1}), flush=True)

def wait_inference():
    targets = {key: item['api_base'] + '/models' for key, item in cfg.deployment()['lanes'].items()}
    targets['meter'] = 'http://127.0.0.1:18083/v1/models'
    until = time.monotonic() + 900
    while targets and time.monotonic() < until:
        recover_inference()
        for name, url in list(targets.items()):
            try:
                with urlopen(url, timeout=2) as response:
                    data = json.load(response)
                if any(item['id'] == cfg.MODEL and item.get('max_model_len') == 65536
                       for item in data.get('data', [])):
                    del targets[name]
            except (OSError, ValueError):
                pass
        if targets:
            time.sleep(5)
    if targets:
        raise TimeoutError('Inference endpoints not ready: ' + ','.join(targets))
    until = time.monotonic() + 120
    while time.monotonic() < until:
        recover_inference()
        try:
            with urlopen('http://127.0.0.1:18084/v1/models', timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(3)
    raise TimeoutError('CPU embedding endpoint not ready')


def verify_archive_receipt():
    receipt = cfg.read(cfg.STATE / 'archive_verified.json')
    if (receipt.get('status') != 'archive_verified' or receipt.get('instance_id') != cfg.INSTANCE
            or receipt.get('generated') != {'simplemem': 500, 'lightmem': 500}
            or receipt.get('fresh_run_sha256') != cfg.sha(cfg.STATE / 'FRESH_RUN.json')):
        raise ValueError('Final archive receipt identity differs')
    archive = Path(receipt['archive']).resolve()
    if (not archive.is_relative_to((cfg.ROOT / 'archives').resolve())
            or cfg.sha(archive) != receipt['archive_sha256']):
        raise ValueError('Final archive file differs')
    return receipt

def main():
    import fcntl
    if os.environ.get('CONTAINER_ID') != cfg.INSTANCE:
        raise ValueError('Wrong instance')
    for key in list(os.environ):
        if any(word in key.upper() for word in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            os.environ.pop(key, None)
    os.environ.update(PYTHONPATH=str(cfg.ROOT) + ':' + str(cfg.ROOT / 'fast_native2_20260911'),
                      HF_HOME='/workspace/.hf_home', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      NLTK_DATA='/root/.cache/longmemeval_s_native7_20260910/nltk_data')
    with (cfg.STATE / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (cfg.STATE / 'archive_verified.json').exists():
            run('archive_results.py', 3600)
            verify_archive_receipt()
            report('complete', archive_receipt=str(cfg.STATE / 'archive_verified.json'))
            return
        report('waiting_for_preparation')
        wait_service_finished('quad-install', 3900)
        wait_service_finished('quad-models', 9000)
        report('preparing_fresh_dataset')
        run('prepare_fresh.py', 600)
        with (cfg.STATE / 'nltk.log').open('a') as stream:
            subprocess.run(['bash', str(cfg.STATE / 'prepare_nltk.sh')], stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=330)
        report('verifying_frozen_artifacts')
        run('verify_fresh.py', 900)
        report('starting_inference')
        for name in ('quad-sm0', 'quad-sm1', 'quad-lm0', 'quad-lm1', 'quad-meter', 'quad-minilm'):
            start(name)
        wait_inference()
        if not (cfg.STATE / 'READY.json').exists():
            report('proving_all_four_gpus')
            run('probe_replicas.py', 1200)
        cfg.require_ready()
        report('starting_fresh_queues')
        for name in ('quad-simplemem', 'quad-lightmem'):
            start(name)
        report('experiments_running', fresh_run_sha256=cfg.sha(cfg.STATE / 'FRESH_RUN.json'))
        while True:
            recover_inference()
            infos = [rpc().getProcessInfo(name) for name in ('quad-simplemem', 'quad-lightmem')]
            if all(info['statename'] == 'EXITED' and info['exitstatus'] == 0 for info in infos):
                break
            if any(info['statename'] in {'FATAL', 'STOPPED', 'UNKNOWN'}
                   or info['statename'] == 'EXITED' and info['exitstatus'] != 0 for info in infos):
                raise RuntimeError('Experiment coordinator needs attention; artifacts preserved')
            time.sleep(30)
        report('verifying_final_archive')
        run('archive_results.py', 3600)
        verify_archive_receipt()
        report('complete', archive_receipt=str(cfg.STATE / 'archive_verified.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        report('needs_attention', error_type=type(error).__name__, message=str(error))
        sys.exit(1)