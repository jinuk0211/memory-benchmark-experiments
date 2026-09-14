"""Drain native queues, verify admission, and resume; rollback on failure or signal."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import ProxyHandler, build_opener

import quad_config as cfg
from pipeline import rpc

Q = cfg.STATE
RECEIPT = Q / 'admission_rollout_attempt_2.json'
WRAPPER = Q / 'quad_metered_proxy.py'
PENDING = Q / 'quad_metered_proxy.admission_pending.py'
BACKUP = Q / 'admission_wrapper_backup.py'
ORIGINAL_SHA = '4012c2676c0e97b1d9758b7d738ecd064ba50d8b0406e740c265106bbb09b8bd'
WRAPPER_SHA = '700b043d38268e64a9114c3d6bf1e7b82d88b5510ad24a1b5e4cf1eaa8b0ac90'
PINS = {'quad_admission.py': '1a785ac42c264b75acdf6b8c996b5e6df4c45e7710fd3ad28d7e12437a723423',
        'canary_admission.py': '5e98177c743dded53920f7acf2044c6c4b193ee34177395c3d67f9c244b1dd6e'}
QUEUES = {'quad-simplemem': 'quad_simplemem_queue.py', 'quad-lightmem': 'quad_lightmem_queue.py'}


def atomic_bytes(path, raw):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def health():
    with build_opener(ProxyHandler({})).open('http://127.0.0.1:18083/health', timeout=5) as response:
        return json.load(response)


def argv(pid):
    return Path('/proc').joinpath(str(pid), 'cmdline').read_bytes().decode().split('\0')


def info(name):
    return rpc().getProcessInfo(name)


def wait_for(predicate, deadline, message):
    while True:
        if time.time() >= deadline:
            raise TimeoutError(message)
        if predicate():
            return
        time.sleep(1)


class Rollout:
    def __init__(self):
        self.state = cfg.read(RECEIPT) if RECEIPT.exists() else {}
        self.locks = ExitStack()
        self.paused = False
        self.locked = False
        self.cleaning = False
        self.canary = None
        self.terminated_pids = set()

    def report(self, phase, *, best_effort=False, **details):
        self.state.update(phase=phase, updated_at=time.time(), **details)
        try:
            cfg.atomic(RECEIPT, self.state)
        except OSError:
            if not (best_effort or self.cleaning):
                raise

    def interrupted(self, _signal, _frame):
        if not self.cleaning:
            raise RuntimeError('Rollout interrupted; restoring the original wrapper')

    def engines(self):
        result = {}
        for lane in ('sm0', 'sm1', 'lm0', 'lm1'):
            process = info('quad-' + lane)
            args = argv(process['pid']) if process['pid'] else []
            expected = '2' if lane.startswith('sm') else '1'
            if (process['statename'] != 'RUNNING' or args.count('--max-num-seqs') != 1
                    or args[args.index('--max-num-seqs') + 1] != expected):
                raise ValueError('Current engine sequence limit differs: ' + lane)
            result[lane] = {'pid': process['pid'], 'max_num_seqs': int(expected)}
        return result

    def preflight(self):
        if os.environ.get('CONTAINER_ID') != cfg.INSTANCE:
            raise ValueError('Wrong instance')
        cfg.require_ready()
        if cfg.sha(PENDING) != WRAPPER_SHA or any(cfg.sha(Q / name) != digest for name, digest in PINS.items()):
            raise ValueError('Pending admission/canary code differs')
        identity = {**cfg.fresh_identity(), 'instance_id': cfg.INSTANCE, 'deployment_sha256': cfg.DEPLOYMENT_SHA,
                    'original_wrapper_sha256': ORIGINAL_SHA, 'pending_wrapper_sha256': WRAPPER_SHA,
                    'code_sha256': PINS, 'launcher_sha256': cfg.sha(Q / 'serve_qwen.sh')}
        if self.state and self.state.get('identity') != identity:
            raise ValueError('Persisted rollout identity differs')
        self.state.setdefault('identity', identity)
        if self.state.get('phase') == 'complete':
            if cfg.sha(WRAPPER) != WRAPPER_SHA:
                raise ValueError('Previously verified wrapper changed')
            return 'complete'
        if self.state.get('phase') in ('rolled_back', 'rollback_failed'):
            raise ValueError('Attempt 2 apply budget exhausted')
        if self.state.get('phase') and self.state.get('attempts') != 2:
            raise ValueError('Attempt 2 receipt identity differs')
        interrupted = bool(self.state.get('phase'))
        if cfg.sha(WRAPPER) not in (ORIGINAL_SHA, WRAPPER_SHA):
            raise ValueError('Active wrapper is neither original nor pending')
        if cfg.sha(WRAPPER) == WRAPPER_SHA and not BACKUP.exists():
            raise ValueError('New wrapper has no preserved original backup')
        if BACKUP.exists() and cfg.sha(BACKUP) != ORIGINAL_SHA:
            raise ValueError('Original wrapper backup differs')
        if not interrupted:
            if cfg.sha(WRAPPER) != ORIGINAL_SHA:
                raise ValueError('A fresh rollout must begin with the original wrapper')
            if cfg.read(Q / 'pipeline_status.json').get('phase') != 'experiments_running':
                raise ValueError('Pipeline must be in its steady experiment phase')
            self.state.update(attempts=2, started_at=time.time(), installed=False,
                              drain_deadline=time.time() + 5700)
        self.state['engines'] = self.engines()
        self.report('recovering_interrupted_rollout' if interrupted else 'prepared')
        return 'recover' if interrupted else 'apply'

    def pause(self):
        self.paused = True  # Cleanup resumes the monitor even if the RPC reply is lost.
        self.report('pausing_pipeline')
        if info('quad-pipeline')['statename'] in ('RUNNING', 'STARTING'):
            rpc().stopProcess('quad-pipeline', False)
        wait_for(lambda: info('quad-pipeline')['statename'] in ('STOPPED', 'EXITED', 'FATAL'),
                 time.time() + 90, 'Pipeline monitor did not stop')

    def drain(self):
        self.report('draining_queues')
        for name, script in QUEUES.items():
            process = info(name)
            if process['statename'] in ('RUNNING', 'STARTING'):
                pid = process['pid']
                if str(Q / script) not in argv(pid):
                    raise ValueError('Coordinator PID identity differs: ' + name)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self.terminated_pids.add(pid)
        deadline = self.state.get('rollback_drain_deadline', self.state['drain_deadline']) if self.cleaning else self.state['drain_deadline']
        wait_for(lambda: all(info(name)['statename'] in ('STOPPED', 'EXITED', 'FATAL') for name in QUEUES),
                 deadline, 'Current native histories did not drain; no workers killed')
        self.acquire_locks(deadline)

    def acquire_locks(self, deadline):
        import fcntl
        if self.locked:
            return
        for method in ('simplemem', 'lightmem'):
            lock = self.locks.enter_context((cfg.ROOT / 'fast_native2_20260911' / (method + '_queue.lock')).open('a'))

            def acquire():
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    return False
            wait_for(acquire, deadline, 'A native child still holds its queue lock')
        self.locked = True
        self.report('queues_drained_and_locked')

    def idle_meter(self, maximum=900):
        def idle():
            current = info('quad-meter')['statename']
            if current in ('STOPPED', 'EXITED', 'FATAL'):
                return True
            item = health()
            if item.get('status') != 'ok':
                raise ValueError('Meter journal is not healthy')
            return item.get('in_flight') == 0
        wait_for(idle, time.time() + maximum, 'Meter requests did not drain')

    def restart_meter(self):
        if info('quad-meter')['statename'] in ('RUNNING', 'STARTING'):
            rpc().stopProcess('quad-meter', False)
        wait_for(lambda: info('quad-meter')['statename'] in ('STOPPED', 'EXITED', 'FATAL'),
                 time.time() + 90, 'Meter did not stop')
        rpc().startProcess('quad-meter', False)

        def healthy():
            if info('quad-meter')['statename'] != 'RUNNING':
                return False
            try:
                item = health()
                return item.get('status') == 'ok' and item.get('in_flight') == 0
            except OSError:
                return False
        wait_for(healthy, time.time() + 90, 'Restarted meter is not healthy')

    def stop_canary(self):
        if self.canary is None or self.canary.poll() is not None:
            return
        try:
            os.killpg(self.canary.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.canary.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.canary.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.canary.wait(timeout=5)

    def run_canary(self):
        self.report('running_canary')
        started = time.time()
        environment = {key: value for key, value in os.environ.items()
                       if not any(word in key.upper() for word in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
        command = ['timeout', '--signal=TERM', '--kill-after=10', '890',
                   str(cfg.ROOT / '.venv-inference/bin/python'), str(Q / 'canary_admission.py')]
        self.canary = subprocess.Popen(command, env=environment, cwd=cfg.ROOT, start_new_session=True)
        try:
            code = self.canary.wait(timeout=905)
        finally:
            self.stop_canary()
        if code:
            raise RuntimeError('Admission canary failed with exit ' + str(code))
        receipts = []
        for path in Q.glob('canary_admission_*.json'):
            proof = cfg.read(path)
            if proof.get('started_at', 0) >= started:
                receipts.append((path, proof))
        if len(receipts) != 1:
            raise ValueError('Expected one fresh canary receipt')
        path, proof = receipts[0]
        identity = self.state['identity']
        if (proof.get('status') != 'admission_canary_verified' or proof.get('instance_id') != cfg.INSTANCE
                or proof.get('fresh_run_sha256') != identity['fresh_run_sha256']
                or proof.get('deployment_sha256') != cfg.DEPLOYMENT_SHA
                or proof.get('canary_script_sha256') != PINS['canary_admission.py']
                or proof.get('code_sha256', {}).get('quad_metered_proxy.py') != WRAPPER_SHA
                or proof.get('code_sha256', {}).get('quad_admission.py') != PINS['quad_admission.py']
                or proof.get('summary', {}).get('sm0', {}).get('max_admission_wait_seconds', 0) <= 600):
            raise ValueError('Current canary proof differs')
        if self.engines() != self.state['engines']:
            raise ValueError('An inference engine or sequence limit changed')
        self.report('canary_verified', canary_receipt=str(path), canary_receipt_sha256=cfg.sha(path))

    def restore(self):
        self.stop_canary()
        if self.locked:
            if not BACKUP.exists() or cfg.sha(BACKUP) != ORIGINAL_SHA:
                raise ValueError('Cannot restore an unverified original wrapper')
            atomic_bytes(WRAPPER, BACKUP.read_bytes())
            if cfg.sha(WRAPPER) != ORIGINAL_SHA:
                raise ValueError('Wrapper rollback verification failed')
            self.restart_meter()
            self.report('original_wrapper_restored', restored_wrapper_sha256=cfg.sha(WRAPPER),
                        original_meter_healthy=True)

    def resume(self):
        if self.paused:
            if info('quad-meter')['statename'] != 'RUNNING' or health().get('status') != 'ok':
                raise ValueError('Queues cannot resume without a running healthy meter')
        self.locks.close()
        self.locked = False
        if not self.paused:
            return
        self.report('resuming_pipeline', best_effort=True)
        if info('quad-pipeline')['statename'] not in ('RUNNING', 'STARTING'):
            rpc().startProcess('quad-pipeline', False)

        def running():
            if info('quad-pipeline')['statename'] not in ('RUNNING', 'STARTING'):
                raise RuntimeError('Pipeline failed while resuming')
            queues = [info(name) for name in QUEUES]
            return all(item['statename'] == 'RUNNING' and item['pid'] not in self.terminated_pids for item in queues)
        wait_for(running, time.time() + 1500, 'Both native coordinators did not resume')

    def execute(self):
        mode = self.preflight()
        if mode == 'complete':
            return 0
        changed = mode == 'recover'
        try:
            self.pause()
            self.drain()
            self.idle_meter()
            if not BACKUP.exists():
                if cfg.sha(WRAPPER) != ORIGINAL_SHA:
                    raise ValueError('Cannot back up a non-original wrapper')
                atomic_bytes(BACKUP, WRAPPER.read_bytes())
            if mode == 'recover':
                raise RuntimeError('Interrupted rollout requires original-wrapper recovery')
            raw = PENDING.read_bytes()
            if (hashlib.sha256(raw).hexdigest() != WRAPPER_SHA or cfg.sha(BACKUP) != ORIGINAL_SHA
                    or any(cfg.sha(Q / name) != digest for name, digest in PINS.items())
                    or self.engines() != self.state['engines']):
                raise ValueError('Staged code or inference engines changed during drain')
            self.report('installing_wrapper')
            changed = True  # Includes an ambiguous interrupted atomic replacement.
            atomic_bytes(WRAPPER, raw)
            self.report('wrapper_installed', installed=True, installed_at=time.time())
            self.restart_meter()
            self.run_canary()
            self.resume()
            self.report('complete', queues_resumed=True, error_class=None)
            return 0
        except BaseException as error:
            self.cleaning = True
            self.report('rolling_back', error_class=type(error).__name__)
            try:
                self.stop_canary()
                if changed or self.state.get('installed') or cfg.sha(WRAPPER) != ORIGINAL_SHA:
                    # A failed resume may already have released locks and launched fresh histories.
                    if not self.locked:
                        self.state.setdefault('rollback_drain_deadline', time.time() + 5700)
                        self.pause()
                        self.drain()
                    self.restore()
            except BaseException as rollback_error:
                self.state['rollback_error'] = type(rollback_error).__name__
            finally:
                if self.state.get('rollback_error'):
                    self.locks.close()
                    self.locked = False
                else:
                    try:
                        self.resume()
                    except BaseException as resume_error:
                        self.state['resume_error'] = type(resume_error).__name__
            failed = bool(self.state.get('rollback_error') or self.state.get('resume_error'))
            self.report('rollback_failed' if failed else 'rolled_back', queues_resumed=not failed,
                        needs_attention=failed)
            return 1


def main():
    import fcntl
    socket.setdefaulttimeout(10)
    with (Q / 'admission_rollout.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rollout = Rollout()
        signal.signal(signal.SIGTERM, rollout.interrupted)
        signal.signal(signal.SIGINT, rollout.interrupted)
        result = rollout.execute()
    print(json.dumps({'phase': rollout.state.get('phase'), 'receipt': str(RECEIPT)}), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())