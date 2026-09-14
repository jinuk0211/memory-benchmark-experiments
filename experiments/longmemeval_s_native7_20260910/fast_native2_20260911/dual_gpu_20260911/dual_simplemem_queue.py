"""Coordinate two isolated native workers without changing the frozen protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

LANES = {
    'local': {'instance_id': '50468468', 'api_base': 'http://127.0.0.1:18081/v1'},
    'v100': {'instance_id': '50558359', 'api_base': 'http://127.0.0.1:18091/v1'},
}
HISTORY_TIMEOUT = 5400
REMOTE_WAIT = 1800
PROTOCOL_SHA = '099254c45e1077d44c851907f5f37bd1e0a36e7127e9eef87e406181d479977b'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_ready(endpoint):
    try:
        with urlopen(endpoint + '/models', timeout=10) as response:
            return any(item['id'] == 'Qwen/Qwen3.5-9B' for item in json.load(response)['data'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def remote_enabled(path):
    try:
        receipt = json.loads(path.read_text(encoding='utf-8'))
        return receipt.get('status') == 'inference_verified' and receipt.get('instance_id') == '50558359'
    except (OSError, ValueError, AttributeError):
        return False


class Coordinator:
    def __init__(self, queue, native, root, runner, data, protocol, receipt_path, remote_file):
        self.queue, self.native, self.root, self.runner = queue, native, root, runner
        self.run_dir = root / 'runs/simplemem_native_dialogues_v4'
        self.state_dir = root / 'fast_native2_20260911'
        self.dual_dir = root / 'queue/dual_gpu_20260911'
        self.dual_dir.mkdir(parents=True, exist_ok=True)
        self.protocol, self.protocol_hash = protocol, native.digest(protocol)
        self.receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        self.remote_file = remote_file
        self.rows = {row['question_id']: row for row in data}
        self.ids = list(self.rows)
        self.histories = {qid: self.run_dir / 'histories' / hashlib.sha256(qid.encode()).hexdigest()[:24]
                          for qid in self.ids}
        self.identities = {qid: {'protocol_sha256': self.protocol_hash,
            'source_sha256': native.digest(native.source_only(row)),
            'query_sha256': native.digest({key: row[key] for key in ('question_id', 'question', 'question_date')})}
            for qid, row in self.rows.items()}
        self.predictions, self.failures, self.inflight = {}, {}, {}
        self.stopping = False
        self.health = dict.fromkeys(LANES, False)
        self.remote_down_since, self.remote_disabled = None, False
        self.next_health = 0
        self.route_path = self.dual_dir / 'routes.json'
        initialized_path = self.dual_dir / 'initialized.json'
        initialized = initialized_path.exists()
        marker = {'schema': 'dual-simplemem-queue-initialized-v1',
                  'protocol_sha256': self.protocol_hash, 'lanes': LANES}
        if initialized:
            if native.read_json(initialized_path) != marker:
                raise ValueError('Initialization marker protocol/lanes differs')
            if not self.route_path.exists():
                raise ValueError('Initialized queue lost its routes manifest; restore it before restart')
        self.routes = native.read_json(self.route_path) if self.route_path.exists() else {
            'protocol_sha256': self.protocol_hash, 'lanes': LANES, 'histories': {}}
        if (self.routes.get('protocol_sha256') != self.protocol_hash or self.routes.get('lanes') != LANES
                or not isinstance(self.routes.get('histories'), dict)
                or set(self.routes['histories']) - self.rows.keys()):
            raise ValueError('Route file protocol/lanes/population differs')
        for route in self.routes['histories'].values():
            lane = route.get('lane')
            if (lane not in LANES or route.get('instance_id') != LANES[lane]['instance_id']
                    or route.get('protocol_sha256') != self.protocol_hash):
                raise ValueError('Invalid persisted route')
        # Independent attempt evidence also detects loss of both coordinator state files.
        for qid in self.ids:
            for path in self.histories[qid].glob('attempt_*/dual_route.json'):
                route = self.routes['histories'].get(qid)
                if route is None or native.read_json(path) != {'question_id': qid, **route}:
                    raise ValueError('Dual attempt route is missing or differs from its durable receipt')
        for qid in self.ids:
            saved = native.verified(self.histories[qid], self.identities[qid])
            if saved is not None:
                if saved.get('question_id') != qid:
                    raise ValueError('Completed question ID differs')
                self.predictions[qid] = saved
            elif self.attempts(qid):
                # Every pre-migration attempt ran on the original local instance.
                if qid not in self.routes['histories']:
                    if initialized:
                        raise ValueError('Initialized queue lost a history route; restore it before restart')
                    self.assign(qid, 'local')
                result = self.attempts(qid)[-1] / 'coordinator_result.json'
                self.failures[qid] = native.read_json(result) if result.exists() else {
                    'question_id': qid, 'attempts': len(self.attempts(qid)),
                    'reason': 'Previous native attempt incomplete; artifacts preserved'}
        self.save_routes()
        if not initialized:
            self.queue.atomic(initialized_path, marker)

    def attempts(self, qid):
        return sorted(path for path in self.histories[qid].glob('attempt_*') if path.is_dir())

    def save_routes(self):
        self.routes['updated_at'] = time.time()
        self.queue.atomic(self.route_path, self.routes)

    def assign(self, qid, lane):
        old = self.routes['histories'].get(qid)
        if old is not None:
            if old['lane'] != lane:
                raise ValueError('An assigned history cannot change lanes')
            return
        self.routes['histories'][qid] = {
            'lane': lane, 'instance_id': LANES[lane]['instance_id'],
            'protocol_sha256': self.protocol_hash, 'assigned_at': time.time()}
        self.save_routes()

    def candidate(self, lane):
        busy = {job['qid'] for job in self.inflight.values()}
        eligible = [qid for qid in self.ids if qid not in self.predictions and qid not in busy
                    and len(self.attempts(qid)) < self.queue.MAX_ATTEMPTS]
        for assigned in (True, False):
            for qid in eligible:
                route = self.routes['histories'].get(qid)
                if (assigned and route is not None and route['lane'] == lane) or (not assigned and route is None):
                    return qid
        return None

    def launch(self, lane, qid):
        if self.stopping or lane in self.inflight or qid != self.candidate(lane):
            raise ValueError('Dispatch would duplicate, exhaust or reassign work')
        self.assign(qid, lane)  # Persist BEFORE allocating or launching any native work.
        self.save_routes()
        attempt = self.native.next_attempt(self.histories[qid])
        self.queue.atomic(attempt / 'dual_route.json', {'question_id': qid, **self.routes['histories'][qid]})
        row = self.rows[qid]
        for name, value in (
            ('source.json', self.native.source_only(row)),
            ('query.json', {key: row[key] for key in ('question_id', 'question', 'question_date')}),
            ('worker.json', {'identity': self.identities[qid], 'runtime': self.protocol['runtime'],
                             'source_files_sha256': self.protocol['source_files_sha256']}),
        ):
            self.native.save_json(attempt / name, value)
        output = (attempt / 'console.log').open('w', encoding='utf-8')
        job = {'qid': qid, 'attempt': attempt, 'started': time.monotonic(), 'output': output,
               'timeout_at': None}
        try:
            job['process'] = subprocess.Popen(
                [sys.executable, str(self.runner), '--worker-dir', str(attempt.resolve())],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            output.close()
            self.finish(job, 127, reason=f'Worker launch failed: {exc}')
            return
        self.inflight[lane] = job

    def finish(self, job, code, reason=None):
        qid, attempt = job['qid'], job['attempt']
        result = {'question_id': qid, 'attempt': str(attempt), 'attempts': len(self.attempts(qid)),
                  'exit_code': 124 if job['timeout_at'] is not None else code,
                  'process_returncode': code, 'completed_at': time.time()}
        if job['timeout_at'] is not None:
            result['reason'] = 'history_timeout_5400_seconds'
        elif reason:
            result['reason'] = reason
        self.queue.atomic(attempt / 'coordinator_result.json', result)
        saved = self.native.verified(self.histories[qid], self.identities[qid])
        if result['exit_code'] == 0 and saved is not None:
            self.predictions[qid] = saved
            self.failures.pop(qid, None)
        else:
            self.failures[qid] = result
            if not (attempt / 'failure.json').exists():
                self.native.save_json(attempt / 'failure.json', result)

    @staticmethod
    def signal_group(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def reap(self, now):
        for lane, job in list(self.inflight.items()):
            process = job['process']
            code = process.poll()
            if code is None and job['timeout_at'] is None and now - job['started'] >= HISTORY_TIMEOUT:
                job['timeout_at'] = now
                self.signal_group(process, signal.SIGTERM)
            if job['timeout_at'] is not None and (code is not None or now - job['timeout_at'] >= 10):
                # Kill surviving descendants too, even if their worker parent already exited.
                self.signal_group(process, signal.SIGKILL)
            if code is not None:
                process.wait()
                job['output'].close()
                del self.inflight[lane]
                self.finish(job, code)

    def check_health(self, now):
        if now < self.next_health:
            return
        self.next_health = now + 30
        self.health['local'] = model_ready(LANES['local']['api_base'])
        self.health['v100'] = (not self.remote_disabled and remote_enabled(self.remote_file)
                               and model_ready(LANES['v100']['api_base']))
        if self.health['v100']:
            self.remote_down_since = None
        elif self.remote_down_since is None:
            self.remote_down_since = now
        if self.remote_down_since is not None and now - self.remote_down_since >= REMOTE_WAIT:
            self.remote_disabled = True
            self.health['v100'] = False

    def publish(self, phase='running'):
        active = [job['qid'] for job in self.inflight.values()]
        complete = self.queue.publish(self.run_dir, self.state_dir, self.ids, self.predictions,
                                      self.failures, active=active[0] if active else None, phase=phase)
        self.queue.atomic(self.dual_dir / 'status.json', {
            'protocol_sha256': self.protocol_hash, 'runtime_receipt_sha256': self.receipt_sha,
            'phase': phase, 'generated': len(self.predictions), 'failed': len(self.failures),
            'pending_ids': [qid for qid in self.ids if qid not in self.predictions and qid not in active],
            'lanes': {lane: {**config, 'healthy': self.health[lane]} for lane, config in LANES.items()},
            'remote_disabled': self.remote_disabled, 'remote_enabled_file': str(self.remote_file),
            'inflight': {lane: {'question_id': job['qid'], 'attempt': str(job['attempt']),
                'pid': job['process'].pid, 'seconds': time.monotonic() - job['started'],
                'timed_out': job['timeout_at'] is not None} for lane, job in self.inflight.items()},
            'stopping': self.stopping, 'updated_at': time.time()})
        return complete

    def stop(self, *_):
        self.stopping = True  # SIGTERM stops dispatch; live work keeps its full timeout.

    def execute(self):
        try:
            while True:
                self.reap(time.monotonic())
                if self.stopping:
                    if not self.inflight:
                        break
                    self.publish('draining')
                else:
                    self.check_health(time.monotonic())
                    disk_ready = shutil.disk_usage(self.root).free >= 2 * 1024**3
                    for lane in LANES:
                        qid = self.candidate(lane)
                        if self.stopping:
                            break
                        if lane not in self.inflight and self.health[lane] and disk_ready and qid is not None:
                            self.launch(lane, qid)
                    self.publish('running' if self.inflight else 'waiting_for_model' if disk_ready else 'waiting_for_disk')
                    if not self.inflight and self.candidate('local') is None and (
                            self.remote_disabled or self.candidate('v100') is None):
                        break
                time.sleep(5)
        finally:
            # Unexpected coordinator errors also drain, never orphan a running history.
            self.stopping = self.stopping or bool(self.inflight)
            while self.inflight:
                try:
                    self.reap(time.monotonic())
                    self.publish('draining')
                except Exception as exc:
                    try:
                        print(f'Drain metadata error: {exc}', file=sys.stderr, flush=True)
                    except OSError:
                        pass  # A broken log stream must not orphan native workers.
                if self.inflight:
                    time.sleep(5)
        complete = self.publish('paused' if self.stopping else 'finished')
        return 0 if complete else 130 if self.stopping else 1


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-receipt', type=Path, required=True)
    parser.add_argument('--remote-enabled-file', type=Path)
    args = parser.parse_args()
    queue = load_module('existing_simplemem_queue', Path(__file__).resolve().parents[1] / 'simplemem_queue.py')
    root = queue.ROOT.resolve()
    runner = root / 'official_recovery/simplemem_native_dialogues_v4/runner.py'
    if hashlib.sha256(runner.read_bytes()).hexdigest() != queue.RUNNER_SHA:
        raise ValueError('Native SimpleMem runner changed')
    receipt = json.loads(args.runtime_receipt.read_text(encoding='utf-8'))
    if (receipt.get('status') != 'runtime_verified' or receipt.get('candidate') != str(runner.parent)
            or receipt['integrity']['runtime_files']['official_recovery/simplemem_native_dialogues_v4/runner.py'] != queue.RUNNER_SHA):
        raise ValueError('Expected receipt binding the verified native runtime')
    state_dir = root / 'fast_native2_20260911'
    state_dir.mkdir(exist_ok=True)
    with (state_dir / 'simplemem_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = root / 'runs/simplemem_native_dialogues_v4'
        if queue.live_native_workers(runner, run):
            raise RuntimeError('Existing native process must exit before queue starts')
        raw = (root / 'longmemeval_s_cleaned.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != queue.DATA_SHA:
            raise ValueError('Canonical dataset changed')
        data = json.loads(raw)
        ids = [row['question_id'] for row in data]
        if len(ids) != 500 or len(set(ids)) != 500:
            raise ValueError('Expected canonical 500 unique histories')
        native = load_module('native_simplemem_v4_dual', runner)
        protocol = native.read_json(run / 'protocol.json')
        runtime = protocol['runtime']
        embedding = Path(runtime['embedding_model']).resolve()
        if not embedding.is_dir() or embedding.name != native.MINILM_REVISION:
            raise ValueError('Expected pinned local MiniLM snapshot directory')
        expected_runtime = {'method': 'simplemem', 'api_base': 'http://127.0.0.1:18083/v1',
                            'model': native.MODEL, 'embedding_model': str(embedding),
                            'embedding_revision': native.MINILM_REVISION}
        expected = {'dataset_sha256': queue.DATA_SHA, 'population_ids': ids, 'runtime': expected_runtime,
                    'source_files_sha256': native.source_hashes(), 'policy': native.POLICY,
                    'embedding_config_sha256': {name: native.sha(embedding / name)
                        for name in ('config.json', 'modules.json', 'sentence_bert_config.json')}}
        if protocol != expected or native.digest(protocol) != PROTOCOL_SHA:
            raise ValueError('Native protocol/source/runtime mismatch')
        remote_file = args.remote_enabled_file or root / 'queue/dual_gpu_20260911/inference_verified.json'
        coordinator = Coordinator(queue, native, root, runner, data, protocol, args.runtime_receipt, remote_file)
        signal.signal(signal.SIGTERM, coordinator.stop)
        signal.signal(signal.SIGINT, coordinator.stop)
        return coordinator.execute()


if __name__ == '__main__':
    sys.exit(main())
