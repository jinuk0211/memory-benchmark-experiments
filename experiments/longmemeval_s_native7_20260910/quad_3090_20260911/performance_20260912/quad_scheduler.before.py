"""Two isolated histories per coordinator; bounded retries and process-group draining."""
import os
import shutil
import signal
import subprocess
import time
from urllib.request import urlopen
import json

import quad_config as config


def model_ready(endpoint):
    try:
        with urlopen(endpoint + '/models', timeout=10) as response:
            return any(row.get('id') == config.MODEL and row.get('max_model_len') == 65536
                       for row in json.load(response)['data'])
    except (OSError, ValueError, KeyError):
        return False


class Scheduler:
    def __init__(self, adapter, routes, lock_fd):
        self.adapter, self.routes, self.lock_fd = adapter, routes, lock_fd
        self.lanes = {name: value for name, value in routes.config['lanes'].items()
                      if value['method'] == adapter.method}
        self.active, self.unavailable_since, self.disabled = {}, {}, set()
        self.health = dict.fromkeys(self.lanes, False)
        self.next_health, self.stopping = 0, False

    def stop(self, *_):
        self.stopping = True

    def candidate(self, lane):
        busy = {job['qid'] for job in self.active.values()}
        eligible = [qid for qid in self.adapter.ids if qid not in self.adapter.completed
                    and qid not in busy and self.adapter.attempts(qid) < config.MAX_ATTEMPTS]
        # Finish the first pass (including in-flight first attempts) before retries.
        first_pass = any(self.adapter.attempts(qid) == 0 for qid in eligible)
        first_pass = first_pass or any(job['ordinal'] == 1 for job in self.active.values())
        for qid in eligible:
            if first_pass and self.adapter.attempts(qid) != 0:
                continue
            if self.routes.lane(qid) in (None, lane):
                return qid
        return None

    def launch(self, lane, qid):
        if self.stopping or lane in self.active or self.candidate(lane) != qid:
            raise ValueError('Duplicate, exhausted or reassigned dispatch')
        config.require_ready(self.routes.config)
        if not model_ready(self.lanes[lane]['api_base']):
            self.health[lane] = False
            self.unavailable_since.setdefault(lane, time.monotonic())
            self.next_health = 0
            return
        route = self.routes.assign(qid, lane)
        job = self.adapter.prepare(qid, lane, route, self.routes)
        job.update(qid=qid, lane=lane, started=time.monotonic(), timeout_at=None)
        job['output'] = job['log'].open('a', encoding='utf-8')
        try:
            job['process'] = subprocess.Popen(job['argv'], env=config.child_environment(self.lanes[lane], self.adapter.method),
                stdout=job['output'], stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(self.lock_fd,))
        except OSError:
            job['output'].close()
            self.adapter.finish(job, 127, 127)
            return
        self.active[lane] = job

    @staticmethod
    def signal_group(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def reap(self, now):
        failure = None
        for lane, job in list(self.active.items()):
            process = job['process']
            code = process.poll()
            if code is None and job['timeout_at'] is None and now - job['started'] >= config.HISTORY_TIMEOUT:
                job['timeout_at'] = now
                self.signal_group(process, signal.SIGTERM)
            if job['timeout_at'] is not None and (code is not None or now - job['timeout_at'] >= 10):
                self.signal_group(process, signal.SIGKILL)
            if code is not None:
                process.wait()
                job['output'].close()
                try:
                    self.adapter.finish(job, 124 if job['timeout_at'] is not None else code, code)
                except Exception as error:
                    failure = failure or error
                finally:
                    del self.active[lane]
        if failure is not None:
            raise failure

    def publish(self, phase):
        self.adapter.publish(self.active, phase)
        config.atomic(config.STATE / (self.adapter.method + '_quad_status.json'), {
            'method': self.adapter.method, 'phase': phase, 'completed': len(self.adapter.completed),
            'inflight': {lane: {'question_id': job['qid'], 'ordinal': job['ordinal'],
                                'pid': job['process'].pid} for lane, job in self.active.items()},
            'disabled_lanes': sorted(self.disabled), 'deployment_sha256': config.DEPLOYMENT_SHA,
            'updated_at': time.time(), **self.routes.fresh})

    def execute(self):
        try:
            while True:
                self.reap(time.monotonic())
                if self.stopping:
                    break
                config.require_ready(self.routes.config)
                now = time.monotonic()
                if now >= self.next_health:
                    self.next_health = now + 30
                    for lane, binding in self.lanes.items():
                        self.health[lane] = lane not in self.disabled and model_ready(binding['api_base'])
                        if self.health[lane]:
                            self.unavailable_since.pop(lane, None)
                        else:
                            self.unavailable_since.setdefault(lane, now)
                            if now - self.unavailable_since[lane] >= 1800:
                                self.disabled.add(lane)
                if shutil.disk_usage(config.ROOT).free < 2 * 1024**3:
                    self.stopping = True
                    break
                for lane in self.lanes:
                    qid = self.candidate(lane)
                    if lane not in self.active and self.health[lane] and qid is not None:
                        self.launch(lane, qid)
                self.publish('running' if self.active else 'waiting_for_model')
                if not self.active and all(lane in self.disabled or self.candidate(lane) is None for lane in self.lanes):
                    break
                time.sleep(2)
        finally:
            self.stopping = True
            while self.active:
                try:
                    self.reap(time.monotonic())
                    self.publish('draining')
                except Exception as error:
                    try:
                        print(json.dumps({'status': 'drain_error', 'error_type': type(error).__name__}), flush=True)
                    except OSError:
                        pass
                if self.active:
                    time.sleep(2)
        self.publish('finished')
        return len(self.adapter.completed) == 500 and not self.adapter.failures
