"""V100 self-stop guard: bounded setup and inference idle billing protection."""
import fcntl
import http.client
import json
import os
from pathlib import Path
import shutil
import time

INSTANCE = '50558359'
ROOT = Path('/workspace/longmemeval_s_native7_20260910/queue/dual_gpu_20260911')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_save(path, state):
    try:
        save(path, state)
        return True
    except OSError as error:
        state.setdefault('stop_reason', 'state_write_failed_preserve_results')
        state.update(persistence_error=type(error).__name__, persistence='unavailable')
        try:
            print(json.dumps({'status': 'state_write_failed_stop_required', 'error_type': type(error).__name__}), flush=True)
        except OSError:
            pass
        return False

def provider(stop=False):
    if os.environ.get('CONTAINER_ID') != INSTANCE or not os.environ.get('CONTAINER_API_KEY'):
        raise ValueError('Wrong instance or missing scoped credential')
    connection = http.client.HTTPSConnection('console.vast.ai', timeout=20)
    try:
        connection.request('PUT' if stop else 'GET', '/api/v0/instances/' + INSTANCE + '/',
                           body=b'{"state":"stopped"}' if stop else None,
                           headers={'Authorization': 'Bearer ' + os.environ['CONTAINER_API_KEY'], 'Content-Type': 'application/json'})
        response = connection.getresponse()
        value = json.loads(response.read(262144))
        if response.status != 200:
            raise RuntimeError('ProviderHTTP' + str(response.status))
        if stop:
            if value.get('success') is not True:
                raise RuntimeError('StopNotAccepted')
            return {'accepted': True}
        row = value['instances']
        if str(row['id']) != INSTANCE:
            raise ValueError('Provider instance mismatch')
        return {k: row.get(k) for k in ('id', 'actual_status', 'intended_status', 'cur_state')}
    finally:
        connection.close()


def decision(state, now, activated, last_success, free_bytes, finished):
    if finished:
        return 'coordinator_requested_stop_after_result_preservation'
    if free_bytes < 1024**3:
        return 'low_disk_preserve_results'
    if now - state['started_at'] >= 72 * 3600:
        return 'maximum_72_hour_budget'
    if not activated and now - state['started_at'] >= 3600:
        return 'setup_not_verified_within_60_minutes'
    if activated and now - max(state.get('activated_at', now), last_success) >= 1800:
        return 'no_successful_inference_30_minutes'
    return None


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'guard.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = ROOT / 'v100_guard_state.json'
    state = json.loads(path.read_text()) if path.exists() else {'instance_id': INSTANCE, 'started_at': time.time(), 'put_attempts': 0}
    if state['instance_id'] != INSTANCE:
        raise ValueError('State instance mismatch')
    while True:
        now = time.time()
        try:
            active = (ROOT / 'inference_verified.json').is_file()
            if active:
                state.setdefault('activated_at', now)
            activity = ROOT / 'inference_activity.json'
            last = json.loads(activity.read_text()).get('last_success', 0) if activity.exists() else 0
            request = ROOT / 'stop_requested.json'
            finished = request.is_file() and str(json.loads(request.read_text()).get('instance_id')) == INSTANCE
            free = shutil.disk_usage(ROOT).free
            reason = state.get('stop_reason') or decision(state, now, active, last, free, finished)
            state.update(observed_at=now, status='armed', activated=active, last_success=last, free_bytes=free)
            state.pop('observer_error_since', None)
        except Exception as error:
            state.setdefault('observer_error_since', now)
            state.update(status='observer_error', error_type=type(error).__name__)
            reason = state.get('stop_reason') or ('observer_unavailable_30_minutes' if now-state['observer_error_since'] >= 1800 else None)
        if reason:
            state.update(stop_reason=reason, status='stop_intent_persisted')
            safe_save(path, state)
            try:
                current = provider()
                state['provider'] = current
                if current['actual_status'] == 'stopped':
                    state['status'] = 'observed_stopped'
                    safe_save(path, state)
                    return
                if current.get('intended_status') != 'stopped' and current.get('cur_state') != 'stopped':
                    state['put_attempts'] += 1
                    safe_save(path, state)
                    state['stop_ack'] = provider(True)
                state['status'] = 'stop_requested_not_yet_observed'
            except Exception as error:
                state.update(status='stop_retry_pending', error_type=type(error).__name__)
        safe_save(path, state)
        time.sleep(30)


if __name__ == '__main__':
    main()