"""Fixed-instance, server-local idle-cost protection. Never destroys data."""
import argparse
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/workspace/longmemeval_s_native7_20260910')
INSTANCE = '50468468'
STATE = ROOT / 'queue/native2_cost_guard/state.json'
QUEUES = ('native2-simplemem', 'native2-lightmem')
TERMINAL = {'EXITED', 'FATAL', 'STOPPED', 'MISSING'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def provider(stop=False):
    if os.environ.get('CONTAINER_ID') != INSTANCE or not os.environ.get('CONTAINER_API_KEY'):
        raise ValueError('Wrong instance or missing scoped API credential')
    connection = http.client.HTTPSConnection('console.vast.ai', timeout=20)
    try:
        connection.request('PUT' if stop else 'GET', f'/api/v0/instances/{INSTANCE}/',
            body=b'{"state":"stopped"}' if stop else None,
            headers={'Authorization': 'Bearer ' + os.environ['CONTAINER_API_KEY'], 'Content-Type': 'application/json'})
        response = connection.getresponse()
        payload = json.loads(response.read(262144))
        if response.status != 200:
            raise RuntimeError('ProviderHTTP' + str(response.status))
        if stop:
            if payload.get('success') is not True:
                raise RuntimeError('StopNotAccepted')
            return {'stop_accepted': True}
        row = payload['instances']
        if str(row['id']) != INSTANCE:
            raise ValueError('Provider instance mismatch')
        return {key: row.get(key) for key in ('id', 'actual_status', 'intended_status', 'cur_state')}
    finally:
        connection.close()


def observe(state, now):
    result = subprocess.run(['supervisorctl', 'status'], capture_output=True, text=True, timeout=15)
    services = {parts[0]: parts[1] for line in result.stdout.splitlines() if len(parts := line.split()) >= 2}
    if not services:
        raise RuntimeError('Supervisor inventory unavailable')
    generated = {}
    for method in ('simplemem', 'lightmem'):
        path = ROOT / 'fast_native2_20260911' / (method + '_status.json')
        generated[method] = json.loads(path.read_text()).get('generated', 0) if path.exists() else 0
    journal = ROOT / 'logs/chat_usage.jsonl'
    offset = state.get('journal_offset', journal.stat().st_size if journal.exists() else 0)
    successes = state.get('successes', 0)
    if journal.exists():
        with journal.open('rb') as stream:
            stream.seek(min(offset, journal.stat().st_size))
            for _ in range(10000):
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break
                offset = stream.tell()
                record = json.loads(line)
                if record.get('success') is True:
                    successes += 1
    semantic = sum(generated.values())
    if successes > state.get('successes', 0) or semantic > state.get('generated_total', 0):
        state['last_progress'] = now
    state.update(successes=successes, journal_offset=offset, generated_total=semantic)
    terminal = all(services.get(name, 'MISSING') in TERMINAL for name in QUEUES)
    if terminal:
        state.setdefault('terminal_since', now)
    else:
        state.pop('terminal_since', None)
    finish_path = ROOT / 'queue/fast_native2_finish/status.json'
    finish = json.loads(finish_path.read_text()).get('status') if finish_path.exists() else None
    return {'services': {key: services.get(key, 'MISSING') for key in (*QUEUES, 'native2-finish', 'native7-qwen', 'native7-minilm')},
            'generated': generated, 'all_terminal': terminal, 'finish': finish,
            'free_bytes': shutil.disk_usage(ROOT).free}


def decide(observation, state, now):
    if observation['free_bytes'] < 2 * 1024**3:
        return 'low_disk_preserve_partial_results'
    if now - state['started_at'] >= 72 * 3600:
        return 'maximum_72_hour_budget'
    if now - state['started_at'] < 600:
        return None
    if observation['all_terminal']:
        if observation['finish'] in {'verified_ready', 'stop_accepted'}:
            return None  # The stricter verified-completion finalizer owns normal STOP.
        if now - state.get('terminal_since', now) >= 600:
            return 'both_queues_terminal_without_verified_completion'
    if now - state['last_progress'] >= 1800:
        # Give full-result verification and compression up to one hour.
        if sum(observation['generated'].values()) == 1000 and now - state['last_progress'] < 3600:
            return None
        return 'no_successful_response_or_question_progress_30_minutes'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(provider()))
        return
    import fcntl
    STATE.parent.mkdir(parents=True, exist_ok=True)
    lock = (STATE.parent / 'guard.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    now = time.time()
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        'instance_id': INSTANCE, 'started_at': now, 'last_progress': now, 'put_attempts': 0}
    if state['instance_id'] != INSTANCE:
        raise ValueError('State instance mismatch')
    while True:
        now = time.time()
        try:
            observation = observe(state, now)
            state.update(observation=observation, observed_at=now, status='armed')
            state.pop('observation_error_since', None)
            reason = state.get('stop_reason') or decide(observation, state, now)
        except Exception as error:
            state.setdefault('observation_error_since', now)
            state.update(status='observer_error', error_type=type(error).__name__)
            reason = state.get('stop_reason')
            if now - state['observation_error_since'] >= 1800:
                reason = 'observer_unavailable_30_minutes'
        if reason:
            state.update(stop_reason=reason, status='stop_intent_persisted')
            save(STATE, state)
            try:
                remote = provider()
                state['provider'] = remote
                if remote['actual_status'] == 'stopped':
                    state['status'] = 'observed_stopped'
                    save(STATE, state)
                    return
                if remote.get('intended_status') != 'stopped' and remote.get('cur_state') != 'stopped':
                    state['put_attempts'] += 1
                    save(STATE, state)
                    state['stop_ack'] = provider(stop=True)
                state['status'] = 'stop_requested_not_yet_observed'
            except Exception as error:
                state.update(status='stop_retry_pending', error_type=type(error).__name__)
        save(STATE, state)
        print(json.dumps({key: state.get(key) for key in ('status', 'stop_reason', 'generated_total', 'put_attempts')}), flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
