"""Start the LightMem tuner once, only after a completed admission rollout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

Q = Path(__file__).resolve().parent
MAX_WAIT_SECONDS = 4 * 60 * 60
POLL_SECONDS = 15
FAILED_PHASES = {'rolled_back', 'rollback_failed', 'needs_attention'}


def save(path, state, status, **details):
    state.update(status=status, updated_at=time.time(), **details)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    print(json.dumps({'status': status, 'receipt': str(path)}), flush=True)


def watch(quad):
    receipt = quad / 'post_admission_lightmem.json'
    source = quad / 'admission_rollout.json'
    state = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
    if state.get('start_attempted') or state.get('status') not in (None, 'waiting'):
        return 0 if state.get('status') == 'tuner_started' else 1
    state.setdefault('started_at', time.time())
    state.setdefault('deadline_at', state['started_at'] + MAX_WAIT_SECONDS)
    remaining = min(MAX_WAIT_SECONDS, max(0, state['deadline_at'] - time.time()))
    deadline = time.monotonic() + remaining
    save(receipt, state, 'waiting', start_attempted=False)
    while time.monotonic() < deadline:
        try:
            raw = source.read_bytes()
            rollout = json.loads(raw)
            if not isinstance(rollout, dict):
                rollout = {}
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            rollout = {}
        phase = rollout.get('phase')
        if phase in FAILED_PHASES:
            save(receipt, state, 'admission_failed', admission_phase=phase)
            return 1
        if phase == 'complete' and rollout.get('queues_resumed') is True:
            # Persist the claim before invoking Supervisor; ambiguous starts are never retried.
            save(receipt, state, 'start_requested', start_attempted=True,
                 admission_phase=phase, admission_rollout_sha256=hashlib.sha256(raw).hexdigest(),
                 admission_identity=rollout.get('identity'))
            try:
                result = subprocess.run(['supervisorctl', 'start', 'quad-tune-lightmem'],
                                        capture_output=True, text=True, timeout=30, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                save(receipt, state, 'start_failed', error_class=type(error).__name__)
                return 1
            save(receipt, state, 'tuner_started' if result.returncode == 0 else 'start_failed',
                 supervisor_returncode=result.returncode)
            return 0 if result.returncode == 0 else 1
        time.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
    save(receipt, state, 'timed_out')
    return 1


def main():
    import fcntl
    with (Q / 'post_admission_lightmem.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return watch(Q)


if __name__ == '__main__':
    raise SystemExit(main())
