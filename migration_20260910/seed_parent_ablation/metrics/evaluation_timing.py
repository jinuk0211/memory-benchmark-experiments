"""Sidecar timing of an unchanged evaluate_sample call; no model dependencies."""
from collections.abc import Callable
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid


def canonical_bytes(value: Any) -> bytes:
    """Same canonical JSON digest convention as source/refine.py, without newline."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


class TimingRecorder:
    """Record each call in a fresh directory, preserving partial attempts.

    I/O and digest work are outside the timed interval. Failure to start or
    finalize a receipt raises; an error while recording an original exception
    never replaces that exception. Existing evaluation caches are untouched.
    """

    def __init__(self, sidecar_root: Path, metadata: dict[str, Any]) -> None:
        self.session_id = uuid.uuid4().hex
        self.root = Path(sidecar_root) / self.session_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.metadata = json.loads(canonical_bytes(metadata))

    @staticmethod
    def _write(path: Path, record: dict[str, Any]) -> str:
        body = canonical_bytes(record) + b'\n'
        temporary = path.with_suffix('.tmp')
        with temporary.open('xb') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        return hashlib.sha256(body).hexdigest()

    def wrap(self, original: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(original)

        @functools.wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            rt, sample = arguments.arguments['rt'], arguments.arguments['sample']
            runtime_args = rt.args
            invocation_id = uuid.uuid4().hex
            directory = self.root / invocation_id
            directory.mkdir(exist_ok=False)
            started = {
                'schema_version': 1, 'status': 'started', 'session_id': self.session_id,
                'invocation_id': invocation_id, 'method': arguments.arguments['method'],
                'dataset': arguments.arguments['dataset'], 'conversation_id': str(sample['sample_id']),
                'metadata': self.metadata,
                'runtime': {
                    'out': str(runtime_args.out), 'model': getattr(runtime_args, 'model', None),
                    'model_metadata': getattr(rt, 'model_meta', None),
                    'embed_model': getattr(runtime_args, 'embed_model', None),
                    'embedding_metadata': getattr(rt, 'embed_meta', None),
                    'seed': getattr(runtime_args, 'seed', None),
                },
                'cache_lifecycle': self.metadata.get('cache_lifecycle', 'existing_cache_state_unchanged'),
                'timing_scope': 'Only original evaluate_sample call, including its existing cache reads and all reader work.',
                'clock': 'time.perf_counter_ns',
            }
            started_sha256 = self._write(directory / 'started.json', started)
            began = time.perf_counter_ns()
            try:
                rows = original(*args, **kwargs)
            except BaseException as exc:
                ended = time.perf_counter_ns()
                error = {**started, 'status': 'error', 'started_sha256': started_sha256,
                         'elapsed_seconds': (ended - began) / 1e9,
                         'exception_type': type(exc).__name__}
                try:
                    self._write(directory / 'error.json', error)
                except BaseException as logging_error:
                    BaseException.add_note(
                        exc, f'Timing error receipt write failed ({type(logging_error).__name__}); '
                        'started receipt remains incomplete')
                raise
            ended = time.perf_counter_ns()
            completed = {**started, 'status': 'complete', 'started_sha256': started_sha256,
                         'elapsed_seconds': (ended - began) / 1e9, 'question_count': len(rows),
                         'question_ids': [row['question_id'] for row in rows],
                         'returned_rows_sha256': hashlib.sha256(canonical_bytes(rows)).hexdigest(),
                         'returned_rows_digest_format': 'UTF-8 canonical JSON, sort_keys=True, ensure_ascii=False, no newline'}
            self._write(directory / 'completed.json', completed)
            return rows

        return measured
