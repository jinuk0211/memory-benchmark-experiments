"""Durable, instance-local observation of native calls; never changes their results."""
import dataclasses
import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

_CONTEXT = {'phase': 'unattributed'}


def jsonable(value: object) -> Any:
    """Retain native structs/logprobs without lossy repr fallbacks."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'nonfinite_float': str(value)}
    if isinstance(value, (Path, Enum)):
        return str(value) if isinstance(value, Path) else jsonable(value.value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [jsonable(v) for v in value]
        return {'native_type': type(value).__name__,
                'items': sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False))}
    if hasattr(value, 'detach'):
        return jsonable(value.detach().cpu().tolist())
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if dataclasses.is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    fields = getattr(value, '__struct_fields__', None)
    if fields is not None:
        return {name: jsonable(getattr(value, name)) for name in fields}
    if hasattr(value, '__dict__'):
        return jsonable(vars(value))
    raise ValueError(f'Unsupported native logging value: {type(value).__name__}')


def set_runtime_context(phase: str, **metadata: Any) -> None:
    global _CONTEXT
    if not isinstance(phase, str) or not phase:
        raise ValueError('Runtime phase must be nonempty')
    _CONTEXT = jsonable({'phase': phase, **metadata})


def token_count(ids: object) -> int | None:
    return len(ids) if isinstance(ids, list) and all(type(n) is int and n >= 0 for n in ids) else None


def generation_usage(outputs: object, request_count: int) -> dict:
    rows = outputs if isinstance(outputs, list) else []
    prompts, completions, finishes, count = [], [], [], 0
    for row in rows:
        prompts.append(token_count(getattr(row, 'prompt_token_ids', None)))
        candidates = getattr(row, 'outputs', None)
        candidates = candidates if isinstance(candidates, list) else []
        count += len(candidates)
        values = [token_count(getattr(c, 'token_ids', None)) for c in candidates]
        completions.append(sum(values) if values and all(n is not None for n in values) else None)
        finishes.extend(getattr(c, 'finish_reason', None) for c in candidates)
    prompt = sum(prompts) if prompts and all(n is not None for n in prompts) else None
    completion = sum(completions) if completions and all(n is not None for n in completions) else None
    complete = (prompt is not None and completion is not None and len(rows) == request_count
                and all(getattr(row, 'finished', None) is True for row in rows))
    return {'prompt_tokens': prompt if complete else None, 'completion_tokens': completion if complete else None,
            'recorded_prompt_tokens': sum(n for n in prompts if n is not None),
            'recorded_completion_tokens': sum(n for n in completions if n is not None),
            'total_tokens': prompt + completion if complete else None, 'request_count': request_count,
            'response_count': len(rows), 'generation_count': count, 'finish_reasons': finishes,
            'usage_complete': complete}


def embedding_usage(features: dict) -> dict:
    masks, ids = jsonable(features['attention_mask']), jsonable(features['input_ids'])
    if not isinstance(masks, list) or not masks or not all(isinstance(row, list) for row in masks):
        raise ValueError('Unsupported native embedding attention mask')
    width = len(masks[0])
    if (not width or width > 256 or any(len(row) != width or any(type(n) is not int or n not in (0, 1) for n in row) for row in masks)
            or not isinstance(ids, list) or len(ids) != len(masks)
            or any(not isinstance(row, list) or len(row) != width or token_count(row) is None for row in ids)):
        raise ValueError('Invalid native MiniLM input IDs/masks/window')
    counts = [sum(row) for row in masks]
    return {'input_ids': ids, 'attention_mask': masks, 'shape': [len(masks), width],
            'tokens_per_input': counts, 'input_tokens': sum(counts), 'padded_tokens': len(masks) * width,
            'usage_complete': True, 'measure': 'Actual attempted native post-256 attention-mask tokens, including failed forwards.'}


class Meter:
    def __init__(self, out: Path, stage: str, profile: dict) -> None:
        if not isinstance(stage, str) or not stage or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in stage):
            raise ValueError('Invalid runtime stage')
        self.root = Path(out) / 'runtime' / stage / str(os.getpid()) / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.profile, self.sequence, self.active, self.attempt = profile, 0, None, None
        self.counts = {'generation': 0, 'embedding': 0}

    def write(self, path: Path, value: Any) -> str:
        body = (json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_suffix('.tmp')
        with temp.open('xb') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        return hashlib.sha256(body).hexdigest()

    def record(self, kind: str, payload: dict, call: Callable[[], Any], initial: dict | None = None,
               summarize: Callable[[Any], dict] | None = None, raw: bool = False) -> Any:
        self.sequence += 1
        seq, started = self.sequence, time.perf_counter()
        stem = self.root / kind / f'{seq:08d}'
        receipt = {'schema_version': 1, 'kind': kind, 'sequence': seq, 'profile': self.profile,
                   'context': jsonable(_CONTEXT), 'operation_id': self.active['id'] if self.active else None,
                   'operation_kind': self.active['kind'] if self.active else None, 'encode_attempt_id': self.attempt,
                   'status': 'error', **(initial or {})}
        try:
            receipt['request_sha256'] = self.write(stem.with_suffix('.request.json'), {**receipt, 'payload': payload})
            if kind in self.counts:
                self.counts[kind] += 1
            call_started = time.perf_counter()
            try:
                result = call()
            finally:
                receipt['call_duration_s'] = time.perf_counter() - call_started
                if kind in self.counts:
                    receipt['inference_duration_s'] = receipt['call_duration_s']
            if summarize:
                receipt.update(summarize(result))
            if raw:
                receipt['response_sha256'] = self.write(stem.with_suffix('.response.json'), result)
            receipt['status'] = 'success'
            return result
        except BaseException as exc:
            receipt.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            receipt['duration_s'] = time.perf_counter() - started
            if kind == 'operation':
                observed = {key: self.counts[key] - payload['counts_before'][key] for key in self.counts}
                receipt.update(actual_invocations=observed, zero_invocations_observed=not any(observed.values()), cache_hit_proven=False)
            self.write(stem.with_suffix('.receipt.json'), receipt)

    def operation(self, kind: str, inputs: Any, call: Callable[[], Any]) -> Any:
        previous = self.active
        self.active = {'id': uuid.uuid4().hex, 'kind': kind}
        try:
            return self.record('operation', {'inputs': inputs, 'counts_before': self.counts.copy()}, call)
        finally:
            self.active = previous


class MeteredLLM:
    def __init__(self, engine: Any, meter: Meter) -> None:
        self.engine, self.meter = engine, meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        prompts = args[0] if args else kwargs.get('prompts')
        if not isinstance(prompts, list):
            raise TypeError('Unsupported native batched prompt contract')
        return self.meter.record('generation', {'args': args, 'kwargs': kwargs},
            lambda: self.engine.generate(*args, **kwargs),
            initial={'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None, 'usage_complete': False},
            summarize=lambda result: generation_usage(result, len(prompts)), raw=True)


class MiniLMAdapter:
    PREFIX = 'Instruct: Retrieve relevant conversation memories to answer the question.\nQuery: '

    def __init__(self, encoder: Any, meter: Meter) -> None:
        self.encoder, self.meter, self.query = encoder, meter, False
        original = encoder.forward
        def observed_forward(features: dict, *args: Any, **kwargs: Any) -> Any:
            measured = embedding_usage(features)
            return meter.record('embedding', {'features': features, 'args': args, 'kwargs': kwargs},
                lambda: original(features, *args, **kwargs), initial=measured)
        # ST6 encode invokes self.forward directly; torch register_forward_hook is insufficient.
        encoder.forward = observed_forward

    def encode(self, texts: list[str], *args: Any, **kwargs: Any) -> Any:
        prepared = texts
        if self.query:
            if not all(isinstance(text, str) and text.startswith(self.PREFIX) for text in texts):
                raise ValueError('Inherited Qwen query-prefix contract changed')
            prepared = [text[len(self.PREFIX):] for text in texts]
        previous = self.meter.attempt
        self.meter.attempt = uuid.uuid4().hex
        try:
            return self.meter.record('encode_attempt', {'texts': prepared, 'query': self.query, 'args': args, 'kwargs': kwargs},
                lambda: self.encoder.encode(prepared, *args, **kwargs))
        finally:
            self.meter.attempt = previous
