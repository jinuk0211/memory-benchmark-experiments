"""Opt-in pinned comparison execution and durable, non-caching native accounting."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import threading
import time
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any

from .config import Config

MODEL_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
EMBED_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    raise TypeError(f'Unsupported accounting value: {type(value).__name__}')


def _encoded(value: Any) -> bytes:
    return (json.dumps(_safe(value), ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def _fields(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: _safe(getattr(value, name, None)) for name in names}


def _token_count(ids: Any) -> int | None:
    if not isinstance(ids, (list, tuple)) or any(type(item) is not int or item < 0 for item in ids):
        return None
    return len(ids)


def comparison_config(cfg: Config, out_dir: str | Path) -> Config:
    """Use a fresh telemetry directory; automatic resume/replay is not supported."""
    configured = replace(
        cfg, model='Qwen/Qwen3.5-9B', model_revision=MODEL_REVISION,
        embed_model='sentence-transformers/all-MiniLM-L6-v2', embed_revision=EMBED_REVISION,
        dtype='half', embedding_device='cpu', max_model_len=49152,
        gpu_memory_utilization=0.92, max_num_seqs=48, max_num_batched_tokens=8192,
        comparison_enabled=True, comparison_output_dir=str(out_dir), out_dir=str(out_dir))
    configured._comparison_runtime = NativeRuntime(configured)
    return configured


class NativeRuntime:
    """One writer per engine, unique records for every real invocation, never a cache."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Path(cfg.comparison_output_dir) / 'runtime'
        self.root.mkdir(parents=True, exist_ok=False)
        for kind in ('generation', 'embedding'):
            (self.root / kind).mkdir()
        self.context: dict[str, Any] = {}
        self._sequence = {'generation': 0, 'embedding': 0}
        self._lock = threading.Lock()

    def set_context(self, **metadata: Any) -> None:
        _encoded(metadata)  # Fail before inference if the context cannot be persisted.
        self.context = copy.deepcopy(metadata)

    def _write(self, kind: str, sequence: int, suffix: str, data: Any) -> str:
        payload = _encoded(data)
        path = self.root / kind / f'{sequence:08d}.{suffix}.json'
        temporary = path.with_suffix('.json.partial')
        if path.exists():
            raise FileExistsError(path)
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != 'nt':
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return hashlib.sha256(payload).hexdigest()

    def _begin(self, kind: str, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._sequence[kind] += 1
            sequence = self._sequence[kind]
        request['context'] = copy.deepcopy(self.context)
        request_hash = hashlib.sha256(_encoded(request)).hexdigest()
        request_sha = self._write(kind, sequence, 'request', request)
        return {'type': f'{kind}_batch' if kind == 'generation' else 'embedding_forward',
                'batch_sequence': sequence, 'context': request['context'],
                'request_hash': request_hash, 'request_sha256': request_sha}

    def generate(self, llm: Any, tokenizer: Any, prompts: list[str], params: Any) -> Any:
        sampling = _fields(params, (
            'temperature', 'max_tokens', 'min_tokens', 'top_p', 'top_k', 'min_p', 'seed',
            'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'stop',
            'stop_token_ids', 'ignore_eos', 'skip_special_tokens', 'detokenize', 'n',
            'truncate_prompt_tokens', 'include_stop_str_in_output'))
        row = self._begin('generation', {'model': self.cfg.model,
            'model_revision': self.cfg.model_revision, 'prompts': prompts, 'sampling': sampling})
        row.update(status='error', usage_complete=False, prompt_tokens=None,
                   completion_tokens=None, total_tokens=None, request_count=len(prompts))
        started = time.perf_counter()
        try:
            counts = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
            row['preflight_prompt_tokens'] = counts
            if any(count + params.max_tokens > self.cfg.max_model_len for count in counts):
                row['status'] = 'rejected_preflight'
                raise ValueError('Native prompt plus output budget exceeds comparison context limit')
            inference_started = time.perf_counter()
            outs = llm.generate(prompts, params, use_tqdm=False)
            row['inference_duration_s'] = time.perf_counter() - inference_started
            raw = []
            for output in outs:
                item = _fields(output, ('request_id', 'prompt', 'prompt_token_ids', 'finished',
                                        'num_cached_tokens'))
                item['outputs'] = [_fields(candidate, (
                    'index', 'text', 'token_ids', 'finish_reason', 'stop_reason', 'cumulative_logprob'))
                    for candidate in output.outputs]
                item['metrics'] = _fields(getattr(output, 'metrics', None), (
                    'arrival_time', 'first_scheduled_time', 'first_token_time',
                    'last_token_time', 'finished_time', 'time_in_queue'))
                raw.append(item)
            row['response_sha256'] = self._write('generation', row['batch_sequence'], 'response',
                                                {'responses': raw})
            prompt_counts = [_token_count(item['prompt_token_ids']) for item in raw]
            candidates = [candidate for item in raw for candidate in item['outputs']]
            completion_counts = [_token_count(item['token_ids']) for item in candidates]
            row['response_count'], row['generation_count'] = len(raw), len(candidates)
            row['finish_reasons'] = [item['finish_reason'] for item in candidates]
            row['prompt_tokens'] = sum(prompt_counts) if all(n is not None for n in prompt_counts) else None
            row['completion_tokens'] = sum(completion_counts) if all(n is not None for n in completion_counts) else None
            complete = (len(raw) == len(prompts) and all(item['outputs'] for item in raw)
                        and row['prompt_tokens'] is not None and row['completion_tokens'] is not None)
            if not complete:
                raise ValueError('Missing actual token IDs or incomplete native response batch')
            row.update(status='success', usage_complete=True,
                       total_tokens=row['prompt_tokens'] + row['completion_tokens'])
            return outs
        except Exception as exc:
            row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            row['duration_s'] = time.perf_counter() - started
            self._write('generation', row['batch_sequence'], 'receipt', row)

    def instrument_embedding(self, embedder: Any) -> None:
        model = embedder[0].auto_model
        original_forward = model.forward
        if getattr(original_forward, '_certmem_runtime', None) is not None:
            raise ValueError('Encoder forward is already instrumented')
        signature = inspect.signature(original_forward)

        @wraps(original_forward)
        def metered_forward(*args: Any, **kwargs: Any) -> Any:
            # ST 6 calls .forward directly, bypassing torch Module forward hooks.
            mask = signature.bind(*args, **kwargs).arguments.get('attention_mask')
            if mask is None:
                raise ValueError('Actual encoder attention_mask required for comparison accounting')
            counts = {'input_tokens': int(mask.sum().item()), 'padded_tokens': int(mask.numel()),
                      'tokens_per_input': mask.sum(dim=1).tolist(), 'shape': list(mask.shape)}
            row = self._begin('embedding', {
                'model': self.cfg.embed_model, 'model_revision': self.cfg.embed_revision,
                'native_max_seq_length': embedder.max_seq_length, **counts})
            row.update(counts)
            row.update(status='error', usage_complete=False)
            started = time.perf_counter()
            try:
                output = original_forward(*args, **kwargs)
                row.update(status='success' if output is not None else 'error',
                           usage_complete=output is not None)
                return output
            except Exception as exc:
                row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                raise
            finally:
                row['duration_s'] = time.perf_counter() - started
                self._write('embedding', row['batch_sequence'], 'receipt', row)

        metered_forward._certmem_runtime = self
        model.forward = metered_forward


def load_embedding(cfg: Config) -> Any:
    from sentence_transformers import SentenceTransformer

    if not cfg.comparison_enabled:
        return SentenceTransformer(cfg.embed_model, device='cuda')
    if getattr(cfg, '_comparison_runtime', None) is None:
        raise ValueError('Use comparison_config to create a fresh metered runtime')
    embedder = SentenceTransformer(cfg.embed_model, device=cfg.embedding_device,
                                  revision=cfg.embed_revision, local_files_only=True)
    if embedder.max_seq_length != 256:
        raise ValueError('Pinned MiniLM native max_seq_length must remain 256')
    cfg._comparison_runtime.instrument_embedding(embedder)
    return embedder
