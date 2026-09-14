"""Isolated FP16/MiniLM adapter; native cache, generation and NLL algorithms reused."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import refine as core
from evidence_utility import POLICY
from run_evidence_utility import Scorer as FrozenScorer

from chat_tokenizer_compat import ListChatTokenizer
from runtime_meter import Meter, MeteredLLM, MiniLMAdapter, set_runtime_context

__all__ = ['Runtime', 'Scorer', 'set_runtime_context']

RUNTIME_FLAGS = {"language_model_only": True}
MODEL = 'Qwen/Qwen3.5-9B'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
EMBED_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def model_metadata(environment: dict, name: str, expected_name: str, revision: str) -> dict:
    if name != expected_name:
        raise ValueError('Unexpected comparison model')
    meta = environment['models'][name]
    if not isinstance(meta, dict) or meta.get('revision') != revision or not isinstance(meta.get('path'), str) or not meta['path']:
        raise ValueError('Missing/mismatched model revision metadata')
    return meta


class Runtime(core.Runtime):
    def __init__(self, args: Any, environment: dict) -> None:
        self.model_meta = model_metadata(environment, args.model, MODEL, REVISION)
        self.embed_meta = model_metadata(environment, args.embed_model, EMBED_MODEL, EMBED_REVISION)
        if args.seed != 20260907:
            raise ValueError('Native Runtime seed changed')
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from vllm import LLM

        torch.set_num_threads(8)
        self.args, self.np = args, np
        self.cache = Path(args.out) / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.meter = Meter(Path(args.out), args.stage, {'role': 'runtime', 'model': self.model_meta,
            'embedding': self.embed_meta, 'dtype': 'float16', 'quantization': None, 'seed': args.seed,
            'embedding_device': 'cpu', 'embedding_dtype': 'float32', 'native_max_seq_length': 256,
            'model_metadata_is_not_weight_integrity_proof': True})
        def load_encoder() -> Any:
            encoder = SentenceTransformer(self.embed_meta['path'], device='cpu').float()
            if encoder.max_seq_length != 256:
                raise ValueError('MiniLM native max_seq_length must be 256')
            return encoder
        encoder = self.meter.operation('embedding_initialization',
            {'path': self.embed_meta['path'], 'device': 'cpu', 'dtype': 'float32', 'native_max_seq_length': 256}, load_encoder)
        self.embed = MiniLMAdapter(encoder, self.meter)
        engine_config = dict(
            model=self.model_meta["path"], dtype="half", quantization=None, max_model_len=8192,
            gpu_memory_utilization=0.78, max_num_seqs=24, max_num_batched_tokens=8192,
            enable_prefix_caching=True, enforce_eager=True, seed=args.seed,
            **RUNTIME_FLAGS,
        )
        engine = self.meter.operation('llm_initialization', engine_config, lambda: LLM(**engine_config))
        self.llm = MeteredLLM(engine, self.meter)
        self.tok = ListChatTokenizer(self.llm.get_tokenizer())
        self.ntok = lru_cache(maxsize=100000)(
            lambda text: len(self.tok.encode(text, add_special_tokens=False))
        )

    def generate(self, system: str, users: list[str], max_tokens: int = 512) -> list[str | None]:
        return self.meter.operation('generation', {'system': system, 'users': users, 'max_tokens': max_tokens},
            lambda: super(Runtime, self).generate(system, users, max_tokens))

    def encode(self, texts: list[str], query: bool = False) -> Any:
        previous, self.embed.query = self.embed.query, query
        try:
            return self.meter.operation('embedding', {'texts': texts, 'query': query},
                lambda: super(Runtime, self).encode(texts, query))
        finally:
            self.embed.query = previous


class Scorer(FrozenScorer):
    def __init__(self, args: Any, environment: dict, out: Path) -> None:
        self.model = model_metadata(environment, args.model, MODEL, REVISION)
        from vllm import LLM

        self.cache = out / "cache"
        self.meter = Meter(out, args.stage, {'role': 'scorer', 'model': self.model, 'dtype': 'float16',
            'quantization': None, 'seed': POLICY['seed'], 'model_metadata_is_not_weight_integrity_proof': True})
        engine_config = dict(
            model=self.model["path"], dtype="half", quantization=None, max_model_len=8192,
            gpu_memory_utilization=0.78, max_num_seqs=8, max_num_batched_tokens=512, enable_chunked_prefill=True,
            enable_prefix_caching=False, enforce_eager=True, seed=POLICY["seed"],
            **RUNTIME_FLAGS,
        )
        engine = self.meter.operation('llm_initialization', engine_config, lambda: LLM(**engine_config))
        self.llm = MeteredLLM(engine, self.meter)
        self.tok = ListChatTokenizer(self.llm.get_tokenizer())
        self.ntok = lambda text: len(self.tok.encode(text, add_special_tokens=False))

    def score(self, jobs: list[dict]) -> None:
        return self.meter.operation('source_utility_nll', {'jobs': jobs}, lambda: super(Scorer, self).score(jobs))

    def generate(self, controls: dict) -> dict:
        return self.meter.operation('source_utility_generation', {'controls': controls},
            lambda: super(Scorer, self).generate(controls))
