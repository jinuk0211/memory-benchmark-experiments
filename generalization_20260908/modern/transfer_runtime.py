"""Modern vLLM initialization; all frozen generation/embedding/scoring methods reused."""

from functools import lru_cache
from pathlib import Path

import refine as core
from run_evidence_utility import Scorer as FrozenScorer
from evidence_utility import POLICY
from chat_tokenizer_compat import ListChatTokenizer

RUNTIME_FLAGS = {"language_model_only": True}


class Runtime(core.Runtime):
    def __init__(self, args, environment):
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from vllm import LLM

        torch.set_num_threads(8)
        self.args, self.np = args, np
        self.cache = Path(args.out) / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model_meta = environment["models"][args.model]
        self.embed_meta = environment["models"][args.embed_model]
        self.embed = SentenceTransformer(self.embed_meta["path"], device="cuda")
        self.llm = LLM(
            model=self.model_meta["path"], dtype="bfloat16", max_model_len=8192,
            gpu_memory_utilization=0.78, max_num_seqs=24, max_num_batched_tokens=8192,
            enable_prefix_caching=True, enforce_eager=True, seed=args.seed,
            **RUNTIME_FLAGS,
        )
        self.tok = ListChatTokenizer(self.llm.get_tokenizer())
        self.ntok = lru_cache(maxsize=100000)(
            lambda text: len(self.tok.encode(text, add_special_tokens=False))
        )


class Scorer(FrozenScorer):
    def __init__(self, args, environment, out):
        from vllm import LLM

        self.model = environment["models"][args.model]
        self.cache = out / "cache"
        self.llm = LLM(
            model=self.model["path"], dtype="bfloat16", max_model_len=8192,
            gpu_memory_utilization=0.78, max_num_seqs=8, max_num_batched_tokens=8192,
            enable_prefix_caching=False, enforce_eager=True, seed=POLICY["seed"],
            **RUNTIME_FLAGS,
        )
        self.tok = ListChatTokenizer(self.llm.get_tokenizer())
        self.ntok = lambda text: len(self.tok.encode(text, add_special_tokens=False))

