"""Model parameterization of the frozen vLLM runtime; algorithm functions are reused."""

import refine as core
from run_evidence_utility import Scorer as FrozenScorer
from evidence_utility import POLICY


Runtime = core.Runtime


class Scorer(FrozenScorer):
    def __init__(self, args, environment, out):
        from vllm import LLM

        self.model = environment['models'][args.model]
        self.cache = out / 'cache'
        self.llm = LLM(model=self.model['path'], dtype='bfloat16', max_model_len=8192,
                       gpu_memory_utilization=0.78, max_num_seqs=8, max_num_batched_tokens=8192,
                       enable_prefix_caching=False, enforce_eager=True, seed=POLICY['seed'])
        self.tok = self.llm.get_tokenizer()
        self.ntok = lambda s: len(self.tok.encode(s, add_special_tokens=False))


