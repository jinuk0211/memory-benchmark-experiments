"""CPU tests for the modern initializer flags and frozen method inheritance."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import run_transfer
import run_reader_transfer
import transfer_runtime


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


class ModernRuntimeTests(unittest.TestCase):
    def test_generation_embedding_and_likelihood_methods_are_exactly_inherited(self) -> None:
        core = run_transfer.core
        self.assertIs(transfer_runtime.Runtime.generate, core.Runtime.generate)
        self.assertIs(transfer_runtime.Runtime.encode, core.Runtime.encode)
        self.assertIs(run_reader_transfer.ReaderRuntime.generate, core.Runtime.generate)
        self.assertIs(transfer_runtime.Scorer.score, transfer_runtime.FrozenScorer.score)
        self.assertIs(transfer_runtime.Scorer.generate, transfer_runtime.FrozenScorer.generate)

    def test_writer_scorer_reader_initializers_apply_language_model_only(self) -> None:
        calls = []

        def fake_llm(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(get_tokenizer=lambda: FakeTokenizer())

        fake_torch = ModuleType("torch")
        fake_torch.set_num_threads = Mock()
        fake_sentence_transformers = ModuleType("sentence_transformers")
        fake_sentence_transformers.SentenceTransformer = Mock(return_value=object())
        fake_vllm = ModuleType("vllm")
        fake_vllm.LLM = fake_llm
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "torch": fake_torch,
            "sentence_transformers": fake_sentence_transformers,
            "vllm": fake_vllm,
        }):
            args = SimpleNamespace(
                model="Qwen/Qwen3.5-9B", embed_model="Qwen/Qwen3-Embedding-0.6B",
                out=Path(directory), seed=20260907, embed_batch_size=4,
            )
            environment = {"models": {
                args.model: {"path": "/fake/pinned/language-model", "revision": "language-revision"},
                args.embed_model: {"path": "/fake/pinned/embed-model", "revision": "embed-revision"},
            }}
            writer = transfer_runtime.Runtime(args, environment)
            scorer = transfer_runtime.Scorer(args, environment, args.out)
            reader = run_reader_transfer.ReaderRuntime(args, environment)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call["language_model_only"] is True for call in calls))
            self.assertTrue(all(call["max_model_len"] == 8192 for call in calls))
            self.assertEqual([item.ntok("two words") for item in (writer, scorer, reader)], [2, 2, 2])
            fake_sentence_transformers.SentenceTransformer.assert_called_once_with(
                "/fake/pinned/embed-model", device="cuda"
            )
        self.assertEqual(transfer_runtime.RUNTIME_FLAGS, {"language_model_only": True})
        self.assertIs(run_transfer.RUNTIME_FLAGS, transfer_runtime.RUNTIME_FLAGS)
        self.assertIs(run_reader_transfer.RUNTIME_FLAGS, transfer_runtime.RUNTIME_FLAGS)


if __name__ == "__main__":
    unittest.main()

