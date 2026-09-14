"""Independent CPU contract checks; never starts a GPU engine."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "seed_parent_ablation"
sys.path[:0] = [str(ROOT), str(ROOT / "source")]

import numpy as np  # noqa: E402
import evidence_utility as utility  # noqa: E402
from runtime_meter import Meter, MiniLMAdapter  # noqa: E402
from transfer_runtime import Runtime  # noqa: E402


class NativeContracts(unittest.TestCase):
    def test_actual_target_span_and_mean_logprob_direction(self) -> None:
        """Ignore prefix logprobs and retain token-mean rather than sum gain."""
        output = SimpleNamespace(
            prompt_token_ids=[900, 7, 8],
            prompt_logprobs=[None, {7: SimpleNamespace(logprob=-1.0)},
                              {8: SimpleNamespace(logprob=-3.0)}],
        )
        result = utility.extract_answer_logprob(output, 1, [7, 8])
        self.assertEqual(result["mean_logprob"], -2.0)
        self.assertEqual(result["sum_logprob"], -4.0)
        self.assertEqual(result["answer_tokens"], 2)
        self.assertGreater(result["mean_logprob"] - (-3.0), 0)

    def test_scoring_refuses_wrong_or_missing_actual_target(self) -> None:
        output = SimpleNamespace(prompt_token_ids=[1, 7], prompt_logprobs=[None, {}])
        with self.assertRaises(ValueError):
            utility.extract_answer_logprob(output, 1, [7])
        with self.assertRaises(ValueError):
            utility.extract_answer_logprob(output, 1, [8])
        output.prompt_logprobs[1] = {7: SimpleNamespace(logprob=float("nan"))}
        with self.assertRaises(ValueError):
            utility.extract_answer_logprob(output, 1, [7])

    def _runtime(self, directory: Path, fail: bool = False) -> tuple[Runtime, object]:
        class Encoder:
            def __init__(self) -> None:
                self.seen: list[list[str]] = []

            def forward(self, features: dict) -> np.ndarray:
                if fail:
                    raise RuntimeError("independent synthetic forward failure")
                return np.asarray([[1.0, 0.0]], dtype=np.float32)

            def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
                self.seen.append(list(texts))
                return self.forward({"input_ids": [[101, 42, 102]],
                                     "attention_mask": [[1, 1, 1]]})

        runtime = Runtime.__new__(Runtime)
        runtime.args = SimpleNamespace(embed_batch_size=32)
        runtime.np = np
        runtime.cache = directory / "cache"
        runtime.embed_meta = {"path": "pinned-synthetic", "revision": "test"}
        runtime.meter = Meter(directory, "independent_cpu", {"synthetic": True})
        encoder = Encoder()
        runtime.embed = MiniLMAdapter(encoder, runtime.meter)
        return runtime, encoder

    def test_cpu_embedding_prefix_cache_and_native_meter_are_consistent(self) -> None:
        """One actual forward, two operations; query text excludes Qwen prefix."""
        with tempfile.TemporaryDirectory(prefix="independent_", dir=Path(__file__).parent) as raw:
            directory = Path(raw)
            runtime, encoder = self._runtime(directory)
            # core.encode imports torch only for its OOM type; no real torch work.
            fake_torch = SimpleNamespace(OutOfMemoryError=MemoryError)
            with patch.dict(sys.modules, {"torch": fake_torch}):
                first = runtime.encode(["Where did Mira travel?"], query=True)
                second = runtime.encode(["Where did Mira travel?"], query=True)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(encoder.seen, [["Where did Mira travel?"]])
            self.assertIs(runtime.embed.query, False)
            self.assertIsNone(runtime.meter.active)
            self.assertIsNone(runtime.meter.attempt)
            receipts = [json.loads(path.read_text()) for path in directory.rglob("*.receipt.json")]
            actual = [row for row in receipts if row["kind"] == "embedding"]
            operations = [row for row in receipts if row["kind"] == "operation"]
            self.assertEqual(len(actual), 1)
            self.assertEqual(actual[0]["input_tokens"], 3)
            self.assertEqual(actual[0]["operation_kind"], "embedding")
            self.assertEqual(len(operations), 2)
            self.assertEqual(sorted(row["actual_invocations"]["embedding"] for row in operations), [0, 1])

    def test_failed_embedding_retains_error_receipt_and_restores_flags(self) -> None:
        with tempfile.TemporaryDirectory(prefix="independent_", dir=Path(__file__).parent) as raw:
            directory = Path(raw)
            runtime, _ = self._runtime(directory, fail=True)
            fake_torch = SimpleNamespace(OutOfMemoryError=MemoryError)
            with patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
                    runtime.encode(["Question"], query=True)
            self.assertFalse(runtime.embed.query)
            self.assertIsNone(runtime.meter.active)
            self.assertIsNone(runtime.meter.attempt)
            receipts = [json.loads(path.read_text()) for path in directory.rglob("*.receipt.json")]
            actual = [row for row in receipts if row["kind"] == "embedding"]
            self.assertEqual(len(actual), 1)
            self.assertEqual(actual[0]["status"], "error")
            self.assertEqual(actual[0]["input_tokens"], 3)
            self.assertEqual(runtime.meter.counts["embedding"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

