"""Compare cache-only replay with the actual frozen evaluate_sample function."""
import ast
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import tempfile
import time
from types import MethodType, SimpleNamespace as NS
import unittest

from replay_qwen_reader import reference_digest, replay


ROOT = Path(__file__).resolve().parents[2] / "generalization_20260908"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ReaderReplayTests(unittest.TestCase):
    def test_matches_native_dense_sparse_packing_and_cached_generation(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            (cache / "embeddings").mkdir()
            (cache / "generations").mkdir()
            core = NS(**load_functions(
                ROOT / "source/refine.py", {"lexical", "pack", "generic_f1"},
                {"re": re, "Counter": Counter, "lru_cache": lru_cache}))
            core.digest = reference_digest
            core.READER = "fixture reader"
            protocol = dict(model={"revision": "fixture"}, embedding={"revision": "fixture"},
                            config=dict(seed=7, embed_batch_size=4), reader_prompt=core.READER)
            target = dict(question="alpha", question_date="2023/05/20", answer="answer",
                          question_type="multi-session")
            units = [
                dict(text="X" * 3000, index_text="alpha", sources=["D1:1"]),
                dict(text="tiny alpha", sources=["D1:2"]),
                dict(text="한국어\n\nsecond paragraph", sources=["D1:3"]),
            ]
            matrices = [
                ([unit.get("index_text", unit["text"]) for unit in units], False,
                 np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)),
                ([target["question"]], True, np.array([[1, 0]], dtype=np.float32)),
            ]
            for texts, is_query, values in matrices:
                key = reference_digest([
                    protocol["embedding"], texts, is_query, 4, "oom_backoff_v1"])
                np.save(cache / "embeddings" / f"{key}.npy", values)
            runtime = NS(cache=cache, model_meta=protocol["model"],
                         embed_meta=protocol["embedding"], args=NS(seed=7, embed_batch_size=4),
                         np=np, ntok=len)
            tree = ast.parse((ROOT / "source/refine.py").read_text(encoding="utf-8-sig"))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Runtime")
            encode = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "encode")
            scope = {"digest": reference_digest}
            exec(compile(ast.Module(body=[encode], type_ignores=[]), "native_encode", "exec"), scope)
            runtime.encode = MethodType(scope["encode"], runtime)

            def fixture_generation(system, users, max_tokens):
                for user in users:
                    key = reference_digest([runtime.model_meta, 7, system, user, max_tokens, False])
                    (cache / "generations" / f"{key}.json").write_text(json.dumps(
                        dict(text="answer", finish_reason="length", input_tokens=44, output_tokens=96)))
                return ["answer"] * len(users)

            runtime.generate = fixture_generation
            sample = dict(
                sample_id="q1", qa=[dict(target, category=target["question_type"], is_abstention=False)],
                evaluation_metadata=dict(session_map=[dict(session="session_1", is_answer_session=True)]),
            )
            scope = load_functions(ROOT / "full_transfer/run_transfer.py", {"evaluate_sample"},
                                   {"core": core, "time": time,
                                    "read": lambda path: json.loads(Path(path).read_bytes())})
            native = scope["evaluate_sample"](runtime, units, sample, "seed", "longmemeval")[0]
            result = replay(units, target, protocol, cache, len)
            for field, value in result.items():
                self.assertEqual(value, native[field], field)
            self.assertNotIn("X" * 3000, result["context"])
            before = {path: path.read_bytes() for path in cache.rglob("*") if path.is_file()}
            replay(units, target, protocol, cache, len)
            self.assertEqual(before, {path: path.read_bytes() for path in cache.rglob("*") if path.is_file()})
            next((cache / "generations").glob("*.json")).unlink()
            with self.assertRaises(FileNotFoundError):
                replay(units, target, protocol, cache, len)


if __name__ == "__main__":
    unittest.main()