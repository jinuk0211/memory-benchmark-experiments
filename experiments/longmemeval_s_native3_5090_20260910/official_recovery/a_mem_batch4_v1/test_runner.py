"""Actual pinned native classes with fake CPU embeddings/HTTP; no model or network calls."""

import argparse
import ast
import copy
from contextlib import nullcontext, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import runner as r


class Response:
    def __init__(self, value, finish="stop"):
        self.text = value if isinstance(value, str) else json.dumps(value)
        self.choices = [
            SimpleNamespace(
                finish_reason=finish, message=SimpleNamespace(content=self.text)
            )
        ]
        self.usage = {"prompt_tokens": 5, "completion_tokens": 7}

    def model_dump(self, **kwargs):
        return {
            "choices": [
                {
                    "finish_reason": self.choices[0].finish_reason,
                    "message": {"content": self.text},
                }
            ],
            "usage": self.usage,
        }


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.requests = []
        self.embedding_calls = []
        self.fenced_metadata = False
        self.invalid_evolution = False
        self.invalid_metadata = False
        self.metadata_transport_error = False
        self.unhandled_evolution = False
        self.blank_qa = False
        self.environment = patch.dict(
            os.environ,
            OPENAI_BASE_URL="http://127.0.0.1:18083/v1",
            OPENAI_API_KEY="EMPTY",
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        outer = self

        class FakeOpenAI:
            def __init__(self, api_key):
                if api_key != "EMPTY":
                    raise AssertionError("Non-placeholder credential")
                self.base_url = os.environ["OPENAI_BASE_URL"]
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self.create)
                )

            def create(self, **kwargs):
                outer.requests.append(kwargs)
                keys = set(
                    kwargs["response_format"]["json_schema"]["schema"]["properties"]
                )
                if keys == {"keywords", "context", "tags"}:
                    if outer.metadata_transport_error:
                        raise ConnectionError("synthetic local transport failure")
                    if outer.invalid_metadata:
                        return Response("{", "length")
                    value = {
                        "keywords": ["native"],
                        "context": "native metadata",
                        "tags": ["tag"],
                    }
                    return Response(
                        "```json\n" + json.dumps(value) + "\n```"
                        if outer.fenced_metadata
                        else value
                    )
                if "should_evolve" in keys:
                    if outer.unhandled_evolution:
                        return Response({"wrong_field": False})
                    if outer.invalid_evolution:
                        return Response("{", "length")
                    return Response(
                        {
                            "should_evolve": False,
                            "actions": [],
                            "suggested_connections": [],
                            "tags_to_update": [],
                            "new_context_neighborhood": [],
                            "new_tags_neighborhood": [],
                        }
                    )
                if keys == {"keywords"}:
                    return Response({"keywords": "rewritten-keywords"})
                return Response({"answer": " " if outer.blank_qa else "native answer"})

        class FakeEmbedding:
            max_seq_length = 256

            def __init__(self, name, device):
                outer.embedding_calls.append((name, device))

            def get_sentence_embedding_dimension(self):
                return 384

            def parameters(self):
                return [SimpleNamespace(dtype="torch.float32")]

            def encode(self, documents):
                return np.ones((len(documents), 384), dtype=np.float32)

            def get_config_dict(self):
                return {"model_name": "all-MiniLM-L6-v2"}

        modules = {}
        for name in (
            "openai",
            "sentence_transformers",
            "sklearn",
            "sklearn.metrics",
            "sklearn.metrics.pairwise",
        ):
            modules[name] = ModuleType(name)
        modules["openai"].OpenAI = FakeOpenAI
        modules["sentence_transformers"].SentenceTransformer = FakeEmbedding
        modules["sklearn.metrics.pairwise"].cosine_similarity = lambda q, e: np.ones(
            (len(q), len(e))
        )
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.calls = r.Calls(self.root / "calls.jsonl", "http://127.0.0.1:18083/v1")
        self.agent = r.make_agent(self.root / r.MINILM_REVISION, self.calls)

    def sessions(self, *counts):
        return [
            {
                "session_id": f"session_{index}",
                "date": "2023/05/20 (Sat) 13:00",
                "turns": [
                    {
                        "role": "user" if n % 2 == 0 else "assistant",
                        "content": f"session{index} turn{n}\nraw text",
                    }
                    for n in range(count)
                ],
            }
            for index, count in enumerate(counts)
        ]

    def prepare(self, source):
        embedding = self.root / r.MINILM_REVISION
        embedding.mkdir(exist_ok=True)
        for name in ("config.json", "modules.json", "sentence_bert_config.json"):
            r.save_json(embedding / name, {})
        runtime = {
            "method": "a_mem",
            "api_base": self.calls.endpoint,
            "model": r.MODEL,
            "embedding_model": str(embedding),
            "embedding_revision": r.MINILM_REVISION,
        }
        hashes = r.source_hashes()
        protocol = {
            "dataset_sha256": r.DATA_SHA256,
            "population_ids": ["q"] + [str(i) for i in range(499)],
            "runtime": runtime,
            "source_files_sha256": hashes,
            "policy": r.POLICY,
            "embedding_config_sha256": {
                name: r.sha(embedding / name)
                for name in ("config.json", "modules.json", "sentence_bert_config.json")
            },
        }
        run = self.root / ("run_" + str(len(list(self.root.glob("run_*")))))
        r.save_json(run / "protocol.json", protocol)
        query = {
            "question_id": "q",
            "question": "public question",
            "question_date": "D",
        }
        identity = {
            "protocol_sha256": r.digest(protocol),
            "source_sha256": r.digest(source),
            "query_sha256": r.digest(query),
        }
        attempt = r.next_attempt(
            run / "histories" / hashlib.sha256(b"q").hexdigest()[:24]
        )
        for name, value in (
            ("source.json", source),
            ("query.json", query),
            (
                "worker.json",
                {
                    "identity": identity,
                    "runtime": runtime,
                    "source_files_sha256": hashes,
                },
            ),
        ):
            r.save_json(attempt / name, value)
        (attempt / "console.log").write_text("mock parent owns closed console\n")
        return attempt, identity

    def execute_worker(self, source, *, finalize=True):
        attempt, identity = self.prepare(source)
        utils = ModuleType("utils")
        utils.__path__ = []
        metering = ModuleType("utils.request_metering")
        metering.install_request_metering = lambda: None
        metering.meter_operation = lambda *args, **kwargs: nullcontext()
        with (
            patch.dict(
                sys.modules, {"utils": utils, "utils.request_metering": metering}
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = r.worker(attempt)
        if result == 0 and finalize:
            r.finalize_attempt(attempt, identity)
        return result, attempt, identity

    def test_four_turn_notes_never_cross_sessions_and_reconstruct_raw_source(self):
        source = self.sessions(5, 0, 3, 8)
        source[0]["turns"][0]["content"] = (
            "x" * 5001 + "\nSpeaker assistantsays : literal"
        )
        plan = r.batch_plan(source)
        self.assertEqual(
            [
                (b["session_index"], b["turn_start"], b["turn_stop"])
                for b in plan["batches"]
            ],
            [(0, 0, 4), (0, 4, 5), (2, 0, 3), (3, 0, 4), (3, 4, 8)],
        )
        reconstructed = [[] for _ in source]
        for batch in plan["batches"]:
            turns = source[batch["session_index"]]["turns"][
                batch["turn_start"] : batch["turn_stop"]
            ]
            reconstructed[batch["session_index"]].extend(turns)
            self.assertEqual(batch["source_turns_sha256"], r.digest(turns))
            self.assertEqual(
                batch["session_date"], source[batch["session_index"]]["date"]
            )
            self.assertEqual(batch["native_time"], "202305201300")
        self.assertEqual(reconstructed, [session["turns"] for session in source])
        result, attempt, identity = self.execute_worker(source)
        self.assertEqual(
            result, 0, r.read_json(attempt / "failure.json") if result else None
        )
        self.assertEqual(len(r.read_json(attempt / "memory.json")), 5)
        self.assertEqual(
            len(self.requests), 12
        )  # Two unchanged writes per note, rewrite+QA.
        self.assertEqual(
            r.verified(attempt.parent, identity)["hypothesis"], "native answer"
        )
        self.assertTrue(
            all(
                x["model"] == r.MODEL
                and x["max_tokens"] == 1000
                and x["temperature"] == 0.7
                for x in self.requests
            )
        )

    def test_native_algorithm_prompts_embedding_and_qa_ast_are_unchanged(self):
        old = ast.parse((r.HERE.with_name("a_mem_paper_v2") / "runner.py").read_text())
        new = ast.parse((r.HERE / "runner.py").read_text())
        names = {
            "Calls",
            "native_fallbacks",
            "make_agent",
            "answer",
            "compile_classes",
            "local_endpoint",
        }

        def extract(tree):
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names
            }

        self.assertEqual(extract(old), extract(new))
        for name in r.read_json(r.HERE / "source_manifest.json")["files"]:
            self.assertEqual(
                (r.HERE / "upstream" / name).read_bytes(),
                (r.HERE.with_name("a_mem_paper_v2") / "upstream" / name).read_bytes(),
            )

    def test_actual_native_render_time_qa_and_retrieval_are_preserved(self):
        source = self.sessions(4)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                r.ingest_batches(self.agent, source, self.calls, self.root), 4
            )
        note = next(iter(self.agent.memory_system.memories.values()))
        self.assertEqual(
            note.content,
            "\n".join(
                "Speaker " + t["role"] + "says : " + t["content"]
                for t in source[0]["turns"]
            ),
        )
        self.assertEqual(note.timestamp, "202305201300")
        seen = []
        self.agent.retrieve_memory = lambda query, k: (
            seen.append((query, k)) or "evidence"
        )
        with redirect_stdout(io.StringIO()):
            answer, detail = r.answer(
                self.agent,
                {"question": "public question", "question_date": "D"},
                self.calls,
            )
        self.assertEqual(
            (answer, seen), ("native answer", [("rewritten-keywords", 10)])
        )
        self.assertIn("Question date: D", detail["prompt"])
        self.agent.memory_system.consolidate_memories()
        self.assertTrue(
            all(
                path == str(self.root / r.MINILM_REVISION) and device == "cpu"
                for path, device in self.embedding_calls
            )
        )

    def test_handled_fallbacks_count_per_batch_without_repair(self):
        self.invalid_metadata = self.invalid_evolution = True
        result, attempt, identity = self.execute_worker(self.sessions(9))
        self.assertEqual(result, 0)
        degradation = r.read_json(attempt / "native_degradation.json")
        self.assertEqual(
            (
                degradation["native_metadata_fallback_count"],
                degradation["native_evolution_fallback_count"],
            ),
            (3, 3),
        )
        self.assertEqual(len(self.requests), 8)
        self.assertEqual(
            r.verified(attempt.parent, identity)["native_handled_fallback_count"], 6
        )

    def test_transport_and_unhandled_errors_remain_failed_with_partial_evidence(self):
        for flag in ("metadata_transport_error", "unhandled_evolution", "blank_qa"):
            with self.subTest(flag=flag):
                setattr(self, flag, True)
                result, attempt, identity = self.execute_worker(self.sessions(5))
                self.assertEqual(result, 1)
                self.assertTrue((attempt / "failure.json").exists())
                self.assertTrue((attempt / "batch_manifest.json").exists())
                self.assertTrue((attempt / "llm_calls.jsonl").exists())
                self.assertFalse((attempt / "completion.json").exists())
                self.assertIsNone(r.verified(attempt.parent, identity))
                setattr(self, flag, False)

    def test_post_exit_finalization_and_exact_inventory_cache_rejection(self):
        result, attempt, identity = self.execute_worker(
            self.sessions(5), finalize=False
        )
        self.assertEqual(result, 0)
        self.assertFalse((attempt / "completion.json").exists())
        self.assertIsNone(r.verified(attempt.parent, identity))
        r.finalize_attempt(attempt, identity)
        self.assertTrue(
            r.read_json(attempt / "completion.json")["created_after_worker_exit"]
        )
        (attempt / "extra.bin").write_bytes(b"unsealed")
        with self.assertRaises(ValueError):
            r.verified(attempt.parent, identity)
        (attempt / "extra.bin").unlink()
        r.save_json(attempt / "failure.json", {"error": "later failure"})
        with self.assertRaises(ValueError):
            r.verified(attempt.parent, identity)

    def test_semantic_coverage_rejects_resealed_missing_duplicate_cross_session_and_changed_note(
        self,
    ):
        _, attempt, identity = self.execute_worker(self.sessions(5, 3))
        originals = {p.name: p.read_bytes() for p in attempt.iterdir() if p.is_file()}
        for kind in (
            "missing",
            "duplicate",
            "cross_session",
            "wrong_note",
            "wrong_time",
            "unfinished",
            "missing_source",
        ):
            with self.subTest(kind=kind):
                for name, raw in originals.items():
                    (attempt / name).write_bytes(raw)
                journal = [
                    json.loads(line)
                    for line in (attempt / "batch_calls.jsonl").read_text().splitlines()
                ]
                if kind == "missing":
                    journal.pop()
                if kind == "duplicate":
                    journal[1] = journal[0]
                if kind == "cross_session":
                    journal[0]["batch"]["turn_stop"] = 5
                if kind in ("missing", "duplicate", "cross_session"):
                    (attempt / "batch_calls.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in journal)
                    )
                if kind in ("wrong_note", "wrong_time"):
                    memory = r.read_json(attempt / "memory.json")
                    memory[journal[0]["note_id"]][
                        "content" if kind == "wrong_note" else "timestamp"
                    ] = "wrong"
                    r.save_json(attempt / "memory.json", memory)
                if kind == "unfinished":
                    r.save_json(
                        attempt / "construction_progress.json", {"status": "adding"}
                    )
                if kind == "missing_source":
                    r.save_json(attempt / "batch_manifest.json", {})
                receipt = r.read_json(attempt / "completion.json")
                receipt["files_sha256"] = r.attempt_files(attempt)
                r.save_json(attempt / "completion.json", receipt)
                with self.assertRaises(ValueError):
                    r.verified(attempt.parent, identity)

    def test_old_paper_version_and_changed_worker_or_protocol_never_reuse(self):
        _, attempt, identity = self.execute_worker(self.sessions(4))
        receipt = r.read_json(attempt / "completion.json")
        receipt["variant"] = "a_mem_paper_v2"
        r.save_json(attempt / "completion.json", receipt)
        with self.assertRaises(ValueError):
            r.verified(attempt.parent, identity)
        receipt["variant"] = r.VERSION
        r.save_json(attempt / "completion.json", receipt)
        worker = r.read_json(attempt / "worker.json")
        worker["runtime"] = {}
        r.save_json(attempt / "worker.json", worker)
        receipt["files_sha256"] = r.attempt_files(attempt)
        r.save_json(attempt / "completion.json", receipt)
        with self.assertRaises(ValueError):
            r.verified(attempt.parent, identity)
        protocol = r.read_json(attempt.parents[2] / "protocol.json")
        protocol["policy"]["harness_version"] = "a_mem_paper_v2"
        r.save_json(attempt.parents[2] / "protocol.json", protocol)
        with self.assertRaises(ValueError):
            r.verified(attempt.parent, identity)

    def test_canonical_500_batch_oracle_and_gold_labels_excluded(self):
        path = r.HARNESS / "longmemeval_s_cleaned.json"
        if not path.exists():
            path = (
                r.HARNESS.parents[1]
                / "MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json"
            )
        self.assertEqual(r.sha(path), r.DATA_SHA256)
        records = r.read_json(path)
        totals = [0, 0, 0, 0]
        for row in records:
            source = r.source_only(row)
            plan = r.batch_plan(source)
            totals[0] += plan["sessions"]
            totals[1] += plan["turns"]
            totals[2] += plan["batch_count"]
            totals[3] += sum(
                b["turn_stop"] - b["turn_start"] < 4 for b in plan["batches"]
            )
            if row["question_id"] == "e47becba":
                self.assertEqual(
                    (plan["sessions"], plan["turns"], plan["batch_count"]),
                    (53, 550, 146),
                )
        self.assertEqual((len(records), totals), (500, [23867, 246750, 65885, 8386]))
        labelled = {
            "haystack_sessions": [
                [{"role": "user", "content": "fact", "has_answer": True}]
            ],
            "haystack_dates": ["2023/05/20 (Sat) 13:00"],
            "haystack_session_ids": ["s"],
            "answer": "GOLD_SENTINEL",
            "question_type": "LABEL_SENTINEL",
        }
        serial = json.dumps(r.batch_plan(r.source_only(labelled)))
        self.assertNotIn("GOLD_SENTINEL", serial)
        self.assertNotIn("has_answer", serial)

    def test_full_population_status_and_stop_on_first_failure(self):
        records = [
            {
                "question_id": str(i),
                "question": "Q",
                "question_date": "D",
                "haystack_sessions": [[{"role": "user", "content": "fact"}]],
                "haystack_dates": ["2023/05/20 (Sat) 13:00"],
                "haystack_session_ids": ["s"],
            }
            for i in range(500)
        ]
        dataset = self.root / "dataset.json"
        r.save_json(dataset, records)
        embedding = self.root / r.MINILM_REVISION
        embedding.mkdir()
        for name in ("config.json", "modules.json", "sentence_bert_config.json"):
            r.save_json(embedding / name, {})
        args = argparse.Namespace(
            dataset=dataset,
            run_dir=self.root / "run",
            ids_file=None,
            embedding_model=embedding,
            api_base=self.calls.endpoint,
        )
        with (
            patch.object(r, "DATA_SHA256", r.sha(dataset)),
            patch.object(
                r.subprocess, "run", return_value=SimpleNamespace(returncode=7)
            ) as child,
        ):
            self.assertEqual(r.run(args), 1)
        self.assertEqual(child.call_count, 1)
        status = r.read_json(args.run_dir / "status.json")
        self.assertEqual(
            (
                status["planned"],
                status["generated"],
                status["failed"],
                status["status"],
            ),
            (500, 0, 1, "generation_incomplete"),
        )

    def test_boolean_numeric_artifacts_are_not_equal_to_source_ranges(self):
        _, attempt, identity = self.execute_worker(self.sessions(4))
        originals = {p.name: p.read_bytes() for p in attempt.iterdir() if p.is_file()}
        for filename in (
            "batch_manifest.json",
            "batch_calls.jsonl",
            "construction_progress.json",
        ):
            with self.subTest(filename=filename):
                for name, raw in originals.items():
                    (attempt / name).write_bytes(raw)
                path = attempt / filename
                if filename == "batch_manifest.json":
                    value = r.read_json(path)
                    value["batches"][0]["turn_start"] = False
                    r.save_json(path, value)
                elif filename == "batch_calls.jsonl":
                    value = json.loads(path.read_text())
                    value["batch"]["batch_index"] = False
                    path.write_text(json.dumps(value) + "\n")
                else:
                    value = r.read_json(path)
                    value["completed_batches"] = True
                    r.save_json(path, value)
                receipt = r.read_json(attempt / "completion.json")
                receipt["files_sha256"] = r.attempt_files(attempt)
                r.save_json(attempt / "completion.json", receipt)
                with self.assertRaises(ValueError):
                    r.verified(attempt.parent, identity)

    def test_finalizer_rejects_wrong_question_and_non500_protocol(self):
        _, attempt, identity = self.execute_worker(self.sessions(4), finalize=False)
        prediction = r.read_json(attempt / "prediction.json")
        prediction["question_id"] = "other"
        r.save_json(attempt / "prediction.json", prediction)
        with self.assertRaises(ValueError):
            r.finalize_attempt(attempt, identity)
        prediction["question_id"] = "q"
        r.save_json(attempt / "prediction.json", prediction)
        protocol_path = attempt.parents[2] / "protocol.json"
        original = r.read_json(protocol_path)
        for population in (["q"], ["q"] * 500, ["q"] + list(range(499))):
            with self.subTest(population_size=len(population)):
                protocol = copy.deepcopy(original)
                protocol["population_ids"] = population
                r.save_json(protocol_path, protocol)
                altered = dict(identity, protocol_sha256=r.digest(protocol))
                worker = r.read_json(attempt / "worker.json")
                worker["identity"] = altered
                r.save_json(attempt / "worker.json", worker)
                with self.assertRaises(ValueError):
                    r.validate_inputs(attempt, altered)

    def test_second_controller_is_rejected_before_dispatch_or_aggregate_write(self):
        fcntl = ModuleType("fcntl")
        fcntl.LOCK_EX, fcntl.LOCK_NB = 2, 4
        acquisitions = []

        def flock(fd, flags):
            self.assertEqual(flags, 6)
            acquisitions.append(fd)
            if len(acquisitions) > 1:
                raise BlockingIOError("another parent owns this run")

        fcntl.flock = flock
        command = [
            "runner.py",
            "--dataset",
            str(self.root / "data.json"),
            "--run-dir",
            str(self.root / "run"),
            "--api-base",
            self.calls.endpoint,
            "--embedding-model",
            str(self.root / r.MINILM_REVISION),
        ]

        def first_dispatch(args):
            with self.assertRaises(BlockingIOError):
                r.main()
            self.assertFalse((args.run_dir / "predictions.json").exists())
            return 0

        with (
            patch.dict(sys.modules, {"fcntl": fcntl}),
            patch.object(sys, "argv", command),
            patch.object(r, "run", side_effect=first_dispatch) as dispatch,
        ):
            self.assertEqual(r.main(), 0)
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(len(acquisitions), 2)


if __name__ == "__main__":
    unittest.main()
