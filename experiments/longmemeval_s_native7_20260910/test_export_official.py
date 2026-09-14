"""Synthetic exporter coverage; no real labels, model calls, or judge calls."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import export_official as exporter


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = [f"qid-{index}" for index in range(500)]
        self.dataset = self.root / "dataset.json"
        self.dataset.write_text(json.dumps([
            {"question_id": qid, "answer": "GOLD_MUST_NEVER_APPEAR", "question_type": "SECRET_TYPE"}
            for qid in self.ids]), encoding="utf-8")
        self.data_hash = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        self.vendor = self.root / "official"
        self.upstream = {}
        for name in exporter.UPSTREAM_HASHES:
            path = self.vendor / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic source; never executed.\n", encoding="utf-8")
            self.upstream[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.patches = [
            patch.object(exporter, "DATA_SHA256", self.data_hash),
            patch.object(exporter, "UPSTREAM_HASHES", self.upstream),
        ]
        for active in self.patches:
            active.start()
            self.addCleanup(active.stop)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def make_run(self, method):
        run_dir = self.root / method
        if method == "lightmem":
            protocol = {"dataset_sha256": self.data_hash, "model": "Qwen",
                        "embedding_model": "MiniLM", "upstream_sha256": "f" * 64}
            status = {"method": method, "selected": 500, "completed": 500, "population": 500,
                      "failed": [], "generation_complete": True}
        elif method == "higmem":
            protocol = {"method": method, "dataset_sha256": self.data_hash, "question_ids": self.ids}
            status = {"method": method, "selected": 500, "results": 500, "failed_ids": [],
                      "run_complete": True, "invalid_native_answers": 0, "benchmark_complete": True}
        else:
            protocol = {"runtime": {"method": method}, "dataset_sha256": self.data_hash,
                        "population_ids": self.ids}
            status = {"method": method, "planned": 500, "generated": 500, "failed": 0,
                      "status": "generation_complete"}
        self.save(run_dir / "protocol.json", protocol)
        self.save(run_dir / ("completion.json" if method == "higmem" else "status.json"), status)
        rows = []
        for index, qid in enumerate(self.ids):
            row = {"question_id": qid, "hypothesis": f"  응답 {index}\n"}
            if method == "lightmem":
                row.update(protocol)
                self.save(run_dir / qid / "prediction.json", row)
            else:
                row.update(status="generated", identity={
                    "protocol_sha256": exporter.digest(protocol),
                    "source_sha256": f"{index:064x}", "query_sha256": "b" * 64})
                if method == "higmem":
                    row.update(method=method, native_answer_valid=True)
                rows.append(row)
        if method != "lightmem":
            self.save(run_dir / "predictions.json", list(reversed(rows)))
        return argparse.Namespace(method=method, run_dir=run_dir, dataset=self.dataset,
                                  vendor=self.vendor, output=self.root / f"{method}.hypotheses.jsonl")

    def test_all_seven_native_formats_export_only_exact_hypotheses_in_canonical_order(self):
        for method in exporter.METHODS:
            with self.subTest(method=method):
                args = self.make_run(method)
                receipt = exporter.export(args)
                output = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["question_id"] for row in output], self.ids)
                self.assertTrue(all(set(row) == {"question_id", "hypothesis"} for row in output))
                self.assertEqual(output[0]["hypothesis"], "  응답 0\n")
                self.assertNotIn("GOLD_MUST", args.output.read_text(encoding="utf-8"))
                self.assertEqual(receipt["expected_count"], 500)
                self.assertEqual(receipt["official_judge_calls_by_exporter"], 0)

    def alter_predictions(self, args, change):
        path = args.run_dir / "predictions.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        change(rows)
        self.save(path, rows)

    def test_missing_prediction_cannot_be_exported_as_full_500(self):
        args = self.make_run("e_mem")
        self.alter_predictions(args, lambda rows: rows.pop())
        with self.assertRaises(ValueError):
            exporter.export(args)
        self.assertFalse(args.output.exists())

    def test_duplicate_id_is_rejected(self):
        args = self.make_run("mem0")
        self.alter_predictions(args, lambda rows: rows[0].update(question_id=rows[1]["question_id"]))
        with self.assertRaises(ValueError):
            exporter.export(args)

    def test_failed_and_empty_answers_are_rejected(self):
        for method, change in (
            ("simplemem", lambda rows: rows[0].update(hypothesis=" \n")),
            ("langmem", lambda rows: rows[0].update(status="failed")),
            ("higmem", lambda rows: rows[0].update(native_answer_valid=False)),
        ):
            with self.subTest(method=method):
                args = self.make_run(method)
                self.alter_predictions(args, change)
                with self.assertRaises(ValueError):
                    exporter.export(args)

    def test_score_annotations_are_never_accepted_as_prediction_inputs(self):
        args = self.make_run("a_mem")
        self.alter_predictions(args, lambda rows: rows[0].update(autoeval_label={"label": True}))
        with self.assertRaises(ValueError):
            exporter.export(args)

    def test_other_protocol_prediction_is_rejected(self):
        args = self.make_run("e_mem")
        self.alter_predictions(args, lambda rows: rows[0]["identity"].update(protocol_sha256="c" * 64))
        with self.assertRaises(ValueError):
            exporter.export(args)

    def test_modified_official_script_is_rejected_without_output(self):
        args = self.make_run("langmem")
        (self.vendor / next(iter(self.upstream))).write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            exporter.export(args)
        self.assertFalse(args.output.exists())

    def test_lightmem_environment_mismatch_is_rejected(self):
        args = self.make_run("lightmem")
        path = args.run_dir / self.ids[0] / "prediction.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["model"] = "wrong"
        self.save(path, row)
        with self.assertRaises(ValueError):
            exporter.export(args)

    def test_existing_export_is_not_overwritten(self):
        args = self.make_run("e_mem")
        exporter.export(args)
        first = args.output.read_bytes()
        with self.assertRaises(FileExistsError):
            exporter.export(args)
        self.assertEqual(args.output.read_bytes(), first)

    def test_real_producer_writers_and_protocols_are_accepted(self):
        """Exercise production writers; replace only native/model integrations."""
        import contextlib
        import io
        import os
        import sys
        from types import ModuleType, SimpleNamespace
        import native_five
        import native_higmem
        import native_lightmem

        record = {"question": "Fixture question", "question_date": "2026/01/01",
                  "haystack_sessions": [[{"role": "user", "content": "source", "has_answer": True},
                                         {"role": "assistant", "content": "response"}]],
                  "haystack_dates": ["2025/01/01"], "haystack_session_ids": ["session-a"],
                  "answer": "GOLD_MUST_NEVER_APPEAR", "question_type": "SECRET_TYPE"}
        self.dataset.write_text(json.dumps([{**record, "question_id": qid} for qid in self.ids]), encoding="utf-8")
        data_hash = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        source_root = self.root / "source"
        source_root.mkdir()
        for name in native_higmem.SOURCE_FILES:
            (source_root / name).write_text("source fixture", encoding="utf-8")
        hypothesis = "  Production writer 응답\n"
        fake_controller = SimpleNamespace(llm=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0))

        def higmem_factory(*args):
            return SimpleNamespace(
                add_turn=lambda **kwargs: None, finalize_memory_build=lambda: None,
                build_indices=lambda: None, turn_notes={}, events={}, profiles={},
                executor=SimpleNamespace(shutdown=lambda **kwargs: None),
                retrieve_for_query=lambda **kwargs: ("native context", {"mode": "full"}),
                _get_llm_json_response=lambda *args, **kwargs: {"answer": hypothesis})

        fake_memory = SimpleNamespace(
            add_memory=lambda **kwargs: {"add_input_prompt": [], "add_output_prompt": [], "api_call_nums": 0},
            retrieve=lambda *args, **kwargs: ["native memory"])
        fake_reader = SimpleNamespace(call=lambda messages: hypothesis)
        fake_agent = SimpleNamespace(save_agent=lambda: None, close=lambda: None, tokenizer=object())
        utils_package = ModuleType("utils")
        utils_package.__path__ = []
        meter_module = ModuleType("utils.request_metering")
        meter_module.install_request_metering = lambda: None
        meter_module.meter_operation = lambda *args, **kwargs: contextlib.nullcontext()
        agent_module = ModuleType("utils.agent")
        agent_module.AgentWrapper = object

        def inline_worker(command, **kwargs):
            attempt = Path(command[-1])
            code = native_five.worker(attempt)
            self.assertEqual(code, 0, (attempt / "failure.json").read_text(encoding="utf-8") if code else "")
            return SimpleNamespace(returncode=code)

        for method, producer in (("e_mem", native_five), ("lightmem", native_lightmem), ("higmem", native_higmem)):
            with self.subTest(method=method), contextlib.ExitStack() as stack:
                run_dir = self.root / ("actual_" + method)
                stack.enter_context(patch.object(exporter, "DATA_SHA256", data_hash))
                stack.enter_context(patch.object(producer, "DATA_SHA256", data_hash))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if method == "e_mem":
                    stack.enter_context(patch.dict(sys.modules, {"utils": utils_package,
                        "utils.request_metering": meter_module, "utils.agent": agent_module}))
                    stack.enter_context(patch.dict(os.environ, {}, clear=True))
                    stack.enter_context(patch.object(native_five.os, "chdir"))
                    stack.enter_context(patch.object(sys, "path", list(sys.path)))
                    stack.enter_context(patch.object(native_five, "source_hashes", return_value={"native.py": "f" * 64}))
                    stack.enter_context(patch.object(native_five, "build_config", return_value=({"tokenizer_model": "fixture-tokenizer"}, {})))
                    stack.enter_context(patch.object(native_five, "create_agent", return_value=fake_agent))
                    stack.enter_context(patch.object(native_five, "tokenizer_runtime", return_value={"fixture": True}))
                    stack.enter_context(patch.object(native_five, "ingest"))
                    stack.enter_context(patch.object(native_five, "native_answer", return_value=(hypothesis, {})))
                    stack.enter_context(patch.object(native_five.subprocess, "run", side_effect=inline_worker))
                    args = argparse.Namespace(dataset=self.dataset, run_dir=run_dir, source_root=source_root,
                        ids_file=None, method=method, api_base="http://fixture/v1", embedding_api_base="http://fixture/v1",
                        model="Qwen/Qwen3.5-9B", embedding_model="MiniLM", embedding_dims=384, tokenizer=None)
                    self.assertEqual(native_five.run(args), 0)
                else:
                    argv = [method, "--dataset", str(self.dataset), "--run-dir", str(run_dir),
                            "--source-root", str(source_root), "--embedding-model", str(source_root),
                            "--api-base", "http://fixture/v1"]
                    if method == "lightmem":
                        argv += ["--compressor-model", str(source_root)]
                        stack.enter_context(patch.object(native_lightmem, "load_native", return_value={
                            "load_lightmem": lambda qid: fake_memory, "LLMModel": lambda *args: fake_reader}))
                    else:
                        stack.enter_context(patch.object(native_higmem, "load_native", return_value=(
                            fake_controller, higmem_factory,
                            lambda *args: {"keyword_query": "query", "profile_retrieval_keys": []},
                            lambda **kwargs: "native prompt")))
                    stack.enter_context(patch.object(sys, "argv", argv))
                    producer.main()
                args = argparse.Namespace(method=method, run_dir=run_dir, dataset=self.dataset,
                    vendor=self.vendor, output=self.root / f"actual_{method}.jsonl")
                exporter.export(args)
                output = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["question_id"] for row in output], self.ids)
                self.assertTrue(all(row["hypothesis"] == hypothesis for row in output))
                self.assertTrue(all(set(row) == {"question_id", "hypothesis"} for row in output))
                self.assertNotIn("GOLD_MUST_NEVER_APPEAR", args.output.read_text(encoding="utf-8"))
                for source_path in run_dir.rglob("source.json"):
                    self.assertNotIn("has_answer", source_path.read_text(encoding="utf-8"))

    def test_existing_receipt_without_export_is_preserved(self):
        args = self.make_run("e_mem")
        receipt = args.output.with_name(args.output.name + ".receipt.json")
        receipt.write_text("preserve preexisting receipt", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            exporter.export(args)
        self.assertEqual(receipt.read_text(encoding="utf-8"), "preserve preexisting receipt")
        self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
