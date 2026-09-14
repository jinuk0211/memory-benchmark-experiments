"""Runtime gate tests use temporary files and fake HTTP/models; no GPU or sockets."""
import base64
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import verify_runtime as target


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch_count = patch.object(target, "MIN_SOURCE_FILES", 3)
        self.patch_count.start()
        self.addCleanup(self.patch_count.stop)
        self.snapshot = {}
        for key, name in target.SOURCE_GROUPS.items():
            path = self.root / "source" / name / "original.py"
            path.parent.mkdir(parents=True)
            path.write_text("pinned source", encoding="utf-8")
            self.snapshot[key] = {"original.py": sha(path)}
        target.write_json(self.root / "source/source_snapshot.json", self.snapshot)
        self.runtime_files = {}
        for name in target.REQUIRED_RUNTIME:
            path = self.root / name
            path.write_text("pinned runner", encoding="utf-8")
            self.runtime_files[name] = sha(path)
        self.extra = self.root / "source/MemoryData/methods/simplemem/source/SimpleMem/config.py"
        self.extra.parent.mkdir(parents=True)
        self.extra.write_text("pinned supplement", encoding="utf-8")
        self.runtime_files[self.extra.relative_to(self.root).as_posix()] = sha(self.extra)
        models = {}
        for name in ("qwen", "minilm"):
            directory = self.root / "models" / name
            directory.mkdir(parents=True)
            files = {"config.json": "{}", "tokenizer_config.json": "{}", "tokenizer.json": "{}"}
            if name == "qwen":
                shards = [f"model-{index}.safetensors" for index in range(4)]
                files.update({filename: "fixture weights" for filename in shards})
                files["model.safetensors.index.json"] = json.dumps({"weight_map": {str(i): name for i, name in enumerate(shards)}})
            else:
                files.update({"model.safetensors": "fixture weights", "modules.json": "[]", "sentence_bert_config.json": '{"max_seq_length":256}'})
            for filename, content in files.items():
                (directory / filename).write_text(content, encoding="utf-8")
            models[name] = {"path": f"models/{name}", "files": {filename: sha(directory / filename) for filename in files}}
        self.manifest = {"runtime_files": self.runtime_files, "models": models}
        self.manifest_path = self.root / "model_integrity.json"
        self.dataset = self.root / "dataset.json"
        target.write_json(self.dataset, [{"question_id": f"q{i}"} for i in range(500)])
        self.patch_sha = patch.object(target, "DATA_SHA256", sha(self.dataset))
        self.patch_sha.start()
        self.addCleanup(self.patch_sha.stop)

    def verify(self):
        target.write_json(self.manifest_path, self.manifest)
        return target.verify_integrity(self.root, self.manifest_path, self.dataset)

    def test_all_declared_sources_runtime_supplements_and_weights_are_verified(self):
        result = self.verify()
        self.assertEqual(result["dataset"]["questions"], 500)
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(result["models"]["qwen"]["files"], self.manifest["models"]["qwen"]["files"])
        self.assertEqual(result["runtime_files"], self.runtime_files)

    def test_changed_source_runner_and_model_each_fail(self):
        for name in ("source/MemoryData/original.py", "native_five.py", "models/qwen/model-0.safetensors"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    self.verify()
                path.write_bytes(original)

    def test_unhashed_simplemem_addition_fails(self):
        self.runtime_files.pop(self.extra.relative_to(self.root).as_posix())
        with self.assertRaisesRegex(ValueError, "Unhashed supplemental SimpleMem"):
            self.verify()

    def test_missing_weight_or_tokenizer_hash_fails(self):
        for filename in ("model-0.safetensors", "tokenizer.json"):
            with self.subTest(filename=filename):
                expected = self.manifest["models"]["qwen"]["files"]
                old = expected.pop(filename)
                with self.assertRaisesRegex(ValueError, "hashes missing"):
                    self.verify()
                expected[filename] = old

    def test_duplicate_ids_fail_even_with_matching_dataset_hash(self):
        target.write_json(self.dataset, [{"question_id": "same"}] * 500)
        with patch.object(target, "DATA_SHA256", sha(self.dataset)), self.assertRaisesRegex(ValueError, "500 unique"):
            self.verify()

    def test_snapshot_cannot_silently_drop_files(self):
        with patch.object(target, "MIN_SOURCE_FILES", 4), self.assertRaisesRegex(ValueError, "1029 frozen"):
            self.verify()

    def test_manifest_rejects_parent_traversal_and_absolute_names(self):
        for filename in ("../secret", "/absolute", "directory\\file"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "safe relative"):
                target.relative_file(self.root, filename)


class LogProofTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "qwen"
        self.log = self.root / "server.log"
        self.process = {"pid": 123, "started_at": 0, "command": ["python", "vllm", "--model", str(self.model), "--dtype", "float16"]}

    def test_actual_latest_engine_initialization_proves_fp16(self):
        self.log.write_text(f"Initializing a V1 LLM engine with config: model='{self.model}', dtype=torch.float16, max_seq_len=65536\n")
        result = target.verify_fp16_log(self.log, self.process, self.model)
        self.assertEqual(result["actual_dtype"], "float16")
        self.assertEqual(result["process"]["pid"], 123)

    def test_dtype_flag_alone_is_not_actual_fp16_proof(self):
        self.log.write_text("non-default args: {'dtype': 'float16'}\nApplication startup complete\n")
        with self.assertRaisesRegex(ValueError, "No actual"):
            target.verify_fp16_log(self.log, self.process, self.model)

    def test_latest_bfloat16_engine_blocks_old_float16_evidence(self):
        self.log.write_text("\n".join(f"Initializing a V1 LLM engine with config: model='{self.model}', dtype=torch.{dtype}," for dtype in ("float16", "bfloat16")))
        with self.assertRaisesRegex(ValueError, "not FP16"):
            target.verify_fp16_log(self.log, self.process, self.model)

    def test_old_log_cannot_attest_current_process(self):
        self.log.write_text(f"Initializing a V1 LLM engine with config: model='{self.model}', dtype=torch.float16,")
        self.process["started_at"] = self.log.stat().st_mtime + 1
        with self.assertRaisesRegex(ValueError, "predates"):
            target.verify_fp16_log(self.log, self.process, self.model)


class ServiceTests(unittest.TestCase):
    def response(self, vectors, encoding="float", tokens=3):
        return {"model": target.MINILM, "data": [
            {"index": index, "embedding": vector if encoding == "float" else base64.b64encode(struct.pack("<384f", *vector)).decode()}
            for index, vector in enumerate(vectors)], "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

    def test_float_base64_same_finite_normalized_vector(self):
        vectors = [[1.0] + [0.0] * 383]
        floats = target.embedding_vectors(self.response(vectors), 1, "float")
        encoded = target.embedding_vectors(self.response(vectors, "base64"), 1, "base64")
        self.assertEqual(floats, encoded)

    def test_bad_embedding_dimension_nan_norm_and_order_fail(self):
        for defect in ("dimension", "nan", "norm", "order"):
            with self.subTest(defect=defect):
                vector = [1.0] + [0.0] * 383
                response = self.response([vector])
                if defect == "dimension":
                    vector.pop()
                elif defect == "nan":
                    vector[0] = float("nan")
                elif defect == "norm":
                    vector[0] = 2.0
                else:
                    response["data"][0]["index"] = 1
                with self.assertRaises(ValueError):
                    target.embedding_vectors(response, 1, "float")

    def test_stream_must_end_and_contain_native_content(self):
        events = ["data: " + json.dumps({"choices": [{"delta": {"content": "READY"}, "finish_reason": None}]}),
                  "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}), "data: [DONE]"]
        with patch.object(target, "urlopen", return_value=io.BytesIO("\n\n".join(events).encode())):
            result = target.streamed_chat("http://fixture", {})
        self.assertEqual(result["content"], "READY")
        with patch.object(target, "urlopen", return_value=io.BytesIO(events[0].encode())), self.assertRaisesRegex(ValueError, "complete nonempty"):
            target.streamed_chat("http://fixture", {})

    def test_qwen_checks_explicit_tool_choice_and_arguments(self):
        requests = []
        def http(base, route, payload=None):
            requests.append((base, route, payload))
            if route == "/models":
                return {"data": [{"id": target.QWEN}]}
            if "tools" in payload:
                self.assertEqual(payload["tool_choice"], {"type": "function", "function": {"name": "record_probe"}})
                return {"choices": [{"message": {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "record_probe", "arguments": '{"value":"native7"}'}}]}}]}
            return {"model": target.QWEN, "choices": [{"message": {"content": "READY"}}], "usage": {"prompt_tokens": 10}}
        with patch.object(target, "http_json", side_effect=http), patch.object(target, "streamed_chat", return_value={"done": True}) as stream:
            target.verify_qwen("http://fixture")
        self.assertEqual(len(requests), 3)
        stream.assert_called_once()

    def test_missing_tool_call_refuses_service_verification(self):
        responses = [{"data": [{"id": target.QWEN}]}, {"model": target.QWEN, "choices": [{"message": {"content": "READY"}}], "usage": {"prompt_tokens": 1}}, {"choices": [{"message": {"content": "I called the tool"}}]}]
        with patch.object(target, "http_json", side_effect=responses), patch.object(target, "streamed_chat", return_value={}), self.assertRaisesRegex(ValueError, "native tool call"):
            target.verify_qwen("http://fixture")

    def test_minilm_native_window_and_usage_verified_against_fake_local_model(self):
        import numpy as np
        vectors = np.zeros((4, 384), dtype=np.float32)
        vectors[0, 0], vectors[1, 1], vectors[2:, 2] = 1, 1, 1
        class Native:
            max_seq_length = 256
            def get_sentence_embedding_dimension(self):
                return 384
            def tokenizer(self, text, **kwargs):
                return {"input_ids": [1] * (1030 if text.startswith("hello ") else 8)}
            def tokenize(self, texts):
                masks = np.zeros((4, 256), dtype=int)
                masks[:2, :8], masks[2:, :] = 1, 1
                return {"attention_mask": masks}
            def encode(self, texts, **kwargs):
                return vectors
        module = ModuleType("sentence_transformers")
        module.SentenceTransformer = lambda path, **kwargs: Native()
        def http(_base, _route, payload):
            return self.response(vectors.tolist(), payload["encoding_format"], 528)
        with patch.dict(sys.modules, {"sentence_transformers": module}), patch.object(target, "http_json", side_effect=http):
            result = target.verify_minilm("http://fixture", Path("fixture"))
        self.assertEqual(result["attention_tokens"], [8, 8, 256, 256])
        self.assertTrue(result["native_vector_match"])
        self.assertTrue(result["suffix_beyond_window_ignored"])


class RuntimeProcessTests(unittest.TestCase):
    def test_gpu_probe_uses_fp16_and_records_packages_and_nvidia_queries(self):
        calls = []
        class Tensor:
            dtype = "torch.float16"
            def __matmul__(self, other):
                return self
            def sum(self):
                return 4096
        module = ModuleType("torch")
        module.float16 = "float16"
        module.ones = lambda shape, **kwargs: calls.append((shape, kwargs)) or Tensor()
        module.isfinite = lambda result: SimpleNamespace(all=lambda: True)
        module.cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None,
                                      get_device_name=lambda index: "fixture GPU", get_device_capability=lambda index: (12, 0))
        module.version = SimpleNamespace(cuda="fixture CUDA")
        with patch.dict(sys.modules, {"torch": module}), patch.object(target.importlib.metadata, "version", return_value="fixture"), patch.object(target, "command_output", return_value="fixture query") as command:
            proof = target.verify_cuda()
        self.assertEqual(calls, [((16, 16), {"device": "cuda", "dtype": "float16"})])
        self.assertEqual(proof["matmul_sum"], 4096)
        self.assertEqual(proof["device"], "fixture GPU")
        self.assertEqual(set(proof["packages"]), {"torch", "vllm", "transformers", "sentence-transformers"})
        self.assertEqual(command.call_count, 2)
        self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in command.call_args_list))

    def test_unavailable_cuda_cannot_issue_proof(self):
        module = ModuleType("torch")
        module.cuda = SimpleNamespace(is_available=lambda: False)
        with patch.dict(sys.modules, {"torch": module}), self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
            target.verify_cuda()

    def test_vllm_process_must_actually_own_the_listening_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "net").mkdir()
            # /proc/net/tcp columns: sl, local, remote, state, queues, timer, retr, uid, timeout, inode.
            (root / "net/tcp").write_text("header\n0: 0100007F:46A1 00000000:0000 0A 0 0 0 0 0 555\n")
            (root / "net/tcp6").write_text("header\n")
            (root / "stat").write_text("btime 1000\n")
            process = root / "123"
            (process / "fd").mkdir(parents=True)
            (process / "fd/4").touch()
            command = ["python", "vllm.entrypoints.openai.api_server", "--port", "18081"]
            (process / "cmdline").write_bytes("\0".join(command).encode() + b"\0")
            (process / "stat").write_text("123 (python with spaces) " + " ".join(["S"] + ["0"] * 18 + ["200"]))
            def proc_path(value):
                return root / str(value).removeprefix("/proc").lstrip("/")
            with patch.object(target, "Path", side_effect=proc_path), patch.object(target.os, "sysconf", return_value=100, create=True), patch.object(target.os, "readlink", return_value="socket:[555]"):
                proof = target.vllm_process(18081)
            self.assertEqual(proof["pid"], 123)
            self.assertEqual(proof["started_at"], 1002)
            with patch.object(target, "Path", side_effect=proc_path), patch.object(target.os, "readlink", return_value="socket:[999]"), self.assertRaisesRegex(ValueError, "one actual vLLM process"):
                target.vllm_process(18081)

class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.process = {"pid": 42}
        self.integrity = {"models": {"qwen": {"path": "qwen"}, "minilm": {"path": "minilm"}}}
        for name, result in (("verify_integrity", self.integrity), ("verify_cuda", {}), ("vllm_process", self.process),
                             ("verify_fp16_log", {}), ("verify_qwen", {}), ("verify_minilm", {})):
            mocked = patch.object(target, name, return_value=result)
            setattr(self, name, mocked.start())
            self.addCleanup(mocked.stop)

    def verify(self):
        return target.verify(self.root, self.root / "manifest.json", self.root / "data.json", self.root / "server.log")

    def test_receipt_is_written_only_after_all_live_checks(self):
        self.verify_minilm.side_effect = lambda *args: self.assertFalse((self.root / "runtime_receipt.json").exists()) or {}
        result = self.verify()
        self.assertEqual(result["status"], "runtime_verified")
        self.assertEqual(target.read_json(self.root / "runtime_receipt.json"), result)
        self.verify_cuda.assert_called_once()
        self.verify_qwen.assert_called_once()
        self.verify_minilm.assert_called_once()

    def test_failed_live_check_removes_stale_verified_receipt_and_preserves_failure(self):
        old = {"status": "runtime_verified", "old": True}
        target.write_json(self.root / "runtime_receipt.json", old)
        self.verify_qwen.side_effect = RuntimeError("live service failed")
        with self.assertRaisesRegex(RuntimeError, "live service failed"):
            self.verify()
        self.assertFalse((self.root / "runtime_receipt.json").exists())
        previous = list(self.root.glob("runtime_receipt.previous_*.json"))
        self.assertEqual(len(previous), 1)
        self.assertEqual(target.read_json(previous[0]), old)
        self.assertEqual(target.read_json(self.root / "runtime_failure.json")["status"], "runtime_verification_failed")
        self.verify_minilm.assert_not_called()

    def test_changed_vllm_process_prevents_verified_receipt(self):
        self.vllm_process.side_effect = [{"pid": 42}, {"pid": 43}]
        with self.assertRaisesRegex(ValueError, "process changed"):
            self.verify()
        self.assertFalse((self.root / "runtime_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()

