"""Offline native3 verifier boundary tests; no GPU, network or benchmark content."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_native3 as verify


class VerifierTests(unittest.TestCase):
    def process(self):
        return {"command": ["python", "vllm", "--dtype", "float16", "--max-model-len", "65536",
            "--gpu-memory-utilization", "0.78", "--max-num-seqs", "4", "--max-num-batched-tokens", "8192",
            "--seed", "20260909", "--tool-call-parser", "qwen3_coder", "--generation-config", "vllm",
            "--served-model-name", verify.shared.QWEN, "--host", "127.0.0.1", "--enforce-eager",
            "--enable-chunked-prefill", "--enable-prefix-caching", "--language-model-only",
            "--enable-auto-tool-choice", "--default-chat-template-kwargs", '{"enable_thinking":false}']}

    def test_matching_live_flags(self):
        self.assertEqual(verify.verify_flags(self.process())["--max-model-len"], "65536")

    def test_wrong_context_dtype_thinking_or_tool_parser_rejected(self):
        for flag, value in (("--dtype", "bfloat16"), ("--max-model-len", "32768"),
                            ("--default-chat-template-kwargs", '{"enable_thinking":true}'),
                            ("--tool-call-parser", "hermes")):
            with self.subTest(flag=flag):
                process = self.process()
                process["command"][process["command"].index(flag) + 1] = value
                with self.assertRaises(ValueError):
                    verify.verify_flags(process)

    def test_candidate_file_omission_rejected_and_complete_coverage_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            dataset = root / "dataset.json"
            manifest.write_text(json.dumps({"models": {"qwen": {}, "minilm": {}}}))
            dataset.write_text('[{"question_id":"e47becba"}]')
            files = {}
            for name in verify.CANDIDATES:
                folder = root / "official_recovery" / name
                folder.mkdir(parents=True)
                for filename in ("runner.py", "source_manifest.json", "LICENSE"):
                    path = folder / filename
                    path.write_text("frozen")
                    files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            integrity = {"sources": {"MemoryData": {str(i): "sha" for i in range(501)}},
                         "runtime_files": dict(files)}
            with patch.object(verify.shared, "verify_integrity", return_value=integrity):
                result = verify.verify_integrity(root, manifest, dataset)
                self.assertEqual(len(result["candidate_files"]), 9)
                integrity["runtime_files"].pop(next(iter(files)))
                with self.assertRaisesRegex(ValueError, "candidate hash"):
                    verify.verify_integrity(root, manifest, dataset)

    def test_existing_host_receipt_preserved_before_any_live_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "runtime_receipt.json"
            receipt.write_text("existing evidence")
            with patch.object(verify, "verify_integrity") as integrity:
                with self.assertRaisesRegex(ValueError, "existing evidence"):
                    verify.verify(root)
                integrity.assert_not_called()
            self.assertEqual(receipt.read_text(), "existing evidence")

    def test_auxiliary_model_manifest_rejected_before_hash_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"models": {"qwen": {}, "minilm": {}, "compressor": {}}}))
            with patch.object(verify.shared, "verify_integrity") as shared_check:
                with self.assertRaisesRegex(ValueError, "Exactly pinned"):
                    verify.verify_integrity(root, manifest, root / "dataset.json")
                shared_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
