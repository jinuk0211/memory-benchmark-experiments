"""Receipt preservation and failure gates; no GPU or model API calls."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_recovery_runtime as recovery


class RecoveryReceiptTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(prefix="receipt_test_", dir=workspace)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(workspace))
        self.addCleanup(self.temporary.cleanup)
        self.base_path = self.root / "runtime_receipt.json"
        integrity = {
            "runtime_files": {"original.py": "a" * 64},
            "models": {"qwen": {"path": "/qwen", "files": {"weight": "q"}},
                       "minilm": {"path": "/minilm", "files": {"weight": "m"}}},
            "source_snapshot_sha256": "s" * 64,
            "sources": {"MemoryData": {"module.py": "d" * 64}},
        }
        self.base_bytes = json.dumps({"status": "runtime_verified", "integrity": integrity}).encode()
        self.base_path.write_bytes(self.base_bytes)
        self.candidate = self.root / "official_recovery/a_mem_paper_v1"
        (self.candidate / "upstream").mkdir(parents=True)
        for name in ("runner.py", "source_manifest.json", "upstream/native.py"):
            (self.candidate / name).write_text("test fixture")
        self.verifier = self.root / "official_recovery/verify_recovery_runtime.py"
        self.verifier.write_text("test fixture")
        self.file_map = {"original.py": "a" * 64}
        self.file_map.update({path.relative_to(self.root).as_posix(): "b" * 64
                             for path in (self.root / "official_recovery").rglob("*") if path.is_file()})
        self.manifest = self.root / "recovery_manifest.json"
        self.manifest.write_text(json.dumps({"runtime_files": self.file_map}))
        self.receipt = self.root / "new_receipt.json"
        self.stack = self.enterContext(ExitStack())
        self.stack.enter_context(patch.object(recovery, "__file__", str(self.verifier)))
        self.stack.enter_context(patch.object(recovery.original, "__file__", str(self.root / "verify_runtime.py")))
        self.stack.enter_context(patch.object(recovery, "BASE_RECEIPT_SHA256",
                                             hashlib.sha256(self.base_bytes).hexdigest()))
        values = {
            "verify_integrity": json.loads(json.dumps(integrity)),
            "verify_cuda": {"cuda": True}, "vllm_process": {"pid": 123},
            "verify_fp16_log": {"dtype": "float16"}, "verify_qwen": {"completion": "READY"},
            "verify_minilm": {"dimensions": 384},
        }
        self.checks = {name: self.stack.enter_context(patch.object(recovery.original, name, return_value=value))
                       for name, value in values.items()}

    def run_verification(self):
        return recovery.verify_recovery(self.root, self.manifest, self.receipt)

    def assert_base_preserved(self):
        self.assertEqual(self.base_path.read_bytes(), self.base_bytes)

    def test_complete_verification_publishes_separate_receipt(self):
        result = self.run_verification()
        self.assertEqual(json.loads(self.receipt.read_text()), result)
        self.assertEqual(result["status"], "runtime_verified")
        self.assertEqual(result["service_checks"]["minilm"]["dimensions"], 384)
        for check in self.checks.values():
            self.assertTrue(check.called)
        self.assert_base_preserved()

    def test_existing_receipt_never_overwritten(self):
        self.receipt.write_bytes(b"prior proof")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_verification()
        self.assertEqual(self.receipt.read_bytes(), b"prior proof")
        self.checks["verify_integrity"].assert_not_called()
        self.assert_base_preserved()

    def test_failed_live_service_does_not_issue_verified_receipt(self):
        self.checks["verify_qwen"].side_effect = RuntimeError("local model unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.run_verification()
        self.assertFalse(self.receipt.exists())
        self.assert_base_preserved()

    def test_changed_original_receipt_blocks_model_calls(self):
        self.base_path.write_bytes(b"changed proof")
        with self.assertRaisesRegex(ValueError, "Original verified receipt changed"):
            self.run_verification()
        self.checks["verify_cuda"].assert_not_called()
        self.assertEqual(self.base_path.read_bytes(), b"changed proof")
        self.assertFalse(self.receipt.exists())

    def test_changed_original_file_binding_is_rejected(self):
        self.manifest.write_text(json.dumps({"runtime_files": {
            "original.py": "c" * 64, "official_recovery/runner.py": "b" * 64
        }}))
        with self.assertRaisesRegex(ValueError, "Original runtime binding changed"):
            self.run_verification()
        self.assertFalse(self.receipt.exists())
        self.assert_base_preserved()

    def test_model_process_change_prevents_publication(self):
        self.checks["vllm_process"].side_effect = [{"pid": 123}, {"pid": 456}]
        with self.assertRaisesRegex(ValueError, "process changed"):
            self.run_verification()
        self.assertFalse(self.receipt.exists())
        self.assert_base_preserved()

    def test_late_concurrent_receipt_is_not_overwritten(self):
        def finish_embedding(*args):
            self.receipt.write_bytes(b"other completed verifier")
            return {"dimensions": 384}
        self.checks["verify_minilm"].side_effect = finish_embedding
        with self.assertRaises(FileExistsError):
            self.run_verification()
        self.assertEqual(self.receipt.read_bytes(), b"other completed verifier")
        self.assert_base_preserved()

    def test_manifest_without_new_sources_is_rejected(self):
        self.manifest.write_text(json.dumps({"runtime_files": {"original.py": "a" * 64}}))
        with self.assertRaisesRegex(ValueError, "omits candidate"):
            self.run_verification()
        self.checks["verify_integrity"].assert_not_called()
        self.assert_base_preserved()


    def test_self_consistent_different_model_is_rejected(self):
        self.checks["verify_integrity"].return_value["models"]["qwen"]["files"]["weight"] = "different"
        with self.assertRaisesRegex(ValueError, "original verified models"):
            self.run_verification()
        self.checks["verify_cuda"].assert_not_called()
        self.assertFalse(self.receipt.exists())
        self.assert_base_preserved()

    def test_changed_original_sources_are_rejected(self):
        self.checks["verify_integrity"].return_value["source_snapshot_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Original frozen sources changed"):
            self.run_verification()
        self.assertFalse(self.receipt.exists())
        self.assert_base_preserved()

    def test_omitted_upstream_file_blocks_verification(self):
        del self.file_map["official_recovery/a_mem_paper_v1/upstream/native.py"]
        self.manifest.write_text(json.dumps({"runtime_files": self.file_map}))
        with self.assertRaisesRegex(ValueError, "omits candidate"):
            self.run_verification()
        self.checks["verify_integrity"].assert_not_called()
        self.assertFalse(self.receipt.exists())

    def test_imported_verifier_from_another_root_is_rejected(self):
        with patch.object(recovery.original, "__file__", str(self.root / "other/verify_runtime.py")):
            with self.assertRaisesRegex(ValueError, "Imported verifier"):
                self.run_verification()
        self.checks["verify_integrity"].assert_not_called()
        self.assertFalse(self.receipt.exists())

if __name__ == "__main__":
    unittest.main()
