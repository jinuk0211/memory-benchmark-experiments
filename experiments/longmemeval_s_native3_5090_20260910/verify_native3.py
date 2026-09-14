"""Verify this native-three host using frozen native-seven verification helpers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

import verify_runtime_reference as shared

CANDIDATES = ("langmem_native_sessions_v1", "mem0_native_pairs_v1", "a_mem_paper_v1")
REQUIRED = {"native_five.py", "run_three.py", "serve_minilm.py", "serve_qwen.sh",
            "metered_lme_proxy.py", "verify_native3.py", "verify_runtime_reference.py",
            "native3-services.conf", "native3-queue.conf", "run_one.sh", "launch_three.sh"}


def verify_integrity(root: Path, manifest: Path, dataset: Path) -> dict:
    expected = shared.read_json(manifest)
    shared.require(set(expected.get("models", {})) == {"qwen", "minilm"},
                   "Exactly pinned Qwen and MiniLM model maps are required")
    # Only the shared MemoryData closure is applicable; HiGMem/LightMem and their
    # auxiliary model are not deployed. The reference checker otherwise remains unchanged.
    shared.SOURCE_GROUPS = {"source_files_sha256": "MemoryData"}
    shared.MIN_SOURCE_FILES = 501
    shared.REQUIRED_RUNTIME = REQUIRED
    integrity = shared.verify_integrity(root, manifest, dataset)
    shared.require(len(integrity["sources"]["MemoryData"]) == 501,
                   "Expected the exact frozen 501-file MemoryData closure")
    candidate_files = {}
    for name in CANDIDATES:
        candidate = root / "official_recovery" / name
        shared.require((candidate / "runner.py").is_file()
                       and (candidate / "source_manifest.json").is_file(),
                       f"Missing native candidate: {name}")
        for path in candidate.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                key = path.relative_to(root).as_posix()
                actual = shared.file_sha256(path)
                shared.require(integrity["runtime_files"].get(key) == actual,
                               f"Missing or incorrect candidate hash: {key}")
                candidate_files[key] = actual
    integrity["candidate_files"] = candidate_files
    shared.require(shared.read_json(dataset)[0]["question_id"] == "e47becba",
                   "Shared first-history smoke ID differs")
    return integrity


def verify_flags(process: dict) -> dict:
    command = process["command"]
    expected = {"--dtype": "float16", "--max-model-len": "65536",
                "--gpu-memory-utilization": "0.78", "--max-num-seqs": "4",
                "--max-num-batched-tokens": "8192", "--seed": "20260909",
                "--tool-call-parser": "qwen3_coder", "--generation-config": "vllm",
                "--served-model-name": shared.QWEN, "--host": "127.0.0.1"}
    for flag, value in expected.items():
        shared.require(shared.argument(command, flag) == value,
                       f"Live Qwen service differs at {flag}")
    for flag in ("--enforce-eager", "--enable-chunked-prefill", "--enable-prefix-caching",
                 "--language-model-only", "--enable-auto-tool-choice"):
        shared.require(flag in command, f"Missing live Qwen service flag: {flag}")
    template = json.loads(shared.argument(command, "--default-chat-template-kwargs") or "null")
    shared.require(template == {"enable_thinking": False}, "Live thinking-off default differs")
    return {**expected, "default_chat_template_kwargs": template}


def verify(root: Path) -> dict:
    receipt_path = root / "runtime_receipt.json"
    shared.require(not receipt_path.exists(), "Use a fresh receipt path; existing evidence is preserved")
    started = datetime.now(timezone.utc).isoformat()
    try:
        integrity = verify_integrity(root, root / "model_integrity_native3.json",
                                     root / "longmemeval_s_cleaned.json")
        cuda = shared.verify_cuda()
        process = shared.vllm_process(18081)
        flags = verify_flags(process)
        log_proof = shared.verify_fp16_log(root / "logs/native3-qwen.log", process,
                                          Path(integrity["models"]["qwen"]["path"]))
        qwen = shared.verify_qwen("http://127.0.0.1:18083/v1")
        minilm = shared.verify_minilm("http://127.0.0.1:18084/v1",
                                      Path(integrity["models"]["minilm"]["path"]))
        shared.require(shared.vllm_process(18081) == process,
                       "Live Qwen process changed during verification")
        receipt = {"status": "runtime_verified", "started_at": started,
                   "verified_at": datetime.now(timezone.utc).isoformat(),
                   "hostname": socket.gethostname(), "root": str(root),
                   "methods": ["langmem", "mem0", "a_mem"],
                   "integrity": integrity, "runtime": cuda, "service_flags": flags,
                   "fp16_log_proof": log_proof, "service_checks": {"qwen": qwen, "minilm": minilm},
                   "client_environment_verification": "Separate installation/import receipts required",
                   "officially_judged": 0}
        shared.write_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        shared.write_json(root / f"runtime_failure_{time.time_ns()}.json",
                          {"status": "runtime_verification_failed", "started_at": started,
                           "hostname": socket.gethostname(), "error_type": type(error).__name__,
                           "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    print(json.dumps({"status": result["status"], "hostname": result["hostname"],
                      "receipt": str(args.root.resolve() / "runtime_receipt.json")}))
