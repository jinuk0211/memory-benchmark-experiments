"""Verify recovery sources and live services without replacing the original receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import verify_runtime as original

BASE_RECEIPT_SHA256 = "ad0ac34ce4a68101d39b466e5c208a338123868fc008979c97a85cf7f180897d"


def verify_recovery(root: Path, manifest: Path, receipt_path: Path,
                    candidate: Path | None = None) -> dict:
    root, manifest, receipt_path = root.resolve(), manifest.resolve(), receipt_path.resolve()
    candidate = (candidate or root / "official_recovery/a_mem_paper_v1").resolve()
    original.require(Path(original.__file__).resolve() == root / "verify_runtime.py",
                     "Imported verifier does not belong to the verified root")
    verifier_path = Path(__file__).resolve()
    original.require(verifier_path == root / "official_recovery/verify_recovery_runtime.py",
                     "Recovery verifier does not belong to the verified root")
    original.require(candidate.is_relative_to(root / "official_recovery") and candidate.is_dir(),
                     "Candidate must be an existing recovery directory")
    original.require(manifest.is_relative_to(root), "Recovery manifest must be inside the experiment")
    original.require(receipt_path.is_relative_to(root), "Recovery receipt must be inside the experiment")
    original.require(not receipt_path.exists(), "Recovery receipt already exists; choose a new path")
    base_receipt = root / "runtime_receipt.json"
    original.require(original.file_sha256(base_receipt) == BASE_RECEIPT_SHA256,
                     "Original verified receipt changed")
    base = original.read_json(base_receipt)
    original.require(base.get("status") == "runtime_verified", "Original runtime is not verified")
    expected = original.read_json(manifest).get("runtime_files", {})
    for name, digest in base["integrity"]["runtime_files"].items():
        original.require(expected.get(name) == digest, f"Original runtime binding changed: {name}")
    required = {str(verifier_path.relative_to(root).as_posix())}
    for filename in ("runner.py", "source_manifest.json"):
        original.require((candidate / filename).is_file(), f"Missing candidate file: {filename}")
    required.update(path.relative_to(root).as_posix() for path in candidate.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    original.require(required <= expected.keys(),
                     f"Recovery manifest omits candidate/verifier files: {sorted(required - expected.keys())}")
    started = datetime.now(timezone.utc).isoformat()
    integrity = original.verify_integrity(root, manifest, root / "longmemeval_s_cleaned.json")
    original.require(integrity["models"] == base["integrity"]["models"],
                     "Recovery model paths or weight/config hashes differ from the original verified models")
    original.require(integrity["source_snapshot_sha256"] == base["integrity"]["source_snapshot_sha256"]
                     and integrity["sources"] == base["integrity"]["sources"],
                     "Original frozen sources changed")
    cuda = original.verify_cuda()
    process = original.vllm_process(18081)
    log_proof = original.verify_fp16_log(root / "logs/native7-qwen.log", process,
                                       Path(integrity["models"]["qwen"]["path"]))
    qwen = original.verify_qwen("http://127.0.0.1:18083/v1")
    minilm = original.verify_minilm("http://127.0.0.1:18084/v1",
                                    Path(integrity["models"]["minilm"]["path"]))
    original.require(original.vllm_process(18081) == process,
                     "vLLM process changed during runtime verification")
    original.require(original.file_sha256(base_receipt) == BASE_RECEIPT_SHA256,
                     "Original receipt changed during runtime verification")
    receipt = {
        "status": "runtime_verified", "started_at": started,
        "verified_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "base_receipt": {"path": str(base_receipt), "sha256": BASE_RECEIPT_SHA256},
        "candidate": str(candidate), "required_recovery_files": sorted(required),
        "integrity": integrity, "runtime": cuda, "fp16_log_proof": log_proof,
        "service_checks": {"qwen": qwen, "minilm": minilm}, "officially_judged": 0,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a late competing verification from overwriting a receipt.
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    verify_recovery(args.root, args.manifest, args.receipt, args.candidate)
    print(json.dumps({"status": "runtime_verified", "receipt": str(args.receipt.resolve())}))


if __name__ == "__main__":
    main()
