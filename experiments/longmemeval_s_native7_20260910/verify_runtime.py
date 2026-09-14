"""Verify native7 files, CUDA and live local services before issuing a receipt."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
MIN_SOURCE_FILES = 1029
QWEN = "Qwen/Qwen3.5-9B"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_RUNTIME = {"native_five.py", "native_lightmem.py", "native_higmem.py", "run_all.py",
                    "serve_minilm.py", "serve_qwen.sh", "metered_lme_proxy.py", "verify_runtime.py"}
SOURCE_GROUPS = {"source_files_sha256": "MemoryData", "higmem_files_sha256": "HiGMem",
                 "lightmem_files_sha256": "LightMem"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def relative_file(base: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    require(bool(name) and not relative.is_absolute() and ".." not in relative.parts and "\\" not in name,
            f"Expected a safe relative manifest filename: {name!r}")
    # HF snapshots intentionally contain symlinks into their sibling blob directory.
    return base.joinpath(*relative.parts)


def verify_file_map(base: Path, expected: dict[str, str]) -> dict[str, str]:
    require(isinstance(expected, dict) and bool(expected), f"Empty file map: {base}")
    checked = {}
    for name, expected_sha in sorted(expected.items()):
        require(isinstance(expected_sha, str) and re.fullmatch(r"[a-f0-9]{64}", expected_sha) is not None,
                f"Invalid expected SHA256: {name}")
        actual = file_sha256(relative_file(base, name))
        require(actual == expected_sha, f"SHA256 mismatch: {base / name}")
        checked[name] = actual
    return checked


def verify_integrity(root: Path, manifest_path: Path, dataset: Path) -> dict:
    manifest = read_json(manifest_path)
    snapshot_path = root / "source/source_snapshot.json"
    snapshot = read_json(snapshot_path)
    sources, covered = {}, set()
    for key, directory in SOURCE_GROUPS.items():
        files = verify_file_map(root / "source" / directory, snapshot[key])
        sources[directory] = files
        covered.update(f"source/{directory}/{name}" for name in files)
    require(sum(map(len, sources.values())) >= MIN_SOURCE_FILES, "Source snapshot must cover at least 1029 frozen files")
    runtime_files = manifest.get("runtime_files", {})
    require(REQUIRED_RUNTIME <= runtime_files.keys(), "Expected hashes missing for native7 runtime files")
    runtime = verify_file_map(root, runtime_files)
    covered.update(runtime)
    simplemem_root = root / "source/MemoryData/methods/simplemem"
    for path in simplemem_root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            require(path.relative_to(root).as_posix() in covered, f"Unhashed supplemental SimpleMem file: {path}")
    models = manifest.get("models", {})
    require({"qwen", "minilm"} <= models.keys(), "Qwen and MiniLM model maps are required")
    model_checks = {}
    for name, spec in models.items():
        model_path = Path(spec["path"])
        if not model_path.is_absolute():
            model_path = root / model_path
        model_path = model_path.resolve()
        expected = spec["files"]
        require({"config.json", "tokenizer_config.json", "tokenizer.json"} <= expected.keys(),
                f"Expected config/tokenizer hashes missing: {name}")
        weights = {p.relative_to(model_path).as_posix() for p in model_path.rglob("*.safetensors")}
        require(bool(weights) and weights <= expected.keys(), f"Expected model-weight hashes missing: {name}")
        if name == "qwen":
            require(len(weights) == 4, "Expected four pinned Qwen weight shards")
        if name == "minilm":
            require({"modules.json", "sentence_bert_config.json"} <= expected.keys(), "MiniLM native config hashes missing")
        files = verify_file_map(model_path, expected)
        index = model_path / "model.safetensors.index.json"
        if index.exists():
            require(index.name in expected, f"Unhashed model weight index: {name}")
            referenced = set(read_json(index)["weight_map"].values())
            require(referenced <= weights, f"Missing model shards referenced by index: {name}")
        model_checks[name] = {"path": str(model_path), "files": files}
    require(file_sha256(dataset) == DATA_SHA256, "Canonical LongMemEval-S dataset SHA256 mismatch")
    ids = [row["question_id"] for row in read_json(dataset)]
    require(len(ids) == 500 and all(isinstance(qid, str) and qid for qid in ids) and len(set(ids)) == 500,
            "Expected 500 unique canonical question IDs")
    return {"manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            "source_snapshot_sha256": file_sha256(snapshot_path), "sources": sources,
            "runtime_files": runtime, "models": model_checks,
            "dataset": {"path": str(dataset), "sha256": DATA_SHA256, "questions": len(ids)}}


def command_output(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=30).stdout.strip()


def verify_cuda() -> dict:
    import torch

    require(torch.cuda.is_available(), "CUDA is unavailable in the verifier interpreter")
    x = torch.ones((16, 16), device="cuda", dtype=torch.float16)
    result = x @ x
    torch.cuda.synchronize()
    total = float(result.sum())
    require(total == 4096.0 and bool(torch.isfinite(result).all()), "CUDA FP16 16x16 matmul failed")
    return {"python": sys.version, "executable": sys.executable, "pid": os.getpid(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "vllm", "transformers", "sentence-transformers")},
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "matmul_dtype": str(result.dtype),
            "matmul_sum": total,
            "nvidia_gpu": command_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"]),
            "nvidia_processes": command_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])}


def argument(command: list[str], flag: str) -> str | None:
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def vllm_process(port: int) -> dict:
    inodes = set()
    for filename in ("tcp", "tcp6"):
        for line in (Path("/proc/net") / filename).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
                inodes.add(fields[9])
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if not any("vllm" in part for part in command) or argument(command, "--port") != str(port):
                continue
            sockets = {os.readlink(fd) for fd in (entry / "fd").iterdir()}
            if not any(f"socket:[{inode}]" in sockets for inode in inodes):
                continue
            ticks = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[19])
            boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
            matches.append({"pid": int(entry.name), "command": command,
                            "started_at": boot + ticks / os.sysconf("SC_CLK_TCK"), "listening_port": port})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    require(len(matches) == 1, f"Expected one actual vLLM process listening on port {port}")
    return matches[0]


def verify_fp16_log(log_path: Path, process: dict, model_path: Path) -> dict:
    require(argument(process["command"], "--dtype") in ("float16", "half"),
            "Live vLLM process dtype flag conflicts with FP16 startup evidence")
    require(Path(argument(process["command"], "--model") or "").resolve() == model_path.resolve(),
            "Live vLLM process uses a different model snapshot")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    require(log_path.stat().st_mtime >= process["started_at"], "vLLM log predates the current process")
    engine_lines = [line for line in log.splitlines() if "Initializing" in line and "LLM engine" in line and "dtype=" in line]
    require(bool(engine_lines), "No actual vLLM engine initialization dtype evidence in log")
    actual = engine_lines[-1]
    require(re.search(r"dtype=(?:torch\.)?float16(?:[,\s)])", actual) is not None,
            "Actual latest vLLM engine initialization is not FP16")
    require(str(model_path) in actual, "vLLM engine initialization log names another model")
    return {"path": str(log_path), "sha256_at_verification": file_sha256(log_path),
            "actual_engine_initialization": actual, "actual_dtype": "float16", "process": process}


def http_json(base: str, route: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(base + route, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def streamed_chat(base: str, payload: dict) -> dict:
    request = Request(base + "/chat/completions", data=json.dumps({**payload, "stream": True}).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    text, chunks, done, finish = [], 0, False, None
    with urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                done = True
                break
            event = json.loads(value)
            require("error" not in event, "Streaming service returned an error")
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    text.append(content)
                finish = choice.get("finish_reason") or finish
            chunks += 1
    result = "".join(text)
    require(done and chunks > 0 and bool(result.strip()) and finish in ("stop", "length"),
            "Streaming chat did not yield complete nonempty content")
    return {"content": result, "chunks": chunks, "done": done, "finish_reason": finish}


def verify_qwen(base: str) -> dict:
    models = http_json(base, "/models")
    require(QWEN in {item["id"] for item in models["data"]}, "Qwen service model identity mismatch")
    payload = {"model": QWEN, "messages": [{"role": "user", "content": "Reply with the single word READY."}],
               "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}
    response = http_json(base, "/chat/completions", payload)
    content = response["choices"][0]["message"].get("content")
    require(response.get("model") == QWEN and isinstance(content, str) and bool(content.strip()), "Empty or wrong-model Qwen completion")
    require(response.get("usage", {}).get("prompt_tokens", 0) > 0, "Qwen completion usage missing")
    stream = streamed_chat(base, payload)
    tool_payload = {**payload, "messages": [{"role": "user", "content": "Call record_probe with value native7."}],
                    "max_tokens": 128, "tools": [{"type": "function", "function": {
                        "name": "record_probe", "description": "Record the runtime probe value.",
                        "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                                       "required": ["value"], "additionalProperties": False}}}],
                    "tool_choice": {"type": "function", "function": {"name": "record_probe"}}}
    tool_response = http_json(base, "/chat/completions", tool_payload)
    calls = tool_response["choices"][0]["message"].get("tool_calls", [])
    require(len(calls) == 1 and calls[0].get("type") == "function" and bool(calls[0].get("id")), "Explicit tool choice did not produce one native tool call")
    call = calls[0]["function"]
    require(call["name"] == "record_probe" and json.loads(call["arguments"]) == {"value": "native7"}, "Tool-call name/arguments mismatch")
    return {"base_url": base, "completion": response, "stream": stream, "explicit_tool_choice": tool_response}


def embedding_vectors(response: dict, count: int, encoding: str) -> list[list[float]]:
    require(response.get("model") == MINILM and len(response.get("data", [])) == count, "Embedding response model/count mismatch")
    vectors = []
    for index, item in enumerate(response["data"]):
        require(item.get("index") == index, "Embedding response order mismatch")
        value = item["embedding"]
        if encoding == "base64":
            raw = base64.b64decode(value, validate=True)
            require(len(raw) == 384 * 4, "Base64 embedding must contain 384 float32 values")
            value = list(struct.unpack("<384f", raw))
        require(isinstance(value, list) and len(value) == 384 and
                all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value),
                "Embedding must contain 384 finite numbers")
        require(abs(sum(x * x for x in value) - 1.0) < 1e-4, "Embedding is not normalized")
        vectors.append(value)
    return vectors


def close_vectors(left: list, right: list, tolerance: float = 1e-6) -> bool:
    return len(left) == len(right) and all(len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))
                                         for a, b in zip(left, right))


def verify_minilm(base: str, model_path: Path) -> dict:
    from sentence_transformers import SentenceTransformer

    native = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    require(native.max_seq_length == 256 and native.get_sentence_embedding_dimension() == 384,
            "Pinned MiniLM native settings must be 256 tokens / 384 dimensions")
    texts = ["The native runtime is ready.", "Bananas grow on tropical plants.", "hello " * 1024 + "suffix alpha", "hello " * 1024 + "suffix beta"]
    raw_counts = [len(native.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]) for text in texts]
    require(min(raw_counts[2:]) > 256, "Truncation probe must exceed 256 native tokens")
    features = native.tokenize(texts)
    token_counts = [int(mask.sum()) for mask in features["attention_mask"]]
    require(token_counts[2:] == [256, 256], "Native tokenizer did not truncate to 256")
    expected = native.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    payload = {"model": MINILM, "input": texts, "dimensions": 384}
    floats = http_json(base, "/embeddings", {**payload, "encoding_format": "float"})
    encoded = http_json(base, "/embeddings", {**payload, "encoding_format": "base64"})
    vectors = embedding_vectors(floats, len(texts), "float")
    decoded = embedding_vectors(encoded, len(texts), "base64")
    require(close_vectors(vectors, decoded), "Float/base64 embedding vectors differ")
    require(close_vectors(vectors, expected), "HTTP embeddings differ from pinned native MiniLM")
    require(close_vectors([vectors[2]], [vectors[3]]) and not close_vectors([vectors[0]], [vectors[1]]),
            "Truncation/sensitivity embedding probe failed")
    for response in (floats, encoded):
        require(response.get("usage") == {"prompt_tokens": sum(token_counts), "total_tokens": sum(token_counts)},
                "Embedding usage must count post-truncation attention tokens")
    return {"base_url": base, "model": MINILM, "dimensions": 384, "native_max_seq_length": 256,
            "raw_tokens": raw_counts, "attention_tokens": token_counts, "usage": floats["usage"],
            "float_base64_match": True, "native_vector_match": True, "normalized_finite": True,
            "suffix_beyond_window_ignored": True, "distinct_short_inputs_differ": True,
            "vectors_sha256": hashlib.sha256(json.dumps(vectors).encode()).hexdigest()}


def verify(root: Path, manifest: Path, dataset: Path, log_path: Path, vllm_port: int = 18081) -> dict:
    receipt_path = root / "runtime_receipt.json"
    if receipt_path.exists():
        receipt_path.replace(root / f"runtime_receipt.previous_{time.time_ns()}.json")
    started = datetime.now(timezone.utc).isoformat()
    try:
        integrity = verify_integrity(root, manifest, dataset)
        cuda = verify_cuda()
        process = vllm_process(vllm_port)
        log_proof = verify_fp16_log(log_path, process, Path(integrity["models"]["qwen"]["path"]))
        qwen = verify_qwen("http://127.0.0.1:18083/v1")
        minilm = verify_minilm("http://127.0.0.1:18084/v1", Path(integrity["models"]["minilm"]["path"]))
        require(vllm_process(vllm_port) == process, "vLLM process changed during runtime verification")
        receipt = {"status": "runtime_verified", "started_at": started,
                   "verified_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
                   "integrity": integrity, "runtime": cuda, "fp16_log_proof": log_proof,
                   "service_checks": {"qwen": qwen, "minilm": minilm}, "officially_judged": 0}
        write_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        write_json(root / "runtime_failure.json", {"status": "runtime_verification_failed", "started_at": started,
                   "failed_at": datetime.now(timezone.utc).isoformat(), "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model-integrity", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--vllm-log", type=Path)
    parser.add_argument("--vllm-port", type=int, default=18081)
    args = parser.parse_args()
    root = args.root.resolve()
    verify(root, args.model_integrity or root / "model_integrity.json",
           args.dataset or root / "longmemeval_s_cleaned.json",
           args.vllm_log or root / "logs/native7-qwen.log", args.vllm_port)
    print(json.dumps({"status": "runtime_verified", "receipt": str(root / "runtime_receipt.json")}))


if __name__ == "__main__":
    main()


