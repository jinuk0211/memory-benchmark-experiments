"""Pinned SimpleMem core, original bulk Dialogue ingestion and native ask."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from types import ModuleType, SimpleNamespace
from typing import List
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
sys.path.insert(0, str(HARNESS))
from native_five import DATA_SHA256, digest, next_attempt, read_json, save_json, source_only, write_once

PIN = "db80b6a7c591e0ea730a058e9f5fc4eb06572299"
MODEL = "Qwen/Qwen3.5-9B"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
POLICY = {"official_commit": PIN, "ingestion": "original convert_to_dialogues then one bulk add_dialogues + finalize",
          "source_turn_fields": ["role", "content"], "window_size": 40, "overlap_size": 2,
          "parallel_workers": 16, "retrieval_workers": 8, "streaming": True,
          "planning": True, "reflection": True, "max_reflection_rounds": 2,
          "max_tokens": "Not sent by original LLMClient; no new limit added",
          "temperature": "Original caller-specific values unchanged",
          "normalization": "Compatibility v3: location list[str] joined with comma-space; original raw response audited; no token splitting",
          "question_date": "Prefixed through public ask(question) argument",
          "native_errors": "Original retries/fallback behavior observed; failed histories excluded"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    manifest = read_json(HERE / "source_manifest.json")
    if manifest["commit"] != PIN:
        raise ValueError("Unexpected original SimpleMem revision")
    hashes = {}
    for name, item in manifest["files"].items():
        actual = sha(HERE / "upstream" / name)
        if actual != item["sha256"]:
            raise ValueError(f"Original source changed: {name}")
        hashes[f"upstream/{name}"] = actual
    for name in ("runner.py", "source_manifest.json", "location_compat.py"):
        hashes[name] = sha(HERE / name)
    hashes["harness/native_five.py"] = sha(HARNESS / "native_five.py")
    hashes["harness/request_metering.py"] = sha(HARNESS / "source/MemoryData/utils/request_metering.py")
    return hashes


def local_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1" or not parsed.port):
        raise ValueError("A loopback HTTP /v1 endpoint with explicit port is required")
    return value.rstrip("/")


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(runtime: dict, attempt: Path):
    # Explicit package path avoids any installed simplemem distribution/router being imported.
    if any(name == "simplemem" or name.startswith("simplemem.") for name in sys.modules):
        raise RuntimeError("SimpleMem must load in a fresh per-history worker")
    package = ModuleType("simplemem")
    package.__path__ = [str(HERE / "upstream/simplemem")]
    sys.modules["simplemem"] = package

    config = ModuleType("config")
    path = HERE / "upstream/config.py.example"
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), config.__dict__)
    settings_tree = ast.parse((HERE / "upstream/simplemem/core/settings.py").read_text(encoding="utf-8"))
    defaults = next(node.value for node in settings_tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "_DEFAULTS" for target in node.targets))
    for key, value in ast.literal_eval(defaults).items():
        if not hasattr(config, key):
            setattr(config, key, value)  # Keep native defaults, neutralize unrelated inherited env settings.
    config.OPENAI_API_KEY, config.OPENAI_BASE_URL, config.LLM_MODEL = "EMPTY", runtime["api_base"], MODEL
    config.EMBEDDING_MODEL = runtime["embedding_model"]
    config.EMBEDDING_DIMENSION, config.EMBEDDING_CONTEXT_LENGTH = 384, 256
    config.LANCEDB_PATH, config.MEMORY_TABLE_NAME = str(attempt / "database"), "memory_entries"
    config.ENABLE_THINKING = False
    sys.modules["config"] = config
    return config


def convert_dialogues(source: list[dict], dialogue_class) -> list:
    tree = ast.parse((HERE / "upstream/test_locomo10.py").read_text(encoding="utf-8"))
    tester = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LoCoMoTester")
    function = next(node for node in tester.body if isinstance(node, ast.FunctionDef) and node.name == "convert_to_dialogues")
    scope = {"List": List, "Dialogue": dialogue_class, "LoCoMoSample": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(HERE / "upstream/test_locomo10.py"), "exec"), scope)
    # Integer keys reproduce native chronological sorting while retaining canonical session order.
    sessions = {index: SimpleNamespace(date_time=session["date"], turns=[
        SimpleNamespace(speaker=turn["role"], text=turn["content"]) for turn in session["turns"]])
        for index, session in enumerate(source)}
    return scope["convert_to_dialogues"](None, SimpleNamespace(conversation=SimpleNamespace(sessions=sessions)))


class Calls:
    def __init__(self, path: Path):
        self.path = path
        self.phase = "memory_write"
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.count = 0
        self.lock = threading.Lock()

    def bind(self, client, endpoint: str):
        if client.model != MODEL or str(client.client.base_url).rstrip("/") != endpoint:
            raise ValueError("SimpleMem escaped fixed local model/endpoint")
        original = client.chat_completion

        def observed(messages, *args, **kwargs):
            record = {"phase": self.phase, "messages": messages, "args": args, "kwargs": kwargs}
            started = time.monotonic()
            try:
                # Worker threads inherit the existing outer phase through request_metering.
                result = original(messages, *args, **kwargs)
                record["response"] = result
                return result
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["seconds"] = time.monotonic() - started
                with self.lock:
                    self.count += 1
                    with self.path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        client.chat_completion = observed

    def check(self):
        if self.errors:
            raise RuntimeError(f"Native failure fallback observed ({len(self.errors)}); see failure.json")


@contextmanager
def native_fallbacks(calls: Calls):
    """Observe native final error returns in main and parallel worker threads."""
    lines = {}
    paths = ("memory_builder.py", "hybrid_retriever.py", "answer_generator.py",
             "database/vector_store.py", "database/vector_store_backend.py")
    for name in paths:
        path = HERE / "upstream/simplemem/core" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = {}
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
            for ret in (node for node in ast.walk(handler) if isinstance(node, ast.Return)):
                fatal = name not in {"hybrid_retriever.py", "answer_generator.py"}
                if name == "answer_generator.py":
                    fatal = isinstance(ret.value, ast.Constant) and ret.value.value == "Failed to generate answer"
                selected[ret.lineno] = fatal
            if name.startswith("database/"):
                selected[handler.body[0].lineno] = True  # Swallowed backend/row/FTS exceptions.
        if name == "memory_builder.py":
            selected[361] = True  # Original parallel future failed and was discarded.
        lines[str(path)] = selected
    previous, previous_thread = sys.gettrace(), threading.gettrace()

    def observe(frame, event, arg):
        if frame.f_code.co_filename not in lines:
            return None
        if event == "line" and frame.f_lineno in lines[frame.f_code.co_filename]:
            with calls.lock:
                target = calls.errors if lines[frame.f_code.co_filename][frame.f_lineno] else calls.warnings
                target.append(f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}:{frame.f_code.co_name}")
        return observe

    sys.settrace(observe)
    threading.settrace(observe)
    try:
        yield
    finally:
        sys.settrace(previous)
        threading.settrace(previous_thread)


def make_system(runtime: dict, attempt: Path, calls: Calls):
    config = configure(runtime, attempt)
    import sentence_transformers
    original_embedding = sentence_transformers.SentenceTransformer

    def pinned_embedding(name, *args, **kwargs):
        if name != runtime["embedding_model"]:
            raise ValueError("Unexpected embedding model/fallback")
        kwargs["device"] = "cpu"
        model = original_embedding(name, *args, **kwargs)
        if model.get_sentence_embedding_dimension() != 384 or model.max_seq_length != 256 or any(str(p.dtype) != "torch.float32" for p in model.parameters()):
            raise ValueError("MiniLM shape/context/dtype changed")
        return model

    sentence_transformers.SentenceTransformer = pinned_embedding
    module = load_file("simplemem_original_system", HERE / "upstream/main.py")
    system = module.SimpleMemSystem(api_key="EMPTY", model=MODEL, base_url=runtime["api_base"],
                                   db_path=config.LANCEDB_PATH, table_name=config.MEMORY_TABLE_NAME,
                                   clear_db=True, enable_thinking=False)
    for component in (system.memory_builder, system.hybrid_retriever, system.answer_generator):
        if component.llm_client is not system.llm_client:
            raise ValueError("Unexpected separate native LLM client")
    compatibility = load_file("simplemem_location_compat_v3", HERE / "location_compat.py")
    compatibility.install(system.memory_builder, HERE / "upstream/simplemem/core/memory_builder.py",
                          attempt / "location_compat_audit.jsonl")
    calls.bind(system.llm_client, runtime["api_base"])
    imported = {name: str(Path(mod.__file__).resolve()) for name, mod in list(sys.modules.items())
                if name.startswith("simplemem.core") and getattr(mod, "__file__", None)}
    for name, filename in imported.items():
        if not Path(filename).is_relative_to(HERE / "upstream/simplemem/core"):
            raise ValueError(f"Imported non-pinned module: {name}")
    save_json(attempt / "native_runtime.json", {"config": {key: getattr(config, key) for key in (
        "WINDOW_SIZE", "OVERLAP_SIZE", "USE_STREAMING", "ENABLE_PLANNING", "ENABLE_REFLECTION",
        "MAX_PARALLEL_WORKERS", "MAX_RETRIEVAL_WORKERS", "MAX_REFLECTION_ROUNDS", "USE_JSON_FORMAT")},
        "imports": imported, "model": MODEL, "embedding_model": runtime["embedding_model"]})
    return system, importlib.import_module("simplemem.core.models.memory_entry").Dialogue


def require_fts(system):
    backend = system.vector_store.backend
    if backend.count() > 0 and backend._fts_initialized is not True:
        raise RuntimeError("Native Tantivy FTS index was not created; keyword retrieval is unavailable")

def seal(attempt: Path, identity: dict):
    files = [p for p in attempt.rglob("*") if p.is_file() and p.name not in {"completion.json", "console.log"}]
    save_json(attempt / "completion.json", {"identity": identity,
        "files_sha256": {p.relative_to(attempt).as_posix(): sha(p) for p in files}})


def verified(history: Path, identity: dict) -> dict | None:
    for path in sorted(history.glob("attempt_*/completion.json")):
        receipt = read_json(path)
        if receipt["identity"] != identity:
            raise ValueError("Completed attempt protocol/input changed")
        required = {"source.json", "query.json", "worker.json", "prediction.json", "memory.json",
                    "dialogues.json", "build_complete.json", "usage.json", "native_runtime.json", "llm_calls.jsonl", "location_compat_audit.jsonl"}
        if not required.issubset(receipt["files_sha256"]):
            raise ValueError("Incomplete native receipt")
        for name, expected in receipt["files_sha256"].items():
            target = (path.parent / name).resolve()
            if not target.is_relative_to(path.parent.resolve()) or sha(target) != expected:
                raise ValueError("Successful native artifact changed")
        prediction = read_json(path.parent / "prediction.json")
        if prediction.get("identity") != identity or prediction.get("status") != "generated" or not isinstance(prediction.get("hypothesis"), str) or not prediction["hypothesis"].strip():
            raise ValueError("Invalid completed prediction")
        return prediction
    return None


def worker(attempt: Path) -> int:
    spec = read_json(attempt / "worker.json")
    identity, runtime = spec["identity"], spec["runtime"]
    started, calls = time.monotonic(), None
    try:
        source, query = read_json(attempt / "source.json"), read_json(attempt / "query.json")
        protocol = read_json(attempt.parents[2] / "protocol.json")
        if (digest(protocol) != identity["protocol_sha256"] or protocol["runtime"] != runtime
                or digest(source) != identity["source_sha256"] or digest(query) != identity["query_sha256"]
                or source_hashes() != spec["source_files_sha256"]):
            raise ValueError("Worker inputs/source/protocol changed")
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.update(OPENAI_API_KEY="EMPTY", OPENAI_BASE_URL=local_endpoint(runtime["api_base"]),
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
            NO_PROXY="localhost,127.0.0.1,::1", no_proxy="localhost,127.0.0.1,::1",
            METER_RUN_ID=identity["protocol_sha256"], METER_METHOD="simplemem",
            METER_LLM_PROXY_ORIGIN=runtime["api_base"], METER_EMBEDDING_PROXY_ORIGIN=runtime["api_base"],
            METER_TIMING_JOURNAL=str(attempt / "timing.jsonl"))
        sys.path.insert(0, str(HARNESS / "source/MemoryData"))
        from utils.request_metering import install_request_metering, meter_operation
        install_request_metering()
        calls = Calls(attempt / "llm_calls.jsonl")
        with meter_operation("memory_init", sample_id=query["question_id"]):
            system, Dialogue = make_system(runtime, attempt, calls)
        dialogues = convert_dialogues(source, Dialogue)
        save_json(attempt / "dialogues.json", [item.model_dump(mode="json") for item in dialogues])
        with native_fallbacks(calls), meter_operation("memory_write", sample_id=query["question_id"]):
            system.add_dialogues(dialogues)
            system.finalize()
        calls.check()
        require_fts(system)
        save_json(attempt / "memory.json", [entry.model_dump(mode="json") for entry in system.get_all_memories()])
        save_json(attempt / "build_complete.json", {"identity": identity, "dialogues": len(dialogues),
            "processed_count": system.memory_builder.processed_count, "count_scope": "Native window count includes overlap; source coverage is dialogues.json",
            "fts_initialized": system.vector_store.backend._fts_initialized, "seconds": time.monotonic() - started})
        calls.phase = "qa"
        qa_started = time.monotonic()
        question = f'Question date: {query["question_date"]}\nQuestion: {query["question"]}'
        with native_fallbacks(calls), meter_operation("qa", sample_id=query["question_id"], question_id=query["question_id"]):
            hypothesis = system.ask(question)
        calls.check()
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("Native ask returned an empty answer")
        save_json(attempt / "usage.json", {"total_seconds": time.monotonic() - started,
            "qa_seconds": time.monotonic() - qa_started, "logical_calls": calls.count,
            "physical_usage": "Authoritative local proxy journal includes native/SDK retries and stream usage",
            "wall_time_scope": "Includes shared GPU queue waiting", "native_recovery_warnings": calls.warnings, "officially_judged": False})
        save_json(attempt / "prediction.json", {"question_id": query["question_id"], "identity": identity,
            "status": "generated", "hypothesis": hypothesis, "officially_judged": False,
            "native_answer_path": "Pinned SimpleMemSystem.ask -> HybridRetriever -> AnswerGenerator"})
        seal(attempt, identity)
        return 0
    except Exception:
        save_json(attempt / "failure.json", {"identity": identity, "traceback": traceback.format_exc(),
            "native_errors": calls.errors if calls else [], "native_recovery_warnings": calls.warnings if calls else [], "seconds": time.monotonic() - started})
        return 1

def run(args: argparse.Namespace) -> int:
    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Canonical dataset hash mismatch")
    records = json.loads(raw)
    by_id = {row["question_id"]: row for row in records}
    if len(records) != 500 or len(by_id) != 500:
        raise ValueError("Expected 500 unique histories")
    selected = read_json(args.ids_file) if args.ids_file else list(by_id)
    if not isinstance(selected, list) or not selected or any(not isinstance(qid, str) for qid in selected) or len(set(selected)) != len(selected) or set(selected) - by_id.keys():
        raise ValueError("Invalid selected IDs")
    embedding = args.embedding_model.resolve()
    if not embedding.is_dir() or embedding.name != MINILM_REVISION:
        raise ValueError("Expected pinned local MiniLM snapshot directory")
    runtime = {"method": "simplemem", "api_base": local_endpoint(args.api_base), "model": MODEL,
               "embedding_model": str(embedding), "embedding_revision": MINILM_REVISION}
    hashes = source_hashes()
    protocol = {"dataset_sha256": DATA_SHA256, "population_ids": list(by_id), "runtime": runtime,
                "source_files_sha256": hashes, "policy": POLICY,
                "embedding_config_sha256": {name: sha(embedding / name) for name in ("config.json", "modules.json", "sentence_bert_config.json")}}
    write_once(args.run_dir / "protocol.json", protocol)
    predictions, failures = [], []
    for qid in selected:
        row = by_id[qid]
        source = source_only(row)
        query = {key: row[key] for key in ("question_id", "question", "question_date")}
        identity = {"protocol_sha256": digest(protocol), "source_sha256": digest(source), "query_sha256": digest(query)}
        history = args.run_dir / "histories" / hashlib.sha256(qid.encode()).hexdigest()[:24]
        prediction = verified(history, identity)
        if prediction is None:
            attempt = next_attempt(history)
            for name, value in (("source.json", source), ("query.json", query), ("worker.json", {
                    "identity": identity, "runtime": runtime, "source_files_sha256": hashes})):
                save_json(attempt / name, value)
            with (attempt / "console.log").open("w", encoding="utf-8") as output:
                try:
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-dir", str(attempt.resolve())],
                                            stdout=output, stderr=subprocess.STDOUT, check=False, timeout=5400)
                except subprocess.TimeoutExpired:
                    save_json(attempt / "failure.json", {"identity": identity, "reason": "history_timeout_5400_seconds"})
                    result = SimpleNamespace(returncode=124)
            prediction = verified(history, identity)
            if result.returncode or prediction is None:
                failures.append({"question_id": qid, "attempt": str(attempt), "exit_code": result.returncode})
        if prediction is not None and not failures:
            predictions.append(prediction)
        save_json(args.run_dir / "predictions.json", predictions)
        save_json(args.run_dir / "failures.json", failures)
        save_json(args.run_dir / "status.json", {"method": "simplemem", "planned": len(selected), "generated": len(predictions),
                  "failed": len(failures), "population": 500, "officially_judged": 0,
                  "status": "generation_incomplete" if failures else "generation_complete" if len(predictions) == len(selected) else "running"})
        if failures:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--method", choices=["simplemem"], default="simplemem")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--api-base")
    parser.add_argument("--model", choices=[MODEL], default=MODEL)
    parser.add_argument("--embedding-model", type=Path, help="Pinned local MiniLM snapshot path")
    parser.add_argument("--ids-file", type=Path)
    args = parser.parse_args()
    if args.worker_dir:
        return worker(args.worker_dir.resolve())
    if any(getattr(args, key) is None for key in ("dataset", "run_dir", "api_base", "embedding_model")):
        parser.error("--dataset, --run-dir, --api-base and --embedding-model are required")
    args.run_dir = args.run_dir.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())