"""Pinned LangMem/Trustcall with role-preserving LongMemEval session inputs.

This is a cross-dataset library integration, not an official LongMemEval driver.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
SOURCE = HARNESS / "source/MemoryData"
sys.path.insert(0, str(HARNESS))
import native_five as harness
from native_five import DATA_SHA256, digest, next_attempt, read_json, save_json, source_only, write_once

PIN = "9d033b47d9ce53e37e92c92241b0496c0278932e"
MODEL = "Qwen/Qwen3.5-9B"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CLIENT_VERSIONS = {"langmem": "0.0.30", "trustcall": "0.0.39", "langchain-openai": "0.3.35",
                   "langchain-core": "0.3.86", "langgraph": "1.0.1", "openai": "2.54.0"}
POLICY = {
    "official_commit": PIN, "langmem_version": "0.0.30", "trustcall_version": "0.0.39",
    "ingestion": "One original-role message list per public session, in dataset order",
    "date_mapping": "Separate system message: Session {session_id} ({date}); source turns unchanged",
    "adaptation": "Explicit LongMemEval mapping; no official LongMemEval evaluation driver claimed",
    "source_turn_fields": ["role", "content"], "enable_inserts": True,
    "enable_deletes": False, "query_limit": 5, "max_steps": 1,
    "trustcall_max_attempts": 3, "memory_temperature": 0,
    "temperature_scope": "Existing benchmark ChatOpenAI setting, not an upstream universal default",
    "answer": "Unchanged benchmark retrieve_num=10, context packing and QA prompt integration",
    "timeout": "No transport timeout, retry, output cap or model parameter changes",
    "failure": "Preserve native retries; reject logged terminal errors and lost PatchDocs",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    manifest = read_json(HERE / "source_manifest.json")
    if (manifest["commit"] != PIN or manifest["langmem_version"] != "0.0.30"
            or manifest["trustcall_version"] != "0.0.39"):
        raise ValueError("Unexpected native source version")
    hashes = {}
    for name, item in manifest["files"].items():
        path = (HERE / "upstream" / name).resolve()
        if not path.is_relative_to(HERE / "upstream") or sha(path) != item["sha256"]:
            raise ValueError(f"Pinned source changed: {name}")
        hashes[f"upstream/{name}"] = item["sha256"]
    for name in ("runner.py", "source_manifest.json"):
        hashes[name] = sha(HERE / name)
    hashes.update({"harness/" + name: value for name, value in harness.source_hashes(SOURCE, "langmem").items()})
    return hashes


def client_versions() -> dict[str, str]:
    from importlib.metadata import version
    actual = {name: version(name) for name in CLIENT_VERSIONS}
    if actual != CLIENT_VERSIONS:
        raise ValueError(f"Client packages differ from the recorded experiment: {actual}")
    return actual


def local_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1" or not parsed.port):
        raise ValueError("A loopback HTTP /v1 endpoint with explicit port is required")
    return value.rstrip("/")


def load_native() -> None:
    for package in ("langmem", "trustcall"):
        if any(name == package or name.startswith(package + ".") for name in sys.modules):
            raise ValueError(f"Native package already loaded before pinning: {package}")
    sys.path.insert(0, str(HERE / "upstream/trustcall"))
    sys.path.insert(0, str(HERE / "upstream/langmem/src"))
    importlib.import_module("trustcall")
    importlib.import_module("langmem")
    imported_native()


def imported_native() -> dict:
    manifest = read_json(HERE / "source_manifest.json")["files"]
    imports = {}
    for name, module in list(sys.modules.items()):
        if not any(name == p or name.startswith(p + ".") for p in ("langmem", "trustcall")):
            continue
        path = Path(module.__file__).resolve()
        prefix = HERE / ("upstream/langmem/src/langmem" if name.startswith("langmem") else "upstream/trustcall/trustcall")
        if not path.is_relative_to(prefix):
            raise ValueError(f"Imported non-pinned native module: {name}")
        relative = path.relative_to(HERE / "upstream").as_posix()
        if relative not in manifest or sha(path) != manifest[relative]["sha256"]:
            raise ValueError(f"Native module hash differs: {name}")
        imports[name] = {"path": str(path), "sha256": manifest[relative]["sha256"]}
    if not {"langmem", "trustcall"}.issubset(imports):
        raise ValueError("Native modules not loaded")
    return imports


class NativeFailures(logging.Handler):
    """Observe terminal losses while preserving original successful repair paths."""
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records, self.warnings = [], []
        self.native_file = str((HERE / "upstream/trustcall/trustcall/_base.py").resolve())

    def emit(self, record):
        event = {"logger": record.name, "function": record.funcName,
                 "file": record.pathname, "line": record.lineno, "message": record.getMessage()}
        recoverable = (str(Path(record.pathname).resolve()) == self.native_file
                       and record.funcName == "_get_message_op" and record.lineno in {1476, 1480})
        lost_patch = (record.funcName == "_teardown"
                      and record.getMessage().startswith("Could not find existing schema for"))
        target = self.records if (record.levelno >= logging.ERROR and not recoverable) or lost_patch else self.warnings
        target.append(event)

    @contextmanager
    def observe(self):
        # Exact source hash is checked before use. These branches discard work or terminate repair.
        terminal = {435, 468, 795, 804, 823, 829, 1084, 1101}
        previous, previous_thread = sys.gettrace(), threading.gettrace()

        def trace(frame, event, arg):
            if frame.f_code.co_filename != self.native_file:
                return None
            if event == "line" and frame.f_lineno in terminal:
                self.records.append({"file": self.native_file, "line": frame.f_lineno,
                                     "function": frame.f_code.co_name, "event": "native_terminal_loss"})
            return trace

        sys.settrace(trace)
        threading.settrace(trace)
        try:
            yield
        finally:
            sys.settrace(previous)
            threading.settrace(previous_thread)

    def check(self):
        if self.records:
            raise RuntimeError("Native memory execution reported errors or discarded patches")


def session_messages(session: dict) -> list[dict]:
    """Keep dataset roles/content verbatim, including assistant-first sessions."""
    turns = []
    for turn in session["turns"]:
        if turn["role"] not in {"user", "assistant", "system"} or not isinstance(turn["content"], str):
            raise ValueError("Invalid source role/content")
        turns.append({"role": turn["role"], "content": turn["content"]})
    if not turns or not isinstance(session["date"], str) or not isinstance(session["session_id"], str):
        raise ValueError("Empty session or invalid public session metadata")
    return [{"role": "system", "content": f'Session {session["session_id"]} ({session["date"]})'}] + turns


def ingest_sessions(manager, source: list[dict], failures: NativeFailures, journal: Path) -> dict:
    turns = 0
    for index, session in enumerate(source):
        messages = session_messages(session)
        started = time.monotonic()
        puts = manager.invoke({"messages": messages})
        failures.check()
        if not isinstance(puts, list):
            raise ValueError("Native manager returned an invalid result type")
        turns += len(session["turns"])
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"session_index": index, "session_id": session["session_id"],
                "messages_sha256": digest(messages), "source_turns": len(session["turns"]),
                "native_puts": len(puts), "seconds": time.monotonic() - started}) + "\n")
    return {"sessions": len(source), "source_turns": turns}


class Calls:
    """Guard local routing and observe native SDK calls without changing kwargs."""
    def __init__(self, path: Path, runtime: dict):
        self.path, self.runtime = path, runtime
        self.lock, self.count, self.phase = threading.Lock(), 0, "memory_write"
        self.path.touch(exist_ok=False)

    def record(self, value: dict):
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")

    def validate(self, kind: str, client, kwargs: dict):
        expected_base = self.runtime["api_base" if kind == "chat" else "embedding_api_base"]
        expected_model = MODEL if kind == "chat" else EMBEDDING_MODEL
        if (local_endpoint(str(client.base_url)) != expected_base or client.api_key != "EMPTY"
                or kwargs.get("model") != expected_model):
            raise ValueError("Native model request escaped fixed local routing")
        if kind == "embedding":
            value = kwargs.get("input")
            if not (isinstance(value, str) or isinstance(value, list) and all(isinstance(x, str) for x in value)):
                raise ValueError("Embedding input must remain original strings, not tokenizer IDs")

    def bind(self, client, asynchronous: bool):
        for kind, resource in (("chat", client.chat.completions), ("embedding", client.embeddings)):
            original = resource.create

            def begin(kwargs, kind=kind):
                self.validate(kind, client, kwargs)
                with self.lock:
                    self.count += 1
                    number = self.count
                entry = {"call": number, "phase": self.phase, "kind": kind,
                         "model": kwargs["model"], "base_url": str(client.base_url),
                         "request_parameters": {k: kwargs[k] for k in ("temperature", "max_tokens", "max_completion_tokens", "stream", "tool_choice") if k in kwargs},
                         "tool_names": [x.get("function", {}).get("name") for x in kwargs.get("tools", [])]}
                self.record({**entry, "event": "begin"})
                return entry, time.monotonic()

            def end(entry, started, result=None, error=None):
                self.record({**entry, "event": "end", "seconds": time.monotonic() - started,
                             "response": result.model_dump(mode="json") if hasattr(result, "model_dump") else None,
                             "error_class": type(error).__name__ if error else None})

            if asynchronous:
                async def observed(*args, _original=original, _begin=begin, _end=end, **kwargs):
                    entry, started = _begin(kwargs)
                    try:
                        result = await _original(*args, **kwargs)
                    except Exception as exc:
                        _end(entry, started, error=exc)
                        raise
                    _end(entry, started, result=result)
                    return result
            else:
                def observed(*args, _original=original, _begin=begin, _end=end, **kwargs):
                    entry, started = _begin(kwargs)
                    try:
                        result = _original(*args, **kwargs)
                    except Exception as exc:
                        _end(entry, started, error=exc)
                        raise
                    _end(entry, started, result=result)
                    return result
            resource.create = observed

    @contextmanager
    def clients(self):
        import openai
        originals = [(cls, cls.__init__) for cls in (openai.OpenAI, openai.AsyncOpenAI)]
        for (cls, original), asynchronous in zip(originals, (False, True)):

            def initialize(client, *args, _original=original, _async=asynchronous, **kwargs):
                _original(client, *args, **kwargs)
                if local_endpoint(str(client.base_url)) not in {self.runtime["api_base"], self.runtime["embedding_api_base"]} or client.api_key != "EMPTY":
                    raise ValueError("Native client escaped fixed local routing")
                self.bind(client, _async)
            cls.__init__ = initialize
        try:
            yield
        finally:
            for cls, original in originals:
                cls.__init__ = original


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
                    "native_sessions.json", "session_calls.jsonl", "build_complete.json", "usage.json",
                    "native_runtime.json", "native_config.json", "native_dataset_config.json", "llm_calls.jsonl"}
        if not required.issubset(receipt["files_sha256"]) or (path.parent / "failure.json").exists():
            raise ValueError("Incomplete or failed native receipt")
        actual_files = {p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*")
                        if p.is_file() and p.name not in {"completion.json", "console.log"}}
        if actual_files != set(receipt["files_sha256"]):
            raise ValueError("Successful artifact inventory changed")
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
    started, failures = time.monotonic(), NativeFailures()
    logging.getLogger().addHandler(failures)
    agent = None
    try:
        source, query = read_json(attempt / "source.json"), read_json(attempt / "query.json")
        protocol = read_json(attempt.parents[2] / "protocol.json")
        if (digest(protocol) != identity["protocol_sha256"] or protocol["runtime"] != runtime
                or digest(source) != identity["source_sha256"] or digest(query) != identity["query_sha256"]
                or protocol["source_files_sha256"] != spec["source_files_sha256"]
                or source_hashes() != spec["source_files_sha256"]):
            raise ValueError("Worker inputs/source/protocol changed")
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.update(OPENAI_API_KEY="EMPTY", OPENAI_BASE_URL=local_endpoint(runtime["api_base"]),
            EMBEDDING_BASE_URL=local_endpoint(runtime["embedding_api_base"]), BASELINE_STRICT_COMPARISON="1",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
            LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING_V2="false",
            LANGCHAIN_TRACING="false", LANGSMITH_TRACING="false", SKIP_SBERT_SIM="1",
            NO_PROXY="localhost,127.0.0.1,::1", no_proxy="localhost,127.0.0.1,::1",
            METER_RUN_ID=identity["protocol_sha256"], METER_METHOD="langmem",
            METER_LLM_PROXY_ORIGIN=runtime["api_base"], METER_EMBEDDING_PROXY_ORIGIN=runtime["embedding_api_base"],
            METER_TIMING_JOURNAL=str(attempt / "timing.jsonl"))
        sys.path.insert(0, str(SOURCE))
        os.chdir(SOURCE)
        from utils.request_metering import install_request_metering, meter_operation
        install_request_metering()
        calls = Calls(attempt / "llm_calls.jsonl", runtime)
        with calls.clients(), failures.observe():
            load_native()
            packages = client_versions()
            from utils.agent import AgentWrapper
            args = argparse.Namespace(**runtime, source_root=SOURCE)
            config, dataset = harness.build_config(args, attempt)
            save_json(attempt / "native_config.json", config)
            save_json(attempt / "native_dataset_config.json", dataset)
            state = attempt / "memory"
            state.mkdir()
            with meter_operation("memory_init", sample_id=query["question_id"]):
                agent = harness.create_agent(AgentWrapper, config, dataset, state, "langmem")
                manager = agent.benchmark_memory.manager
                if (manager.query_limit != 5 or manager.enable_deletes is not False
                        or manager.enable_inserts is not True or manager.query_model is not None
                        or manager.phases or manager.model.model_name != MODEL or manager.model.temperature != 0):
                    raise ValueError("Documented native manager configuration changed")
                tokenizer = harness.tokenizer_runtime(agent.tokenizer, runtime["tokenizer"])
            save_json(attempt / "native_runtime.json", {"imports": imported_native(), "tokenizer": tokenizer,
                "runtime": runtime, "policy": POLICY, "python": sys.version, "packages": packages})
            save_json(attempt / "native_sessions.json", [session_messages(s) for s in source])
            with meter_operation("memory_write", sample_id=query["question_id"]):
                counts = ingest_sessions(manager, source, failures, attempt / "session_calls.jsonl")
            with meter_operation("memory_finalize", sample_id=query["question_id"]):
                agent.save_agent()
            failures.check()
            save_json(attempt / "memory.json", [{"key": str(x.key), "value": x.value}
                for x in agent.benchmark_memory._live_items()])
            save_json(attempt / "build_complete.json", {"identity": identity, **counts,
                "seconds": time.monotonic() - started})
            calls.phase = "qa"
            qa_started = time.monotonic()
            with meter_operation("qa", sample_id=query["question_id"], question_id=query["question_id"]):
                hypothesis, detail = harness.native_answer(agent, "langmem", query)
            failures.check()
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                raise ValueError("Benchmark QA returned an empty answer")
            save_json(attempt / "native_runtime.json", {"imports": imported_native(), "tokenizer": tokenizer,
                "runtime": runtime, "policy": POLICY, "python": sys.version, "packages": packages})
            save_json(attempt / "usage.json", {"total_seconds": time.monotonic() - started,
                "qa_seconds": time.monotonic() - qa_started, "logical_sdk_calls": calls.count,
                "physical_usage": "Authoritative local proxy journal includes SDK retries",
                "wall_time_scope": "Includes shared GPU queue waiting", "native_qa_detail": detail,
                "native_recovery_warnings": failures.warnings,
                "officially_judged": False})
            save_json(attempt / "prediction.json", {"question_id": query["question_id"], "identity": identity,
                "status": "generated", "hypothesis": hypothesis, "officially_judged": False,
                "native_answer_path": POLICY["answer"]})
        if agent is not None:
            agent.close()
            agent = None
        failures.check()
        seal(attempt, identity)
        return 0
    except Exception:
        save_json(attempt / "failure.json", {"identity": identity, "traceback": traceback.format_exc(),
            "native_errors": failures.records, "native_recovery_warnings": failures.warnings,
            "seconds": time.monotonic() - started})
        return 1
    finally:
        logging.getLogger().removeHandler(failures)
        if agent is not None:
            agent.close()


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
    tokenizer = args.tokenizer.resolve()
    if not tokenizer.is_dir() or tokenizer.name != TOKENIZER_REVISION:
        raise ValueError("Expected pinned Qwen tokenizer snapshot path")
    runtime = {"method": "langmem", "api_base": local_endpoint(args.api_base), "model": MODEL,
        "embedding_api_base": local_endpoint(args.embedding_api_base), "embedding_model": EMBEDDING_MODEL,
        "embedding_dims": 384, "tokenizer": str(tokenizer)}
    hashes = source_hashes()
    protocol = {"dataset_sha256": DATA_SHA256, "population_ids": list(by_id), "runtime": runtime,
                "source_files_sha256": hashes, "policy": POLICY,
                "tokenizer_config_sha256": {name: sha(tokenizer / name) for name in ("tokenizer_config.json", "tokenizer.json")}}
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
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-dir", str(attempt.resolve())],
                                        stdout=output, stderr=subprocess.STDOUT, check=False)
            prediction = verified(history, identity)
            if result.returncode or prediction is None:
                failures.append({"question_id": qid, "attempt": str(attempt), "exit_code": result.returncode})
        if prediction is not None and not failures:
            predictions.append(prediction)
        save_json(args.run_dir / "predictions.json", predictions)
        save_json(args.run_dir / "failures.json", failures)
        save_json(args.run_dir / "status.json", {"method": "langmem", "planned": len(selected), "generated": len(predictions),
                  "failed": len(failures), "population": 500, "officially_judged": 0,
                  "status": "generation_incomplete" if failures else "generation_complete" if len(predictions) == len(selected) else "running"})
        if failures:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--method", choices=["langmem"], default="langmem")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--api-base")
    parser.add_argument("--embedding-api-base")
    parser.add_argument("--model", choices=[MODEL], default=MODEL)
    parser.add_argument("--embedding-model", choices=[EMBEDDING_MODEL], default=EMBEDDING_MODEL)
    parser.add_argument("--embedding-dims", type=int, choices=[384], default=384)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--ids-file", type=Path)
    args = parser.parse_args()
    if args.worker_dir:
        return worker(args.worker_dir.resolve())
    if any(getattr(args, key) is None for key in ("dataset", "run_dir", "api_base", "embedding_api_base", "tokenizer")):
        parser.error("--dataset, --run-dir, --api-base, --embedding-api-base and --tokenizer are required")
    args.run_dir = args.run_dir.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())