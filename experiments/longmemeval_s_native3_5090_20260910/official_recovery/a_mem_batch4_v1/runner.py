"""A-Mem-batch4: four consecutive source turns per note; original paper algorithm unchanged."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import subprocess
import sys
import time
import traceback
import typing
from urllib.parse import urlsplit
import uuid

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
sys.path.insert(0, str(HARNESS))
from native_five import DATA_SHA256, digest, next_attempt, read_json, save_json, source_only, write_once  # noqa: E402

PIN = "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"
MODEL = "Qwen/Qwen3.5-9B"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CLASSES = {"BaseLLMController", "OpenAIController", "LLMController", "MemoryNote",
           "SimpleEmbeddingRetriever", "AgenticMemorySystem"}
VERSION = "a_mem_batch4_v1"
DISPLAY_NAME = "A-Mem-batch4"
BATCH_SIZE = 4
POLICY = {"harness_version": VERSION, "name": DISPLAY_NAME, "paper_commit": PIN,
          "ingestion": "Within each original session, four consecutive complete turns per add_memory/add_note; final remainder retained",
          "batch_size": BATCH_SIZE, "rendering": "Newline-join unchanged Speaker <role>says : <content> turn strings; original session timestamp",
          "source_coverage": "Raw label-free source plus exact session/turn ranges and hashes; one successful native note per batch",
          "retrieve_k": 10, "category": 1, "answer_argument": "",
          "question_date": "Prefixed to the public question argument, not the paper prompt template",
          "native_max_tokens": 1000, "native_temperature": 0.7,
          "compatibility": ["Bind missing standard-library re global", "Pinned MiniLM on CPU FP32"],
          "native_errors": "Reject transport and unhandled errors; preserve and count native handled metadata/evolution fallbacks",
          "handled_fallbacks": "Original native returned values, no repair, extra requests or parameter changes",
          "completion": "All source turns covered by verified batch4 notes and valid QA generated; not an error-free metadata/evolution claim"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    manifest = read_json(HERE / "source_manifest.json")
    if manifest["commit"] != PIN:
        raise ValueError("Unexpected paper revision")
    hashes = {}
    for name, item in manifest["files"].items():
        actual = sha(HERE / "upstream" / name)
        if actual != item["sha256"]:
            raise ValueError(f"Upstream changed: {name}")
        hashes[f"upstream/{name}"] = actual
    for name in ("runner.py", "source_manifest.json"):
        hashes[name] = sha(HERE / name)
    hashes["harness/native_five.py"] = sha(HARNESS / "native_five.py")
    hashes["harness/request_metering.py"] = sha(HARNESS / "source/MemoryData/utils/request_metering.py")
    return hashes


def local_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1" or not parsed.port):
        raise ValueError("A fixed loopback HTTP /v1 endpoint with port is required")
    return value.rstrip("/")


def compile_classes(path: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("Pinned class boundary differs")
    # Compile original AST nodes unchanged; no evaluator imports or module-level downloads.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Calls:
    """Record native requests, handled fallbacks and terminal errors separately."""
    def __init__(self, path: Path, endpoint: str):
        self.path, self.endpoint = path, endpoint
        self.errors: list[str] = []
        self.phase = "initialization"
        self.records: list[dict] = []
        self.handled_fallbacks: list[dict] = []

    def bind(self, controller) -> None:
        if controller.model != MODEL or str(controller.client.base_url).rstrip("/") != self.endpoint:
            raise ValueError("Native controller escaped the fixed local model/endpoint")
        original = controller.client.chat.completions.create

        def observed(**kwargs):
            record = {"phase": self.phase, "request": kwargs, "started_at": time.time()}
            try:
                if kwargs.get("model") != MODEL or kwargs.get("max_tokens") != 1000 or kwargs.get("temperature") != 0.7:
                    raise ValueError("Paper native generation policy changed")
                response = original(**kwargs)
                record["response"] = response.model_dump(mode="json")
                if response.choices[0].finish_reason != "stop":
                    record["warning"] = "Non-stop completion; original native parser determines acceptance"
                return response
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.errors.append(message)
                record["error"] = message
                raise
            finally:
                record["seconds"] = time.time() - record["started_at"]
                self.records.append(record)
                with self.path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        controller.client.chat.completions.create = observed

    def check(self) -> None:
        if self.errors:
            raise RuntimeError(f"Native requests failed ({len(self.errors)}); see llm_calls.jsonl")

    def degradation_counts(self) -> dict:
        return {"native_handled_fallback_count": len(self.handled_fallbacks),
                "native_metadata_fallback_count": sum(row["kind"] == "metadata" for row in self.handled_fallbacks),
                "native_evolution_fallback_count": sum(row["kind"] == "evolution" for row in self.handled_fallbacks)}

    def degradation(self) -> dict:
        return {"harness_version": POLICY["harness_version"], **self.degradation_counts(),
                "native_handled_fallbacks": self.handled_fallbacks,
                "counting": "One exact pinned native fallback branch per event; native return values unchanged",
                "error_free_metadata_and_evolution": False if self.handled_fallbacks else None}


@contextmanager
def native_fallbacks(calls: Calls):
    """Observe original fallback branches without changing their returned values."""
    filename = str(HERE / "upstream/memory_layer.py")
    # Pinned original: metadata fallback assignments/return; evolution JSON fallback return.
    fallback_sites = {("analyze_content", 385): "metadata", ("analyze_content", 396): "metadata",
                      ("process_memory", 827): "evolution"}
    previous = sys.gettrace()

    def observe(frame, event, arg):
        if frame.f_code.co_filename != filename:
            return None
        kind = fallback_sites.get((frame.f_code.co_name, frame.f_lineno))
        if event == "line" and kind:
            calls.handled_fallbacks.append({"kind": kind, "file": "upstream/memory_layer.py",
                                           "function": frame.f_code.co_name, "line": frame.f_lineno,
                                           "phase": calls.phase, "native_call_ordinal": len(calls.records)})
        return observe

    sys.settrace(observe)
    try:
        yield
    finally:
        sys.settrace(previous)

def make_agent(embedding_path: Path, calls: Calls):
    import numpy as np
    from abc import ABC, abstractmethod
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer

    def pinned_embedding(model_name):
        if model_name not in {"all-MiniLM-L6-v2", str(embedding_path)}:
            raise ValueError("Unexpected native embedding model")
        model = SentenceTransformer(str(embedding_path), device="cpu")
        if model.get_sentence_embedding_dimension() != 384 or model.max_seq_length != 256:
            raise ValueError("MiniLM dimension/context differs")
        if any(str(p.dtype) != "torch.float32" for p in model.parameters()):
            raise ValueError("MiniLM is not FP32")
        return model

    namespace = {"__name__": "a_mem_paper_native", "json": json, "re": re, "uuid": uuid,
                 "datetime": datetime, "np": np, "cosine_similarity": cosine_similarity,
                 "SentenceTransformer": pinned_embedding, "os": os, "pickle": pickle,
                 "Path": Path, "ABC": ABC, "abstractmethod": abstractmethod}
    namespace.update({name: getattr(typing, name) for name in ("List", "Dict", "Optional", "Literal", "Any", "Union")})
    compile_classes(HERE / "upstream/memory_layer.py", CLASSES, namespace)

    def memory_factory(**kwargs):
        kwargs["model_name"] = str(embedding_path)
        return namespace["AgenticMemorySystem"](**kwargs)

    qa = {"__name__": "a_mem_paper_qa", "json": json, "random": random,
          "AgenticMemorySystem": memory_factory, "LLMController": namespace["LLMController"]}
    compile_classes(HERE / "upstream/test_advanced.py", {"advancedMemAgent"}, qa)
    agent = qa["advancedMemAgent"](MODEL, "openai", 10, 0.5)
    for controller in (agent.memory_system.llm_controller.llm, agent.retriever_llm.llm):
        calls.bind(controller)
    return agent


def batch_plan(sessions: list[dict]) -> dict:
    batches = []
    turns = 0
    for session_index, session in enumerate(sessions):
        if (set(session) != {"session_id", "date", "turns"}
                or not isinstance(session["session_id"], str) or not isinstance(session["date"], str)
                or not isinstance(session["turns"], list)):
            raise ValueError("Expected exact label-free source session")
        stamp = datetime.strptime(session["date"], "%Y/%m/%d (%a) %H:%M").strftime("%Y%m%d%H%M")
        for turn in session["turns"]:
            if (set(turn) != {"role", "content"} or turn["role"] not in {"user", "assistant"}
                    or not isinstance(turn["content"], str)):
                raise ValueError("Expected unchanged user/assistant role and content")
        for start in range(0, len(session["turns"]), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(session["turns"]))
            original = session["turns"][start:stop]
            content = render_batch(original)
            batches.append({"batch_index": len(batches), "session_index": session_index,
                "session_id": session["session_id"], "session_date": session["date"], "native_time": stamp,
                "turn_start": start, "turn_stop": stop, "source_turns_sha256": digest(original),
                "note_content_sha256": hashlib.sha256(content.encode()).hexdigest()})
            turns += len(original)
    return {"harness_version": VERSION, "batch_size": BATCH_SIZE, "source_sha256": digest(sessions),
            "sessions": len(sessions), "turns": turns, "batch_count": len(batches), "batches": batches}


def render_batch(turns: list[dict]) -> str:
    return "\n".join("Speaker " + turn["role"] + "says : " + turn["content"] for turn in turns)


def ingest_batches(agent, sessions: list[dict], calls: Calls, attempt: Path) -> int:
    plan = batch_plan(sessions)
    write_once(attempt / "batch_manifest.json", plan)
    count = 0
    with (attempt / "batch_calls.jsonl").open("x", encoding="utf-8") as journal:
        for batch in plan["batches"]:
            save_json(attempt / "construction_progress.json", {"status": "adding", "batch": batch,
                      "completed_batches": batch["batch_index"], "completed_turns": count})
            calls.phase = "memory_write"
            start_call, started = len(calls.records), time.monotonic()
            previous_ids = set(agent.memory_system.memories)
            original = sessions[batch["session_index"]]["turns"][batch["turn_start"]:batch["turn_stop"]]
            agent.add_memory(render_batch(original), time=batch["native_time"])
            calls.check()
            added_ids = set(agent.memory_system.memories) - previous_ids
            if len(added_ids) != 1:
                raise ValueError("Original add_memory did not create exactly one note")
            note_id = added_ids.pop()
            journal.write(json.dumps({"batch": batch, "note_id": note_id,
                "llm_call_start": start_call, "llm_call_stop": len(calls.records),
                "seconds": time.monotonic() - started, "added_at": time.time()}, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            count += batch["turn_stop"] - batch["turn_start"]
            save_json(attempt / "construction_progress.json", {"status": "added", "batch": batch,
                      "completed_batches": batch["batch_index"] + 1, "completed_turns": count})
    return count


def validate_inputs(attempt: Path, identity: dict) -> tuple[list[dict], dict]:
    source, query, worker = (read_json(attempt / name) for name in ("source.json", "query.json", "worker.json"))
    protocol = read_json(attempt.parents[2] / "protocol.json")
    runtime = protocol["runtime"]
    expected_runtime = {"method": "a_mem", "api_base": local_endpoint(runtime["api_base"]), "model": MODEL,
                        "embedding_model": runtime["embedding_model"], "embedding_revision": MINILM_REVISION}
    embedding = Path(runtime["embedding_model"])
    if (digest(protocol.get("policy")) != digest(POLICY) or protocol.get("dataset_sha256") != DATA_SHA256
            or not isinstance(protocol.get("population_ids"), list)
            or len(protocol["population_ids"]) != 500
            or any(not isinstance(qid, str) for qid in protocol["population_ids"])
            or len(set(protocol["population_ids"])) != 500
            or digest(protocol) != identity["protocol_sha256"] or runtime != expected_runtime
            or embedding.name != MINILM_REVISION
            or protocol["embedding_config_sha256"] != {name: sha(embedding / name) for name in
                ("config.json", "modules.json", "sentence_bert_config.json")}
            or protocol["source_files_sha256"] != source_hashes()
            or worker != {"identity": identity, "runtime": runtime, "source_files_sha256": protocol["source_files_sha256"]}
            or digest(source) != identity["source_sha256"] or digest(query) != identity["query_sha256"]
            or query["question_id"] not in protocol["population_ids"]
            or attempt.parent.name != hashlib.sha256(query["question_id"].encode()).hexdigest()[:24]):
        raise ValueError("Batch4 protocol/source/query/worker identity differs")
    batch_plan(source)
    return source, query


def verify_batch_coverage(attempt: Path, identity: dict) -> None:
    source, _ = validate_inputs(attempt, identity)
    plan = batch_plan(source)
    if digest(read_json(attempt / "batch_manifest.json")) != digest(plan):
        raise ValueError("Batch manifest does not cover the unchanged source")
    journal = [json.loads(line) for line in (attempt / "batch_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    llm_calls = [json.loads(line) for line in (attempt / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    memory = read_json(attempt / "memory.json")
    if len(journal) != plan["batch_count"]:
        raise ValueError("Missing, repeated or extra successful batch calls")
    note_ids, previous_call = set(), 0
    for batch, row in zip(plan["batches"], journal):
        first, last = row.get("llm_call_start"), row.get("llm_call_stop")
        if (digest(row.get("batch")) != digest(batch) or type(first) is not int or type(last) is not int
                or first != previous_call or not first < last <= len(llm_calls)
                or not isinstance(row.get("note_id"), str) or row["note_id"] in note_ids
                or row["note_id"] not in memory
                or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                       for k in ("seconds", "added_at"))
                or any(call.get("phase") != "memory_write" or call.get("error") for call in llm_calls[first:last])):
            raise ValueError("Batch/native call/time proof differs")
        original = source[batch["session_index"]]["turns"][batch["turn_start"]:batch["turn_stop"]]
        note = memory[row["note_id"]]
        if note.get("content") != render_batch(original) or note.get("timestamp") != batch["native_time"]:
            raise ValueError("Native note lost source content/time")
        note_ids.add(row["note_id"])
        previous_call = last
    if set(memory) != note_ids:
        raise ValueError("Native memory contains foreign or missing batch notes")
    if any(call.get("phase") != "qa" or call.get("error") for call in llm_calls[previous_call:]):
        raise ValueError("Unaccounted native calls after construction")
    build = read_json(attempt / "build_complete.json")
    if (build.get("identity") != identity or type(build.get("turns")) is not int or build["turns"] != plan["turns"]
            or type(build.get("batches")) is not int or build["batches"] != plan["batch_count"]):
        raise ValueError("Build does not cover all source batches/turns")
    progress = read_json(attempt / "construction_progress.json")
    if not plan["batches"] or digest(progress) != digest({"status": "added", "batch": plan["batches"][-1],
            "completed_batches": plan["batch_count"], "completed_turns": plan["turns"]}):
        raise ValueError("Final construction progress is incomplete")


def answer(agent, query: dict, calls: Calls) -> tuple[str, dict]:
    calls.phase = "qa"
    question = f'Question date: {query["question_date"]}\nQuestion: {query["question"]}'
    raw, prompt, context = agent.answer_question(question, category=1, answer="")
    calls.check()
    value = json.loads(raw)
    if set(value) != {"answer"} or not isinstance(value["answer"], str) or not value["answer"].strip():
        raise ValueError("Native QA did not return a nonempty answer")
    return value["answer"], {"raw": raw, "prompt": prompt, "retrieved_context": context}


REQUIRED = {"source.json", "query.json", "worker.json", "prediction.json", "memory.json",
            "build_complete.json", "usage.json", "llm_calls.jsonl", "native_answer.json",
            "retriever.pkl", "embeddings.npy", "native_degradation.json", "console.log",
            "batch_manifest.json", "batch_calls.jsonl", "construction_progress.json"}


def attempt_files(attempt: Path) -> dict[str, str]:
    paths = list(attempt.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Native artifact symlink refused")
    return {path.relative_to(attempt).as_posix(): sha(path) for path in paths
            if path.is_file() and path != attempt / "completion.json"}


def finalize_attempt(attempt: Path, identity: dict) -> None:
    """Only the parent calls this after subprocess exit0 and closed console/cache files."""
    files = attempt_files(attempt)
    if not REQUIRED.issubset(files) or "failure.json" in files:
        raise ValueError("Incomplete or failed batch attempt")
    verify_batch_coverage(attempt, identity)
    prediction = read_json(attempt / "prediction.json")
    query = read_json(attempt / "query.json")
    if (prediction.get("question_id") != query["question_id"] or prediction.get("identity") != identity or prediction.get("status") != "generated"
            or prediction.get("variant") != VERSION or prediction.get("error") or prediction.get("error_type")
            or not isinstance(prediction.get("hypothesis"), str) or not prediction["hypothesis"].strip()):
        raise ValueError("Invalid batch prediction")
    write_once(attempt / "completion.json", {"identity": identity, "variant": VERSION,
               "created_after_worker_exit": True, "files_sha256": files})


def verified(history: Path, identity: dict) -> dict | None:
    for path in sorted(history.glob("attempt_*/completion.json")):
        receipt = read_json(path)
        if (receipt.get("identity") != identity or receipt.get("variant") != VERSION
                or receipt.get("created_after_worker_exit") is not True):
            raise ValueError("Completed attempt belongs to another protocol/input/version")
        files = attempt_files(path.parent)
        if not REQUIRED.issubset(files) or "failure.json" in files or receipt.get("files_sha256") != files:
            raise ValueError("Incomplete, failed or modified successful batch artifacts")
        verify_batch_coverage(path.parent, identity)
        prediction = read_json(path.parent / "prediction.json")
        query = read_json(path.parent / "query.json")
        if (prediction.get("identity") != identity or prediction.get("status") != "generated"
                or prediction.get("variant") != VERSION or prediction.get("question_id") != query["question_id"]
                or prediction.get("error") or prediction.get("error_type")
                or not isinstance(prediction.get("hypothesis"), str) or not prediction["hypothesis"].strip()):
            raise ValueError("Invalid completed batch prediction")
        return prediction
    return None


def worker(attempt: Path) -> int:
    spec = read_json(attempt / "worker.json")
    identity, runtime = spec["identity"], spec["runtime"]
    calls = Calls(attempt / "llm_calls.jsonl", runtime["api_base"])
    started = time.monotonic()
    try:
        source, query = validate_inputs(attempt, identity)
        if digest(source) != identity["source_sha256"] or digest(query) != identity["query_sha256"] or source_hashes() != spec["source_files_sha256"]:
            raise ValueError("Worker inputs/source changed")
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.update(OPENAI_API_KEY="EMPTY", OPENAI_BASE_URL=local_endpoint(runtime["api_base"]),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                          NO_PROXY="localhost,127.0.0.1,::1", no_proxy="localhost,127.0.0.1,::1",
                          METER_RUN_ID=identity["protocol_sha256"], METER_METHOD="a_mem_batch4",
                          METER_LLM_PROXY_ORIGIN=runtime["api_base"], METER_EMBEDDING_PROXY_ORIGIN=runtime["api_base"],
                          METER_TIMING_JOURNAL=str(attempt / "timing.jsonl"))
        sys.path.insert(0, str(HARNESS / "source/MemoryData"))
        from utils.request_metering import install_request_metering, meter_operation
        install_request_metering()
        with meter_operation("memory_init", sample_id=query["question_id"]):
            agent = make_agent(Path(runtime["embedding_model"]), calls)
        with meter_operation("memory_write", sample_id=query["question_id"]):
            with native_fallbacks(calls):
                count = ingest_batches(agent, source, calls, attempt)
        memory = {key: vars(note) for key, note in agent.memory_system.memories.items()}
        save_json(attempt / "memory.json", memory)
        # Native retriever save keeps corpus/embeddings; note JSON avoids dynamic-class pickle.
        agent.memory_system.retriever.save(attempt / "retriever.pkl", attempt / "embeddings.npy")
        save_json(attempt / "native_degradation.json", calls.degradation())
        save_json(attempt / "build_complete.json", {"identity": identity, "turns": count, "batches": batch_plan(source)["batch_count"],
                  "seconds": time.monotonic() - started, **calls.degradation_counts()})
        qa_started = time.monotonic()
        with meter_operation("qa", sample_id=query["question_id"], question_id=query["question_id"]):
            hypothesis, detail = answer(agent, query, calls)
        save_json(attempt / "native_answer.json", detail)
        save_json(attempt / "usage.json", {"total_seconds": time.monotonic() - started,
                  "qa_seconds": time.monotonic() - qa_started, "calls": len(calls.records),
                  **calls.degradation_counts(),
                  "meter_run_id": identity["protocol_sha256"], "officially_judged": False,
                  "embedding": "Local native SentenceTransformer CPU FP32, 384 dimensions, max_seq_length 256",
                  "wall_time_scope": "Includes any shared-GPU queuing; not isolated throughput"})
        save_json(attempt / "prediction.json", {"question_id": query["question_id"], "identity": identity,
                  "status": "generated", "hypothesis": hypothesis, "variant": VERSION, "officially_judged": False,
                  **calls.degradation_counts(),
                  "native_answer_path": "paper advancedMemAgent.answer_question(category=1, answer='')"})
        verify_batch_coverage(attempt, identity)
        return 0
    except Exception:
        save_json(attempt / "native_degradation.json", calls.degradation())
        save_json(attempt / "failure.json", {"identity": identity, "native_errors": calls.errors,
                  "traceback": traceback.format_exc(), "seconds": time.monotonic() - started,
                  **calls.degradation_counts()})
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
    runtime = {"method": "a_mem", "api_base": local_endpoint(args.api_base), "model": MODEL,
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
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-dir", str(attempt.resolve())],
                                        stdout=output, stderr=subprocess.STDOUT, check=False)
            completion_error = None
            if result.returncode == 0:
                try:
                    finalize_attempt(attempt, identity)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    completion_error = type(error).__name__
            prediction = verified(history, identity)
            if result.returncode or prediction is None:
                failures.append({"question_id": qid, "attempt": str(attempt), "exit_code": result.returncode, "completion_error": completion_error})
        if prediction is not None and not failures:
            predictions.append(prediction)
        save_json(args.run_dir / "predictions.json", predictions)
        save_json(args.run_dir / "failures.json", failures)
        save_json(args.run_dir / "status.json", {"method": "a_mem", "variant": VERSION, "name": DISPLAY_NAME, "planned": 500, "selected_count": len(selected), "generated": len(predictions),
                  "failed": len(failures), "population": 500, "officially_judged": 0,
                  "status": "generation_incomplete" if failures else "generation_complete" if len(predictions) == 500 else "subset_generation_complete" if len(predictions) == len(selected) else "running"})
        if failures:
            return 1
    return 0


@contextmanager
def execution_lock(run_dir: Path):
    """One Linux parent owns attempt allocation and aggregate writes for a run."""
    import fcntl
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "execution.lock"
    if path.is_symlink():
        raise ValueError("Execution lock symlink refused")
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--method", choices=["a_mem"], default="a_mem")
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
    with execution_lock(args.run_dir):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())