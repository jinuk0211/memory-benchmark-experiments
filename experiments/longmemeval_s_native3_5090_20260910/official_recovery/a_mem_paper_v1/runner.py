"""Pinned A-MEM paper pipeline with a label-free LongMemEval data adapter."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
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
from native_five import DATA_SHA256, digest, next_attempt, read_json, save_json, source_only, write_once

PIN = "0c8039f28fdcc08189a23c07a3437d9d2482f9c2"
MODEL = "Qwen/Qwen3.5-9B"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CLASSES = {"BaseLLMController", "OpenAIController", "LLMController", "MemoryNote",
           "SimpleEmbeddingRetriever", "AgenticMemorySystem"}
POLICY = {"paper_commit": PIN, "ingestion": "one complete turn per add_memory",
          "retrieve_k": 10, "category": 1, "answer_argument": "",
          "question_date": "Prefixed to the public question argument, not the paper prompt template",
          "native_max_tokens": 1000, "native_temperature": 0.7,
          "compatibility": ["Bind missing standard-library re global", "Pinned MiniLM on CPU FP32"],
          "native_errors": "Record and reject invalid histories; no corrective retries or scored fallback"}


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
    """Observe native requests and reject invalid output without repairing it."""
    def __init__(self, path: Path, endpoint: str):
        self.path, self.endpoint = path, endpoint
        self.errors: list[str] = []
        self.phase = "initialization"
        self.records: list[dict] = []

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


@contextmanager
def native_fallbacks(calls: Calls):
    """Observe original fallback branches without changing their returned values."""
    filename = str(HERE / "upstream/memory_layer.py")
    # Pinned original: metadata fallback assignments/return; evolution JSON fallback return.
    fallback_lines = {385, 396, 827}
    previous = sys.gettrace()

    def observe(frame, event, arg):
        if frame.f_code.co_filename != filename:
            return None
        if event == "line" and frame.f_lineno in fallback_lines:
            calls.errors.append(f"Native fallback: {frame.f_code.co_name}:{frame.f_lineno}")
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


def ingest_turns(agent, sessions: list[dict], calls: Calls) -> int:
    count = 0
    for session in sessions:
        stamp = datetime.strptime(session["date"], "%Y/%m/%d (%a) %H:%M").strftime("%Y%m%d%H%M")
        for turn in session["turns"]:
            calls.phase = "memory_write"
            agent.add_memory("Speaker " + turn["role"] + "says : " + turn["content"], time=stamp)
            calls.check()
            count += 1
    return count


def answer(agent, query: dict, calls: Calls) -> tuple[str, dict]:
    calls.phase = "qa"
    question = f'Question date: {query["question_date"]}\nQuestion: {query["question"]}'
    raw, prompt, context = agent.answer_question(question, category=1, answer="")
    calls.check()
    value = json.loads(raw)
    if set(value) != {"answer"} or not isinstance(value["answer"], str) or not value["answer"].strip():
        raise ValueError("Native QA did not return a nonempty answer")
    return value["answer"], {"raw": raw, "prompt": prompt, "retrieved_context": context}


def seal(attempt: Path, identity: dict) -> None:
    files = [p for p in attempt.iterdir() if p.is_file() and p.name not in {"completion.json", "console.log"}]
    save_json(attempt / "completion.json", {"identity": identity, "files_sha256": {p.name: sha(p) for p in files}})


def verified(history: Path, identity: dict) -> dict | None:
    for path in sorted(history.glob("attempt_*/completion.json")):
        receipt = read_json(path)
        if receipt["identity"] != identity:
            raise ValueError("Completed attempt belongs to another protocol/input")
        required = {"source.json", "query.json", "worker.json", "prediction.json", "memory.json",
                    "build_complete.json", "usage.json", "llm_calls.jsonl", "native_answer.json",
                    "retriever.pkl", "embeddings.npy"}
        if not required.issubset(receipt["files_sha256"]):
            raise ValueError("Incomplete successful-attempt receipt")
        for name, expected in receipt["files_sha256"].items():
            if Path(name).name != name or sha(path.parent / name) != expected:
                raise ValueError("Successful native artifact was modified")
        prediction = read_json(path.parent / "prediction.json")
        if prediction.get("identity") != identity or prediction.get("status") != "generated" or not prediction.get("hypothesis", "").strip():
            raise ValueError("Invalid completed prediction")
        return prediction
    return None


def worker(attempt: Path) -> int:
    spec = read_json(attempt / "worker.json")
    identity, runtime = spec["identity"], spec["runtime"]
    calls = Calls(attempt / "llm_calls.jsonl", runtime["api_base"])
    started = time.monotonic()
    try:
        source, query = read_json(attempt / "source.json"), read_json(attempt / "query.json")
        if digest(source) != identity["source_sha256"] or digest(query) != identity["query_sha256"] or source_hashes() != spec["source_files_sha256"]:
            raise ValueError("Worker inputs/source changed")
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.update(OPENAI_API_KEY="EMPTY", OPENAI_BASE_URL=local_endpoint(runtime["api_base"]),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                          NO_PROXY="localhost,127.0.0.1,::1", no_proxy="localhost,127.0.0.1,::1",
                          METER_RUN_ID=identity["protocol_sha256"], METER_METHOD="a_mem",
                          METER_LLM_PROXY_ORIGIN=runtime["api_base"], METER_EMBEDDING_PROXY_ORIGIN=runtime["api_base"],
                          METER_TIMING_JOURNAL=str(attempt / "timing.jsonl"))
        sys.path.insert(0, str(HARNESS / "source/MemoryData"))
        from utils.request_metering import install_request_metering, meter_operation
        install_request_metering()
        with meter_operation("memory_init", sample_id=query["question_id"]):
            agent = make_agent(Path(runtime["embedding_model"]), calls)
        with meter_operation("memory_write", sample_id=query["question_id"]):
            with native_fallbacks(calls):
                count = ingest_turns(agent, source, calls)
        memory = {key: vars(note) for key, note in agent.memory_system.memories.items()}
        save_json(attempt / "memory.json", memory)
        # Native retriever save keeps corpus/embeddings; note JSON avoids dynamic-class pickle.
        agent.memory_system.retriever.save(attempt / "retriever.pkl", attempt / "embeddings.npy")
        save_json(attempt / "build_complete.json", {"identity": identity, "turns": count, "seconds": time.monotonic() - started})
        qa_started = time.monotonic()
        with meter_operation("qa", sample_id=query["question_id"], question_id=query["question_id"]):
            hypothesis, detail = answer(agent, query, calls)
        save_json(attempt / "native_answer.json", detail)
        save_json(attempt / "usage.json", {"total_seconds": time.monotonic() - started,
                  "qa_seconds": time.monotonic() - qa_started, "calls": len(calls.records),
                  "meter_run_id": identity["protocol_sha256"], "officially_judged": False,
                  "embedding": "Local native SentenceTransformer CPU FP32, 384 dimensions, max_seq_length 256",
                  "wall_time_scope": "Includes any shared-GPU queuing; not isolated throughput"})
        save_json(attempt / "prediction.json", {"question_id": query["question_id"], "identity": identity,
                  "status": "generated", "hypothesis": hypothesis, "officially_judged": False,
                  "native_answer_path": "paper advancedMemAgent.answer_question(category=1, answer='')"})
        seal(attempt, identity)
        return 0
    except Exception:
        save_json(attempt / "failure.json", {"identity": identity, "native_errors": calls.errors,
                  "traceback": traceback.format_exc(), "seconds": time.monotonic() - started})
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
            prediction = verified(history, identity)
            if result.returncode or prediction is None:
                failures.append({"question_id": qid, "attempt": str(attempt), "exit_code": result.returncode})
        if prediction is not None and not failures:
            predictions.append(prediction)
        save_json(args.run_dir / "predictions.json", predictions)
        save_json(args.run_dir / "failures.json", failures)
        save_json(args.run_dir / "status.json", {"method": "a_mem", "planned": len(selected), "generated": len(predictions),
                  "failed": len(failures), "population": 500, "officially_judged": 0,
                  "status": "generation_incomplete" if failures else "generation_complete" if len(predictions) == len(selected) else "running"})
        if failures:
            return 1
    return 0


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
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())