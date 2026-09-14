"""Native five-method LongMemEval-S runner; no replacement common reader.

E-Mem/SimpleMem retain native answering. A-MEM, Mem0 and LangMem retain
native memory APIs and their existing benchmark answer integrations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator

METHODS = ("e_mem", "simplemem", "langmem", "mem0", "a_mem")
POLICY_OVERRIDES = {
    "simplemem": {"simplemem_window_size": 40, "simplemem_use_streaming": True,
                  "simplemem_max_parallel_workers": 16, "simplemem_max_retrieval_workers": 8},
    "mem0": {"temperature": 0.1},
}
POLICY_PROVENANCE = {
    "simplemem": {"source": "methods/simplemem/upstream/config.py.example",
                  "prior_yaml": {"window_size": 8, "use_streaming": False,
                                 "parallel_workers": 4, "retrieval_workers": 3},
                  "selected_upstream_sample": {"window_size": 40, "use_streaming": True,
                                              "parallel_workers": 16, "retrieval_workers": 8}},
    "langmem": {"source": "methods/langmem/src/langmem/knowledge/extraction.py:create_memory_store_manager",
               "prior_adapter_enable_deletes": True, "selected_upstream_default_enable_deletes": False,
               "override": "Scoped constructor keyword override; original adapter files unchanged"},
    "mem0": {"source": "methods/mem0/source/mem0/configs/llms/base.py",
             "prior_yaml_temperature": 0.7, "selected_upstream_default_temperature": 0.1},
    "a_mem": {"answer_scope": "Existing adapter QA integration, not an upstream end-to-end answer API",
              "retained_adapter_settings": {"retrieve_k": 2, "answer_max_tokens": 1000,
                                             "answer_context_character_cap": 60000}},
    "e_mem": {"source": "methods/e-mem/src/conversation_manager/factory.py",
              "retained_defaults": {"max_blocks": 5, "context_window": 32768,
                                    "block_size_ratio": 0.125, "overlap_ratio": 0.1,
                                    "summary_max_tokens": 8192, "query_max_tokens": 8192,
                                    "aggregation_max_tokens": 2048, "chat_max_new_tokens": 1024}},
}
DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
NATIVE_PATHS = {
    "e_mem": "EMemAdapter.manager.chat: native tools, block reasoning, aggregation, answer",
    "simplemem": "SimpleMemAdapter.ask -> SimpleMemSystem.ask",
    "a_mem": "Native A-MEM memory API + existing adapter ask_with_retrieved_context integration",
    "mem0": "Native Mem0 memory API + existing benchmark answer integration",
    "langmem": "Native LangMem memory API + existing benchmark answer integration",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_once(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing identity/input differs: {path}")
    else:
        save_json(path, value)


def source_only(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Whitelist source fields before a worker sees data; discard all labels."""
    sessions, dates, ids = (record[name] for name in
                            ("haystack_sessions", "haystack_dates", "haystack_session_ids"))
    if not len(sessions) == len(dates) == len(ids):
        raise ValueError("Source session/date/ID counts differ")
    return [{"session_id": sid, "date": date,
             "turns": [{"role": turn["role"], "content": turn["content"]} for turn in turns]}
            for sid, date, turns in zip(ids, dates, sessions)]


def source_chunks(sessions: list[dict[str, Any]]) -> Iterator[tuple[str, str]]:
    """Existing official loader's 4096-character complete-turn chunks."""
    for session in sessions:
        header = f'Session {session["session_id"]} ({session["date"]})\n'
        lines, size = [], 0
        for turn in session["turns"]:
            line = f'{turn["role"]}: {turn["content"]}'
            if lines and size + len(line) > 4096:
                yield session["date"], header + "\n".join(lines)
                lines, size = [], 0
            lines.append(line)
            size += len(line)
        if lines:
            yield session["date"], header + "\n".join(lines)


def ingest(agent: Any, method: str, date: str, text: str, question_id: str) -> None:
    if method in ("e_mem", "langmem"):
        agent.benchmark_memory.add_chunk(text)
    elif method == "simplemem":
        agent.simplemem.add_chunk(text, timestamp=date)
    elif method == "a_mem":
        stamp = datetime.strptime(date, "%Y/%m/%d (%a) %H:%M").strftime("%Y%m%d%H%M")
        agent.a_mem.add_chunk(text, timestamp=stamp)
    elif method == "mem0":
        agent.send_message(text, memorizing=True, context_id=question_id)
    else:
        raise ValueError(method)


def native_answer(agent: Any, method: str, query: dict[str, str]) -> tuple[str, Any]:
    question = f'Question date: {query["question_date"]}\nQuestion: {query["question"]}'
    if method == "e_mem":
        with agent.benchmark_memory._storage_scope():
            answer = agent.benchmark_memory.manager.chat(question)
        return answer, {"last_queried_memory": agent.benchmark_memory.manager.last_queried_memory}
    if method == "simplemem":
        return agent.simplemem.ask(question), None
    if method == "a_mem":
        return agent.a_mem.ask(question), None
    if method in ("mem0", "langmem"):
        response = agent.send_message(
            question, memorizing=False, query_id=query["question_id"],
            context_id=query["question_id"],
            eval_metadata={"question_id": query["question_id"],
                           "question_date": query["question_date"]})
        return response["output"], response
    raise ValueError(method)


def verified_prediction(history: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    for path in sorted(history.glob("attempt_*/prediction.json")):
        prediction = read_json(path)
        if prediction.get("identity") != identity:
            raise ValueError(f"Prediction protocol/source/query differs: {path}")
        if (prediction.get("status") == "generated" and isinstance(prediction.get("hypothesis"), str)
                and prediction["hypothesis"].strip()):
            return prediction
    return None


def next_attempt(history: Path) -> Path:
    number = max((int(p.name.split("_")[1]) for p in history.glob("attempt_*") if p.is_dir()), default=0) + 1
    attempt = history / f"attempt_{number:04d}"
    attempt.mkdir(parents=True)
    return attempt


class NativeErrors(logging.Handler):
    """Mark swallowed native errors as failures instead of scoring fallbacks."""
    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def source_hashes(root: Path, method: str) -> dict[str, str]:
    folders = [root / "utils", root / "benchmark/memoryagentbench/prompts", root / "methods" / method]
    if method == "e_mem":
        folders.append(root / "methods/e-mem/src")
    files = {Path(__file__).resolve(), root / "scripts/run_six_baselines.py",
             root / "benchmark/longmemeval/config/LongMemEval_s_official.yaml"}
    if method == "simplemem":
        files.add(root / "methods/simplemem/upstream/config.py.example")
    files.update((root / "config").glob("*.yaml"))
    for folder in folders:
        files.update(folder.rglob("*.py"))
    return {str(p.relative_to(root)) if p.is_relative_to(root) else "native_five.py":
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def build_config(args: argparse.Namespace, attempt: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import yaml
    from scripts.run_six_baselines import METHODS as CONFIGS, build_agent_config

    native = yaml.safe_load((args.source_root / "config" / CONFIGS[args.method]).read_text(encoding="utf-8-sig"))
    supplied = argparse.Namespace(model=args.model, base_url=args.api_base,
                                  embedding_base_url=args.embedding_api_base,
                                  embedding_model=args.embedding_model, embedding_dim=args.embedding_dims)
    config = build_agent_config(args.method, supplied)
    # Undo generic helper overrides of method policy; endpoint/backbone changes stay.
    config.update({key: native[key] for key in
                   ("temperature", "input_length_limit", "buffer_length", "agent_chunk_size")
                   if key in native})
    config.update(POLICY_OVERRIDES.get(args.method, {}))
    config.update(artifact_root=str(attempt), output_dir=str(attempt / "outputs"),
                  tokenizer_model=args.tokenizer or args.model, tokenizer_encoding=None,
                  e_mem_tokenizer_model=args.tokenizer or args.model,
                  e_mem_memory_model=args.model, record_llm_io=True)
    dataset = yaml.safe_load((args.source_root / "benchmark/longmemeval/config/LongMemEval_s_official.yaml").read_text(encoding="utf-8-sig"))
    return config, dataset


def create_agent(factory: Any, config: dict[str, Any], dataset: dict[str, Any],
                 state: Path, method: str) -> Any:
    """Use the library default for deletion without modifying the archived adapter."""
    if method != "langmem":
        return factory(config, dataset, load_agent_from=str(state))
    importlib.import_module("methods.langmem.langmem_adapter")
    langmem = importlib.import_module("langmem")
    original = langmem.create_memory_store_manager

    def upstream_default(*args: Any, **kwargs: Any) -> Any:
        kwargs["enable_deletes"] = False
        return original(*args, **kwargs)

    langmem.create_memory_store_manager = upstream_default
    try:
        return factory(config, dataset, load_agent_from=str(state))
    finally:
        langmem.create_memory_store_manager = original


def tokenizer_runtime(tokenizer: Any, pinned_model: str) -> dict[str, Any]:
    """Reject generic tokenizer fallbacks and retain reproducible CPU evidence."""
    from utils.provider_utils import HuggingFaceTokenizerAdapter

    if not isinstance(tokenizer, HuggingFaceTokenizerAdapter):
        kind = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
        raise RuntimeError(f"Expected HuggingFaceTokenizerAdapter; refusing tokenizer fallback: {kind}")
    backend = tokenizer._tokenizer
    sample = "Memory runtime check: 2023-05-20, yesterday; 서울 café."
    ids = tokenizer.encode(sample, add_special_tokens=False)
    if not isinstance(ids, list) or not ids or any(type(token_id) is not int for token_id in ids):
        raise RuntimeError("HF tokenizer did not return a nonempty integer token sequence")
    return {
        "tokenizer_adapter": {"module": type(tokenizer).__module__, "class": type(tokenizer).__name__},
        "tokenizer_backend": {"module": type(backend).__module__, "class": type(backend).__name__,
                              "name_or_path": getattr(backend, "name_or_path", None)},
        "pinned_tokenizer_model": pinned_model,
        "sample_text": sample, "sample_encode_ids": ids, "add_special_tokens": False,
    }



def worker(attempt: Path) -> int:
    spec = read_json(attempt / "worker.json")
    args = argparse.Namespace(**spec["runtime"])
    args.source_root = Path(args.source_root)
    source, query = read_json(attempt / "source.json"), read_json(attempt / "query.json")
    identity = spec["identity"]
    if digest(source) != identity["source_sha256"] or digest(query) != identity["query_sha256"]:
        raise ValueError("Worker input changed")
    if source_hashes(args.source_root, args.method) != spec["source_files_sha256"]:
        raise ValueError("Frozen runner/adapter source changed")
    os.chdir(args.source_root)
    sys.path.insert(0, str(args.source_root))
    os.environ.update(
        OPENAI_BASE_URL=args.api_base, EMBEDDING_BASE_URL=args.embedding_api_base,
        MEM0_DIR=str(attempt / "memory/mem0_home"), BASELINE_STRICT_COMPARISON="1",
        METER_RUN_ID=identity["protocol_sha256"], METER_METHOD=args.method,
        METER_LLM_PROXY_ORIGIN=args.api_base, METER_EMBEDDING_PROXY_ORIGIN=args.embedding_api_base,
        METER_TIMING_JOURNAL=str(attempt / "timing.jsonl"))
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("SIMPLEMEM_API_KEY", os.environ["OPENAI_API_KEY"])
    os.environ.setdefault("SIMPLEMEM_EMBEDDING_API_KEY", os.environ["OPENAI_API_KEY"])
    from utils.request_metering import install_request_metering, meter_operation
    install_request_metering()
    from utils.agent import AgentWrapper
    config, dataset = build_config(args, attempt)
    save_json(attempt / "native_config.json", config)
    save_json(attempt / "native_dataset_config.json", dataset)
    state = attempt / "memory"
    state.mkdir()
    errors = NativeErrors()
    logging.getLogger().addHandler(errors)
    agent = None
    started = time.monotonic()
    try:
        with meter_operation("memory_init", sample_id=query["question_id"]):
            agent = create_agent(AgentWrapper, config, dataset, state, args.method)
            runtime = tokenizer_runtime(agent.tokenizer, config['tokenizer_model'])
            save_json(attempt / 'native_runtime.json', runtime)
        with meter_operation("memory_write", sample_id=query["question_id"]):
            count = 0
            for date, text in source_chunks(source):
                ingest(agent, args.method, date, text, query["question_id"])
                count += 1
            agent.save_agent()
        if errors.records:
            raise RuntimeError(f"Native construction logged errors: {errors.records}")
        save_json(attempt / "build_complete.json",
                  {"identity": identity, "chunks": count, "seconds": time.monotonic() - started})
        answer_started = time.monotonic()
        with meter_operation("qa", sample_id=query["question_id"], question_id=query["question_id"]):
            answer, detail = native_answer(agent, args.method, query)
        if errors.records:
            raise RuntimeError(f"Native answering logged errors: {errors.records}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Native answer must be nonempty text")
        save_json(attempt / "native_answer_detail.json", detail)
        save_json(attempt / "usage.json", {
            "native_reported": detail if args.method in ("mem0", "langmem") else None,
            "scope": "Native counters can exclude construction/internal calls; physical proxy journal is authoritative.",
            "meter_run_id": identity["protocol_sha256"], "meter_method": args.method,
            "sample_id": query["question_id"], "timing_journal": "timing.jsonl",
            "total_seconds": time.monotonic() - started,
            "answer_seconds": time.monotonic() - answer_started})
        save_json(attempt / "prediction.json", {
            "identity": identity, "question_id": query["question_id"], "status": "generated",
            "hypothesis": answer, "officially_judged": False,
            "native_answer_path": NATIVE_PATHS[args.method],
            "completed_at": datetime.now(timezone.utc).isoformat()})
        return 0
    except Exception:
        save_json(attempt / "failure.json", {
            "identity": identity, "traceback": traceback.format_exc(), "native_errors": errors.records})
        return 1
    finally:
        logging.getLogger().removeHandler(errors)
        if agent is not None:
            agent.close()


def run(args: argparse.Namespace) -> int:
    dataset_bytes = args.dataset.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset is not the frozen complete LongMemEval-S release")
    records = json.loads(dataset_bytes)
    by_id = {str(row["question_id"]): row for row in records}
    if len(records) != 500 or len(by_id) != 500:
        raise ValueError("Expected 500 unique LongMemEval-S question IDs")
    selected = list(by_id)
    if args.ids_file:
        selected = read_json(args.ids_file)
        if not isinstance(selected, list) or not all(isinstance(qid, str) for qid in selected):
            raise ValueError("--ids-file must contain a JSON array of question IDs")
        if not selected or len(set(selected)) != len(selected) or set(selected) - by_id.keys():
            raise ValueError("IDs must be a nonempty unique subset of the frozen 500")
    args.source_root, args.run_dir = args.source_root.resolve(), args.run_dir.resolve()
    hashes = source_hashes(args.source_root, args.method)
    runtime = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
               if key in ("method", "source_root", "api_base", "embedding_api_base",
                          "model", "embedding_model", "embedding_dims", "tokenizer")}
    protocol = {
        "runtime": runtime, "dataset_sha256": DATA_SHA256, "population_ids": list(by_id),
        "source_files_sha256": hashes, "native_answer_path": NATIVE_PATHS[args.method],
        "method_policy_provenance": POLICY_PROVENANCE[args.method],
        "method_policy_overrides": POLICY_OVERRIDES.get(args.method, {}),
        "environment_changes": {
            "backbone": "Qwen3.5-9B FP16; thinking disabled at serving boundary",
            "embedding": "MiniLM native 256-token input, 384 dimensions",
            "integration_input_capacity": "Native YAML policy retained; server overcapacity is a failure",
            "tokenizer": args.tokenizer or args.model,
            "native_method_temperature": "Recorded YAML except Mem0 upstream default 0.1; see policy provenance",
            "common_reader": False, "library_answer_integration": args.method in ("mem0", "langmem", "a_mem")}}
    write_once(args.run_dir / "protocol.json", protocol)
    predictions, failures = [], []
    for qid in selected:
        record = by_id[qid]
        source = source_only(record)
        query = {"question_id": qid, "question": record["question"], "question_date": record["question_date"]}
        identity = {"protocol_sha256": digest(protocol),
                    "source_sha256": digest(source), "query_sha256": digest(query)}
        history = args.run_dir / "histories" / hashlib.sha256(qid.encode()).hexdigest()[:24]
        prediction = verified_prediction(history, identity)
        if prediction is None:
            attempt = next_attempt(history)
            save_json(attempt / "source.json", source)
            save_json(attempt / "query.json", query)
            save_json(attempt / "worker.json", {"runtime": runtime, "identity": identity,
                                               "source_files_sha256": hashes})
            with (attempt / "console.log").open("w", encoding="utf-8") as output:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker-dir", str(attempt)],
                    stdout=output, stderr=subprocess.STDOUT, check=False)
            prediction = verified_prediction(history, identity)
            if prediction is None:
                failures.append({"question_id": qid, "attempt": str(attempt), "exit_code": result.returncode})
                if not (attempt / "failure.json").exists():
                    save_json(attempt / "failure.json", failures[-1])
        if prediction is not None:
            predictions.append(prediction)
        save_json(args.run_dir / "predictions.json", predictions)
        save_json(args.run_dir / "failures.json", failures)
        status = ("running" if len(predictions) + len(failures) < len(selected)
                  else "generation_complete" if not failures else "generation_incomplete")
        save_json(args.run_dir / "status.json", {
            "method": args.method, "planned": len(selected), "generated": len(predictions),
            "failed": len(failures), "officially_judged": 0, "status": status})
    return int(bool(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--api-base")
    parser.add_argument("--embedding-api-base")
    parser.add_argument("--model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dims", type=int)
    parser.add_argument("--ids-file", type=Path)
    local_root = Path(__file__).resolve().parents[2] / "MemoryData"
    parser.add_argument("--source-root", type=Path, default=os.environ.get(
        "MEMORYDATA_SOURCE_ROOT", str(local_root) if local_root.exists()
        else "/workspace/longmemeval_s_native7_20260910/source/MemoryData"))
    parser.add_argument("--tokenizer", help="Pinned local Qwen tokenizer; defaults to --model")
    args = parser.parse_args()
    if args.worker_dir:
        return worker(args.worker_dir.resolve())
    for key in ("method", "dataset", "run_dir", "api_base", "embedding_api_base",
                "model", "embedding_model", "embedding_dims"):
        if not getattr(args, key):
            parser.error(f"--{key.replace('_', '-')} is required")
    if args.model != "Qwen/Qwen3.5-9B":
        parser.error("This run is frozen to Qwen/Qwen3.5-9B")
    if args.embedding_model != "sentence-transformers/all-MiniLM-L6-v2":
        parser.error("This run is frozen to sentence-transformers/all-MiniLM-L6-v2")
    if args.embedding_dims != 384:
        parser.error("Frozen MiniLM condition requires --embedding-dims 384")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
