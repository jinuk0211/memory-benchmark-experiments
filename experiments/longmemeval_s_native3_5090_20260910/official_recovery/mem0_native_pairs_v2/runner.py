"""Mem0 OSS 0.1.94 + documented LongMemEval QA integration; local candidate."""
from __future__ import annotations

import argparse
import gzip
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import native_five as harness

VERSION = "mem0_native_pairs_v2"
WHEEL_SHA256 = "862aaccf4ec41d65a5c935dc49c7b8175c96f6b60b29abeda338d8a335027e2c"
ANSWER_PATH = "Official Mem0 OSS 0.1.94 search + unchanged AgentWrapper Mem0 LongMemEval QA integration"


class NativeObservations(harness.NativeErrors):
    """Count authenticated native parse fallbacks; keep transport/storage errors fatal."""

    def __init__(self, attempt: Path) -> None:
        super().__init__()
        self.attempt = attempt
        self.warnings: list[dict] = []
        self.call = 0
        self.pair = -1
        self.pair_calls = 0
        self.latest: dict | None = None

    def begin_pair(self, index: int) -> None:
        self.pair, self.pair_calls, self.latest = index, 0, None

    def observe(self, completions: Any) -> None:
        original = completions.create

        def create(*args, **kwargs):
            self.call += 1
            self.pair_calls += 1
            event = {"call": self.call, "pair_index": self.pair,
                     "phase": "facts" if self.pair_calls == 1 else "update"}
            self.latest = None
            try:
                response = original(*args, **kwargs)
                raw = response.model_dump(mode="json")
                body = json.dumps(raw, ensure_ascii=False)
                event.update(response=raw, response_sha256=hashlib.sha256(body.encode()).hexdigest())
                with gzip.open(self.attempt / "memory_responses.jsonl.gz", "at", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                self.latest = event
                return response
            except Exception as error:
                self.records.append(f"Memory response/transport failure: {type(error).__name__}")
                raise

        completions.create = create

    def emit(self, record: logging.LogRecord) -> None:
        native_path = HERE / "vendor/mem0/memory/main.py"
        matched = (Path(record.pathname).resolve() == native_path.resolve()
                   and record.funcName == "_add_to_vector_store"
                   and record.lineno in {233, 278, 331})
        event = self.latest
        if not matched or event is None or event["pair_index"] != self.pair:
            super().emit(record)
            return
        choices = event["response"].get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        malformed, parsed = False, None
        try:
            strip_blocks = sys.modules["mem0.memory.main"].remove_code_blocks
            parsed = json.loads(strip_blocks(content))
        except (ValueError, TypeError, AttributeError):
            malformed = True
        supported = (
            record.lineno == 233 and event["phase"] == "facts"
            and (malformed or not isinstance(parsed, dict) or "facts" not in parsed)
        ) or (
            event["phase"] == "update" and (
                record.lineno == 278 and malformed
                or record.lineno == 331 and (malformed or not isinstance(parsed, dict)
                    or not isinstance(parsed.get("memory", []), list))
            )
        )
        if not supported:
            super().emit(record)
            return
        self.warnings.append({"pair_index": self.pair, "call": event["call"],
            "phase": event["phase"], "native_path": str(native_path),
            "native_function": record.funcName, "native_line": record.lineno,
            "message": record.getMessage(), "response_sha256": event["response_sha256"],
            "finish_reason": choices[0].get("finish_reason") if choices else None})

    def counts(self) -> dict:
        return {"native_warning_logs": len(self.warnings),
                "native_rejected_responses": len({row["call"] for row in self.warnings}),
                "native_affected_pairs": len({row["pair_index"] for row in self.warnings})}

    def save(self) -> None:
        harness.save_json(self.attempt / "native_degradation.json", {
            "policy": "Preserve native parse/shape fallbacks; no repair or added retries",
            "warnings": self.warnings, "fatal_errors": self.records, **self.counts()})


def verify_official_source() -> dict[str, str]:
    manifest = harness.read_json(HERE / "source_manifest.json")
    if manifest["wheel_sha256"] != WHEEL_SHA256:
        raise ValueError("Unrecognized official Mem0 distribution")
    expected = {**manifest["files_sha256"], manifest["wheel_file"]: WHEEL_SHA256}
    actual = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in expected}
    if actual != expected:
        raise ValueError("Official Mem0 bytes changed")
    python_files = {p.relative_to(HERE).as_posix() for p in (HERE / "vendor").rglob("*.py")}
    if python_files != {name for name in expected if name.endswith(".py")}:
        raise ValueError("Unexpected Python files in official vendor tree")
    actual["source_manifest.json"] = hashlib.sha256((HERE / "source_manifest.json").read_bytes()).hexdigest()
    actual["runner.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return actual


def source_hashes(source_root: Path) -> dict[str, str]:
    # Hash the archived integration too; its altered Mem0 core is never imported.
    integration = harness.source_hashes(source_root, "mem0")
    return {**{f"integration/{k}": v for k, v in integration.items()},
            **{f"candidate/{k}": v for k, v in verify_official_source().items()}}


def session_pairs(sessions: list[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], list[dict[str, str]]]]:
    """Keep dataset session order, boundaries, roles and full text; never add labels."""
    for session_index, session in enumerate(sessions):
        turns = [{"role": t["role"], "content": t["content"]} for t in session["turns"]]
        for turn in turns:
            if (turn["role"] not in ("user", "assistant", "system")
                    or not isinstance(turn["content"], str) or not turn["content"].strip()):
                raise ValueError("Source contains an unsupported role or blank/non-text turn")
        stamp = datetime.strptime(session["date"], "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)
        for start in range(0, len(turns), 2):
            yield {"session_id": session["session_id"], "session_date": session["date"],
                   "timestamp": int(stamp.timestamp()), "session_index": session_index,
                   "turn_start": start, "turn_stop": min(start + 2, len(turns))}, turns[start:start + 2]


def memory_config(args: argparse.Namespace, state: Path) -> dict[str, Any]:
    return {
        "llm": {"provider": "openai", "config": {"model": args.model, "api_key": "EMPTY",
                "openai_base_url": args.api_base, "temperature": 0.1, "max_tokens": 2000, "top_p": 0.1}},
        "embedder": {"provider": "openai", "config": {"model": args.embedding_model, "api_key": "EMPTY",
                     "openai_base_url": args.embedding_api_base, "embedding_dims": 384}},
        "vector_store": {"provider": "qdrant", "config": {"path": str(state / "qdrant"),
                         "collection_name": "mem0_longmemeval_s_official", "embedding_model_dims": 384,
                         "on_disk": True}},
        "history_db_path": str(state / "history.db"),
    }


def dependency_versions() -> dict[str, str]:
    """Reject unsupported runtime before importing the historical official package."""
    from packaging.requirements import Requirement
    from email.parser import Parser
    metadata = Parser().parsestr((HERE / "vendor/mem0ai-0.1.94.dist-info/METADATA").read_text(encoding="utf-8"))
    versions = {}
    for text in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        installed = importlib.metadata.version(requirement.name)
        if installed not in requirement.specifier:
            raise RuntimeError(f"Official Mem0 dependency mismatch: {requirement}; installed {installed}")
        versions[requirement.name] = installed
    return versions


def create_agent(args: argparse.Namespace, attempt: Path, config: dict, dataset: dict) -> Any:
    # Load the authenticated package before the archived adapter adds vendor paths.
    sys.path.insert(0, str(HERE / "vendor"))
    from mem0.memory.main import Memory
    import mem0
    if Path(sys.modules["mem0.memory.main"].__file__).resolve() != HERE / "vendor/mem0/memory/main.py":
        raise RuntimeError("Mem0 resolved outside the authenticated official snapshot")
    if mem0.__version__ != "0.1.94":
        raise RuntimeError("Mem0 distribution metadata did not resolve to 0.1.94")
    from utils.agent import AgentWrapper

    class OfficialMem0Agent(AgentWrapper):
        def _initialize_mem0_agent(self, agent_config: dict, dataset_config: dict) -> None:
            self.retrieve_num = agent_config["retrieve_num"]
            self.mem0_add_infer = True
            self.chunk_size = agent_config.get("agent_chunk_size", dataset_config.get("chunk_size"))
            self.mem0_source_map_path = str(attempt / "memory/mem0_source_map.json")
            self.mem0_source_map = {}
            self.context = ""
            self.client = self._create_oai_client()
            self.memory = Memory.from_config(memory_config(args, attempt / "memory"))
            self.agent_start_time = time.time()

    return OfficialMem0Agent(config, dataset, load_agent_from=str(attempt / "memory"))


def ingest_pairs(memory: Any, source: list[dict], user_id: str, progress_path: Path,
                 errors: harness.NativeErrors) -> int:
    count = 0
    for metadata, messages in session_pairs(source):
        event = {"completed_pairs": count, "metadata": metadata, "messages_sha256": harness.digest(messages)}
        harness.save_json(progress_path, {**event, "status": "adding"})
        errors.begin_pair(count)
        result = memory.add(messages, user_id=user_id, metadata=metadata, infer=True)
        if errors.records:
            raise RuntimeError("Official Mem0 logged construction errors; preserving failed attempt")
        count += 1
        harness.save_json(progress_path, {**event, "completed_pairs": count, "status": "added",
                                         "native_result": result})
    return count


def worker(attempt: Path) -> int:
    errors = NativeObservations(attempt)
    logging.getLogger().addHandler(errors)
    spec = harness.read_json(attempt / "worker.json")
    identity = spec["identity"]
    started = time.monotonic()
    try:
        protocol = harness.read_json(attempt.parents[2] / "protocol.json")
        if (harness.digest(protocol) != identity["protocol_sha256"]
                or protocol["runtime"] != spec["runtime"]
                or protocol["source_files_sha256"] != spec["source_files_sha256"]):
            raise ValueError("Worker does not match the immutable run protocol")
        args = argparse.Namespace(**spec["runtime"])
        args.source_root = Path(args.source_root)
        validate_runtime(args)
        source, query = harness.read_json(attempt / "source.json"), harness.read_json(attempt / "query.json")
        if harness.digest(source) != identity["source_sha256"] or harness.digest(query) != identity["query_sha256"]:
            raise ValueError("Worker source/query changed")
        if source_hashes(args.source_root) != spec["source_files_sha256"]:
            raise ValueError("Frozen candidate/integration changed")
        versions = dependency_versions()
        os.chdir(args.source_root)
        sys.path.insert(0, str(args.source_root))
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.update(OPENAI_API_KEY="EMPTY", OPENAI_BASE_URL=args.api_base,
                          MEM0_TELEMETRY="false", MEM0_DIR=str(attempt / "memory/mem0_home"),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                          NO_PROXY="localhost,127.0.0.1,::1", no_proxy="localhost,127.0.0.1,::1",
                          BASELINE_STRICT_COMPARISON="1", METER_RUN_ID=identity["protocol_sha256"],
                          METER_METHOD="mem0", METER_LLM_PROXY_ORIGIN=args.api_base,
                          METER_EMBEDDING_PROXY_ORIGIN=args.embedding_api_base,
                          METER_TIMING_JOURNAL=str(attempt / "timing.jsonl"))
        from utils.request_metering import install_request_metering, meter_operation
        install_request_metering()
        config, dataset = harness.build_config(args, attempt)
        (attempt / "memory").mkdir()
        harness.save_json(attempt / "native_config.json", {"memory": memory_config(args, attempt / "memory"),
                          "qa_integration": config, "dataset": dataset, "implementation": VERSION})
        with meter_operation("memory_init", sample_id=query["question_id"]):
            agent = create_agent(args, attempt, config, dataset)
            errors.observe(agent.memory.llm.client.chat.completions)
        from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT
        harness.save_json(attempt / "native_runtime.json", {"dependencies": versions,
                          "official_fact_retrieval_prompt": FACT_RETRIEVAL_PROMPT,
                          "official_fact_retrieval_prompt_sha256": hashlib.sha256(FACT_RETRIEVAL_PROMPT.encode()).hexdigest(),
                          "prompt_observed_at_utc": datetime.now(timezone.utc).isoformat(),
                          **harness.tokenizer_runtime(agent.tokenizer, config["tokenizer_model"])})
        with meter_operation("memory_write", sample_id=query["question_id"]):
            count = ingest_pairs(agent.memory, source, f'context_{query["question_id"]}_{agent.sub_dataset}',
                                 attempt / "construction_progress.json", errors)
            agent.save_agent()
        if errors.records:
            raise RuntimeError("Official Mem0 construction logged errors")
        harness.save_json(attempt / "build_complete.json", {"identity": identity, "pairs": count,
                          "sessions": len(source), "seconds": time.monotonic() - started, **errors.counts()})
        with meter_operation("qa", sample_id=query["question_id"], question_id=query["question_id"]):
            answer, detail = harness.native_answer(agent, "mem0", query)
        if errors.records or not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Mem0 QA failed, logged errors, or returned a blank answer")
        harness.save_json(attempt / "native_answer_detail.json", detail)
        harness.save_json(attempt / "prediction.json", {"identity": identity, "question_id": query["question_id"],
                          "hypothesis": answer, "status": "generated", "officially_judged": False,
                          "native_answer_path": ANSWER_PATH, "implementation": VERSION, **errors.counts()})
        return 0
    except Exception:
        harness.save_json(attempt / "failure.json", {"identity": identity, "traceback": traceback.format_exc(),
                          "native_errors": errors.records, "seconds": time.monotonic() - started})
        return 1
    finally:
        errors.save()
        logging.getLogger().removeHandler(errors)


def validate_runtime(args: argparse.Namespace) -> None:
    if (args.model != "Qwen/Qwen3.5-9B" or args.embedding_model != "sentence-transformers/all-MiniLM-L6-v2"
            or args.embedding_dims != 384):
        raise ValueError("This candidate is fixed to local Qwen3.5-9B and MiniLM 384")
    for endpoint in (args.api_base, args.embedding_api_base):
        url = urlsplit(endpoint)
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.username or url.password:
            raise ValueError("Only local unauthenticated HTTP model endpoints are permitted")


def attempt_files(attempt: Path) -> dict[str, str]:
    return {p.relative_to(attempt).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(attempt.rglob("*")) if p.is_file() and p != attempt / "completion.json"}


def finalize_attempt(attempt: Path, identity: dict) -> None:
    """Called only after the worker exits, so SQLite/Qdrant/log files are closed."""
    required = {"source.json", "query.json", "worker.json", "native_config.json", "native_runtime.json",
                "build_complete.json", "native_answer_detail.json", "prediction.json", "memory/history.db",
                "native_degradation.json", "memory_responses.jsonl.gz"}
    files = attempt_files(attempt)
    if not required.issubset(files) or not any(p.startswith("memory/qdrant/") for p in files):
        raise ValueError("Completed attempt lacks required native construction artifacts")
    if "failure.json" in files:
        raise ValueError("Failed attempt cannot receive a completion receipt")
    source, query = harness.read_json(attempt / "source.json"), harness.read_json(attempt / "query.json")
    prediction = harness.read_json(attempt / "prediction.json")
    if (harness.digest(source) != identity["source_sha256"] or harness.digest(query) != identity["query_sha256"]
            or harness.read_json(attempt / "build_complete.json").get("identity") != identity
            or prediction.get("identity") != identity or prediction.get("question_id") != query["question_id"]
            or prediction.get("status") != "generated" or prediction.get("error") or prediction.get("error_type")
            or not isinstance(prediction.get("hypothesis"), str) or not prediction["hypothesis"].strip()):
        raise ValueError("Attempt source/query/build/prediction proof differs")
    harness.write_once(attempt / "completion.json", {"identity": identity, "status": "generated",
                       "files_sha256": files, "created_after_worker_exit": True})


def verified_prediction(history: Path, identity: dict) -> dict | None:
    for path in sorted(history.glob("attempt_*/prediction.json")):
        row = harness.read_json(path)
        if row.get("identity") != identity:
            raise ValueError(f"Prediction identity changed: {path}")
        attempt = path.parent
        if (attempt / "failure.json").exists() or not (attempt / "completion.json").exists():
            continue
        receipt = harness.read_json(attempt / "completion.json")
        if receipt.get("identity") != identity or receipt.get("files_sha256") != attempt_files(attempt):
            raise ValueError(f"Completed attempt artifacts changed: {attempt}")
        if not {"native_degradation.json", "memory_responses.jsonl.gz"}.issubset(receipt["files_sha256"]):
            raise ValueError("Missing native degradation/response evidence")
        if row.get("status") == "generated" and isinstance(row.get("hypothesis"), str) and row["hypothesis"].strip():
            return row
    return None


def run(args: argparse.Namespace) -> int:
    validate_runtime(args)
    data = args.dataset.read_bytes()
    if hashlib.sha256(data).hexdigest() != harness.DATA_SHA256:
        raise ValueError("Expected the unchanged canonical LongMemEval-S 500 dataset")
    records = json.loads(data)
    by_id = {str(row["question_id"]): row for row in records}
    if len(records) != 500 or len(by_id) != 500:
        raise ValueError("Expected 500 unique question IDs")
    selected = harness.read_json(args.ids_file) if args.ids_file else list(by_id)
    if (not isinstance(selected, list) or not selected or not all(isinstance(q, str) for q in selected)
            or len(set(selected)) != len(selected) or set(selected) - by_id.keys()):
        raise ValueError("IDs must be a nonempty unique subset of the canonical 500")
    args.source_root, args.run_dir = args.source_root.resolve(), args.run_dir.resolve()
    hashes = source_hashes(args.source_root)
    runtime = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k in ("method", "source_root", "api_base", "embedding_api_base", "model", "embedding_model", "embedding_dims", "tokenizer")}
    protocol = {"runtime": runtime, "dataset_sha256": harness.DATA_SHA256, "population_ids": list(by_id),
                "source_files_sha256": hashes, "implementation": VERSION, "native_answer_path": ANSWER_PATH,
                "ingestion": "Within each original session, consecutive original role/content pairs; final singleton retained",
                "memory_prompts": "Unmodified official OSS defaults", "metadata": "Session ID/date/UTC timestamp; not inserted into prompts",
                "qa_integration": "Existing benchmark integration unchanged: retrieval 100, temperature 0.1, output 256",
                "cloud_paper_reproduction": False, "common_reader": False,
                "native_fallbacks": "Count authenticated caught JSON/shape fallbacks; transport/storage failures remain fatal",
                "native_response_capture": "Original SDK responses retained; requests and retries unchanged"}
    harness.write_once(args.run_dir / "protocol.json", protocol)
    predictions, failures = [], []
    for qid in selected:
        row = by_id[qid]
        source = harness.source_only(row)
        query = {"question_id": qid, "question": row["question"], "question_date": row["question_date"]}
        identity = {"protocol_sha256": harness.digest(protocol), "source_sha256": harness.digest(source), "query_sha256": harness.digest(query)}
        history = args.run_dir / "histories" / hashlib.sha256(qid.encode()).hexdigest()[:24]
        prediction = verified_prediction(history, identity)
        if prediction is None:
            attempt = harness.next_attempt(history)
            for name, value in (("source", source), ("query", query), ("worker", {"runtime": runtime, "identity": identity, "source_files_sha256": hashes})):
                harness.save_json(attempt / f"{name}.json", value)
            harness.save_json(args.run_dir / "current.json", {"question_id": qid, "attempt": str(attempt), "status": "running"})
            with (attempt / "console.log").open("w", encoding="utf-8") as output:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-dir", str(attempt)],
                                        stdout=output, stderr=subprocess.STDOUT, check=False)
            completion_error = None
            if result.returncode == 0:
                try:
                    finalize_attempt(attempt, identity)
                except (ValueError, OSError, KeyError) as error:
                    completion_error = str(error)
            prediction = verified_prediction(history, identity)
            if result.returncode or prediction is None:
                prediction = None
                failure = {"question_id": qid, "attempt": str(attempt), "exit_code": result.returncode, "completion_error": completion_error}
                failures.append(failure)
                if not (attempt / "failure.json").exists():
                    harness.save_json(attempt / "failure.json", failure)
        if prediction is not None:
            predictions.append(prediction)
        harness.save_json(args.run_dir / "predictions.json", predictions)
        harness.save_json(args.run_dir / "failures.json", failures)
        harness.save_json(args.run_dir / "status.json", {"method": "mem0", "implementation": VERSION,
                          "planned": len(selected), "generated": len(predictions), "failed": len(failures),
                          "officially_judged": 0, "status": "generation_incomplete" if failures else
                          "generation_complete" if len(predictions) == len(selected) else "running"})
        if failures:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT / "source/MemoryData")
    parser.add_argument("--api-base")
    parser.add_argument("--embedding-api-base")
    parser.add_argument("--model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dims", type=int, default=384)
    parser.add_argument("--tokenizer")
    parser.add_argument("--ids-file", type=Path)
    parser.set_defaults(method="mem0")
    args = parser.parse_args()
    if args.worker_dir:
        return worker(args.worker_dir.resolve())
    for key in ("dataset", "run_dir", "api_base", "embedding_api_base", "model", "embedding_model", "tokenizer"):
        if not getattr(args, key):
            parser.error(f"--{key.replace('_', '-')} is required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())