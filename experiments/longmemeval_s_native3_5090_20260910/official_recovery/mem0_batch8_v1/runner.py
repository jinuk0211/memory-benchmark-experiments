"""Mem0 OSS 0.1.94 + documented LongMemEval QA integration; local candidate."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import copy
import gzip
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
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
import native_five as harness  # noqa: E402 - pinned experiment root is selected above.

VERSION = "mem0_batch8_v1"
DISPLAY_NAME = "Mem0-batch8"
BATCH_PAIRS = 8
INGESTION = "Within each original session, batch up to eight consecutive original dialogue pairs (16 turns) per Memory.add(infer=True); retain remaining pairs, final singleton and empty-string turns verbatim without truncation or cross-session batching"
BATCH_EVIDENCE = "Ordered successful-add journal binds original source hash, batch index, immutable session metadata, exact turn range, message hash, pair/turn counts and observed chat call count"
WHEEL_SHA256 = "862aaccf4ec41d65a5c935dc49c7b8175c96f6b60b29abeda338d8a335027e2c"
ANSWER_PATH = "Official Mem0 OSS 0.1.94 search + unchanged AgentWrapper Mem0 LongMemEval QA integration"


class NativeObservations(harness.NativeErrors):
    """Count authenticated native response losses; keep transport/storage errors fatal."""

    def __init__(self, attempt: Path) -> None:
        super().__init__()
        self.attempt = attempt
        self.warnings: list[dict] = []
        self.call = 0
        self.batch = -1
        self.batch_calls = 0
        self.latest: dict | None = None

    def begin_batch(self, index: int) -> None:
        self.batch, self.batch_calls, self.latest = index, 0, None

    def observe(self, completions: Any) -> None:
        original = completions.create

        def create(*args, **kwargs):
            self.call += 1
            self.batch_calls += 1
            event = {"call": self.call, "batch_index": self.batch,
                     "phase": "facts" if self.batch_calls == 1 else "update"}
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
                   and record.lineno in {233, 278, 329, 331})
        event = self.latest
        if not matched or event is None or event["batch_index"] != self.batch:
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
        action = native_action_evidence(sys.exception(), parsed) if record.lineno == 329 and event["phase"] == "update" else None
        if not supported and action is None:
            super().emit(record)
            return
        self.warnings.append({"warning_id": len(self.warnings) + 1,
            "kind": "native_action_drop" if action is not None else "native_parse_fallback",
            "batch_index": self.batch, "call": event["call"],
            "phase": event["phase"], "native_path": str(native_path),
            "native_function": record.funcName, "native_line": record.lineno,
            "message": record.getMessage(), "response_sha256": event["response_sha256"],
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            **({"action_evidence": action} if action is not None else {})})

    def counts(self) -> dict:
        return warning_counts(self.warnings)

    def save(self) -> None:
        harness.save_json(self.attempt / "native_degradation.json", {
            "implementation": VERSION,
            "policy": "Preserve native parse/shape and pre-storage action drops; no repair or added retries",
            "interpretation": "Affected responses may retain partial native writes; not whole-response rejection or recovered writes",
            "lossless_writes": False if self.warnings else None,
            "warnings": self.warnings, "fatal_errors": self.records, **self.counts()})


def warning_counts(warnings: list[dict]) -> dict:
    return {"native_warning_logs": len(warnings),
            "native_affected_responses": len({row["call"] for row in warnings}),
            "native_affected_batches": len({row["batch_index"] for row in warnings}),
            "native_action_drops": sum(row["kind"] == "native_action_drop" for row in warnings)}


def action_cause(line: int, exception_type: str, entry: Any, mapping_keys: list[str]) -> bool:
    """Recognize only JSON entry access and temporary-ID lookup before storage calls."""
    if line == 286:
        return exception_type == "AttributeError" and not isinstance(entry, dict)
    if (line not in (304, 318) or not isinstance(entry, dict) or not entry.get("text")
            or entry.get("event") != ("UPDATE" if line == 304 else "DELETE")):
        return False
    if line == 304 and "id" not in entry:
        return exception_type == "KeyError"
    value = entry.get("id")
    if isinstance(value, (list, dict)):
        return exception_type == "TypeError"
    return exception_type == "KeyError" and value not in mapping_keys


def action_matches_response(evidence: dict, parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    entries, keys = parsed.get("memory"), evidence.get("mapping_keys")
    if (not isinstance(entries, (list, dict, str)) or not isinstance(keys, list)
            or keys != [str(i) for i in range(len(keys))]
            or "response_entry" not in evidence or type(evidence.get("origin_line")) is not int):
        return False
    entry = evidence["response_entry"]
    return (any(harness.digest(entry) == harness.digest(item) for item in entries)
            and action_cause(evidence["origin_line"], evidence.get("exception_type"), entry, keys))


def native_action_evidence(error: BaseException | None, parsed: Any) -> dict | None:
    """Use the active native catch; log text alone cannot justify an action drop."""
    if error is None or error.__traceback__ is None:
        return None
    tb = error.__traceback__
    frame, path = tb.tb_frame, HERE / "vendor/mem0/memory/main.py"
    if (tb.tb_next is not None or frame.f_code.co_filename != str(path.resolve())
            or frame.f_code.co_name != "_add_to_vector_store"
            or frame.f_locals.get("e") is not error
            or frame.f_locals.get("new_memories_with_actions") != parsed):
        return None
    mapping = frame.f_locals.get("temp_uuid_mapping")
    if not isinstance(mapping, dict) or "resp" not in frame.f_locals:
        return None
    evidence = {"origin_line": tb.tb_lineno, "exception_type": type(error).__name__,
                "exception_message": str(error), "response_entry": frame.f_locals["resp"],
                "mapping_keys": list(mapping), "native_traceback_frames": 1}
    return json.loads(json.dumps(evidence)) if action_matches_response(evidence, parsed) else None


def parsed_response(raw: dict) -> tuple[Any, bool]:
    """Evidence-only equivalent of the pinned remove_code_blocks plus JSON decoder."""
    try:
        content = raw["choices"][0]["message"]["content"]
        match = re.match(r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$", content.strip())
        return json.loads(match.group(1).strip() if match else content.strip()), False
    except (ValueError, TypeError, AttributeError, KeyError, IndexError):
        return None, True


def verify_degradation(attempt: Path) -> None:
    data = harness.read_json(attempt / "native_degradation.json")
    warnings = data.get("warnings")
    if data.get("implementation") != VERSION or data.get("fatal_errors") != [] or not isinstance(warnings, list):
        raise ValueError("Native degradation version/fatal state differs")
    native_path = str(HERE / "vendor/mem0/memory/main.py")
    for index, warning in enumerate(warnings, 1):
        if (not isinstance(warning, dict) or type(warning.get("warning_id")) is not int
                or warning["warning_id"] != index or type(warning.get("call")) is not int
                or warning["call"] <= 0 or type(warning.get("batch_index")) is not int
                or warning["batch_index"] < 0 or warning.get("native_path") != native_path
                or warning.get("native_function") != "_add_to_vector_store"
                or type(warning.get("native_line")) is not int):
            raise ValueError("Invalid native warning provenance")
    for key, expected in warning_counts(warnings).items():
        if data.get(key) != expected or type(data.get(key)) is not type(expected):
            raise ValueError("Native warning accounting differs")
    if warnings and data.get("lossless_writes") is not False:
        raise ValueError("Native losses cannot claim lossless writes")
    pending = {w["call"] for w in warnings}
    with gzip.open(attempt / "memory_responses.jsonl.gz", "rt", encoding="utf-8") as stream:
        previous = 0
        for line in stream:
            event = json.loads(line)
            if type(event.get("call")) is not int or event["call"] != previous + 1:
                raise ValueError("Native response call sequence differs")
            previous = event["call"]
            body = json.dumps(event["response"], ensure_ascii=False)
            if event.get("response_sha256") != hashlib.sha256(body.encode()).hexdigest():
                raise ValueError("Native response hash differs")
            for warning in (w for w in warnings if w["call"] == event["call"]):
                verify_warning_response(warning, event)
            pending.discard(event["call"])
    if pending:
        raise ValueError("Native warning lacks its original response")


def verify_warning_response(warning: dict, event: dict) -> None:
    choices = event["response"].get("choices") or []
    finish = choices[0].get("finish_reason") if choices else None
    if (warning.get("finish_reason") != finish
            or any(warning.get(key) != event.get(key) for key in ("batch_index", "phase", "response_sha256"))):
        raise ValueError("Native warning response provenance differs")
    parsed, malformed = parsed_response(event["response"])
    line, phase = warning["native_line"], warning["phase"]
    if warning.get("kind") == "native_action_drop":
        evidence = warning.get("action_evidence")
        if (line != 329 or phase != "update" or not isinstance(evidence, dict)
                or type(evidence.get("native_traceback_frames")) is not int
                or evidence["native_traceback_frames"] != 1
                or not isinstance(evidence.get("exception_message"), str)
                or warning.get("message") != "Error in new_memories_with_actions: " + evidence["exception_message"]
                or not action_matches_response(evidence, parsed)):
            raise ValueError("Native action-drop evidence differs")
    elif warning.get("kind") != "native_parse_fallback" or not (
            line == 233 and phase == "facts" and (malformed or not isinstance(parsed, dict) or "facts" not in parsed)
            or phase == "update" and (line == 278 and malformed or line == 331 and (
                malformed or not isinstance(parsed, dict) or not isinstance(parsed.get("memory", []), list)))):
        raise ValueError("Native parse-fallback evidence differs")


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


def session_batches(sessions: list[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], list[dict[str, str]]]]:
    """Eight consecutive pairs at most, retaining source order and session boundaries."""
    batch_index = 0
    for session_index, session in enumerate(sessions):
        turns = [{"role": t["role"], "content": t["content"]} for t in session["turns"]]
        if not turns:
            raise ValueError("An empty original session cannot be silently omitted")
        for turn in turns:
            if (turn["role"] not in ("user", "assistant", "system")
                    or not isinstance(turn["content"], str)):
                raise ValueError("Source contains an unsupported role or non-text turn")
        stamp = datetime.strptime(session["date"], "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)
        for start in range(0, len(turns), 2 * BATCH_PAIRS):
            stop = min(start + 2 * BATCH_PAIRS, len(turns))
            yield {"session_id": session["session_id"], "session_date": session["date"],
                   "timestamp": int(stamp.timestamp()), "session_index": session_index,
                   "batch_index": batch_index, "turn_start": start, "turn_stop": stop}, turns[start:stop]
            batch_index += 1


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


def disable_telemetry_threads() -> dict[str, Any]:
    """Keep native telemetry opt-out while preventing unused PostHog consumers."""
    telemetry = importlib.import_module("mem0.memory.telemetry")
    posthog = importlib.import_module("posthog")
    version = importlib.metadata.version("posthog")
    if (version != "3.25.0" or telemetry.MEM0_TELEMETRY is not False
            or Path(telemetry.__file__).resolve() != HERE / "vendor/mem0/memory/telemetry.py"
            or telemetry.Posthog is not posthog.Posthog):
        raise ValueError("Expected pinned native telemetry with explicit opt-out")
    telemetry.client_telemetry.close()
    state = {"posthog_version": version, "send": False, "sync_mode": True,
             "disabled": True, "created_clients": 0,
             "scope": "Only native Mem0 telemetry; memory/model calls unchanged"}
    original = telemetry.Posthog

    def disabled_client(*args, **kwargs):
        kwargs.update(send=False, sync_mode=True, disabled=True)
        client = original(*args, **kwargs)
        if (client.send is not False or client.sync_mode is not True
                or client.disabled is not True or client.consumers is not None):
            raise RuntimeError("Telemetry unexpectedly created asynchronous consumers")
        state["created_clients"] += 1
        return client

    telemetry.Posthog = disabled_client
    return state


def create_agent(args: argparse.Namespace, attempt: Path, config: dict, dataset: dict) -> Any:
    # Load the authenticated package before the archived adapter adds vendor paths.
    sys.path.insert(0, str(HERE / "vendor"))
    from mem0.memory.main import Memory
    import mem0
    if Path(sys.modules["mem0.memory.main"].__file__).resolve() != HERE / "vendor/mem0/memory/main.py":
        raise RuntimeError("Mem0 resolved outside the authenticated official snapshot")
    if mem0.__version__ != "0.1.94":
        raise RuntimeError("Mem0 distribution metadata did not resolve to 0.1.94")
    telemetry_state = disable_telemetry_threads()
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

    agent = OfficialMem0Agent(config, dataset, load_agent_from=str(attempt / "memory"))
    agent._native_telemetry_state = telemetry_state
    return agent


def batch_input(source_sha256: str, metadata: dict, messages: list[dict]) -> dict:
    return {"batch_index": metadata["batch_index"], "source_sha256": source_sha256,
            "metadata": copy.deepcopy(metadata), "messages_sha256": harness.digest(messages),
            "source_turns": len(messages), "source_pairs": (len(messages) + 1) // 2}


def ingest_batches(memory: Any, source: list[dict], user_id: str, progress_path: Path,
                   errors: NativeObservations) -> int:
    count = 0
    source_sha256 = harness.digest(source)
    with (progress_path.parent / "batch_calls.jsonl").open("x", encoding="utf-8") as journal:
        for metadata, messages in session_batches(source):
            event = batch_input(source_sha256, metadata, messages)
            harness.save_json(progress_path, {**event, "completed_batches": count, "status": "adding"})
            errors.begin_batch(count)
            before, started = errors.call, time.monotonic()
            result = memory.add(messages, user_id=user_id, metadata=copy.deepcopy(metadata), infer=True)
            if errors.records:
                raise RuntimeError("Official Mem0 logged construction errors; preserving failed attempt")
            record = {**event, "chat_calls": errors.call - before, "seconds": time.monotonic() - started}
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            journal.flush()
            count += 1
            harness.save_json(progress_path, {**event, "completed_batches": count, "status": "added",
                                             "native_result": result})
    return count


def worker(attempt: Path) -> int:
    errors = NativeObservations(attempt)
    logging.getLogger().addHandler(errors)
    spec = harness.read_json(attempt / "worker.json")
    identity = spec["identity"]
    started = time.monotonic()
    agent = None
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
            count = ingest_batches(agent.memory, source, f'context_{query["question_id"]}_{agent.sub_dataset}',
                                 attempt / "construction_progress.json", errors)
            agent.save_agent()
        if errors.records:
            raise RuntimeError("Official Mem0 construction logged errors")
        harness.save_json(attempt / "build_complete.json", {"identity": identity, "batches": count,
                          "sessions": len(source), "turns": sum(len(s["turns"]) for s in source),
                          "pairs": sum((len(s["turns"]) + 1) // 2 for s in source),
                          "seconds": time.monotonic() - started, **errors.counts()})
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
        if agent is not None:
            harness.save_json(attempt / "native_telemetry.json", agent._native_telemetry_state)
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


def verify_batch_coverage(attempt: Path, identity: dict) -> None:
    """Recheck original source order and complete batch coverage, including cache reads."""
    source = harness.read_json(attempt / "source.json")
    if harness.digest(source) != identity["source_sha256"]:
        raise ValueError("Batch source identity differs")
    batches = list(session_batches(source))
    if not batches:
        raise ValueError("Completed construction has no original sessions")
    records = []
    with (attempt / "batch_calls.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Partial batch journal line")
            records.append(json.loads(line))
    if len(records) != len(batches):
        raise ValueError("Batch journal coverage differs from source")
    observed = [0] * len(batches)
    previous = -1
    with gzip.open(attempt / "memory_responses.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            index = event.get("batch_index")
            if (type(index) is not int or not 0 <= index < len(batches) or index < previous
                    or event.get("phase") != ("facts" if observed[index] == 0 else "update")):
                raise ValueError("Response batch attribution or phase differs")
            previous = index
            observed[index] += 1
    for index, (record, (metadata, messages)) in enumerate(zip(records, batches)):
        expected = batch_input(identity["source_sha256"], metadata, messages)
        if (not isinstance(record, dict) or set(record) != set(expected) | {"chat_calls", "seconds"}
                or any(record.get(key) != value for key, value in expected.items())
                or any(type(record[key]) is not int for key in ("batch_index", "source_turns", "source_pairs", "chat_calls"))
                or any(type(record["metadata"][key]) is not int for key in
                       ("batch_index", "session_index", "timestamp", "turn_start", "turn_stop"))
                or record["chat_calls"] <= 0 or record["chat_calls"] != observed[index]
                or type(record["seconds"]) not in (int, float) or not math.isfinite(record["seconds"])
                or record["seconds"] < 0):
            raise ValueError("Batch journal source binding or accounting differs")
    build = harness.read_json(attempt / "build_complete.json")
    totals = {"batches": len(batches), "sessions": len(source),
              "turns": sum(len(s["turns"]) for s in source),
              "pairs": sum((len(s["turns"]) + 1) // 2 for s in source)}
    if (build.get("identity") != identity or any(type(build.get(key)) is not int
            or build[key] != value for key, value in totals.items())):
        raise ValueError("Completed batch/session/turn/pair totals differ")
    progress = harness.read_json(attempt / "construction_progress.json")
    last = records[-1]
    if (progress.get("status") != "added" or type(progress.get("completed_batches")) is not int
            or progress["completed_batches"] != len(batches)
            or any(type(progress.get(key)) is not int for key in ("batch_index", "source_turns", "source_pairs"))
            or not isinstance(progress.get("metadata"), dict)
            or any(type(progress["metadata"].get(key)) is not int for key in
                   ("batch_index", "session_index", "timestamp", "turn_start", "turn_stop"))
            or any(progress.get(key) != last[key] for key in
                   ("batch_index", "source_sha256", "metadata", "messages_sha256", "source_turns", "source_pairs"))):
        raise ValueError("Final batch progress differs from journal")


def validated_attempt_files(attempt: Path, identity: dict) -> dict[str, str]:
    required = {"source.json", "query.json", "worker.json", "native_config.json", "native_runtime.json",
                "build_complete.json", "native_answer_detail.json", "prediction.json", "memory/history.db",
                "native_degradation.json", "memory_responses.jsonl.gz", "native_telemetry.json",
                "batch_calls.jsonl", "construction_progress.json", "console.log"}
    files = attempt_files(attempt)
    if not required.issubset(files) or not any(p.startswith("memory/qdrant/") for p in files):
        raise ValueError("Completed attempt lacks required native construction artifacts")
    if "failure.json" in files:
        raise ValueError("Failed attempt cannot receive a completion receipt")
    telemetry = harness.read_json(attempt / "native_telemetry.json")
    if (telemetry.get("posthog_version") != "3.25.0" or telemetry.get("send") is not False
            or telemetry.get("sync_mode") is not True or telemetry.get("disabled") is not True
            or type(telemetry.get("created_clients")) is not int or telemetry["created_clients"] <= 0):
        raise ValueError("Native telemetry opt-out evidence differs")
    protocol = harness.read_json(attempt.parents[2] / "protocol.json")
    population = protocol.get("population_ids", [])
    if (len(population) != 500 or len(set(population)) != 500
            or protocol != protocol_for(protocol["runtime"], population,
                                        source_hashes(Path(protocol["runtime"]["source_root"])))
            or harness.digest(protocol) != identity.get("protocol_sha256")
            or harness.read_json(attempt / "worker.json") != {"identity": identity,
                "runtime": protocol["runtime"], "source_files_sha256": protocol["source_files_sha256"]}):
        raise ValueError("Completed attempt protocol or worker binding differs")
    source, query = harness.read_json(attempt / "source.json"), harness.read_json(attempt / "query.json")
    prediction = harness.read_json(attempt / "prediction.json")
    if (harness.digest(source) != identity["source_sha256"] or harness.digest(query) != identity["query_sha256"]
            or query["question_id"] not in population or prediction.get("identity") != identity
            or prediction.get("question_id") != query["question_id"]
            or prediction.get("implementation") != VERSION or prediction.get("native_answer_path") != ANSWER_PATH
            or prediction.get("status") != "generated" or prediction.get("error") or prediction.get("error_type")
            or not isinstance(prediction.get("hypothesis"), str) or not prediction["hypothesis"].strip()):
        raise ValueError("Attempt source/query/prediction proof differs")
    verify_degradation(attempt)
    verify_batch_coverage(attempt, identity)
    return files


def finalize_attempt(attempt: Path, identity: dict) -> None:
    """Called only after worker exit, when SQLite/Qdrant/log files are closed."""
    files = validated_attempt_files(attempt, identity)
    harness.write_once(attempt / "completion.json", {"identity": identity, "status": "generated",
                       "files_sha256": files, "created_after_worker_exit": True})


def verified_prediction(history: Path, identity: dict) -> dict | None:
    for attempt in sorted(history.glob("attempt_*")):
        worker_path = attempt / "worker.json"
        if worker_path.is_file() and harness.read_json(worker_path).get("identity") != identity:
            raise ValueError("Attempt identity changed; old-version cache cannot be reused")
        if not (attempt / "completion.json").exists():
            continue
        receipt = harness.read_json(attempt / "completion.json")
        if (receipt.get("identity") != identity or receipt.get("status") != "generated"
                or receipt.get("created_after_worker_exit") is not True
                or receipt.get("files_sha256") != attempt_files(attempt)):
            raise ValueError(f"Completed attempt artifacts changed: {attempt}")
        validated_attempt_files(attempt, identity)
        return harness.read_json(attempt / "prediction.json")
    return None


def protocol_for(runtime: dict, population_ids: list[str], hashes: dict[str, str]) -> dict:
    """Scientific identity excludes execution concurrency and subset scheduling."""
    return {"runtime": runtime, "dataset_sha256": harness.DATA_SHA256, "population_ids": population_ids,
            "source_files_sha256": hashes, "implementation": VERSION, "display_name": DISPLAY_NAME,
            "native_answer_path": ANSWER_PATH, "batch_pairs": BATCH_PAIRS,
            "ingestion": INGESTION, "batch_evidence": BATCH_EVIDENCE,
            "memory_prompts": "Unmodified official OSS defaults",
            "metadata": "Session ID/date/UTC timestamp and original batch turn range; not inserted into prompts",
            "qa_integration": "Existing benchmark integration unchanged: retrieval 100, temperature 0.1, output 256",
            "cloud_paper_reproduction": False, "common_reader": False,
            "native_fallbacks": "Count authenticated JSON/shape and one-frame pre-storage action drops; transport/storage failures remain fatal",
            "native_response_capture": "Original SDK responses retained; requests and retries unchanged",
            "telemetry": "PostHog 3.25.0: send=False, sync_mode=True, disabled=True; no background consumers"}


@contextmanager
def run_lock(run_dir: Path):
    """One parent owns a run's attempts and aggregate files until all children drain."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "execution.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def run(args: argparse.Namespace) -> int:
    with run_lock(args.run_dir.resolve()):
        return run_locked(args)


def run_locked(args: argparse.Namespace) -> int:
    validate_runtime(args)
    if type(args.workers) is not int or not 1 <= args.workers <= 4:
        raise ValueError("Workers must be between one and four")
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
    protocol = protocol_for(runtime, list(by_id), hashes)
    harness.write_once(args.run_dir / "protocol.json", protocol)
    jobs, predictions, failures = {}, {}, []
    for qid, row in by_id.items():
        source = harness.source_only(row)
        list(session_batches(source))  # Reject invalid input before any new model work.
        query = {"question_id": qid, "question": row["question"], "question_date": row["question_date"]}
        identity = {"protocol_sha256": harness.digest(protocol), "source_sha256": harness.digest(source),
                    "query_sha256": harness.digest(query)}
        history = args.run_dir / "histories" / hashlib.sha256(qid.encode()).hexdigest()[:24]
        jobs[qid] = {"source": source, "query": query, "identity": identity, "history": history}
        prediction = verified_prediction(history, identity)
        if prediction is not None:
            predictions[qid] = prediction
    history_root = args.run_dir / "histories"
    if history_root.exists() and {p.name for p in history_root.iterdir()} - {j["history"].name for j in jobs.values()}:
        raise ValueError("Unknown history in the batch8 run")
    pending = deque(qid for qid in selected if qid not in predictions)
    active = {}

    def publish(final: bool = False) -> None:
        status = "running"
        if final or failures:
            status = ("generation_incomplete" if failures else "generation_complete" if len(predictions) == 500
                      else "subset_generation_complete")
        harness.save_json(args.run_dir / "predictions.json", [predictions[qid] for qid in by_id if qid in predictions])
        harness.save_json(args.run_dir / "failures.json", failures)
        harness.save_json(args.run_dir / "status.json", {
            "method": "mem0", "implementation": VERSION, "display_name": DISPLAY_NAME,
            "planned": 500, "population": 500, "selected_count": len(selected), "generated": len(predictions),
            "failed": len(failures), "officially_judged": 0, "workers": args.workers,
            "status": status})
        harness.save_json(args.run_dir / "current.json", {"status": status, "pending": len(pending),
            "active": [{"question_id": qid, "attempt": str(attempt), "pid": child.pid}
                       for qid, (attempt, child) in active.items()]})

    def failed(qid: str, attempt: Path, code: int | None, error: Exception) -> None:
        failure = {"question_id": qid, "attempt": str(attempt), "exit_code": code, "error": str(error)}
        failures.append(failure)
        if not (attempt / "failure.json").exists():
            harness.save_json(attempt / "failure.json", failure)

    def settle(qid: str, attempt: Path, code: int) -> None:
        try:
            if code != 0:
                raise ValueError(f"Native worker failed: exit={code}")
            finalize_attempt(attempt, jobs[qid]["identity"])
            prediction = verified_prediction(jobs[qid]["history"], jobs[qid]["identity"])
            if prediction is None:
                raise ValueError("Exited native worker lacks verified completion")
            predictions[qid] = prediction
        except Exception as error:
            failed(qid, attempt, code, error)

    try:
        while active or (pending and not failures):
            for qid, (attempt, child) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                settle(qid, attempt, code)
                del active[qid]
            while pending and not failures and len(active) < args.workers:
                qid = pending.popleft()
                job = jobs[qid]
                attempt = harness.next_attempt(job["history"])
                try:
                    for name, value in (("source", job["source"]), ("query", job["query"]),
                            ("worker", {"runtime": runtime, "identity": job["identity"], "source_files_sha256": hashes})):
                        harness.save_json(attempt / f"{name}.json", value)
                    with (attempt / "console.log").open("x", encoding="utf-8") as output:
                        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-dir", str(attempt)],
                                                 stdout=output, stderr=subprocess.STDOUT)
                    active[qid] = (attempt, child)
                except Exception as error:
                    failed(qid, attempt, None, error)
            publish()
            if active:
                time.sleep(0.2)
    except BaseException as parent_error:
        failures.append({"error": f"Parent interrupted: {type(parent_error).__name__}: {parent_error}"})
        for qid, (attempt, child) in list(active.items()):
            try:
                settle(qid, attempt, child.wait())
                del active[qid]
            except Exception:
                pass  # Preserve the original parent error when persistence is unavailable.
        try:
            publish(final=True)
        except Exception:
            pass
        raise
    publish(final=True)
    return 1 if failures else 0


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
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1,
                        help="Independent question processes sharing one endpoint; scheduling only")
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