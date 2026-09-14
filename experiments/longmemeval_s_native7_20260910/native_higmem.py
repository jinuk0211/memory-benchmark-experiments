"""Preserve upstream HiGMem's paper pipeline on complete LongMemEval-S histories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
PAPER_SETTINGS = dict(use_character_profile=False, use_event_metadata_mode=True,
                      ablation_no_link=True, k_event_affiliation=10)
SOURCE_FILES = ("README.md", "run_fphm_evaluation.py", "fphm_core.py", "memory_layer.py",
                "prompts.py", "fphm_structures.py", "fphm_logger.py", "load_dataset.py", "utils.py")
ANSWER_SCHEMA = {"name": "response", "schema": {"type": "object", "properties": {
    "answer": {"type": "string"}}, "required": ["answer"]}}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def is_native_answer(prediction: Any) -> bool:
    return isinstance(prediction, str) and bool(prediction.strip()) and prediction.strip() != "Could not generate answer."


def valid_generated(outcome: dict) -> bool:
    return outcome.get("status") == "generated" and outcome.get("native_answer_valid") is True and is_native_answer(outcome.get("hypothesis"))


def source_only(record: dict) -> list[dict]:
    """Exclude evaluation questions, gold answers, types and source annotations."""
    sessions, dates, ids = (record[key] for key in
                            ("haystack_sessions", "haystack_dates", "haystack_session_ids"))
    if not (len(sessions) == len(dates) == len(ids)):
        raise ValueError("History sessions, dates and IDs do not align")
    result = []
    for sid, date, turns in zip(ids, dates, sessions):
        if not isinstance(date, str):
            raise TypeError("Session date must be text")
        clean = []
        for turn in turns:
            if not isinstance(turn["role"], str) or not isinstance(turn["content"], str):
                raise TypeError("History turns require complete role/content strings")
            clean.append({"role": turn["role"], "content": turn["content"]})
        result.append({"session_id": sid, "date": date, "turns": clean})
    return result


def source_turns(sessions: list[dict]) -> list[tuple[str, str, str, str]]:
    """Retain every user and assistant turn, in original dataset order."""
    return [(f"D{si + 1}:{ti + 1}", turn["content"], turn["role"], session["date"])
            for si, session in enumerate(sessions) for ti, turn in enumerate(session["turns"])]


def build_source(system: Any, sessions: list[dict]) -> int:
    turns = source_turns(sessions)
    for turn_id, content, speaker, timestamp in turns:
        system.add_turn(turn_id=turn_id, turn_content=content, speaker=speaker, timestamp=timestamp)
    system.finalize_memory_build()
    system.build_indices()
    return len(turns)


def native_answer(system: Any, controller: Any, question: str, question_date: str,
                  generate_keyword_query: Any, build_category_prompt: Any) -> tuple[str, dict]:
    # LME has no LoCoMo category: retain the upstream general QA prompt branch.
    query_text = f"Question date: {question_date}\nQuestion: {question}"
    rewritten = generate_keyword_query(controller, query_text)
    context, retrieval = system.retrieve_for_query(
        original_query=query_text, keyword_query=rewritten["keyword_query"],
        profile_retrieval_keys=rewritten["profile_retrieval_keys"],
        k_profile=3, k_event=10, k_turn=10, return_trace=True)
    prompt = build_category_prompt(category=0, context=context, question=query_text)
    response = system._get_llm_json_response(
        prompt, ANSWER_SCHEMA, caller="final_answer_generation", temperature=0.0)
    prediction = response.get("answer", "Could not generate answer.") if response else "Could not generate answer."
    if not isinstance(prediction, str):
        raise TypeError("Native final answer must be text")
    return prediction, {"query": query_text, "rewritten_query": rewritten, "retrieval": retrieval,
                        "context": context, "final_prompt": prompt, "native_response": response}


def select_records(records: list[dict], ids_file: Path | None, limit: int) -> list[dict]:
    by_id = {row["question_id"]: row for row in records}
    if len(records) != 500 or len(by_id) != 500:
        raise ValueError("Expected 500 unique LongMemEval-S question IDs")
    if limit < 0:
        raise ValueError("--limit cannot be negative")
    if ids_file:
        ids = json.loads(ids_file.read_text(encoding="utf-8"))
        if not isinstance(ids, list) or not all(isinstance(qid, str) for qid in ids):
            raise ValueError("--ids-file must be a JSON array of IDs")
        if not ids or len(ids) != len(set(ids)) or any(qid not in by_id for qid in ids):
            raise ValueError("Subset IDs must be nonempty, unique and present")
        selected = [by_id[qid] for qid in ids]
    else:
        selected = records
    return selected[:limit] if limit else selected


def load_native(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(args.source_root.resolve()))
    import memory_layer
    from fphm_core import FPHMSystem
    from run_fphm_evaluation import build_category_prompt, generate_keyword_query
    from sentence_transformers import SentenceTransformer

    # Pin native MiniLM without changing encode defaults, truncation or prefixes.
    encoder = SentenceTransformer(str(args.embedding_model), device="cpu")
    if encoder.get_sentence_embedding_dimension() != 384 or encoder.max_seq_length != 256:
        raise ValueError("Expected pinned 384-dimensional, native-256 MiniLM")
    original_factory = memory_layer._get_sentence_transformer

    def pinned_encoder(model_name: str, device: str | None = None) -> Any:
        if model_name != "all-MiniLM-L6-v2" or device is not None:
            raise ValueError("Unexpected HiGMem embedding configuration")
        return encoder

    controller = memory_layer.LLMController(
        backend="openai", model=args.model, api_base=args.api_base, api_key=args.api_key)

    def make_system(run_name: str, log_dir: Path) -> Any:
        memory_layer._get_sentence_transformer = pinned_encoder
        try:
            return FPHMSystem(controller, run_name, log_dir=str(log_dir), **PAPER_SETTINGS)
        finally:
            memory_layer._get_sentence_transformer = original_factory

    return controller, make_system, generate_keyword_query, build_category_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "run-dir", "source-root", "embedding-model"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="First N IDs for smoke; zero selects all")
    args = parser.parse_args()
    if args.model != "Qwen/Qwen3.5-9B":
        parser.error("--model is fixed to Qwen/Qwen3.5-9B")
    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Canonical LongMemEval-S cleaned SHA-256 mismatch")
    population = json.loads(raw)
    records = select_records(population, args.ids_file, args.limit)
    protocol = {
        "method": "higmem", "dataset_sha256": DATA_SHA256, "question_ids": [r["question_id"] for r in population],
        "model": args.model, "api_base": args.api_base, "embedding_model": str(args.embedding_model),
        "paper_settings": PAPER_SETTINGS, "retrieval_k": {"profile": 3, "event": 10, "turn": 10},
        "source_sha256": {name: hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "construction_fields": ["session_id", "date", "turns.role", "turns.content"],
        "answer_path": "upstream general build_category_prompt + FPHMSystem._get_llm_json_response",
        "dataset_adaptation": "Original source order/roles; question_date in query; no LoCoMo category mapping",
        "context_truncation": False, "generation_budget_override": False,
        "embedding": {"device": "cpu", "dimension": 384, "max_seq_length": 256, "query_prefix": None},
        "upstream_failure_policy": "Preserved; empty/fallback native answers flagged separately"}
    protocol_path = args.run_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Run protocol differs; use a separate run directory")
    write_json(protocol_path, protocol)
    controller, make_system, rewrite, prompt_builder = load_native(args)
    outcomes = []
    for record in records:
        qid = record["question_id"]
        history_dir = args.run_dir / "histories" / hashlib.sha256(qid.encode()).hexdigest()[:24]
        source = source_only(record)
        query = {key: record[key] for key in ("question_id", "question", "question_date")}
        identity = {"protocol_sha256": digest(protocol), "source_sha256": digest(source), "query_sha256": digest(query)}
        attempts = sorted(history_dir.glob("attempt_*"))
        saved = None
        for previous in reversed(attempts):
            path = previous / "prediction.json"
            if path.exists():
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if candidate["question_id"] != qid or candidate["identity"] != identity:
                    raise ValueError("Saved history identity mismatch")
                if valid_generated(candidate):
                    saved = candidate
                    break
        if saved is not None:
            outcomes.append(saved)
            continue
        attempt = history_dir / f"attempt_{len(attempts) + 1:04d}"
        write_json(attempt / "source.json", source)
        write_json(attempt / "query.json", query)
        write_json(attempt / "native_config.json", protocol)
        started = time.monotonic()
        system = None
        counters = ("prompt_tokens", "completion_tokens", "total_tokens")
        before = {key: getattr(controller.llm, key, 0) for key in counters}
        outcome = {"question_id": qid, "method": "higmem", "identity": identity, "officially_judged": False}
        try:
            system = make_system(history_dir.name, attempt / "native_logs")
            turn_count = build_source(system, source)
            prediction, trace = native_answer(system, controller, query["question"], query["question_date"], rewrite, prompt_builder)
            write_json(attempt / "trace.json", trace)
            valid_answer = is_native_answer(prediction)
            outcome.update(hypothesis=prediction, status="generated" if valid_answer else "failed",
                           source_sessions=len(source), source_turns=turn_count,
                           memory_turns=len(system.turn_notes), events=len(system.events),
                           profiles=len(system.profiles), native_answer_valid=valid_answer)
            if not valid_answer:
                outcome.update(error_type="NativeAnswerFailure", error="Upstream returned an empty or fallback answer")
        except Exception as error:
            outcome.update(hypothesis="", status="failed", error_type=type(error).__name__, error=str(error))
        finally:
            if system is not None:
                system.executor.shutdown(wait=True)
        outcome.update(elapsed_seconds=time.monotonic() - started)
        write_json(attempt / "usage.json", {key: getattr(controller.llm, key, 0) - before[key] for key in counters})
        write_json(attempt / "prediction.json", outcome)
        outcomes.append(outcome)
        write_json(args.run_dir / "predictions.json", outcomes)
        write_json(args.run_dir / "status.json", {"selected": len(records), "processed": len(outcomes),
                   "generated": sum(valid_generated(r) for r in outcomes), "latest_question_id": qid})
        print(json.dumps({"question_id": qid, "status": outcome["status"], "processed": len(outcomes)}), flush=True)
    write_json(args.run_dir / "predictions.json", outcomes)
    predictions = args.run_dir / "predictions.jsonl"
    temporary = predictions.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"]}, ensure_ascii=False) + "\n" for r in outcomes if valid_generated(r)), encoding="utf-8")
    temporary.replace(predictions)
    failed = [r["question_id"] for r in outcomes if not valid_generated(r)]
    write_json(args.run_dir / "completion.json", {"method": "higmem", "selected": len(records),
               "results": len(outcomes), "valid_predictions": sum(valid_generated(r) for r in outcomes), "failed_ids": failed, "benchmark_complete": not failed and len(outcomes) == 500,
               "run_complete": not failed, "invalid_native_answers": sum(not r.get("native_answer_valid", False) for r in outcomes),
               "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(), "official_accuracy": None})
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
