"""Validate the full baseline population and separate source from evaluation data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from controlled_reader import DATA_SHA256, validate_reference

REFERENCE_SHA256 = "bf3425ad00d32e43c1022c7a11c98258a5c396268c8a09449c038dc8efa7cf9c"
# User excluded LightMem and HiGMem on 2026-09-08 before any baseline run.
METHODS = ("e_mem", "simplemem", "mem0", "langmem", "a_mem")


def source_only(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only original session identity, dates and complete role/content turns."""
    sessions, dates, ids = (record[key] for key in
                            ("haystack_sessions", "haystack_dates", "haystack_session_ids"))
    if not (len(sessions) == len(dates) == len(ids)):
        raise ValueError("Session IDs, dates and histories must align")
    result = []
    for sid, date, turns in zip(ids, dates, sessions):
        clean = []
        for turn in turns:
            if not isinstance(turn["role"], str) or not isinstance(turn["content"], str):
                raise TypeError("Expected complete text turns")
            clean.append({"role": turn["role"], "content": turn["content"]})
        result.append({"session_id": sid, "date": date, "turns": clean})
    return result


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preflight(dataset: Path, reference: Path) -> dict[str, Any]:
    """Audit every source history; no model call or result-based selection."""
    reference_bytes = reference.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Current Qwen full500 reference protocol changed")
    protocol = json.loads(reference_bytes)
    validate_reference(protocol)
    raw = dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Canonical LongMemEval dataset changed")
    records = json.loads(raw)
    del raw
    by_id = {record["question_id"]: record for record in records}
    selected_ids = protocol["selection"]["selected_ids"]
    if (len(records) != 500 or len(by_id) != 500 or len(selected_ids) != 500
            or len(set(selected_ids)) != 500 or set(selected_ids) != set(by_id)):
        raise ValueError("Exact500 unique question coverage mismatch")
    counts: Counter[str] = Counter()
    source_manifest = []
    sessions = turns = 0
    for qid in selected_ids:
        record = by_id[qid]
        source = source_only(record)
        session_count = len(source)
        turn_count = sum(len(session["turns"]) for session in source)
        sessions += session_count
        turns += turn_count
        counts[record["question_type"]] += 1
        source_manifest.append({
            "question_id": qid, "source_sha256": digest(source),
            "sessions": session_count, "turns": turn_count,
        })
    abstention = sum(qid.endswith("_abs") for qid in selected_ids)
    clusters = len({qid.removesuffix("_abs") for qid in selected_ids})
    if (sessions, turns, abstention, clusters) != (23867, 246750, 30, 471):
        raise ValueError("Canonical full500 source or subgroup coverage mismatch")
    return {
        "status": "source_population_verified_baselines_not_launched",
        "dataset_sha256": DATA_SHA256, "reference_protocol_sha256": REFERENCE_SHA256,
        "required_methods": list(METHODS), "question_count": 500,
        "session_count": sessions, "turn_count": turns,
        "abstention_count": abstention, "base_question_clusters": clusters,
        "question_type_counts": dict(sorted(counts.items())),
        "source_fields": ["session_id", "date", "turns.role", "turns.content"],
        "source_manifest": source_manifest,
        "controlled_reader": {
            "model": protocol["config"]["model"], "revision": protocol["model"]["revision"],
            "reader_prompt": protocol["reader_prompt"], "read_budget": 2048,
            "max_answer_tokens": 96, "temperature": 0, "thinking": False,
            "tokenizer_counting": "Qwen pinned tokenizer; add_special_tokens=False",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args.dataset, args.reference)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out.exists():
        if args.out.read_text(encoding="utf-8") != rendered:
            raise ValueError("Existing preflight differs; preserve it and use a new path")
    else:
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("source_manifest", "controlled_reader")}))


if __name__ == "__main__":
    main()
