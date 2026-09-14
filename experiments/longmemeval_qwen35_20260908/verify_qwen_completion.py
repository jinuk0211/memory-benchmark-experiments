"""Read-only completion audit before the current Qwen GPU can change jobs."""
import argparse
from collections.abc import Callable
from functools import lru_cache
import importlib.metadata
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from controlled_reader import DATA_SHA256, validate_reference
from materialize_inputs import write_once
from prepare_inputs import REFERENCE_SHA256
from run_history import verify_files
from replay_qwen_reader import reference_digest, replay

ARMS = ("seed", "r40_fused_four_turn", "s_parent_single_2000")
HOLD_HASHES = {
    "smoke_writer65k.py": "a103a56e8f191907683e38e2bd9dfa2c228266da5b125f86c3edae65cbc4a0d6",
    "GEMMA_FULL500_HOLD.json": "21f5601eb148635f227020c972e8ea3537295d294658b089023ac73d16dc984c",
    "run_full_queue.sh": "24e744b5ab7cb0b00cfc874a8ec8c85d85da94fed83e62e7e951231cfe505470",
}


class NotReady(RuntimeError):
    """The current job has not reached a verified terminal completion."""


def qwen_source(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct the original transfer adapter's source-only session format."""
    result = []
    for number, (turns, date) in enumerate(
        zip(record["haystack_sessions"], record["haystack_dates"]), start=1
    ):
        clean = []
        for index, turn in enumerate(turns, start=1):
            tid, role, text = f"D{number}:{index}", turn["role"], turn["content"]
            clean.append(dict(id=tid, speaker=role, body=text, text=f"[{tid}] {role}: {text}"))
        result.append(dict(num=number, date=date, turns=clean))
    return result


def verify_arm(
    path: Path, method: str, expected: dict[str, dict[str, Any]], item_dir: Path,
    replay_question: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Require all500 actual per-question outputs, including failed/empty answers."""
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [row["question_id"] for row in rows]
    if len(rows) != 500 or len(set(ids)) != 500 or set(ids) != set(expected):
        raise ValueError(f"{method}: expected exactly500 unique question IDs")
    empty = length_stops = 0
    for row in rows:
        qid = row["question_id"]
        target = expected[qid]
        if row["method"] != method or row["conversation_id"] != qid:
            raise ValueError(f"{method}: mismatched method/history")
        for field, value in (
            ("question", target["question"]), ("question_date", target["question_date"]),
            ("question_type", target["question_type"]), ("gold", str(target["answer"])),
            ("is_abstention", qid.endswith("_abs")),
        ):
            if type(row.get(field)) is not type(value) or row[field] != value:
                raise ValueError(f"{method}/{qid}: wrong {field}")
        prediction = row.get("hypothesis")
        if not isinstance(prediction, str) or row.get("prediction") != prediction:
            raise ValueError(f"{method}/{qid}: missing generation text")
        status = "ok" if prediction.strip() else "generation_empty"
        if row["status"] != status or row["finish_reason"] not in ("stop", "length"):
            raise ValueError(f"{method}/{qid}: invalid generation status")
        for field, maximum in (("read_tokens", 2048), ("output_tokens", 96)):
            if type(row.get(field)) is not int or not 0 <= row[field] <= maximum:
                raise ValueError(f"{method}/{qid}: invalid {field}")
        replayed = replay_question(qid)
        for field, value in replayed.items():
            if type(row.get(field)) is not type(value) or row[field] != value:
                raise ValueError(f"{method}/{qid}: generation/retrieval replay mismatch: {field}")
        if json.loads((item_dir / f"{qid}.json").read_bytes()) != [row]:
            raise ValueError(f"{method}/{qid}: aggregate and per-item result differ")
        empty += status == "generation_empty"
        length_stops += row["finish_reason"] == "length"
    return dict(questions=500, empty_generations=empty, length_stops=length_stops,
                file_sha256=hashlib.sha256(raw).hexdigest())


def verify_artifacts(run_dir: Path, transfer_root: Path, dataset: Path) -> dict[str, Any]:
    """Verify the complete population, frozen sources and all actual memory files."""
    state = json.loads((run_dir / "status.json").read_bytes())
    if state.get("phase") != "generation_complete":
        raise NotReady(f"Qwen generation is not complete: {state.get('phase')}")
    if type(state.get("questions")) is not int or state["questions"] != 500:
        raise ValueError("Qwen completion status has the wrong population")
    raw_protocol = (run_dir / "protocol.json").read_bytes()
    if hashlib.sha256(raw_protocol).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Qwen full500 protocol changed")
    protocol = json.loads(raw_protocol)
    validate_reference(protocol)
    if protocol["methods"] != list(ARMS):
        raise ValueError("Qwen comparison arms changed")
    verify_files(transfer_root, protocol["source_hashes"])
    verify_files(transfer_root, HOLD_HASHES)
    raw_data = dataset.read_bytes()
    if hashlib.sha256(raw_data).hexdigest() != DATA_SHA256:
        raise ValueError("Canonical dataset changed")
    records = json.loads(raw_data)
    del raw_data
    expected = {row["question_id"]: row for row in records}
    if len(records) != 500 or len(expected) != 500 or set(expected) != set(
        protocol["selection"]["selected_ids"]
    ):
        raise ValueError("Dataset population mismatch")
    sources = json.loads((run_dir / "source_sessions.json").read_bytes())
    if sources != {qid: qwen_source(row) for qid, row in expected.items()}:
        raise ValueError("Qwen source histories differ from the canonical dataset")
    memories = {
        method: {qid: json.loads((run_dir / "memories" / method / f"{qid}.json").read_bytes())
                 for qid in expected}
        for method in ARMS
    }
    lock_bytes = (run_dir / "memory_lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    if (lock.get("benchmark_questions_used") is not False
            or lock["protocol_sha256"] != reference_digest(protocol)
            or lock["source_sha256"] != reference_digest(sources)
            or lock["memories_sha256"] != reference_digest(memories)):
        raise ValueError("Qwen protocol/source/memory lock mismatch")
    for package in ("transformers", "numpy", "rank-bm25"):
        if importlib.metadata.version(package) != protocol["actual_packages"][package]:
            raise ValueError(f"Use the original Qwen runtime for CPU replay: {package}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(protocol["model"]["path"], local_files_only=True)

    @lru_cache(maxsize=100000)
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    arms = {}
    for method in ARMS:
        def replay_question(qid: str) -> dict[str, Any]:
            return replay(memories[method][qid], expected[qid], protocol,
                          run_dir / "cache", count_tokens)
        arms[method] = verify_arm(run_dir / f"{method}.jsonl", method, expected,
                                 run_dir / "items" / method, replay_question)
    return dict(
        status="qwen_artifacts_verified", reference_protocol_sha256=REFERENCE_SHA256,
        dataset_sha256=DATA_SHA256, arms=arms, source_files_verified=len(protocol["source_hashes"]),
        memory_files_verified=1500, reader_replays_verified=1500, memory_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        gemma_held=True, official_judge_complete=False,
    )


def queue_state() -> str:
    result = subprocess.run(
        ["supervisorctl", "status", "memory-transfer-full-queue"],
        check=False, capture_output=True, text=True, timeout=20,
    )
    fields = result.stdout.strip().split()
    if result.returncode not in (0, 3) or len(fields) < 2 or fields[0] != "memory-transfer-full-queue":
        raise RuntimeError("Cannot establish current Supervisor queue state")
    return fields[1]


def require_idle_gpu() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    if result.stdout.strip():
        raise NotReady("GPU0 still has compute processes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "transfer-root", "dataset", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    queue = queue_state()
    try:
        result = verify_artifacts(args.run_dir, args.transfer_root, args.dataset)
    except NotReady:
        if queue not in ("RUNNING", "STARTING", "STOPPING", "BACKOFF"):
            raise RuntimeError(f"Qwen queue is {queue} before generation completed") from None
        raise
    queue = queue_state()
    if queue not in ("EXITED", "STOPPED"):
        raise NotReady(f"Qwen queue is still {queue}")
    require_idle_gpu()
    result.update(status="qwen_generation_verified_gpu_idle", queue_state=queue,
                  verified_at=datetime.now(timezone.utc).isoformat())
    write_once(args.out, result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotReady as exc:
        print(json.dumps({"status": "not_ready", "reason": str(exc)}))
        raise SystemExit(2) from None