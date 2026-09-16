"""Create a one-conversation, one-question integrity-checked ours smoke input."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ours_input_gemma"
DESTINATION = ROOT / "ours_input_smoke"
CONVERSATION = "conv-26"


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir()
    for name in ("plans.json",):
        shutil.copy2(SOURCE / name, DESTINATION / name)
    sessions = json.loads((SOURCE / "source_sessions.json").read_text(encoding="utf-8"))
    utility = json.loads((SOURCE / "source_utility.json").read_text(encoding="utf-8"))
    questions = [
        row for row in json.loads((SOURCE / "reader_questions.json").read_text(encoding="utf-8"))
        if row["conv_id"] == CONVERSATION
    ][:1]
    if len(questions) != 1:
        raise ValueError("Smoke question selection failed")
    smoke_sessions = [dict(sessions[CONVERSATION][0])]
    smoke_sessions[0]["turns"] = smoke_sessions[0]["turns"][:10]
    source_ids = {turn["id"] for turn in smoke_sessions[0]["turns"]}
    smoke_utility = [
        row for row in utility[CONVERSATION]
        if set(row.get("source_ids", ())).issubset(source_ids)
        and set(row.get("candidate_context_ids", ())).issubset(source_ids)
    ]
    if not smoke_utility:
        raise ValueError("No source-utility probes remain in smoke input")
    write(DESTINATION / "source_sessions.json", {CONVERSATION: smoke_sessions})
    write(DESTINATION / "source_utility.json", {CONVERSATION: smoke_utility})
    write(DESTINATION / "reader_questions.json", questions)
    category_counts = Counter(str(row["category"]) for row in questions)
    write(
        DESTINATION / "selection.json",
        {
            "records": [
                {key: row[key] for key in ("id", "conv_id", "qa_index", "category")}
                for row in questions
            ],
            "selected": 1,
            "source_population": 1540,
            "selection": "Deterministic first canonical question for API smoke only",
            "category_counts": dict(category_counts),
            "conversation_counts": {CONVERSATION: 1},
        },
    )
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {
        name: digest(DESTINATION / name)
        for name in ("plans.json", "reader_questions.json", "selection.json",
                     "source_sessions.json", "source_utility.json")
    }
    manifest["smoke_only"] = {"conversation": CONVERSATION, "questions": 1, "turns": 10, "utility_probes": len(smoke_utility)}
    write(DESTINATION / "manifest.json", manifest)
    print(DESTINATION)


if __name__ == "__main__":
    main()