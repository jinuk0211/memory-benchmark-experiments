"""Create a four-turn, one-question LoCoMo smoke dataset for HiGMem."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "MemoryData" / "datasets" / "LoCoMo" / "locomo10.json"
DESTINATION = ROOT / "higmem_smoke_locomo.json"
SELECTION = ROOT / "higmem_smoke_selection.json"
CONVERSATION = "conv-26"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    sample = copy.deepcopy(next(row for row in source if row["sample_id"] == CONVERSATION))
    question = copy.deepcopy(sample["qa"][0])
    if question.get("evidence") != ["D1:3"]:
        raise ValueError("Expected canonical smoke evidence D1:3")
    conversation = sample["conversation"]
    sample["conversation"] = {
        "speaker_a": conversation["speaker_a"],
        "speaker_b": conversation["speaker_b"],
        "session_1_date_time": conversation["session_1_date_time"],
        "session_1": conversation["session_1"][:4],
    }
    ids = [turn["dia_id"] for turn in sample["conversation"]["session_1"]]
    if ids != ["D1:1", "D1:2", "D1:3", "D1:4"]:
        raise ValueError("Unexpected canonical turn order")
    smoke = [sample if row["sample_id"] == CONVERSATION else row for row in source]
    if sum(qa.get("category") in (1, 2, 3, 4) for row in smoke for qa in row["qa"]) != 1540:
        raise ValueError("Smoke file must preserve the official 1,540-question population")
    write(DESTINATION, smoke)
    write(
        SELECTION,
        {
            "source": str(SOURCE),
            "source_sha256": sha256(SOURCE),
            "conversation": CONVERSATION,
            "turn_ids": ids,
            "questions": 1540,
            "executed_questions": 1,
            "smoke_sha256": sha256(DESTINATION),
            "smoke_only": True,
        },
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
