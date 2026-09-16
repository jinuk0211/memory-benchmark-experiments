"""Create the all-1540 source-only input for the Gemma no-binding run."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ours_source" / "input"
CODE = ROOT / "ours_source" / "code"
DATASET = ROOT / "MemoryData" / "datasets" / "LoCoMo" / "locomo10.json"
DEST = ROOT / "ours_input_gemma"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    original_manifest = read(SOURCE / "manifest.json")
    for name in ("source_sessions.json", "plans.json", "source_utility.json"):
        if digest(SOURCE / name) != original_manifest["files"][name]:
            raise RuntimeError(f"Frozen source input changed: {name}")
        DEST.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, DEST / name)

    dataset = read(DATASET)
    questions = []
    for sample in dataset:
        cid = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            if int(qa["category"]) not in (1, 2, 3, 4):
                continue
            questions.append({
                "id": f"{cid}:{index}",
                "conv_id": cid,
                "qa_index": index,
                "category": int(qa["category"]),
                "question": qa["question"],
            })
    questions.sort(key=lambda row: (row["conv_id"], row["qa_index"]))
    if len(questions) != 1540 or len({row["id"] for row in questions}) != 1540:
        raise RuntimeError("Expected 1540 unique category 1-4 questions")
    save(DEST / "reader_questions.json", questions)
    save(DEST / "selection.json", {
        "records": [{key: row[key] for key in ("id", "conv_id", "qa_index", "category")} for row in questions],
        "selected": 1540,
        "source_population": 1540,
        "selection": "All canonical category1-4 questions; no sampling",
        "category_counts": dict(Counter(row["category"] for row in questions)),
        "conversation_counts": dict(Counter(row["conv_id"] for row in questions)),
    })
    file_hashes = {
        path.name: digest(path)
        for path in sorted(DEST.glob("*.json"))
        if path.name != "manifest.json"
    }
    code_hashes = {
        str(path.relative_to(CODE)).replace("\\", "/"): digest(path)
        for path in sorted(CODE.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    save(DEST / "manifest.json", {
        "files": file_hashes,
        "code": code_hashes,
        "original_dataset_sha256": digest(DATASET),
        "sample_count": 1540,
        "arms": ["no_binding"],
        "source_utility_count": len(read(DEST / "source_utility.json")),
        "answers_uploaded": False,
        "model_transfer": "Gemma writer and reader; frozen source-only utility and recipes",
    })
    print(json.dumps({"questions": len(questions), "dataset_sha256": digest(DATASET)}, indent=2))


if __name__ == "__main__":
    main()