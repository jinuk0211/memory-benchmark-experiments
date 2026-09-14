"""Materialize immutable source-only histories and separate read-time queries."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_inputs import digest, preflight, source_only
from controlled_reader import DATA_SHA256


def write_once(path: Path, value: Any) -> None:
    """Keep existing identical artifacts; reject an incompatible experiment."""
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError(f"Existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    population = preflight(args.dataset, args.reference)
    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset changed after preflight")
    by_id = {row["question_id"]: row for row in json.loads(raw)}
    del raw
    rows = []
    for entry in population["source_manifest"]:
        qid, source_hash = entry["question_id"], entry["source_sha256"]
        record = by_id[qid]
        source = source_only(record)
        query = {key: record[key] for key in ("question_id", "question", "question_date")}
        query_hash = digest(query)
        write_once(args.out / "sources" / f"{source_hash}.json", source)
        write_once(args.out / "queries" / f"{query_hash}.json", query)
        rows.append(dict(entry, query_sha256=query_hash))
    manifest = {
        "dataset_sha256": population["dataset_sha256"],
        "reference_protocol_sha256": population["reference_protocol_sha256"],
        "required_methods": population["required_methods"], "questions": rows,
        "source_files": len({row["source_sha256"] for row in rows}),
        "source_fields": population["source_fields"],
    }
    write_once(args.out / "manifest.json", manifest)
    print(json.dumps({"questions": len(rows), "source_files": manifest["source_files"],
                      "status": "source_query_files_materialized_no_inference"}))


if __name__ == "__main__":
    main()