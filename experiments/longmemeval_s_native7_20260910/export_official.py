"""Export complete native-seven hypotheses for the unchanged official evaluator.

This module never calls a judge or reads reference answer/type fields. It emits
only question_id and the exact hypothesis, plus a separate provenance receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_all import DATA_SHA256, METHODS, check_complete

UPSTREAM_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
UPSTREAM_HASHES = {
    "src/evaluation/evaluate_qa.py": "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
    "src/evaluation/print_qa_metrics.py": "e9283933a0cefb7a0ded7365e436ae3d1be5aac41853325e6155d83bf07607f0",
}
FORBIDDEN = {"answer", "reference", "reference_answer", "autoeval_label", "label", "question_type"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def export(args: argparse.Namespace) -> dict[str, Any]:
    companion = args.output.with_name(args.output.name + ".receipt.json")
    if args.output.exists() or companion.exists():
        raise FileExistsError("Existing exports are preserved; choose a fresh output path")
    input_hashes: dict[Path, str] = {}

    def read(path: Path) -> Any:
        data = path.read_bytes()
        input_hashes[path] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    reference = read(args.dataset)
    if input_hashes[args.dataset] != DATA_SHA256:
        raise ValueError("Canonical LongMemEval-S hash mismatch")
    ids = [row["question_id"] for row in reference]
    del reference
    if len(ids) != 500 or any(not isinstance(qid, str) or not qid for qid in ids) or len(set(ids)) != 500:
        raise ValueError("Reference must contain exactly 500 unique IDs")
    for relative, expected in UPSTREAM_HASHES.items():
        path = args.vendor / relative
        input_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        if input_hashes[path] != expected:
            raise ValueError(f"Unchanged official evaluator hash mismatch: {relative}")
    run_dir = args.run_dir
    status_path = run_dir / ("completion.json" if args.method == "higmem" else "status.json")
    check_complete(args.method, read(status_path), 500)
    protocol = read(run_dir / "protocol.json")
    if protocol.get("dataset_sha256") != DATA_SHA256:
        raise ValueError("Generation protocol uses another reference")
    if args.method == "lightmem":
        rows = [read(path) for path in sorted(run_dir.glob("*/prediction.json"))]
    else:
        declared_method = protocol.get("method", protocol.get("runtime", {}).get("method"))
        population = protocol.get("question_ids", protocol.get("population_ids"))
        if declared_method != args.method or not isinstance(population, list) or len(population) != 500 or set(population) != set(ids):
            raise ValueError("Generation method or full population differs")
        rows = read(run_dir / "predictions.json")
    if not isinstance(rows, list) or len(rows) != 500 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Predictions must contain 500 objects")
    by_id: dict[str, dict[str, Any]] = {}
    protocol_hash = digest(protocol)
    for row in rows:
        qid, hypothesis = row.get("question_id"), row.get("hypothesis")
        if not isinstance(qid, str) or qid in by_id or qid not in ids:
            raise ValueError("Duplicate, unknown, or malformed prediction ID")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(f"Failed or empty native answer: {qid}")
        if FORBIDDEN.intersection(row):
            raise ValueError(f"Prediction input contains evaluation annotations: {qid}")
        if row.get("method", args.method) != args.method:
            raise ValueError(f"Mixed methods in predictions: {qid}")
        if args.method == "lightmem":
            for key in ("dataset_sha256", "model", "embedding_model", "upstream_sha256"):
                if not isinstance(protocol.get(key), str) or row.get(key) != protocol[key]:
                    raise ValueError(f"LightMem prediction/protocol mismatch: {qid}/{key}")
        else:
            identity = row.get("identity")
            if not isinstance(identity, dict) or identity.get("protocol_sha256") != protocol_hash:
                raise ValueError(f"Prediction belongs to another protocol: {qid}")
            for key in ("source_sha256", "query_sha256"):
                value = identity.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"Missing source/query provenance: {qid}")
            if row.get("status") != "generated" or row.get("error") or row.get("error_type"):
                raise ValueError(f"Incomplete prediction: {qid}")
            if args.method == "higmem" and row.get("native_answer_valid") is not True:
                raise ValueError(f"Invalid native HiGMem answer: {qid}")
        by_id[qid] = row
    if set(by_id) != set(ids):
        raise ValueError("Prediction coverage differs from all 500 canonical IDs")
    payload = "".join(json.dumps({"question_id": qid, "hypothesis": by_id[qid]["hypothesis"]},
                                 ensure_ascii=False) + "\n" for qid in ids)
    for path, expected in input_hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Completed input changed during export: {path}")
    receipt = {
        "schema": "native-seven-official-hypotheses-v1", "method": args.method,
        "scope": "full_canonical_500", "expected_count": 500, "expected_question_ids": ids,
        "dataset_sha256": DATA_SHA256, "inputs_sha256": {str(p): h for p, h in input_hashes.items()},
        "hypotheses_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "answer_postprocessing": "none", "officialjudge_pending": True,
        "upstream_commit": UPSTREAM_COMMIT, "upstream_files_sha256": UPSTREAM_HASHES,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status_validator_sha256": hashlib.sha256(Path(__file__).with_name("run_all.py").read_bytes()).hexdigest(),
        "judge_cli_key": "gpt-4o", "judge_snapshot": "gpt-4o-2024-08-06",
        "official_judge_calls_by_exporter": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    with companion.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    for name in ("run-dir", "dataset", "vendor", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(export(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
