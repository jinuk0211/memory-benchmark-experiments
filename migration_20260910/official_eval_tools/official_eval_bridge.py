"""Export hypotheses and audit unchanged LongMemEval labels; never call a judge."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CANONICAL_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
UPSTREAM_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
UPSTREAM_HASHES = {
    "src/evaluation/evaluate_qa.py": "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
    "src/evaluation/print_qa_metrics.py": "e9283933a0cefb7a0ded7365e436ae3d1be5aac41853325e6155d83bf07607f0",
}
JUDGE_MODEL = "gpt-4o-2024-08-06"
QUESTION_TYPES = (
    "single-session-user", "single-session-preference", "single-session-assistant",
    "multi-session", "temporal-reasoning", "knowledge-update",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def unique_ids(values: object, label: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label}: nonempty ID list required")
    if any(not isinstance(qid, str) or not qid for qid in values):
        raise ValueError(f"{label}: IDs must be nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}: duplicate IDs")
    return values


def indexed(rows: list[dict], expected: list[str], label: str) -> dict[str, dict]:
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label}: each row must be an object")
    ids = unique_ids([row.get("question_id") for row in rows], label)
    if set(ids) != set(expected):
        raise ValueError(f"{label}: ID coverage mismatch; "
                         f"missing={len(set(expected) - set(ids))}, "
                         f"extra={len(set(ids) - set(expected))}")
    return dict(zip(ids, rows))


def reference_metadata(path: Path) -> dict[str, str]:
    if sha256(path) != CANONICAL_SHA256:
        raise ValueError("Canonical cleaned-S reference SHA256 mismatch")
    # Read no question, answer, history, or previous outcome fields.
    rows = read_json(path)
    if sha256(path) != CANONICAL_SHA256:
        raise ValueError("Canonical reference changed while reading metadata")
    ids = unique_ids([row["question_id"] for row in rows], "reference")
    types = [row["question_type"] for row in rows]
    if any(qtype not in QUESTION_TYPES for qtype in types):
        raise ValueError("Unknown reference question type")
    return dict(zip(ids, types))


def expected_ids(path: Path, partition: str | None, reference: dict[str, str]) -> list[str]:
    manifest = read_json(path)
    if isinstance(manifest, dict):
        data_hash = manifest.get("dataset", {}).get("sha256")
        if data_hash is not None and data_hash != CANONICAL_SHA256:
            raise ValueError("Expected-ID manifest dataset hash mismatch")
        if "partitions" in manifest:
            if partition not in manifest["partitions"]:
                raise ValueError("Frozen split manifest requires a valid --partition")
            ids = manifest["partitions"][partition]["question_ids"]
        else:
            if partition is not None:
                raise ValueError("--partition requires a frozen split manifest")
            ids = manifest.get("question_ids", manifest.get("selected_ids"))
    else:
        if partition is not None:
            raise ValueError("--partition requires a frozen split manifest")
        ids = manifest
    ids = unique_ids(ids, "expected population")
    if not set(ids) <= set(reference):
        raise ValueError("Expected population contains unknown reference IDs")
    selected = set(ids)
    return [qid for qid in reference if qid in selected]


def verify_upstream(vendor: Path) -> dict[str, str]:
    for relative, expected in UPSTREAM_HASHES.items():
        if sha256(vendor / relative) != expected:
            raise ValueError(f"Frozen official file changed: {relative}")
    return UPSTREAM_HASHES.copy()


def prepare(args: argparse.Namespace) -> tuple[list[dict], dict, dict[str, str]]:
    upstream = verify_upstream(args.vendor)
    input_hashes = {name: sha256(getattr(args, name))
                    for name in ("predictions", "protocol", "expected_ids")}
    metadata = reference_metadata(args.reference)
    ids = expected_ids(args.expected_ids, args.partition, metadata)
    protocol = read_json(args.protocol)
    data_hash = protocol.get("dataset_sha256")
    if data_hash != CANONICAL_SHA256:
        if protocol.get("canonical_parent_sha256") != CANONICAL_SHA256:
            raise ValueError("Derived generation data requires canonical_parent_sha256")
        if (not isinstance(data_hash, str) or len(data_hash) != 64
                or any(char not in "0123456789abcdef" for char in data_hash)):
            raise ValueError("Missing generation data SHA256")
    if args.method not in protocol.get("evaluation_methods", protocol.get("methods", [])):
        raise ValueError("Requested arm is not declared by generation protocol")
    selection = unique_ids(protocol.get("selection", {}).get("selected_ids"),
                           "generation protocol selection")
    if set(selection) != set(ids):
        raise ValueError("Generation protocol selection differs from expected population")
    predictions = indexed(read_jsonl(args.predictions), ids, "predictions")
    hypotheses = []
    empty_ids = []
    for qid in ids:
        row = predictions[qid]
        hyp = row.get("hypothesis")
        if not isinstance(hyp, str):
            raise ValueError(f"Non-string hypothesis: {qid}")
        if row.get("method") != args.method:
            raise ValueError(f"Mixed or inconsistent method: {qid}")
        if row.get("prediction") != hyp:
            raise ValueError(f"prediction/hypothesis mismatch: {qid}")
        status = "ok" if hyp.strip() else "generation_empty"
        if row.get("status") != status:
            raise ValueError(f"Failed or inconsistent generation status: {qid}")
        if not hyp.strip():
            empty_ids.append(qid)
        hypotheses.append({"question_id": qid, "hypothesis": hyp})
    if any(sha256(getattr(args, name)) != digest for name, digest in input_hashes.items()):
        raise ValueError("Input changed during export/verification; use completed frozen inputs")
    config = protocol.get("config", {})
    receipt = {
        "schema": "longmemeval-official-bridge-v1",
        "scope": "full_canonical_500" if set(ids) == set(metadata) else "diagnostic_subset",
        "partition": args.partition,
        "expected_question_ids": ids,
        "expected_count": len(ids),
        "canonical_population_count": len(metadata),
        "method": args.method,
        "model_name": config.get("model"),
        "embedding_name": config.get("embed_model"),
        "model": protocol.get("model"),
        "embedding": protocol.get("embedding"),
        "read_budget": protocol.get("read_budget"),
        "extra_storage_budget": protocol.get("extra_storage_budget"),
        "max_answer_tokens": protocol.get("max_answer_tokens"),
        "generation_data_sha256": data_hash,
        "canonical_parent_sha256": protocol.get("canonical_parent_sha256"),
        "reference_sha256": CANONICAL_SHA256,
        "predictions_sha256": input_hashes["predictions"],
        "protocol_sha256": input_hashes["protocol"],
        "expected_ids_manifest_sha256": input_hashes["expected_ids"],
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_sha256": upstream,
        "bridge_sha256": sha256(Path(__file__)),
        "completed_empty_question_ids": empty_ids,
        "answer_postprocessing": "none",
    }
    return hypotheses, receipt, metadata


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def receipt_path(hypotheses: Path) -> Path:
    return hypotheses.with_name(hypotheses.name + ".receipt.json")


def export(args: argparse.Namespace) -> dict:
    hypotheses, receipt, _ = prepare(args)
    companion = receipt_path(args.hypotheses)
    if args.hypotheses.exists() or companion.exists():
        raise FileExistsError("Use a fresh hypothesis filename; existing attempts are preserved")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in hypotheses)
    receipt["hypotheses_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with args.hypotheses.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    with companion.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json_text(receipt))
    return receipt


def score_group(labels: list[bool]) -> dict:
    return {"correct": sum(labels), "count": len(labels),
            "accuracy": sum(labels) / len(labels) if labels else None}


def verify(args: argparse.Namespace) -> dict:
    hypotheses, receipt, metadata = prepare(args)
    receipt["hypotheses_sha256"] = sha256(args.hypotheses)
    if read_json(receipt_path(args.hypotheses)) != receipt:
        raise ValueError("Export provenance changed or receipt does not match current inputs")
    if read_jsonl(args.hypotheses) != hypotheses:
        raise ValueError("Exported hypotheses changed")
    ids = receipt["expected_question_ids"]
    results_hash = sha256(args.results)
    results = indexed(read_jsonl(args.results), ids, "official results")
    by_type = {qtype: [] for qtype in QUESTION_TYPES}
    abstention = []
    all_labels = []
    for hyp in hypotheses:
        qid = hyp["question_id"]
        row = results[qid]
        if set(row) != {"question_id", "hypothesis", "autoeval_label"}:
            raise ValueError(f"Unexpected official output fields: {qid}")
        if row["hypothesis"] != hyp["hypothesis"]:
            raise ValueError(f"Official result hypothesis changed: {qid}")
        verdict = row["autoeval_label"]
        if not isinstance(verdict, dict) or set(verdict) != {"model", "label"}:
            raise ValueError(f"Malformed official label: {qid}")
        if verdict["model"] != JUDGE_MODEL or type(verdict["label"]) is not bool:
            raise ValueError(f"Wrong judge snapshot or non-Boolean label: {qid}")
        label = verdict["label"]  # Preserve upstream labels; do not parse judge text.
        all_labels.append(label)
        by_type[metadata[qid]].append(label)
        if "_abs" in qid:  # Exact upstream abstention grouping.
            abstention.append(label)
    per_type = {qtype: score_group(labels) for qtype, labels in by_type.items()}
    macro = (sum(group["accuracy"] for group in per_type.values()) / len(QUESTION_TYPES)
             if all(group["count"] for group in per_type.values()) else None)
    if sha256(args.results) != results_hash:
        raise ValueError("Official results changed during verification")
    report = {
        "status": "VALIDATED_OFFICIAL_LABELS",
        "execution_evidence": "Run unchanged scripts and preserve commands and stdout separately; labels alone do not prove execution.",
        "provenance": receipt,
        "hypotheses_receipt_sha256": sha256(receipt_path(args.hypotheses)),
        "official_results_sha256": results_hash,
        "judge_requested_model": JUDGE_MODEL,
        "overall": score_group(all_labels),
        "per_question_type": per_type,
        "six_type_macro_accuracy": macro,
        "abstention": score_group(abstention),
        "metric_units": "fraction in [0, 1]",
        "label_semantics": "Unmodified upstream Boolean labels; no custom verdict interpretation.",
        "limitations": "No raw judge text, returned model identity, or API usage is available in upstream result rows.",
    }
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json_text(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "verify"):
        command = commands.add_parser(name)
        for option in ("predictions", "protocol", "reference", "expected-ids", "vendor", "hypotheses"):
            command.add_argument("--" + option, type=Path, required=True)
        command.add_argument("--method", required=True)
        command.add_argument("--partition")
        if name == "verify":
            command.add_argument("--results", type=Path, required=True)
            command.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = export(args) if args.command == "export" else verify(args)
    print(json_text(result))


if __name__ == "__main__":
    main()
