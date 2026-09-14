"""Verify the four preserved LangMem conversations without rewriting their run ID."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import locomo_server_queue as queue
from scripts.finalize_locomo_comparison import get_template, normalize_records
from scripts.score_locomo_comparison import (
    expected_questions, read_jsonl, summarize_usage, validate_predictions, write_report,
)


def audit_reuse(plan):
    """Require pinned, complete original shards and every attributed request."""
    reuse = plan["reused_langmem"]
    source = Path(reuse["source"]).resolve()
    expected = expected_questions(plan["dataset"], 1540)
    samples = list(dict.fromkeys(sample for sample, _ in expected))[:4]
    selected = {key: qa for key, qa in expected.items() if key[0] in samples}
    global_ids = {(sample, str(qa.get("question_id") or f"{sample}_qa{index}")): offset
                  for offset, ((sample, index), qa) in enumerate(expected.items())}
    item = next(item for item in plan["methods"] if item["method"] == "langmem")
    agent = yaml.safe_load(Path(item["agent_config"]).read_text())
    dataset = yaml.safe_load(Path(item["dataset_config"]).read_text())
    template = get_template(dataset["sub_dataset"], "query", agent["agent_name"])
    hashes, records = {}, []

    def pinned(path):
        path = path.resolve()
        digest = plan["file_sha256"].get(str(path))
        if not digest or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Reused LangMem artifact is unpinned or changed")
        hashes[str(path)] = digest
        return path

    for index, sample in enumerate(samples):
        context = source / f"context_{index:02d}"
        marker = json.loads(pinned(context / "completion.json").read_text())
        expected_count = sum(key[0] == sample for key in selected)
        raw_path = Path(marker["result_path"]).resolve()
        if (marker.get("context_index") != index or marker.get("sample_id") != sample
                or marker.get("questions") != expected_count or not raw_path.is_relative_to(context)):
            raise ValueError("Reused LangMem completion marker differs from its original context")
        rows = json.loads(pinned(raw_path).read_text(encoding="utf-8"))["data"]
        if (len(rows) != expected_count or any(
                row.get("eval_metadata", {}).get("sample_id") != sample for row in rows)):
            raise ValueError("Reused LangMem context has wrong QA count or identity")
        for row in rows:
            key = (sample, row["eval_metadata"].get("question_id"))
            if (type(row.get("query_id")) is not int or row["query_id"] != global_ids.get(key)
                    or type(row.get("context_id")) is not int or row["context_id"] != index):
                raise ValueError("Reused LangMem original global/context indices changed")
        records.extend(rows)
    normalized = normalize_records(records, selected, template)
    with tempfile.TemporaryDirectory(prefix="langmem-reuse-audit-") as scratch:
        path = Path(scratch) / "normalized.json"
        write_report(path, {"data": normalized})
        predictions = queue.canonical_predictions([path], selected)
    valid, issues, missing = validate_predictions(predictions, selected)
    if issues or missing or any(not row["prediction"].strip() for row in valid):
        raise ValueError("Reused LangMem predictions are incomplete or empty")
    # Include failed as well as successful requests for these conversations.
    # Unfinished conversations from the old attempt remain separate overhead.
    original_usage = read_jsonl(pinned(source / "usage.jsonl"))
    all_samples = {sample for sample, _ in expected}
    if any(row.get("run_id") != reuse["run_id"] or row.get("method") != "langmem"
           or row.get("sample_id") not in all_samples for row in original_usage):
        raise ValueError("Original LangMem journal contains unattributed or foreign requests")
    usage = [row for row in original_usage if row["sample_id"] in samples]
    queue.audit_coverage(usage, selected, reuse["run_id"], "langmem")
    summary = summarize_usage(usage, True)
    if not summary["complete"] or summary["failed_requests"] or summary["truncated_requests"]:
        raise ValueError("Reused LangMem usage fails completeness or quality audit")
    return {"complete": True, "questions": len(valid), "samples": samples, "run_id": reuse["run_id"],
            "usage": summary, "sources_sha256": hashes, "raw_artifacts_modified": False,
            "runtime_note": "Old attempt GPU time/cost includes unfinished conversations and is not "
                            "allocated proportionally to reused QA. New generation is a separate attempt."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    queue.validate_plan(plan)
    if args.output.exists():
        raise ValueError("Refusing to overwrite a prior reuse audit")
    report = audit_reuse(plan)
    write_report(args.output, report)
    print(json.dumps({"reused_questions": report["questions"], "original_run_id": report["run_id"]}), flush=True)


if __name__ == "__main__":
    main()
