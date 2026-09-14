"""Offline official LoCoMo F1 alongside, not instead of, native CertMem scores."""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

DATASET_SHA256 = "cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993"
SCORER_SHA256 = "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
POLICIES = {
    (config, policy)
    for config in ("full", "no_adaptive", "no_residual")
    for policy in (["units_only", "certified"]
                   + [f"{kind}_{budget}" for kind in ("uniform", "ours", "certified")
                      for budget in (400, 1600, 4000)]
                   + (["full_raw", "raw_rag_400", "raw_rag_1600", "raw_rag_4000"]
                      if config == "full" else []))
}


def build_report(dataset: list, items: list, score_fn) -> dict:
    """Validate all 37 policy groups before scoring any of them."""
    expected = {}
    for sample in dataset:
        cid = sample["sample_id"]
        if cid in expected:
            raise ValueError("Duplicate dataset conversation")
        expected[cid] = [(index, qa) for index, qa in enumerate(sample["qa"])
                         if qa["category"] in (1, 2, 3, 4)]
    identities = {(cid, ordinal) for cid, qas in expected.items() for ordinal in range(len(qas))}
    grouped = defaultdict(dict)
    for row in items:
        group = (row["config"], row["policy"])
        cid, ordinal = row["conv_id"], int(row["question_ordinal"])
        key = (cid, ordinal)
        if group not in POLICIES or key not in identities or key in grouped[group]:
            raise ValueError("Unexpected or duplicate prediction identity")
        index, qa = expected[cid][ordinal]
        if (row["question"] != qa["question"] or row["gold"] != str(qa["answer"])
                or int(row["category"]) != qa["category"] or not isinstance(row["pred"], str)):
            raise ValueError("Prediction metadata differs from original QA")
        native_f1, native_lenient = float(row["f1"]), float(row["lenient"])
        if not math.isfinite(native_f1) or not 0 <= native_f1 <= 1 or native_lenient not in (0, 1):
            raise ValueError("Invalid native score")
        record = {**qa, "sample": cid, "index": index, "prediction": row["pred"]}
        grouped[group][key] = (record, native_f1, native_lenient)
    if set(grouped) != POLICIES or any(set(rows) != identities for rows in grouped.values()):
        raise ValueError("Incomplete config/policy/question coverage")
    results = []
    empty = 0
    for (config, policy), rows in sorted(grouped.items()):
        records = [rows[key][0] for key in sorted(identities)]
        scores = [float(value) for value in score_fn(records)]
        if len(scores) != len(records) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("Invalid official scorer output")
        empty_count = sum(not record["prediction"].strip() for record in records)
        empty += empty_count
        results.append({"config": config, "policy": policy, "qa_count": len(records),
                        "official_f1": statistics.mean(scores),
                        "native_normalized_f1": statistics.mean(row[1] for row in rows.values()),
                        "native_lenient": statistics.mean(row[2] for row in rows.values()),
                        "empty_predictions": empty_count})
    return {"schema_version": 1, "predictions_complete": True,
            "qa_per_policy": len(identities), "reader_outputs": len(items),
            "empty_predictions": empty, "by_policy": results,
            "note": "Official F1 uses the unchanged original QA metadata. Native normalized F1 is a different metric. "
                    "This report validates prediction coverage only; token/runtime completeness is audited separately. "
                    "Native empty or output-capped answers remain in the denominator."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing score report")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Unexpected dataset hash")
    if hashlib.sha256(args.scorer.read_bytes()).hexdigest() != SCORER_SHA256:
        raise ValueError("Unexpected official scorer hash")
    spec = importlib.util.spec_from_file_location("locomo_official_certmem", args.scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def score(records):
        with contextlib.redirect_stdout(io.StringIO()):
            values, _, _ = module.eval_question_answering(records, eval_key="prediction", metric="f1")
        return values

    with args.items.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    report = build_report(json.loads(args.dataset.read_bytes()), rows, score)
    report["sources_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in (("dataset", args.dataset), ("items", args.items), ("scorer", args.scorer))}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"predictions_complete": True, "qa_per_policy": report["qa_per_policy"],
                      "reader_outputs": report["reader_outputs"]}))


if __name__ == "__main__":
    main()
