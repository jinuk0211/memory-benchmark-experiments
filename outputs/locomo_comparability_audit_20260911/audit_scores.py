"""Offline LoCoMo F1, answer lengths, and paired conversation bootstrap."""
import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lightmem_analysis_deps"))
import numpy as np

DATASET_HASH = "cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993"
SCORER_HASH = "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
OUR = "s_parent_single_2000"
WRAPPER = ("Search Archival Memory and answer the question as concisely as you can, "
           "using a single phrase if possible.\n\n {question} \n\n Answer:")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_scorer(path: Path) -> Any:
    source = path.read_bytes()
    require(sha256(source) == SCORER_HASH, "Pinned official scorer hash mismatch")
    tree = ast.parse(source, filename=str(path))
    omitted = [node for node in tree.body
               if isinstance(node, ast.ImportFrom) and node.module == "bert_score"]
    require(len(omitted) == 1, "Expected exactly one unused bert_score import")
    tree.body = [node for node in tree.body if node not in omitted]
    namespace: dict[str, Any] = {"__name__": "official_f1_audit"}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["eval_question_answering"]


def parse_rows(archive: zipfile.ZipFile, entry: dict[str, Any],
               canonical: dict[tuple[str, int], dict[str, Any]]) -> tuple[list, dict]:
    found = {}
    hashes = {}
    method = entry["method"]
    for name in entry["paths"]:
        content = archive.read(name)
        hashes[name] = sha256(content)
        text = content.decode("utf-8-sig")
        if name.endswith(".jsonl"):
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif name.endswith(".csv"):
            rows = list(csv.DictReader(io.StringIO(text)))
        else:
            payload = json.loads(text)
            rows = payload if isinstance(payload, list) else payload.get("data", payload.get("results"))
        require(isinstance(rows, list), f"Unrecognized rows: {name}")
        light_keys = []
        if method == "LightMem_official":
            light_keys = [key for key in canonical if key[0] == payload["sample_id"]]
            require(len(rows) == len(light_keys), f"LightMem sample coverage: {name}")
        for position, row in enumerate(rows):
            if method == "LightMem_official":
                key = light_keys[position]
                question, gold, prediction = row["question"], row["reference"], row["prediction"]
            elif method == "E-Mem":
                cid, ordinal = row["qa_pair_id"].rsplit("_qa", 1)
                key = (cid, int(ordinal))
                require(key in canonical, f"Unknown E-Mem ID: {key}")
                question = canonical[key]["question"]
                require(row["query"] == WRAPPER.format(question=question), f"E-Mem query: {key}")
                require(row["sample_id"] == cid and row["question_id"] == row["qa_pair_id"], f"E-Mem ID: {key}")
                gold, prediction = row["answer"], row["parsed_output"]
            elif method in ("CertMem_v15", "Naive"):
                csv_keys = [key for key in canonical if key[0] == row["conv_id"]]
                ordinal = int(row["question_ordinal"])
                require(0 <= ordinal < len(csv_keys), f"CSV ordinal: {row['conv_id']} {ordinal}")
                key = csv_keys[ordinal]  # Ordinal counts category-filtered QA, unlike raw index.
                policy = "certified" if method == "CertMem_v15" else "full_raw"
                require(row["config"] == "full" and row["policy"] == policy, f"CSV policy: {key}")
                question, gold, prediction = row["question"], row["gold"], row["pred"]
            elif "conversation_id" in row:
                cid, ordinal = row["question_id"].rsplit(":", 1)
                key = (cid, int(ordinal))
                require(row["method"] == method and row["conversation_id"] == cid, f"Internal method/ID: {key}")
                require(row["prediction"] == row["hypothesis"], f"Internal answer alias: {key}")
                question, gold, prediction = row["question"], row["gold"], row["prediction"]
            else:
                key = (row["sample"], int(row["index"]))
                question, gold, prediction = row["question"], row["answer"], row["prediction"]
            require(key in canonical and key not in found, f"Unknown/duplicate {method} ID: {key}")
            qa = canonical[key]
            require(question == qa["question"] and str(gold) == str(qa["answer"])
                    and int(row["category"]) == qa["category"], f"Canonical metadata: {method} {key}")
            require(isinstance(prediction, str), f"Non-string prediction: {method} {key}")
            found[key] = {**qa, "prediction": prediction}
    require(found.keys() == canonical.keys(), f"Incomplete canonical coverage: {method}")
    return [found[key] for key in canonical], hashes


def main() -> None:
    dataset_path = ROOT / "certmem_primary_results_r1/canonical-locomo10.json"
    dataset_bytes = dataset_path.read_bytes()
    require(sha256(dataset_bytes) == DATASET_HASH, "Canonical dataset hash mismatch")
    dataset = json.loads(dataset_bytes)
    canonical = {(sample["sample_id"], index): qa
                 for sample in dataset for index, qa in enumerate(sample["qa"])
                 if qa["category"] in (1, 2, 3, 4)}
    require(len(canonical) == 1540 and len(dataset) == 10, "Wrong canonical population")
    score_fn = load_scorer(ROOT / "HiGMem/official_locomo_evaluation.py")
    export = ROOT / "outputs/baseline_drive_export_20260909"
    coverage_bytes = (export / "analysis_coverage.json").read_bytes()
    coverage = json.loads(coverage_bytes)
    archive_path = export / "Analysis_bundle.zip"
    categories = np.array([qa["category"] for qa in canonical.values()])
    conversations = [sample["sample_id"] for sample in dataset]
    conv_index = np.array([conversations.index(cid) for cid, _ in canonical])
    counts = np.bincount(conv_index, minlength=10)
    rng = np.random.default_rng(20260909)
    samples = rng.integers(0, 10, size=(20000, 10))
    sampled_counts = counts[samples].sum(axis=1)
    scores, methods = {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        for entry in coverage["methods"]:
            method = entry["method"]
            rows, hashes = parse_rows(archive, entry, canonical)
            with contextlib.redirect_stdout(io.StringIO()):
                values, _, _ = score_fn(rows, eval_key="prediction", metric="f1")
            values = np.array(values, dtype=float)
            require(values.shape == (1540,) and np.isfinite(values).all()
                    and ((values >= 0) & (values <= 1)).all(), f"Invalid F1: {method}")
            scores[method] = values
            words = np.array([len(row["prediction"].split()) for row in rows])
            methods[method] = {
                "n": len(rows), "unique_canonical_ids": len(rows), "metadata_validation": "PASS",
                "empty_predictions": int((words == 0).sum()), "f1_percent": float(values.mean() * 100),
                "categories": {str(cat): {"n": int((categories == cat).sum()),
                    "f1_percent": float(values[categories == cat].mean() * 100)} for cat in (1, 2, 3, 4)},
                "answer_words": {"definition": "Python str.split whitespace-separated words; not model tokens",
                    "mean": float(words.mean()), "median": float(np.median(words)),
                    "p95": float(np.percentile(words, 95)), "maximum": int(words.max()),
                    "over_30": int((words > 30).sum()), "over_30_percent": float((words > 30).mean() * 100)},
                "source_members_sha256": hashes,
            }
    require(len(methods) == 15, "Expected fifteen archived configurations")
    paired = {}
    for method, values in scores.items():
        if method == OUR:
            continue
        delta = (scores[OUR] - values) * 100
        sums = np.bincount(conv_index, weights=delta, minlength=10)
        replicates = sums[samples].sum(axis=1) / sampled_counts
        paired[method] = {
            "delta_f1_pp": float(delta.mean()),
            "conversation_bootstrap_95_interval_pp": np.percentile(replicates, [2.5, 97.5]).tolist(),
            "better": int((delta > 1e-12).sum()), "worse": int((delta < -1e-12).sum()),
            "tied": int((np.abs(delta) <= 1e-12).sum()),
            "categories": {str(cat): {"n": int((categories == cat).sum()),
                "delta_f1_pp": float(delta[categories == cat].mean()),
                "weighted_overall_contribution_pp": float(delta[categories == cat].sum() / 1540)}
                for cat in (1, 2, 3, 4)},
            "conversation_delta_f1_pp": {cid: float(sums[i] / counts[i]) for i, cid in enumerate(conversations)},
        }
    aggregate_path = ROOT / "results/comparisons/all_methods_20260909/ALL_CATEGORY_SCORES.json"
    aggregate_bytes = aggregate_path.read_bytes()
    archived_rows = {row["method"]: row for row in json.loads(aggregate_bytes)["rows"]
                     if row["population"] == "full1540"}
    require(archived_rows.keys() == methods.keys(), "Archived aggregate method coverage mismatch")
    differences = []
    for method, row in archived_rows.items():
        require(row["n"] == methods[method]["n"], f"Archived aggregate count: {method}")
        differences.append(abs(row["f1_percent"] - methods[method]["f1_percent"]))
        archived_categories = {str(item["category"]): item for item in row["categories"]}
        require(archived_categories.keys() == methods[method]["categories"].keys(), f"Archived categories: {method}")
        for cat, values in methods[method]["categories"].items():
            require(archived_categories[cat]["n"] == values["n"], f"Archived category count: {method}/{cat}")
            differences.append(abs(archived_categories[cat]["f1_percent"] - values["f1_percent"]))
    require(max(differences) < 1e-10, "Recomputed scores differ from archived aggregates")
    result = {
        "schema_version": 1, "model_requests": 0, "scored_question_method_pairs": 1540 * len(methods),
        "metric": "Pinned official LoCoMo category-specific F1, multiplied by 100",
        "evaluator_loading": "AST omitted only unused top-level from bert_score import score; all function bodies unchanged",
        "dataset_sha256": sha256(dataset_bytes), "scorer_sha256": SCORER_HASH,
        "archive_sha256": sha256(archive_path.read_bytes()), "coverage_sha256": sha256(coverage_bytes),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "bootstrap": {"unit": "conversation", "conversations": 10, "replicates": 20000,
            "seed": 20260909, "rng": "numpy.default_rng / PCG64",
            "weighting": "question-weighted with resampled conversation multiplicity",
            "limitations": "Descriptive conversation resampling; not independent-run uncertainty or correction for configuration selection."},
        "archived_aggregate_comparison": {"path": str(aggregate_path), "sha256": sha256(aggregate_bytes),
            "method_count": len(archived_rows), "overall_and_category_values_compared": len(differences),
            "maximum_absolute_difference_pp": max(differences), "tolerance_pp": 1e-10, "status": "PASS"},
        "methods": methods, "our_minus_comparators": paired,
    }
    output_path = Path(__file__).with_name("scores_audit.json")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for method, report in methods.items():
        print(f"{method}: {report['f1_percent']:.6f}; mean words {report['answer_words']['mean']:.3f}; >30 words {report['answer_words']['over_30']}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

