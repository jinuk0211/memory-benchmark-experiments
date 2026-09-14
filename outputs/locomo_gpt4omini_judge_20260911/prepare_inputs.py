"""Prepare the nine archived configurations for a method-blind common judge."""
import ast
import collections
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

ROOT = Path("D:/MemoryData")
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / "outputs/locomo_comparability_audit_20260911/audit_scores.py"
METHODS = ["Naive", "Mem0", "A-MEM", "LangMem", "SimpleMem",
           "LightMem_official", "HiGMem", "E-Mem", "s_parent_single_2000"]
LABELS = dict(zip(METHODS, ["Full-context reference", "Mem0", "A-MEM", "LangMem",
              "SimpleMem", "LightMem", "HiGMem", "E-Mem", "Our method"]))
spec = importlib.util.spec_from_file_location("audit", AUDIT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def main():
    dataset_path = ROOT / "certmem_primary_results_r1/canonical-locomo10.json"
    raw = dataset_path.read_bytes()
    audit.require(audit.sha256(raw) == audit.DATASET_HASH, "Dataset hash mismatch")
    dataset = json.loads(raw)
    canonical = {(sample["sample_id"], i): qa for sample in dataset
                 for i, qa in enumerate(sample["qa"]) if qa["category"] in (1, 2, 3, 4)}
    audit.require(len(canonical) == 1540, "Population mismatch")
    export = ROOT / "outputs/baseline_drive_export_20260909"
    coverage_path = export / "analysis_coverage.json"
    coverage = {x["method"]: x for x in json.loads(coverage_path.read_bytes())["methods"]}
    archive_path = export / "Analysis_bundle.zip"
    scorer = audit.load_scorer(ROOT / "HiGMem/official_locomo_evaluation.py")
    source_path = ROOT / "lightmem_official_20260909/upstream/experiments/locomo/llm_judge.py"
    source = source_path.read_bytes()
    prompt = next(ast.literal_eval(n.value) for n in ast.parse(source).body
                  if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "ACCURACY_PROMPT" for t in n.targets))
    (OUT / "official_accuracy_prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")
    mapped, members = {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        for method in METHODS:
            rows, hashes = audit.parse_rows(archive, coverage[method], canonical)
            with contextlib.redirect_stdout(io.StringIO()):
                values, _, _ = scorer(rows, eval_key="prediction", metric="f1")
            mapped[method] = (rows, values)
            members[method] = hashes
    rows = []
    for ordinal, (cid, index) in enumerate(canonical):
        for method in METHODS:
            source_row = mapped[method][0][ordinal]
            rows.append({
                "id": f"{method}/{cid}:{index}", "method": method,
                "conversation_id": cid, "qa_index": index,
                "category": source_row["category"], "question": source_row["question"],
                "gold_answer": str(source_row["answer"]),
                "generated_answer": source_row["prediction"],
                "official_f1": float(mapped[method][1][ordinal]),
            })
    serialized = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    input_path = OUT / "input.jsonl"
    input_path.write_text(serialized, encoding="utf-8", newline="\n")
    unique = {prompt.format(question=r["question"], gold_answer=r["gold_answer"],
                           generated_answer=r["generated_answer"]) for r in rows}
    manifest = {
        "protocol": "LightMem official-distribution LoCoMo GPT-4o-mini judge; separate from original LoCoMo F1",
        "official_source_url": "https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/experiments/locomo/llm_judge.py",
        "official_source_path": str(source_path),
        "official_source_sha256": audit.sha256(source),
        "prompt_sha256": audit.sha256(prompt.encode("utf-8")),
        "api_settings": {"model": "gpt-4o-mini", "temperature": 0.0,
                         "response_format": {"type": "json_object"}, "max_tokens": "omitted as upstream"},
        "method_labels": LABELS, "method_order": METHODS, "total_rows": len(rows),
        "questions_per_method": len(canonical), "unique_question_gold_prediction_prompts": len(unique),
        "identical_payload_policy": "Reuse a single valid judge result for an identical complete API payload; retain every logical row.",
        "ordering": "Canonical QA order, methods interleaved within each question",
        "category_counts_per_method": dict(collections.Counter(q["category"] for q in canonical.values())),
        "gold_preprocessing": "str(canonical answer); no category-specific F1 preprocessing for the judge",
        "source_predictions_modified": False, "dataset_sha256": audit.sha256(raw),
        "archive_sha256": audit.sha256(archive_path.read_bytes()),
        "coverage_sha256": audit.sha256(coverage_path.read_bytes()),
        "input_sha256": audit.sha256(input_path.read_bytes()),
        "loader_sha256": audit.sha256(AUDIT.read_bytes()), "source_members_sha256": members,
        "price_usd_per_million": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "price_source": "https://developers.openai.com/api/docs/models/gpt-4o-mini",
        "price_checked_date": "2026-09-11", "run_budget_usd": 5.0,
    }
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "unique_prompts": len(unique),
                      "category_counts": manifest["category_counts_per_method"],
                      "input_sha256": manifest["input_sha256"]}))
if __name__ == "__main__":
    main()
