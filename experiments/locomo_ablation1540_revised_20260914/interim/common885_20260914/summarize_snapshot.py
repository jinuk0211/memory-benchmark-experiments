"""Descriptive snapshot only: all16 arms on the same six completed histories."""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import report_results as report  # noqa: E402


def main() -> None:
    snapshot_path = HERE / "snapshot.json.gz"
    expected_sha = "847ec0ddab9918b9ba5a066147832aae8b20e0cbec733820680477b4c35d1843"
    if report.sha(snapshot_path) != expected_sha:
        raise ValueError("Snapshot fingerprint changed")
    with gzip.open(snapshot_path, "rt", encoding="utf-8") as stream:
        snapshot = json.load(stream)
    inputs, code, memories = ROOT / "package/input", ROOT / "package/code", ROOT / "package/run"
    manifest = report.read(inputs / "manifest.json")
    for name, expected in manifest["code"].items():
        if report.sha(code / name) != expected:
            raise ValueError("Frozen scorer/code changed")
    runner = report.load_runner(code)
    if snapshot["arms"] != list(report.ARMS) or tuple(runner.ARMS) != report.ARMS:
        raise ValueError("Incomplete arm population")
    dataset = ROOT.parent / "recursive_minilm_20260909/data/locomo10.json"
    if report.sha(dataset) != manifest["dataset_sha256"]:
        raise ValueError("Canonical dataset changed")
    canonical = []
    for sample in report.read(dataset):
        cid = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            if int(qa["category"]) in report.CATEGORIES:
                canonical.append({"id": f"{cid}:{index}", "conv_id": cid, "qa_index": index,
                                  "category": int(qa["category"]), "question": qa["question"],
                                  "gold": str(qa["answer"])})
    canonical.sort(key=lambda row: (row["conv_id"], row["qa_index"]))
    if len(canonical) != 1540 or canonical != report.read(ROOT / "evaluation/gold.json"):
        raise ValueError("Canonical full population differs")
    questions = [{key: value for key, value in row.items() if key != "gold"} for row in canonical]
    if questions != report.read(inputs / "reader_questions.json"):
        raise ValueError("Canonical questions differ")
    if report.sha(inputs / "reader_questions.json") != snapshot["canonical_questions_sha256"]:
        raise ValueError("Snapshot question source changed")
    histories = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44"]
    if snapshot["common_histories"] != histories or snapshot["questions_per_arm"] != 885:
        raise ValueError("Frozen interim common population changed")
    selected = [row for row in canonical if row["conv_id"] in histories]
    gold = {row["id"]: row for row in selected}
    tokenizer = runner.TokenRuntime(ROOT.parents[1] / "model_staging/qwen35_9b_c202236/tokenizer.json",
                                    manifest["tokenizer_sha256"])
    scored, summary = {}, {}
    for arm in report.ARMS:
        rows = snapshot["rows"][arm]
        if [row["id"] for row in rows] != [row["id"] for row in selected]:
            raise ValueError("Incomplete or reordered paired interim population")
        units = {}
        for cid in histories:
            path = memories / "memories" / arm / f"{cid}.json"
            lock = report.read(memories / "locks" / f"{cid}.json")
            if report.sha(path) != lock["memories"][arm] or lock["benchmark_qa_used"]:
                raise ValueError("Memory fingerprint differs")
            units[cid] = report.read(path)
        scored[arm] = []
        for row in rows:
            reference = gold[row["id"]]
            if any(row[key] != reference[key] for key in ("id", "conv_id", "qa_index", "category", "question")):
                raise ValueError("Question metadata changed")
            context = "\n\n".join(units[row["conv_id"]][i]["text"] for i in row["memory_indices"])
            report.validate_context({**row, "context": context}, units[row["conv_id"]], tokenizer.ntok, runner.core.digest)
            user = report.reader_request(context, row["question"])
            key = runner.core.digest([snapshot["protocol"]["environment"]["models"]["Qwen/Qwen3.5-9B"],
                                      20260907, snapshot["protocol"]["reader_system"], user, 96, False])
            report.validate_receipt(row, row["native_cache"], key)
            scored[arm].append({"id": row["id"], "conv_id": row["conv_id"], "category": row["category"],
                                "prediction": row["prediction"],
                                "official_f1": runner.core.f1(row["prediction"], reference["gold"], row["category"])})
        summary[arm] = {"f1": 100 * statistics.mean(row["official_f1"] for row in scored[arm]),
                        "read_tokens": statistics.mean(row["read_tokens"] for row in rows),
                        "categories": {name: 100 * statistics.mean(row["official_f1"] for row in scored[arm]
                                                                 if row["category"] == category)
                                       for category, name in report.CATEGORIES.items()}}
    random_per_question = [statistics.mean(scored[arm][i]["official_f1"] for arm in report.RANDOM_ARMS)
                           for i in range(len(selected))]
    seed_f1 = [summary[arm]["f1"] for arm in report.RANDOM_ARMS]
    summary["random_mean"] = {"f1": 100 * statistics.mean(random_per_question),
                              "seed_f1_sd": statistics.stdev(seed_f1),
                              "seed_f1_min": min(seed_f1), "seed_f1_max": max(seed_f1)}
    for value in summary.values():
        value["control_minus_ours_pp"] = value["f1"] - summary["ours"]["f1"]
    result = {"status": "INTERIM_DESCRIPTIVE_NOT_FINAL", "questions": len(selected), "planned_questions": 1540,
              "common_histories": histories, "observed_at_utc": snapshot["observed_at_utc"],
              "snapshot_sha256": expected_sha, "script_sha256": report.sha(Path(__file__)),
              "category_counts": dict(Counter(row["category"] for row in selected)),
              "summary": summary, "rows": scored,
              "statistical_tests_performed": False, "confidence_intervals_computed": False,
              "limits": "Only six common completed histories; changing interim sample. No significance or final-full1540 claim.",
              "validation": "Same canonical885 IDs in16 arms; pinned official F1; exact locked context/token reconstruction and native request/output/cache records."}
    report.write_json(HERE / "INTERIM_RESULTS.json", result)
    lines = ["# LoCoMo 중간 결과 — 공통 완료 885문항", "",
             "최종 1,540문항 결과가 아니다. 모든16개 설정이 공통으로 완료한6개 대화의885문항만 비교했다.",
             "현재 1차 실험이며, source marginal utility를 수정한 후속 실험의 결과가 아니다.",
             "중간 결과에 대한 신뢰구간이나 p-value를 계산하지 않았으며 유의성 판단에 사용하지 않는다.", "",
             "| 설정 | F1 (%) | 설정 − Ours (pp) |", "|---|---:|---:|"]
    for arm in report.TABLE_ARMS:
        value = summary[arm]
        lines.append(f'| {report.LABELS[arm]} | {value["f1"]:.4f} | {value["control_minus_ours_pp"]:+.4f} |')
    random = summary["random_mean"]
    lines += ["", f'Random10 seed F1 범위: {random["seed_f1_min"]:.4f}–{random["seed_f1_max"]:.4f}; '
              f'seed 간 표본 표준편차: {random["seed_f1_sd"]:.4f}pp.', "",
              "표본은 점수와 무관하게 snapshot 시점에 모든 설정의 예측 파일이 완료된 대화의 교집합으로 선택했다.",
              "질문·정답은 canonical dataset에서, F1은 최종 reporter와 같은 고정 scorer에서 다시 계산했다.", ""]
    (HERE / "INTERIM_RESULTS_KO.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
