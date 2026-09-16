"""Validate and report the predeclared full-LoCoMo component comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
RANDOM_ARMS = tuple(f"random_{i:02d}" for i in range(10))
ARMS = ("ours", "no_cues", "no_audit", "no_temporal", "with_binding", "payload_keys") + RANDOM_ARMS
TABLE_ARMS = ("ours", "no_cues", "no_audit", "no_temporal", "with_binding", "random_mean", "payload_keys")
LABELS = {"ours": "Ours", "no_cues": "w/o cues (base memory)",
          "no_audit": "w/o omission audit", "no_temporal": "w/o temporal normalization",
          "with_binding": "+ evidence binding", "payload_keys": "Payload retrieval keys",
          "random_mean": "Random cue selection (10 seeds)"}
CATEGORIES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop"}
ACCOUNT_FIELDS = ("parent_payload_tokens", "cue_payload_tokens", "distinct_key_tokens", "total_stored_tokens")
STAT_SEED = 20260914


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def paired_differences(left: list[dict], right: list[dict]) -> tuple[list[str], np.ndarray]:
    """Align question IDs, rejecting duplicate or mismatched populations."""
    lhs, rhs = {r["id"]: r for r in left}, {r["id"]: r for r in right}
    if len(lhs) != len(left) or len(rhs) != len(right) or lhs.keys() != rhs.keys() or not lhs:
        raise ValueError("Paired populations differ or contain duplicate IDs")
    ids = sorted(lhs)
    for qid in ids:
        if any(lhs[qid][key] != rhs[qid][key] for key in ("conv_id", "category")):
            raise ValueError("Paired question identity changed")
    return [lhs[qid]["conv_id"] for qid in ids], np.array([
        100 * (lhs[qid]["official_f1"] - rhs[qid]["official_f1"]) for qid in ids])


def paired_statistics(left: list[dict], right: list[dict], *, draws: int = 5000,
                      seed: int = STAT_SEED) -> dict:
    """Question-weighted cluster bootstrap and exact cluster sign flips."""
    cids, differences = paired_differences(left, right)
    histories = sorted(set(cids))
    totals = np.array([sum(d for cid, d in zip(cids, differences) if cid == name) for name in histories])
    counts = np.array([cids.count(name) for name in histories])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(histories), size=(draws, len(histories)))
    boot = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    observed = float(differences.mean())
    masks = np.arange(2 ** len(histories), dtype=np.uint64)[:, None]
    bits = (masks >> np.arange(len(histories), dtype=np.uint64)) & 1
    signs = 2 * bits.astype(np.int8) - 1
    permutations = (signs @ totals) / len(differences)
    p_value = float(np.mean(np.abs(permutations) >= abs(observed) - 1e-12))
    return {"delta_pp": observed, "paired_95ci_pp": np.quantile(boot, [.025, .975]).tolist(),
            "sign_flip_p": p_value, "permutations": len(permutations),
            "histories": len(histories), "questions": len(differences),
            "bootstrap_draws": draws, "bootstrap_seed": seed}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    adjusted, running = {}, 0.0
    ordered = sorted(p_values, key=p_values.get)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def summarize(rows: list[dict], accounting: dict[str, dict]) -> dict:
    storage: dict[str, float] = {}
    for row in rows:
        prior = storage.setdefault(row["conv_id"], row["stored_tokens"])
        if prior != row["stored_tokens"]:
            raise ValueError("Within-history storage changed")
    if set(accounting) != set(storage):
        raise ValueError("Accounting population differs")
    for cid, fields in accounting.items():
        if sum(fields[key] for key in ACCOUNT_FIELDS[:3]) != fields["total_stored_tokens"]:
            raise ValueError("Storage decomposition does not sum")
        if fields["total_stored_tokens"] != storage[cid]:
            raise ValueError("Accounting differs from prediction storage")
    tokens = np.array([r["read_tokens"] for r in rows])
    return {"n": len(rows), "histories": len(storage),
            "f1": statistics.mean(r["official_f1"] for r in rows) * 100,
            "category_f1": {str(c): statistics.mean(r["official_f1"] for r in rows if r["category"] == c) * 100
                            for c in sorted({r["category"] for r in rows})},
            "stored_tokens": statistics.mean(storage.values()), "read_tokens": float(tokens.mean()),
            "read_token_quantiles": dict(zip(("min", "p25", "median", "p75", "max"),
                                             np.quantile(tokens, [0, .25, .5, .75, 1]).tolist())),
            "storage_breakdown": {key: statistics.mean(a[key] for a in accounting.values()) for key in ACCOUNT_FIELDS},
            "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
            "empty_answers": sum(not r["prediction"].strip() for r in rows)}


def random_mean_rows(scores: dict[str, list[dict]]) -> list[dict]:
    reference = scores[RANDOM_ARMS[0]]
    for arm in RANDOM_ARMS[1:]:
        paired_differences(reference, scores[arm])
    lookup = {arm: {r["id"]: r for r in scores[arm]} for arm in RANDOM_ARMS}
    return [{**{key: row[key] for key in ("id", "conv_id", "category", "qa_index", "question")},
             "official_f1": statistics.mean(lookup[arm][row["id"]]["official_f1"] for arm in RANDOM_ARMS)}
            for row in reference]


def aggregate(scores: dict[str, list[dict]], accounting: dict[str, dict]) -> tuple[dict, dict, list[dict]]:
    summary = {arm: {"label": LABELS.get(arm, arm), **summarize(scores[arm], accounting[arm])} for arm in ARMS}
    random_rows = random_mean_rows(scores)
    random_results = [summary[arm] for arm in RANDOM_ARMS]
    random_summary = {"label": LABELS["random_mean"], "n": len(random_rows),
                      "histories": random_results[0]["histories"], "seeds": len(RANDOM_ARMS),
                      "f1": statistics.mean(r["f1"] for r in random_results),
                      "seed_f1_sd": statistics.stdev(r["f1"] for r in random_results),
                      "stored_tokens": statistics.mean(r["stored_tokens"] for r in random_results),
                      "read_tokens": statistics.mean(r["read_tokens"] for r in random_results),
                      "category_f1": {str(c): statistics.mean(r["category_f1"][str(c)] for r in random_results)
                                      for c in CATEGORIES},
                      "storage_breakdown": {key: statistics.mean(r["storage_breakdown"][key] for r in random_results)
                                            for key in ACCOUNT_FIELDS},
                      "individual_seed_results": {arm: summary[arm] for arm in RANDOM_ARMS}}
    summary["random_mean"] = random_summary
    comparisons = {"ours_minus_" + arm: paired_statistics(scores["ours"], random_rows if arm == "random_mean" else scores[arm])
                   for arm in ("no_cues", "random_mean", "payload_keys")}
    adjusted = holm_adjust({name: row["sign_flip_p"] for name, row in comparisons.items()})
    for name, row in comparisons.items():
        row["holm_p"] = adjusted[name]
    for arm in (*ARMS, "random_mean"):
        summary[arm].update(paired_statistics(random_rows if arm == "random_mean" else scores[arm], scores["ours"]))
    return summary, comparisons, random_rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tex_table(caption: str, label: str, columns: str, header: str, rows: list[str]) -> str:
    return "\n".join([r"\begin{table}[t]", r"\centering\small", r"\caption{" + caption + "}",
                      r"\label{" + label + "}", r"\setlength{\tabcolsep}{4pt}",
                      r"\renewcommand{\arraystretch}{1.10}", r"\resizebox{0.92\linewidth}{!}{%",
                      r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule", header + r" \\",
                      r"\midrule", *[row + r" \\" for row in rows], r"\bottomrule",
                      r"\end{tabular}%", "}", r"\end{table}", ""])


def export(out: Path, scores: dict, accounting: dict, protocol: dict,
           validation: dict, selection: dict) -> None:
    summary, primary, random_rows = aggregate(scores, accounting)
    out.mkdir(parents=True, exist_ok=True)
    interpretation = ("All 1,540 questions and ten histories were previously exposed. "
                      "This is an exploratory follow-up after a finite predeclared reader selection "
                      "on the previously used 300-question subset, not held-out evaluation. "
                      "Random controls retain source-utility admission; only selection is randomized.")
    write_json(out / "RESULTS.json", {"results": summary, "primary_comparisons": primary,
                                      "reader_selection": selection, "protocol": protocol,
                                      "validation": validation, "interpretation": interpretation})
    write_json(out / "scored_predictions.json", scores)
    write_json(out / "VALIDATION.json", validation)
    table_rows = [{"arm": arm, **summary[arm]} for arm in TABLE_ARMS]
    write_csv(out / "table.csv", ["arm", "label", "f1", "delta_pp", "stored_tokens", "read_tokens"], table_rows)
    rows_tex, rows_md = [], ["| Variant | F1 (%) | Delta (pp) | Stored tokens | Read tokens |",
                            "|---|---:|---:|---:|---:|"]
    for arm in TABLE_ARMS:
        row = summary[arm]
        label = r"\textbf{Ours}" if arm == "ours" else row["label"]
        rows_tex.append(f'{label} & {row["f1"]:.4f} & {row["delta_pp"]:+.4f} & '
                        f'{row["stored_tokens"]:,.1f} & {row["read_tokens"]:.1f}')
        rows_md.append(f'| {row["label"]} | {row["f1"]:.4f} | {row["delta_pp"]:+.4f} | '
                       f'{row["stored_tokens"]:,.1f} | {row["read_tokens"]:.1f} |')
    caption = (r"Component ablations on all 1,540 LoCoMo questions with Qwen3.5-9B. "
               r"Ours uses separate dialogue blocks and facts (without evidence binding). "
               r"Differences are relative to Ours. Stored tokens are averaged over ten histories; "
               r"read tokens over 1,540 questions. Random selection averages ten predeclared seeds. "
               r"All variants use a 2,048-token evidence limit. All questions were previously exposed; "
               r"the reader was selected from two predeclared settings on a previously used 300-question subset.")
    (out / "table.tex").write_text(tex_table(caption, "tab:exp-ablation1540-revised", "lrrrr",
        r"\textbf{Variant} & \textbf{F1 (\%)} & $\boldsymbol{\Delta}$\textbf{ (pp)} & \textbf{Stored tokens} & \textbf{Read tokens}",
        rows_tex), encoding="utf-8")
    category_rows = [{"arm": arm, "label": summary[arm]["label"],
                      **{name: summary[arm]["category_f1"][str(c)] for c, name in CATEGORIES.items()},
                      "Overall": summary[arm]["f1"]} for arm in TABLE_ARMS]
    write_csv(out / "category_table.csv", ["arm", "label", *CATEGORIES.values(), "Overall"], category_rows)
    (out / "category_table.tex").write_text(tex_table(
        r"Category F1 (\%) on the same 1,540 previously exposed LoCoMo questions. Random values average ten seeds.",
        "tab:exp-ablation1540-categories", "lrrrrr", "Variant & " + " & ".join([*CATEGORIES.values(), "Overall"]),
        [row["label"] + " & " + " & ".join(f'{row[name]:.4f}' for name in [*CATEGORIES.values(), "Overall"])
         for row in category_rows]), encoding="utf-8")
    stat_rows = [{"contrast": name, **stats, "ci_lower_pp": stats["paired_95ci_pp"][0],
                  "ci_upper_pp": stats["paired_95ci_pp"][1]} for name, stats in primary.items()]
    write_csv(out / "primary_statistics.csv", ["contrast", "delta_pp", "ci_lower_pp", "ci_upper_pp", "sign_flip_p", "holm_p"], stat_rows)
    (out / "primary_statistics.tex").write_text(tex_table(
        r"Predeclared Ours-minus-control contrasts. Intervals use 5,000 paired conversation-cluster bootstrap draws. "
        r"Two-sided sign-flip sensitivity tests enumerate all 1,024 conversation sign assignments; "
        r"Holm correction covers the three contrasts. Random scores are first averaged across ten seeds.",
        "tab:exp-ablation1540-statistics", "lrrrr", r"Contrast & $\Delta$ (pp) & 95\% CI (pp) & $p$ & $p_{\mathrm{Holm}}$",
        [f'Ours $-$ {LABELS[row["contrast"].removeprefix("ours_minus_")]} & {row["delta_pp"]:+.4f} & '
         f'[{row["ci_lower_pp"]:+.4f}, {row["ci_upper_pp"]:+.4f}] & {row["sign_flip_p"]:.6f} & {row["holm_p"]:.6f}'
         for row in stat_rows]), encoding="utf-8")
    write_csv(out / "per_question_scores.csv", ["arm", "id", "conv_id", "qa_index", "category", "question", "gold",
                                               "prediction", "official_f1", "stored_tokens", "read_tokens", "finish_reason",
                                               "input_tokens", "output_tokens", "generation_cache_key"],
              [row for arm in ARMS for row in scores[arm]])
    random_lookup = {row["id"]: row["official_f1"] for row in random_rows}
    comparisons = [{key: row[key] for key in ("id", "conv_id", "category", "question", "gold")} for row in scores["ours"]]
    score_lookup = {arm: {row["id"]: row for row in scores[arm]} for arm in ARMS}
    for row in comparisons:
        for arm in ARMS:
            row[arm + "_f1"] = score_lookup[arm][row["id"]]["official_f1"] * 100
        row["random_mean_f1"] = random_lookup[row["id"]] * 100
    write_csv(out / "per_question_comparison.csv", ["id", "conv_id", "category", "question", "gold",
                                                   *[arm + "_f1" for arm in ARMS], "random_mean_f1"], comparisons)
    write_csv(out / "storage_breakdown.csv", ["arm", "conv_id", *ACCOUNT_FIELDS, "parent_unit_count", "cue_unit_count", "parent_sha256"],
              [{"arm": arm, "conv_id": cid, **account} for arm in ARMS for cid, account in accounting[arm].items()])
    write_csv(out / "random_seed_results.csv", ["arm", "seed", "f1", "delta_pp", "stored_tokens", "read_tokens"],
              [{"arm": arm, "seed": 2026091400 + i, **summary[arm]} for i, arm in enumerate(RANDOM_ARMS)])
    disclosure = json.dumps(selection, ensure_ascii=False, indent=2)
    lines = ["LoCoMo 전체 1,540문항 ablation 결과. Ours는 evidence binding을 제거한 구성이다.", "",
             *rows_md, "", "저장 토큰은 10개 대화에 동일 가중치, 읽기 토큰과 F1은 1,540문항에 동일 가중치로 집계했다.",
             "각 제거 조건은 같은 Ours에서 구성했다. Random은 양의 utility를 포함한 기존 admission을 유지한 후보 내 선택 대조다.",
             "", f'Random 10개 seed의 평균 F1은 {summary["random_mean"]["f1"]:.4f}, seed 간 표본 표준편차는 '
             f'{summary["random_mean"]["seed_f1_sd"]:.4f}pp다. 전체 seed 결과는 random_seed_results.csv에 있다.', "",
             "| 사전 지정 비교 | Ours − control (pp) | 대화 bootstrap 95% CI | sign-flip p | Holm p |",
             "|---|---:|---:|---:|---:|"]
    for row in stat_rows:
        lines.append(f'| {row["contrast"]} | {row["delta_pp"]:+.4f} | [{row["ci_lower_pp"]:+.4f}, {row["ci_upper_pp"]:+.4f}] | '
                     f'{row["sign_flip_p"]:.6f} | {row["holm_p"]:.6f} |')
    lines += ["", "대화 단위 부호 교환 검정은 교환가능성/대칭성 가정 아래의 민감도 분석이다. Random 비교의 CI는 고정한 10개 seed 평균을 대상으로 한다.",
              "모든 1,540문항과 10개 대화가 이전에 노출되었다. 미노출 평가가 아닌 후속 탐색 실험이며, 300문항의 사전 지정 후보 선택을 포함한다.",
              "", "Reader 선택 기록:", "```json", disclosure, "```", "",
              "유형별 5개 F1은 category_table.csv/.tex, 문항별 원점수는 per_question_scores.csv, paired 점수는 per_question_comparison.csv에 있다.",
              "scored_predictions.json에는 답변·문맥·원본 영수증 식별자를 보존했다. 저장량 분해는 storage_breakdown.csv, 검증은 VALIDATION.json에 있다."]
    (out / "RESULTS_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(rows_md))




def reader_request(context: str, question: str) -> str:
    return f"Conversation memory:\n{context}\n\nQuestion: {question}\nAnswer:"


def validate_context(row: dict, units: list[dict], ntok: Any, digest: Any) -> str:
    indices = row["memory_indices"]
    if (any(type(i) is not int or not 0 <= i < len(units) for i in indices)
            or len(indices) != len(set(indices)) or len(indices) > 120):
        raise ValueError("Invalid packed memory indices")
    context = "\n\n".join(units[i]["text"] for i in indices)
    if context != row["context"] or digest(context) != row["context_sha256"]:
        raise ValueError("Context differs from the locked selected payloads")
    if (type(row["read_tokens"]) is not int or not 0 <= row["read_tokens"] <= 2048
            or ntok(context) != row["read_tokens"]):
        raise ValueError("Independent reader-token recount or evidence cap failed")
    return context


def validate_receipt(row: dict, native: dict, expected_key: str) -> None:
    if row["generation_cache_key"] != expected_key:
        raise ValueError("Reader cache key differs from the frozen request")
    if not isinstance(row["prediction"], str) or native["text"] != row["prediction"] or any(
            native[field] != row[field] for field in ("finish_reason", "input_tokens", "output_tokens")):
        raise ValueError("Native reader receipt mismatch")
    if (row["finish_reason"] not in {"stop", "length"}
            or type(row["input_tokens"]) is not int or type(native["input_tokens"]) is not int
            or type(row["output_tokens"]) is not int or type(native["output_tokens"]) is not int
            or row["input_tokens"] <= 0 or row["input_tokens"] + 96 > 8192
            or not 0 <= row["output_tokens"] <= 96):
        raise ValueError("Native stop/token receipt gate failed")


def validate_pilot(args: argparse.Namespace, runner: Any, gold: dict,
                   tokenizer: Any, selection: dict) -> dict:
    contract_path = args.pilot / "protocol.json"
    contract = read(contract_path)
    contexts_path = args.pilot_inputs / "contexts.json"
    prompts = read(args.pilot_inputs / "reader_prompts.json")
    code = args.inputs.parent / "code"
    if (contract["population"] != "previously_exposed_dev300"
            or contract["context_sha256"] != sha(contexts_path)
            or contract["prompts"] != prompts or prompts != runner.READERS
            or contract["environment"] != read(code / "environment.json")
            or contract["seed"] != 20260907 or contract["output_cap"] != 96
            or contract["script_sha256"] != sha(code / "reader_pilot.py")
            or contract["selection_rule"] != "Highest dev300 Ours overall official F1; ties prefer legacy"):
        raise ValueError("Pilot contract differs from the predeclared inputs/code/readers")
    expected_runtime = {p.relative_to(code).as_posix(): sha(p) for p in (code / "vendor").rglob("*.py")}
    if contract["runtime_hashes"] != expected_runtime:
        raise ValueError("Pilot runtime fingerprint mismatch")
    contexts = read(contexts_path)
    expected_ids = {row["id"] for row in read(args.inputs / "dev300_questions.json")}
    if len(contexts) != 300 or len({row["id"] for row in contexts}) != 300 or {
            row["id"] for row in contexts} != expected_ids:
        raise ValueError("Pilot does not cover the exact frozen development300 population")
    for row in contexts:
        if ("gold" in row or "answer" in row or any(row[k] != gold[row["id"]][k]
                for k in ("question", "conv_id", "category"))
                or runner.core.digest(row["context"]) != row["context_sha256"]
                or type(row["read_tokens"]) is not int or not 0 <= row["read_tokens"] <= 2048
                or tokenizer.ntok(row["context"]) != row["read_tokens"]):
            raise ValueError("Frozen pilot question/context mismatch")
    checks = {}
    for reader in ("legacy", "grounded"):
        path = args.pilot / f"{reader}.json"
        if (selection["dev_protocol_sha256"][reader] != sha(contract_path)
                or selection["dev_prediction_sha256"][reader] != sha(path)):
            raise ValueError("Reader selection record does not fingerprint the pilot artifacts")
        rows = read(path)
        if [row["id"] for row in rows] != [row["id"] for row in contexts]:
            raise ValueError("Incomplete, duplicated or reordered pilot responses")
        values = []
        for row, original in zip(rows, contexts):
            if row["reader"] != reader or any(row[k] != v for k, v in original.items()):
                raise ValueError("Pilot answer differs from its frozen question/context")
            prompt = reader_request(row["context"], row["question"])
            key = runner.core.digest([contract["environment"]["models"]["Qwen/Qwen3.5-9B"],
                                      20260907, prompts[reader], prompt, 96, False])
            native = read(args.pilot / "runtime/cache/generations" / f"{key}.json")
            validate_receipt(row, native, key)
            values.append(runner.core.f1(row["prediction"], gold[row["id"]]["gold"], row["category"]))
        measured = statistics.mean(values) * 100
        if abs(measured - selection["results"][reader]["f1"]) > 1e-10:
            raise ValueError("Reader selection summary differs from independently scored pilot answers")
        checks[reader] = {"n": len(values), "f1": measured, "prediction_sha256": sha(path)}
    return {"protocol_sha256": sha(contract_path), "results": checks,
            "native_receipts_verified": 600, "frozen_before_full1540": "declared in selection record"}


def load_runner(code: Path) -> Any:
    sys.path.insert(0, str(ROOT / "report_deps"))
    spec = importlib.util.spec_from_file_location("ablation_report_runner", code / "run_revised.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_selection(selection: dict, reader: str) -> None:
    if (selection["candidate_configurations"] != ["legacy", "grounded"]
            or selection["population"] != "dev300" or selection["arms"] != ["ours"]
            or selection["criterion"] != "highest_overall_f1_tie_legacy"
            or selection["frozen_before_full1540"] is not True):
        raise ValueError("Reader selection differs from the predeclared finite protocol")
    results = selection["results"]
    if set(results) != {"legacy", "grounded"} or any(
            row["n"] != 300 or not 0 <= row["f1"] <= 100 for row in results.values()):
        raise ValueError("Incomplete reader development results")
    winner = "grounded" if results["grounded"]["f1"] > results["legacy"]["f1"] else "legacy"
    if selection["chosen_reader"] != winner or reader != winner:
        raise ValueError("Chosen reader violates the predeclared rule or full-run protocol")


def validate_run(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    inputs, memory_root, run = args.inputs, args.memory_root, args.run
    code = inputs.parent / "code"
    manifest = read(inputs / "manifest.json")
    for name, digest in manifest["files"].items():
        if sha(inputs / name) != digest:
            raise ValueError(f"Frozen input fingerprint mismatch: {name}")
    for name, digest in manifest["code"].items():
        if sha(code / name) != digest:
            raise ValueError(f"Frozen code fingerprint mismatch: {name}")
    runner = load_runner(code)
    if (tuple(runner.ARMS) != ARMS or manifest["arms"] != list(ARMS)
            or runner.RANDOM_SEEDS != {arm: 2026091400 + i for i, arm in enumerate(RANDOM_ARMS)}):
        raise ValueError("The full report requires all six deterministic and ten random arms")
    protocol = read(run / "protocol.json")
    expected = runner.evaluation_contract(argparse.Namespace(
        inputs=inputs, memory_root=memory_root, out=run, arms=list(ARMS),
        reader=protocol["reader"], population="full1540", shards=protocol["shards"]))
    if protocol != expected or protocol["question_count"] != 1540:
        raise ValueError("Downloaded evaluation contract differs from frozen full1540 inputs")
    expected_prepare = {"stage": "prepare", "manifest": manifest, "arms": list(ARMS),
                        "random_seeds": runner.RANDOM_SEEDS, "read_budget": 2048, "extra_budget": 2000}
    if read(memory_root / "protocol.json") != expected_prepare:
        raise ValueError("Memory preparation contract differs")
    selection = read(args.selection)
    validate_selection(selection, protocol["reader"])
    dataset_path = ROOT.parent / "recursive_minilm_20260909/data/locomo10.json"
    if sha(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Canonical dataset fingerprint changed")
    canonical = []
    for sample in read(dataset_path):
        cid = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            if int(qa["category"]) in CATEGORIES:
                canonical.append({"id": f"{cid}:{index}", "conv_id": cid, "qa_index": index,
                                  "category": int(qa["category"]), "question": qa["question"],
                                  "gold": str(qa["answer"])})
    canonical.sort(key=lambda row: (row["conv_id"], row["qa_index"]))
    if read(ROOT / "evaluation/gold.json") != canonical:
        raise ValueError("Local gold differs from canonical full1540 labels")
    questions = read(inputs / "reader_questions.json")
    if questions != [{key: value for key, value in row.items() if key != "gold"} for row in canonical]:
        raise ValueError("Reader questions differ from canonical full1540 population")
    selected = [{key: row[key] for key in ("id", "conv_id", "qa_index", "category")} for row in canonical]
    if read(inputs / "selection.json")["records"] != selected:
        raise ValueError("Selection identities differ from canonical full1540")
    gold = {row["id"]: row for row in canonical}
    histories = sorted({row["conv_id"] for row in canonical})
    if len(gold) != 1540 or len(histories) != 10:
        raise ValueError("Expected 1,540 unique questions in ten histories")
    tokenizer = runner.TokenRuntime(args.tokenizer, manifest["tokenizer_sha256"])
    pilot_validation = validate_pilot(args, runner, gold, tokenizer, selection)
    sources = read(inputs / "source_sessions.json")
    locks = {cid: read(memory_root / "locks" / f"{cid}.json") for cid in histories}
    scores, accounts, prediction_hashes, native_hashes = {}, {}, {}, {}
    for arm in ARMS:
        rows, accounts[arm] = [], {}
        for cid in histories:
            lock = locks[cid]
            memory_path = memory_root / "memories" / arm / f"{cid}.json"
            units = read(memory_path)
            if sha(memory_path) != lock["memories"][arm] or lock["benchmark_qa_used"]:
                raise ValueError("Memory fingerprint or source-only lock mismatch")
            if {sid for unit in units for sid in unit["sources"]} != {
                    turn["id"] for session in sources[cid] for turn in session["turns"]}:
                raise ValueError("Memory source coverage differs")
            account = lock["accounting"][arm]
            parent = units[:account["parent_unit_count"]]
            if runner.accounting(units, parent, tokenizer.ntok) != account:
                raise ValueError("Independent storage recount differs from the locked decomposition")
            accounts[arm][cid] = account
            if account["total_stored_tokens"] != lock["stored_tokens"][arm]:
                raise ValueError("Stored-token lock and accounting disagree")
            if arm == "ours" and sha(memory_path) != sha(inputs / "archived_no_binding" / f"{cid}.json"):
                raise ValueError("Ours differs from archived no_binding baseline")
            path = run / "predictions" / arm / f"{cid}.json"
            part = read(path)
            expected_ids = [row["id"] for row in questions if row["conv_id"] == cid]
            if [row["id"] for row in part] != expected_ids:
                raise ValueError(f"Incomplete, duplicated or reordered population: {arm}/{cid}")
            prediction_hashes[f"{arm}/{cid}"] = sha(path)
            for raw in part:
                row = dict(raw)
                reference = gold[row["id"]]
                if any(row[key] != reference[key] for key in ("question", "category", "conv_id", "qa_index")):
                    raise ValueError("Question identity differs from canonical gold")
                if row["arm"] != arm or row["stored_tokens"] != account["total_stored_tokens"]:
                    raise ValueError("Prediction arm/storage metadata mismatch")
                context = validate_context(row, units, tokenizer.ntok, runner.core.digest)
                prompt = reader_request(context, row["question"])
                key = runner.core.digest([protocol["environment"]["models"]["Qwen/Qwen3.5-9B"],
                                          20260907, protocol["reader_system"], prompt, 96, False])
                shard = histories.index(cid) % protocol["shards"]
                native_path = run / f"evaluate_{shard}/cache/generations" / f"{key}.json"
                native = read(native_path)
                validate_receipt(row, native, key)
                native_hashes[f"evaluate_{shard}/{key}"] = sha(native_path)
                row["gold"] = reference["gold"]
                row["official_f1"] = runner.core.f1(row["prediction"], row["gold"], row["category"])
                rows.append(row)
        if len(rows) != 1540 or len({row["id"] for row in rows}) != 1540:
            raise ValueError("Complete full1540 population is required for every arm")
        scores[arm] = rows
    validation = {"status": "PASS", "arms": len(ARMS), "questions_per_arm": 1540,
                  "histories": 10, "prediction_count": sum(map(len, scores.values())),
                  "pilot_validation": pilot_validation,
                  "native_receipt_count": len(native_hashes), "input_manifest_sha256": sha(inputs / "manifest.json"),
                  "protocol_sha256": sha(run / "protocol.json"), "scorer_sha256": sha(code / "vendor/source/refine.py"),
                  "tokenizer_sha256": sha(args.tokenizer), "selection_record_sha256": sha(args.selection),
                  "prediction_file_sha256": prediction_hashes, "native_receipt_sha256": native_hashes,
                  "checks": ["input/code fingerprints", "frozen prepare/evaluation contracts", "canonical full1540 gold",
                             "all16 complete paired populations", "archived no_binding baseline identity",
                             "source coverage", "exact independent storage/read token recount", "context payload reconstruction",
                             "native generation keys and receipts", "reader selection rule"]}
    return scores, accounts, protocol, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "collected/full1540")
    parser.add_argument("--memory-root", type=Path, default=ROOT / "package/run")
    parser.add_argument("--inputs", type=Path, default=ROOT / "package/input")
    parser.add_argument("--selection", type=Path, default=ROOT / "READER_SELECTION.json")
    parser.add_argument("--pilot", type=Path, default=ROOT / "collected/reader_pilot")
    parser.add_argument("--pilot-inputs", type=Path, default=ROOT / "package/pilot_input")
    parser.add_argument("--tokenizer", type=Path, default=ROOT.parents[1] / "model_staging/qwen35_9b_c202236/tokenizer.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    scores, accounts, protocol, validation = validate_run(args)
    export(args.out, scores, accounts, protocol, validation, read(args.selection))


if __name__ == "__main__":
    main()
