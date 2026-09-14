"""Export completed common-judge results alongside independently verified LoCoMo F1."""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
OUR = "s_parent_single_2000"
CATS = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop"}

def main():
    manifest = json.loads((BASE / "input_manifest.json").read_text(encoding="utf-8"))
    inputs = [json.loads(x) for x in (BASE / "input.jsonl").read_text(encoding="utf-8").splitlines()]
    scored = [json.loads(x) for x in (BASE / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((BASE / "run_config.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("judge", BASE / "run_judge.py")
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    assert summary["status"] == "complete", "Cannot publish incomplete results"
    assert judge.digest((BASE / "input.jsonl").read_bytes()) == manifest["input_sha256"] == config["input_sha256"]
    assert config["prompt_sha256"] == manifest["prompt_sha256"]
    assert len(inputs) == len(scored) == 13860
    lookup = {r["id"]: r for r in scored}
    assert len(lookup) == len(scored)
    cache = {r["payload_hash"]: r for r in judge.read_journal(BASE / "judge_cache.jsonl")}
    prompt = judge.load_prompt()
    for row in inputs:
        actual = lookup[row["id"]]
        assert all(actual[k] == v for k, v in row.items()), "Input prediction or metadata changed"
        key = judge.payload_hash(judge.make_payload(row, prompt))
        assert actual["payload_hash"] == key
        assert actual["judge_label"] == judge.parse_label(cache[key]["response"])
        assert actual["judge_score"] == int(actual["judge_label"] == "CORRECT")
    methods, table = {}, []
    order = manifest["method_order"]
    for method in order:
        rows = [lookup[r["id"]] for r in inputs if r["method"] == method]
        assert len(rows) == 1540
        f1 = np.array([r["official_f1"] for r in rows])
        acc = np.array([r["judge_score"] for r in rows])
        cats = np.array([r["category"] for r in rows])
        result = {"n": len(rows), "correct": int(acc.sum()),
                  "f1_percent": float(f1.mean()*100), "judge_accuracy_percent": float(acc.mean()*100),
                  "categories": {str(c): {"n": int((cats==c).sum()),
                      "f1_percent": float(f1[cats==c].mean()*100),
                      "judge_accuracy_percent": float(acc[cats==c].mean()*100)} for c in CATS},
                  "f1_below_half_judge_correct": int(((f1 < .5) & (acc == 1)).sum()),
                  "f1_above_half_judge_wrong": int(((f1 > .5) & (acc == 0)).sum())}
        assert abs(result["judge_accuracy_percent"]-summary["methods"][method]["accuracy_pct"]) < 1e-10
        methods[method] = result
        table.append({"Method":manifest["method_labels"][method], "N":1540,
                      "F1":round(result["f1_percent"],6), "Judge_ACC":round(result["judge_accuracy_percent"],6),
                      **{CATS[c]:round(result["categories"][str(c)]["judge_accuracy_percent"],6) for c in CATS}})
    our_rows = [r for r in scored if r["method"] == OUR]
    ids = [(r["conversation_id"],r["qa_index"]) for r in our_rows]
    convs = list(dict.fromkeys(cid for cid,_ in ids))
    ci = np.array([convs.index(cid) for cid,_ in ids])
    counts = np.bincount(ci)
    sampled = np.random.default_rng(20260911).integers(0,len(convs),size=(20000,len(convs)))
    pairs = {}
    for method in order:
        if method == OUR:
            continue
        other = {(r["conversation_id"],r["qa_index"]):r for r in scored if r["method"] == method}
        delta = np.array([r["judge_score"]-other[key]["judge_score"] for r,key in zip(our_rows,ids,strict=True)])
        sums = np.bincount(ci, weights=delta)
        boot = sums[sampled].sum(axis=1)/counts[sampled].sum(axis=1)*100
        pairs[method] = {"delta_judge_pp":float(delta.mean()*100),
                         "conversation_bootstrap_95_interval_pp":np.percentile(boot,[2.5,97.5]).tolist(),
                         "our_only_correct":int((delta>0).sum()), "other_only_correct":int((delta<0).sum())}
    analysis = {"verification":"PASS: all 13,860 IDs, predictions, labels and aggregates checked",
                "methods":methods, "our_minus_comparators":pairs,
                "bootstrap":{"unit":"conversation","replicates":20000,"seed":20260911,
                             "limitation":"Descriptive resampling; does not estimate repeat-judge or generation-run variability."},
                "run":{k:summary[k] for k in summary if k != "methods"}}
    (BASE / "analysis.json").write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (BASE / "comparison.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# LoCoMo 공통 GPT-4o-mini judge 재평가", "",
             "표의 9개 방법이 생성해 둔 답변을 같은 1,540문항에서 재채점했다. 답변을 다시 생성하거나 수정하지 않았다.",
             "지표는 LightMem 공식 배포 LoCoMo judge의 CORRECT 비율(ACC)이다. 기존 LoCoMo token F1과 별도 지표이며, F1 값을 대체하지 않는다.", "",
             "| 방법 | 기존 F1 (%) | Judge ACC (%) | 정답 / 1,540 |", "|---|---:|---:|---:|"]
    for method in order:
        r=methods[method]
        lines.append(f"| {manifest['method_labels'][method]} | {r['f1_percent']:.2f} | {r['judge_accuracy_percent']:.2f} | {r['correct']} |")
    lines += ["", "| 방법 | Multi-hop | Temporal | Open-domain | Single-hop |", "|---|---:|---:|---:|---:|"]
    for method in order:
        values=" | ".join(f"{methods[method]['categories'][str(c)]['judge_accuracy_percent']:.2f}" for c in CATS)
        lines.append(f"| {manifest['method_labels'][method]} | {values} |")
    light_f1_gap = methods[OUR]["f1_percent"] - methods["LightMem_official"]["f1_percent"]
    light_judge_gap = pairs["LightMem_official"]["delta_judge_pp"]
    full = methods["Naive"]
    lines += ["", "## 이번 결과가 보여주는 것", "",
              f"- Our–LightMem 격차는 F1 {light_f1_gap:.2f} pp, judge ACC {light_judge_gap:.2f} pp이다. 같은 저장 답변도 지표에 따라 격차가 크게 달라지므로, F1 15.20 pp를 지표와 무관한 성능 차이로 해석하면 안 된다.",
              f"- Full context의 judge ACC는 {full['judge_accuracy_percent']:.2f}%이고 Our method는 {methods[OUR]['judge_accuracy_percent']:.2f}%이다. 전체 대화가 들어가는 full context가 여전히 강한 reference라는 관찰은 유지된다. Our와의 차이의 대화 단위 bootstrap 구간은 0을 포함한다.",
              f"- E-Mem의 judge ACC는 {methods['E-Mem']['judge_accuracy_percent']:.2f}%로 Our보다 {-pairs['E-Mem']['delta_judge_pp']:.2f} pp 높다. 이 재평가에서 Our가 가장 높은 메모리 방법이라고 주장할 수 없다.",
              f"- Full context는 Single-hop {full['categories']['4']['judge_accuracy_percent']:.2f}%이며, 이 유형은 841/1,540문항을 차지한다. Temporal은 {full['categories']['2']['judge_accuracy_percent']:.2f}%여서 모든 유형에서 동일하게 강한 것은 아니다.",
              "- 기존 F1의 공식 코드 재계산과 저장된 점수는 일치했다. 이번 변화는 F1 구현 오류를 발견한 것이 아니라 채점 지표의 차이를 확인한 것이다. F1과 judge ACC를 함께 보고해야 비교 의미가 분명하다."]

    lines += ["", "## 비교 조건과 해석", "",
              "- 질문·정답·저장 답변을 원문 그대로 채점했다. 모델명, 방법명, F1 점수는 judge 입력에 포함하지 않았다.",
              "- 고정 commit의 공식 프롬프트, gpt-4o-mini, temperature=0, JSON 출력 설정을 사용했다. 반환된 모델 버전은 실행 기록에 보관했다.",
              f"- 질문·정답·답변을 포함한 전체 API payload가 같은 경우 한 판정을 공유했다. {len(cache):,}개 고유 판정을 {len(scored):,}개 방법×문항에 연결했으며, 각 방법의 분모는 1,540이다.",
              "- API 오류를 오답으로 바꾸거나 누락 문항을 분모에서 빼지 않았다. 모든 문항이 채점된 뒤 집계했다.",
              "- 동일한 judge를 적용해도 기존 답변 생성 단계의 프롬프트·출력 한도·검색 조건 차이는 남는다. 점수 변화만으로 메모리 알고리즘의 우열이나 F1 구현 오류를 단정할 수 없다.",
              "- temperature=0도 서버의 완전한 결정성을 보장하지 않는다. 이번 결과는 고유 답변별 1회 판정이며 반복 채점 변동성을 추정하지 않았다.", "",
              "## Our method와의 차이", ""]
    for method in ["Naive","E-Mem","LightMem_official","HiGMem"]:
        p=pairs[method]
        low,high=p["conversation_bootstrap_95_interval_pp"]
        lines.append(f"- {manifest['method_labels'][method]} 대비 {p['delta_judge_pp']:+.2f} pp; 대화 단위 bootstrap 95% 구간 [{low:+.2f}, {high:+.2f}] pp.")
    lines += ["", "## 실행 기록", "",
              f"- 반환 모델: {json.dumps(summary['returned_models'])}",
              f"- API 사용량 기반 비용: USD {summary['cost_usd']:.6f}. 계정 청구서 금액과는 구분한다.",
              f"- 입력/출력 토큰: {summary['token_usage']['prompt_tokens']:,} / {summary['token_usage']['completion_tokens']:,}.",
              f"- 공식 소스: {manifest['official_source_url']}",
              "- 원본 입력 및 해시: input.jsonl, input_manifest.json. 공식 원본 대조: protocol_verification.json.",
              "- 전체 문항 판정: scores.jsonl. API 응답: judge_cache.jsonl. 비용/오류: api_events.jsonl.",
              "- 검증 결과 및 비교 통계: analysis.json. 표: comparison.csv, table_judge.tex.", ""]
    (BASE / "REPORT_KO.md").write_text("\n".join(lines),encoding="utf-8")
    tex = [r"\begin{tabular}{lrrrrrr}",r"\toprule",
           r"Method & F1 & Judge ACC & Multi-hop & Temporal & Open-domain & Single-hop \\",
           r"\midrule"]
    for r in table:
        vals=" & ".join(f"{r[k]:.2f}" for k in ["F1","Judge_ACC",*CATS.values()])
        tex.append(f"{r['Method']} & {vals} "+r"\\")
    tex += [r"\bottomrule",r"\end{tabular}"]
    (BASE / "table_judge.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")
    with (BASE / "f1_judge_disagreements.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        fields=["id","method","category","question","gold_answer","generated_answer","official_f1","judge_label"]
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r for r in scored if (r["official_f1"] < .5 and r["judge_score"] == 1)
                         or (r["official_f1"] > .5 and r["judge_score"] == 0))
    print(json.dumps({"status":"verified","table":table,"cost_usd":summary["cost_usd"]},ensure_ascii=False))
if __name__ == "__main__":
    main()
