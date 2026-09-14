# LoCoMo 55.53 대 40.33 비교 감사

2026-09-11. 저장된 코드·실행 manifest·질문별 예측·로그를 조사했다. 새 모델 추론은 실행하지 않았다. 공통 재채점 결과는 scores_audit.json, 재현 코드는 audit_scores.py에 기록한다.

**관측된 점수 격차는 설명할 수 있지만, 15.20포인트 전부를 제안한 refinement의 우수성으로 귀속할 수는 없다.** 같은 질문과 공식 채점기를 사용한 시스템 비교이며, reader·검색·메모리 보존·출력 제한까지 통일한 메모리 품질 실험은 아니다.

## 결과 위치

| 용도 | 실제 로컬 파일 |
|---|---|
| 모든 완료 설정의 점수 | `ALL_RESULTS.md` (historical source: `results/comparisons/all_methods_20260909/ALL_RESULTS.md`) |
| 통합 정밀 수치 | `ALL_CATEGORY_SCORES.json` (historical source: `results/comparisons/all_methods_20260909/ALL_CATEGORY_SCORES.json`) |
| 15개 설정의 분석용 예측 패키지 | [Analysis_bundle.zip](../baseline_drive_export_20260909/Analysis_bundle.zip) |
| baseline 원본·실행 출처 | `BASELINE_INDEX.md` (historical source: `upload_preparation_20260910/BASELINE_INDEX.md`) |
| baseline 풀린 예측 출처 | `baseline_manifest.json` (historical source: `upload_preparation_20260910/extracted_baseline_results/baseline_manifest.json`) |
| Our 55.53 예측 | `s_parent_single_2000.jsonl` (historical source: `experiments/refinement_usage_20260909/evidence/runs/qwen35_baseline/s_parent_single_2000.jsonl`) |
| Our 실행 당시 protocol | `protocol.json` (historical source: `experiments/refinement_usage_20260909/evidence/runs/qwen35_baseline/protocol.json`) |
| Seed / r40 | `seed.jsonl` (historical source: `experiments/refinement_usage_20260909/evidence/runs/qwen35_baseline/seed.jsonl`), `r40_fused_four_turn.jsonl` (historical source: `experiments/refinement_usage_20260909/evidence/runs/qwen35_baseline/r40_fused_four_turn.jsonl`) |
| 논문 보관본 | `INPUT_MANUSCRIPT.tex` (historical source: `migration_20260910/INPUT_MANUSCRIPT.tex:419`) |

논문 주석이 참조하는 analysis/export_tables.py는 로컬에서 확보되지 않았다는 기록이 있다(`README_BASE.md` (historical source: `upload_preparation_20260910/README_BASE.md:35`)). 예전 개발 탐색의 일부 질문별 원본도 없다. 모든 과거 결과가 완전히 보존됐다고 말할 수는 없지만 이번 표의 완료 예측은 확보돼 있다.

## 1. 동일 채점 여부

canonical 데이터 SHA256은 cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993, 공식 scorer SHA256은 8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd다. Windows 소스 사본은 LF 줄바꿈으로 정규화해 hash를 비교한다.

공식 코드는 정규화와 Porter stemming 후 token F1을 계산한다. category 1 Multi-hop은 comma로 prediction/gold를 나누고 각 gold 항목의 best-match F1을 평균한다. category 3 Open-domain은 gold의 첫 semicolon 앞부분을 사용한다. category 2·3·4는 단일 F1이다. 이번 표는 category 5를 제외한다. [공식 평가 코드](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py), `로컬 사본` (historical source: `HiGMem/official_locomo_evaluation.py:126`).

문항 수는 Multi-hop282 / Temporal321 / Open-domain96 / Single-hop841. 전체는 질문별 평균이며 category macro 평균이 아니다. LightMem 원본 metrics의 judge 관련 값이나 E-Mem 원본의 일반 F1 필드를 그대로 평균하면 안 된다. 최종 answer를 canonical gold에 공통 scorer로 넣어야 한다.

같은 evaluator는 같은 답변 생성 조건을 뜻하지 않는다. 짧은 명사구, 날짜 형식, comma list 지시가 token F1에 영향을 줄 수 있어 reader 조건은 별도 점검 대상이다.

## 2. 최종 답변 조건과 native 동작

| 방법 | 확인된 주요 조건 / 해석 |
|---|---|
| Our | Qwen3.5-9B FP16, thinking off, temperature0. BM25+dense RRF60, 후보 top120, evidence2048 tokens, 답변96 tokens. exact short answer·comma list·session date 해석 지시. |
| Full-context | 전체 원문의 별도 reference. shortest exact-answer prompt, 답변32 tokens. Our와 동일 prompt/cap이 아니다. |
| LightMem official | native add/search, LLMLingua2 rate0.6·STM768·offline update·combined top60. native speaker/date prompt, QA temperature0, 출력 cap 명시 없음. writer는 temperature0.1 등 자체 설정. |
| HiGMem | turn10+event10 → event별 LLM turn 선택 → LLM relevance filter → native category별 reader/JSON answer. QA cap 명시 없음. |
| Mem0 | native update/top100 검색에 benchmark extraction/answer 연결. 수정 없는 upstream 재현이라고 부르면 안 된다. |
| A-MEM | native metadata/evolution schema, 메모리 출력 cap1000, JSON 실패 시 skip 복원. 별도 A-MEM reader. 실제 evolution 실패가 많았다. |
| LangMem | native memory manager/search에 공통 reader. Trustcall patch failure가 기록됐다. |
| SimpleMem | native hybrid retrieval 내부 planning/reflection은 유지. retrieved entries를 top10으로 자른 후 공통 reader로 답변하여 native AnswerGenerator 전체 경로와 다르다. |
| E-Mem | native block query/aggregation과 benchmark query template.1538 성공 답변 유지+2 복구. 복구에 XML guidance/repetition penalty 변경이 있어 기록해야 한다. |

공통 baseline YAML의 generation_max_length=1024를 실제 제한이라고 쓰면 틀릴 수 있다. 실행 comparison proxy가 max_tokens/max_completion_tokens를 제거한다. A-MEM은 특정 memory_add 요청에만1000을 복구했다. QA와 writer cap을 구분해야 한다. [상세 조건·실행별 코드 줄 번호](baseline_conditions.md).

공통 wrapper에 Current Time의 실행시각 fallback도 있다. 실제 선택 QA 전체에 적용됐는지는 이번에 확정하지 못했으므로 날짜 오류의 원인으로 단정하지 않는다.

같은 Qwen checkpoint와 temperature/thinking은 비교를 돕지만 위 차이를 없애지는 않는다.2048-token evidence budget도 모든 baseline의 공통 제한이 아니다.

## 3. HiGMem 의혹에 대한 직접 답

HiGMem의 profile=false, event metadata, immediate turn links=false는 공식 README의 paper setting과 맞는다. ablation이라는 플래그 이름만으로 임의 축소판이라고 단정하면 안 된다. [공식 paper setting](https://github.com/ZeroLoss-Lab/HiGMem#reproducing-the-paper-setting), `로컬 README` (historical source: `HiGMem/README.md:110`).

최종 tar archive에서 초기 검색1540회, event→turn LLM 선택15400회, 최종 relevance filter3743회, 답변1540회를 확인했다. core·prompts·memory layer hash도 실행 manifest와 일치한다. **이벤트 계층 및 LLM 선택을 제거한 vector-only 변형이라는 의혹은 실제 기록과 맞지 않는다.**

query 문자열로 유일하게 연결되고 gold evidence가 있는1514문항에서 정답 turn의 문항별 평균 recall은 초기 vector 검색45.67% → event 확장 후보68.22% → 최종 필터60.79%였다. 후보의 정답 근거가 필터 뒤 모두 사라진 질문은121개였다. 전체1540개 context 중67개는 비어 있었다.

이는 event 계층이 도움을 주면서도 최종 선택에서 근거 손실이 생긴다는 진단이다.121문항을121개 오답이나 특정 F1 손실로 환산할 수 없다. gold evidence annotation도 가능한 모든 근거를 표시한다는 보장은 없다.

DoorDash 퇴사 시점 질문은 정답 evidence D1:3을 회수했으나 January, 2023 대신 this month로 답해 F1=0이었다(`질문별 결과` (historical source: `upload_preparation_20260910/extracted_baseline_results/higmem/scored_predictions.json:5`)). 날짜 해석·답변 방식도 점수에 관여한다. [상세 HiGMem 감사](higmem_audit.md).

## 4. 격차를 설명할 때 함께 밝혀야 할 관측

**LightMem 설정 민감도.** 대표 공식 설정은40.3262이지만 다른 완료 adapted pipeline은46.5954다. Our 차이는 각각+15.2016pp와+8.9324pp다. 후자를 native 최고값으로 대신 주장할 수도 없지만,15.20을 모든 설정에 걸친 고정 효과로 표현해서도 안 된다. direct38.20도 별도 변형이다. `실행 protocol` (historical source: `lightmem_official_20260909/PROTOCOL.md`).

**A-MEM 실행 품질.** evolution5882회 중 JSON 실패로3569회가 생략됐다(약60.68%). 메모리 출력 length 종료는3575회다. QA는 전부 정상 stop이므로 빈 답변0개가 건강한 메모리 구축을 보장하지 않는다. 그렇다고 실패를 제거하면 몇 점 오른다고 추정할 근거도 없다. native 실패 정책 결과와 별도 수정판을 구분해야 한다. `최종 보고` (historical source: `amem_final_results_r26/results_summary.md:20`).

**Our가 보존하는 정보.** Seed는 session fact extraction → source audit → 모든 원문 turn 추가를 거친다. r40는 날짜 처리와 사실 필터 후 원문4턴 블록을 사실과 결합한다. Our는 source-probe cue를 더한다.2000은 추가 storage budget이며 전체 저장량이 아니다.2048은 reader evidence budget이다. 평균 Our 저장량은 약61.9k tokens/history다. `구축 체인` (historical source: `experiments/recursive_minilm_20260909/run_transfer.py:170`), `원문 결합` (historical source: `experiments/recursive_minilm_20260909/source/memory_ops.py:48`), `검색·reader` (historical source: `experiments/recursive_minilm_20260909/run_transfer.py:57`).

**마지막 refinement의 효과.** Seed56.2597 → r4055.5224 → Our55.5278이다. 마지막 cue augmentation의+0.0054pp와 외부 baseline 대비+15.20pp는 서로 다른 비교다. Seed도 단순 raw-RAG가 아니다. 같은 reader·retriever·budget 아래 raw-only / facts-only / facts+raw / +date / +cue가 있어야 단계별 효과를 분리할 수 있다.

**E-Mem 및 Full-context.** E-Mem56.9567은 Our보다1.4289pp 높다. Full-context55.1689와 Our 차이는+0.3589pp다. 각각 단일 완료 실행을 비교한 것이며 그 자체로 통계적으로 확립된 승패를 뜻하지 않는다.

**개발 노출.** 모든 LoCoMo history와 이전 aggregate가 개발 중 노출됐다는 protocol이 있다. benchmark QA를 메모리 구축에 직접 전달하지 않는 코드 경로와 미노출 test generalization은 별개다. `PROTOCOL.md` (historical source: `experiments/recursive_minilm_20260909/PROTOCOL.md:16`).

## 5. 논문에서 방어 가능한 설명

> 동일 LoCoMo category1–4의1540문항에서 저장된 답변을 공통 공식 F1 evaluator로 재채점했다. 제안 시스템은55.53, 선택한 LightMem 공식 설정 재실행은40.33, E-Mem은56.96이다. 각 baseline의 native reader와 retrieval 조건이 달라 이 격차는 완료된 시스템 구성들의 차이를 나타내며, 단일 메모리 연산의 통제된 인과 효과로 해석하지 않는다. HiGMem의 이벤트 계층과 LLM 기반 선택은 실행 로그로 확인됐다. 추가 cue augmentation 자체의 증분은 내부 base 대비+0.0054pp다.

재실험을 한다면 다음 순서가 유용하다. 이번 감사에서는 실행하지 않았다.

1. **저장 context에 공통 reader 적용:** 동일 checkpoint/thinking/prompt/temperature/output cap으로 재답변하고 native 점수와 나란히 보고한다. context 미보존 방법은 정확한 설정으로 먼저 재추출한다. 기존 답변을 사후 잘라 F1을 높이는 것은 reader 통제 실험이 아니다.
2. **Our 구성 단계별 내부 비교:** raw-only, facts-only, facts+raw, 날짜 처리, cue를 동일2048-token reader와 retrieval로 비교한다.
3. **native 실패 민감도:** A-MEM JSON/cap 복구, HiGMem final filter on/off를 별도 변형으로 보고한다. original을 조용히 대체하지 않는다. SimpleMem은 native AnswerGenerator 경로도 추가한다.
4. **대표 설정과 외부 평가:** LightMem 모든 완료 설정을 부록에 공개하고 선택 이유를 명시한다. 미노출 데이터에서 고정 recipe로 평가한다. conversation 단위 uncertainty와 단일 실행·개발 노출 한계를 유지한다.

공통 reader 실험은 native 시스템 평가를 대체하지 않고 reader 영향의 민감도를 확인한다. retrieval량·storage까지 맞추는 것은 별도 통제 축이다. 모든 요소를 한꺼번에 바꾸면 원인을 다시 분리할 수 없다.

## 6. 이번 감사에서 다시 계산한 수치

15개 완료 설정 × 1,540문항 = **23,100개**의 공식 F1을 같은 코드로 재계산했다. 모두 canonical ID·question·gold·category 검증을 통과했다. 원본 답변은 수정하지 않았다. [기계 판독 결과](scores_audit.json), [재현 스크립트](audit_scores.py).

| 표 행 | 재계산 F1 ×100 |
|---|---:|
| Full-context | 55.168900 |
| Mem0 | 36.498359 |
| A-MEM | 36.845900 |
| LangMem | 29.932364 |
| SimpleMem | 38.189102 |
| LightMem official | 40.326188 |
| HiGMem | 40.333102 |
| E-Mem | 56.956713 |
| Our | 55.527814 |

표의 반올림값 55.53−40.33은15.20이다. 원점수 차이는 LightMem 대비15.201625, HiGMem 대비15.194711이다. 따라서 원점수 기반 소수 둘째 자리 격차는 각각15.20과15.19다.

**LightMem 대비15.2016pp의 구성:**

| 유형 | 문항 수 | 유형 내 F1 차이 | 전체 차이에 기여한 pp |
|---|---:|---:|---:|
| Multi-hop | 282 | +6.1560 | +1.1273 |
| Temporal | 321 | +5.7950 | +1.2079 |
| Open-domain | 96 | −1.1647 | −0.0726 |
| Single-hop | 841 | +23.6934 | +12.9391 |
| 전체 | 1,540 | | +15.2016 |

Single-hop은 문항의54.61%이며, 순 전체 격차의 약85.1%인12.9391pp를 만든다. 이는 산술적 분해이며 왜 그 유형에서 잘했는지의 인과 추정은 아니다. HiGMem 대비는 Single-hop8.0973pp, Temporal5.0540pp, Multi-hop1.8644pp, Open-domain0.1791pp로 합계15.1947pp다.

**답변 길이 진단:** 공백 구분 단어 수 평균은 Our4.319, LightMem4.989, HiGMem4.420, Mem04.451, SimpleMem5.145, E-Mem5.733, LangMem6.079, A-MEM6.641이다. 모델 tokenizer의 token 수가 아니다.30단어 초과 답변은 Our4개, LightMem0개, HiGMem6개다. baseline의 광범위한 장문 출력만으로 격차를 설명할 근거는 약하다. 이 진단이 prompt/cap 영향 자체를 배제하는 것은 아니다.

**대화 단위 paired bootstrap:** 10개 대화를 복원추출해20,000회 계산했다(seed20260909). 각 replicate는 뽑힌 대화의 모든 질문과 중복 횟수를 포함해 질문별 가중 평균을 낸다.

| Our − comparator | 차이 pp | 기술적 95% 구간 |
|---|---:|---:|
| LightMem official | +15.2016 | [+13.41,+17.07] |
| HiGMem | +15.1947 | [+11.29,+19.36] |
| LightMem adapted pipeline | +8.9324 | [+5.24,+13.53] |
| E-Mem | −1.4289 | [−3.75,+0.64] |
| Full-context | +0.3589 | [−1.67,+2.44] |
| Seed | −0.7319 | [−2.95,+1.47] |
| r40 base | +0.0054 | [−0.42,+0.43] |

이 구간은 관측된10개 대화에 대한 기술적 재표본화다. 독립 model run의 변동성, 설정 탐색 보정, reader 차이 통제, 미노출 일반화를 제공하지 않는다. 따라서 LightMem/HiGMem 대비 큰 관측 차이와 개별 메모리 연산의 인과 효과는 구별한다.

Our의 실제 저장값에서 모든1,540 답변은 stop 종료였다. 평균 reader evidence2011.87tokens, 대화당 저장61909.7tokens. Seed의96-token length 종료는1개, r40와 Our는0개다.

재검증 결과: 기존 full1540 집계의15개 설정·75개 전체/유형별 수치와 대조했으며 최대 차이는7.11e−15pp(부동소수점 정밀도 수준)였다. 별도 Python 코드 리뷰에서 공식 함수 보존,15개 결과 어댑터,bootstrap 가중 방식,단어 수 정의를 점검했고 정확성 결함은 발견되지 않았다.

재현 명령: python D:/MemoryData/outputs/locomo_comparability_audit_20260911/audit_scores.py

범위: 새 GPU 추론0회. 원본 실험·예측·논문은 수정하지 않았으며, 이번 감사 폴더에 분석 코드·결과·보고서만 추가했다.