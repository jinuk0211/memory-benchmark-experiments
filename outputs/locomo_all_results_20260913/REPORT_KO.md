# LoCoMo 결과 모음 — 2026-09-13

로컬에 보존된 완료 결과를 통합했다. 새 GPU 실행·답변 생성·judge API 호출은 하지 않았다. 동일 실행의 중간 집계나 복제본은 별도 방법으로 중복 집계하지 않았다.

전체 1,540문항 완료 설정 16개, 그중 공통 GPT-4o-mini judge 완료 9개, 과거 dev70 구성 129개와 모델 전이 6개 집단별 비교를 포함한다. 모든 과거 GPU 실행·실패 시도까지 복구했다는 뜻은 아니다.

## 1. 전체 1,540문항 결과

Qwen3.5-9B, LoCoMo 10대화의 category 1–4 전체. Category 5는 제외한다. F1은 공식 category-specific token F1×100의 질문별 평균이다. 우리 5개 구성의 임베딩은 MiniLM이다. baseline은 저장된 각 구현의 설정을 사용했다.

**Judge는 gpt-4o-mini-2024-07-18이다. 앞서 LongMemEval 50문항을 채점한 GPT-4o와 구별한다.** LightMem 배포 LoCoMo ACCURACY_PROMPT의 CORRECT 비율이며 F1과 다른 지표다.

| 방법 | 공식 F1×100 | Judge 정답/문항 | Judge 정답률 |
|---|---:|---:|---:|
| 우리 Seed | 56.26 | 미채점 | — |
| 우리 r40 | 55.52 | 미채점 | — |
| 우리 Refined (Our method) | 55.53 | 1120/1540 | 72.73% |
| 우리 Recursive v1 | 55.86 | 미채점 | — |
| 우리 Seed-parent | 56.17 | 미채점 | — |
| Full-context | 55.17 | 1099/1540 | 71.36% |
| E-Mem | 56.96 | 1188/1540 | 77.14% |
| LightMem official | 40.33 | 1008/1540 | 65.45% |
| HiGMem | 40.33 | 860/1540 | 55.84% |
| SimpleMem | 38.19 | 842/1540 | 54.68% |
| A-MEM | 36.85 | 799/1540 | 51.88% |
| Mem0 | 36.50 | 839/1540 | 54.48% |
| LangMem | 29.93 | 644/1540 | 41.82% |
| LightMem pipeline | 46.60 | 미채점 | — |
| LightMem direct | 38.20 | 미채점 | — |
| CertMem v15 | 42.23 | 미채점 | — |

Our method, Paper Refined, s_parent_single_2000은 같은 설정이다. Seed-parent는 별도 후속 대조다. LightMem official/pipeline/direct 역시 서로 다른 실행 설정이다.

검증: 기존 15개 구성 23,100개 방법×문항의 공통 재채점은 보존 집계와 일치했다. Seed-parent는 별도 로컬 검증 결과에서 가져왔다. judge scores.jsonl 13,860행을 다시 집계해 9개 방법 각각 1,540개 및 정답 수의 summary 일치를 확인했다. 실제 고유 judge 응답 10,624개, 오류 0개이며 완전히 같은 API 입력에만 판정을 공유했다.

## 2. 유형별 공식 F1×100

| 방법 | Multi-hop (282) | Temporal (321) | Open-domain (96) | Single-hop (841) |
|---|---:|---:|---:|---:|
| 우리 Seed | 46.08 | 41.76 | 13.57 | 70.08 |
| 우리 r40 | 39.68 | 49.70 | 14.09 | 67.78 |
| 우리 Refined (Our method) | 38.13 | 50.64 | 15.58 | 67.79 |
| 우리 Recursive v1 | 38.76 | 50.48 | 16.01 | 68.19 |
| 우리 Seed-parent | 46.70 | 42.34 | 13.52 | 69.49 |
| Full-context | 41.77 | 36.24 | 15.66 | 71.40 |
| E-Mem | 44.87 | 54.67 | 30.64 | 64.89 |
| LightMem official | 31.97 | 44.85 | 16.74 | 44.09 |
| HiGMem | 27.95 | 26.40 | 12.70 | 52.96 |
| SimpleMem | 29.57 | 24.87 | 23.33 | 47.86 |
| A-MEM | 28.98 | 29.09 | 17.43 | 44.66 |
| Mem0 | 37.58 | 13.29 | 24.53 | 46.36 |
| LangMem | 28.37 | 27.27 | 21.65 | 32.42 |
| LightMem pipeline | 42.01 | 34.36 | 24.31 | 55.35 |
| LightMem direct | 36.49 | 22.71 | 24.67 | 46.23 |
| CertMem v15 | 26.93 | 34.97 | 9.92 | 53.81 |

Seed에서 r40로 바꾸면 temporal은 41.76→49.70으로 올랐지만 multi-hop 46.08→39.68, single-hop 70.08→67.78로 내려가 전체 평균은 감소했다. Refined의 r40 대비 추가분은 +0.0054점이다. Recursive v1은 Refined보다 +0.3282점이지만 Seed보다 −0.4037점이다.

## 3. 과거 자기개선 개발 기록

**Qwen3-8B + Qwen3-Embedding-0.6B, 동일 dev70** 결과다. 위 Qwen3.5-9B 전체 평가와 직접 비교하지 않는다. 당시 8B 실행은 개발 이력으로 보존하며 9B 결과로 바꾸어 표기하지 않는다.

18개 history 장부 215행을 중복 제거하면 129개 구성이다. 초기 기준 1개와 재현 전용 2개를 제외한 후보/대조는 126개, 기록상 개선 채택은 7개다. r40는 설정 이름이며 40회 재귀를 뜻하지 않는다. 가장 큰 숫자 접두어는 r68이다.

| 채택 경로 | dev70 F1×100 | 변경 |
|---|---:|---|
| r00_session10 | 47.5552 | 대화 세션 전체를 읽고 세션당 최대 10개 사실을 추출하는 시작 기준. 이름의 10은 턴 수가 아님. |
| r02_audit | 48.6016 | 기존 메모리를 원문과 대조해 빠진 사실을 보강. |
| r03_dialogue_residual | 52.7943 | 추출 메모리에 원문 대화 블록을 추가해 요약에서 누락되는 정보 보존. |
| r05_calendar_anchor | 54.3894 | 대화 기록일을 기준으로 상대 날짜 표현을 달력 날짜로 정규화. |
| r06_calendar_month | 54.8601 | 월 단위 해석을 포함한 날짜 정규화 변형. 선택된 최종 전처리. |
| r12_filter_current_best | 57.1189 | 당시 최고 후보에 상투적인 사교 사실 필터 적용. 최종 계보에 채택. |
| r40_fused_four_turn | 58.6203 | 사실과 원문 4턴/2턴 중첩 블록을 결합하고 날짜 정보를 유지. 채택된 r40. |
| s_parent_single_2000 | 59.2744 | 출처를 덮는 기존 부모 전체 단위를 본문으로, 원문 질문을 검색키로 추가. 단일+전체 근거 후보; 추가 2000토큰. |

전체 129개 구성의 설명·원본 점수·채택 여부·원본 결과 경로·SHA256은 [HISTORY_129.csv](HISTORY_129.csv)에 있다. 21개 실험 계열의 상세 설명은 [기존 자기개선 감사](../self_improvement_review_20260912/SELF_IMPROVEMENT_REVIEW_KO.md)를 참조한다. 그 보고서의 이후 LongMemEval 채점 미완료 서술은 현재 최신 결과로 사용하지 않는다.

## 4. 동결 후 모델 전이

Writer·source utility scorer·reader를 각각 Qwen3.5-9B 또는 Gemma4-E4B-it로 다시 실행했다. 공통 임베딩은 Qwen3-Embedding-0.6B이며 위 MiniLM 전체 평가와 구별한다. 아래는 모두 F1×100이다.

| 모델 | 평가 집단 | n | Seed | r40 | Refined |
|---|---|---:|---:|---:|---:|
| Qwen/Qwen3.5-9B | 세 대화 전체 | 507 | 54.8340 | 56.0704 | 56.5787 |
| Qwen/Qwen3.5-9B | 새 질문 | 377 | 54.0947 | 55.6871 | 56.3052 |
| Qwen/Qwen3.5-9B | 기존 audit | 100 | 54.3146 | 52.9788 | 53.2878 |
| google/gemma-4-E4B-it | 세 대화 전체 | 507 | 47.1824 | 51.7907 | 51.6723 |
| google/gemma-4-E4B-it | 새 질문 | 377 | 46.2218 | 50.5582 | 50.7410 |
| google/gemma-4-E4B-it | 기존 audit | 100 | 49.1909 | 52.3379 | 50.8647 |

507=새 질문 377+기존 audit100+별도 기존 pilot30이므로 집단을 합산해 독립 평가 수로 세면 안 된다. 새 질문 377에서도 대화 3개 자체는 과거 노출됐다. Refined−r40의 95% 대화 bootstrap CI는 Qwen [−0.0908,+1.6645], Gemma [−2.1968,+1.4989]로 둘 다 0을 포함한다.

## 5. 추가 재귀·대조 결과와 미실행 항목

- dev70 검색·응답 내부 재귀 t: 59.1275. 최대 3회 중 2변경 채택 후 중단. 별도 source 표현을 사용한 v: 56.6619, 1변경 후 중단. 둘 모두 위 129개 안에 포함된다.
- Recursive source rehearsal v1: 전체 F1 55.8560. 10대화×3=30 source 라운드, 6대화에서 7변경 채택. 재작성 검색키 옵션 779 instances 중 실제 선택은 0개였고 변화는 근거 배분에서 발생했다. Refined 대비 탐색적 CI [+0.0271,+0.7727]점.
- Seed-parent: 전체 F1 56.1682. Seed보다 −0.0915점, CI [−1.2308,+1.0921]. 최종 방법으로 채택하지 않았다.
- packed-marginal LoCoMo conv49: source audit F1 18.3333→8.6806으로 나빠져 Seed로 되돌렸다. 고정 후 source QB3은 66.6667→66.6667. 이 값은 benchmark 전체 성능이 아니다.
- Recursive v2: 구현·로컬 테스트 기록은 있으나 저장된 GPU 평가 점수 없음. 계획된 Gemma18 transfer도 v1 기준 미달로 실행되지 않았다.
- source-reconstruction: 설계 단계에서 철회되어 새 평가 점수 없음.

## 6. 보존된 비용·시간

단위 M=백만 LLM 입력+출력 토큰. 선택 구성의 구축/준비+QA 비용이며 임베딩과 전체 탐색 비용은 제외한다. reported는 원 실행 보고값, derived는 필요한 준비와 캐시 중립 logical reader를 합친 재구성값, lower_bound는 관측 하한이다. 이들을 모두 독립 cold-run 실제 비용으로 해석하지 않는다.

| 방법 | LLM 토큰 M | 비용 집계 | 질의 시간 초 | 시간 범위 |
|---|---:|---|---:|---|
| 우리 Seed | 4.153930 | derived | 0.4138 | observed_batch_amortized_llm_and_embedding_call_time |
| 우리 r40 | 4.112703 | derived | 0.3860 | observed_batch_amortized_llm_and_embedding_call_time |
| 우리 Refined (Our method) | 10.356854 | derived | 0.3032 | observed_batch_amortized_llm_and_embedding_call_time |
| 우리 Recursive v1 | 13.149892 | derived | 0.3625 | observed_batch_amortized_llm_and_embedding_call_time |
| 우리 Seed-parent | 전체 합계 미확인 | incomplete_total | 0.4833 | incremental_reader_evaluation_timer |
| Full-context | 54.106911 | reported | 미확인 | — |
| E-Mem | ≥60.802774 | lower_bound | 36.3957 | saved_native_question_timer |
| LightMem official | 5.300706 | reported | 1.0803 | saved_retrieval_plus_answer_api_component_sum |
| HiGMem | ≥38.514830 | lower_bound | 미확인 | — |
| SimpleMem | 19.777906 | reported | 13.7781 | saved_native_question_timer |
| A-MEM | 27.285662 | reported | 0.6905 | saved_native_question_timer |
| Mem0 | 27.370629 | reported | 0.3200 | saved_native_question_timer |
| LangMem | 25.942379 | reported | 0.4097 | saved_native_question_timer |
| LightMem pipeline | 5.582404 | reported | 미확인 | — |
| LightMem direct | 15.267609 | reported | 미확인 | — |
| CertMem v15 | 27.163164 | reported | 미확인 | — |

Seed-parent의 위 전체 토큰 합계는 비워 두었다. 확인된 추가 reader 사용량만 LLM 3.322521M, embedding 0.697131M이며 수입된 구축·utility 비용은 추가다.

우리 기존 4개 구성의 시간은 배치·캐시 효과가 있는 LLM+embedding 호출 시간 합/1,540이다. baseline native 질문 타이머, LightMem 검색+답변 API 시간 합, Seed-parent 평가 타이머와 범위가 다르므로 동일한 end-to-end 지연 순위로 해석하지 않는다. 전체 구성에 공통인 평균 구축시간은 이 집계에 없다.

## 해석과 출처

우리 전체 평가 F1 최고는 Seed 56.2597, 전체 비교 최고는 E-Mem 56.9567이다. 저장된 9개 judge 비교에서도 E-Mem 77.14%가 Refined 72.73%보다 높다. Refined−Full-context judge 차이는 +1.36pp이며 대화 bootstrap CI [−1.00,+3.64]로 0을 포함한다.

같은 질문·모델·scorer를 사용해도 reader 프롬프트, 출력 제한, 검색 흐름과 예산은 baseline마다 다르다. 모든 LoCoMo 대화는 과거 개발 중 노출됐다. 개발 dev70에서 선택한 최고 59.2744를 전체 평가 또는 미노출 일반화 점수로 제시하지 않는다.

- [공통 F1 감사](../locomo_comparability_audit_20260911/scores_audit.json)
- [공통 judge 원본](../locomo_gpt4omini_judge_20260911/REPORT_KO.md)
- [Seed-parent 독립 검증](../../migration_20260910/pilot_runs/seed_parent_locomo_qwen35_minilm_r2/LOCAL_RESULT_VERIFIED.json)
- [모델 전이](../../generalization_20260908/modern/LOCOMO_MODEL_TRANSFER_RESULTS.md)
- [개발 기록 수 감사](../../generalization_20260908/modern/REFINEMENT_HISTORY_COUNT.md)
- [토큰·시간 정의](../locomo_pareto_20260911/README_KO.md)
- [baseline 실행 조건 감사](../locomo_comparability_audit_20260911/REPORT_KO.md)
