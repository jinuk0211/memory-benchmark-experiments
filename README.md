# Memory benchmark experiments

LoCoMo, LongMemEval, recursive self-improvement, ablation study의 실행 코드와 저장된 결과를 모은 연구 아카이브입니다. 2026-09-14에 기존 작업 폴더에서 범위를 선별했습니다. 모델 추론이나 judge API를 새로 호출한 결과가 아닙니다.

## 결과 바로가기

| 실험 | 보고서 | 데이터 |
|---|---|---|
| LoCoMo 전체 1,540문항, 16개 설정 | [통합 보고서](outputs/locomo_all_results_20260913/REPORT_KO.md) | [전체·유형별 F1 및 judge](outputs/locomo_all_results_20260913/FULL1540.csv) |
| LongMemEval 공통 50문항 GPT-4o | [결과](outputs/longmemeval_common50_gpt4o_20260912/RESULTS_KO.md) | [합친 표](outputs/longmemeval_common50_gpt4o_20260912/combined_gpt4o_summary.csv), [개별 판정](outputs/longmemeval_common50_gpt4o_20260912/scored_answers.csv) |
| 자기개선 129개 구성 | [실험별 설명](outputs/self_improvement_review_20260912/SELF_IMPROVEMENT_REVIEW_KO.md) | [129개 이력](outputs/locomo_all_results_20260913/HISTORY_129.csv) |
| 동결 후 모델 전이 | [Qwen/Gemma 보고서](generalization_20260908/modern/LOCOMO_MODEL_TRANSFER_RESULTS.md) | [집단별 표](outputs/locomo_all_results_20260913/MODEL_TRANSFER.csv) |
| Ablation 7조건 × 300문항 | [결과](experiments/locomo_ablation300_20260913/results/RESULTS_KO.md) | [CSV](experiments/locomo_ablation300_20260913/results/table.csv), [질문별 결과](experiments/locomo_ablation300_20260913/results/scored_predictions.json) |
| Binding ablation 2조건 × 1,540문항 | [결과](experiments/locomo_binding1540_20260913/results/RESULTS_KO.md) | [CSV](experiments/locomo_binding1540_20260913/results/table.csv), [질문별 비교](experiments/locomo_binding1540_20260913/results/per_question_comparison.csv) |
| 토큰·시간 | [집계 범위](outputs/locomo_pareto_20260911/README_KO.md) | [토큰](outputs/locomo_pareto_20260911/token_points.csv), [시간](outputs/locomo_pareto_20260911/latency_points.csv) |

## LoCoMo 전체 평가

Qwen3.5-9B, 10대화, category 1–4의 1,540문항. F1은 공식 category-specific token F1 ×100이고, judge 정답률과 다른 지표입니다. 우리 5개 설정은 MiniLM 임베딩을 사용했습니다. 저장된 LLM judge는 **gpt-4o-mini-2024-07-18**입니다.

| 방법 | F1 ×100 | GPT-4o-mini 정답/문항 | 정답률 |
|---|---:|---:|---:|
| Seed | 56.26 | 미채점 | — |
| r40 | 55.52 | 미채점 | — |
| Refined / Our method | 55.53 | 1120/1540 | 72.73% |
| Recursive v1 | 55.86 | 미채점 | — |
| Seed-parent | 56.17 | 미채점 | — |
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

`Refined = Our method = s_parent_single_2000`. Seed-parent는 별도 후속 실험입니다. 동일한 질문을 평가했지만 baseline마다 검색·답변 프롬프트, 출력 한도와 예산은 다릅니다. 모든 LoCoMo 대화는 개발 중 노출됐습니다.

## LongMemEval GPT-4o 평가

Judge: **gpt-4o-2024-08-06**. baseline 50문항과 우리 방법의 기존 12문항은 서로 다른 평가 집합입니다. 50문항은 모두 `single-session-user`이며 전체 500문항 점수가 아닙니다.

| 방법 | 정답/평가 문항 | 정답률 |
|---|---:|---:|
| Full-context | 46/50 | 92.00% |
| Seed | 11/12 | 91.67% |
| r40 | 7/12 | 58.33% |
| Refined | 6/12 | 50.00% |
| Seed-parent | 11/12 | 91.67% |
| LangMem | 37/50 | 74.00% |
| SimpleMem | 38/50 | 76.00% |
| LightMem | 45/50 | 90.00% |

공통50은 200개 답변에 대한 182개 고유 요청의 실제 GPT-4o 응답을 보존합니다. 우리 12문항 표는 `official_results.json`의 `minilm_*`를 사용합니다. 같은 폴더의 초기 `results.json`은 응답 파싱 진단 단계이므로 최종 표와 혼용하지 않습니다.

## 자기개선과 모델 전이

과거 개발은 **Qwen3-8B + Qwen3-Embedding-0.6B, dev70**입니다. 129개 구성 중 초기 기준·재현 3개를 제외하면 후보/대조는 126개, 개선 채택은 7개입니다. r40는 설정 이름입니다.

| 채택 경로 | dev70 F1 ×100 |
|---|---:|
| r00 초기 추출 | 47.5552 |
| r02 누락 사실 감사 | 48.6016 |
| r03 원문 보강 / Seed | 52.7943 |
| r05 상대 날짜 연결 | 54.3894 |
| r06 월 단위 날짜 | 54.8601 |
| r12 사교 표현 필터 | 57.1189 |
| r40 사실 + 4턴 원문 | 58.6203 |
| Refined 부모 근거·검색키 | 59.2744 |

동결 후 Qwen3.5-9B와 Gemma4-E4B-it 전이는 Qwen3-Embedding-0.6B를 사용했습니다. 507문항은 새 질문377 + 기존 audit100 + pilot30이며 중첩된 집단을 독립 표본처럼 합산하지 않습니다. 대화 자체는 개발 중 노출됐고, 두 모델 모두 Refined−r40의 새 질문 집단 95% CI가 0을 포함합니다.

Recursive v1은 10대화×3라운드에서 7변경을 채택했으며 전체 F1 55.8560으로 Seed보다 낮았습니다. Seed-parent는 56.1682, Seed 대비 −0.0915점입니다. Packed-marginal source audit는 18.33→8.68로 하락해 Seed로 복귀했고 새 benchmark 전체 점수는 없습니다. Recursive v2는 구현만 보존되며 저장된 GPU 평가와 Gemma18 전이 점수는 없습니다.

129개 구성은 점수·설명·채택 여부·원본 경로·SHA256 감사 기록을 보존합니다. **129개 원본 실행의 개별 질문 결과 전체를 복구한 아카이브는 아닙니다.**

## Ablation study

아래는 초기 추출을 공유해 새로 구축한 별도 실행입니다. 기존 Refined 55.5278과 구별합니다.

| 300문항 조건 | F1 ×100 |
|---|---:|
| Our method | 55.7975 |
| w/o cues | 55.4263 |
| w/o audit | 54.7889 |
| w/o temporal | 52.2010 |
| w/o evidence binding | 58.0424 |
| random cues | 55.1647 |
| payload keys | 55.6363 |

전체 1,540문항 후속 paired 비교에서는 fresh Our **55.3733**, w/o evidence binding **58.7143**, 차이 **+3.3410점**, 대화 bootstrap 95% CI **[+2.2271,+4.5079]**입니다. 이 결과에서 binding 제거가 더 높은 F1을 보였습니다.

## 코드와 재현

- [코드 지도·재채점 명령](REPRODUCIBILITY.md)
- [파일별 출처와 SHA256](MANIFEST.json)
- [외부 코드·데이터 고지](THIRD_PARTY.md)

실험별 Python 소스와 동결 입력은 바이트 그대로 보존했습니다. 일부 Markdown의 로컬 링크만 저장소 상대 링크로 바꿨습니다. 과거 보고서의 원격 경로·미완료 서술은 당시 기록이며, 위 최신 결과 표를 우선합니다. 모델 가중치, 가상환경, 접속 키와 대량 중복 복구 백업은 포함하지 않습니다.
