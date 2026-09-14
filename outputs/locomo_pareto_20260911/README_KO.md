# LoCoMo Pareto figures

메인 표의 9개 구성과 추가 실험을 포함한 15개 구성을 모두 그렸습니다. Pareto 선에 연결되는 점의 수와 그림에 표시된 전체 점의 수는 다릅니다.

- `tokens_all_dark.png`: 전체 15개 구성, 첨부 예시와 같은 어두운 배경.
- `tokens_all_light.png / .pdf / .svg`: 전체 15개 구성, 논문용 밝은 배경.
- `tokens_main_dark.png`: 메인 표 9개 구성.
- `tokens_main_light.png / .pdf / .svg`: 메인 표 9개 구성, 논문용.
- `latency_dark.png`: latency 기록이 있는 baseline 6개 + 내부 구성 4개.
- `latency_paper.png / .pdf / .svg`: 같은 latency 그림의 논문용 버전.
- `token_points.csv`, `token_provenance.json`, `latency_points.csv`: 그림의 원자료와 출처.

모든 F1은 Qwen3.5-9B로 답한 동일한 LoCoMo 1,540개 질문의 공식 F1 × 100입니다. 질문별 재채점 검증 결과는 `../locomo_comparability_audit_20260911/scores_audit.json`에서 읽습니다. 15개는 서로 다른 논문 15편이 아니라 **baseline 및 내부 변형을 포함한 15개 실행 구성**입니다.

## Token figure

가로축은 **선택된 구축·준비·QA 경로의 LLM 입력 + 출력 토큰 합계**, 단위는 백만(M)입니다. 임베딩 토큰과 API 가격은 포함하지 않습니다. 첨부 예시처럼 로그축을 뒤집어 오른쪽일수록 토큰이 적고, 위쪽일수록 F1이 높습니다.

한 점보다 토큰이 같거나 적고 F1이 같거나 높은 다른 점이 있으며, 둘 중 하나가 엄격히 더 좋으면 그 점은 dominated입니다. 표시한 모든 점에 대해 이 관계를 계산했습니다. 반올림 전 수치를 사용했습니다.

| 그림 범위 | Pareto 경계, 적은 토큰부터 |
|---|---|
| 메인 표 9개 구성 | LightMem official → Our method → E-Mem |
| 전체 15개 구성 | R40 → Seed → E-Mem |

**전체 구성을 포함하면 Seed가 Our method보다 적은 토큰으로 더 높은 F1을 기록합니다.** 메인 9개 그림의 경계를 모든 내부 구성까지 포괄하는 결과로 해석하면 안 됩니다. 이는 선택된 실행들의 관측상 관계이며 통계적 유의성을 뜻하지 않습니다.

내부 구성의 비용은 독립적인 cold run 실측 총액이 아니라 필요한 준비 비용과 전체 logical reader 사용량을 합친 재구성 값입니다. 캐시로 재사용한 답변의 logical reader 토큰도 포함해 신규 API 호출분만 합산할 때 생기는 유리한 효과를 제거했습니다. 탐색 전체의 개발·튜닝 비용은 이 축에 포함하지 않았습니다.

| 구성 | 필요한 준비 LLM 토큰 | Logical reader | 합계 |
|---|---:|---:|---:|
| Seed | 831,791 | 3,322,139 | 4,153,930 |
| R40 | 831,791 | 3,280,912 | 4,112,703 |
| Our method | 7,073,329 | 3,283,525 | 10,356,854 |
| Recursive v1 | 9,852,376 | 3,297,516 | 13,149,892 |

Our 준비 비용은 seed build 831,791 + source QA 795,965 + utility scoring 5,445,573입니다. Recursive는 추가 준비 2,779,047을 더합니다. Seed와 R40에는 source QA/utility 비용이 필요하지 않습니다. R40의 anchor_time → filter_social_facts → fuse_evidence 변환은 추가 LLM generation 없이 동작합니다. 근거는 `experiments/refinement_usage_20260909/evidence/run_transfer.py`와 `experiments/recursive_minilm_20260909/source/portable_parent.py`, 같은 source 폴더의 `continuous_v2.py`, `memory_ops.py` 및 세 plan입니다. 원시 사용량은 `experiments/refinement_usage_20260909/reports/current_verified/USAGE_REPORT.json`에 있습니다. 모든 경로는 프로젝트 루트 기준입니다.

E-Mem ≥60,802,774와 HiGMem ≥38,514,830은 보존된 원장에서 확인된 LLM 토큰 하한입니다. 실패 요청의 일부가 미계측이고, E-Mem은 계측된 실패 시도, HiGMem은 이전 invocation/retry도 포함합니다. **모든 행이 완전히 동일하게 계측된 독립 실행 총비용은 아닙니다.** 높은 비용 방향인 왼쪽으로 화살표를 표시하고, 하한 좌표를 사용하는 Pareto 구간은 점선으로 구분했습니다. 점 사이의 선은 순서 안내이며 중간 비용에서 그 성능을 달성한다는 의미가 아닙니다.

## Latency figure

저장된 측정 범위에 맞춰 패널을 나눴습니다.

- 왼쪽: Mem0, A-MEM, LangMem, SimpleMem, E-Mem의 native 질문 타이머. LightMem official은 retrieval 평균 0.0480428228초 + answer API 평균 1.0322810312초의 합 1.0803238540초이며 별도 마커를 사용합니다. 구성별 runtime/concurrency와 타이머 범위가 같다고 보장되지 않아 공동 frontier를 그리지 않았습니다.
- 오른쪽: Seed, R40, Our method, Recursive v1의 **기록된 LLM + embedding 호출시간 합 / 1,540**입니다. 배치와 캐시 재사용의 영향을 포함한 상각 시간이며 질문별 end-to-end 응답시간이 아닙니다. 같은 저장 지표에서의 관측 경계만 연결했습니다. 오른쪽 F1 축은 차이를 볼 수 있도록 확대했습니다.

Our method, Full-context, HiGMem에는 이 그림에 사용할 공통 질문별 end-to-end latency가 없습니다. 해당 값을 추정하거나 0으로 채우지 않았습니다. SimpleMem의 848개 부분 실행 latency addendum도 사용하지 않았습니다.

## Reproduce

프로젝트의 원본 결과를 유지한 상태에서 다음을 실행합니다. 새 모델 호출은 없습니다.

```powershell
C:\Python314\python.exe D:\MemoryData\outputs\locomo_pareto_20260911\plot_tokens.py
C:\Python314\python.exe D:\MemoryData\outputs\locomo_pareto_20260911\plot_latency.py
```

Python 3.14, Matplotlib 3.10.8에서 생성했습니다. Python 리뷰와 데이터 대조를 수행했고, 이미지의 축·화살표·레이블 겹침을 확인했습니다.

## Suggested English captions

**Main token figure.** Quality–token trade-off for the nine main-table configurations on the same 1,540 LoCoMo questions with Qwen3.5-9B. The reversed logarithmic x-axis shows accounted LLM input and output tokens for each selected build/preparation and QA path, excluding embeddings. Internal-method costs reconstruct the required preparation plus cache-normalized logical reader usage. E-Mem and HiGMem show recorded token lower bounds; dashed segments involve a lower-bound coordinate. The frontier describes the plotted accounting figures under native pipeline conditions.

**Extended token figure.** Quality–token trade-off across all 15 verified configurations, including internal variants and alternative LightMem pipelines. Under the displayed selected-path accounting, R40, Seed, and E-Mem are nondominated. Costs and lower-bound notation follow the main token figure; the connecting line does not imply interpolated achievable performance.

**Latency figure.** Quality and recorded latency for ten configurations with available timing. The left panel reports native question timers, with LightMem shown separately as the sum of recorded retrieval and answer-API components. The right panel reports observed LLM-plus-embedding call time amortized over 1,540 questions for four internal variants and uses a zoomed F1 scale. Batching and cache reuse affect the latter metric; it is not end-to-end query latency. No frontier is shared across panels or baseline timing scopes.