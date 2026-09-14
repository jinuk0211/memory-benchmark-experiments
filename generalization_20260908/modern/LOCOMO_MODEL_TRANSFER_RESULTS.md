# LoCoMo 모델 전이 결과: Qwen3.5-9B와 Gemma4-E4B-it

현재 증거로는 고정 refined 방법의 r40 대비 추가 이득이 모델을 바꿔도 안정적으로 유지된다고 결론낼 수 없다. 주 분석에서 Qwen3.5는 +0.618점, Gemma는 +0.183점이지만 두 신뢰구간 모두0을 포함한다. Gemma의 전체507문항과 기존 audit100에서는 refined가 r40보다 낮았다. Seed 대비 전체 구성의 이득과 새 utility 단계의 추가 효과를 구분해야 한다.

두 모델 모두 writer, source likelihood scorer, reader를 해당 모델로 다시 실행했다. 방법·프롬프트·utility 기준·메모리 예산·retrieval·평가 질문 집단은 결과를 보기 전에 고정했다. Qwen3-8B는 실수로 수행된 개발 이력으로만 보존하며 새 기준 결과로 사용하지 않는다.

점수는 질문별 category-specific F1의 평균을100배한 값이다. 95% 신뢰구간은 같은 대화3개를 묶어10,000회 paired cluster bootstrap한 결과다.

| 모델 | 사전 고정 분석 대상 | n | Seed | r40 | Refined | Refined−r40 | 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-9B | 주 분석: 이전 평가에 없던 질문 | 377 | 54.095 | 55.687 | 56.305 | +0.618 | [−0.091,+1.664] |
| Gemma4-E4B-it | 주 분석: 이전 평가에 없던 질문 | 377 | 46.222 | 50.558 | 50.741 | +0.183 | [−2.197,+1.499] |
| Qwen3.5-9B | 보조 분석: 기존 audit | 100 | 54.315 | 52.979 | 53.288 | +0.309 | [−0.373,+1.028] |
| Gemma4-E4B-it | 보조 분석: 기존 audit | 100 | 49.191 | 52.338 | 50.865 | −1.473 | [−2.626,0.000] |
| Qwen3.5-9B | 생성 전체 | 507 | 54.834 | 56.070 | 56.579 | +0.508 | [−0.055,+1.134] |
| Gemma4-E4B-it | 생성 전체 | 507 | 47.182 | 51.791 | 51.672 | −0.118 | [−2.145,+0.736] |

주 분석의 refined−seed는 Qwen +2.211점(CI[+0.554,+3.682]), Gemma +4.519점(CI[+3.307,+5.198])이다. 하지만 seed→r40에서 이미 Qwen +1.592점, Gemma +4.336점이 발생했다. 따라서 이 큰 차이를 새 source-utility refinement만의 기여로 해석하면 안 된다. 기존 calendar/filter/four-turn 처리도 포함된 전체 구성의 결과다. Qwen의 보조 audit100에서는 refined가 seed보다1.027점 낮았다.

대화별 주 분석 refined−r40도 일관되지 않다.

| 대화 | 질문 수 | Qwen3.5 차이 | Gemma 차이 |
|---|---:|---:|---:|
| conv-42 | 155 | −0.091 | +0.849 |
| conv-47 | 107 | +0.520 | −2.197 |
| conv-50 | 115 | +1.664 | +1.499 |

두 모델 모두3개 방법×507개의 완전한 출력이 있고 빈 생성은0개다. 각 모델1521개의 F1을 독립적으로 재계산했으며 원본 질문 집단,105개 source hash, protocol와 memory lock,6개 paired comparison을 검증했다. 주 분석377개 질문은 이전 평가에 없었지만 대화 이력3개 자체는 개발 중 노출됐다. 따라서 완전히 미노출된 대화 평가가 아니며, 대화 cluster가3개뿐이라는 불확실성도 남는다. 전체507개는377개 주 분석+100개 audit+30개 다른 이전 pilot 질문으로 구성된다.

Refined는 r40에 최대2000 stored tokens의 메모리와 질문 검색 키를 추가한다. 현재 비교는 동일 저장 예산에서 utility 선택 자체의 우월성을 분리하지 않는다. Qwen3-Embedding-0.6B는 두 모델에서 공통으로 고정됐으므로, generative 모델 역할의 전이 검증이며 embedding 모델까지 교체한 검증은 아니다. 방법의 독창성이나 규칙 없는 동작을 이 점수만으로 주장하지 않는다.

Gemma의 두 unused pair 진단 NLL과 마지막 cache 값 사이에 차이가 관찰됐다. 같은 key의 뒤따른 answer_removed 결과가 cache를 덮어쓴 경우이며, 실제 parent_single이 사용하는 empty/full/single 점수와 모든 generation control은 일치했다. 현재 선택 메모리와 최종 답변 점수에는 영향을 주지 않았다. 자세한 관찰 범위는 역할·캐시 감사에 기록했다.

LongMemEval은 고정된12문항 실행성 점검에 진입했고, 두 모델 전체500문항 큐가 이어서 대기한다. 전체 실행은65k GPU 검증 후 시작하며, 공식 gpt-4o-2024-08-06 채점용 API 설정은 아직 없다. 아직 데이터 전이 이득에 대한 결론은 없다. 여기서 관찰한 결과에 맞춰 방법이나 threshold를 바꾸지 않는다.

검증 가능한 원본:

- [Qwen 결과](verified_qwen35/predeclared_locomo_results.json) · [Qwen 독립 감사](QWEN35_RESULT_AUDIT.md)
- [Gemma 결과](verified_gemma4/predeclared_locomo_results.json) · [Gemma 독립 감사](GEMMA_RESULT_AUDIT.md)
- [Gemma 역할·캐시 감사](GEMMA_ROLE_CACHE_AUDIT.md) · `전체500 실행 안내` (historical source: `generalization_20260908/full_transfer/FULL500_RUNBOOK.md`)
- [사전 고정 모집단](../locomo_transfer/evaluation_populations.json)
사전 계산된 유형별 보조 집계도 함께 보면 효과가 고르지는 않다. 아래는 주 분석377개에서 refined−r40의 F1 차이다. 유형별 평균에 동일 가중치를 준 macro 차이는 Qwen +0.359점, Gemma +1.949점이며, 사전 지정 주 지표인 질문 평균을 대체하지 않는다. 위의95% CI는 질문 평균 차이에 대한 것으로 macro 차이에 적용할 수 없다.

| LoCoMo 유형 | n | Qwen3.5 차이 | Gemma 차이 |
|---|---:|---:|---:|
| 1 | 67 | +1.084 | +0.799 |
| 2 | 78 | −0.671 | −0.717 |
| 3 | 24 | 0.000 | +8.333 |
| 4 | 208 | +1.023 | −0.619 |

[Source-only 동작 진단](METHOD_TRANSFER_DIAGNOSTICS.md)에서는 두 모델 모두138개 fit probe의 full−empty likelihood 기준을 통과했고, full-source answer F1 기준을 통과한 수는 Qwen89개, Gemma83개였다. 선택된 추가 메모리는36개/32개, 실제 사용한 추가 예산은5821/5734토큰(세 대화 합계6000토큰 상한)이었다. 따라서 추가 단계가 실행되지 않았거나 예산을 크게 남겼다는 설명은 맞지 않는다. 단, 모델이 만든 source 질문 집단은 거의 서로 다르므로 이 admission 수를 모델 능력의 paired 비교나 benchmark 차이의 원인으로 해석하면 안 된다. 질문·정답 오류 사례를 이용한 튜닝이나 새 추론은 수행하지 않았다.