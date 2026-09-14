# Qwen3.5-9B LoCoMo 재검증 결과

Qwen3.5-9B에서 고정 refined 방법의 평균 F1은 r40보다 소폭 높았다. 사전 지정 주 분석377문항에서 +0.618점이지만, 대화 cluster bootstrap95% 신뢰구간[-0.091,+1.664]점이0을 포함하므로 안정적인 추가 개선이 확립됐다고 단정할 수 없다. 전체 모델·데이터 일반성 결론은 Gemma와 LongMemEval 결과가 필요하다.

2026-09-08 06:54 UTC 완료. Writer, source utility scorer, reader 모두 Qwen/Qwen3.5-9B이며, 고정 revision은 c202236235762e1c871ad0ccb60c8ee5ba337b9a이다. 세 방법 각각507개 고유 질문의 답변이 완료됐고 빈 생성은0개다. Qwen3-8B 결과는 개발 이력으로만 보존한다.

F1은0–100 척도이며 차이는 동일 문항에서 계산했다.

| 사전 고정 분석 대상 | 문항 수 | Seed | r40 | Refined | Refined−r40 | Refined−seed |
|---|---:|---:|---:|---:|---:|---:|
| 주 분석: 이전 평가에 없던 질문 | 377 | 54.095 | 55.687 | 56.305 | +0.618 | +2.211 |
| 보조 분석: 기존 audit | 100 | 54.315 | 52.979 | 53.288 | +0.309 | −1.027 |
| 생성 전체: 세 대화 category1–4 | 507 | 54.834 | 56.070 | 56.579 | +0.508 | +1.745 |

Refined−r40의95% 신뢰구간은 주 분석[-0.091,+1.664], 보조 분석[-0.373,+1.028], 생성 전체[-0.055,+1.134]점이다. 주 분석에서는13문항 향상,7문항 하락,357문항 동률이었다. Refined−seed 주 분석은 +2.211점이며95% 신뢰구간[+0.554,+3.682]점이다. 기존 audit100에서는 refined가 seed보다 낮았다는 결과도 함께 보존한다.

모든 분석은 같은 대화3개(conv-42/47/50)를 사용한다. 주 분석의377개 질문은 이전 pilot/audit에 없었지만, 대화 이력 자체는 이전 개발 과정에 노출됐다. 따라서 완전히 미노출된 대화에 대한 평가가 아니다. 대화 cluster가3개뿐이라 bootstrap 구간의 해석에도 제약이 있다. 전체507개에는 주 분석377개, audit100개, 별도 이전 pilot30개가 포함된다.

방법은 결과 확인 전에 고정했다. Refined는 r40 위에 source-only utility로 선택한 추가 메모리와 질문 검색 키를 최대2000 stored tokens 예산으로 붙인다. 이 결과만으로 동일 저장 예산에서의 우월성, 규칙 없는 방법, 새 방법론의 독창성까지 입증하지는 않는다. Gemma 및 LongMemEval 결과를 보고 규칙이나 threshold를 바꾸지 않는다.

검증 근거:

- 원본 집계: [predeclared_locomo_results.json](verified_qwen35/predeclared_locomo_results.json)
- 실행 protocol: [protocol.json](verified_qwen35/protocol.json)
- 모델 역할·캐시 감사: [QWEN35_ROLE_CACHE_AUDIT.md](QWEN35_ROLE_CACHE_AUDIT.md)
- 사전 고정 모집단: [evaluation_populations.json](../locomo_transfer/evaluation_populations.json)

Protocol SHA256:54eb61046be0f4a6f67755fbb6250ad3f81dc2edd6ccc396f92c23447988dc26. Population SHA256:615b81dad77a64dafcea718f60472bc4b0a526d0f19cafe290c520fb0c17d631. Dataset SHA256:696e2090c4c419229c7243d088f6a69e8d30507c9313c2dd7d2aa6cffd636c0d.