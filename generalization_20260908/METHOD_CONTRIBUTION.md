# Methodological contribution boundary (2026-09-08, before target scores)

이 문서는 현재 target 성능을 보고 만든 refinement 제안이 아니다. 고정 평가의 해석 기준과 추후 연구 방향이다.

## Existing ideas that should not be claimed as new

- 자동 생성 QA를 검색 가능한 기억으로 사용하는 발상은 [PAQ / RePAQ](https://arxiv.org/abs/2102.07033)에 선행한다.
- value granularity를 조절하고 factual keys를 확장하는 접근은 [LongMemEval 원 논문](https://arxiv.org/abs/2410.10813) 자체의 분석과 겹친다.
- counterfactual utility로 문맥을 선택하고 예산 내 memory cards를 만드는 넓은 발상도 [Decision-Aware Memory Cards](https://arxiv.org/abs/2606.08151) 등과 인접한다. 여기서는 해당 논문의 abstract 수준 인접성만 확인했으므로 구체적인 동일성/차별성은 추가 full-text 검토가 필요하다.
- Knapsack, 원문 provenance, 기존 payload 재사용은 각각만으로 독창성 주장이 되지 않는다.

## Testable claim for the current frozen method

Source 대화에서 생성한 질의의 행동적 유용성을 측정하고, 검증된 원문 ID를 기존 parent payload에 연결하여 제한된 추가 저장 예산을 배분하면, 고정 reader 예산에서 다른 데이터 및 다른 reader에도 이득이 유지되는가?

현재 구현은 singleton/full option만 선택하며 pair synergy나 behavioral verification을 최종 방법으로 포함하지 않는다. calendar/social filtering과 같은 기존 전처리도 포함하므로, bundle의 전체 성능 향상을 순수한 utility-selection 기여로 설명하면 안 된다.

Parent − r40가 증분 대비다. 다만 parent는 +2,000 stored tokens를 허용하므로 이 대비만으로 선택 기준의 우수성을 저장량 증가와 완전히 분리하지 못한다. 원래 연구의 identity/random/coverage controls가 무엇을 이미 검증했는지 함께 보고, 필요한 경우 같은 추가 저장량의 대조군을 별도 고정 실험으로 둔다.

## If transfer weakens: prospective directions, not implemented changes

1. **여러 reader에 대한 source utility 안정성.** 한 모델의 절대 likelihood threshold에 의존하는 대신, source-only probe에서 모델별 상대 개선의 안정성과 불확실성을 측정한다. 새 데이터 정답으로 threshold를 고르지 않는다. 추가 계산량을 함께 비교해야 한다.
2. **답변 가능성 보존과 coverage를 함께 보는 예산 배분.** 자주 생성되는 QA만 반복 보존하지 않도록 source information coverage와 behavioral preservation 사이의 trade-off를 명시한다. 구체적 목적함수는 데이터셋별 question type 규칙 없이 정의해야 한다.
3. **원문 보존과 검색 키 기여를 분리한 통제 실험.** payload와 read tokens를 고정하고 key construction/selection만 바꾸며, source-only 유용성이 실제 downstream retrieval 및 answer gain을 예측하는지 검증한다.

어느 방향도 현재 target 결과를 보고 같은 test set에서 승자를 고르는 식으로 진행하지 않는다. 이미 본 target 집합은 차기 버전에서 development로 취급하거나 별도 unseen evaluation을 확보한다. 현재 frozen transfer 실험이 우선이다.

