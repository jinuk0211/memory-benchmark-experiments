# Frozen method claim audit — 2026-09-08

현재 코드가 뒷받침하는 중심 주장은 **source 대화만으로 생성한 질의의 counterfactual utility를 이용해 기존 기억에 추가할 검색 키와 근거 payload를 고르고, 이 고정 절차의 증분 이득이 모델·데이터 변경 뒤에도 유지되는가**이다. 문헌상 최초성은 이 코드 감사로 입증되지 않는다. [METHOD_CONTRIBUTION.md](/D:/MemoryData/generalization_20260908/METHOD_CONTRIBUTION.md)의 선행연구 목록을 확장하거나 새로운 논문상 차별성을 주장하지 않는다.

## 실제 구현과 기여의 경계

- **Source-only 선택 신호.** 원문 10-turn/overlap-2 창에서 QA를 생성하고, 답변이 원문에 실제 등장하는 1–12 lexical-word span인 경우만 컴파일한다. 세션별 최대 2개를 고정 seed로 뽑고 세션 일부를 source audit으로 남긴다. 현재 transfer runner는 남은 모든 `probe_fit`을 사용한다. `POLICY.fit_per_conversation=4`를 최종 실행의 표본 수로 기술하면 틀린다. [컴파일·분할](/D:/MemoryData/generalization_20260908/modern/source/compile_research_data.py:19), [호출 경로](/D:/MemoryData/generalization_20260908/modern/source/portable_parent.py:25).
- **Counterfactual surrogate.** 동일한 source 답변 토큰열의 평균 log probability를 empty/full/single/pair 문맥에서 비교한다. Full source의 답변 F1 ≥ 0.8, full−empty ≥ 0.1 nats/token 조건을 적용하고, 옵션 가치는 `clip((ℓ(option)−ℓ(empty))/(ℓ(full)−ℓ(empty)), 0, 1)`이다. 정답 문맥이 2,048 tokens를 넘으면 해당 source probe를 제외한다. 이는 benchmark 문항을 제외하는 규칙과 다르지만, 데이터별 probe coverage 편향은 생길 수 있다. [NLL 계산](/D:/MemoryData/generalization_20260908/modern/source/evidence_utility.py:62), [옵션 값·입장 조건](/D:/MemoryData/generalization_20260908/modern/source/budgeted_evidence.py:71).
- **기존 기억에 대한 routing.** 필요한 source ID를 덮는 기존 r40 unit들의 집합을 찾고, 그 unit 텍스트를 통째로 이어 붙여 추가 payload로 사용한다. 검색 키 `index_text`는 source QA의 질문이다. 선택된 payload를 새로 요약하지 않고 기존 parent를 유지한다는 구조적 성질은 코드로 확인된다. 다만 source ID cover는 의미적 정답 보존의 증명이 아니다. [Cover와 mapping](/D:/MemoryData/generalization_20260908/modern/source/parent_evidence.py:8).
- **저장 예산 보장.** Payload와 다른 검색 키의 token cost를 함께 계산하고, source probe당 최대 한 옵션을 8-token quantum의 multiple-choice knapsack으로 선택한다. 추가 비용은 최대 2,000 tokens다. 최적성은 이산화된 가산 surrogate 목적함수에 한정되며 downstream accuracy의 최적성을 뜻하지 않는다. 원래 parent와 추가 payload의 중복도 비용에 포함된다. [비용·배분](/D:/MemoryData/generalization_20260908/modern/source/budgeted_evidence.py:12), [최종 구성](/D:/MemoryData/generalization_20260908/modern/source/parent_evidence.py:81).

이 조합의 연구적 후보는 **benchmark QA 없이 얻은 답변 의존성 신호를 기존 기억의 추가 인덱싱과 저장 배분에 연결하는 절차**다. 생성 QA, provenance, 최소 비용 cover, knapsack 각각이나 최종 bundle의 성능만으로 독창성을 주장하지 않는다.

## 주장하면 안 되는 것

1. **Certified NLL/answer preservation.** 0.2-nat tolerance는 진단용 `minimum`/`single_policy`를 정하는 데 사용된다. 최종 옵션에는 tolerance를 만족하지 않는 `best_single`도 양의 gain이면 들어갈 수 있다. 원문 옵션을 parent payload로 mapping한 뒤 NLL을 재측정하지 않으며, `portable_parent.augment`는 verifier 결과 없이 `parent_single`을 호출한다. 따라서 raw subset의 utility를 최종 payload에 대한 보장으로 옮기면 안 된다. [진단](/D:/MemoryData/generalization_20260908/modern/source/evidence_utility.py:90), [실제 최종 호출](/D:/MemoryData/generalization_20260908/modern/source/portable_parent.py:17).
2. **Pair synergy가 최종 향상의 원인.** Pair NLL과 생성은 진단 과정에 존재하지만 `parent_single`은 pair option을 제거한다. 계산한 모든 진단을 최종 방법의 활성 구성요소로 열거하지 않는다.
3. **Rule-free 또는 domain-unbiased memory.** Calendar/month 처리, social-fact regex, four-turn/overlap-2 anchor 구성, QA prompt의 사실 유형 지정, literal-span 조건, 정해진 임계값·예산·probe sampling을 상속한다. 이 규칙들이 benchmark gold leakage라는 뜻은 아니지만, 데이터 편향이 없다는 뜻도 아니다.
4. **같은 저장량에서 utility selection이 더 우수함.** `parent − r40`는 추가 저장량 최대 2,000 tokens를 허용한다. 추가 저장, 질문형 검색 키, parent payload 선택이 함께 바뀌므로 양의 차이는 이 bundle의 효용을 보여 준다. utility 기준 자체의 우수성이나 key와 payload 각각의 인과 효과를 분리하지 못한다. `parent − seed`에는 상속된 전처리 차이까지 포함된다.

## 고정 전이 결과로 판단할 수 있는 주장

모든 차이는 같은 문항·모델·실행 설정에서의 `parent − r40`를 우선하고 `parent − seed`를 보조로 본다. 기준 모델은 사용자 지정 Qwen3.5-9B이다. 과거 Qwen3-8B 산출물을 이 기준 실행의 재현 점수로 대체하지 않는다.

| 검증할 주장 | 지지하는 관측 | 반대 관측과 해석 한계 |
|---|---|---|
| Qwen3.5에서 실제 기준 이득이 존재한다 | 고정 LoCoMo population의 full-role Qwen3.5에서 양의 paired 차이 | 기준 이득이 없으면 이후 결과를 단순히 “살아남은 원래 이득”이라고 부를 수 없다. |
| 동일 절차가 다른 데이터로 전이된다 | Qwen3.5 full-role LongMemEval에서도 양의 차이 | LoCoMo에서 양수이고 LME에서 명확히 음수면 현재 절차의 해당 데이터 전이가 약화된다. 넓은 CI를 동반한 0 근처 결과는 불확실성이다. |
| 동일 절차가 다른 모델 계열로 전이된다 | Gemma로 writer/scorer/reader와 기억을 모두 재구성해도 같은 데이터에서 양의 차이 | Gemma의 명확한 음의 차이는 해당 모델 조건에서의 실패다. 전체 역할을 동시에 바꾸므로 writer·scorer·reader 중 원인을 식별하지 못한다. |
| 모델과 데이터를 함께 바꿔도 이득이 유지된다 | 두 모델의 LME 결과가 모두 양수이며 불확실성·비용을 함께 보고 | 두 모델 성공은 이 두 설정의 증거이지 모든 모델·데이터에 대한 보장이 아니다. 한 셀 실패를 다른 셀의 큰 평균 이득으로 숨기지 않는다. |
| 고정 기억이 다른 reader에도 유용하다 | 별도의 reader-only 실험에서 memory/payload/index를 그대로 두고 reader만 바꿔도 이득 | Full-role 전이는 이 주장을 직접 검증하지 않는다. Reader-only 실패와 full-role 성공은 모델별 재구성이 도움이 될 가능성을 보여 줄 뿐, 어느 단계가 원인인지 확정하지 않는다. |

LoCoMo의 audit100/fresh377은 이미 사용된 세 대화에서의 질문 분리다. 세 대화 cluster로 얻는 불확실성의 한계를 유지한다. LME12는 탐색적 pilot이며, 전체 500개/471 base-ID cluster 결과와 혼합하지 않는다. LME token F1는 진단치이고 공식 judge accuracy를 대신하지 않는다. Positive point estimate만으로 확실한 전이를 선언하지 않으며, 모델별 paired effect와 CI를 함께 제시한다.

실행 실패·빈 출력·누락 문항을 선택적으로 삭제하지 않는다. 초기 8K writer capacity의 실패와 full500의 별도 65,536-token(64K) writer capacity 변형의 결과를 분리한다. 현재 Qwen3.5와 Gemma 모두 writer는 65,536, reader/scorer는 8,192 tokens로 고정하며, 실제 target tokenizer의 capacity audit은 아직 완료되지 않았다. 용량 변형은 동일 알고리즘을 더 큰 처리 용량에서 평가하는 실험이며 동일 실행 설정의 재현은 아니다. 각 모델의 tokenizer가 다르므로 같은 2,048/2,000/96 token 규칙도 동일 문자량이나 계산량을 뜻하지 않는다.

## 현재 실험 후에야 분리할 수 있는 원인

현재 고정 benchmark arm과 임계값은 추가·변경하지 않는다. 이후 별도 사전 고정 실험에서 다음 대조가 필요하다.

- **선택 기준:** parent, 후보 payload/key bank, reader 예산, 추가 저장 허용량을 맞춘 utility/random/coverage 등의 선택 대조가 있어야 선택 기준의 효과를 논할 수 있다. 실제 사용량도 맞추거나 명시적으로 통제한다.
- **검색 키와 payload:** payload 집합을 고정한 key 대조와 key 생성·검색 조건을 고정한 payload 선택 대조를 분리해야 한다. 현재 bundle 결과 하나로 어느 쪽이 향상을 만들었는지 판정하지 않는다.
- **Surrogate의 타당성:** 원문 subset utility와 mapping 후 payload utility, 실제 retrieval/answer gain 사이의 관계를 따로 확인해야 한다. 현 실행은 이 일치성을 보장하지 않는다.

기존 identity/random/coverage라는 실험명 자체를 위 대조의 증거로 간주하지 않는다. 동일한 비용·후보·질의 분리 조건이 확인된 결과만 해당 기여 주장에 연결한다. 이 메모는 target 성능으로 새 규칙을 만드는 제안이 아니라, 고정 실험 완료 후 허용되는 결론의 범위다.
