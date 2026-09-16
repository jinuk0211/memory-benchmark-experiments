# LoCoMo 전체 ablation 실험 기록

최종 평가 모집단은 LoCoMo 10개 대화의 category 1–4 전체 **1,540문항**이다. Multi-hop 282, Temporal 321, Open-domain 96, Single-hop 841문항이며 모든 비교에서 동일한 canonical ID를 사용한다. 데이터 SHA256은 `cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993`이다.

## 고정된 비교와 현재 단계

| 비교 | 바뀌는 내용 | 전체 평가 범위 | 현재 단계 |
|---|---|---|---|
| Revised reference | evidence binding을 제거한 Ours 기준, temporal 헤더 혼합 수정, grounded reader 고정 | 16개 설정 × 1,540문항 | 원격 GPU 평가 실행 중 |
| Source marginal utility | 같은 부모 메모리·후보·예산·DP에서 실제 base retrieval 대비 source answer likelihood 증가량 사용 | 14개 설정 × 1,540문항 | 입력·parity 검증 완료, reference 종료 후 GPU 실행 대기 |
| Marginal component extension | no_audit/no_temporal/with_binding 각각의 부모에 대해 source utility 재측정 후 cue 재선택 | 3개 설정 × 1,540문항 | CPU 입력·parity 완료, utility 전체 평가 종료 후 GPU 실행 대기 |
| 최종 GPT-4o-mini 채점 | 최종 선택된 Ours의 원본 예측을 기존과 동일한 semantic rubric으로 채점 | 1,540문항 전체 | 입력·출력 검증 코드 준비, 실제 새 채점은 미실행 |

위 단계 표는 준비 시점의 기록이다. 실제 최신 실행 상태는 각 실험의 STATUS.json과 원격 Supervisor/native 산출물을 확인한다. 실행 중인 부분 결과를 전체 F1로 발표하지 않는다.

## 비교 해석

Grounded reader는 전체 실행 전에 기존 300문항에서 비교하여 고정했다. 이 300문항은 최종 1,540문항과 겹치므로 독립적인 holdout이 아니다. 전체 LoCoMo도 이전 개발에서 노출된 데이터다. 따라서 이번 수치는 탐색적 개선 결과로 보고하며 새로운 독립 검증이라고 설명하지 않는다.

Source marginal utility의 construction에는 benchmark 질문·정답·평가 결과를 넣지 않는다. 기존 source full-context generated answer와 같은 source probe를 사용한다. Source-only construction이라는 사실이 이미 노출된 benchmark에 대한 방법 선택의 편향까지 제거하는 것은 아니다.

모든 사전 지정 random seed 10개의 결과를 유지한다. Random 행은 seed 평균으로 보고하고 seed별 값과 표준편차를 함께 남긴다. 가장 높은 seed만 선택하지 않는다. Uniform-gain control은 동일 후보·DP·예산에서 source utility 값의 역할을 분리한다. 기존 reference 결과와 이후 개선안 결과를 각각 보존한다.

## 통계와 표 작성

- 문항별 F1을 먼저 계산한 뒤 전체 1,540문항에 같은 가중치를 준다.
- 같은 대화의 질문 간 의존성을 반영해 대화 단위 bootstrap 5,000회로 CI를 구한다. 대화는 10개뿐이라는 한계를 명시한다.
- 대화 단위의 paired sign-flip 1,024개를 전수 계산한다.
- Reference의 primary family는 no_cues/random mean/payload_keys 3개 비교다. Marginal의 primary family는 이에 uniform_gain을 추가한 4개 비교다. 각각 Holm 보정을 적용한다.
- Marginal component 3개 비교는 같은 marginal Ours 기준의 별도 secondary 분석으로 보고한다. Primary family에 사후 혼합하지 않으며 secondary p값은 미보정임을 표시한다.
- 개선 여부와 관계없이 시도한 round 및 설정을 기록한다. 이후 방법을 선택하더라도 탐색 결과의 CI·p값을 독립 confirmatory 검정으로 해석하지 않는다.

## Token accounting

Stored tokens는 10개 히스토리의 **parent payload + 추가 cue payload + 별도 retrieval key**를 tokenizer로 세어 비가중 평균한 값이다. Read tokens는 reader에 전달한 retrieval evidence text를 1,540문항에 걸쳐 평균한 값이다. 두 열은 임베딩 벡터 bytes, 전체 system/question prompt 또는 생성·구축 API 비용을 나타내지 않는다.

모든 cue 조건의 추가 payload+key 예산은 2,000 tokens다. 동일 상한은 실제 선택 개수나 총 저장량이 같다는 뜻이 아니다. Payload-key control은 Ours의 payload와 순서를 유지하면서 별도 question key만 제거하므로 해당 key 저장량 감소를 그대로 반영한다.

기존 temporal ablation은 반복 날짜 헤더 길이도 바뀌어 통제가 섞였다. 이번 temporal 비교는 canonical 헤더와 메모리 단위 identity/order를 고정하고 본문 상대시간 normalization만 제거한다. 구체적인 분해 및 재현 근거는 TEMPORAL_ACCOUNTING_AUDIT_KO.md에 있다.

## 남아 있는 범위

새 전체 F1·semantic Accuracy는 각 native 결과가 완성되어 검증되기 전까지 미정이다. 과거 GPT-4o의 75.1299%를 GPT-4o-mini 열에 옮겨 적지 않는다. LongMemEval의 핵심 전체 ablation은 아직 실행되지 않았으며, 이번 LoCoMo 결과만으로 그 지적까지 해결했다고 주장하지 않는다.
