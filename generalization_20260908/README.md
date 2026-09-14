# Frozen-method validation and generalization — 2026-09-08

사용자 확인(2026-09-08): **원래 기준 모델은 Qwen3.5-9B이다. Qwen3-8B로 실행한 것은 실수였다.** Gemma는 모델 일반성 검증용이다. 이 확인이 과거의 잠정적인 source/target 구분을 대체한다.

고정 방법은 기존 실수로 실행한 8B 실험에서 선택된 `s_parent_single_2000`이다. 그 개발 이력은 보존하되, Qwen3.5-9B에서 검증된 방법이라고 소급해서 주장하지 않는다. 현재 우선순위는 Qwen3.5-9B로 writer, source-only utility scorer, reader를 모두 다시 실행해 seed / r40 / refined 점수를 확인하는 것이다. 이후 같은 알고리즘의 Gemma 및 LongMemEval 일반성을 검증한다. Target 결과에 따른 규칙·threshold·방법 재선택은 하지 않는다.

원본 코드와 설정은 source/ 및 frozen-refinement-*.tgz에 보관했다. 원격 작업 공간은 /workspace/generalization_20260908이다. Qwen3-8B 평가와 이전 queue는 05:00 UTC에 중단했으며, 추가 8B 추론이나 backend bridge는 실행하지 않는다. Mistral 역시 제외했다.

## Frozen comparisons

session10 → audit → dialogue residual seed; 기존 calendar anchoring/filter/four-turn fusion r40; source-only grounded probes 및 likelihood utility; singleton/full parent routing과 +2000 stored-token knapsack을 고정한다. Retrieval은 Qwen3-Embedding-0.6B + BM25 RRF60, top120, read2048, answer96, temperature0, nonthinking이다.

Primary contrast: s_parent_single_2000 − r40_fused_four_turn. Secondary: refined − seed. 기존 heuristic preprocessing도 동결했다. Embedding 모델의 Qwen3 표기는 이번 잘못된 LLM 8B 실행과 별개이다.

## Evaluation populations

- Qwen3.5-9B × LoCoMo: 전체 역할을 다시 실행한다. conv-42/47/50의 원래 이력·QA 배열을 그대로 사용해 category1–4 507문항을 생성한다. 미리 고정한 새 질문377개를 주 분석, 기존 audit100을 보조 분석으로 보고한다. 대화 이력3개 자체는 이전에 노출됐으므로 완전히 미노출된 대화 평가가 아니다.
- Gemma4-E4B-it × LoCoMo: 같은 동결 방법의 전체 역할 모델 일반성 평가.
- 두 모델 × LongMemEval-S cleaned: 먼저 점수 선택에 사용하지 않는 사전고정12문항으로 실행성을 확인하고, 전체500문항 평가로 확장한다. 12문항에는 abstention이 없으므로 일반성 결론에 충분하지 않다.
- LongMemEval 데이터 SHA256: d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442. 500질문/471 base-qid clusters/30 abstention/23867 sessions. 모든 이력을 보존하고 answer_ ID, has_answer, gold, question_type을 메모리 생성 입력에서 제거한다.

LoCoMo는 원본 category-specific F1와 대화 cluster bootstrap을 사용한다. LongMemEval 공식 정확도는 고정 gpt-4o-2024-08-06 rubric judge가 필요하며 승인된 API 설정은 아직 없다. Token F1는 진단 지표만이다. 누락·중복·미채점 행을 완료된 평가로 취급하지 않는다.

## Current evidence and remaining work

Qwen3.5-9B LoCoMo 평가가 완료됐다. 주 분석377개 F1×100은 seed54.095/r4055.687/refined56.305이며 refined−r40는 +0.618점(95% CI[−0.091,+1.664])이다. 보조 audit100에서 refined−seed는 −1.027점이다. Gemma도 완료됐다. 주 분석377개 r4050.558/refined50.741, 차이+0.183점(CI[−2.197,+1.499]); audit100은−1.473점, 전체507은−0.118점이다. 추가 utility 이득의 안정적인 모델 일반성은 확인되지 않았다. 두 모델 비교는 modern/LOCOMO_MODEL_TRANSFER_RESULTS.md에 있다. 기존 8B audit100의 refined − r40는 +0.444pp, 95% 대화 cluster CI [-1.001,+2.278]pp였으며 이것은 실수로 수행된 개발 이력일 뿐 Qwen3.5-9B 성능 증거가 아니다.

두 모델 가중치와 modern runtime 설치·검증이 완료됐고, 실제 Qwen3.5 GPU smoke가 통과했다. Qwen3.5 LoCoMo 세 방법 각각507개 평가가 완료됐고,1521개 F1과 고정 모집단·집계를 독립적으로 재검증했다. Gemma도 GPU 검증을 통과했다. 두 실제 tokenizer로 LongMemEval 500개 전체를 점검해 session10/source-QA/raw-fallback 시나리오 모두65536 한도에서 overflow0을 확인했다. 최대 입력+출력예산은 Qwen3.5 39120, Gemma38905토큰이다. 이 CPU 점검은 GPU 메모리 적합성이나 모든 생성 메모리 경우의 상한을 보장하지 않는다. Full500은 writer capacity65536과 최종 메모리에 사용되지 않는 계산의 제거를 별도 protocol로 기록한다. Reader/scorer8192, memory/read/output 예산은 고정한다. 실제 Qwen3.5 utility138개를 재사용한15개 대화/예산 비교에서도 계산 생략 후 선택 결과가 동일했다. 이는 기록된 출력을 고정한 재생 검증이며 새 GPU 배치의 수치적 동일성을 주장하지 않는다.

현재 실행 상태와 다음 단계는 modern/NEXT.md, 모델/런타임은 modern/README.md, full500 protocol은 full_transfer/에 기록한다. 목표는 실제 Qwen3.5-9B 성능 확인과 모델·데이터 일반성 판단까지 계속 active이다.

전체500문항 큐(memory-transfer-full-queue)는 등록·시작됐으며 현재 modern 실행의 성공 종료를 기다린다. 새로운 full500 plan과 GPU 추론은 아직 시작하지 않았다. 실행 전65k GPU 검증과 저장 공간 확인을 거친다.

07:42UTC 현재 Qwen3.5 LongMemEval12의 메모리 생성이 진행 중이다. 공식 full500 집계기는 로컬 검토와8개 테스트를 통과했으며, 결과를 읽어와 로컬에서 실행한다. 집계기 원격 전송은 자동 승인 검토가 거절해 재시도하지 않았다. GPU 큐는 영향을 받지 않았고 공식채점API는 여전히 미설정이다.
