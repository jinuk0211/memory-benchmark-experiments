# 55.69 기준선과 기존 메모리 방법 비교 감사

2026-09-08. 저장된 결과와 코드를 확인하고 과거 답변을 같은 문항에서 재채점했다. GPU 추론이나 실험 설정 변경은 수행하지 않았다.

55.69는 refinement 이전의 원시 메모리 점수가 아니다. 현재 Seed부터 사실 추출, 원문 기반 누락 사실 보강, 전체 원문 대화 보존을 거친 자체 메모리다. r40는 여기에 날짜 처리, 사실 필터, 원문 4턴 블록과 사실의 결합을 적용한다. Refined−r40는 이 기존 구성 위에 추가한 source-utility 단계의 증분이다. E-Mem/LangMem/LightMem에 이 단계를 적용한 전후 실험이 아니다.

## 같은 문항으로 재집계

category 1–4의 질문별 category-specific F1 평균 × 100. 기존 세 방법은 원본 예측을 공식 scorer로 재채점했으며 전체 1,540문항 재계산이 과거 보고서와 일치했다. 현대 실험은 저장된 질문별 F1을 모집단 ID로 필터링해 평균을 재계산했다.

| 방법 | 기존 전체 1,540문항 | 동일 3개 대화 507문항 | 현재 주 분석과 같은 377문항 |
|---|---:|---:|---:|
| E-Mem | 56.956713 | 54.906688 | 54.739064 |
| LightMem | 46.595381 | 46.660236 | 45.610083 |
| LangMem | 29.932364 | 32.974622 | 33.070580 |
| 자체 Seed | — | 54.834016 | 54.094655 |
| 자체 r40 | — | 56.070440 | 55.687063 |
| 자체 Refined | — | 56.578669 | 56.305168 |

같은 377문항에서 r40−E-Mem은 +0.948000점, Refined−r40는 +0.618105점이다. 전자는 질문 집단만 맞춘 비교이며 방법 자체의 통제된 인과 효과를 뜻하지 않는다.

## 구성과 해석 범위

- 생성 모델은 양쪽 모두 Qwen3.5-9B다. 임베딩은 기존 all-MiniLM-L6-v2와 현재 Qwen3-Embedding-0.6B가 다르다. 프롬프트와 검색·메모리 구성도 같게 통제하지 않았다.
- modern/run_transfer.py:175–177은 session10 → audit → dialogue_residual을 Seed로 만든다. modern/source/refine.py:196–204,323–324의 residual은 모든 원문 turn을 이전 turn과 묶어 사실 메모리에 추가한다. 누락된 원문만 넣는 구현이 아니다.
- r40는 전체 원문을 4턴/overlap-2 블록으로 구성하고 사실을 붙인다. 고정 체인은 source/portable_parent.py:10–14, 결합 구현은 source/memory_ops.py:48–87에 있다.
- Qwen 출력에 기록된 Seed는 대화당 72,328–79,992 stored tokens, r40는 61,312–68,371 stored tokens다. 실제 reader 문맥 상한은 질문당 2,048 tokens다. 전체 원문을 한 번에 넣는다는 뜻이 아니다.
- 현재 Qwen3.5/377에는 raw-RAG 또는 session10-only arm이 없다. 아무 개선도 하기 전 점수 대비 전체 refinement 효과는 이 표에서 분리할 수 없다.
- 377개는 기록된 pilot/audit에서 제외된 질문이지만 대화 3개는 개발 중 노출됐다. 기억 생성에는 대화만 넘기는 코드 경로가 확인되지만 미노출 대화 일반화 평가는 아니다.
- Qwen의 Refined−r40 95% CI는 [−0.091,+1.664]점이다. 새 마지막 단계의 추가 효과가 확실하지 않다는 결과이며, 모든 refinement가 효과 없다는 결론은 아니다.

## 원본과 검증

모집단: generalization_20260908/locomo_transfer/evaluation_populations.json의 primary_question_population/generated_population. 원본 gold/category: 같은 디렉터리의 input/locomo10.json.

기존 공식 scorer: lightmem_export_20260907/reference/official_locomo_evaluation.py의 normalize_answer, f1_score, f1 및 category dispatch. 현재 source/refine.py:73–95와 해당 범위의 채점 규칙이 일치한다.

- LightMem: D:/MemoryData/lightmem_export_20260907/qa_results_scored.jsonl.
- LangMem: D:/MemoryData/results/locomo/langmem/langmem-complete-50156979-r1-backup.tar.zst 내부 locomo-langmem-composite-50156979-r1/predictions.jsonl.
- E-Mem: D:/MemoryData/results/locomo/emem/emem-r4-r11-closed-artifacts-50156979.tar.gz 내부 locomo-emem-saved-recovery-r11/artifacts/outputs/e_mem/LoCoMo/locomo_qa_official_4cat_in10000000_size1024_shots0_max_samples10_categories_multi_hop-open_domain-single_hop-temporal_k5_chunk1_results.json.
- 현재 결과: D:/MemoryData/generalization_20260908/modern/verified_qwen35/의 seed.jsonl, r40_fused_four_turn.jsonl, s_parent_single_2000.jsonl, predeclared_locomo_results.json.

기존 답변의 ID/gold/category를 원본과 대조했다. LightMem/LangMem 질문 텍스트도 대조했다. E-Mem 최종 1,540답변이 기존 1,538답변을 그대로 보존하고 conv-26_qa74, conv-49_qa38만 추가했음을 확인했다. 중복 백업을 추가 문항으로 세지 않았다. 기존 E-Mem 보고서의 계측/응답 감사 미완료 범위를 이 재채점으로 해소했다고 주장하지 않는다.

`기존 전체 비교` (historical source: `LOCOMO_RESULTS_20260908_0535UTC.md`) · [현재 Qwen 결과 감사](../../../../generalization_20260908/modern/QWEN35_RESULT_AUDIT.md)

