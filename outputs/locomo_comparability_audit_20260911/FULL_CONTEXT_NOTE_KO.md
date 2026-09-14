# Full-context 55.17이 지나치게 높은가?

2026-09-11 추가 확인. **Full-context가 메모리 시스템보다 높다는 사실만으로 평가 오류를 뜻하지 않는다. 원논문 자체에도 같은 패턴이 있다.** 다만 공통 scorer의 정확성과 baseline 재현 조건의 공정성은 별도로 판단해야 한다.

## 원논문에서 직접 확인한 비교

| 원논문·설정 | Full context/text | 해당 방법 | 지표 |
|---|---:|---:|---|
| Mem0, GPT-4o-mini | 72.90 | Mem0 66.88 / graph 68.44 | LLM-as-a-Judge |
| LightMem, Qwen3-30B-A3B-Instruct-2507 | 74.87 | LightMem(.6,768) 71.36 / (.8,1024) 72.60 | judge 기반 ACC |

[Mem0 Table2 및 Section4.3](https://arxiv.org/html/2504.19413v1#S4.SS3)은 full-context가 가장 높은 judge score임을 명시하며 메모리의 품질·비용·지연 절충을 논의한다. [LightMem Table3 및 Section5.1](https://arxiv.org/html/2510.18866v4#S5)은 Qwen에서 full-text가 모든 표기된 memory 구성보다 높은 ACC를 보고한다. 이 값은 우리 표의 token F1과 수치 자체를 직접 비교하면 안 된다.

논문의 성능 향상은 다른 memory/RAG baseline 대비, 특정 모델·설정에서, 또는 비용 대비 품질에서의 향상일 수 있다. 모든 모델에서 모든 기억 시스템이 원문 전체를 읽는 방식보다 정확해야 한다는 주장은 아니다.

## 우리 full-context의 의미

원본 r1/r2 CSV에서 full/full_raw를 고른1,540문항은 빈 답변0, thinking tokens0이다. 대화별 원문 context는21,113–40,191 tokens이며 설정된49,152-token context window 안에 들어간다. 모든369–689개 turn을 합치며 날짜·speaker·텍스트·image caption을 보존한다. prompt 구성에 정답이나 gold evidence를 넣지 않고, 해당 정보는 이후 평가용 metadata로 보관한다.

소스: `full_raw 분기` (historical source: `certmem_primary_deploy_r2/scripts/run_system_v15.py:319`), `전체 문맥 preflight 검사` (historical source: `certmem_primary_deploy_r2/scripts/run_certmem_full.py:86`), `공식 점수` (historical source: `certmem_primary_results_r1/official_primary_scores.json`). 추가 세부 확인은 원시 요청 원장 범위에 따라 제한될 수 있다.

따라서 우리 실험은 full-context가 과거 대화를 창 밖으로 잃어버리는 조건이 아니다. 질문마다 전체 원문을 다시 읽는 강한 reference다. 메모리 경로에는 저장 시 생략·변형, retrieval miss, selector의 근거 제거가 추가될 수 있다. 반대로 잘 설계된 검색·정리는 잡음과 날짜 혼동을 줄여 full-context를 이길 수도 있어 이론적 상한으로 취급하지 않는다.

## 우리 표의 판단

- 공식 category-specific F1 재계산으로55.168900이 재현됐다. 점수 산술/분모 오류 증거는 없다.
- F1은 정답률이 아니다. 예: 정답7 May2023와 답변8 May2023는 날짜가 틀려도 다른 두 단어가 겹쳐 부분 점수를 받을 수 있다.
- Full-context Single-hop71.40이고 Single-hop이841/1540문항(54.61%)이라 전체 평균에 크게 기여한다. 반면 Temporal36.24는 LightMem44.85·Our50.64·E-Mem54.67보다 낮다. Full-context가 모든 능력에서 우월한 패턴도 아니다.
- Qwen3.5-9B로 바꾸면 원논문 GPT/Qwen3에서의 상대 순위가 보장되지 않는다. Full-context는 reader 능력에 직접 의존하지만 메모리 시스템은 writer·JSON/tool 수행·retriever/selector·reader에 모두 의존한다. A-MEM evolution60.68% skip, HiGMem final filter의 evidence 손실이 실제 예다.
- 공통 evaluator가 맞더라도 native reader prompt/cap 차이 및 adapter 변경은 남는다. 그러므로 전체 비교가 완전히 공정하다고 단정하지 않는다.

가장 직접적인 추가 검증은 동일 full-context reader를 각 방법의 저장 retrieval context에 적용하는 paired reader 통제 실험이다. 이어 같은 저장 답변에 공통 judge 또는 맹검 수동 의미 정확도 평가를 더하면 token F1 특유의 부분일치 효과를 확인할 수 있다. 이번에는 추가 GPU/API 추론을 하지 않았다.

추가 preflight 확인: 가장 긴 chat prompt40,311 + 답변상한32 =40,343 tokens로49,152 안에 들어간다. `원본 preflight` (historical source: `certmem-v15-token-preflight-r1.json:6`). 입력 truncation은 false이며 초과 시 runtime은 예외를 발생시킨다. 기존39,000-token guard는 사전 적합성 검사 후 해제하여453문항을 누락 없이 포함했다. r1/r2 결합은 selection_by_score=false다. 출력 length 종료와 입력 truncation은 다르다. r1 full-context885문항 중 출력 length 종료3개가 있었고, r2는 이번 조사에서 policy별 length 수를 분리하지 않았다. capped 답변도 평가 분모에 남겼다.