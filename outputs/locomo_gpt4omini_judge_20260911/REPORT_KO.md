# LoCoMo 공통 GPT-4o-mini judge 재평가

표의 9개 방법이 생성해 둔 답변을 같은 1,540문항에서 재채점했다. 답변을 다시 생성하거나 수정하지 않았다.
지표는 LightMem 공식 배포 LoCoMo judge의 CORRECT 비율(ACC)이다. 기존 LoCoMo token F1과 별도 지표이며, F1 값을 대체하지 않는다.

| 방법 | 기존 F1 (%) | Judge ACC (%) | 정답 / 1,540 |
|---|---:|---:|---:|
| Full-context reference | 55.17 | 71.36 | 1099 |
| Mem0 | 36.50 | 54.48 | 839 |
| A-MEM | 36.85 | 51.88 | 799 |
| LangMem | 29.93 | 41.82 | 644 |
| SimpleMem | 38.19 | 54.68 | 842 |
| LightMem | 40.33 | 65.45 | 1008 |
| HiGMem | 40.33 | 55.84 | 860 |
| E-Mem | 56.96 | 77.14 | 1188 |
| Our method | 55.53 | 72.73 | 1120 |

| 방법 | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---:|---:|---:|---:|
| Full-context reference | 71.28 | 43.93 | 25.00 | 87.16 |
| Mem0 | 61.35 | 27.73 | 40.62 | 63.97 |
| A-MEM | 48.23 | 38.94 | 34.38 | 60.05 |
| LangMem | 47.52 | 26.79 | 39.58 | 45.90 |
| SimpleMem | 48.94 | 43.30 | 39.58 | 62.66 |
| LightMem | 63.12 | 60.44 | 38.54 | 71.22 |
| HiGMem | 47.52 | 38.01 | 31.25 | 68.25 |
| E-Mem | 72.34 | 67.91 | 50.00 | 85.37 |
| Our method | 60.28 | 65.73 | 27.08 | 84.78 |

## 이번 결과가 보여주는 것

- Our–LightMem 격차는 F1 15.20 pp, judge ACC 7.27 pp이다. 같은 저장 답변도 지표에 따라 격차가 크게 달라지므로, F1 15.20 pp를 지표와 무관한 성능 차이로 해석하면 안 된다.
- Full context의 judge ACC는 71.36%이고 Our method는 72.73%이다. 전체 대화가 들어가는 full context가 여전히 강한 reference라는 관찰은 유지된다. Our와의 차이의 대화 단위 bootstrap 구간은 0을 포함한다.
- E-Mem의 judge ACC는 77.14%로 Our보다 4.42 pp 높다. 이 재평가에서 Our가 가장 높은 메모리 방법이라고 주장할 수 없다.
- Full context는 Single-hop 87.16%이며, 이 유형은 841/1,540문항을 차지한다. Temporal은 43.93%여서 모든 유형에서 동일하게 강한 것은 아니다.
- 기존 F1의 공식 코드 재계산과 저장된 점수는 일치했다. 이번 변화는 F1 구현 오류를 발견한 것이 아니라 채점 지표의 차이를 확인한 것이다. F1과 judge ACC를 함께 보고해야 비교 의미가 분명하다.

## 비교 조건과 해석

- 질문·정답·저장 답변을 원문 그대로 채점했다. 모델명, 방법명, F1 점수는 judge 입력에 포함하지 않았다.
- 고정 commit의 공식 프롬프트, gpt-4o-mini, temperature=0, JSON 출력 설정을 사용했다. 반환된 모델 버전은 실행 기록에 보관했다.
- 질문·정답·답변을 포함한 전체 API payload가 같은 경우 한 판정을 공유했다. 10,624개 고유 판정을 13,860개 방법×문항에 연결했으며, 각 방법의 분모는 1,540이다.
- API 오류를 오답으로 바꾸거나 누락 문항을 분모에서 빼지 않았다. 모든 문항이 채점된 뒤 집계했다.
- 동일한 judge를 적용해도 기존 답변 생성 단계의 프롬프트·출력 한도·검색 조건 차이는 남는다. 점수 변화만으로 메모리 알고리즘의 우열이나 F1 구현 오류를 단정할 수 없다.
- temperature=0도 서버의 완전한 결정성을 보장하지 않는다. 이번 결과는 고유 답변별 1회 판정이며 반복 채점 변동성을 추정하지 않았다.

## Our method와의 차이

- Full-context reference 대비 +1.36 pp; 대화 단위 bootstrap 95% 구간 [-1.00, +3.64] pp.
- E-Mem 대비 -4.42 pp; 대화 단위 bootstrap 95% 구간 [-8.19, -0.79] pp.
- LightMem 대비 +7.27 pp; 대화 단위 bootstrap 95% 구간 [+4.01, +10.32] pp.
- HiGMem 대비 +16.88 pp; 대화 단위 bootstrap 95% 구간 [+13.23, +20.39] pp.

## 실행 기록

- 반환 모델: {"gpt-4o-mini-2024-07-18": 10624}
- API 사용량 기반 비용: USD 0.684138. 계정 청구서 금액과는 구분한다.
- 입력/출력 토큰: 4,305,869 / 63,763.
- 공식 소스: https://github.com/zjunlp/LightMem/blob/8449d574df6bae1bdf3314a1564da65e2f37e046/experiments/locomo/llm_judge.py
- 원본 입력 및 해시: input.jsonl, input_manifest.json. 공식 원본 대조: protocol_verification.json.
- 전체 문항 판정: scores.jsonl. API 응답: judge_cache.jsonl. 비용/오류: api_events.jsonl.
- 검증 결과 및 비교 통계: analysis.json. 표: comparison.csv, table_judge.tex.
