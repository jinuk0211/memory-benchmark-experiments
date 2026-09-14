# GPT-4o 실제 채점 결과

제공된 API 인증으로 **gpt-4o-2024-08-06을 실제 호출**했다. 공식 LongMemEval의 고정 프롬프트·유형 분기·모델·요청 파라미터·최종 판정식을 사용했다. 다음 값은 앞선 Codex 잠정 수동 판정을 대체한다.

| 방법 | 정답/평가 문항 | GPT-4o 정답률 |
|---|---:|---:|
| Full-context | 9/12 | 75.00% |
| 우리 Seed | 11/12 | 91.67% |
| 우리 r40 | 7/12 | 58.33% |
| 우리 Refined | 6/12 | 50.00% |
| 우리 Seed-parent | 11/12 | 91.67% |
| LangMem | 0/1 | 0.00% |
| SimpleMem | 1/1 | 100.00% |
| LightMem | 1/1 | 100.00% |

자체 네 방법은9/10 Qwen3.5-9B+MiniLM 조건이다. Full-context와 자체4방법은 같은12개 질문이다. LangMem·SimpleMem·LightMem은 통근시간 질문1개만 확보됐다. 이 세 행은1문항 결과이며12개·전체500의 정확도가 아니다. 모든12개가 이전에 노출된DEV이고 원래 생성 프롬프트·출력 길이·하드웨어는 통제되지 않았으므로 일반화 우위나 통계적 우위를 주장하지 않는다.

## 통근시간1개

- Full-context: **No**. 45분을 다른 사용자 정보라고 하며 질문 사용자의 통근시간은 모른다고 답했다.
- LangMem: **No**. Unknown.
- SimpleMem: **Yes**. 편도45분.
- LightMem: **Yes.** 편도45분과 출처 설명.
- 자체4방법: **Yes**. 편도45분.

LightMem의 이전 F1 19.05는 설명 길이에 따른 점수였다. 실제 GPT-4o 판정은 정답이다.

## 예전 모든 모델·embedding 조합

아래 각 방법은 동일12문항이다. 동일한 전체 API 요청은 방법 사이에 판정을 공유했다.

| 실행 조건 | Seed | r40 | Refined |
|---|---:|---:|---:|
| 9/8 Qwen3.5 + Qwen embedding | 83.33% | 58.33% | 58.33% |
| 9/8 Gemma4 + Qwen embedding | 58.33% | 58.33% | 58.33% |
| 9/10 Qwen3.5 + MiniLM | 91.67% | 58.33% | 50.00% |

Seed-parent는9/10 MiniLM에서11/12=91.67%다.

## 앞선 수동 판정에서 달라진 점

수동 검토한63개 방법·문항 쌍 중 **5개 판정**이 달랐다.

- Seed와Seed-parent의 영상 답변은 링크가 없어도 정확한 제목과Mayo Clinic을 담아 GPT-4o가 **Yes.**로 판정했다.
- r40와Refined의2주 질문에 대한3 weeks는 GPT-4o가 **No**로 판정했다. 공식 프롬프트에는 ±1 허용 문구가 있으나, 실제 judge가 항상 동일하게 적용한다고 가정하지 않고 반환값을 그대로 기록했다.
- Full-context의2개월 질문에 대한 긴3개월 답변은 **No**로 판정했다.

Full-context가 필요한 피연산자를 제시하면서 계산을 거부한 두 답변은 GPT-4o가 모두 **Yes.**로 판정했다. 영상 제목만 있는 r40/Refined 답변은No.였고, 제목과by the Mayo Clinic을 덧붙인Seed/parent는Yes.였다. 이는 이번단일judge실행의 관측이며 판정 이유를 따로 요청하지 않았으므로, 위 차이에 대한 내부 이유를 확정적으로 설명하지 않는다.

## 문항별 실제 판정

| question_id | Full-context | Seed | r40 | Refined | Seed-parent |
|---|---|---|---|---|---|
| cc6d1ec1 | 오답 | 정답 | 정답 | 정답 | 정답 |
| 157a136e | 오답 | 정답 | 오답 | 정답 | 정답 |
| e4e14d04 | 정답 | 정답 | 오답 | 오답 | 정답 |
| e6041065 | 정답 | 정답 | 정답 | 오답 | 정답 |
| d6062bb9 | 정답 | 오답 | 오답 | 오답 | 오답 |
| gpt4_b5700ca9 | 정답 | 정답 | 정답 | 오답 | 정답 |
| 118b2229 | 오답 | 정답 | 정답 | 정답 | 정답 |
| 27016adc | 정답 | 정답 | 오답 | 오답 | 정답 |
| 8cf51dda | 정답 | 정답 | 정답 | 정답 | 정답 |
| 41275add | 정답 | 정답 | 오답 | 오답 | 정답 |
| 4baee567 | 정답 | 정답 | 정답 | 정답 | 정답 |
| gpt4_7de946e7 | 정답 | 정답 | 정답 | 정답 | 정답 |

## 실행과 실제 토큰

- 기존 답변135개, 고유 API 요청54개, 성공 응답54개, HTTP오류0.
- 반환 모델54개 모두gpt-4o-2024-08-06, finish_reason54개 모두stop.
- 실제 보고된 입력10,292토큰 + 출력82토큰 = **10,374토큰**.
- 판정 텍스트: Yes17개, No9개, Yes.16개, No.12개.
- 모델 입력에는 방법명·과거 점수·수동 판정을 넣지 않았다.
- 질문·정답·hypothesis를 포함한 전체payload가 같은 경우만 중복 제거했다. 원본 답변은 수정하지 않았다.
- API키를 파일이나결과에 저장하지 않았고 숨김stdin으로 실행 프로세스에만 전달했다.

초기 실행기는 yes/no만 허용하는 파서를 사용해 마침표가 붙은28응답을미완료로 표시했다. 원시 응답을 보존한 채 고정 upstream의 실제 판정식 `'yes' in eval_response.lower()`(앞서strip)를 적용해 집계했다. **추가 API 호출이나 선택적 재채점은 없었다.** 초기results.json은원본진단으로보존하고 최종값은official_results.json으로구별한다. RUN_JUDGE.ps1을 다시 실행할 필요가 없다.

별도 전송·재개 실행기를 사용했으며 upstream CLI 자체를 실행했다는 뜻은 아니다. 프롬프트·모델·요청 파라미터·판정식은 고정 upstream과 동일하다.

## 근거

- [최종 점수와 실제 사용량](official_results.json)
- [문항별135개 CSV](official_scores.csv)
- [실제54개 API 응답·사용량 영수증](judge_responses.jsonl)
- [고정 요청54개](requests.json)
- [집계 재현 코드](finalize_official_results.py)
- [공식 upstream 기준](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)
