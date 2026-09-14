# 같은50문항의 GPT-4o 실제 재채점

**Judge는 gpt-4o-2024-08-06이다.** 기존 mini 실행의50개질문과200개원문답변을그대로유지하고, 요청payload에서model만변경했다. 모든182고유요청을이번GPT-4o실행에서새로호출했다.

| 방법 | GPT-4o 정답/문항 | GPT-4o 점수 | 이전 mini 점수(같은50개) |
|---|---:|---:|---:|
| Full-context | 46/50 | **92.00%** | 90.00% |
| LangMem | 37/50 | **74.00%** | 72.00% |
| SimpleMem | 38/50 | **76.00%** | 74.00% |
| LightMem | 45/50 | **90.00%** | 88.00% |

## 기존 우리 방법과 합친 표

| 방법 | GPT-4o 정답/평가 문항 | 정답률 |
|---|---:|---:|
| Full-context | 46/50 | **92.00%** |
| 우리 Seed | 11/12 | **91.67%** |
| 우리 r40 | 7/12 | **58.33%** |
| 우리 Refined | 6/12 | **50.00%** |
| 우리 Seed-parent | 11/12 | **91.67%** |
| LangMem | 37/50 | **74.00%** |
| SimpleMem | 38/50 | **76.00%** |
| LightMem | 45/50 | **90.00%** |

우리4방법은앞선9/10 Qwen3.5-9B+MiniLM의12문항GPT-4o결과를인용했다. 이50문항으로우리방법을새로평가했다는뜻은아니다. Judge모델은같지만문항집합이다르다. 이전Full-context12문항은9/12=75%였으며, 이번46/50=92%는다른문항집합이다.

## 선택과 평가 범위

- 선택은기존과동일한공통52개중canonical데이터순서첫50개다. 제외ID도86f00804,8e9d538c로그대로다.
- 50개모두single-session-user유형이다. 전체LongMemEval500문항점수가아니다.
- 고정공식get_anscheck_prompt와유형분기,temperature0,n1,max_tokens10을사용했다. 방법이름이나이전점수는judge에게보내지않았다.
- 정상yes/no및마침표응답에공식표현식 `'yes' in eval_response.lower()`를적용했다.
- 전송/재개실행기는자체도구이며공식CLI전체를실행했다는뜻은아니다.
- 기존생성단계의시간·토큰은변하지않았다. 같은50개계측값은[기존효율자료](../longmemeval_common50_gpt4omini_20260912/matched50_efficiency.json)를참조한다.

## 실제 호출 검증

- 답변200개, 고유API요청182개, 실제완료응답182개, 누락0.
- 모든반환모델gpt-4o-2024-08-06, 모든finish_reason=stop, 모든status=ok.
- 입력28,450토큰 + 출력273토큰 = **28,723토큰**.
- 원시응답에서공식판정식으로다시계산한50문항점수가results.json과일치한다.
- mini와달라진판정은6개방법×문항쌍이다. 각방법의총정답수는mini보다1개씩늘었다.
- 문항선택·기준정답·원문답변·프롬프트·생성파라미터는전부mini와같고judge모델만다름을검증했다.
- 원본결과와mini파일은보존했다.

## 결과 파일

- [최종결과 JSON](results.json)
- [50문항4방법 요약 CSV](summary.csv)
- [200개 개별 판정 CSV](scored_answers.csv)
- [실제 API 응답 영수증](judge_responses.jsonl)
- [모델 외 변경 없음 검증](MODEL_ONLY_CHANGE_VERIFIED.json)
- [최종 독립 재집계 검증](VERIFICATION.json)
- [기존 우리12문항 GPT-4o 결과](../longmemeval_semantic_dev12_20260912/official_results.json)
