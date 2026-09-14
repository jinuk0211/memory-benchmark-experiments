# 동일50문항 · GPT-4o-mini 실제 채점

네 방법이 모두 답변한52개 중 **원본 데이터 순서의 첫50개**를 점수를 보기 전에 고정했다. 제외된2개 ID는86f00804와8e9d538c다. 선택된50개는 모두**single-session-user(한 세션의 사용자 정보)** 유형이다.

| 방법 | 정답/문항 | GPT-4o-mini 점수 | 평균 구축시간 | 평균 질의시간 | 구축·질의 관측 LLM 토큰 합 |
|---|---:|---:|---:|---:|---:|
| Full-context | 45/50 | **90%** | 0.00초 | 35.71초 | 5,788,016 |
| LangMem | 36/50 | **72%** | 390.73초 | 0.42초 | 13,172,012 |
| SimpleMem | 37/50 | **74%** | 1766.11초 | 37.08초 | ≥ 11,835,645 |
| LightMem | 44/50 | **88%** | 836.96초 | 2.55초 | ≥ 2,829,491 |

- 모든 행의 점수·시간·토큰은 정확히 같은50개ID 기준이다. 시간과 구축·질의 토큰은 기존 실행의 계측 기록이며 이번에 답변을 다시 생성하지 않았다.
- 총토큰은 입력+출력, 보고된 재시도 포함이다. 임베딩과이번judge토큰은 제외했다.
- SimpleMem476호출, LightMem2호출의 사용량이 누락돼 두 토큰 합은하한(≥)이다. 누락은소비0이 아니고 distinctGPUcrash수도아니다.
- LightMem의질의시간은prediction파일mtime−construction파일mtime대용치다. 다른 방법의직접QA wall과측정경계가같지는않다. 모든시간에원래GPU·큐대기차이가반영될수있다.
- Full-context45개와LightMem44개의차이는이번50개에서1문항이다. 전체LongMemEval500개 또는다른문항유형의성능으로확대하지않는다.

## Judge 실행

- 실제 모델: **gpt-4o-mini-2024-07-18**.
- 고정된공식LongMemEval get_anscheck_prompt, n=1, temperature=0, max_tokens=10.
- 전체200답변. 질문·정답·원문답변을포함한전체API payload가완전히같은경우만판정을공유하여**182회실제호출**.
- 모델에방법명·과거점수는보내지않았다. 이전GPT-4o판정캐시를mini판정에재사용하지않았다.
- 성공182/182, 미완료0, 사용량누락0.
- 이번judge입력 **28,450토큰** + 출력 **182토큰** = **28,632토큰**.
- 정답판정은공식표현식 `'yes' in eval_response.lower()`를적용했다. 완결된yes/no및정상마침표응답만허용했고모델·finish_reason을검증했다.
- 자체전송/재개실행기를사용했으며공식CLI전체를그대로실행했다는뜻은아니다. 프롬프트·모델·샘플링·판정식은공식mini조건과같다.

## 선택 및 자료 범위

52개원시답변은2026-09-12 18:53:12KST GPU서버의완료파일을읽어고정한스냅샷이다. 이번작업은이미완료된예측의judge채점이며GPU실험·큐·서비스를변경하지않았다.

우리Seed/r40/Refined/Seed-parent의기존답변은이50개중1개만겹친다. 앞선12문항GPT-4o점수를이50문항GPT-4o-mini표와같은조건인것처럼합치지않았다.

## 산출물

- [4방법 요약 CSV](summary.csv)
- [200개 답변별 판정 CSV](scored_answers.csv)
- [실제 점수·judge 토큰 JSON](results.json)
- [동일50개 효율 계측·누락 내역](matched50_efficiency.json)
- [선택50개 ID와 제외2개](selection.json)
- [실제182개응답 영수증](judge_responses.jsonl)
- [공식 upstream 채점 소스](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)
