> 완료: 실제 GPT-4o 채점 결과는 [GPT4O_RESULTS_KO.md](GPT4O_RESULTS_KO.md)를 참조하세요. 아래는 실행 전 준비 또는 잠정 판정의 보존 기록입니다.

# LongMemEval DEV12 의미 채점 정정 · 실행 준비

이 문서는 앞선 F1 중심 비교를 정정한다. 현재 공식 GPT-4o 판정은 **아직 실행되지 않았다**. 기존 F1 13.78/70.79는 정답률이 아니며, 방법 간 정확도 차이의 근거로 사용하지 않는다.

## 고정한 채점 기준

- upstream commit: 9e0b455f4ef0e2ab8f2e582289761153549043fc
- [공식 evaluate_qa.py](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)
- 파일 SHA256: ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251
- judge: gpt-4o-2024-08-06, temperature=0, n=1, max_tokens=10.
- 공식 get_anscheck_prompt 함수를 고정 파일의 AST에서 그대로 불러온다. 질문 유형과 _abs 여부에 따라 공식 분기를 사용한다.
- 모델에 보내는 요청에는 방법 이름과 과거 점수를 넣지 않는다. 질문·기준 정답·수정하지 않은 답변만 넣는다.
- 완전히 동일한 API payload만 판정을 공유한다.
- 재개·사용량 기록을 위한 별도 실행기이며 upstream CLI 자체를 실행했다는 주장은 하지 않는다. 공식 프롬프트·모델·요청 파라미터는 같다.
- yes/no 외 응답, 모델 불일치, 잘린 판정은 미완료로 남긴다. upstream의 단순 yes 부분문자열 파서보다 엄격하며, 이 차이를 protocol.json에 기록했다.

공식 규칙은 단어 겹침 F1와 크게 다르다. 긴 설명도 정답을 포함하면 통과할 수 있다. temporal-reasoning은 일·주·개월 등의 ±1 오차를 허용한다. 일반 유형은 정답을 도출할 중간 과정이 모두 담겨 있어도 통과할 수 있다. 따라서 임의의 엄격한 수동 판정률로 F1를 바꾸는 것도 같은 공식 평가가 아니다.

## 확보 범위와 실행 상태

| 대상 | 기존 답변 | 공식 판정 | 기존12 대비 누락 |
|---|---:|---:|---:|
| Full-context | 12 | 대기 | 0 |
| 자체 Qwen/MiniLM Seed·r40·Refined·Seed-parent | 각각12 | 대기 | 0 |
| 자체 이전 Qwen/Qwen embedding 3방법 | 각각12 | 대기 | 0 |
| 자체 이전 Gemma/Qwen embedding 3방법 | 각각12 | 대기 | 0 |
| LangMem | 1 | 대기 | 11 |
| SimpleMem | 1 | 대기 | 11 |
| LightMem | 1 | 대기 | 11 |

총135개 기존 답변과54개 고유 요청을 검증했다. 원본 데이터 SHA, 질문·정답·유형, 각 예측 파일 SHA와 ID를 대조했다. 정답·답변을 고쳐 쓰거나 다시 생성하지 않았다.

진행률이 다른 방법을 각기 다른 분모로 순위화하지 않는다. 모든12개가 판정된 방법만 accuracy_on_same12를 제공한다. 1개 baseline은 해당1개 판정만 보여주고12문항 정확도는 null로 둔다. 누락11개를 오답으로 채우지 않는다.

## 통근시간 문항의 정정

| 방법 | 실제 답변의 의미 | 이번 공식 판정 |
|---|---|---|
| Full-context | 45분을 언급하지만 다른 사용자의 정보라며 자신의 통근시간은 모른다고 답함 | 대기: 숫자 포함과 답변 부정의 충돌을 judge가 판정 |
| LangMem | Unknown | 대기 |
| SimpleMem | 편도45분 | 대기 |
| LightMem | 편도45분과 출처 설명 | 대기 |
| 자체4방법 | 편도45분 | 대기 |

SimpleMem과 LightMem은 같은 통근시간을 답했다. LightMem의 예전 F1 19.05를 정답률19.05% 또는 오답으로 표현하지 않는다. Full-context는 답을 부정하는 문장이 있어 단순한45분 문자열 검사로 정답을 강제하지 않는다.

## 현재 실행을 막는 두 조건

1. **공식 judge 인증 없음.** 프로세스·사용자·시스템 OPENAI_API_KEY가 없고 기존 프로젝트 인증 파일도 없다. 예전 LoCoMo 실행은 stdin/getpass로 받은 키를 메모리에서만 썼다. 과거 계획된 DEV12 공식48판정도 실제 결과/실행 영수증이 없어 재사용할 판정이 없다.
2. **세 baseline의 나머지11예측 없음.** 2026-09-12 16:25~16:26 KST 서버 읽기 확인: LangMem139/500, SimpleMem52/500, LightMem111/500이고 기존12 coverage는 각각1뿐이다. GPU4개는 LangMem이 사용 중이고 Simple/Light는 이후 재개 대기다. 별도 worker를 추가하거나 진행 중 작업을 중단하지 않았다. 동일12를 완성하려면 현재 운영 작업의 다음 배정 순서와 조정해야 한다.

API 인증이 주어지면 기존135답변의54개 요청을 바로 실행할 수 있다. 이는 누락33개 답변의 생성까지 해결한다는 뜻은 아니다.

## 실행

`숨김 입력 실행기` (historical source: `outputs/longmemeval_semantic_dev12_20260912/RUN_JUDGE.ps1`)를 PowerShell 터미널에서 실행하면 API 키를 화면에 표시하지 않고 받는다. 키를 파일이나 명령행 인수에 저장하지 않는다.

```powershell
& 'D:\MemoryData\outputs\longmemeval_semantic_dev12_20260912\RUN_JUDGE.ps1'
```

이미 환경변수가 설정되어 있다면:

```powershell
python -X utf8 'D:\MemoryData\outputs\longmemeval_semantic_dev12_20260912\run_semantic_judge.py' --run
```

실행 없이 입력을 검증하려면 --run을 생략한다. 실제 응답은 judge_responses.jsonl, 방법별·문항별 집계와 토큰은 results.json에 저장한다. 기존 성공 판정은 재사용한다. 원본 입력 변경이나 훼손된 체크포인트는 임의 복구하지 않고 멈춘다.

검증 완료: 원본135행·고유54요청, 정답/질문/유형/해시 일치, Python 코드 리뷰 및 행동 테스트4개 통과. 현 시점 API 호출0·공식 판정0이다.

- [고정 프로토콜](protocol.json)
- [변형하지 않은 답변135개](inputs.json)
- [집계 및 판정 상태](results.json)
