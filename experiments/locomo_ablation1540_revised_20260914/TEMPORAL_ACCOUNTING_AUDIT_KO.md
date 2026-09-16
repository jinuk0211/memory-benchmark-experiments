# Temporal ablation 저장량 감사

- 작성일: 2026-09-14
- 범위: 기존 LoCoMo 300 실험에서 생성한 10개 히스토리의 메모리. 해당 `no_binding` 메모리는 후속 1,540문항 실험에서도 동일하게 사용되었다.
- 상태: 저장량과 construction 재현 감사 완료. 새 ablation의 답변 생성 또는 F1 측정 결과를 보고하는 문서가 아니다.

## 결론

기존 `w/o temporal normalization`의 저장량 64,294.5 tokens는 **토큰 집계 오류가 아니다**. 해당 분기가 상대시간 본문 annotation뿐 아니라 반복되는 세션 날짜 헤더의 압축도 함께 제거했다. 따라서 논문의 “단순히 시간 annotation을 제거한 ablation”이라는 해석에는 통제 조건의 혼합이 있다.

기존 코드의 `anchor_units()`는 다음 두 작업을 동시에 수행한다.

1. `(Session date: 1:56 pm on 8 May, 2023)`를 `(Recorded on: 8 May 2023)`로 바꾼다.
2. 본문 `yesterday` 등을 `7 May 2023 [original relative expression: yesterday]`로 바꾼다.

날짜 헤더는 모든 fact와 dialogue block에 반복된다. 헤더 압축의 절약량이 본문 annotation의 증가량보다 커서 normalization을 끄면 총 저장량이 증가한다.

## 동일 메모리에서의 정확한 토큰 분해

모든 값은 히스토리 10개에 대한 비가중 평균이며, cue를 제외한 base 저장량이다.

| 단계 | 평균 tokens | 직전 단계 대비 |
|---|---:|---:|
| 기존 no_temporal base | 62,347.9 | — |
| 동일 단위와 본문에서 날짜 헤더만 canonical 형식으로 변경 | 58,788.3 | −3,559.6 |
| 같은 단위 경계에서 본문 상대시간 normalization 추가 | 59,960.3 | +1,172.0 |
| 실제 normalized fusion의 fact 배치/단위 경계 반영 | 59,960.6 | +0.3 |

즉 `62,347.9 − 3,559.6 + 1,172.0 + 0.3 = 59,960.6`이다. 본문 normalization을 계산할 때 단위 경계는 기존 no_temporal과 동일하게 고정했다. 마지막 차이는 normalized 문자열로 fusion의 640-token cap을 다시 적용할 때 발생하는 구조 차이다.

| 히스토리 | 기존 no_temporal base | 헤더만 변경 | 본문 normalization, 기존 단위 경계 | 실제 normalized base | 헤더 변화 | 본문 변화 | 단위 배치 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conv-26 | 46,433 | 44,101 | 45,329 | 45,329 | −2,332 | +1,228 | 0 |
| conv-30 | 38,839 | 36,459 | 37,248 | 37,248 | −2,380 | +789 | 0 |
| conv-41 | 71,143 | 67,283 | 69,274 | 69,274 | −3,860 | +1,991 | 0 |
| conv-42 | 63,831 | 60,070 | 61,340 | 61,340 | −3,761 | +1,270 | 0 |
| conv-43 | 73,514 | 69,453 | 70,474 | 70,474 | −4,061 | +1,021 | 0 |
| conv-44 | 69,069 | 65,331 | 65,789 | 65,791 | −3,738 | +458 | +2 |
| conv-47 | 70,302 | 66,139 | 67,864 | 67,864 | −4,163 | +1,725 | 0 |
| conv-48 | 68,460 | 64,459 | 65,647 | 65,647 | −4,001 | +1,188 | 0 |
| conv-49 | 54,995 | 51,648 | 52,741 | 52,741 | −3,347 | +1,093 | 0 |
| conv-50 | 66,893 | 62,940 | 63,897 | 63,898 | −3,953 | +957 | +1 |
| 평균 | 62,347.9 | 58,788.3 | 59,960.3 | 59,960.6 | −3,559.6 | +1,172.0 | +0.3 |

양쪽 모두 dialogue block 수는 전체 2,739개이다. fusion cap 때문에 개별 residual fact는 no_temporal 전체 5개(conv-44 4개, conv-50 1개), normalized 전체 2개(conv-44 2개)이다. fact 내용이 누락된 것이 아니라 fused payload 내부와 개별 residual unit 사이에서 저장 위치가 달라진다.

## Cue를 포함한 기존 표의 복원

| 기존 arm | Base tokens | Cue payload | 추가 question-key tokens | Cue 총 tokens | 전체 tokens |
|---|---:|---:|---:|---:|---:|
| ours, with binding | 59,960.6 | 1,837.4 | 113.5 | 1,950.9 | 61,911.5 |
| no_temporal, with binding | 62,347.9 | 1,837.4 | 109.2 | 1,946.6 | 64,294.5 |
| no_binding | 58,206.7 | 1,666.7 | 243.0 | 1,909.7 | 60,116.4 |

따라서 기존 no_temporal의 전체 저장량 증가는 `+2,387.3` base tokens와 `−4.3` cue tokens의 합인 **+2,383.0 tokens**이다. 모든 cue arm의 추가 저장량은 동일한 상한 2,000 tokens 이내이다. 상한은 동일하지만 실현된 cue 저장량이 동일하다는 뜻은 아니다.

저장 비용은 각 unit의 `ntok(text)`와, `index_text != text`인 경우의 `ntok(index_text)`를 더한 값이다. 동일 문자열이 여러 unit의 payload에 저장되면 각각 센다. base payload, cue payload, 별도 retrieval key가 분리된 accounting이며, 임베딩 벡터의 bytes나 모델 실행 전체 prompt/completion 사용량을 뜻하지 않는다.

## 구성요소별 평균 저장량과 개수

| Arm | 구성요소 | 평균 units | Payload tokens | 별도 key tokens |
|---|---|---:|---:|---:|
| ours, with binding | fused_dialogue | 273.9 | 59,954.2 | 0.0 |
| ours, with binding | residual fact | 0.2 | 6.4 | 0.0 |
| ours, with binding | parent_routed_evidence | 11.4 | 1,837.4 | 113.5 |
| no_temporal, with binding | fused_dialogue | 273.9 | 62,327.2 | 0.0 |
| no_temporal, with binding | residual fact | 0.5 | 20.7 | 0.0 |
| no_temporal, with binding | parent_routed_evidence | 11.0 | 1,837.4 | 109.2 |
| no_binding | dialogue_block | 273.9 | 49,015.5 | 0.0 |
| no_binding | fact | 302.1 | 9,191.2 | 0.0 |
| no_binding | parent_routed_evidence | 24.4 | 1,666.7 | 243.0 |

최종 Ours로 사용하는 no_binding의 base는 **평균 576.0 units, 58,206.7 tokens**이다. Cue까지 포함하면 평균 600.4 units, 60,116.4 tokens이다.

## 새 no_binding 기준 temporal ablation의 통제 조건

모든 arm에서 같은 canonical 날짜 헤더를 공통 전처리로 사용하고, **본문의 `calendar_anchor(body, date, 'month')`만 토글**한다. Fact 추출 결과, social filter, dialogue block size 4/overlap 2, 단위 순서, source IDs와 binding 제거 상태를 고정한다. 기존의 상호 배타적 `arm == 'no_binding'` / `arm == 'no_temporal'` 분기는 이 조합을 표현하지 못하므로 독립 설정이 필요하다.

이 조건으로 기존 audited extraction을 재구성한 결과:

| Base 조건, cue 미포함 | 평균 units | 평균 tokens |
|---|---:|---:|
| no_binding + 상대시간 본문 normalization | 576.0 | 58,206.7 |
| no_binding + 상대시간 본문 normalization 제거, 동일 날짜 헤더 | 576.0 | **57,034.7** |
| 차이 | 0.0 | **−1,172.0** |

히스토리별 base unit 수는 `[379, 381, 626, 606, 661, 605, 679, 652, 543, 628]`이며, 위 표의 두 조건에서 각 unit의 kind, sources와 순서를 모두 대조했다. **57,034.7은 같은 기존 추출 결과와 동일 단위 구성에 대한 base-only accounting invariant**이다. 새 writer/extractor 또는 다른 block 구성에는 자동으로 적용되지 않는다.

Cue routing과 selection을 다시 실행하면 본문 길이 및 최소 parent-cover 비용 변화로 선택된 cue가 달라질 수 있다. 새 전체 저장량을 57,034.7에 과거 cue 저장량을 임의로 더해서 작성하면 안 된다. 동일 selection 규칙과 2,000-token 상한을 적용한 후 새 실측 cue 저장량을 기록해야 한다.

표의 행 이름은 `w/o relative-time normalization`으로 구체화하고, canonical date headers는 모든 arm에서 공통이라는 점을 방법 또는 caption에 명시하는 것을 권장한다. 기존 300문항 no_temporal F1은 with-binding 및 긴 날짜 헤더 조건에서 나온 값이므로 새 arm의 F1로 재사용할 수 없다.

## 검증 방법과 재현 환경

- 감사 입력: `D:\MemoryData\experiments\locomo_ablation300_20260913\collected\run\memories`, `locks`, `construction`, `prepare_0/1\audited`.
- 고정 source sessions/plans: 같은 실험의 `package\input\source_sessions.json`, `plans.json`.
- 기존 Python: `C:\Users\tgc04\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe`.
- Tokenizer 라이브러리: `tokenizers==0.23.1`.
- Qwen model revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (`environment.json` 기록).
- 로컬 tokenizer: `D:\MemoryData\model_staging\qwen35_9b_c202236\tokenizer.json`.
- Tokenizer SHA-256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.

읽기 전용 PowerShell/Python 실행으로 원본 코드 함수를 import하고 다음 절차를 적용했다. 별도 package 설치나 원본 파일 변경은 없었다.

```powershell
Get-FileHash -LiteralPath 'D:\MemoryData\model_staging\qwen35_9b_c202236\tokenizer.json' -Algorithm SHA256
$audit | & 'C:\Users\tgc04\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' -
```

`$audit`의 핵심 계산은 다음과 같다. `r`은 기존 실험 루트이며 `ntok`은 모든 단계에서 동일하다.

```python
from tokenizers import Tokenizer
from types import SimpleNamespace
from budgeted_evidence import storage_cost
import run_ablation as runner

tok = Tokenizer.from_file(
    r'D:/MemoryData/model_staging/qwen35_9b_c202236/tokenizer.json'
)
ntok = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
rt = SimpleNamespace(ntok=ntok)

# cid마다 ours / no_temporal / no_binding의 전체 메모리를 원본 lock과 대조.
assert sum(storage_cost(u, ntok) for u in memory) == lock['stored_tokens'][arm]

# 저장된 audited extraction에서 원본 backend 함수를 그대로 재실행.
seed = runner.core.build(rt, sessions, 'dialogue_residual', audited)
reconstructed = runner.backend(rt, sessions, seed, plans, arm)
assert reconstructed == [
    u for u in memory if u['kind'] != 'parent_routed_evidence'
]
```

1. 히스토리 10개 × arm 3개에 대해 **저장량 lock 30개 항목이 정확히 일치**했다. 이는 lock 파일 30개가 아니라, 히스토리별 lock 파일 10개 내의 arm별 저장량 비교 30회이다.
2. 같은 30개 backend 메모리를 원본 `backend()`로 재구성하여 **JSON unit list 전체가 정확히 일치**함을 확인했다.
3. 기존 no_temporal base의 각 unit에서 `Session date` 헤더를 같은 session의 canonical 날짜로 치환해 단위 경계 고정 상태의 header-only 비용을 계산했다.
4. 이어서 같은 unit 본문에 원본 `calendar_anchor(..., 'month')`를 적용하고, 실제 normalized base와 비교하여 fusion 구조 차이를 분리했다.
5. 새 no_binding 제어군에서는 원본 audited facts와 동일 social filter, 원본 `dialogue_blocks(sessions, 4, 2)`를 사용했다. 두 조건의 kind/sources/order가 같음을 확인한 뒤 같은 canonical 헤더와 원본 상대시간 본문을 합쳐 57,034.7 tokens를 계산했다.

## 코드 근거

- [run_ablation.py:85](/D:/MemoryData/experiments/locomo_ablation300_20260913/run_ablation.py:85): 기존 backend의 normalization/no_binding 분기.
- [run_ablation.py:95](/D:/MemoryData/experiments/locomo_ablation300_20260913/run_ablation.py:95): no_temporal의 fusion anchor 비활성화.
- [continuous.py:85](/D:/MemoryData/experiments/locomo_ablation300_20260913/vendor/source/continuous.py:85): 날짜 헤더와 본문 상대시간을 동시에 변경하는 함수.
- [continuous_v2.py:86](/D:/MemoryData/experiments/locomo_ablation300_20260913/vendor/source/continuous_v2.py:86): 초기 `anchor_time` recipe가 사용하는 같은 변환.
- [memory_ops.py:48](/D:/MemoryData/experiments/locomo_ablation300_20260913/vendor/source/memory_ops.py:48): fact-to-block 배치 및 fusion 640-token cap.
- [parent_evidence.py:42](/D:/MemoryData/experiments/locomo_ablation300_20260913/vendor/source/parent_evidence.py:42): parent cover 재계산과 cue 저장 비용.
- [budgeted_evidence.py:12](/D:/MemoryData/experiments/locomo_ablation300_20260913/vendor/source/budgeted_evidence.py:12): payload 및 별도 key 토큰 집계.
- [report_results.py:32](/D:/MemoryData/experiments/locomo_ablation300_20260913/report_results.py:32): 히스토리별 동일 저장량 검증과 비가중 평균.

## 해석의 한계

이 감사는 기존 저장량을 설명하고 수정된 temporal control의 base accounting을 확인한다. 새 arm의 검색 결과, cue 선택, reader 답변, F1, confidence interval 또는 유의성은 측정하지 않았다. 이 문서만으로 normalization의 성능 효과나 revised method의 성능 향상을 주장할 수 없다. 기존 300문항 결과를 새 no_binding 기준의 1,540문항 결과로 재명명해서는 안 된다.
