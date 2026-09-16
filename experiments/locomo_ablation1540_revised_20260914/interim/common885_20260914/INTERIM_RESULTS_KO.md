# LoCoMo 중간 결과 — 공통 완료 885문항

최종 1,540문항 결과가 아니다. 모든16개 설정이 공통으로 완료한6개 대화의885문항만 비교했다.
현재 1차 실험이며, source marginal utility를 수정한 후속 실험의 결과가 아니다.
중간 결과에 대한 신뢰구간이나 p-value를 계산하지 않았으며 유의성 판단에 사용하지 않는다.

| 설정 | F1 (%) | 설정 − Ours (pp) |
|---|---:|---:|
| Ours | 59.2127 | +0.0000 |
| w/o cues (base memory) | 59.4980 | +0.2853 |
| w/o omission audit | 59.2358 | +0.0231 |
| w/o temporal normalization | 58.3537 | -0.8590 |
| + evidence binding | 55.0716 | -4.1411 |
| Random cue selection (10 seeds) | 59.0114 | -0.2013 |
| Payload retrieval keys | 59.4509 | +0.2382 |

Random10 seed F1 범위: 58.2576–59.6615; seed 간 표본 표준편차: 0.4442pp.

표본은 점수와 무관하게 snapshot 시점에 모든 설정의 예측 파일이 완료된 대화의 교집합으로 선택했다.
질문·정답은 canonical dataset에서, F1은 최종 reporter와 같은 고정 scorer에서 다시 계산했다.
