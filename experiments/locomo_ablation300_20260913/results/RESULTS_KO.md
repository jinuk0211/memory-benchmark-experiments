LoCoMo300 실측 결과. 저장량은 10대화 평균, 읽기 토큰은 300문항 평균.

| Variant | F1 (%) | Delta (pp) | Stored tokens | Read tokens |
|---|---:|---:|---:|---:|
| Our method | 55.7975 | +0.0000 | 61,911.5 | 2013.8 |
| w/o cues (base memory) | 55.4263 | -0.3712 | 59,960.6 | 2013.9 |
| w/o omission audit | 54.7889 | -1.0087 | 60,369.8 | 2015.4 |
| w/o temporal normalization | 52.2010 | -3.5966 | 64,294.5 | 2010.4 |
| w/o evidence binding | 58.0424 | +2.2449 | 60,116.4 | 2038.8 |
| Random cue selection | 55.1647 | -0.6329 | 61,880.2 | 2014.2 |
| Payload retrieval keys | 55.6363 | -0.1612 | 61,798.0 | 2014.0 |

같은 초기 추출을 공유한 새 구성 결과이며 전체 1,540문항 점수와 다른 부분집합이다.

검증: 7조건 모두 동일한 300개 문항, 총 2,100개 예측을 확보했다. 원본 모델 응답과 모두 대조했으며 빈 답변과 출력 길이 제한 종료는 각각 0개다. 동일한 요청은 캐시를 재사용했고, 고유 생성 기록은 1,640개다.

해석: 시간 정규화 제거 시 F1이 3.5966pp 낮아졌다. Evidence binding 제거 시에는 오히려 2.2449pp 높아졌으므로 이 부분집합 결과가 binding의 성능 이득을 뒷받침하지는 않는다. 나머지 제거·대체 조건은 완전한 방법보다 점수가 낮았지만, 아래 대화 단위 bootstrap 구간을 함께 고려해야 한다.

| Variant | Δ F1의 대화 단위 bootstrap 95% 구간 (pp) |
|---|---:|
| w/o cues (base memory) | [-2.0364, +1.3437] |
| w/o omission audit | [-3.5910, +1.6516] |
| w/o temporal normalization | [-6.4969, -1.0788] |
| w/o evidence binding | [-1.3300, +5.8537] |
| Random cue selection | [-1.9581, +0.7077] |
| Payload retrieval keys | [-2.4930, +2.0520] |

구간은 대화 10개를 단위로 5,000회 paired bootstrap해 계산했다. 단일 메모리 구성, 고정된 300문항, 대화별 단일 무작위 선택 실험이며 미노출 테스트셋에 대한 일반화 결과가 아니다. 무작위 선택의 실현 저장량은 utility 선택과 약간 다르다. Stored tokens는 각 대화를 한 번씩 평균했고, Read tokens와 F1은 문항별 평균이다.

실행 기록: 메모리 생성 후 평가 모델을 처음 초기화할 때 GPU 메모리 해제가 지연되어 한 차례 재시작했다. 이때 예측은 아직 생성되지 않았으며, 평가 재개 시 코드·설정·메모리를 변경하지 않았다. 원래 로그와 재개 로그를 모두 보존했다. 모든 평가가 끝난 뒤 GPU 해제를 확인했다.

재현 자료: `../PROTOCOL.md`, `../package/input/manifest.json`, `../collected/run/protocol.json`, `../collected/run/locks/`, `../collected/run/predictions/`, `scored_predictions.json`, `RESULTS.json`, `VALIDATION.json`, `table.csv`, `table.tex`. 정답은 서버로 전송하지 않고 로컬에서 채점했다.
