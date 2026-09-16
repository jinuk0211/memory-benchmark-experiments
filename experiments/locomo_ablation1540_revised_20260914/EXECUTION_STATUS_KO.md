# 실행 상태 — LoCoMo 1,540문항

- 사용자가 instance 50986606 환경 구축, 필요한 연구 자료 전송과 평가 실행을 승인했습니다.
- 현재 `locomo-revised-full1540`에서 10개 대화의 1,540문항 × 16개 설정을 평가 중입니다. 최종 F1 표는 전체 문항 완료·검증 후 작성합니다.
- 사전 고정된 300문항 reader 비교는 완료했습니다. Legacy 58.0350173, grounded 59.0255432 F1이며 grounded를 전체 평가 전에 고정했습니다. Open-domain은 이 사전 비교에서 감소했습니다.
- `STATUS.json`, `READER_SELECTION.json`, `PILOT_REPORT_KO.md`에 기계 판독 상태와 검증 결과가 있습니다.
- 전체 평가 원격 출력: `/workspace/locomo_ablation1540_revised_20260914/full_grounded`.
- utility 포화 원인을 다루는 source-only marginal utility 비교를 별도 `locomo_marginal_utility1540_20260914`에서 입력 검증을 완료하고 첫 전체 평가 종료 후 실행 대기 중입니다. 동일 부모 메모리·후보·예산·DP에 실제 base-RAG 대비 source answer likelihood 증가량을 사용합니다. 이 비교도 1,540문항 전체로 평가합니다.
- random 10개 사전 고정 seed를 모두 보고합니다. Conversation cluster bootstrap CI와 paired sign-flip, Holm 보정을 적용합니다. 모든 기존 설정과 실패한 결과를 보존하며, 이미 노출된 전체 데이터에 대한 탐색적 실험임을 명시합니다.

- 세 marginal component의 source 입력과 20개씩 사전 지정한 retrieval parity 확인을 완료했습니다. 별도 Supervisor locomo-components-after-marginal1540가 utility 전체 평가의 정상 종료를 기다린 뒤 세 조건을 모두 1,540문항씩 실행합니다.
- 최종 GPT-4o-mini 입력 변환·출력 검증 코드가 독립 리뷰를 통과했습니다. 새 Accuracy 채점은 최종 Ours 응답 전체가 완성된 뒤 실행합니다.
