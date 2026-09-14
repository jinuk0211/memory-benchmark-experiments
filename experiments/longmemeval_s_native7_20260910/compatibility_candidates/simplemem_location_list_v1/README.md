# SimpleMem location 배열 호환 수정 후보 v1

**사용 승인·배포 전 검토용입니다.** 현재 frozen source, source manifest, 실행 중 실험에는 적용하지 않았습니다. 이 폴더에는 실행·배포 스크립트가 없습니다.

원본 `MemoryEntry.location`은 `str | None`입니다. 이번 후보는 `_parse_llm_response`에서 **`list[str]`인 location만** `", ".join(location)`으로 표현합니다. 항목 순서와 중복을 유지하며 빈 배열은 빈 문자열이 됩니다. `str`/`None`은 그대로이고 숫자, 객체, 혼합·중첩 배열 등은 기존 Pydantic 검증에서 계속 실패합니다. 항목 안의 쉼표는 이스케이프하지 않으므로 원래 배열을 문자열만으로 완전히 복원할 수는 없습니다. 원래 배열을 감사 기록에 함께 보존합니다.

- `simplemem_location_list_v1.patch`: 실행용 frozen MemoryBuilder 파일에 대한 미적용 패치. 변경은 파서 함수 안에만 있습니다.
- `location_audit.py`: 단일 기록 훅. 기존 attempt 폴더의 **`simplemem_location_normalization.jsonl` 하나에 append**하고 flush/fsync합니다.
- `original_memory_builder.sha256`: 후보를 만든 원본 파일의 SHA256.
- `test_candidate.py`: 실제 frozen 파서와 실제 MemoryEntry 스키마를 사용합니다. 패치는 메모리 안에서만 적용하며 모델·API를 호출하지 않습니다.

기록 정책: 변환을 시도할 때마다 정책 이름, 0부터 시작하는 항목 번호, 원응답 문자열 전체, 원래 location 배열, 변환 문자열을 JSONL에 보존합니다. 이후 같은 응답의 다른 항목이 실패할 수도 있으므로 이 기록은 **변환 시도의 증거이며 construction 성공 증거가 아닙니다.** 정답·question_type·evidence를 읽거나 기록하지 않습니다.

후보 파서는 명시적으로 연결된 `memory_builder.location_normalization_audit` 훅을 요구합니다. 승인 후 별도 통합에서 그 훅을 `record_location_normalization(attempt_dir, event)`에 연결하고, 첫 모델 요청 전에 연결·기록 가능 여부를 검증해야 합니다. 현재 runner에는 연결하지 않았습니다. 훅이 없거나 쓰기에 실패하면 예외가 전파되며, 감사 기록 없이 조용히 정규화하지 않습니다. 현재 strict runner와 함께 쓸 경우 실패 시도가 그대로 남아야 합니다.

방법 충실도 한계: 원본 파서도 location 배열을 거부합니다. 따라서 이 후보는 원본 복원이 아니라 **결정적인 출력 표현 호환 정책 추가**입니다. symbolic location 검색에 들어가는 문자열이 달라질 수 있습니다. 추출 프롬프트, MemoryEntry 스키마, temperature, 재시도 횟수, memory construction 순서는 바꾸지 않습니다. retrieval과 최종 QA의 코드는 수정하지 않지만, 저장 location 문자열이 LIKE 검색 및 reader의 Location 내용에 들어가므로 검색 결과와 답변이 동일하다고 보장하지 않습니다. 질문 정답이나 점수에 근거한 튜닝은 하지 않았습니다. 향후 승인·통합하더라도 원본 strict 실행과 결과를 구분하고 새 source/protocol hash로 기록해야 합니다.

로컬 검증: 이 폴더에서 `python -m unittest -v test_candidate.py`. 정상값 불변, 항목 순서·중복·빈 배열, 원응답 기록, 잘못된 타입의 동일 오류, 훅 누락·기록 실패, 파서 외 AST 불변을 확인합니다. 테스트 임시 기록도 이 후보 폴더 안에서만 생성 후 정리됩니다.
