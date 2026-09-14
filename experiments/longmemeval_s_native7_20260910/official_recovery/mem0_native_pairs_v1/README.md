# Mem0 OSS 0.1.94 + LongMemEval 통합 후보

현재 상태: **로컬 구현·CPU 검증 후보**입니다. 기존 source, 실패한 시도, 서비스, launch plan은 변경하지 않았습니다. 원격 배포 및 모델 실험 결과는 아직 없습니다.

이 후보는 공식 **Mem0 OSS** 메모리 알고리즘과 기존 LongMemEval 답변 통합을 조합합니다. 2025년 논문의 관리형 Cloud 실험 재현이라고 주장하지 않습니다.

| 파일 | 역할 |
|---|---|
| `runner.py` | 원본 역할을 보존하는 세션별 2개 발화 입력, 공식 OSS memory/search, 기존 QA 통합 |
| `vendor/` | 공식 wheel의 97개 파일을 바이트 그대로 추출한 패키지·배포 메타데이터 |
| `mem0ai-0.1.94-py3-none-any.whl` | 해시를 확인한 공식 배포 원본 |
| `source_manifest.json` | 공식 commit, 배포 및 파일별 SHA256, 버전 선택 근거 |
| `requirements.txt`, `requirements-linux-py311.lock`, `ENVIRONMENT_PLAN.md` | 별도 CPU client 의존성·전체 hash lock·미실행 설치 계획 |
| `test_runner.py` | 모델 호출 없는 입력·실패·재개·export 및 원본 알고리즘 회귀 검증 |

## 왜 이 버전인가요?

[논문 제출일은 2025-04-28](https://arxiv.org/abs/2504.19413)입니다. 공식 PyPI에서 그 이전 마지막 안정 배포는 [mem0ai 0.1.94](https://pypi.org/project/mem0ai/0.1.94/)로, 2025-04-26에 공개됐습니다.

- 공식 태그: **v0.1.94**
- 공식 commit: `07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4`
- [해당 commit의 memory/main.py](https://github.com/mem0ai/mem0/blob/07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4/mem0/memory/main.py)는 wheel 안 파일과 정확히 같습니다.
- wheel SHA256: `862aaccf4ec41d65a5c935dc49c7b8175c96f6b60b29abeda338d8a335027e2c`
- memory/main.py SHA256: `7c43b5defd6767f3a101a9bd5d605662e11f1d637462846b82c2ec5658b63371`

기존 vendor는 공식 ancestry가 입증되지 않았고 action 수정 재시도가 들어 있습니다. 설치 설정의 `2.0.20`은 2026년 9월 공개된 다른 API 버전이어서 그 vendor의 정체성을 보증하지 않습니다. 따라서 논문 시기에 공개된 OSS 배포를 기준으로 새 후보를 분리했습니다. `v`가 없는 태그 `0.1.94`는 2024년의 다른 과거 코드이므로 사용하지 않습니다.

## 실제로 보존하거나 바꾼 부분

공식 OSS 파일은 수정하지 않았습니다. 공식 fact extraction prompt, action update prompt, 파서, ADD/UPDATE/DELETE/NONE 처리, 내부 검색과 Qdrant 저장을 그대로 실행합니다. benchmark fact prompt, action 자동 수정, 새 파싱 복구는 추가하지 않았습니다.

입력만 LongMemEval에 연결합니다. 각 질문에 독립된 저장소를 만들고 원래 세션 순서를 유지하며, 각 세션의 연속된 발화 2개를 `Memory.add(messages, user_id=..., metadata=..., infer=True)`로 전달합니다. 마지막 1개 발화도 보존합니다. role/content만 허용하며 긴 발화는 자르지 않습니다. 세션 경계를 넘겨 묶거나 system/단일 user 문자열로 평탄화하지 않습니다. 빈 발화나 잘못된 역할은 실패 처리합니다.

2개 단위는 [공식 LongMemEval loader](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/benchmarks/longmemeval/run.py)의 역할 보존 단위를 참고합니다. 2025 Cloud [LoCoMo 평가](https://github.com/mem0ai/mem0/blob/aae5989e78a6188b3b047c104d960c9ad0927e75/evaluation/src/memzero/add.py)도 2개씩 처리하지만, 사람 두 명의 관점을 뒤집어 각각 저장합니다. 이 후보는 그 이중 사용자 구성을 흉내 내지 않습니다. 현재 공식 loader의 시간순 재정렬·빈 입력 건너뛰기도 가져오지 않고 원본 데이터 순서·내용을 보존합니다.

세션 ID, 원래 날짜, UTC로 해석한 timestamp, 발화 범위는 metadata에 보존합니다. OSS 0.1.94는 이 metadata를 fact 추출 prompt에 넣지 않습니다. 상대 날짜 해석을 강화하기 위한 날짜 삽입은 추가하지 않았습니다. 공식 prompt에 있는 실행 시점 날짜도 원본 동작입니다. 따라서 metadata 보존이 시간 추론 향상을 보증하지 않습니다.

모델 연결은 로컬 **Qwen/Qwen3.5-9B FP16**, 임베딩은 **MiniLM 384차원**으로 고정합니다. 메모리 생성의 공식 기본값 `temperature=0.1`, `max_tokens=2000`, `top_p=0.1`을 보존합니다. FP16, thinking off 및 MiniLM 기본 256-token 한도는 상위 서비스 검증이 담당합니다.

답변은 기존 frozen `AgentWrapper._handle_mem0_agent`의 검색·정리·prompt·reader를 그대로 사용합니다. 검색 한도 100, temperature 0.1, 최종 출력 256 tokens, 공개 question/date 입력이 유지됩니다. 이것은 Mem0 OSS에 붙인 기존 benchmark QA 통합이며, 원래 Cloud 논문의 검색 top-k 30 / 두 사용자 / temperature 0 reader와 같다는 뜻은 아닙니다. 저장 내용이 바뀌므로 검색 결과나 답변의 불변성은 보장하지 않습니다.

## 실행·증거 경계

`OPENAI_API_KEY=EMPTY`, 로컬 HTTP endpoint만 허용하고 `MEM0_TELEMETRY=false`, HF offline을 설정합니다. 유료 API 및 judge를 호출하지 않습니다. 현재 공용 client의 OpenAI 2.54는 공식 OSS의 `<2` 요구 범위를 벗어나므로 이 후보에는 별도 client 환경이 필요합니다. `requirements.txt`는 직접 의존성 pin이며 2025 전체 환경의 재현 lock은 아닙니다. Linux/Python 3.11 전체 hash lock을 별도로 해결했으며, 설치 후 `pip check`와 실제 전체 의존성 목록·해시를 별도 runtime receipt로 보존해야 합니다. 설치 계획은 `ENVIRONMENT_PLAN.md`에 있습니다.

`runner.py`는 canonical 500 데이터 SHA를 검증합니다. `--ids-file`은 해당 500개의 고유 부분집합이며 smoke 선택은 protocol을 바꾸지 않습니다. 새 run-dir을 사용합니다.

```text
python runner.py --dataset <canonical500.json> --run-dir <new-run-dir>
  --source-root <frozen-MemoryData>
  --api-base http://127.0.0.1:18083/<approved-meter-route>/v1
  --embedding-api-base http://127.0.0.1:18084/<approved-meter-route>/v1
  --model Qwen/Qwen3.5-9B
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
  --embedding-dims 384 --tokenizer <pinned-local-Qwen-snapshot>
  [--ids-file <shared-smoke-ids.json>]
```

`protocol.json`은 source/runner 해시와 전체 500개 population을 고정합니다. 각 history의 `attempt_NNNN`에는 source/query/worker, memory, console, 구성·진행·답변 증거가 남습니다. `construction_progress.json`은 현재 세션/발화 범위와 완료 pair 수를 기록합니다. 원본이 JSON 오류 등을 내부에서 삼켜도 ERROR 로그를 감지하면 `failure.json`을 남기고 exit 1로 중단합니다. 원본 응답을 고쳐 재시도하지 않습니다. 실패 시도는 보존하고 다음 실행에서 새 attempt를 만듭니다. worker는 run protocol과 runtime/source 해시를 대조하며, 정상 종료 후 입력·메모리 DB·build·답변·로그의 SHA256을 `completion.json`에 기록합니다. 재사용 전에 모든 artifact를 다시 대조하고 누락·변경 시 중단합니다. 실제 import된 공식 fact prompt의 전체 문자열과 SHA도 native_runtime에 남깁니다.

질문 정답·유형·has_answer/evidence는 memory worker source에 전달하지 않습니다. 최종 질문은 QA에서만 사용합니다. `predictions.json`과 `status.json`은 기존 `export_official.py --method mem0` 형식에 맞춥니다. 모든 500개가 성공해야 전체 export가 가능하고, 공식 judge는 별도입니다.

CPU 검증 명령: `python -m unittest -v test_runner.py`. 실제 원본 `_add_to_vector_store` 함수를 fake I/O와 실행하여 action·파싱 실패 경계를 확인합니다. 임시 500개 fixture를 이용한 controller/export 검증은 실제 모델 품질이나 canonical 500 완료를 뜻하지 않습니다.