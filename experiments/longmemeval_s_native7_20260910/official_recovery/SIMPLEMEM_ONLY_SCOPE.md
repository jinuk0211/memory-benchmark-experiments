# SimpleMem만 추가하는 배포 제안

이 파일은 현재 작업 범위인 HiGMem·LightMem·SimpleMem 각 500개 중, 아직 필요한 SimpleMem 코드만 추가하기 위한 로컬 제안입니다. HiGMem·LightMem의 기존 코드, 성공 캐시와 실행 기록은 그대로 사용합니다. 이 묶음의 작성과 CPU 테스트는 서버 전송·실행 승인 또는 실제 smoke/full500 통과를 의미하지 않습니다. 기존 승인 차단을 우회하는 절차가 아닙니다.

## 포함 범위와 무결성

- 새 archive: `native_recovery_simplemem_only_v1.tgz`. experiment root 기준 상대경로의 일반 파일 28개만 포함합니다.
- 동결 SimpleMem 후보 전체 25개(243,366 B): 공식 upstream 20개와 runner, source manifest, README, CPU test, canonical smoke 입력 audit입니다. 원본 바이트를 그대로 보존합니다. verifier가 subtree 전체를 요구하므로 README/test/audit도 필요합니다.
- 기존 공용 `official_recovery/verify_recovery_runtime.py` 1개를 그대로 포함합니다.
- 새 manifest와 이 적용 범위 문서 2개를 포함합니다. archive에는 모델, 데이터셋, 기존 runtime 파일, venv, wheel, 기존 실패·성공 로그를 넣지 않습니다.
- A-MEM/Mem0/LangMem 후보, 다른 recovery 계획/manifest, 7-method dispatcher, 새 알고리즘은 포함하지 않습니다. 기존 `native_recovery_four_03.tgz`(777,755 B)는 변경하지 않습니다.

원본 `model_integrity.json` SHA256은 `18e3df8b2c2b6b9873619ad31ce8b2f55fde93c987a1ea3b48586b1bd237c02d`입니다. 이 파일과 원래 검증 receipt의 `runtime_files`는 실제 **18개**입니다. 앞선 요청의 161개는 원본 파일 수와 일치하지 않습니다. 새 manifest는 원본 18개를 전부 보존하고 SimpleMem 25개와 공용 verifier 1개만 추가한 **44개 binding**입니다. models의 세 모델·경로·모든 파일 SHA도 원본과 같습니다.

새 `official_recovery/model_integrity_simplemem_recovery_v1.json` SHA256:
`06f8072d0bf27c040629933d646e21ca665bfb86d9ba9b219d452c9b3aeb14b0`

원본 runtime receipt SHA256은 `ad0ac34ce4a68101d39b466e5c208a338123868fc008979c97a85cf7f180897d`, source snapshot SHA256은 `c0448a9a90dfa71bcb62f80a190d2270e943cc36f04f610eb20271ec6951cc70`입니다. 둘 다 그대로 유지해야 합니다. 기존 `native_five.py` helper는 원본 runtime binding으로, `source/MemoryData/utils/request_metering.py`는 원본 source snapshot으로 검증됩니다. 따라서 이 묶음에 두 파일을 다시 넣지 않습니다.

## 검증과 적용 조건

CPU 검증은 기존 파일을 수정하지 않고 다음 두 suite로 실행했습니다. SimpleMem 10개, 공용 verifier 12개 모두 통과했습니다. 실제 LanceDB/Tantivy·GPU·모델 서비스는 이 CPU 테스트 범위에 없습니다.

```text
python -B -m unittest discover -s official_recovery/simplemem_native_dialogues_v1 -p test_runner.py -v
python -B -m unittest discover -s official_recovery -p test_verify_recovery_runtime.py -v
```

서버 전송이 별도로 허용된 이후에만 다음 검증을 고려합니다. `--candidate`는 짧은 이름 `simplemem`이 아니라 아래 실제 디렉터리 경로입니다. 아래 `UNIQUE` receipt는 존재하지 않는 새 경로로 바꿔야 하며, 기존 receipt를 덮어쓰지 않습니다. 공용 verifier는 CUDA·FP16 서비스·MiniLM 검증을 수행하므로 inference 환경에서 실행합니다. 이 명령은 본 제안 작성 중 실행하지 않았습니다.

```sh
R=/workspace/longmemeval_s_native7_20260910
"$R/.venv-inference/bin/python" "$R/official_recovery/verify_recovery_runtime.py" \
  --root "$R" \
  --manifest "$R/official_recovery/model_integrity_simplemem_recovery_v1.json" \
  --candidate "$R/official_recovery/simplemem_native_dialogues_v1" \
  --receipt "$R/runtime_receipt_simplemem_recovery_v1_UNIQUE.json"
```

SimpleMem 실행 환경은 이미 별도로 준비한 `.venv-simplemem-native0253`를 사용하며 재설치하지 않습니다. 원본 LanceDB 0.25.3/Tantivy 경로, Qwen3.5-9B FP16, pinned MiniLM 384차원·native 256토큰과 원본 알고리즘 설정을 유지합니다. 실제 실행 전 해당 환경의 기존 검증 결과와 새 runtime receipt를 확인해야 합니다.

canonical 데이터셋은 `$R/longmemeval_s_cleaned.json`입니다. 동결 후보 README의 예시 데이터셋 경로는 이 실제 root 경로로 해석합니다. SimpleMem run-dir은 `$R/runs/simplemem_native_dialogues_v1`이며, 이전 `$R/runs/simplemem` 실패 시도는 보존합니다. 기존 shared smoke qid `e47becba`의 전체 history를 먼저 검증하고, 성공 후 같은 protocol의 500개 전체 실행을 진행하는 조건을 유지합니다. 본 제안은 smoke/full500를 시작하지 않습니다.

기존 `run_all.py`는 정확히 7개 method 전용이므로 이번 3개 범위의 dispatcher로 호출하지 않습니다. 이 제안에 3-method dispatcher나 새 launch plan은 없습니다. HiGMem·LightMem·SimpleMem 각각 실제 500개 고유 결과와 기존 export 검사를 충족한 뒤에만 완료로 판정합니다. 공식 judge 호출은 포함하지 않습니다.

## archive의 정확한 파일 목록

```text
official_recovery/SIMPLEMEM_ONLY_SCOPE.md
official_recovery/model_integrity_simplemem_recovery_v1.json
official_recovery/simplemem_native_dialogues_v1/README.md
official_recovery/simplemem_native_dialogues_v1/canonical_smoke_input_audit.json
official_recovery/simplemem_native_dialogues_v1/runner.py
official_recovery/simplemem_native_dialogues_v1/source_manifest.json
official_recovery/simplemem_native_dialogues_v1/test_runner.py
official_recovery/simplemem_native_dialogues_v1/upstream/LICENSE
official_recovery/simplemem_native_dialogues_v1/upstream/README.md
official_recovery/simplemem_native_dialogues_v1/upstream/config.py.example
official_recovery/simplemem_native_dialogues_v1/upstream/main.py
official_recovery/simplemem_native_dialogues_v1/upstream/requirements.txt
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/__init__.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/answer_generator.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/config_default.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/database/__init__.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/database/vector_store.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/database/vector_store_backend.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/hybrid_retriever.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/memory_builder.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/models/__init__.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/models/memory_entry.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/settings.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/utils/__init__.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/utils/embedding.py
official_recovery/simplemem_native_dialogues_v1/upstream/simplemem/core/utils/llm_client.py
official_recovery/simplemem_native_dialogues_v1/upstream/test_locomo10.py
official_recovery/verify_recovery_runtime.py
```
