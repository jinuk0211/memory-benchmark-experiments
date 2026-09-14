# 복구 후보 4개를 포함한 7개 방법 실행 계획

현재 파일은 **실행하지 않은 template**입니다. `launch_plan.recovery.template.json`의 receipt 경로와 SHA는 의도적으로 잘못된 placeholder입니다. 실제 새 receipt로 채우기 전에는 `run_all.validate_plan()`이 통과할 수 없습니다. 이 작업에서는 기존 실행기, plan/receipt, 후보, manifest, archive 및 원격 서버를 변경하지 않았습니다.

원격 기준 경로 `R`은 `/workspace/longmemeval_s_native7_20260910`입니다. 모든 방법은 동일한 `R/longmemeval_s_cleaned.json`을 읽습니다. SHA256은 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`이며 500개 고유 질문의 첫 ID는 `e47becba`여야 합니다.

| 방법 | R 아래 실행기 | R 아래 run-dir | Python 환경 |
|---|---|---|---|
| E-Mem | `native_five.py --method e_mem` | `runs/e_mem` 유지 | `.venv-client`, 3.12 |
| SimpleMem | `official_recovery/simplemem_native_dialogues_v1/runner.py` | `runs/simplemem_native_dialogues_v1` 신규 | `.venv-client`, 3.12 |
| LangMem | `official_recovery/langmem_native_sessions_v1/runner.py` | `runs/langmem_native_sessions_v1` 신규 | `.venv-client`, 3.12 |
| Mem0 | `official_recovery/mem0_native_pairs_v1/runner.py` | `runs/mem0_native_pairs_v1` 신규 | `.venv-mem0-oss0194`, 3.11 |
| A-MEM | `official_recovery/a_mem_paper_v1/runner.py` | `runs/a_mem_paper_v1` 신규 | `.venv-client`, 3.12 |
| LightMem | `native_lightmem.py` | `runs/lightmem` 유지 | 기존 `.venv-lightmem`, 3.11 |
| HiGMem | `native_higmem.py` | `runs/higmem` 유지 | `.venv-client`, 3.12 |

E-Mem/LightMem/HiGMem의 argv는 `remote_verification/20260910T0902Z/launch_plan.json`에서 그대로 복사했습니다. 성공 cache와 기존 run-dir을 유지합니다. 예전 `runs/a_mem`, `runs/simplemem`, `runs/mem0`, `runs/langmem`의 실패 기록도 그대로 보존합니다. 새로운 recovery 후보가 아직 원격에 없으므로 신규 4개 run-dir의 성공을 주장하지 않습니다.

## CLI 차이와 모델 조건

네 후보의 실제 `--help`를 로컬에서 확인했습니다. A-MEM·SimpleMem은 `--embedding-model`에 **로컬 MiniLM snapshot 경로**를 받고 embedding API/source-root/tokenizer 인자를 받지 않습니다. LangMem은 **MiniLM repo명 + embedding API + Qwen tokenizer 경로**를 받으며 source-root는 없습니다. Mem0도 repo명/API/tokenizer를 받고 frozen MemoryData source-root를 명시합니다. Mem0에는 `--method` 옵션이 없어 생략했습니다.

공유 모델은 `Qwen/Qwen3.5-9B`, 신규 네 후보의 API는 `http://127.0.0.1:18083/v1`입니다. Mem0/LangMem 임베딩 API는 `http://127.0.0.1:18084/v1`, repo명은 `sentence-transformers/all-MiniLM-L6-v2`, 차원은 384입니다. A-MEM/SimpleMem/LightMem/HiGMem은 동일 MiniLM의 로컬 snapshot을 사용합니다. Qwen revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`의 실제 절대경로를 JSON에 넣었습니다. FP16·thinking off·MiniLM 기본 256-token 한도는 기존 서비스 증거를 유지해야 합니다.

SimpleMem의 동결된 README 실행 예시는 `source/MemoryData/datasets/LongMemEval/longmemeval_s_cleaned.json`을 가리키지만, 이 계획에서는 실제 원격 canonical 경로인 `R/longmemeval_s_cleaned.json`으로 정정했습니다. README와 동결 archive는 수정하지 않습니다.

Mem0 환경 생성에는 실제 3.11인 `.venv-lightmem/bin/python`을 확인 후 사용합니다. `.venv-client`는 실제 3.12.14이므로, 동결된 Mem0 `ENVIRONMENT_PLAN.md`의 생성용 interpreter 두 곳은 운영 기록에서 정정해야 합니다. 새 환경의 `include-system-site-packages=false`, Linux/Python 3.11 hash lock 설치, `pip check` 및 runtime receipt를 확인합니다. 기존 LightMem 패키지를 새 환경에 상속하지 않습니다.

## 최종 receipt가 갖춰야 할 조건

현재 `model_integrity_native_recovery_v1.json`은 A-MEM 9개, SimpleMem 25개, Mem0 107개 파일을 포함하고 **LangMem 42개는 포함하지 않습니다**. 이 3후보 manifest나 그것으로 생성한 receipt를 이 계획에 넣으면 안 됩니다. 기존 manifest/receipt를 덮어쓰지 않고 네 후보 전체를 포함하는 새 manifest와 새 최종 receipt를 생성합니다.

최종 receipt는 기존 runtime/model/source 증거를 유지하고, 네 candidate 디렉터리의 모든 파일(`__pycache__`, `.pyc` 제외) 및 `official_recovery/verify_recovery_runtime.py`의 실제 SHA256을 `integrity.runtime_files`에 포함해야 합니다. 각 후보의 `runner.py`와 `source_manifest.json`도 포함합니다. 새로운 Mem0 환경의 설치·import·tokenizer 검증 증거는 별도 운영 receipt와 연결합니다.

기존 `run_all.validate_plan()`은 receipt의 SHA와 `status=runtime_verified`를 검사하지만 **네 후보 전체가 포함됐는지는 검사하지 않습니다**. 또한 현재 `verify_recovery_runtime.py --candidate`의 필수 파일 검사는 한 candidate를 기준으로 합니다. 따라서 전체 4후보 manifest로 실제 검증을 수행하고 아래 포함 범위 검사를 추가로 통과해야 합니다. 이것은 계획 문서의 운영 확인이며 기존 실행기를 바꾸지 않습니다.

최종 JSON은 template의 새 사본 `R/official_recovery/launch_plan.recovery.final_REPLACE_TIMESTAMP.json`에 작성합니다. `runtime_receipt.path`를 새 실제 receipt로, `runtime_receipt.sha256`을 그 파일의 실제 SHA256으로 채웁니다. 기존 `R/launch_plan.json`, `R/runtime_receipt.json` 및 동결 archive는 건드리지 않습니다.

다음 검증은 모델을 호출하지 않습니다. `P`를 새 최종 plan 경로로 지정한 뒤 R에서 실행하는 예입니다.

```bash
R=/workspace/longmemeval_s_native7_20260910
P="$R/official_recovery/launch_plan.recovery.final_REPLACE_TIMESTAMP.json"
cd "$R"
"$R/.venv-client/bin/python" -B - "$P" <<'PY'
import hashlib, json, sys
from pathlib import Path
from run_all import validate_plan
root = Path('/workspace/longmemeval_s_native7_20260910')
plan = json.loads(Path(sys.argv[1]).read_text())
ids, directories = validate_plan(plan)
assert ids[0] == 'e47becba' and len(ids) == 500
receipt = json.loads(Path(plan['runtime_receipt']['path']).read_text())
bindings = receipt['integrity']['runtime_files']
folders = ('a_mem_paper_v1', 'simplemem_native_dialogues_v1',
           'mem0_native_pairs_v1', 'langmem_native_sessions_v1')
required = {root / 'official_recovery/verify_recovery_runtime.py'}
for name in folders:
    candidate = root / 'official_recovery' / name
    assert (candidate / 'runner.py').is_file()
    assert (candidate / 'source_manifest.json').is_file()
    required.update(p for p in candidate.rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and p.suffix != '.pyc')
for path in required:
    key = path.relative_to(root).as_posix()
    assert bindings.get(key) == hashlib.sha256(path.read_bytes()).hexdigest(), key
print('Verified all four recovery candidates, seven commands and canonical 500 IDs')
PY
```

## 새 queue에서 실행

**수동 HiGMem shared smoke가 완료되기 전에는 새 queue를 시작하지 않습니다.** 같은 run-dir을 사용하는 이전 queue·수동 smoke·worker가 실행 중이지 않은지 확인합니다. 기존 runner에는 중복 실행을 막는 전역 lock이 없습니다. 완료한 성공 cache의 protocol/source/argv가 유지되는지도 확인합니다.

새 state-dir은 `R/queue/recovery_REPLACE_TIMESTAMP`처럼 사용한 적 없는 경로를 선택합니다. 기존 `R/queue`의 상태·계획 snapshot·로그를 덮어쓰지 않습니다. 별도 Supervisor program에서 다음 명령을 실행하도록 설정합니다. 아래는 향후 실행 예이며 이 문서 작성 중 실행하지 않았습니다.

```bash
env -u OPENROUTER_API_KEY \
  OPENAI_API_KEY=EMPTY SIMPLEMEM_API_KEY=EMPTY SIMPLEMEM_EMBEDDING_API_KEY=EMPTY \
  MEM0_TELEMETRY=false HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SKIP_SBERT_SIM=1 \
  NLTK_DATA=/root/.cache/longmemeval_s_native7_20260910/nltk_data \
  NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
  /workspace/longmemeval_s_native7_20260910/.venv-client/bin/python \
  /workspace/longmemeval_s_native7_20260910/run_all.py \
  --plan /workspace/longmemeval_s_native7_20260910/official_recovery/launch_plan.recovery.final_REPLACE_TIMESTAMP.json \
  --state-dir /workspace/longmemeval_s_native7_20260910/queue/recovery_REPLACE_TIMESTAMP
```

`run_all.py`의 원래 순서인 E-Mem → SimpleMem → LangMem → Mem0 → A-MEM → LightMem → HiGMem을 그대로 사용합니다. 먼저 **7개 모두 같은 e47becba 전체 history smoke**를 통과해야 하며, 이후 7개 모두 full500을 순차 실행합니다. template argv에는 `--ids-file`이나 다른 부분집합 제한을 넣지 않았고 smoke 단계에서만 기존 queue가 그 인자를 추가합니다. 하나라도 실패하거나 새 완료 상태를 쓰지 않으면 queue는 중단합니다.

네 후보 모두 전체 500 population을 protocol에 고정하고 선택한 smoke ID는 protocol에서 제외합니다. 따라서 동일 source/model/endpoint/설정이면 smoke에서 full500으로 전환해 완료 cache를 재사용할 수 있습니다. 기존 E-Mem/LightMem/HiGMem도 같은 방식으로 선택 제한을 분리하며, cache 재사용 시 queue가 확인하는 최종 상태 파일을 다시 씁니다. 소스나 endpoint를 바꿔 기존 protocol에 억지로 맞추지 않습니다.

`7 × 500 = 3500`개의 생성이 모두 검증돼야 queue가 `generation_complete`가 됩니다. 공식 export·무료 진단 F1·백업은 그 이후 별도 단계이며, 이 queue는 유료 judge를 호출하지 않습니다.