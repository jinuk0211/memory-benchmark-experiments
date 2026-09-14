# SimpleMem: 원본 Tantivy 검색을 위한 독립 환경

현재 상태는 **계획 및 Linux hash lock 준비 완료, 새 환경 설치·native 검증 미실행**입니다. 기존 LanceDB 0.38.0은 `use_tantivy=True`를 제거해 원본 호출이 실패했습니다. native FTS로 코드를 바꾸지 않고, 공식 SimpleMem requirements의 LanceDB 0.25.3을 별도 환경에 설치하는 계획입니다.

이 폴더만 새로 작성했습니다. 동결된 후보 코드·source manifest·4후보 archive·launch plan·기존 환경은 수정하지 않았습니다. 이 문서는 recovery 코드나 archive의 전송 승인을 대신하지 않습니다.

| 파일 | 역할 |
|---|---|
| `requirements.in` | 실제 실행 경로에 필요한 직접 의존성 12개 |
| `upstream_requirements.txt` | 공식 commit의 requirements 원본 바이트 사본 |
| `upstream_constraints.txt` | 원본의 정확한 버전 pin만 제약으로 적용; 미사용 패키지는 설치하지 않음 |
| `requirements-linux-py311.lock` | uv가 Linux CPython 3.11 대상으로 해결한 전체 55개 패키지와 해시 |
| `dependency_provenance.json`, `package_provenance.json` | 코어/helper import, 원본 해시, 공식 wheel·릴리스 근거 |
| `dependency_download_estimate.json` | 55개 Linux 호환 wheel의 URL·해시·실제 다운로드 크기 |
| `resolver.stderr.txt`, `resolver.stdout.txt` | 실제 resolver 결과와 yanked 경고 |

## 바꾸는 환경과 유지하는 방법

- 새 환경: `/workspace/longmemeval_s_native7_20260910/.venv-simplemem-native0253`.
- 생성기: **실제 CPython 3.11.16인 `.venv-lightmem/bin/python`**. `.venv-client`는 3.12.14여서 생성기로 사용하지 않습니다. `--system-site-packages`를 사용하지 않습니다.
- 원본 pin: LanceDB 0.25.3, PyLance 0.39.0, PyArrow 22.0.0, SentenceTransformers 5.1.1, Transformers 4.57.0, OpenAI 2.3.0 등. 해결된 55개 중 53개는 원본 requirements의 버전입니다.
- 원본에서 미고정한 Tantivy는 같은 2025년 의존성 세대의 0.25.0(2025-09-09 배포), 누락된 Torch는 공식 2.8.0+cpu wheel로 명시합니다. 두 버전을 원저자의 실제 환경이었다고 주장하지 않습니다. CPU Torch wheel은 glibc 2.28 이상을 요구합니다.
- `pylance`는 필수입니다. LanceDB의 일반 의존성에서는 선택 항목이지만, 원본 Tantivy 색인 생성은 `table.to_lance()`를 호출해 `lance` 모듈을 가져옵니다.
- native_five에서 가져오는 helpers는 표준 라이브러리이며, `request_metering`은 httpx만 추가합니다. LoCoMo 변환 함수만 AST로 가져오므로 평가용 nltk·bert-score·rouge-score와 LangChain 등은 필요 없습니다.
- CUDA/nvidia/triton/vLLM/LLMLingua, cloud embedding extras, 별도 모델 서버는 설치하지 않습니다. `lance-namespace`와 urllib3 client는 LanceDB/PyLance의 필수 전이 의존성이므로 남깁니다.
- Qwen FP16 모델·기존 localhost endpoint(계량 proxy `18083/v1`)·프롬프트·생성 설정을 유지합니다. MiniLM은 기존 고정 snapshot을 CPU FP32로 읽고 **384차원·native 256토큰**을 유지합니다. 원본 Tantivy `en_stem`, semantic/keyword/structured 검색 코드는 그대로입니다.

## Transformers 4.57.0 yanked 처리

[공식 PyPI](https://pypi.org/project/transformers/4.57.0/)의 이유는 “Error in the setup causing installation issues”입니다. [공식 4.57.1 릴리스](https://github.com/huggingface/transformers/releases/tag/v4.57.1)는 optional `optax>=0.08` 때문에 Poetry 파싱이 실패한 문제와 `>=0.0.8` 수정을 설명합니다. 이번 lock은 Flax/optax extras를 포함하지 않으며 uv가 원본 exact pin을 실제 해결했습니다. 경고도 보존했습니다.

**이는 실제 import/encode 성공 증거가 아닙니다.** 설치 후 아래 gate에서 실패하면 새 환경을 실패 상태로 남기고 원인·traceback을 기록합니다. 그때 검증된 4.57.6 또는 최소 patch 변경의 필요성과 차이를 검토하며, 자동 업그레이드나 원본 후보 수정을 하지 않습니다.

## 설치 명령 — 운영자가 별도로 실행

공개 의존성 설치가 승인되고 이 환경 자료의 해시를 확인한 뒤 실행합니다. 기존 환경이나 경로가 있으면 중단합니다. 아래 명령은 이 작업에서 실행하지 않았습니다. 새 proof 경로는 root가 관리하며 기존 receipt를 덮어쓰지 않습니다.

```bash
set -euo pipefail
R=/workspace/longmemeval_s_native7_20260910
C="$R/runtime_compatibility/simplemem_lancedb0253_v1"
E="$R/.venv-simplemem-native0253"
P="$R/runtime_compatibility_receipts/simplemem_lancedb0253_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$E"
test ! -e "$P"
mkdir -p "$P"
"$R/.venv-lightmem/bin/python" -I -c 'import sys; print(sys.version); assert sys.version_info[:3] == (3,11,16)' > "$P/creator-python.txt"
getconf GNU_LIBC_VERSION > "$P/glibc.txt"
# 위 glibc가 2.28 이상인지 확인한 뒤 계속합니다.
sha256sum "$C/requirements-linux-py311.lock" "$C/requirements.in" "$C/upstream_constraints.txt" > "$P/input-sha256.txt"
"$R/.venv-lightmem/bin/python" -I -m venv "$E"
"$E/bin/python" -I -m pip --version > "$P/pip-version.txt"
"$E/bin/python" -I -m pip --isolated install --disable-pip-version-check --no-cache-dir --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --report "$P/pip-install-report.json" -r "$C/requirements-linux-py311.lock" > "$P/install.log" 2>&1
"$E/bin/python" -I -m pip check > "$P/pip-check.txt" 2>&1
"$E/bin/python" -I -m pip freeze --all > "$P/pip-freeze.txt"
```

설치 report의 실제 URL·archive SHA256·버전을 `dependency_download_estimate.json`과 대조합니다. 시스템 glibc에 따라 lock에 포함된 다른 호환 wheel이 선택되면 실제 wheel과 크기를 receipt에 남깁니다. bootstrap pip/setuptools는 `venv` 생성기의 버전을 따르고 freeze로 기록합니다. 설치 실패 시 새 환경과 로그를 보존하고 자동 삭제·재사용하지 않습니다.

## 환경 검증 순서

1. `sys.version_info[:3] == (3,11,16)`, `sys.prefix == E`, `include-system-site-packages = false`, glibc ≥2.28, `pip check`를 확인합니다. `import lancedb, lance, tantivy, pyarrow, openai, pydantic, dateparser, numpy, httpx, torch, transformers; from sentence_transformers import SentenceTransformer`가 통과해야 합니다. Torch가 `2.8.0+cpu`, `torch.version.cuda is None`인지 확인하고 설치 패키지에 nvidia/CUDA/triton/vLLM이 없는지 대조합니다. Qwen 서버를 import하거나 요청하지 않습니다.
2. **공개 backend API만**으로 별도 scratch 경로에 2행을 한 번에 넣습니다. 원본 schema는 문자열 `entry_id/lossless_restatement/timestamp/location/topic`, 문자열 배열 `keywords/persons/entities`, `pa.list_(pa.float32(), 384)`인 `vector`입니다. A/B는 서로 다른 단어·사람·위치와 직교 단위벡터를 사용합니다.
3. 다음 native API가 성공해야 합니다. scratch 경로·각 결과 ID·필드·오류를 기록하고 실제 후보 통과라고 부르지 않습니다.

```python
# table은 위 원본 schema로 생성하고 synthetic 두 행을 table.add([a,b])한 상태입니다.
table.create_fts_index("lossless_restatement", use_tantivy=True,
                       tokenizer_name="en_stem", replace=True)
assert table.count_rows() == 2
semantic = table.search(a["vector"]).limit(1).to_list()
assert semantic[0]["entry_id"] == a["entry_id"]
assert isinstance(semantic[0]["_distance"], float)
lexical = table.search("velociraptor").limit(2).to_list()
assert [x["entry_id"] for x in lexical] == [a["entry_id"]]
assert isinstance(lexical[0]["_score"], float)
expr = ("array_has_any(persons, make_array('Alice')) AND "
        "location LIKE '%Seoul%' AND "
        "array_has_any(entities, make_array('velociraptor')) AND "
        "timestamp >= '2025-01-01T00:00:00' AND "
        "timestamp <= '2025-01-31T23:59:59'")
assert [x["entry_id"] for x in table.search().where(expr, prefilter=True).limit(2).to_list()] == [a["entry_id"]]
# DB를 다시 연결하고 같은 table을 open_table하여 세 검색을 재확인합니다.
```

4. HF offline/local-only 상태에서 기존 MiniLM snapshot을 `SentenceTransformer(snapshot, device='cpu', local_files_only=True)`로 불러옵니다. snapshot은 `/workspace/.hf_home/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41`입니다. 384차원·`max_seq_length==256`·CPU FP32를 확인하고 공개 synthetic 문자열을 `encode(..., normalize_embeddings=True)`로 검증합니다. 기존 통과한 native-vs-HTTP 수치 검증과 **동일 입력·판정 기준**을 재사용해 다른 Transformers/Torch 환경에서도 일치하는지 별도 기록합니다. 새 모델을 다운로드하거나 실제 history를 입력하지 않습니다.
5. 그 뒤 recovery source/실행이 허용되었을 때만 원본 runner `--help`, source/runtime/model 해시 검증, 공유 smoke `e47becba`를 진행합니다. 위 공개 의존성 검사만으로 native candidate 또는 500개 실행이 통과했다고 기록하지 않습니다.

정상 LongMemEval 일괄 경로는 모든 window의 결과를 모아 단일 DB insert 후 FTS를 만듭니다. 후속 insert가 Tantivy 색인을 자동 갱신하지 않는 원본 특성은 그대로입니다. 예외로 순차 fallback이 발생한 실행은 별도 audit가 필요하며, 이 환경 계획은 그 코드를 변경하거나 검색 결과 불변을 보증하지 않습니다.

## 용량과 최종 실행 연결

55개 Linux wheel의 다운로드 합계는 **431,753,616 bytes (411.75 MiB)**입니다. 설치 후 크기는 실측이 아닌 **1.21–2.01 GiB 추정**이며, 설치 임시 공간까지 3 GiB 여유를 권장합니다. `--no-cache-dir`로 영구 wheel 복사본을 만들지 않고, 기존 Qwen/MiniLM weights를 재사용합니다. 원본 Tantivy writer 기본 1 GiB는 RAM 예산이며 이 디스크 합계와 별개입니다.

root가 최종 launch plan의 새 실사용 복사본을 만들 때에만 SimpleMem의 Python을 `.venv-simplemem-native0253/bin/python`으로 바꿉니다. 나머지 SimpleMem argv와 새 run-dir `runs/simplemem_native_dialogues_v1`을 유지하고 실제 argv·새 환경 receipt·최종 runtime/model/source 검증을 함께 기록합니다. 동결된 template나 4후보 archive를 덮어쓰지 않으며, 수동 HiGMem smoke와 중복 큐를 시작하지 않습니다.
