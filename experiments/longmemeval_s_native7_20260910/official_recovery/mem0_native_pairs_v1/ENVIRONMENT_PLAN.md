# 별도 Mem0 OSS 환경 설치 계획 — 아직 원격에서 실행하지 않음

대상은 새 `/workspace/longmemeval_s_native7_20260910/.venv-mem0-oss0194`입니다. 기존 client/inference/LightMem 환경은 수정하지 않습니다. 아래 명령은 후보 bundle 검토와 업로드를 마친 뒤 실행하는 계획입니다.

`requirements-linux-py311.lock`은 Linux x86_64/Python 3.11 대상으로 해결한 전체 62개 패키지와 공식 배포 SHA256입니다. `requirements_resolution_windows_py311.json`은 참고용 Windows dry-run으로, Linux 설치 증거로 사용하지 않습니다.

```bash
set -euo pipefail
R=/workspace/longmemeval_s_native7_20260910
C="$R/official_recovery/mem0_native_pairs_v1"
V="$R/.venv-mem0-oss0194"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
E="$R/queue/mem0_oss0194_environment_$STAMP"
if [ -e "$V" ]; then
  echo 'Refusing to replace an existing environment' >&2
  exit 1
fi
mkdir -p "$E"
"$R/.venv-client/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11); print(sys.executable, sys.version)'
sha256sum "$C/runner.py" "$C/source_manifest.json" "$C/requirements-linux-py311.lock" > "$E/inputs.sha256"
df -B1 "$R" > "$E/disk_before.txt"
"$R/.venv-client/bin/python" -m venv "$V"
"$V/bin/python" -m pip install --no-cache-dir --disable-pip-version-check --only-binary=:all: --require-hashes --report "$E/pip_install_report.json" -r "$C/requirements-linux-py311.lock" > "$E/install.log" 2>&1
"$V/bin/python" -m pip check > "$E/pip_check.txt"
"$V/bin/python" -m pip freeze --all > "$E/pip_freeze.txt"
"$V/bin/python" -m pip inspect > "$E/pip_inspect.json"
"$V/bin/python" -c 'import sys; print(sys.executable, sys.version)' > "$E/python.txt"
du -sb "$V" > "$E/environment_disk_bytes.txt"
df -B1 "$R" > "$E/disk_after.txt"
```

그 다음 모델 호출 없이 candidate의 `verify_official_source()`·`dependency_versions()`와 실제 official Mem0 import 경로/버전, 고정 tokenizer 인코딩 및 native 기본값을 검증합니다. `MEM0_TELEMETRY=false`, `OPENAI_API_KEY=EMPTY`, HF offline, `OPENROUTER_API_KEY` 제거를 먼저 적용합니다. import/preflight의 MEM0_DIR도 이 환경 receipt 아래 독립 경로를 사용합니다. 이러한 import 검증과 상위 서비스/model/source 검증까지 모두 통과한 뒤 root가 새 recovery runtime receipt를 생성해야 합니다. 기존 receipt를 덮어쓰지 않습니다.

공식 wheel 및 실제 import된 vendor 파일들의 해시를 source_manifest와 대조합니다. 설치 report에는 배포 URL과 hash가, pip inspect/freeze에는 실제 환경 버전이 남습니다. 실패하면 기존 환경으로 fallback하지 않고 새 설치 로그를 보존합니다.

`dependency_download_estimate.json`의 CPython 3.11 / x86_64 / glibc 2.28까지 호환되는 Linux wheel 선택 기준으로 다운로드 합계는 **70,933,200 bytes(약 67.6 MiB)**이고 누락된 wheel은 없습니다. torch/nvidia/cuda 패키지는 **0개**입니다. 설치 후 점유량은 아직 측정하지 않았으며 **약 0.2–0.5 GiB**를 임시 예산으로 잡습니다. 실제 `du`가 최종 근거입니다. Qwen/MiniLM 모델 및 GPU 라이브러리는 기존 서비스가 제공하므로 이 환경에 다시 설치하지 않습니다. 로컬 Hugging Face tokenizer는 transformers/tokenizers만 사용합니다.