# Mem0 OSS 0.1.94: 공개 의존성만 설치하는 범위

**상태: 공식 메타데이터 대조 완료, 설치·원격 접근·모델 호출 미실행.** 동결된 `official_recovery/mem0_native_pairs_v1`의 기존 Linux Python 3.11 선택을 그대로 사용합니다. 재해결하거나 버전을 바꾸지 않았습니다.

- 기존 lock SHA256: `54ca0f16442a7cdd6616b94d30294959c2f291f88710e8e70727876a2fd31fd5`.
- 정확히 **62개 / 70,933,200 bytes (67.6472 MiB)**. 각 filename·version·size·SHA256을 공식 PyPI JSON API와 대조했고, 같은 패키지의 SHA가 동결 lock에 들어 있음을 확인했습니다.
- `selected-linux-wheels.lock`은 `https://files.pythonhosted.org/`의 정확한 wheel URL과 SHA만 지정합니다. Python 3.11/Linux x86_64 호환 태그이며 glibc 2.28 이상을 전제로 합니다. Torch·GPU·CUDA 패키지는 없습니다.
- `package_provenance.json`에 패키지별 공식 metadata URL, wheel URL, 바이트·해시·업로드 시각과 대조 결과를 기록했습니다. wheel 파일 자체는 다운로드하지 않았습니다.
- 새 환경의 생성기는 **`.venv-lightmem/bin/python` = CPython 3.11.16**, 대상은 **`.venv-mem0-oss0194`**입니다. 기존 `.venv-client`나 다른 환경에 설치하지 않습니다.

아래는 root가 별도로 검토할 설치 명령입니다. 이 작업에서는 실행하지 않았습니다. 기존 대상이나 receipt 경로가 있으면 중단하고, 실패한 새 환경과 로그도 보존합니다.

```bash
set -euo pipefail
R=/workspace/longmemeval_s_native7_20260910
C="$R/runtime_compatibility/mem0_oss0194_public_env_v1"
E="$R/.venv-mem0-oss0194"
P="$R/runtime_compatibility_receipts/mem0_oss0194_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$E"
test ! -e "$P"
mkdir -p "$P"
"$R/.venv-lightmem/bin/python" -I -c 'import sys; print(sys.version); assert sys.version_info[:3] == (3,11,16)' > "$P/creator-python.txt"
getconf GNU_LIBC_VERSION > "$P/glibc.txt"
# glibc >=2.28인지 확인한 뒤 계속합니다.
sha256sum "$C/selected-linux-wheels.lock" "$C/package_provenance.json" > "$P/inputs-sha256.txt"
"$R/.venv-lightmem/bin/python" -I -m venv "$E"
"$E/bin/python" -I -m pip --version > "$P/pip-version.txt"
"$E/bin/python" -I -m pip --isolated install --no-index --no-deps --require-hashes --only-binary=:all: --no-cache-dir --disable-pip-version-check --report "$P/pip-install-report.json" -r "$C/selected-linux-wheels.lock" > "$P/install.log" 2>&1
"$E/bin/python" -I -m pip check > "$P/pip-check.txt" 2>&1
"$E/bin/python" -I -m pip freeze --all > "$P/pip-freeze.txt"
```

설치 후 report의 62개 이름·버전·URL·archive SHA를 `package_provenance.json`과 대조합니다. `sys.prefix`와 설치 경로가 새 환경 안에 있고 `include-system-site-packages=false`인지 확인합니다. bootstrap pip/setuptools는 생성기의 버전을 기록하며, `pip check` 실패 시 임의 패키지 추가·업그레이드 없이 실패 원인을 보고합니다.

**이 범위는 공개 배포 패키지 설치뿐입니다.** 환경 설치가 성공해도 차단된 recovery source bundle의 전송·설치·실행은 허용되지 않습니다. 기존 동결 후보, archive, source manifest, 실행 계획을 변경하지 않으며 모델 API나 benchmark를 호출하지 않습니다.
