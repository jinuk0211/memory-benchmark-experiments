#!/bin/bash
set -Eeuo pipefail
R=/workspace/longmemeval_s_native7_20260910
E="$R/.venv-mem0-oss0194"
P=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
trap 'printf "%s\n" "$?" > "$P/exit_code.txt"' EXIT
test ! -e "$E"
printf 'preflight\n' > "$P/phase.txt"
"$R/.venv-lightmem/bin/python" -I - <<'PY'
import sys,platform,shutil
assert sys.version_info[:3]==(3,11,16)
assert platform.libc_ver()[0]=='glibc' and tuple(map(int,platform.libc_ver()[1].split('.'))) >= (2,28)
assert shutil.disk_usage('/workspace/longmemeval_s_native7_20260910').free >= 2*1024**3
print(sys.version,platform.libc_ver())
PY
sha256sum "$R/runtime_receipt.json" "$R/source/source_snapshot.json" > "$P/protected-before.sha256"
df -B1 "$R" > "$P/disk-before.txt"
"$R/.venv-lightmem/bin/python" -I -m venv "$E"
"$E/bin/python" -I - <<'PY'
import sys,pathlib
p=pathlib.Path('/workspace/longmemeval_s_native7_20260910/.venv-mem0-oss0194')
assert sys.version_info[:3]==(3,11,16) and pathlib.Path(sys.prefix)==p
assert 'include-system-site-packages = false' in (p/'pyvenv.cfg').read_text().lower()
PY
printf 'installing_public_locked_wheels\n' > "$P/phase.txt"
"$E/bin/python" -I -m pip --isolated install --disable-pip-version-check --no-cache-dir --no-deps --force-reinstall --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --report "$P/pip-install-report.json" -r "$P/selected-linux-wheels.lock" > "$P/install.log" 2>&1
printf 'checking_dependencies\n' > "$P/phase.txt"
"$E/bin/python" -I -m pip check > "$P/pip-check.txt" 2>&1
"$E/bin/python" -I -m pip freeze --all > "$P/pip-freeze.txt"
"$E/bin/python" -I -m pip inspect > "$P/pip-inspect.json"
"$E/bin/python" -I - "$P" <<'PY'
import sys,pathlib,json,urllib.parse,hashlib
from packaging.utils import parse_wheel_filename,canonicalize_name
p=pathlib.Path(sys.argv[1]);expected=set()
for line in (p/'selected-linux-wheels.lock').read_text().splitlines():
 name,rest=line.split(' @ ',1);url,digest=rest.split(' --hash=sha256:',1)
 wheel=urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit('/',1)[-1]);distribution,version,_,_=parse_wheel_filename(wheel)
 assert canonicalize_name(name)==distribution
 expected.add((distribution,str(version),digest))
report=json.loads((p/'pip-install-report.json').read_text())
actual={(canonicalize_name(i['metadata']['name']),i['metadata']['version'],i['download_info']['archive_info']['hashes']['sha256']) for i in report['install']}
assert len(expected)==62 and expected==actual,(len(expected),len(actual),sorted(expected-actual))
assert not any(n.startswith(('nvidia-','cuda-')) or n in {'torch','torchvision','torchaudio','triton','vllm','llmlingua'} for n,_,_ in actual)
result={'status':'public_wheel_install_and_dependency_check_pass','packages':len(actual),'selected_lock_sha256':hashlib.sha256((p/'selected-linux-wheels.lock').read_bytes()).hexdigest(),'python':sys.version,'environment':sys.prefix,'native_mem0_backend_tokenizer_sdk_preflight':'PENDING','recovery_archive_transferred':False}
(p/'installation_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
sha256sum -c "$P/protected-before.sha256" > "$P/protected-after-check.txt"
du -sb "$E" > "$P/environment-bytes.txt"
df -B1 "$R" > "$P/disk-after.txt"
printf 'public_environment_installed_native_preflight_pending\n' > "$P/phase.txt"
printf 'installation_complete\n' > "$P/INSTALLATION_COMPLETE"
