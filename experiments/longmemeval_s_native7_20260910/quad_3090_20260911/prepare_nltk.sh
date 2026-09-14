#!/usr/bin/env bash
# Reconstruct the original three NLTK resources from immutable public repository bytes.
set -euo pipefail
if [[ "${1:-}" != --bounded ]]; then
  exec timeout --signal=TERM --kill-after=10s 300 bash "$0" --bounded
fi
test "${CONTAINER_ID:-}" = 50577514
unset CONTAINER_API_KEY OPENROUTER_API_KEY
export PATH=/venv/main/bin:$PATH
export NLTK_DATA=/root/.cache/longmemeval_s_native7_20260910/nltk_data
M=/workspace/quad_3090_20260911
R=/workspace/longmemeval_s_native7_20260910
python3 - "$M/nltk_recovery_files.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
from urllib.request import build_opener, ProxyHandler
import zipfile

commit = '550b6625bcef1f2abff2ff770a5a0d272c9c6b2a'
index_sha = '97dce5e72320cd9850b7c20130196006710c18f9c03134c822a37da330198bf6'
packages = (
    ('tokenizers', 'punkt', 13905355, '51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec', True),
    ('tokenizers', 'punkt_tab', 4319076, 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106', True),
    ('corpora', 'wordnet', 10775600, 'cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59', False),
)
root = Path(os.environ['NLTK_DATA'])
root.mkdir(parents=True, exist_ok=True, mode=0o700)
opener = build_opener(ProxyHandler({}))
files, sources = {}, {}
for folder, name, size, expected, unpack in packages:
    parent = root / folder
    parent.mkdir(exist_ok=True)
    target = parent / (name + '.zip')
    url = f'https://raw.githubusercontent.com/nltk/nltk_data/{commit}/packages/{folder}/{name}.zip'
    if not target.exists():
        temporary = target.with_suffix('.partial')
        with opener.open(url, timeout=30) as response, temporary.open('wb') as output:
            while chunk := response.read(1024**2):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != size or hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Public NLTK archive verification failed: {name}')
        temporary.replace(target)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if target.stat().st_size != size or actual != expected:
        raise ValueError(f'Existing NLTK archive differs; preserved for inspection: {target}')
    files[str(target.relative_to(root))] = actual
    sources[name] = {'url': url, 'archive_sha256': actual, 'archive_bytes': size}
    if unpack:
        with zipfile.ZipFile(target) as archive:
            for entry in archive.infolist():
                relative = PurePosixPath(entry.filename)
                if (relative.is_absolute() or '..' in relative.parts or
                        not relative.parts or relative.parts[0] != name or
                        stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError('Unsafe NLTK archive member')
                if entry.is_dir():
                    continue
                output = parent / entry.filename
                raw = archive.read(entry)
                if output.exists() and output.read_bytes() != raw:
                    raise ValueError(f'Existing NLTK resource differs; preserved: {output}')
                if not output.exists():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_name(output.name + '.tmp')
                    temporary.write_bytes(raw)
                    temporary.replace(output)
                files[str(output.relative_to(root))] = hashlib.sha256(raw).hexdigest()
receipt = {'status': 'corpus_files_verified', 'instance_id': '50577514',
           'repository_commit': commit, 'source_index_sha256': index_sha,
           'provenance': 'Public reconstruction at repository revision before original runtime receipt; original server corpus hashes unavailable',
           'data_path': str(root), 'sources': sources, 'files_sha256': files, 'verified_at': time.time()}
path = Path(sys.argv[1])
temporary = path.with_suffix('.tmp')
with temporary.open('w') as output:
    json.dump(receipt, output, indent=2)
    output.flush()
    os.fsync(output.fileno())
temporary.replace(path)
print('Pinned punkt, punkt_tab and wordnet bytes verified; no tokenizer fallback installed.', flush=True)
PY
P="$R/.venv-lightmem/bin/python"
if [[ -x "$P" ]] && "$P" -c 'import nltk' >/dev/null 2>&1; then
  "$P" - "$M/nltk_recovery_probe.json" <<'PY'
import json
import os
from pathlib import Path
import sys
import time
import nltk

assert nltk.__version__ == '3.9.2', nltk.__version__
nltk.data.path[:] = [os.environ['NLTK_DATA']]
nltk.data.find('tokenizers/punkt')
nltk.data.find('tokenizers/punkt_tab/english/')
nltk.data.find('corpora/wordnet.zip')
sample = 'Dr. Smith arrived. He remembered the meeting.'
sentences = nltk.sent_tokenize(sample)
tokens = nltk.word_tokenize(sample)
assert sentences == ['Dr. Smith arrived.', 'He remembered the meeting.']
assert tokens == ['Dr.', 'Smith', 'arrived', '.', 'He', 'remembered', 'the', 'meeting', '.']
assert nltk.corpus.wordnet.synsets('memory')
path = Path(sys.argv[1])
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps({'status': 'nltk_native_probe_verified',
    'nltk_version': nltk.__version__, 'data_path': nltk.data.path[0],
    'sentences': sentences, 'tokens': tokens, 'english_wordnet_available': True,
    'verified_at': time.time()}, indent=2))
temporary.replace(path)
print('Pinned LightMem NLTK native tokenization and English WordNet probes passed.', flush=True)
PY
else
  printf 'Corpus files ready; rerun after pinned LightMem environment installation for native NLTK probes.\n'
fi
