"""Download pinned runtime wheels in verified ranges; preserve complete pip files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time

from filelock import FileLock
import requests

CHUNK = 16 * 1024 * 1024


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid(path, entry):
    return path.is_file() and path.stat().st_size == entry['size'] and digest(path) == entry['sha256']


def fetch_range(job):
    url, path, start, end, total = job
    size = end - start + 1
    if path.is_file() and path.stat().st_size == size:
        return
    temporary = path.with_suffix('.tmp')
    for attempt in range(6):
        try:
            with requests.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                              stream=True, timeout=(20, 90)) as response:
                expected = f'bytes {start}-{end}/{total}'
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError(f'Invalid range: {response.status_code}, {response.headers.get("Content-Range")}')
                written = 0
                with temporary.open('wb') as output:
                    for data in response.iter_content(1024 * 1024):
                        written += len(data)
                        if written > size:
                            raise ValueError('Oversized response')
                        output.write(data)
            if written != size:
                raise ValueError('Short range response')
            temporary.replace(path)
            return
        except (requests.RequestException, OSError, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(10, attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preserve-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = root / 'cuda_wheels.json'
    entries = json.loads(manifest.read_text())
    names = [entry['filename'] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate wheel filename')
    for entry in entries:
        if Path(entry['filename']).name != entry['filename'] or not entry['filename'].endswith('.whl'):
            raise ValueError('Unsafe wheel filename')
        if len(entry['sha256']) != 64 or any(c not in '0123456789abcdef' for c in entry['sha256']):
            raise ValueError('Invalid SHA256')
    wheels, ranges = root / 'wheelhouse', root / 'wheel_ranges'
    wheels.mkdir(exist_ok=True)
    ranges.mkdir(exist_ok=True)
    with FileLock(str(root / 'runtime_wheels.lock'), timeout=0):
        pending = []
        for entry in entries:
            destination = wheels / entry['filename']
            if destination.exists():
                if not valid(destination, entry):
                    raise ValueError(f'Invalid completed wheel: {destination.name}')
                continue
            for candidate in Path('/tmp').glob('pip-unpack-*/' + entry['filename']):
                if valid(candidate, entry):
                    temporary = wheels / (entry['filename'] + '.copying')
                    shutil.copyfile(candidate, temporary)
                    if not valid(temporary, entry):
                        raise ValueError('Preserved wheel failed verification')
                    temporary.replace(destination)
                    print(f'PRESERVED {destination.name}', flush=True)
                    break
            if not destination.exists():
                pending.append(entry)
        print(f'PENDING wheels={len(pending)} bytes={sum(e["size"] for e in pending)}', flush=True)
        if args.preserve_only:
            return
        if shutil.disk_usage(root).free < 2 * sum(e['size'] for e in pending):
            raise ValueError('Insufficient disk for ranges and assembly')
        jobs = []
        for entry in pending:
            parts = ranges / entry['sha256']
            parts.mkdir(exist_ok=True)
            for start in range(0, entry['size'], CHUNK):
                jobs.append((entry['url'], parts / f'{start:012d}', start,
                             min(start + CHUNK, entry['size']) - 1, entry['size']))
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = [pool.submit(fetch_range, job) for job in jobs]
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if count % 16 == 0 or count == len(jobs):
                        print(f'WHEEL_RANGES {count}/{len(jobs)} seconds={time.monotonic() - started:.1f}', flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        for entry in pending:
            parts = ranges / entry['sha256']
            temporary = wheels / (entry['filename'] + '.assembling')
            with temporary.open('wb') as output:
                for start in range(0, entry['size'], CHUNK):
                    with (parts / f'{start:012d}').open('rb') as stream:
                        shutil.copyfileobj(stream, output, length=4 * 1024 * 1024)
            if not valid(temporary, entry):
                raise ValueError(f'Assembled wheel failed SHA256/size: {entry["filename"]}')
            temporary.replace(wheels / entry['filename'])
            if parts.resolve().parent != ranges.resolve():
                raise ValueError('Unsafe parts cleanup path')
            shutil.rmtree(parts)
            print(f'WHEEL_VERIFIED {entry["filename"]}', flush=True)
        receipt = {'manifest_sha256': digest(manifest), 'verified_wheels': names}
        temporary = root / 'runtime_wheels_ready.tmp'
        temporary.write_text(json.dumps(receipt, indent=2))
        temporary.replace(root / 'runtime_wheels_ready.json')


if __name__ == '__main__':
    main()
