"""Fetch a manifest-pinned contemporary checkpoint with verified HTTP ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import time

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
import requests

HUB = Path("/workspace/.hf_home/hub")
CHUNK = 16 * 1024 * 1024


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_range(job: tuple[str, Path, int, int, int, str, str]) -> None:
    """Reject ignored ranges before reading a potentially full-shard response."""
    name, path, start, end, total, model, revision = job
    size = end - start + 1
    if path.exists() and path.stat().st_size == size:
        return
    temporary = path.with_suffix(".tmp")
    url = hf_hub_url(model, name, revision=revision)
    for attempt in range(6):
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                stream=True,
                timeout=(20, 60),
            ) as response:
                expected = f"bytes {start}-{end}/{total}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError(f"Invalid range response: {response.status_code}, {response.headers.get('Content-Range')}")
                written = 0
                with temporary.open("wb") as output:
                    for data in response.iter_content(1024 * 1024):
                        written += len(data)
                        if written > size:
                            raise ValueError("Response exceeds declared range")
                        output.write(data)
            if written != size:
                raise ValueError("Short range response")
            temporary.replace(path)
            return
        except (requests.RequestException, OSError, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(10, attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    locked = json.loads((directory / 'download_lock.json').read_text())
    model = args.model
    entry = locked['models'][model]
    revision = entry['revision']
    expected_files = {name: tuple(values) for name, values in entry['weights'].items()}
    cache = HUB / ('models--' + model.replace('/', '--'))
    root = directory / ('ranges--' + model.replace('/', '--'))
    root.mkdir(exist_ok=True)
    info = HfApi(token=False).model_info(model, revision=revision, files_metadata=True)
    actual = {
        item.rfilename: (item.size, item.lfs.sha256)
        for item in info.siblings
        if item.rfilename in expected_files
    }
    if info.sha != revision or actual != expected_files:
        raise ValueError("Public metadata differs from pinned size/SHA256 manifest")
    manifest = {"model": model, "revision": revision, "files": expected_files, "workers": 64}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lock_dir = HUB / ".locks" / cache.name
    lock_dir.mkdir(parents=True, exist_ok=True)
    (cache / "blobs").mkdir(parents=True, exist_ok=True)
    snapshot = cache / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        locks.enter_context(FileLock(str(root / "download.lock"), timeout=0))
        for _, checksum in expected_files.values():
            locks.enter_context(FileLock(str(lock_dir / f"{checksum}.lock"), timeout=0))
        pending = []
        for name, (size, checksum) in expected_files.items():
            destination = cache / "blobs" / checksum
            if destination.exists():
                if destination.stat().st_size != size or digest(destination) != checksum:
                    raise ValueError(f"Existing completed blob is invalid: {name}")
            else:
                pending.append((name, size, checksum))
        if shutil.disk_usage(root).free < 2 * sum(size for _, size, _ in pending):
            raise ValueError("Insufficient free disk for parts and verified assembly")
        jobs = []
        for name, size, checksum in pending:
            parts = root / checksum
            parts.mkdir(exist_ok=True)
            for start in range(0, size, CHUNK):
                jobs.append((name, parts / f"{start:012d}", start, min(size - 1, start + CHUNK - 1), size, model, revision))
        started = time.monotonic()
        print(f"RANGES_START parts={len(jobs)} bytes={sum(size for _, size, _ in pending)}", flush=True)
        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = {pool.submit(fetch_range, job): job[3] - job[2] + 1 for job in jobs}
            completed_bytes = 0
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    future.result()
                    completed_bytes += futures[future]
                    if count % 16 == 0 or count == len(jobs):
                        elapsed = time.monotonic() - started
                        print(f"RANGES_PROGRESS {count}/{len(jobs)} seconds={elapsed:.1f} completed_MBps={completed_bytes / elapsed / 1e6:.2f}", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        for name, size, checksum in pending:
            temporary = root / f"{checksum}.assembling"
            hasher = hashlib.sha256()
            with temporary.open("wb") as output:
                for start in range(0, size, CHUNK):
                    with (root / checksum / f"{start:012d}").open("rb") as part:
                        while data := part.read(4 * 1024 * 1024):
                            hasher.update(data)
                            output.write(data)
            if temporary.stat().st_size != size or hasher.hexdigest() != checksum:
                raise ValueError(f"Assembled SHA256/size mismatch: {name}")
            temporary.replace(cache / "blobs" / checksum)
            print(f"SHARD_VERIFIED {name} sha256={checksum}", flush=True)
            parts_path = root / checksum
            if parts_path.resolve().parent != root.resolve() or parts_path.name != checksum:
                raise ValueError('Unexpected parts cleanup path')
            shutil.rmtree(parts_path)
        for name, (_, checksum) in expected_files.items():
            link = snapshot / name
            destination = cache / "blobs" / checksum
            if link.is_symlink():
                if link.resolve() != destination.resolve():
                    raise ValueError(f"Unexpected existing snapshot target: {name}")
            elif link.exists():
                if digest(link) != checksum:
                    raise ValueError(f"Unexpected snapshot file: {name}")
            else:
                link.symlink_to(f"../../blobs/{checksum}")
        snapshot_download(model, revision=revision, token=False, cache_dir=HUB,
                          allow_patterns=entry['assets'], max_workers=8)
        if any(not (snapshot / name).is_file() for name in entry['assets']):
            raise ValueError('Pinned configuration/tokenizer assets are incomplete')
        environment = json.loads((directory.parent / 'source/environment.json').read_text())
        environment['models'][model] = {'path': str(snapshot), 'revision': revision}
        environment['packages_note'] = 'Source packages are provenance only; target actual packages are recorded by each run.'
        environment_path = directory / ('environment--' + model.replace('/', '--') + '.json')
        staged = environment_path.with_suffix('.tmp')
        staged.write_text(json.dumps(environment, indent=2))
        staged.replace(environment_path)
        staged_ready = root / 'ready.tmp'
        staged_ready.write_text(json.dumps({**manifest, 'assets': entry['assets'],
                                           'verified': True, 'seconds': time.monotonic() - started}, indent=2))
        staged_ready.replace(root / 'ready.json')
        print(f"model_SHARDS_READY {snapshot}", flush=True)


if __name__ == "__main__":
    main()


