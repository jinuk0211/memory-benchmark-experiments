#!/usr/bin/env python3
"""Download the frozen model inventory once, with resumable verified files."""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

PRINT_LOCK = threading.Lock()
CHUNK = 1024 * 1024
CACHE = Path('/workspace/.hf_home/hub')


def report(**event):
    with PRINT_LOCK:
        print(json.dumps(event, sort_keys=True), flush=True)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * CHUNK):
            value.update(chunk)
    return value.hexdigest()


def verified(path, size, sha256):
    return path.is_file() and path.stat().st_size == size and digest(path) == sha256


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


def allowed_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ''
    return (parsed.scheme == 'https' and not parsed.username and not parsed.password
            and parsed.port in (None, 443)
            and (host == 'huggingface.co' or host.endswith('.huggingface.co')
                 or host == 'hf.co' or host.endswith('.hf.co')))


def open_download(url, token, offset, timeout=60):
    """Follow only HF-owned HTTPS redirects; authenticate only the Hub origin."""
    opener = urllib.request.build_opener(NoRedirect())
    deadline = time.monotonic() + timeout
    for _ in range(8):
        if not allowed_url(url):
            raise ValueError('Download redirect left approved Hugging Face hosts')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Download connection deadline reached')
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'native-memory-pinned-download/1',
                   'Cache-Control': 'no-cache'}
        if urllib.parse.urlsplit(url).hostname == 'huggingface.co' and token:
            headers['Authorization'] = 'Bearer ' + token
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url, headers=headers)
        try:
            return opener.open(request, timeout=remaining)
        except urllib.error.HTTPError as error:
            location = error.headers.get('Location')
            if error.code not in (301, 302, 303, 307, 308) or not location:
                error.close()
                raise
            url = urllib.parse.urljoin(url, location)
            error.close()
    raise ValueError('Too many download redirects')


def transfer(response, partial, offset, size, deadline, label):
    encoding = response.headers.get('Content-Encoding', 'identity')
    if encoding != 'identity':
        raise ValueError('Unexpected compressed response')
    status = response.status
    if status == 206:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if not match:
            raise ValueError('Missing or invalid Content-Range')
        start, end, total = map(int, match.groups())
        if start != offset or total != size or not start <= end < size:
            raise ValueError('Content-Range does not match the pinned file and offset')
        expected_body = end - start + 1
        mode = 'ab' if offset else 'wb'
    elif status == 200:
        # A server may ignore Range. Never append a full response to partial bytes.
        expected_body = size
        mode, offset = 'wb', 0
    else:
        raise ValueError('Unexpected download status')
    length = response.headers.get('Content-Length')
    if length is not None and int(length) != expected_body:
        raise ValueError('Content-Length does not match the pinned file')
    consumed = 0
    last_report = time.monotonic()
    reader = getattr(response, 'read1', response.read)
    with partial.open(mode) as output:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Download attempt deadline reached')
            chunk = reader(CHUNK)
            if not chunk:
                break
            if consumed + len(chunk) > expected_body:
                raise ValueError('Response exceeds pinned size')
            output.write(chunk)
            consumed += len(chunk)
            if time.monotonic() - last_report >= 30:
                output.flush()
                os.fsync(output.fileno())
                report(event='download_progress', file=label, bytes=offset + consumed, total=size)
                last_report = time.monotonic()
        output.flush()
        os.fsync(output.fileno())
    if consumed != expected_body or partial.stat().st_size != size:
        raise OSError('Incomplete response retained for resume')


def download_file(item, token, attempts=4, attempt_seconds=1800,
                  opener=open_download, retry_delay=5, reserve_bytes=3 * 1024**3):
    target = item['target']
    size, sha256 = item['bytes'], item['sha256']
    label = item['model'] + '/' + item['name']
    target.parent.mkdir(parents=True, exist_ok=True)
    if verified(target, size, sha256):
        report(event='file_verified', file=label, bytes=size, reused=True)
        return {'sha256': sha256, 'bytes': size, 'reused': True}
    partial = target.with_name(target.name + '.partial')
    for attempt in range(1, attempts + 1):
        try:
            if partial.exists() and partial.stat().st_size >= size:
                if verified(partial, size, sha256):
                    partial.replace(target)
                    report(event='file_verified', file=label, bytes=size, reused=True)
                    return {'sha256': sha256, 'bytes': size, 'reused': True}
                # Only our exact unverified partial is discarded, never a valid final file.
                partial.unlink()
            if shutil.disk_usage(target.parent).free < reserve_bytes:
                raise OSError('Insufficient download disk reserve')
            offset = partial.stat().st_size if partial.exists() else 0
            report(event='download_attempt', file=label, attempt=attempt, offset=offset, total=size)
            deadline = time.monotonic() + attempt_seconds
            with opener(item['url'], token, offset, min(60, attempt_seconds)) as response:
                transfer(response, partial, offset, size, deadline, label)
            if not verified(partial, size, sha256):
                partial.unlink()
                raise ValueError('Downloaded SHA256 does not match pinned manifest')
            partial.replace(target)
            report(event='file_verified', file=label, bytes=size, reused=False)
            return {'sha256': sha256, 'bytes': size, 'reused': False}
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
            # Exception strings can include signed URLs. Log only safe classification.
            report(event='download_retry' if attempt < attempts else 'download_failed',
                   file=label, attempt=attempt, error_type=type(error).__name__,
                   retained_bytes=partial.stat().st_size if partial.exists() else 0)
            if attempt == attempts:
                raise RuntimeError(f'Download failed: {label}; {type(error).__name__}') from None
            time.sleep(retry_delay)
    raise AssertionError('Unreachable download attempt loop')


def inventory(manifest):
    files = []
    targets = set()
    for name, model in manifest['models'].items():
        provenance = model['provenance']
        repo, revision = provenance['repository'], provenance['revision']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
            raise ValueError('Invalid repository in manifest')
        if not re.fullmatch(r'[a-f0-9]{40}', revision):
            raise ValueError('Model revision is not a pinned commit')
        snapshot = CACHE / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        if Path(model['path']) != snapshot:
            raise ValueError('Manifest model path differs from pinned cache destination')
        metadata = dict(provenance['weights'])
        metadata.update(provenance['metadata'])
        for filename, sha256 in model['files'].items():
            relative = PurePosixPath(filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in filename:
                raise ValueError('Invalid model filename')
            detail = metadata[filename]
            size = detail['bytes']
            if detail['sha256'] != sha256 or not re.fullmatch(r'[a-f0-9]{64}', sha256):
                raise ValueError('Conflicting or invalid pinned SHA256')
            if not isinstance(size, int) or size <= 0:
                raise ValueError('Missing pinned file size')
            target = snapshot / filename
            if target in targets:
                raise ValueError('Duplicate model download destination')
            targets.add(target)
            files.append({'model': name, 'name': filename, 'target': target,
                          'bytes': size, 'sha256': sha256,
                          'url': 'https://huggingface.co/' + repo + '/resolve/' + revision
                                 + '/' + urllib.parse.quote(filename, safe='/')})
    return sorted(files, key=lambda item: item['bytes'], reverse=True)


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 5))
    parser.add_argument('--attempts', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--attempt-seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 60 <= args.attempt_seconds <= 7200:
        parser.error('--attempt-seconds must be between 60 and 7200')
    os.environ.pop('CONTAINER_API_KEY', None)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if os.name != 'posix':
        parser.error('Production downloads require the Linux server lock')
    import fcntl
    with (args.receipt.parent / 'model_download.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        token_stat = args.token_file.stat()
        if stat.S_IMODE(token_stat.st_mode) != 0o600 or token_stat.st_uid != os.getuid():
            raise ValueError('HF token must be an owned mode 0600 file')
        token = args.token_file.read_text(encoding='utf-8').strip()
        if not token.startswith('hf_') or any(character.isspace() for character in token):
            raise ValueError('Invalid HF token file')
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        files = inventory(manifest)
        started = time.time()
        result = {'status': 'downloading', 'started_at': started,
                  'manifest_sha256': digest(args.manifest), 'models': {}, 'errors': []}
        save_json(args.receipt, result)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_file, item, token, args.attempts,
                                   args.attempt_seconds): item for item in files}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    proof = future.result()
                except Exception as error:
                    result['errors'].append({'file': item['model'] + '/' + item['name'],
                                             'error_type': type(error).__name__})
                else:
                    model = manifest['models'][item['model']]
                    entry = result['models'].setdefault(item['model'], {
                        'path': model['path'], 'revision': model['provenance']['revision'],
                        'files_sha256': {}, 'files_bytes': {}})
                    entry['files_sha256'][item['name']] = proof['sha256']
                    entry['files_bytes'][item['name']] = proof['bytes']
                save_json(args.receipt, result)
        result['status'] = 'needs_attention' if result['errors'] else 'model_files_verified'
        result['completed_at'] = time.time()
        result['verified_files'] = sum(len(model['files_sha256']) for model in result['models'].values())
        result['verified_bytes'] = sum(sum(model['files_bytes'].values()) for model in result['models'].values())
        save_json(args.receipt, result)
        report(event=result['status'], verified_files=result['verified_files'],
               verified_bytes=result['verified_bytes'])
        return int(bool(result['errors']))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as failure:
        report(event='model_download_failed', error_type=type(failure).__name__)
        raise SystemExit(1) from None
