"""Fetch exact official PyPI artifacts in resumable ranges; verify full SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, json, os, pathlib, time
import requests
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

root = pathlib.Path('/workspace/locomo-refinement/wheelhouse')
root.mkdir(exist_ok=True)
tags = set(sys_tags())
torch_meta = requests.get('https://pypi.org/pypi/torch/2.8.0/json',timeout=30).json()
versions = {'torch':'2.8.0'}
for raw in torch_meta['info']['requires_dist']:
    req = Requirement(raw)
    if req.name in {'nvidia-cublas-cu12','nvidia-cudnn-cu12'}:
        versions[req.name] = next(iter(req.specifier)).version
files = []
for name, version in versions.items():
    meta = requests.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30).json()
    choices = [f for f in meta['urls'] if f['filename'].endswith('.whl') and parse_wheel_filename(f['filename'])[3] & tags]
    if len(choices) != 1:
        raise ValueError((name,[c['filename'] for c in choices]))
    files.append(choices[0])
(root/'manifest.json').write_text(json.dumps([{k:f[k] for k in ('filename','url','size','digests')} for f in files],indent=2))
chunk = 16*1024*1024
jobs = []
for f in files:
    parts = root/(f['filename']+'.parts')
    parts.mkdir(exist_ok=True)
    for start in range(0,f['size'],chunk):
        end = min(f['size']-1,start+chunk-1)
        jobs.append((f,parts/f'{start:012d}',start,end))

def fetch(job):
    f,path,start,end = job
    size = end-start+1
    if path.exists() and path.stat().st_size == size:
        return
    for attempt in range(5):
        try:
            with requests.get(f['url'],headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=45) as r:
                if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise ValueError((r.status_code,r.headers.get('Content-Range')))
                tmp = path.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    for data in r.iter_content(1024*1024): stream.write(data)
            if tmp.stat().st_size != size: raise ValueError('Short range')
            tmp.replace(path)
            return
        except Exception:
            if attempt == 4: raise
            time.sleep(1+attempt)

started = time.time()
with ThreadPoolExecutor(max_workers=24) as pool:
    futures = [pool.submit(fetch,j) for j in jobs]
    for n,future in enumerate(as_completed(futures),1):
        future.result()
        if n % 8 == 0: print('WHEEL_PARTS',n,len(jobs),'seconds',round(time.time()-started),flush=True)
for f in files:
    target = root/f['filename']
    sha = hashlib.sha256()
    with target.with_suffix('.assembling').open('wb') as output:
        for path in sorted((root/(f['filename']+'.parts')).glob('[0-9]*')):
            with path.open('rb') as source:
                while data := source.read(4*1024*1024):
                    sha.update(data); output.write(data)
    if sha.hexdigest() != f['digests']['sha256']:
        raise ValueError('SHA256 mismatch: '+f['filename'])
    target.with_suffix('.assembling').replace(target)
    print('VERIFIED',f['filename'],flush=True)
print('WHEELS_READY',round(time.time()-started),flush=True)
