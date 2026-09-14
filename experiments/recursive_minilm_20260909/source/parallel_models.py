"""Resume public HF model shards with verified parallel HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, json, os, pathlib, time
import requests
from huggingface_hub import HfApi, hf_hub_url

root = pathlib.Path('/workspace/locomo-refinement/model_ranges')
root.mkdir(exist_ok=True)
cache = pathlib.Path('/workspace/.hf_home/hub/models--Qwen--Qwen3-8B')
revision = (cache/'refs/main').read_text().strip()
info = HfApi().model_info('Qwen/Qwen3-8B',revision=revision,files_metadata=True)
files, jobs = [], []
chunk = 16*1024*1024
for entry in info.siblings:
    if not entry.rfilename.endswith('.safetensors'): continue
    sha = entry.lfs.sha256
    size = entry.size
    dest = cache/'blobs'/sha
    if dest.exists() and dest.stat().st_size == size: continue
    incomplete = dest.with_suffix('.incomplete')
    prefix = incomplete.stat().st_size if incomplete.exists() else 0
    item = {'filename':entry.rfilename, 'sha256':sha, 'size':size,'prefix':prefix,
            'url':hf_hub_url('Qwen/Qwen3-8B',entry.rfilename,revision=revision)}
    files.append(item)
    parts = root/sha
    parts.mkdir(exist_ok=True)
    for start in range(prefix,size,chunk): jobs.append((item,parts/f'{start:012d}',start,min(size-1,start+chunk-1)))
(root/'manifest.json').write_text(json.dumps({'revision':revision,'files':files},indent=2))
print('MODEL_RANGES',len(jobs),'remaining_GB',sum(f['size']-f['prefix'] for f in files)/1e9,flush=True)

def fetch(job):
    f,path,start,end=job
    size=end-start+1
    if path.exists() and path.stat().st_size==size: return
    for attempt in range(8):
        try:
            with requests.get(f['url'],headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=60) as response:
                if response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise ValueError((response.status_code,response.headers.get('Content-Range')))
                temp=path.with_suffix('.tmp')
                with temp.open('wb') as output:
                    for data in response.iter_content(1024*1024): output.write(data)
            if temp.stat().st_size != size: raise ValueError('Short range')
            temp.replace(path)
            return
        except Exception:
            if attempt==7: raise
            time.sleep(min(15,attempt+1))

started=time.time()
with ThreadPoolExecutor(max_workers=64) as pool:
    for n,future in enumerate(as_completed([pool.submit(fetch,j) for j in jobs]),1):
        future.result()
        if n%16==0: print('MODEL_PARTS',n,len(jobs),'seconds',round(time.time()-started),flush=True)
for f in files:
    destination=cache/'blobs'/f['sha256']
    incomplete=destination.with_suffix('.incomplete')
    temp=root/(f['sha256']+'.assembling')
    sha=hashlib.sha256()
    with temp.open('wb') as output:
        if f['prefix']:
            assert incomplete.stat().st_size==f['prefix'], 'Original download still writing'
            with incomplete.open('rb') as source:
                while data:=source.read(4*1024*1024): sha.update(data); output.write(data)
        for part in sorted((root/f['sha256']).glob('[0-9]*')):
            with part.open('rb') as source:
                while data:=source.read(4*1024*1024): sha.update(data); output.write(data)
    if temp.stat().st_size != f['size'] or sha.hexdigest()!=f['sha256']:
        raise ValueError('Model SHA256 mismatch: '+f['filename'])
    temp.replace(destination)
    link=cache/'snapshots'/revision/f['filename']
    if not link.exists(): link.symlink_to('../../blobs/'+f['sha256'])
    print('VERIFIED_MODEL',f['filename'],flush=True)
(root/'ready.json').write_text(json.dumps({'revision':revision,'verified':True,'seconds':time.time()-started}))
print('MODELS_READY',round(time.time()-started),flush=True)
