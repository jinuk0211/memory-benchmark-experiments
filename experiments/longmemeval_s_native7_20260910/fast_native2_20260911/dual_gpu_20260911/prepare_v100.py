"""Validate copied artifacts and start the V100 inference-only service."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time

ROOT = Path('/workspace/longmemeval_s_native7_20260910')
STATE = ROOT/'queue/dual_gpu_20260911'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def report(status, **extra):
    path=STATE/'prepare_state.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'status':status,'at':time.time(),**extra},indent=2))
    temporary.replace(path)
    print(status,flush=True)


def wait_file(name):
    while not (STATE/name).is_file():
        time.sleep(5)
    return json.loads((STATE/name).read_text())


def main():
    unpacked=STATE/'environment_unpacked.json'
    if not unpacked.exists():
        report('waiting_for_environment_archive')
        info=wait_file('environment_archive_ready.json')
        if info['destination_instance']!='50558359' or info['archive']!='inference_environment.tgz':
            raise ValueError('UnexpectedArchiveIdentity')
        archive=STATE/info['archive']
        report('verifying_environment_archive')
        if archive.stat().st_size!=info['bytes'] or digest(archive)!=info['sha256']:
            raise ValueError('ArchiveHashMismatch')
        # Model transfer must wait for this receipt to avoid archive+model peak usage.
        if shutil.disk_usage(ROOT).free < 11473784875 + 2*1024**3:
            raise RuntimeError('InsufficientSpaceForEnvironmentUnpack')
        report('unpacking_environment')
        subprocess.run(['tar','-xzf',str(archive),'-C',str(ROOT),'--no-same-owner'],check=True,timeout=900)
        unpacked.write_text(json.dumps({'status':'unpacked','archive_sha256':info['sha256'],'at':time.time()}))
        archive.unlink()
    report('waiting_for_model')
    info=wait_file('model_transfer_ready.json')
    if info.get('destination_instance')!='50558359' or info.get('revision')!='c202236235762e1c871ad0ccb60c8ee5ba337b9a':
        raise ValueError('UnexpectedModelIdentity')
    manifest=json.loads((STATE/'model_expected.json').read_text())
    model=Path(manifest['path'])
    report('verifying_model_hashes')
    actual={}
    for name,expected in manifest['files'].items():
        actual[name]=digest(model/name)
        if actual[name]!=expected:
            raise ValueError('ModelHashMismatch:'+name)
    (STATE/'model_verified.json').write_text(json.dumps({'instance_id':'50558359','status':'model_verified','files':actual,'at':time.time()},indent=2))
    python=str(ROOT/'.venv-inference/bin/python')
    report('checking_cuda_runtime')
    check=subprocess.run([python,'-c','import torch,vllm,transformers; print(torch.__version__,vllm.__version__,transformers.__version__,torch.cuda.get_device_capability()); assert torch.ones(2,device="cuda").sum().item()==2'],capture_output=True,text=True,timeout=90)
    (STATE/'cuda_check.log').write_text(check.stdout+'\n'+check.stderr)
    if check.returncode:
        raise RuntimeError('CudaCheckFailed')
    report('starting_inference_service')
    for name in ('dual-v100-qwen','dual-v100-meter'):
        result=subprocess.run(['supervisorctl','start',name],capture_output=True,text=True,timeout=40)
        print(result.stdout,flush=True)
        if result.returncode:
            raise RuntimeError('ServiceStartFailed:'+name)
    report('services_started_pending_real_generation_test')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        report('failed',error_type=type(error).__name__,error=str(error))
        raise