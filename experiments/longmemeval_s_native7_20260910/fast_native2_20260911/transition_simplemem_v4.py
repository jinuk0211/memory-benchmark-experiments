"""Load the bounded SimpleMem queue after its current question finishes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/workspace/longmemeval_s_native7_20260910')
if os.environ.get('CONTAINER_ID') != '50468468':
    raise ValueError('Wrong instance')
result=subprocess.check_output(['supervisorctl','pid','native2-simplemem'],text=True,timeout=15)
pid=int(result.strip())
if pid <= 1:
    raise ValueError('No running SimpleMem queue')
proc=Path('/proc')/str(pid)
args=(proc/'cmdline').read_bytes().split(b'\0')
if str(ROOT/'fast_native2_20260911/simplemem_queue.py').encode() not in args:
    raise ValueError('Queue identity mismatch')
children=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    try:
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        if int(stat[1])==pid: children.append(path)
    except (FileNotFoundError,ProcessLookupError): pass
receipt=ROOT/'queue/simplemem_v4_transition.json'
receipt.write_text(json.dumps({'status':'waiting_current_question','pid':pid,'started_at':time.time()}))
os.kill(pid,signal.SIGSTOP)
try:
    deadline=time.monotonic()+5400
    while any(p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z' for p in children):
        if time.monotonic()>=deadline:
            raise TimeoutError('Current question exceeded reload grace')
        time.sleep(1)
    # The child has exited: pending TERM is delivered before the frozen parent runs again.
    os.kill(pid,signal.SIGTERM)
finally:
    try: os.kill(pid,signal.SIGCONT)
    except ProcessLookupError: pass
for _ in range(30):
    if not proc.exists(): break
    time.sleep(1)
if proc.exists(): raise RuntimeError('Parent did not exit')
subprocess.run(['supervisorctl','update','native2-simplemem'],check=True,timeout=30)
receipt.write_text(json.dumps({'status':'reloaded_with_current_question_preserved','finished_at':time.time(),'previous_pid':pid}))
