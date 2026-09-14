"""Start the one-shot read-budget stage only after the prior worker releases GPU."""
from pathlib import Path
import json
import os
import subprocess
import time
import refine as core


def main():
    status=Path('runs/identity_routes_v1/status.json');queued=Path('runs/read_budget_v1/queue_status.json')
    started=time.time()
    while True:
        parent=json.loads(status.read_text()) if status.exists() else {}
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        ready=parent.get('phase')=='controlled_batch_complete' and not pids
        core.save(queued,{'pid':os.getpid(),'started_at':started,'heartbeat_at':time.time(),
                          'phase':'ready' if ready else 'waiting_for_parent_and_idle_gpu','parent_phase':parent.get('phase'),
                          'gpu_has_compute_process':bool(pids)})
        if ready:return
        supervisor=subprocess.run(['supervisorctl','status','locomo-identity-routes-refinement'],capture_output=True,text=True)
        if parent.get('phase')!='controlled_batch_complete' and any(state in supervisor.stdout for state in ('FATAL','BACKOFF','EXITED','STOPPED')):
            raise RuntimeError('Prior worker stopped before completion; do not launch a dependent GPU experiment')
        time.sleep(15)


if __name__=='__main__':main()
