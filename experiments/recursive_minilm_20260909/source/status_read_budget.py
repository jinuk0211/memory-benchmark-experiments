"""Read cost/performance stage progress without source or benchmark QA text."""
import json
from pathlib import Path


def main():
    run=Path('runs/read_budget_v1');result={}
    for name in ('queue_status.json','status.json','reference_gate.json','read_budget_locked.json','source_selection_locked.json','history.json'):
        if not (run/name).exists():continue
        value=json.loads((run/name).read_text())
        if name=='source_selection_locked.json':value={k:value[k] for k in ('locked_at','selected')}
        if name=='history.json':value={**{k:v for k,v in value.items() if k!='rounds'},'rounds':[{'name':r['name'],'f1':r['dev']['official_f1'],'read_tokens':r['dev']['read_tokens'],'mean_reader_total_tokens':r['mean_reader_total_tokens'],'saving':r['reader_token_saving_fraction'],'accepted':r['accepted']} for r in value['rounds']]}
        result[name]=value
    result['source_trials']=[]
    for path in sorted((run/'source_trials').glob('*.json')):
        value=json.loads(path.read_text())['result']
        result['source_trials'].append({k:v for k,v in value.items() if k not in ('q0','fit_a')})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
