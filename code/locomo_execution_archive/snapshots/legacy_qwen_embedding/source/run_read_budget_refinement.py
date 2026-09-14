"""Controlled query-token frontier of the fixed strongest memory."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import refine as core
from read_budget import evaluate,assess
from retrieval_contract import qualified,compare_contract
from crossview_probes import require_fit
from budgeted_evidence import storage_cost
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='runs/read_budget_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B');ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4);ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048);args=ap.parse_args()
    out=Path(args.out);previous=Path('runs/identity_routes_v1')
    assert read(previous/'status.json')['phase']=='controlled_batch_complete'
    previous_protocol=read(previous/'protocol.json')
    for name,digest in previous_protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest
    manifest=read('runs/pilot100_v3/manifest.json');records=[r for r in manifest['records'] if r['split']=='dev']
    cids=sorted({r['conv_id'] for r in records});samples=read('data/locomo10.json')
    assert core.digest(samples)==manifest['dataset_sha256']
    byid={str(s['sample_id']):s for s in samples if str(s['sample_id']) in cids};del samples
    sessions={cid:core.session_data(s) for cid,s in byid.items()}
    assert sessions==read('research_data/source_sessions.json')['sessions_by_id']
    assert not set(cids)&set(manifest['holdout_convs'])
    baseline={cid:read(Path('runs/parent_evidence_v1/memories/s_parent_single_2000')/f'{cid}.json') for cid in cids}
    fit=read('runs/crossview_refinement_v1/query_views/fit_views.json');require_fit(fit);ids={r['id'] for r in fit}
    q0=sorted([r for r in qualified([read(p) for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json')]) if r['id'] in ids],key=lambda r:r['id'])
    assert len(q0)==len(fit)==75 and {r['id'] for r in q0}==ids
    source_names=('run_read_budget_refinement.py','read_budget.py','evaluate_indexed.py','refine.py','retrieval_contract.py',
                  'crossview_probes.py','evidence_fidelity.py','evidence_utility.py','budgeted_evidence.py','run_evidence_utility.py')
    hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_names}
    recipes=[{'name':f'ad_read_{budget}','family':'memory_read_budget','read_budget':budget} for budget in (1536,1024)]
    expected=.5927437370002835
    protocol={'args':vars(args),'source_sha256':hashes,'environment':read('environment.json'),'records':records,
              'previous_protocol_sha256':core.digest(previous_protocol),'baseline_memory_sha256':core.digest(baseline),
              'source_q0_digest':core.digest(q0),'source_fit_view_digest':core.digest(fit),
              'excluded_holdout_conversations':manifest['holdout_convs'],'reader':core.READER,'recipes':recipes,
              'reference':{'name':'s_parent_single_2000','expected_f1':expected,'read_budget':2048},
              'construction':'The fixed strongest memory, text keys, dense vectors, BM25, RRF and unit order are unchanged. Only the packer read-token cap changes to 1536/1024; output maximum remains 96. No memory compression or storage reduction is claimed in this stage.',
              'token_accounting':'Use exact served prompt_token_ids and output token_ids from each reader generation cache, including system instructions, chat framing and the question. Record context-only read tokens separately. Logical served tokens are not a billing or GPU-runtime estimate; shared cached answers are not new inference.',
              'selection':'After policies are locked, use matched source Q0/A75. Admit a lower cap only with zero observed source-answer regressions in both views and fewer mean source reader total tokens. Choose the smallest source total tokens, then smaller cap; otherwise keep 2048. Lock before dev scores.',
              'dev_frontier':'Report every cap. A development candidate can replace the reference only if official F1 does not decrease (1e-12 float tolerance) and mean total reader tokens strictly decreases. Report source choice separately. No audit100/B used.',
              'limits':'A rate/performance diagnostic, not an academic novelty claim. Previously reused dev70 is not fresh generalization. Source equivalence uses the same reader model. Dataset JSON is parsed then QA is stripped from constructor inputs.'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    for path in (previous/'cache').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.npy'):
            target=out/'cache'/path.relative_to(previous/'cache');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():os.link(path,target)
    status={'pid':os.getpid(),'started_at':time.time()}
    def state(phase,**fields):core.save(out/'status.json',{**status,'phase':phase,'heartbeat_at':time.time(),**fields})
    state('model_load');rt=core.Runtime(args,protocol['environment'])
    replay=evaluate(rt,baseline,records,byid,2048,'reference_replay',out)
    keys=('id','question','context','prediction','source_ids','read_tokens','memory_tokens','official_f1')
    assert [{k:r[k] for k in keys} for r in replay]==[{k:r[k] for k in keys} for r in read(previous/'reference_replay_dev_items.json')]
    assert abs(core.summarize(replay)['official_f1']-expected)<1e-12
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':expected,'all_70_contexts_and_predictions_identical':True})
    for cid in cids:
        core.save(out/'base_memories'/f'{cid}.json',baseline[cid])
        vectors=rt.encode([u.get('index_text',u['text']) for u in baseline[cid]])
        path=out/'indexes/baseline'/f'{cid}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,vectors)
        for recipe in recipes:
            name=recipe['name'];core.save(out/'memories'/name/f'{cid}.json',baseline[cid])
            cost=sum(storage_cost(u,rt.ntok) for u in baseline[cid])
            core.save(out/'construction'/name/f'{cid}.json',{'storage_tokens':cost,'storage_cap':cost,'read_budget':recipe['read_budget'],'dense_index_bytes':vectors.nbytes,'memory_unchanged':True})
    for recipe in recipes:core.save(out/'plans'/f"{recipe['name']}.json",{'recipe':recipe,'source_sha256':hashes,'committed_at':time.time()})
    core.save(out/'read_budget_locked.json',{'locked_at':time.time(),'memory_sha256':core.digest(baseline),'recipes':recipes})
    state('source_baseline_assessment')
    baseq,bqt=assess(rt,baseline,sessions,q0,'source_baseline',out,2048,'q0')
    basea,bat=assess(rt,baseline,sessions,fit,'fit_baseline',out,2048,'fit_a')
    core.save(out/'source_baseline_contract.json',{'q0':baseq,'fit_a':basea,'q0_tokens':bqt,'fit_a_tokens':bat});feasible=[]
    for recipe in recipes:
        name=recipe['name'];budget=recipe['read_budget'];state('source_budget_assessment',round=name)
        cq,qt=assess(rt,baseline,sessions,q0,name,out,budget,'q0')
        ca,at=assess(rt,baseline,sessions,fit,name,out,budget,'fit_a')
        dq,da=compare_contract(baseq,cq),compare_contract(basea,ca)
        total=(qt['mean_reader_total_tokens']+at['mean_reader_total_tokens'])/2
        result={'name':name,'read_budget':budget,'q0':dq,'fit_a':da,'mean_source_reader_total_tokens':total,
                'feasible':not dq['regressions'] and not da['regressions'] and total<(bqt['mean_reader_total_tokens']+bat['mean_reader_total_tokens'])/2}
        core.save(out/'source_trials'/f'{name}.json',{'result':result,'q0_contract':cq,'fit_a_contract':ca,'q0_tokens':qt,'fit_a_tokens':at})
        if result['feasible']:feasible.append(result)
        print('READ_BUDGET_SOURCE '+str(result),flush=True)
    choice=min(feasible,key=lambda r:(r['mean_source_reader_total_tokens'],r['read_budget'])) if feasible else None
    core.save(out/'source_selection_locked.json',{'locked_at':time.time(),'selected':choice['name'] if choice else 's_parent_single_2000',
              'result':choice,'memory_sha256':core.digest(baseline),'selection_used_new_dev_outcomes':False,'selection_used_audit_outcomes':False})
    mean_tokens=lambda rows:sum(r['reader_total_tokens'] for r in rows)/len(rows)
    reference_tokens=mean_tokens(replay)
    history={'rounds':[],'best_f1':expected,'winner':'s_parent_single_2000','best_mean_reader_total_tokens':reference_tokens,'reference_mean_reader_total_tokens':reference_tokens}
    for recipe in recipes:
        name=recipe['name'];state('dev_evaluation',round=name)
        rows=evaluate(rt,baseline,records,byid,recipe['read_budget'],name,out);summary=core.summarize(rows);tokens=mean_tokens(rows)
        accepted=summary['official_f1']>=history['best_f1']-1e-12 and tokens<history['best_mean_reader_total_tokens']
        history['rounds'].append({'name':name,'recipe':recipe,'dev':summary,'mean_reader_total_tokens':tokens,
                                  'reader_token_saving_fraction':1-tokens/reference_tokens,'accepted':accepted})
        if accepted:history.update(winner=name,best_f1=summary['official_f1'],best_mean_reader_total_tokens=tokens)
        core.save(out/'history.json',history);print('READ_BUDGET_DEV '+str(history['rounds'][-1]),flush=True)
    state('controlled_batch_complete',winner=history['winner'],best_f1=history['best_f1'],next='Continue storage/rate reduction with no F1 loss; consumed audit100/B remain excluded.')


if __name__=='__main__':main()
