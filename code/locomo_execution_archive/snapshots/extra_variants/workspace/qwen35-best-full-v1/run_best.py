"""Rebuild the fixed best policy on Qwen3.5-9B; evaluate every original QA."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import os
import time
import traceback

import refine as core
from portable_parent import backend,augment,generate_source_probes
from evidence_utility import POLICY,subset_jobs,subset_analysis,generation_controls
from evaluate_indexed import evaluate
from api_runtime import Runtime,Scorer,read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/qwen35_full_v1')
    ap.add_argument('--dataset',default='/workspace/HiGMem/data/locomo10.json')
    ap.add_argument('--base-url',default='http://127.0.0.1:18080')
    ap.add_argument('--model',default='Qwen/Qwen3.5-9B')
    ap.add_argument('--model-path',default='/workspace/.hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-path',default='/workspace/.hf_home/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--api-batch',type=int,default=8)
    ap.add_argument('--api-workers',type=int,default=2)
    ap.add_argument('--cpu-threads',type=int,default=8)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    if args.model!='Qwen/Qwen3.5-9B' or args.budget!=2048:raise ValueError('This run fixes model and best-policy budget')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    bundle=read('policy_bundle.json')
    all_samples=read(args.dataset)
    dataset_sha=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    assert dataset_sha==bundle['dataset_sha256']
    cids=sorted(str(s['sample_id']) for s in all_samples)
    assert len(cids)==10
    source_sessions={str(s['sample_id']):core.session_data(s) for s in all_samples}
    records=[{'id':f"{s['sample_id']}:{i}",'conv_id':str(s['sample_id']),'qa_index':i,
              'category':q['category'],'split':'full1540'}
             for s in all_samples for i,q in enumerate(s['qa']) if q['category'] in (1,2,3,4)]
    assert len(records)==1540 and len({r['id'] for r in records})==1540
    # Only category and positional identifiers survive this preparation boundary.
    del all_samples
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('.').glob('*.py'))}
    for name,value in bundle['frozen_core_sha256'].items():assert sources[name]==value
    protocol={'method':'s_parent_single_2000','args':vars(args),'dataset_sha256':dataset_sha,
              'source_sha256':sources,'policy_bundle':bundle,'utility_policy':POLICY,'records':records,
              'generation_model':{'model':args.model,'path':args.model_path,'revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','dtype':'float16'},
              'embedding_model':{'model':args.embed_model,'path':args.embed_path,'revision':'97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3','device':'cpu'},
              'frozen_method':'session10 -> audit -> dialogue_residual -> r06 -> r12 -> r40 -> source-fit parent_single +2000',
              'selection':'No benchmark-based candidate selection; the already selected policy is fixed before generation.',
              'scope':'Rebuild all memory and source probes with Qwen3.5-9B, then score all 1540 category1-4 QA. Source-only fit/session-audit split is applied separately to the original dev7 and pilot-holdout3 groups.',
              'execution':'Existing shared FP16 Qwen3.5-9B server; identical prompts, limits and selection logic; Qwen embedding retained on CPU to avoid changing the running GPU service. Prefix caching is enabled on the existing server, including verified prompt logprob support.',
              'token_accounting':'Actual API usage for every generation/NLL request including failures when available. Unknown failed-request usage stays unknown. Embedding input tokens and storage tokens separately measured. API preflight is separate overhead.',
              'context_limits':{'memory_read_tokens':2048,'constructor_prompt_plus_output':8192,'server_context':32768},
              'created_at':None}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert read(protocol_path)==protocol
    else:core.save(protocol_path,protocol)
    core.save(out/'source_sessions.json',{'dataset_sha256':dataset_sha,'sessions_by_id':source_sessions})
    core.save(out/'question_manifest.json',{'dataset_sha256':dataset_sha,'records':records})
    started=time.time()
    rt=Runtime(args,protocol)
    def state(phase,cid=None,**fields):
        rt.phase=phase;rt.conversation=cid
        core.save(out/'status.json',{'phase':phase,'conversation':cid,'pid':os.getpid(),
                 'invocation_started_at':started,'heartbeat_at':time.time(),**fields})
        print(json.dumps({'phase':phase,'conversation':cid,**fields}),flush=True)
    backends={}
    for i,cid in enumerate(cids):
        sessions=source_sessions[cid]
        target=out/'memories/r40_fused_four_turn'/f'{cid}.json'
        if target.exists():backends[cid]=read(target);continue
        memory=None
        for strategy in ('session10','audit','dialogue_residual'):
            path=out/'memories'/strategy/f'{cid}.json'
            if path.exists():memory=read(path)
            else:
                state('construction_'+strategy,cid,conversation_number=i+1)
                memory=core.build(rt,sessions,strategy,memory)
                core.save(path,memory)
        state('construction_r40',cid,conversation_number=i+1)
        memory=backend(rt,sessions,memory,bundle['plans'])
        core.save(target,memory);backends[cid]=memory
    pool_path=out/'source_probes.json'
    if pool_path.exists():pool=read(pool_path)
    else:
        pools=[];dumps=[]
        for label,group in bundle['source_groups'].items():
            state('construction_source_questions',label)
            dump,pool=generate_source_probes(rt,{cid:source_sessions[cid] for cid in group},
                           bundle['plans']['r24_source_qa_cards']['recipe']['operations'][0])
            pools.append(pool);dumps.extend(dump['records'])
        pool={'records':[r for p in pools for r in p['records']],
              'counts':{cid:v for p in pools for cid,v in p['counts'].items()},
              'audit_sessions':{cid:v for p in pools for cid,v in p['audit_sessions'].items()},
              'source_groups':bundle['source_groups'],'seed':20260908}
        assert len({r['id'] for r in pool['records']})==len(pool['records'])
        core.save(out/'source_generations.json',{'records':dumps});core.save(pool_path,pool)
    selected=sorted([r for r in pool['records'] if r['split']=='probe_fit'],key=lambda r:r['id'])
    assert {r['conv_id'] for r in selected}==set(cids)
    core.save(out/'source_selection_locked.json',{'ids':[r['id'] for r in selected],
              'pool_sha256':core.digest(pool),'source_session_audit_unused':sum(r['split']=='probe_audit' for r in pool['records'])})
    scorer=Scorer(rt);fit=defaultdict(list)
    for i,probe in enumerate(selected):
        path=out/'utility/items'/f"{probe['id'].replace(':','_')}.json"
        if path.exists():fit[probe['conv_id']].append(read(path));continue
        state('construction_source_utility',probe['conv_id'],completed=i,total=len(selected),probe=probe['id'])
        turns={t['id']:t for s in source_sessions[probe['conv_id']] for t in s['turns']}
        jobs,skip=subset_jobs(probe,turns,rt.ntok,POLICY['read_budget'])
        if skip:row={**probe,'skip':skip}
        else:
            scorer.score(jobs);analysis=subset_analysis(jobs)
            state('construction_source_control_answers',probe['conv_id'],completed=i,total=len(selected),probe=probe['id'])
            generated=scorer.generate(generation_controls(jobs,analysis))
            row={**probe,'analysis':analysis,'generations':generated,
                 'subsets':[{k:v for k,v in j.items() if k not in ('user','answer')} for j in jobs]}
        core.save(path,row);fit[probe['conv_id']].append(row)
    memories={}
    for cid in cids:
        state('construction_parent_selection',cid)
        memory,details,options=augment(rt,source_sessions[cid],backends[cid],fit[cid])
        core.save(out/'memories/s_parent_single_2000'/f'{cid}.json',memory)
        core.save(out/'construction'/f'{cid}.json',details)
        core.save(out/'options'/f'{cid}.json',options)
        memories[cid]=memory
    lock={'method':'s_parent_single_2000','memory_sha256':core.digest(memories),
          'protocol_sha256':core.digest(protocol),'source_results_sha256':core.digest(dict(fit)),
          'benchmark_qa_used_for_construction_or_selection':False}
    lock_path=out/'memory_selection_locked.json'
    if lock_path.exists():assert {k:v for k,v in read(lock_path).items() if k!='locked_at'}==lock
    else:core.save(lock_path,{'locked_at':time.time(),**lock})
    # Benchmark question and answer fields first become evaluation inputs here.
    samples=read(args.dataset);byid={str(s['sample_id']):s for s in samples};del samples
    all_rows=[]
    for cid in cids:
        dest=out/'evaluation'/f'{cid}.json'
        if dest.exists():rows=read(dest)
        else:
            state('evaluation_full1540',cid,completed=len(all_rows),total=1540)
            chosen=[r for r in records if r['conv_id']==cid]
            rows=evaluate(rt,{cid:memories[cid]},chosen,byid,2048,'s_parent_single_2000',out/'evaluation'/cid)
            core.save(dest,rows)
        all_rows+=rows
        core.save(out/'partial_summary.json',{'complete':False,'evaluated':len(all_rows),'summary':core.summarize(all_rows)})
    assert len(all_rows)==1540 and {r['id'] for r in all_rows}=={r['id'] for r in records}
    core.save(out/'s_parent_single_2000_full1540_items.json',all_rows)
    subsets={}
    for name,ids in bundle['comparison_question_ids'].items():
        chosen=[r for r in all_rows if r['id'] in set(ids)]
        assert len(chosen)==len(ids)
        core.save(out/f'{name}_items.json',chosen);subsets[name]=core.summarize(chosen)
    summary={'method':'s_parent_single_2000','model':args.model,'complete':True,'expected':1540,
             'full1540':core.summarize(all_rows),'matched_subsets':subsets,
             'protocol_sha256':core.digest(protocol),'memory_sha256':core.digest(memories),
             'tokens':'Authoritative request-by-request actual usage is in usage.jsonl; independent aggregation required.',
             'invocation_wall_seconds':time.time()-started}
    core.save(out/'final.json',summary)
    state('complete',evaluated=1540,f1=summary['full1540']['official_f1'])


if __name__=='__main__':
    try:main()
    except Exception:
        print(traceback.format_exc(),flush=True)
        raise
