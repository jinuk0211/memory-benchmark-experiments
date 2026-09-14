"""Evaluate fixed memories at explicit read budgets with served-token accounting."""
from collections import defaultdict
from pathlib import Path
import json
import refine as core
from evaluate_indexed import evaluate as indexed_evaluate
from evidence_fidelity import VERIFIER,accepted
from evidence_utility import context_for
from crossview_probes import require_fit


def evaluate(rt,memories,records,samples,budget,name,out):
    values=indexed_evaluate(rt,memories,records,samples,budget,name,out)
    for row in values:
        user=f"Conversation memory:\n{row['context']}\n\nQuestion: {row['question']}\nAnswer:"
        key=core.digest([rt.model_meta,rt.args.seed,core.READER,user,96,False])
        cached=json.loads((rt.cache/'generations'/f'{key}.json').read_text())
        assert cached['text']==row['prediction'] and row['read_tokens']<=budget
        row.update({'read_budget':budget,'reader_cache_key':key,'reader_input_tokens':cached['input_tokens'],
                    'reader_output_tokens':cached['output_tokens'],
                    'reader_total_tokens':cached['input_tokens']+cached['output_tokens']})
    splits={r['split'] for r in records};split=next(iter(splits)) if len(splits)==1 else 'all'
    core.save(Path(out)/f'{name}_{split}_items.json',values)
    return values


def assess(rt,memories,sessions,rows,name,out,budget,view):
    if view not in ('q0','fit_a'):raise ValueError('Only existing source fit views may select read budgets')
    if any(r['split']!='probe_fit' for r in rows):raise ValueError('Source audit is excluded')
    if view=='fit_a':require_fit(rows)
    pseudo=defaultdict(lambda:{'qa':[]});records=[];byid={r['id']:r for r in rows}
    split='source_fit' if view=='q0' else 'source_view_fit'
    for row in sorted(rows,key=lambda r:r['id']):
        cid=row['conv_id'];index=len(pseudo[cid]['qa'])
        pseudo[cid]['qa'].append({'question':row['question'],'answer':row['generations']['full']['text'],'category':4,'evidence':row['source_ids']})
        records.append({'id':row['id'],'conv_id':cid,'qa_index':index,'category':4,'split':split})
    values=evaluate(rt,memories,records,dict(pseudo),budget,name,out);prompts=[]
    for value in values:
        row=byid[value['id']];turns={t['id']:t for s in sessions[row['conv_id']] for t in s['turns']}
        original=context_for(row,turns,row['candidate_context_ids'])
        prompts.append(f"ORIGINAL source:\n{original}\n\nQuestion: {row['question']}\nReference answer: {row['generations']['full']['text']}"
                       f"\n\nCANDIDATE fragment:\n{value['context']}\nCandidate answer: {value['prediction']}\n\nVerdict:")
    verdicts=rt.generate(VERIFIER,prompts,max_tokens=16)
    contract={row['id']:{'preserved':accepted(verdict),'verdict':verdict,'prediction':row['prediction'],'context_tokens':row['read_tokens']}
              for row,verdict in zip(values,verdicts)}
    return contract,{'mean_reader_input_tokens':sum(r['reader_input_tokens'] for r in values)/len(values),
                     'mean_reader_output_tokens':sum(r['reader_output_tokens'] for r in values)/len(values),
                     'mean_reader_total_tokens':sum(r['reader_total_tokens'] for r in values)/len(values)}
