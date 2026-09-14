"""Frozen memory construction and crossed-model/data transfer; no target selection."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
from portable_parent import backend, augment, generate_source_probes
from evidence_utility import POLICY, subset_jobs, subset_analysis, generation_controls
from transfer_runtime import Runtime, Scorer
from transfer_data import load_longmemeval, deterministic_subset, paired_summary

METHODS = ('seed', 'r40_fused_four_turn', 's_parent_single_2000')
PLAN_NAMES = ('r06_calendar_month', 'r12_filter_current_best', 'r40_fused_four_turn', 'r24_source_qa_cards')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def locked(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen artifact changed: {path}; use a new run')
    core.save(path, value)


def state(out, phase, **fields):
    core.save(out / 'status.json', {'phase': phase, 'pid': os.getpid(), 'time': time.time(), **fields})
    print(phase + ' ' + json.dumps(fields), flush=True)


def load_samples(args):
    if args.dataset == 'longmemeval':
        return load_longmemeval(args.data)
    samples = read(args.data)
    return samples


def source_plans():
    plans = {name: read(ROOT / 'source/runs/continuous_v3/plans' / (name + '.json')) for name in PLAN_NAMES}
    for plan in plans.values():
        for name, expected in plan['source_sha256'].items():
            if hashlib.sha256((ROOT / 'source' / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source hash mismatch: ' + name)
    return plans


def evaluate_sample(rt, units, sample, method, dataset):
    """Original BM25+dense RRF and packing; gold accessed only after generation."""
    import numpy as np
    from rank_bm25 import BM25Okapi

    indexes = [u.get('index_text', u['text']) for u in units]
    if not indexes:
        raise ValueError('Empty memory is an infrastructure failure, not a missing row')
    vectors = rt.encode(indexes)
    sparse = BM25Okapi([core.lexical(t) or ['_empty'] for t in indexes])
    qa_indices = [i for i, qa in enumerate(sample['qa']) if dataset != 'locomo' or int(qa['category']) in (1, 2, 3, 4)]
    qas = [sample['qa'][i] for i in qa_indices]
    questions = [qa['question'] for qa in qas]
    queries = rt.encode(questions, query=True)
    prompts, retrievals = [], []
    def ranks(scores):
        order = np.argsort(-scores, kind='stable')
        positions = np.empty(len(order), dtype=np.int64)
        positions[order] = np.arange(len(order))
        return positions
    for qa, query in zip(qas, queries):
        fused = 1 / (60 + ranks(vectors @ query)) + 1 / (60 + ranks(np.asarray(sparse.get_scores(core.lexical(qa['question'])))))
        order = np.argsort(-fused, kind='stable')[:min(len(units), 120)]
        context, hits, tokens = core.pack(units, order, rt.ntok, 2048)
        question = qa['question']
        date = f"Question date: {qa['question_date']}\n" if dataset == 'longmemeval' else ''
        prompts.append(f"Conversation memory:\n{context}\n\n{date}Question: {question}\nAnswer:")
        retrievals.append({'context': context, 'read_tokens': tokens,
                           'source_ids': sorted({sid for i in hits for sid in units[i]['sources']})})
    started = time.time()
    predictions = rt.generate(core.READER, prompts, max_tokens=96)
    if len(predictions) != len(qas):
        raise ValueError('Missing generation outputs')
    storage = sum(rt.ntok(u['text']) + (rt.ntok(index) if index != u['text'] else 0)
                  for u, index in zip(units, indexes))
    rows = []
    for index, (qa, user, pred, retrieval) in enumerate(zip(qas, prompts, predictions, retrievals)):
        key = core.digest([rt.model_meta, rt.args.seed, core.READER, user, 96, False])
        generation = read(rt.cache / 'generations' / (key + '.json'))
        qid = str(sample['sample_id']) if dataset == 'longmemeval' else str(sample['sample_id']) + ':' + str(qa_indices[index])
        row = {'question_id': qid, 'conversation_id': str(sample['sample_id']),
               'method': method, 'question': qa['question'], 'hypothesis': pred, 'prediction': pred,
               'status': 'ok' if pred.strip() else 'generation_empty',
               'gold': str(qa['answer']), 'category': qa['category'], **retrieval,
               'input_tokens': generation['input_tokens'], 'output_tokens': generation['output_tokens'],
               'finish_reason': generation['finish_reason'], 'memory_tokens': storage, 'memory_units': len(units),
               'generation_batch_seconds': time.time() - started}
        if dataset == 'locomo':
            row['official_f1'] = core.f1(pred, qa['answer'], int(qa['category']))
        else:
            row['diagnostic_token_f1'] = core.generic_f1(pred, qa['answer'])
            row['question_type'] = qa['question_type']
            row['is_abstention'] = qa['is_abstention']
            row['question_date'] = qa['question_date']
            gold_sessions = {int(m['session'].split('_')[1]) for m in sample['evaluation_metadata']['session_map'] if m['is_answer_session']}
            retrieved = {int(sid.split(':')[0][1:]) for sid in retrieval['source_ids']}
            row['session_recall_any'] = bool(gold_sessions & retrieved) if gold_sessions else None
            row['session_recall_all'] = gold_sessions <= retrieved if gold_sessions else None
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('plan', 'prepare', 'score', 'construct', 'evaluate'))
    ap.add_argument('--dataset', choices=('locomo', 'longmemeval'), required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--environment', type=Path, default=ROOT / 'environment.json')
    ap.add_argument('--model', default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model', default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20260907)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    out = args.out
    plans = source_plans()
    samples = load_samples(args)
    selected, selection = deterministic_subset(samples, args.limit or len(samples))
    environment = read(args.environment)
    package_names = ('torch', 'vllm', 'transformers', 'sentence-transformers', 'numpy', 'rank-bm25')
    packages = {p: importlib.metadata.version(p) for p in package_names}
    files = list((ROOT / 'source').glob('*.py')) + [ROOT / name for name in ('run_transfer.py', 'transfer_runtime.py', 'transfer_data.py')]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'stage'}
    protocol = {'config': config, 'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'selection': selection, 'methods': METHODS, 'plans': plans, 'source_policy': POLICY,
                'model': environment['models'][args.model], 'embedding': environment['models'][args.embed_model],
                'actual_packages': packages, 'reader_prompt': core.READER, 'read_budget': 2048,
                'extra_storage_budget': 2000, 'max_answer_tokens': 96, 'context_capacity': 8192,
                'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'model_transfer': 'all writer, source likelihood scorer, and reader roles use selected model',
                'target_selection': False, 'history_truncation': False,
                'primary_contrast': 's_parent_single_2000 minus r40_fused_four_turn',
                'secondary_contrast': 's_parent_single_2000 minus seed',
                'longmemeval_primary_metric': 'pending pinned official judge; token F1 is diagnostic only'}
    # JSON roundtrip stabilizes tuple representations on resume.
    locked(out / 'protocol.json', json.loads(json.dumps(protocol)))
    if args.stage == 'plan':
        source_sessions = {str(s['sample_id']): core.session_data(s) for s in selected}
        locked(out / 'source_sessions.json', source_sessions)
        state(out, 'planned', conversations=len(selected), total_population=len(samples), target_outcomes_used=False)
        return
    sessions = read(out / 'source_sessions.json')
    if set(sessions) != {str(s['sample_id']) for s in selected}:
        raise ValueError('Source corpus IDs mismatch')
    if core.digest(sessions) != core.digest({str(s['sample_id']): core.session_data(s) for s in selected}):
        raise ValueError('Source histories changed')
    # Target QA is removed from process-level construction variables.
    if args.stage != 'evaluate':
        del samples, selected
    if args.stage == 'prepare':
        state(out, 'loading_writer', model=args.model)
        rt = Runtime(args, environment)
        for cid, history in sorted(sessions.items()):
            state(out, 'building_seed', conversation=cid)
            seed_path = out / 'memories/seed' / (cid + '.json')
            if seed_path.exists():
                seed = read(seed_path)
            else:
                base = core.build(rt, history, 'session10')
                base = core.build(rt, history, 'audit', base)
                seed = core.build(rt, history, 'dialogue_residual', base)
                core.save(seed_path, seed)
            parent_path = out / 'memories/r40_fused_four_turn' / (cid + '.json')
            if not parent_path.exists():
                core.save(parent_path, backend(rt, history, seed, plans))
        state(out, 'generating_source_probes')
        dump, pool = generate_source_probes(rt, sessions, plans['r24_source_qa_cards']['recipe']['operations'][0])
        core.save(out / 'source_generations.json', dump)
        core.save(out / 'source_probes.json', pool)
        state(out, 'prepared', probes=len(pool['records']))
    elif args.stage == 'score':
        pool = read(out / 'source_probes.json')
        rows = sorted([r for r in pool['records'] if r['split'] == 'probe_fit'], key=lambda r: r['id'])
        locked(out / 'utility_selection.json', {'ids': [r['id'] for r in rows], 'source_pool_sha256': core.digest(pool)})
        state(out, 'loading_scorer', model=args.model)
        scorer = Scorer(args, environment, out / 'utility')
        for index, probe in enumerate(rows):
            path = out / 'utility/items' / (probe['id'].replace(':', '_') + '.json')
            if path.exists():
                continue
            state(out, 'scoring_source', completed=index, total=len(rows))
            turns = {t['id']: t for s in sessions[probe['conv_id']] for t in s['turns']}
            jobs, skip = subset_jobs(probe, turns, scorer.ntok, POLICY['read_budget'])
            if skip:
                row = {**probe, 'skip': skip}
            else:
                scorer.score(jobs)
                analysis = subset_analysis(jobs)
                row = {**probe, 'analysis': analysis, 'generations': scorer.generate(generation_controls(jobs, analysis)),
                       'subsets': [{k: v for k, v in j.items() if k not in ('user', 'answer')} for j in jobs]}
            core.save(path, row)
        state(out, 'scored', probes=len(rows))
    elif args.stage == 'construct':
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(environment['models'][args.model]['path'])
        # Parent routing uses only token counts; it does not generate or embed.
        class TokenRuntime:
            ntok = staticmethod(lambda text: len(tok.encode(text, add_special_tokens=False)))
        rt = TokenRuntime()
        rows = defaultdict(list)
        expected = read(out / 'utility_selection.json')['ids']
        for probe_id in expected:
            row = read(out / 'utility/items' / (probe_id.replace(':', '_') + '.json'))
            if row['id'] != probe_id:
                raise ValueError('Wrong cached probe')
            rows[row['conv_id']].append(row)
        memories = {method: {} for method in METHODS}
        for cid, history in sorted(sessions.items()):
            for method in METHODS[:2]:
                memories[method][cid] = read(out / 'memories' / method / (cid + '.json'))
            refined, details, options = augment(rt, history, memories[METHODS[1]][cid], rows[cid])
            memories[METHODS[2]][cid] = refined
            core.save(out / 'memories' / METHODS[2] / (cid + '.json'), refined)
            core.save(out / 'construction' / (cid + '.json'), details)
        locked(out / 'memory_lock.json', {'protocol_sha256': core.digest(protocol),
               'source_sha256': core.digest(sessions), 'memories_sha256': core.digest(memories),
               'benchmark_questions_used': False})
        state(out, 'memories_locked', conversations=len(sessions))
    else:
        memories = {method: {cid: read(out / 'memories' / method / (cid + '.json')) for cid in sessions} for method in METHODS}
        lock = read(out / 'memory_lock.json')
        if lock['memories_sha256'] != core.digest(memories) or lock['protocol_sha256'] != core.digest(protocol):
            raise ValueError('Memory lock mismatch')
        state(out, 'loading_reader', model=args.model)
        rt = Runtime(args, environment)
        by_method = {method: [] for method in METHODS}
        for sample in selected:
            cid = str(sample['sample_id'])
            for method in METHODS:
                state(out, 'evaluating', method=method, conversation=cid)
                path = out / 'items' / method / (cid + '.json')
                if not path.exists():
                    core.save(path, evaluate_sample(rt, memories[method][cid], sample, method, args.dataset))
                by_method[method].extend(read(path))
        for method, rows in by_method.items():
            target = out / (method + '.jsonl')
            target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        ids = [str(s['sample_id']) if args.dataset == 'longmemeval' else str(s['sample_id']) + ':' + str(i)
               for s in selected for i, qa in enumerate(s['qa'])
               if args.dataset != 'locomo' or int(qa['category']) in (1, 2, 3, 4)]
        metric = 'official_f1' if args.dataset == 'locomo' else 'diagnostic_token_f1'
        comparisons = {base: paired_summary(by_method[base], by_method[METHODS[2]], manifest=ids,
                         dataset=args.dataset, score_key=metric) for base in METHODS[:2]}
        core.save(out / 'paired_diagnostics.json', {'metric': metric, 'comparisons': comparisons,
                   'official_longmemeval_accuracy_pending': args.dataset == 'longmemeval'})
        state(out, 'generation_complete', questions=len(ids), official_judge_pending=args.dataset == 'longmemeval')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # A failed stage remains failed and the Supervisor queue cannot silently advance it.
        print('TRANSFER_FAILED ' + repr(exc), flush=True)
        raise


