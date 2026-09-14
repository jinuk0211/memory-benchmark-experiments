"""Two-history source-only packed-likelihood feasibility experiment."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable

SALT = 'packed-marginal-pilot-v1'
MODEL = 'Qwen/Qwen3.5-9B'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
EMBED = 'sentence-transformers/all-MiniLM-L6-v2'
PREREG_SHA = 'eeb9ad58b563d27735989bbedda8bebb21b37beebc128ca74e27d3ee485f06ff'
POLICY = {'fit_max': 16, 'fit_min': 8, 'audit_max': 4, 'qb_max': 4,
          'read_tokens': 2048, 'extra_tokens': 2000, 'answer_tokens': 96,
          'source_advantage_min': 0.1, 'insufficient_views': 'stop_without_scoring_when_any_history_has_fewer_than_eight',
          'packing_exposure': 'context_bytes_changed; not alias inclusion proof',
          'qb_calibration': 'raw-source eligibility allowed; packed-memory scoring postlock only', 'source_full_f1_min': 0.8,
          'value': 'mean answer logprob QA after packing: Seed+option minus Seed',
          'gate': 'strict whole-set QA likelihood gain and nondecreasing original-audit likelihood AND F1',
          'rounds': 1, 'benchmark_qa_allowed': False}


@dataclass
class Execution:
    out: Path
    support: Any
    scorer: Any
    index: Any


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path: Path | str) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def lock(path: Path | str, value: Any) -> None:
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen artifact differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def ordered(dataset: str, scope: str, values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: (hashlib.sha256(
        f'{SALT}|{dataset}|{scope}|{value}'.encode()).hexdigest(), value))


def admissible(row: dict) -> bool:
    if row['split'] != 'probe_fit' or 'analysis' not in row:
        return False
    if not row['analysis']['source_dependent'] or row['generations']['full']['source_answer_f1'] < 0.8:
        return False
    empty = next(x for x in row['subsets'] if x['kind'] == 'empty')['score']['mean_logprob']
    full = next(x for x in row['subsets'] if x['kind'] == 'full')['score']['mean_logprob']
    realized = {tuple(x['source_ids']) for x in row['generations'].values()}
    return full - empty >= 0.1 and any(x['kind'] in ('single', 'full')
        and tuple(x['source_ids']) in realized and x['score']['mean_logprob'] > empty for x in row['subsets'])


def select(dataset: str, rows: list[dict], pool: list[dict]) -> tuple[str, list[dict], list[dict], list[str]]:
    good = {r['id']: r for r in rows if admissible(r)}
    if len(good) != sum(admissible(r) for r in rows):
        raise ValueError('Duplicate source-fit IDs')
    histories = {r['conv_id'] for r in good.values()}
    viable = [cid for cid in histories if sum(r['conv_id'] == cid for r in good.values()) >= POLICY['fit_min']
              and any(r['conv_id'] == cid and r['split'] == 'probe_audit' for r in pool)]
    if not viable:
        raise ValueError('No history with enough source-fit and audit probes')
    cid = ordered(dataset, 'history', viable)[0]
    fit_ids = ordered(dataset, 'fit', [k for k, r in good.items() if r['conv_id'] == cid])[:16]
    audit = {r['id']: r for r in pool if r['conv_id'] == cid and r['split'] == 'probe_audit'}
    audit_ids = ordered(dataset, 'audit', audit)[:4]
    return cid, [good[k] for k in fit_ids], [audit[k] for k in audit_ids], ordered(dataset, 'qb', fit_ids)[:4]


def support(adapter: Path | str, v2: Path | str) -> tuple[Any, dict[str, str]]:
    adapter, v2 = Path(adapter), Path(v2)
    for path in (v2 / 'source', adapter / 'source', v2, adapter):
        sys.path.insert(0, str(path))
    modules = {name: importlib.import_module(name) for name in
               ('refine', 'transfer_runtime', 'runtime_meter', 'source_views_v2', 'recursive_memory',
                'parent_evidence', 'budgeted_evidence', 'evidence_utility', 'run_recursive_v2')}
    files = [adapter / n for n in ('transfer_runtime.py', 'runtime_meter.py', 'chat_tokenizer_compat.py')]
    files += list((adapter / 'source').glob('*.py'))
    files += [v2 / n for n in ('source_views_v2.py', 'recursive_memory.py', 'recursive_memory_v2.py',
                              'run_recursive_v2.py', 'run_recursive.py', 'summarize_recursive.py')]
    files += [v2 / 'source' / n for n in ('crossview_probes.py', 'evaluate_indexed.py')]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for name in ('import_source', 'source_views_v2_pilot'):
        module = importlib.import_module(name)
        expected = Path(__file__).resolve().with_name(name + '.py')
        if Path(module.__file__).resolve() != expected:
            raise ValueError('Local helper import collision: ' + name)
        modules[name] = module
        files.append(expected)
    hashes = {str(p.resolve()): sha(p) for p in files}
    return SimpleNamespace(**modules), hashes


def candidate_groups(s: Any, token_runtime: Any, sessions: list[dict], seed: list[dict], rows: list[dict]) -> list[list[dict]]:
    groups, mapped = s.parent_evidence.make_options(token_runtime, sessions, seed, rows)
    return [[copy.deepcopy(mapped[o['id']]) for o in group
             if o['kind'] != 'pair' and o['id'] in mapped and mapped[o['id']]['cost'] <= 2000]
            for group in groups]


def plan(args: argparse.Namespace) -> None:
    if sha(args.prereg) != PREREG_SHA:
        raise ValueError('Pre-inference criteria hash differs')
    s, source_hashes = support(args.adapter_root, args.v2_root)
    environment = read(args.environment)
    s.transfer_runtime.model_metadata(environment, MODEL, MODEL, REVISION)
    s.transfer_runtime.model_metadata(environment, EMBED, EMBED, s.transfer_runtime.EMBED_REVISION)
    from transformers import AutoTokenizer
    s.import_source.verify_tokenizer(Path(environment['models'][MODEL]['path']))
    tokenizer = AutoTokenizer.from_pretrained(environment['models'][MODEL]['path'], local_files_only=True)
    token_runtime = SimpleNamespace(ntok=lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    histories = {}
    for dataset, path in (('locomo', args.locomo_source), ('longmemeval', args.lme_source)):
        root = Path(path).resolve()
        protocol = read(root / 'protocol.json')
        if (protocol['model']['revision'] != REVISION or protocol['config']['model'] != MODEL
                or protocol['config']['embed_model'] != EMBED or protocol['read_budget'] != 2048):
            raise ValueError('Imported run is not the fixed Qwen/MiniLM comparison')
        lineage, sessions, seed_memories, per_history = s.import_source.imported_files(root, dataset)
        if dataset == 'longmemeval' and set(sessions) != set(read(args.lme_expected_ids)):
            raise ValueError('LongMemEval source must be exactly the already-exposed DEV12')
        rows = [r for history_rows in per_history.values() for r in history_rows]
        pool = read(root / 'source_probes.json')['records']
        cid, fit, audit, qb_ids = select(dataset, rows, pool)
        seed = seed_memories[cid]
        groups = candidate_groups(s, token_runtime, sessions[cid], seed, fit)
        hashes = {name: item['sha256'] for name, item in lineage['files'].items()}
        histories[dataset] = {'history_id': cid, 'source_root': str(root), 'input_hashes': hashes,
            'import_lineage': lineage, 'sessions': sessions[cid], 'seed': seed, 'fit': fit, 'audit': audit,
            'qb_ids': qb_ids, 'groups': groups,
            'counts': {'fit': len(fit), 'audit': len(audit), 'qb_preselected': len(qb_ids),
                       'mapped_options': sum(map(len, groups))}}
    value = {'schema': SALT, 'policy': POLICY, 'environment': environment,
             'adapter_root': str(Path(args.adapter_root).resolve()), 'v2_root': str(Path(args.v2_root).resolve()),
             'source_hashes': source_hashes, 'pilot_code_sha256': sha(__file__),
             'prereg_sha256': PREREG_SHA, 'histories': histories,
             'model': MODEL, 'embedding': EMBED, 'seed': 20260907, 'scorer_seed': 20260908,
             'cost_scope': 'Imported preparation/source scores are inherited; new native invocations only are incremental.'}
    lock(args.out / 'protocol.json', value)
    lock(args.out / 'PLAN_LOCK.json', {'protocol_sha256': sha(args.out / 'protocol.json'),
         'pilot_code_sha256': sha(__file__), 'benchmark_questions_used': False})
    print(json.dumps({ds: h['counts'] | {'history_id': h['history_id']} for ds, h in histories.items()}), flush=True)


def verified(out: Path) -> dict:
    p = read(out / 'protocol.json')
    if read(out / 'PLAN_LOCK.json')['protocol_sha256'] != sha(out / 'protocol.json'):
        raise ValueError('Frozen pilot protocol changed')
    if p['policy'] != POLICY or p['pilot_code_sha256'] != sha(__file__):
        raise ValueError('Pilot source or policy changed')
    for path, expected in p['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Imported helper source changed: ' + path)
    for h in p['histories'].values():
        for path, expected in h['input_hashes'].items():
            if sha(Path(h['source_root']) / path) != expected:
                raise ValueError('Imported source artifact changed: ' + path)
    return p


def runtime_args(out: Path, stage: str) -> SimpleNamespace:
    return SimpleNamespace(out=out, stage=stage, model=MODEL, embed_model=EMBED, seed=20260907, embed_batch_size=4)


def checked_views(s: Any, path: Path, h: dict) -> dict:
    bundle = read(path)
    manifest = bundle['manifest']
    records = manifest['records']
    if (bundle['manifest_sha256'] != s.refine.digest(manifest)
            or manifest['policy'] != s.source_views_v2.VIEW_POLICY
            or manifest['input_sha256'] != s.refine.digest(h['fit'])
            or bundle['input_count'] != len(h['fit']) or bundle['accepted_count'] != len(records)):
        raise ValueError('Source-view input, manifest, or count differs')
    cues = [u.get('index_text', u['text']) for u in h['seed']] + [r['question'] for r in h['fit']]
    if manifest['stored_cues_sha256'] != s.refine.digest(cues):
        raise ValueError('Source views were generated against different stored cues')
    if records:
        s.source_views_v2.validate_bundle(bundle, h['fit'])
    for record in records:
        for role, field in (('fit', 'fit_question'), ('audit', 'diagnostic_question')):
            quality = bundle['quality'][record['id']][role]
            if (quality['question'] != record[field] or quality['accepted'] is not True
                    or not math.isfinite(quality['f1']) or quality['f1'] < 0.8):
                raise ValueError('A query view lacks raw-source eligibility')
    forbidden = s.source_views_v2.forbidden_views(bundle)
    if len(forbidden) != 2 * len(records):
        raise ValueError('Duplicate independent query views')
    for unit in h['seed'] + [o['unit'] for group in h['groups'] for o in group]:
        if s.source_views_v2.copies_view(unit.get('index_text', unit['text']), forbidden):
            raise ValueError('QA/QB leaked into an index')
    return bundle


def calibration_status(out: Path, bundles: dict) -> dict:
    counts = {ds: b['accepted_count'] for ds, b in bundles.items()}
    sufficient = all(n >= POLICY['fit_min'] for n in counts.values())
    status = {'status': 'ready' if sufficient else 'insufficient_evidence', 'accepted_counts': counts,
              'views_sha256': {ds: sha(out / 'views' / f'{ds}.json') for ds in bundles},
              'packed_memory_scoring_performed': False, 'minimum_fit_per_history': POLICY['fit_min'],
              'reason': None if sufficient else 'No Q0 fallback or outcome-based replacement; no candidate constructed.'}
    lock(out / 'CALIBRATION_STATUS.json', status)
    return status


def calibrate(out: Path) -> None:
    p = verified(out)
    s, _ = support(p['adapter_root'], p['v2_root'])
    pending = [ds for ds in p['histories'] if not (out / 'views' / f'{ds}.json').exists()]
    rt = s.transfer_runtime.Runtime(runtime_args(out / 'calibration', 'calibrate'), p['environment']) if pending else None
    bundles = {}
    for ds, h in p['histories'].items():
        path = out / 'views' / f'{ds}.json'
        if ds in pending:
            s.runtime_meter.set_runtime_context('source_view_calibration', conversation=ds + ':' + h['history_id'])
            # Identical approved preparation except zero records are returned for explicit insufficient status.
            bundle = s.source_views_v2_pilot.prepare_views(rt, h['sessions'], h['fit'], h['seed'])
            lock(path, bundle)
        bundles[ds] = checked_views(s, path, h)
    verified(out)
    print(json.dumps(calibration_status(out, bundles)), flush=True)


def index_runtime(s: Any, args: Any, environment: dict, ntok: Callable[[str], int]) -> Any:
    # The inherited Runtime.encode/core.Runtime.encode cache and query handling remain unchanged.
    class Index(s.transfer_runtime.Runtime):
        def __init__(self) -> None:
            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(8)
            self.args, self.np, self.ntok = args, np, ntok
            self.cache = args.out / 'cache'
            self.embed_meta = environment['models'][EMBED]
            self.meter = s.runtime_meter.Meter(args.out, 'index', {'role': 'source_index',
                'embedding': self.embed_meta, 'embedding_device': 'cpu', 'embedding_dtype': 'float32',
                'native_max_seq_length': 256})
            def load() -> Any:
                encoder = SentenceTransformer(self.embed_meta['path'], device='cpu').float()
                if encoder.max_seq_length != 256:
                    raise ValueError('MiniLM native capacity differs')
                return encoder
            encoder = self.meter.operation('embedding_initialization', {'embedding': self.embed_meta}, load)
            self.embed = s.runtime_meter.MiniLMAdapter(encoder, self.meter)

        def generate(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError('CPU index has no generation engine')
    return Index()


def jobs_for(s: Any, index: Any, memory: list[dict], probes: list[dict]) -> list[dict]:
    contexts = s.recursive_memory.retrieve(index, memory, [r['question'] for r in probes])
    if len(contexts) != len(probes):
        raise ValueError('Missing retrieval contexts')
    return [{'question': r['question'], 'user': s.evidence_utility.reader_user(r['question'], c['context']), 'answer': r['answer'],
             'source_ids': c['source_ids'], 'context_tokens': c['read_tokens'], 'context': c['context'],
             'probe_id': r['id']} for r, c in zip(probes, contexts)]


def logprob(job: dict) -> float:
    result = float(job['score']['mean_logprob'])
    if not math.isfinite(result):
        raise ValueError('Nonfinite source likelihood')
    return result


def value(option_job: dict, baseline_job: dict, mapped_job: dict, empty_job: dict) -> float:
    if option_job['context'] == baseline_job['context']:
        return 0.0
    option, baseline, mapped, empty = map(logprob, (option_job, baseline_job, mapped_job, empty_job))
    if mapped - empty < 0.1:
        return 0.0
    return max(0.0, option - baseline)


def mean(jobs: list[dict]) -> float:
    if not jobs:
        raise ValueError('Empty source comparison')
    return sum(map(logprob, jobs)) / len(jobs)


def paired_delta(before: list[dict], after: list[dict]) -> float:
    def identity(j: dict) -> tuple:
        return j['probe_id'], j['question'], j['answer']
    if list(map(identity, before)) != list(map(identity, after)):
        raise ValueError('Source comparison identities or denominator differ')
    return mean(after) - mean(before)


def gate(fit: tuple[list[dict], list[dict]], audit: tuple[list[dict], list[dict]],
         audit_f1: tuple[float, float]) -> bool:
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in audit_f1):
        raise ValueError('Nonfinite or invalid source F1')
    return paired_delta(*fit) > 0 and paired_delta(*audit) >= 0 and audit_f1[1] >= audit_f1[0]


def generation_guard(scorer: Any, jobs: list[dict]) -> None:
    import refine as core
    for job in jobs:
        prompt = scorer.tok.apply_chat_template(
            [{'role': 'system', 'content': core.READER}, {'role': 'user', 'content': job['user']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if scorer.ntok(prompt) + POLICY['answer_tokens'] > 8192:
            raise ValueError('Source answer prompt exceeds actual 8192 context capacity')


def answer_scores(scorer: Any, jobs: list[dict]) -> dict:
    if not jobs:
        return {'mean_f1': None, 'rows': {}}
    generation_guard(scorer, jobs)
    controls = {str(i): j for i, j in enumerate(jobs)}
    results = scorer.generate(controls)
    if set(results) != set(controls):
        raise ValueError('Missing or unexpected source answers')
    values = [float(r['source_answer_f1']) for r in results.values()]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
        raise ValueError('Nonfinite or invalid generated source F1')
    return {'mean_f1': sum(values) / len(values), 'rows': results}


def score_options(ctx: Execution, h: dict, fit: list[dict], baseline: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    s, scorer, index = ctx.support, ctx.scorer, ctx.index
    by_id = {r['id']: r for r in fit}
    originals = {r['id']: r for r in h['fit']}
    baseline_by_id = {j['probe_id']: j for j in baseline}
    choices, details = [], []
    for group in h['groups']:
        options = []
        for original in group:
            pid = original['unit']['source_probe_id']
            if pid not in by_id:
                continue
            job = jobs_for(s, index, h['seed'] + [original['unit']], [by_id[pid]])[0]
            mapped = {'user': s.evidence_utility.reader_user(originals[pid]['question'], original['unit']['text']),
                      'answer': originals[pid]['answer']}
            empty = {'user': s.evidence_utility.reader_user(originals[pid]['question'], ''), 'answer': originals[pid]['answer']}
            unchanged = job['context'] == baseline_by_id[pid]['context']
            if not unchanged:
                scorer.score([job, mapped, empty])
            marginal = 0.0 if unchanged else value(job, baseline_by_id[pid], mapped, empty)
            details.append({'id': original['id'], 'probe_id': pid, 'unchanged_context': unchanged,
                            'marginal_value': marginal, 'packed': job, 'mapped': mapped, 'empty': empty})
            if marginal > 0:
                options.append({**copy.deepcopy(original), 'gain': marginal})
        choices.append(options)
    return choices, details


def score_history(ctx: Execution, ds: str, h: dict, bundle: dict) -> list[dict]:
    out, s, scorer, index = ctx.out, ctx.support, ctx.scorer, ctx.index
    s.runtime_meter.set_runtime_context('packed_marginal_fit', conversation=ds + ':' + h['history_id'])
    fit = s.source_views_v2.fit_rows(bundle, h['fit'])
    baseline = jobs_for(s, index, h['seed'], fit)
    scorer.score(baseline)
    choices, details = score_options(ctx, h, fit, baseline)
    selected, allocation = s.budgeted_evidence.multiple_choice_budget(choices, 2000, quantum=8)
    if sum(o['cost'] for o in selected) > 2000:
        raise ValueError('Additional storage exceeds fixed budget')
    proposal = h['seed'] + [o['unit'] for o in selected]
    candidate_fit = jobs_for(s, index, proposal, fit)
    base_audit = jobs_for(s, index, h['seed'], h['audit'])
    candidate_audit = jobs_for(s, index, proposal, h['audit'])
    scorer.score(candidate_fit + base_audit + candidate_audit)
    fit_answers = {'seed': answer_scores(scorer, baseline), 'proposal': answer_scores(scorer, candidate_fit)}
    audit_answers = {'seed': answer_scores(scorer, base_audit), 'proposal': answer_scores(scorer, candidate_audit)}
    adopted = bool(selected) and gate((baseline, candidate_fit), (base_audit, candidate_audit),
                                     (audit_answers['seed']['mean_f1'], audit_answers['proposal']['mean_f1']))
    memory = proposal if adopted else h['seed']
    if memory[:len(h['seed'])] != h['seed']:
        raise ValueError('Seed prefix changed')
    receipt = {'adopted': adopted, 'allocation': allocation, 'selected_options': [o['id'] for o in selected],
               'calibrated_fit_count': len(fit), 'audit_count': len(h['audit']), 'option_details': details,
               'baseline_fit': baseline, 'proposal_fit': candidate_fit, 'baseline_audit': base_audit,
               'proposal_audit': candidate_audit, 'fit_answers': fit_answers, 'audit_answers': audit_answers,
               'whole_set_qa_delta': paired_delta(baseline, candidate_fit),
               'source_audit_delta': paired_delta(base_audit, candidate_audit),
               'final_memory_sha256': digest(memory), 'views_sha256': sha(out / 'views' / f'{ds}.json')}
    lock(out / 'construction' / f'{ds}.json', receipt)
    lock(out / 'memories' / f'{ds}.json', memory)
    print(json.dumps({'dataset': ds, 'phase': 'source_selection', 'adopted': adopted,
                      'positive_options': sum(map(len, choices)), 'selected': len(selected),
                      'qa_delta': receipt['whole_set_qa_delta']}), flush=True)
    return memory


def locked_memories(out: Path) -> dict:
    receipt = read(out / 'memory_lock.json')
    if sha(out / 'protocol.json') != receipt['protocol_sha256']:
        raise ValueError('Protocol changed after memory lock')
    for folder, field in (('views', 'views_sha256'), ('construction', 'construction_sha256'),
                          ('memories', 'memory_files_sha256')):
        for ds, expected in receipt[field].items():
            if sha(out / folder / f'{ds}.json') != expected:
                raise ValueError('Frozen artifact changed before QB: ' + folder + '/' + ds)
    memories = {ds: read(out / 'memories' / f'{ds}.json') for ds in receipt['memory_files_sha256']}
    if digest(memories) != receipt['memories_sha256']:
        raise ValueError('Memory lock digest differs')
    return memories


def postlock_qb(ctx: Execution, ds: str, h: dict, bundle: dict) -> None:
    out, s, scorer, index = ctx.out, ctx.support, ctx.scorer, ctx.index
    final = locked_memories(out)
    s.runtime_meter.set_runtime_context('post_lock_qb', conversation=ds + ':' + h['history_id'])
    probes = [{**r['original'], 'question': r['diagnostic_question']} for r in bundle['manifest']['records']
              if r['id'] in h['qb_ids']]
    before = jobs_for(s, index, h['seed'], probes)
    after = jobs_for(s, index, final[ds], probes)
    scorer.score(before + after)
    lock(out / 'qb' / f'{ds}.json', {'count': len(probes), 'selection_allowed': False,
         'preselected_ids': h['qb_ids'], 'actual_ids': [r['id'] for r in probes],
         'memory_lock_sha256': sha(out / 'memory_lock.json'), 'seed': before, 'final': after,
         'mean_logprob_delta': paired_delta(before, after) if probes else None,
         'seed_answers': answer_scores(scorer, before), 'final_answers': answer_scores(scorer, after)})
    locked_memories(out)


def score(out: Path) -> None:
    p = verified(out)
    s, _ = support(p['adapter_root'], p['v2_root'])
    bundles = {ds: checked_views(s, out / 'views' / f'{ds}.json', h) for ds, h in p['histories'].items()}
    status = calibration_status(out, bundles)
    if status['status'] != 'ready':
        print(json.dumps(status), flush=True)
        return
    scorer = s.transfer_runtime.Scorer(runtime_args(out / 'scoring', 'score'), p['environment'], out / 'scoring')
    index = index_runtime(s, runtime_args(out / 'indexing', 'index'), p['environment'], scorer.ntok)
    ctx = Execution(out, s, scorer, index)
    final = {}
    for ds, h in p['histories'].items():
        final[ds] = score_history(ctx, ds, h, bundles[ds])
    verified(out)
    lock(out / 'memory_lock.json', {'protocol_sha256': sha(out / 'protocol.json'),
         'views_sha256': {ds: sha(out / 'views' / f'{ds}.json') for ds in bundles},
         'construction_sha256': {ds: sha(out / 'construction' / f'{ds}.json') for ds in bundles},
         'memory_files_sha256': {ds: sha(out / 'memories' / f'{ds}.json') for ds in bundles},
         'memories_sha256': digest(final), 'benchmark_questions_used': False})
    for ds, h in p['histories'].items():
        postlock_qb(ctx, ds, h, bundles[ds])
    verified(out)
    lock(out / 'COMPLETE.json', {'protocol_sha256': sha(out / 'protocol.json'),
         'memory_lock_sha256': sha(out / 'memory_lock.json'),
         'qb_sha256': {ds: sha(out / 'qb' / f'{ds}.json') for ds in p['histories']},
         'benchmark_evaluation_performed': False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'calibrate', 'score'))
    parser.add_argument('--out', type=Path, required=True)
    for name in ('adapter-root', 'v2-root', 'environment', 'locomo-source', 'lme-source', 'lme-expected-ids', 'prereg'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    started_ns = time.time_ns()
    if args.stage == 'plan':
        for name in ('adapter_root', 'v2_root', 'environment', 'locomo_source', 'lme_source', 'lme_expected_ids', 'prereg'):
            if getattr(args, name) is None:
                parser.error('--' + name.replace('_', '-') + ' is required for plan')
        plan(args)
    else:
        {'calibrate': calibrate, 'score': score}[args.stage](args.out)
    timing = {'stage': args.stage, 'elapsed_seconds': time.perf_counter() - started,
              'scope': 'source-only invocation; includes cache replay and excludes inherited preparation'}
    lock(args.out / 'timings' / f'{args.stage}_{started_ns}.json', timing)
    print(json.dumps(timing), flush=True)


if __name__ == '__main__':
    main()
