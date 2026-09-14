"""Seed-parent ablation with imported, frozen source utility and unchanged reader."""
import argparse
from copy import deepcopy
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
import approved_run_transfer as approved
from portable_parent import augment
from budgeted_evidence import storage_cost
from transfer_runtime import Runtime, EXECUTION_PROFILE, RUNTIME_FLAGS, MODEL, EMBED_MODEL, set_runtime_context
from transfer_data import load_longmemeval
from date_metadata_adapter import install_date_adapter
from import_source import (METHOD, COST_POLICY, TOKENIZER_FILES, read, sha, locked, imported_files,
                           snapshot, verify_snapshot, verify_tokenizer)


# The evaluator below is copied verbatim from approved_run_transfer.py.
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



def source_hashes() -> dict[str, str]:
    """Hash the complete construction, runtime, and evaluator source closure."""
    files = list((ROOT / 'source').glob('*.py')) + [ROOT / name for name in (
        'run_transfer.py', 'approved_run_transfer.py', 'import_source.py', 'transfer_runtime.py',
        'transfer_data.py', 'chat_tokenizer_compat.py', 'runtime_meter.py', 'date_metadata_adapter.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def state(out: Path, phase: str, **fields: Any) -> None:
    """Persist stage progress and assign the native meter context."""
    set_runtime_context(phase, **fields)
    core.save(out / 'status.json', {'phase': phase, 'pid': os.getpid(), 'time': time.time(), **fields})
    print(phase + ' ' + json.dumps(fields), flush=True)


def samples_for(data: Path, protocol: dict) -> list[dict]:
    """Read only the frozen selected population from the canonical dataset."""
    if sha(data) != protocol['dataset_sha256']:
        raise ValueError('Canonical dataset hash changed')
    dataset = protocol['config']['dataset']
    samples = load_longmemeval(data) if dataset == 'longmemeval' else read(data)
    by_id = {str(s['sample_id']): s for s in samples}
    if len(by_id) != len(samples):
        raise ValueError('Duplicate canonical sample IDs')
    return [by_id[cid] for cid in protocol['selection']['selected_ids']]


def verify_histories(samples: list[dict], sessions: dict) -> None:
    """Check that every selected source turn survives the lossless adapter."""
    histories = {str(s['sample_id']): core.session_data(s) for s in samples}
    if core.digest(histories) != core.digest(sessions):
        raise ValueError('Full canonical source histories differ from the imported source')


def plan(args: argparse.Namespace) -> None:
    """Freeze imported lineage and canonical history verification before construction."""
    if any((args.out / name).exists() for name in ('cache', 'runtime', 'items', 'evaluation_timing', 'memories')):
        raise ValueError('Plan requires a fresh output with no reader caches, results, or memories')
    lineage = snapshot(args.import_run, args.out, args.dataset)
    original = read(args.out / 'imported_source/protocol.json')
    sessions = read(args.out / 'imported_source/source_sessions.json')
    samples = samples_for(args.data, original)
    verify_histories(samples, sessions)
    ids = [str(s['sample_id']) if args.dataset == 'longmemeval' else f"{s['sample_id']}:{i}"
           for s in samples for i, qa in enumerate(s['qa'])
           if args.dataset != 'locomo' or int(qa['category']) in (1, 2, 3, 4)]
    del samples
    protocol = deepcopy(original)
    protocol.update({
        'methods': [METHOD], 'evaluation_methods': [METHOD], 'source_hashes': source_hashes(),
        'execution_profile': EXECUTION_PROFILE, 'runtime_flags': RUNTIME_FLAGS,
        'context_capacity': {'writer': 65536, 'reader': 8192, 'scorer': 8192},
        'date_metadata_compatibility': install_date_adapter(),
        'import_lineage_sha256': sha(args.out / 'import_lineage.json'),
        'import_lineage': lineage, 'cost_policy': COST_POLICY,
        'construction': {'parent': 'seed', 'augment': 'unchanged portable_parent.augment',
                         'family': 'parent_single', 'extra_storage_budget': 2000,
                         'source_fit_utility': 'imported frozen source singleton/pair likelihood results',
                         'original_backend_applied': False, 'seed_units_preserved': True},
        'tokenizer_files_sha256': TOKENIZER_FILES,
        'primary_contrast': METHOD + ' minus seed', 'secondary_contrast': METHOD + ' minus s_parent_single_2000',
        'model_transfer': 'This Qwen ablation imports Qwen writer/scorer outputs and uses the same Qwen reader.',
        'target_selection': False, 'history_truncation': False,
        'selection_limitation': 'Previously exposed development examples; no held-out generalization claim.',
        'fresh_output_no_imported_cache': True,
    })
    protocol['config'] = {'dataset': args.dataset, 'out': args.runtime_out, 'model': MODEL,
                          'embed_model': EMBED_MODEL, 'embed_batch_size': 4, 'seed': 20260907}
    locked(args.out / 'protocol.json', protocol)
    locked(args.out / 'source_sessions.json', sessions)
    locked(args.out / 'expected_ids.json', ids)
    locked(args.out / 'history_verification.json', {'dataset_sha256': original['dataset_sha256'],
        'source_histories_digest': core.digest(sessions), 'expected_ids_sha256': sha(args.out / 'expected_ids.json'),
        'full_selected_histories_match_canonical': True, 'benchmark_qa_passed_to_constructor': False})
    state(args.out, 'planned', conversations=len(sessions), questions=len(ids), target_outcomes_used=False)


def verify_run(out: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Revalidate the frozen protocol, source imports, and history verification."""
    protocol, lineage = read(out / 'protocol.json'), read(out / 'import_lineage.json')
    if (protocol['methods'] != [METHOD] or protocol['evaluation_methods'] != [METHOD]
            or protocol['source_hashes'] != source_hashes()
            or protocol['import_lineage'] != lineage
            or protocol['import_lineage_sha256'] != sha(out / 'import_lineage.json')):
        raise ValueError('New protocol, source, or imported lineage changed')
    verify_snapshot(out / 'imported_source', lineage)
    actual, sessions, seeds, rows = imported_files(out / 'imported_source', protocol['config']['dataset'])
    history = read(out / 'history_verification.json')
    if (actual != lineage or read(out / 'source_sessions.json') != sessions
            or history != {'dataset_sha256': protocol['dataset_sha256'],
                'source_histories_digest': core.digest(sessions),
                'expected_ids_sha256': sha(out / 'expected_ids.json'),
                'full_selected_histories_match_canonical': True, 'benchmark_qa_passed_to_constructor': False}):
        raise ValueError('Imported source or canonical-history verification changed')
    return protocol, lineage, sessions, seeds, rows


def construct(args: argparse.Namespace) -> None:
    """Augment intact Seed memories using only the imported source-fit rows."""
    protocol, lineage, sessions, seeds, rows = verify_run(args.out)
    verify_tokenizer(args.tokenizer)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    class TokenRuntime:
        ntok = staticmethod(lambda text: len(tok.encode(text, add_special_tokens=False)))
    rt, memories, summary = TokenRuntime(), {}, {}
    began = time.perf_counter()
    for cid, history in sorted(sessions.items()):
        parent = deepcopy(seeds[cid])
        memory, details, _ = augment(rt, history, parent, rows.get(cid, []))
        if parent != seeds[cid] or memory[:len(parent)] != seeds[cid]:
            raise ValueError('Seed parent was modified')
        added = sum(storage_cost(u, rt.ntok) for u in memory[len(parent):])
        if added > 2000 or details['storage_tokens'] > details['parent_tokens'] + 2000:
            raise ValueError('Added key-plus-payload storage budget exceeded')
        memories[cid] = memory
        locked(args.out / 'memories' / METHOD / (cid + '.json'), memory)
        locked(args.out / 'construction' / (cid + '.json'), details)
        summary[cid] = {'seed_units': len(parent), 'added_units': len(memory) - len(parent),
                        'added_key_and_payload_tokens': added, 'selected_options': details['selected_options']}
    verify_snapshot(args.out / 'imported_source', lineage)
    verify_tokenizer(args.tokenizer)
    locked(args.out / 'memory_lock.json', {'protocol_sha256': core.digest(protocol),
        'source_sha256': core.digest(sessions), 'memories_sha256': core.digest({METHOD: memories}),
        'benchmark_questions_used': False, 'seed_units_unchanged': True,
        'import_lineage_sha256': protocol['import_lineage_sha256']})
    receipt = {'method': METHOD, 'conversations': summary, 'construction_seconds': time.perf_counter() - began,
        'scope': 'CPU augmentation, persistence and final hash rechecks; excludes initial validation and tokenizer loading. No new LLM or embedding calls; imported costs are not zero.',
        'new_llm_tokens': 0, 'new_embedding_tokens': 0, 'cost_policy': COST_POLICY,
        'tokenizer_files_sha256': TOKENIZER_FILES, 'transformers_version': importlib.metadata.version('transformers'),
        'protocol_sha256': sha(args.out / 'protocol.json'), 'memory_lock_sha256': sha(args.out / 'memory_lock.json')}
    # Reconstructing an identical lock must not overwrite the first timing observation.
    if not (args.out / 'construction_receipt.json').exists():
        core.save(args.out / 'construction_receipt.json', receipt)
    state(args.out, 'memories_locked', conversations=len(sessions))


def evaluate(args: argparse.Namespace) -> None:
    """Read the locked new arm with the unchanged, natively metered reader."""
    protocol, lineage, sessions, _, _ = verify_run(args.out)
    memories = {cid: read(args.out / 'memories' / METHOD / (cid + '.json')) for cid in sessions}
    lock = read(args.out / 'memory_lock.json')
    if (lock['protocol_sha256'] != core.digest(protocol) or lock['source_sha256'] != core.digest(sessions)
            or lock['memories_sha256'] != core.digest({METHOD: memories})
            or lock.get('benchmark_questions_used') is not False or lock.get('seed_units_unchanged') is not True):
        raise ValueError('Ablation memory lock changed')
    if str(args.out) != protocol['config']['out']:
        raise ValueError('Actual evaluation output path differs from declared runtime output')
    environment = read(args.environment)
    for name, key in ((MODEL, 'model'), (EMBED_MODEL, 'embedding')):
        if environment['models'][name] != protocol[key]:
            raise ValueError('Runtime model metadata differs from imported protocol')
    packages = {p: importlib.metadata.version(p) for p in protocol['actual_packages']}
    if packages != protocol['actual_packages']:
        raise ValueError('Runtime package versions differ from frozen comparison')
    verify_tokenizer(Path(environment['models'][MODEL]['path']))
    selected = samples_for(args.data, protocol)
    verify_histories(selected, sessions)
    for key in ('model', 'embed_model', 'embed_batch_size', 'seed'):
        setattr(args, key, protocol['config'][key])
    state(args.out, 'loading_reader', model=MODEL)
    rt = Runtime(args, environment)
    result = []
    for sample in selected:
        cid = str(sample['sample_id'])
        state(args.out, 'evaluating', method=METHOD, conversation=cid)
        target = args.out / 'items' / METHOD / (cid + '.json')
        if not target.exists():
            core.save(target, evaluate_sample(rt, memories[cid], sample, METHOD, protocol['config']['dataset']))
        result.extend(read(target))
    if [r['question_id'] for r in result] != read(args.out / 'expected_ids.json'):
        raise ValueError('Final answer coverage differs from the locked population')
    verify_snapshot(args.out / 'imported_source', lineage)
    target = args.out / (METHOD + '.jsonl')
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result), encoding='utf-8')
    state(args.out, 'generation_complete', questions=len(result),
          official_judge_pending=protocol['config']['dataset'] == 'longmemeval')


def main() -> None:
    """Dispatch explicitly separated planning, construction, and evaluation stages."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('plan', 'construct', 'evaluate'))
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--dataset', choices=('locomo', 'longmemeval'))
    ap.add_argument('--import-run', type=Path)
    ap.add_argument('--data', type=Path)
    ap.add_argument('--runtime-out')
    ap.add_argument('--tokenizer', type=Path)
    ap.add_argument('--environment', type=Path)
    args = ap.parse_args()
    required = {'plan': ('dataset', 'import_run', 'data', 'runtime_out'),
                'construct': ('tokenizer',), 'evaluate': ('data', 'environment')}[args.stage]
    for name in required:
        if getattr(args, name) is None:
            ap.error('--' + name.replace('_', '-') + ' is required for ' + args.stage)
    if args.stage == 'construct' and args.data is not None:
        ap.error('Construction accepts no target dataset; use the sealed imported sources')
    original_function = inspect.getsource(approved.evaluate_sample)
    # During timed evaluation the public function is decorated with functools.wraps.
    if inspect.getsource(inspect.unwrap(evaluate_sample)) != original_function:
        raise ValueError('Copied frozen evaluate_sample function changed')
    install_date_adapter()
    {'plan': plan, 'construct': construct, 'evaluate': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()

