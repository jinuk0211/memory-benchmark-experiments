"""Isolated, resumable recursive construction; benchmark QA loads only in evaluate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
import run_transfer as baseline_api
from recursive_memory import RECIPE, refine_memory
from transfer_runtime import Runtime, set_runtime_context

METHOD = RECIPE['name']


def read(path: Path):
    return json.loads(path.read_text(encoding='utf8'))


def lock(path: Path, value) -> None:
    if path.exists() and read(path) != value:
        raise ValueError('Frozen recursive artifact changed: ' + str(path))
    if not path.exists():
        core.save(path, value)


def state(out: Path, phase: str, **fields) -> None:
    core.save(out / 'status.json', {'pid': os.getpid(), 'time': time.time(), 'phase': phase, **fields})
    set_runtime_context(phase, **fields)
    print(phase, json.dumps(fields), flush=True)


def verify_baseline(root: Path) -> tuple[dict, dict, dict]:
    protocol = read(root / 'protocol.json')
    if protocol['config']['embed_model'] != 'sentence-transformers/all-MiniLM-L6-v2':
        raise ValueError('Recursive baseline must use MiniLM')
    for name, expected in protocol['source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Baseline code changed: ' + name)
    sessions = read(root / 'source_sessions.json')
    memories = {method: {cid: read(root / 'memories' / method / (cid + '.json')) for cid in sessions}
                for method in baseline_api.METHODS}
    marker = read(root / 'memory_lock.json')
    if (marker['protocol_sha256'] != core.digest(protocol)
            or marker['source_sha256'] != core.digest(sessions)
            or marker['memories_sha256'] != core.digest(memories)):
        raise ValueError('Starting memory/protocol/source lock mismatch')
    if read(root / 'status.json')['phase'] != 'generation_complete':
        raise ValueError('Original comparison generation is not complete')
    return protocol, sessions, memories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('construct', 'evaluate'))
    parser.add_argument('--baseline', type=Path, default=ROOT / 'runs/qwen35_baseline')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/recursive_v1')
    parser.add_argument('--environment', type=Path, default=ROOT / 'environment.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/locomo10.json')
    args = parser.parse_args()
    parent_protocol, sessions, memories = verify_baseline(args.baseline)
    args.model = parent_protocol['config']['model']
    args.embed_model = parent_protocol['config']['embed_model']
    args.seed = parent_protocol['config']['seed']
    args.embed_batch_size = parent_protocol['config']['embed_batch_size']
    utility_selection = read(args.baseline / 'utility_selection.json')
    probe_pool = read(args.baseline / 'source_probes.json')
    if utility_selection['source_pool_sha256'] != core.digest(probe_pool):
        raise ValueError('Source probe pool differs from the baseline utility origin')
    expected = utility_selection['ids']
    construction_files = ['source_probes.json', 'utility_selection.json'] + [
        'utility/items/' + identity.replace(':', '_') + '.json' for identity in expected]
    input_hashes = {name: hashlib.sha256((args.baseline / name).read_bytes()).hexdigest()
                    for name in construction_files}
    protocol = {'construction_inputs_sha256': input_hashes, 'recipe': RECIPE, 'baseline_protocol_sha256': core.digest(parent_protocol),
                'baseline_memory_lock': read(args.baseline / 'memory_lock.json'),
                'environment': read(args.environment), 'dataset_sha256': parent_protocol['dataset_sha256'],
                'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ('recursive_memory.py', 'run_recursive.py', 'transfer_runtime.py',
                                               'runtime_meter.py', 'run_transfer.py')},
                'method': METHOD, 'data': str(args.data), 'out': str(args.out),
                'benchmark_qa_used_for_construction': False}
    lock(args.out / 'protocol.json', protocol)
    if args.stage == 'construct':
        pool = probe_pool['records']
        utility = [read(args.baseline / 'utility/items' / (identity.replace(':', '_') + '.json'))
                   for identity in expected]
        if [row['id'] for row in utility] != expected:
            raise ValueError('Source utility identity mismatch')
        fit_pool = {row['id']: row for row in pool if row['split'] == 'probe_fit'}
        if set(expected) != set(fit_pool):
            raise ValueError('Source utility partition differs from source probe pool')
        for row in utility:
            if any(row.get(key) != value for key, value in fit_pool[row['id']].items()):
                raise ValueError('Source utility question or provenance changed')
        state(args.out, 'loading_recursive_runtime')
        rt = Runtime(args, protocol['environment'])
        for cid, history in sorted(sessions.items()):
            target = args.out / 'memories' / (cid + '.json')
            if target.exists():
                continue
            state(args.out, 'recursive_construction', conversation=cid)
            def save_round(round_id, memory, record):
                core.save(args.out / 'rounds' / cid / (str(round_id) + '.json'), record)
                core.save(args.out / 'rounds' / cid / (str(round_id) + '_memory.json'), memory)
                state(args.out, 'recursive_round_complete', conversation=cid, round=round_id,
                      accepted=record['accepted'], before=record['before'], after=record['after'])
            result, history_log = refine_memory(rt, history, memories['r40_fused_four_turn'][cid],
                memories['s_parent_single_2000'][cid], [row for row in utility if row['conv_id'] == cid],
                [row for row in pool if row['conv_id'] == cid and row['split'] == 'probe_audit'], save_round)
            core.save(target, result)
            core.save(args.out / 'history' / (cid + '.json'), history_log)
        if input_hashes != {name: hashlib.sha256((args.baseline / name).read_bytes()).hexdigest()
                            for name in construction_files}:
            raise ValueError('Source construction inputs changed during refinement')
        final = {cid: read(args.out / 'memories' / (cid + '.json')) for cid in sessions}
        lock(args.out / 'memory_lock.json', {'protocol_sha256': core.digest(protocol),
             'memories_sha256': core.digest(final), 'source_sha256': core.digest(sessions),
             'benchmark_questions_used': False})
        state(args.out, 'memories_locked', conversations=len(final))
    else:
        # The benchmark file and QA are opened only after all final memories are locked.
        final = {cid: read(args.out / 'memories' / (cid + '.json')) for cid in sessions}
        marker = read(args.out / 'memory_lock.json')
        if marker['protocol_sha256'] != core.digest(protocol) or marker['memories_sha256'] != core.digest(final):
            raise ValueError('Recursive memory lock mismatch')
        if hashlib.sha256(args.data.read_bytes()).hexdigest() != protocol['dataset_sha256']:
            raise ValueError('Benchmark file changed')
        samples = read(args.data)
        if {str(sample['sample_id']) for sample in samples} != set(sessions):
            raise ValueError('Benchmark conversations mismatch')
        state(args.out, 'loading_recursive_reader')
        rt = Runtime(args, protocol['environment'])
        rows = []
        for sample in samples:
            cid = str(sample['sample_id'])
            state(args.out, 'evaluating_recursive_memory', conversation=cid)
            path = args.out / 'items' / (cid + '.json')
            if not path.exists():
                core.save(path, baseline_api.evaluate_sample(rt, final[cid], sample, METHOD, 'locomo'))
            rows.extend(read(path))
        if len(rows) != 1540 or len({row['question_id'] for row in rows}) != 1540:
            raise ValueError('Incomplete1540 recursive predictions')
        target = args.out / (METHOD + '.jsonl')
        target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf8')
        state(args.out, 'generation_complete', questions=1540)


if __name__ == '__main__':
    main()
