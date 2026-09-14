"""Isolated v2 runner; source calibration precedes fixed-view memory rehearsal."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
import run_transfer as baseline_api
import run_recursive as v1
import recursive_memory as rehearsal
from recursive_memory_v2 import RECIPE, refine_memory
from source_views_v2 import prepare_views, validate_bundle
from transfer_runtime import Runtime
from summarize_recursive import validate_rows

METHOD = RECIPE['name']


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inputs(root: Path) -> tuple[list[dict], list[dict], dict]:
    selection = v1.read(root / 'utility_selection.json')
    pool = v1.read(root / 'source_probes.json')
    if selection['source_pool_sha256'] != core.digest(pool):
        raise ValueError('Changed source-probe origin')
    ids = selection['ids']
    utility = [v1.read(root / 'utility/items' / (pid.replace(':', '_') + '.json')) for pid in ids]
    fit = {r['id']: r for r in pool['records'] if r['split'] == 'probe_fit'}
    if len(ids) != len(set(ids)) or [r['id'] for r in utility] != ids or set(ids) != set(fit):
        raise ValueError('Changed source-fit partition')
    for row in utility:
        if any(row.get(k) != value for k, value in fit[row['id']].items()):
            raise ValueError('Changed source question, answer or provenance')
    files = ['utility_selection.json', 'source_probes.json'] + [
        'utility/items/' + pid.replace(':', '_') + '.json' for pid in ids]
    return utility, pool['records'], {name: file_hash(root / name) for name in files}


def completion_value(out: Path, cid: str, protocol: dict, views: dict) -> dict:
    history = v1.read(out / 'history' / (cid + '.json'))
    if len(history) != RECIPE['rounds'] or [r['round'] for r in history] != list(range(1, RECIPE['rounds'] + 1)):
        raise ValueError('Incomplete conversation round history')
    files = ['memories/' + cid + '.json', 'history/' + cid + '.json']
    for record in history:
        stem = 'rounds/' + cid + '/' + str(record['round'])
        if v1.read(out / (stem + '.json')) != record:
            raise ValueError('Conversation history differs from saved round')
        if core.digest(v1.read(out / (stem + '_memory.json'))) != record['accepted_memory_sha256']:
            raise ValueError('Accepted round memory changed')
        files.extend((stem + '.json', stem + '_memory.json'))
    if core.digest(v1.read(out / 'memories' / (cid + '.json'))) != history[-1]['accepted_memory_sha256']:
        raise ValueError('Conversation final memory differs from accepted final round')
    return {'protocol_sha256': core.digest(protocol), 'views_sha256': core.digest(views),
            'artifacts_sha256': {name: file_hash(out / name) for name in files}}


def verify_completion(out: Path, cid: str, protocol: dict, views: dict) -> dict:
    receipt = v1.read(out / 'complete' / (cid + '.json'))
    if receipt != completion_value(out, cid, protocol, views):
        raise ValueError('Conversation completion receipt changed')
    return receipt


def verify_memory_lock(out: Path, protocol: dict, memories: dict, sessions: dict, bundles: dict) -> None:
    expected = {'protocol_sha256': core.digest(protocol), 'memories_sha256': core.digest(memories),
                'source_sha256': core.digest(sessions), 'views_sha256': core.digest(bundles),
                'completion_sha256': core.digest({cid: verify_completion(out, cid, protocol, bundles[cid]) for cid in sessions}),
                'benchmark_questions_used': False}
    if v1.read(out / 'memory_lock.json') != expected:
        raise ValueError('V2 final memory/view/source lock mismatch')


def diagnostic(rt, out: Path, protocol: dict, memories: dict, sessions: dict, bundles: dict) -> None:
    # Calibration against raw sources happens earlier. Candidate-memory QB use starts here.
    verify_memory_lock(out, protocol, memories, sessions, bundles)
    for cid, bundle in sorted(bundles.items()):
        path = out / 'post_lock_diagnostic' / (cid + '.json')
        probes = [{**r['original'], 'question': r['diagnostic_question'], 'split': 'probe_view_audit'}
                  for r in bundle['manifest']['records']]
        expected = {'memory_sha256': core.digest(memories[cid]), 'views_sha256': bundle['manifest_sha256'],
                    'selection_allowed': False}
        if not path.exists():
            v1.state(out, 'post_lock_source_diagnostic', conversation=cid)
            v1.lock(path, {**expected, 'results': rehearsal.rehearse(rt, memories[cid], probes)})
        saved = v1.read(path)
        if (any(saved.get(k) != value for k, value in expected.items())
                or set(saved['results']) != {p['id'] for p in probes}
                or any(r['split'] != 'probe_view_audit' for r in saved['results'].values())):
            raise ValueError('Post-lock diagnostic cache differs from fixed memory or probes')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('construct', 'evaluate'))
    parser.add_argument('--baseline', type=Path, default=ROOT / 'runs/qwen35_baseline')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/recursive_v2')
    parser.add_argument('--environment', type=Path, default=ROOT / 'environment.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/locomo10.json')
    args = parser.parse_args()
    parent_protocol, sessions, starting = v1.verify_baseline(args.baseline)
    utility, pool, hashes = source_inputs(args.baseline)
    for key in ('model', 'embed_model', 'seed', 'embed_batch_size'):
        setattr(args, key, parent_protocol['config'][key])
    sources = ('source_views_v2.py', 'recursive_memory_v2.py', 'run_recursive_v2.py',
               'run_recursive.py', 'recursive_memory.py', 'transfer_runtime.py', 'runtime_meter.py', 'run_transfer.py')
    protocol = {'recipe': RECIPE, 'method': METHOD, 'construction_inputs_sha256': hashes,
                'baseline_protocol_sha256': core.digest(parent_protocol),
                'baseline_memory_lock': v1.read(args.baseline / 'memory_lock.json'),
                'environment': v1.read(args.environment), 'dataset_sha256': parent_protocol['dataset_sha256'],
                'source_sha256': {name: file_hash(ROOT / name) for name in sources},
                'out': str(args.out), 'data': str(args.data), 'benchmark_qa_used_for_construction': False}
    v1.lock(args.out / 'protocol.json', protocol)
    if args.stage == 'construct':
        v1.state(args.out, 'loading_v2_runtime')
        rt = Runtime(args, protocol['environment'])
        for cid, sessions_for_conv in sorted(sessions.items()):
            rows = [r for r in utility if r['conv_id'] == cid]
            view_path = args.out / 'views' / (cid + '.json')
            if not view_path.exists():
                v1.state(args.out, 'calibrating_source_views', conversation=cid)
                views = prepare_views(rt, sessions_for_conv, rows, starting['s_parent_single_2000'][cid])
                v1.lock(view_path, views)
            views = v1.read(view_path)
            validate_bundle(views, rows)
            target = args.out / 'memories' / (cid + '.json')
            receipt = args.out / 'complete' / (cid + '.json')
            if receipt.exists():
                verify_completion(args.out, cid, protocol, views)
                continue
            # Unreceipted partial files are preserved and compared during cache replay.
            v1.state(args.out, 'recursive_v2_construction', conversation=cid)
            def save_round(round_id, memory, record):
                v1.lock(args.out / 'rounds' / cid / (str(round_id) + '.json'), record)
                v1.lock(args.out / 'rounds' / cid / (str(round_id) + '_memory.json'), memory)
                v1.state(args.out, 'recursive_v2_round_complete', conversation=cid, round=round_id,
                         accepted=record['accepted'], before=record['before'], after=record['after'])
            memory, history = refine_memory(rt, sessions_for_conv, starting['r40_fused_four_turn'][cid],
                starting['s_parent_single_2000'][cid], rows,
                [r for r in pool if r['conv_id'] == cid and r['split'] == 'probe_audit'], views, save_round)
            v1.lock(target, memory)
            v1.lock(args.out / 'history' / (cid + '.json'), history)
            v1.lock(receipt, completion_value(args.out, cid, protocol, views))
        if hashes != {name: file_hash(args.baseline / name) for name in hashes}:
            raise ValueError('Construction inputs changed during refinement')
    final = {cid: v1.read(args.out / 'memories' / (cid + '.json')) for cid in sessions}
    bundles = {cid: v1.read(args.out / 'views' / (cid + '.json')) for cid in sessions}
    for cid, bundle in bundles.items():
        validate_bundle(bundle, [r for r in utility if r['conv_id'] == cid])
    if args.stage == 'construct':
        v1.lock(args.out / 'memory_lock.json', {'protocol_sha256': core.digest(protocol),
            'memories_sha256': core.digest(final), 'source_sha256': core.digest(sessions),
            'views_sha256': core.digest(bundles),
            'completion_sha256': core.digest({cid: verify_completion(args.out, cid, protocol, bundles[cid]) for cid in sessions}),
            'benchmark_questions_used': False})
        v1.state(args.out, 'memories_locked', conversations=len(final))
        diagnostic(rt, args.out, protocol, final, sessions, bundles)
        v1.state(args.out, 'source_construction_complete', conversations=len(final))
        return
    verify_memory_lock(args.out, protocol, final, sessions, bundles)
    if file_hash(args.data) != protocol['dataset_sha256']:
        raise ValueError('Canonical benchmark changed')
    samples = v1.read(args.data)
    if {str(s['sample_id']) for s in samples} != set(sessions):
        raise ValueError('Benchmark conversations differ from source population')
    v1.state(args.out, 'loading_v2_reader')
    rt = Runtime(args, protocol['environment'])
    predictions = []
    for sample in samples:
        cid = str(sample['sample_id'])
        path = args.out / 'items' / (cid + '.json')
        v1.state(args.out, 'evaluating_v2_memory', conversation=cid)
        expected = {'protocol_sha256': core.digest(protocol), 'memory_sha256': core.digest(final[cid])}
        if not path.exists():
            v1.lock(path, {**expected, 'rows': baseline_api.evaluate_sample(rt, final[cid], sample, METHOD, 'locomo')})
        saved = v1.read(path)
        if any(saved.get(k) != value for k, value in expected.items()):
            raise ValueError('Prediction cache refers to different memory or protocol')
        predictions.extend(saved['rows'])
    if len(predictions) != 1540 or len({r['question_id'] for r in predictions}) != 1540:
        raise ValueError('Incomplete canonical 1540 prediction set')
    validate_rows(samples, predictions, METHOD)
    target = args.out / (METHOD + '.jsonl')
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in predictions)
    if target.exists() and target.read_text(encoding='utf8') != text:
        raise ValueError('Refusing to replace different predictions')
    target.write_text(text, encoding='utf8')
    v1.state(args.out, 'generation_complete', questions=1540)


if __name__ == '__main__':
    main()
