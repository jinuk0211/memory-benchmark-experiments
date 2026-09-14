"""Freeze the full LoCoMo population against the two existing ablation memories."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'locomo_ablation300_20260913'
ARMS = ('ours', 'no_binding')
DATASET = ROOT.parent / 'recursive_minilm_20260909/data/locomo10.json'
DATASET_SHA = 'cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993'


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Frozen file differs: {path}')
    path.write_text(text, encoding='utf-8')


def main() -> None:
    if sha(DATASET) != DATASET_SHA:
        raise ValueError('Canonical dataset changed')
    prior_manifest = read(PRIOR / 'package/input/manifest.json')
    prior_contract = read(PRIOR / 'collected/run/protocol.json')
    if prior_contract['manifest'] != prior_manifest:
        raise ValueError('Prior protocol mismatch')
    for name, digest in prior_manifest['code'].items():
        if sha(PRIOR / 'package/code' / name) != digest:
            raise ValueError(f'Prior code changed: {name}')
    for name, digest in prior_manifest['files'].items():
        if sha(PRIOR / 'package/input' / name) != digest:
            raise ValueError(f'Prior input changed: {name}')
    package = ROOT / 'package'
    code, inputs = package / 'code', package / 'input'
    shutil.copytree(PRIOR / 'package/code', code, dirs_exist_ok=True)
    runner = (code / 'run_ablation.py').read_text(encoding='utf-8')
    old_arms = '("ours", "no_cues", "no_audit", "no_temporal", "no_binding",\n        "random_cues", "payload_keys")'
    if runner.count(old_arms) != 1:
        raise ValueError('Prior runner arms unexpectedly changed')
    runner = runner.replace(old_arms, '("ours", "no_binding")')
    runner = runner.replace('Matched LoCoMo300 component ablations', 'Matched full-LoCoMo evidence-binding ablation')
    (code / 'run_ablation.py').write_text(runner, encoding='utf-8')
    # The unchanged construction helper still produces seven variants; readers use two.
    tests = (code / 'test_ablation.py').read_text(encoding='utf-8')
    tests = tests.replace('self.assertEqual(set(memories), set(runner.ARMS))',
                          'self.assertEqual(set(memories), {"ours", "no_cues", "no_audit", '
                          '"no_temporal", "no_binding", "random_cues", "payload_keys"})')
    (code / 'test_ablation.py').write_text(tests, encoding='utf-8')
    sources = read(PRIOR / 'package/input/source_sessions.json')
    save(inputs / 'source_sessions.json', sources)
    questions, gold = [], []
    for sample in read(DATASET):
        cid = str(sample['sample_id'])
        for index, qa in enumerate(sample['qa']):
            if int(qa['category']) not in (1, 2, 3, 4):
                continue
            row = {'id': f'{cid}:{index}', 'conv_id': cid, 'qa_index': index,
                   'category': int(qa['category']), 'question': qa['question']}
            questions.append(row)
            gold.append({**row, 'gold': str(qa['answer'])})
    questions.sort(key=lambda r: (r['conv_id'], r['qa_index']))
    gold.sort(key=lambda r: (r['conv_id'], r['qa_index']))
    if len(questions) != 1540 or len({r['id'] for r in questions}) != 1540:
        raise ValueError('Expected1540 unique category1-4 questions')
    for row in read(PRIOR / 'package/input/reader_questions.json'):
        if row not in questions:
            raise ValueError('Original300 population not preserved')
    selection = [{k: r[k] for k in ('id', 'conv_id', 'qa_index', 'category')} for r in questions]
    save(inputs / 'selection.json', {'records': selection, 'selected': 1540,
         'source_population': 1540, 'selection': 'All canonical category1-4 questions; no sampling',
         'category_counts': dict(Counter(r['category'] for r in questions)),
         'conversation_counts': dict(Counter(r['conv_id'] for r in questions))})
    save(inputs / 'reader_questions.json', questions)
    save(ROOT / 'evaluation/gold.json', gold)
    memory_origin = {}
    cache_files = {}
    for cid in sorted(sources):
        original_lock_path = PRIOR / 'collected/run/locks' / f'{cid}.json'
        lock = read(original_lock_path)
        if lock['benchmark_qa_used']:
            raise ValueError('Prior memory used benchmark QA')
        memory_origin[cid] = {'original_lock_sha256': sha(original_lock_path),
                              'memories': {arm: lock['memories'][arm] for arm in ARMS}}
        lock['memories'] = {arm: lock['memories'][arm] for arm in ARMS}
        lock['stored_tokens'] = {arm: lock['stored_tokens'][arm] for arm in ARMS}
        save(inputs / 'locks' / f'{cid}.json', lock)
        for arm in ARMS:
            source = PRIOR / 'collected/run/memories' / arm / f'{cid}.json'
            if sha(source) != lock['memories'][arm]:
                raise ValueError('Prior memory fingerprint mismatch')
            dest = inputs / 'memories' / arm / f'{cid}.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    for shard in range(2):
        source = PRIOR / 'collected/run' / f'evaluate_{shard}/cache/generations'
        dest = inputs / 'generation_cache' / str(shard)
        dest.mkdir(parents=True, exist_ok=True)
        for path in source.glob('*.json'):
            shutil.copyfile(path, dest / path.name)
            cache_files[f'{shard}/{path.name}'] = sha(path)
    save(inputs / 'lineage.json', {'prior_protocol_sha256': sha(PRIOR / 'collected/run/protocol.json'),
         'prior_input_manifest_sha256': sha(PRIOR / 'package/input/manifest.json'),
         'memory_origin': memory_origin, 'generation_cache_sha256': cache_files,
         'cache_policy': 'Reuse only exact content-addressed reader requests; validate all imported cache files.'})
    protocol = '''# Full LoCoMo evidence-binding ablation

All1540 category1-4 questions; two arms: ours and no_binding. Reuse the exact frozen source-only memories constructed for the prior300-question run. No new extraction, audit, cue selection, or outcome-based tuning. Match Qwen3.5-9B FP16, TP2, MiniLM, temperature0, seed20260907,2048 evidence tokens and96 output tokens. All question IDs/categories and imported memory/cache hashes are frozen before execution. Gold remains local. Reuse cached answers only for identical content-addressed prompts. Compare against the same-memory full1540 ours run, not the archived55.5278 baseline. Store each native answer/context/token receipt; score official F1 including empty/length-ended answers. Use a separate output folder and retain the prior300 report. The full set includes the prior300 diagnostic questions and historically exposed conversations; this is a follow-up exploratory comparison, not an unseen holdout.
'''
    (code / 'PROTOCOL.md').write_text(protocol, encoding='utf-8')
    files = {str(p.relative_to(inputs)).replace('\\', '/'): sha(p)
             for p in sorted(inputs.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    code_hashes = {str(p.relative_to(code)).replace('\\', '/'): sha(p)
                  for p in sorted(code.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    save(inputs / 'manifest.json', {'files': files, 'code': code_hashes,
         'original_dataset_sha256': DATASET_SHA, 'sample_count': 1540,
         'arms': list(ARMS), 'parent_protocol_sha256': sha(PRIOR / 'collected/run/protocol.json')})
    shutil.copytree(code / 'vendor', ROOT / 'vendor', dirs_exist_ok=True)
    print(json.dumps({'questions': len(questions), 'arms': ARMS, 'memory_files': 20,
                      'imported_generation_receipts': len(cache_files)}, indent=2))


if __name__ == '__main__':
    main()
