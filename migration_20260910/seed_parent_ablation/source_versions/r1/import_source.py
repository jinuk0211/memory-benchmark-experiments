"""Import source-only fit artifacts; never import target answers or runtime caches."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import refine as core
from compile_research_data import cards_from_generations, split_probe_pool

METHOD = 'seed_parent_single_2000'
ORIGINAL_METHODS = ('seed', 'r40_fused_four_turn', 's_parent_single_2000')
PINNED_PROTOCOLS = {
    'locomo': '50d32eb67ea295ea4d3c125cf301cfa0cbc98f877d946f0667929782a48c98b5',
    'longmemeval': 'e95bda1e33a0241fb53f7ca8c1fefeb5661e3f2c13a17990a05aa2894454aac0',
}
COST_POLICY = 'inherited_costs_separately_reported; new_runtime_is_incremental'
TOKENIZER_FILES = {
    'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715',
    'config.json': 'd0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05',
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
    'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
}


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked(path: Path, value) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen artifact changed: ' + str(path))
        return
    core.save(path, value)


def safe_file(root: Path, name: str) -> Path:
    path = root / name
    if (not name or ':' in name or '\\' in name or Path(name).is_absolute()
            or '..' in Path(name).parts or path.relative_to(root).as_posix() != name
            or not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents) if p != root.parent)):
        raise ValueError('Unsafe imported path: ' + name)
    if not path.is_file():
        raise ValueError('Missing imported file: ' + name)
    return path


def imported_files(root: Path, dataset: str) -> tuple[dict, dict, dict, dict]:
    """Validate source-fit identity and the original complete three-arm memory lock."""
    paths = ['protocol.json', 'memory_lock.json', 'source_sessions.json',
             'source_generations.json', 'source_probes.json', 'utility_selection.json']
    protocol = read(safe_file(root, paths[0]))
    if sha(root / paths[0]) != PINNED_PROTOCOLS[dataset]:
        raise ValueError('Original protocol does not match the frozen dataset pin')
    if (protocol['config']['dataset'] != dataset or protocol['methods'] != list(ORIGINAL_METHODS)
            or protocol.get('target_selection') is not False or protocol.get('history_truncation') is not False):
        raise ValueError('Original protocol is not the frozen source-only comparison')
    for name in paths[1:]:
        safe_file(root, name)
    sessions, pool = read(root / paths[2]), read(root / paths[4])
    selected = protocol['selection']['selected_ids']
    if not sessions or set(sessions) != set(selected) or len(selected) != len(set(selected)):
        raise ValueError('Original source history IDs differ from the frozen selection')
    for cid, history in sessions.items():
        if not history or any('qa' in s or 'evaluation_metadata' in s for s in history):
            raise ValueError('Construction source contains evaluation fields or an empty history')
        ids = [t['id'] for s in history for t in s['turns']]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate source turn IDs')
        if any(char in cid for char in ('/', '\\', ':')):
            raise ValueError('Unsafe conversation ID')
    rebuilt = split_probe_pool(sessions, cards_from_generations(read(root / paths[3])), seed=20260908, per_session=2)
    if rebuilt != pool:
        raise ValueError('Source probe pool cannot be reproduced from imported source generations')
    fit = sorted((r for r in pool['records'] if r['split'] == 'probe_fit'), key=lambda r: r['id'])
    ids = [r['id'] for r in fit]
    selection = read(root / paths[5])
    if (not ids or len(ids) != len(set(ids)) or selection != {'ids': ids, 'source_pool_sha256': core.digest(pool)}):
        raise ValueError('Selected utility IDs differ from all and only source-fit probes')
    rows = defaultdict(list)
    for probe in fit:
        name = 'utility/items/' + probe['id'].replace(':', '_') + '.json'
        row = read(safe_file(root, name))
        if any(row.get(k) != v for k, v in probe.items()):
            raise ValueError('Utility row is not bound to its selected source-fit probe')
        if not row.get('skip') and not all(k in row for k in ('analysis', 'generations', 'subsets')):
            raise ValueError('Incomplete source-utility item')
        paths.append(name)
        rows[probe['conv_id']].append(row)
    memories = {method: {} for method in ORIGINAL_METHODS}
    for method in ORIGINAL_METHODS:
        for cid in sorted(sessions):
            name = f'memories/{method}/{cid}.json'
            memories[method][cid] = read(safe_file(root, name))
            paths.append(name)
    lock = read(root / 'memory_lock.json')
    if (lock.get('protocol_sha256') != core.digest(protocol)
            or lock.get('source_sha256') != core.digest(sessions)
            or lock.get('memories_sha256') != core.digest(memories)
            or lock.get('benchmark_questions_used') is not False):
        raise ValueError('Original memory lock does not verify')
    files = {name: {'sha256': sha(safe_file(root, name)), 'bytes': (root / name).stat().st_size}
             for name in sorted(paths)}
    lineage = {'schema_version': 1, 'original_protocol_sha256': sha(root / 'protocol.json'),
               'original_protocol_digest': core.digest(protocol),
               'original_memory_lock_sha256': sha(root / 'memory_lock.json'),
               'source_histories_digest': core.digest(sessions), 'selected_probe_ids': ids,
               'cost_policy': COST_POLICY, 'fresh_output_no_imported_cache': True, 'files': files}
    return lineage, sessions, memories['seed'], dict(rows)


def snapshot(source: Path, out: Path, dataset: str) -> dict:
    lineage, _, _, _ = imported_files(source, dataset)
    destination = out / 'imported_source'
    for name, wanted in lineage['files'].items():
        target = destination / name
        if target.exists():
            if sha(target) != wanted['sha256']:
                raise ValueError('Existing import differs: ' + name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(source, name), target)
    if imported_files(source, dataset)[0] != lineage:
        raise ValueError('Original source changed during import')
    verify_snapshot(destination, lineage)
    locked(out / 'import_lineage.json', lineage)
    return lineage


def verify_snapshot(root: Path, lineage: dict) -> None:
    found = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if found != set(lineage['files']):
        raise ValueError('Imported source inventory changed')
    for name, wanted in lineage['files'].items():
        path = safe_file(root, name)
        if path.stat().st_size != wanted['bytes'] or sha(path) != wanted['sha256']:
            raise ValueError('Imported source hash changed: ' + name)


def verify_tokenizer(path: Path) -> None:
    for name, wanted in TOKENIZER_FILES.items():
        if sha(path / name) != wanted:
            raise ValueError('Pinned Qwen tokenizer file changed: ' + name)
