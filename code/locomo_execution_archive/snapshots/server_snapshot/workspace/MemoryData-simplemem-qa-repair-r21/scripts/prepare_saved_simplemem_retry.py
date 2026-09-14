"""Copy closed SimpleMem memories and seed native retry tracking, without inference."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from scripts import locomo_server_queue as queue
from scripts.finalize_locomo_comparison import get_template, normalize_records
from scripts.score_locomo_comparison import validate_predictions, write_report
from utils.artifact_paths import (
    build_hashed_runtime_dir,
    generate_agent_save_folder_path,
)

RETRIES = {'conv-43_qa97': (4, 681), 'conv-47_qa45': (6, 930), 'conv-47_qa108': (6, 993)}
RUNS = {'r17': 'locomo-simplemem-failed-contexts-20260908-r17',
        'r13': 'locomo-full-simplemem-validated-20260908-r13'}


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pinned(path: Path, receipt: dict, hashes: dict) -> Path:
    path = path.resolve()
    value = digest(path)
    if receipt['raw_source_sha256'].get(str(path)) != value:
        raise ValueError(f'Saved artifact is unpinned or changed: {path}')
    hashes[str(path)] = value
    return path


def validate_rows(rows: list[dict], expected: dict, template: str) -> None:
    """Validate failed metadata too; preserve native wrapped prompts and global IDs."""
    samples = list(dict.fromkeys(sample for sample, _ in expected))
    identities = {str(qa.get('question_id') or f'{sample}_qa{index}'): (samples.index(sample), offset, sample)
                  for offset, ((sample, index), qa) in enumerate(expected.items())}
    if len(expected) != 1540 or len(rows) != 1540 or len({row.get('qa_pair_id') for row in rows}) != 1540:
        raise ValueError('Native retry requires the exact original 1540 unique QA rows')
    for row in rows:
        qid, metadata = row.get('qa_pair_id'), row.get('eval_metadata') or {}
        identity = identities.get(qid)
        if (identity is None or type(row.get('context_id')) is not int or type(row.get('query_id')) is not int
                or (row['context_id'], row['query_id'], row.get('sample_id')) != identity
                or metadata.get('sample_id') != identity[2] or metadata.get('question_id') != qid
                or metadata.get('qa_pair_id') != qid or row.get('status') not in (None, 'failed')
                or row.get('question_id', qid if row.get('status') == 'failed' else None) != qid):
            raise ValueError('Native retry has corrupt original/context/global metadata')
    normalized = normalize_records(rows, expected, template)
    with tempfile.TemporaryDirectory() as staging:
        path = Path(staging) / 'validation_only.json'
        write_report(path, {'data': [{key: value for key, value in row.items() if key != 'status'} for row in normalized]})
        canonical = queue.canonical_predictions([path], expected)
    if validate_predictions(canonical, expected)[1]:
        raise ValueError('Native retry canonical metadata failed')
    if any(not row['output'].strip() for row in rows if row.get('status') != 'failed'):
        raise ValueError('A skipped native QA is empty')


def _closed(source: Path, receipt: dict, tag: str, hashes: dict) -> None:
    if (receipt.get('method') != 'simplemem' or receipt.get('run_id') != RUNS[tag]
            or receipt.get('original_attempt_closed') is not True):
        raise ValueError('Saved SimpleMem attempt has no closed original receipt')
    status = json.loads(_pinned(source / 'parallel_status.json', receipt, hashes).read_text())
    contexts = list(range(10)) if tag == 'r13' else [i for i in range(10) if i != 5]
    if status != receipt['parallel_status'] or status.get('active_contexts') != [] or status.get('failed') is not True:
        raise ValueError('Saved attempt status is open or inconsistent')
    for event in ('start', 'end'):
        entries = [row for row in status['events'] if row.get('event') == event]
        if (len(entries) != len(contexts) or any(type(row.get('context')) is not int for row in entries)
                or sorted(row['context'] for row in entries) != contexts):
            raise ValueError('Saved attempt lacks exact terminal worker identities')
        if event == 'end':
            failures = [i for i in contexts if i != 5] if tag == 'r13' else [4, 6]
            if any(type(row.get('exit_code')) is not int or (row['exit_code'] != 0) != (row['context'] in failures) for row in entries):
                raise ValueError('Saved terminal worker outcomes changed')


def prepare_artifacts(source: Path, destination: Path, expected: dict, agent: dict,
                      dataset: dict, receipt: dict, old_source: Path, old_receipt: dict) -> dict:
    """Seed r17 1417 + r13 context05 123 solely for skipping; copy ONLY ctx04/06 state.

    Marker, sidecar and every Lance/Tantivy byte are receipt-pinned. Runtime paths
    MUST be rekeyed using the new absolute native agent path. A separate client
    table probe must verify the copied row IDs before starting model services.
    """
    source, old_source, destination = Path(source).resolve(), Path(old_source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if any(root.is_relative_to(destination) or destination.is_relative_to(root) for root in (source, old_source)):
        raise ValueError('Source and recovery artifacts overlap')
    hashes, copies, lineage, rows, relative = {}, {}, [], [], None
    _closed(source, receipt, 'r17', hashes)
    _closed(old_source, old_receipt, 'r13', hashes)
    for context in range(10):
        root, pinned = (old_source, old_receipt) if context == 5 else (source, receipt)
        artifacts = root / f'context_{context:02d}' / 'artifacts'
        paths = list(artifacts.rglob('*_results.json'))
        if len(paths) != 1:
            raise ValueError('Missing or ambiguous original result shard')
        path = _pinned(paths[0], pinned, hashes)
        if relative is not None and path.relative_to(artifacts) != relative:
            raise ValueError('Original native result paths disagree')
        relative = path.relative_to(artifacts)
        shard = json.loads(path.read_text(encoding='utf-8'))['data']
        if any(row.get('context_id') != context for row in shard):
            raise ValueError('Wrong original context shard')
        rows.extend(shard)
        if context not in (4, 6):
            continue
        state = Path(generate_agent_save_folder_path(agent, dataset, context, str(artifacts))).resolve()
        new_state = Path(generate_agent_save_folder_path(agent, dataset, context, str(destination))).resolve()
        runtime = Path(build_hashed_runtime_dir(str(state), '_simplemem_runtime'))
        new_runtime = Path(build_hashed_runtime_dir(str(new_state), '_simplemem_runtime'))
        if (state / 'simplemem_ready.txt').read_text().strip() != 'ready':
            raise ValueError('Saved SimpleMem is not ready')
        source_map = json.loads((state / 'simplemem_source_map.json').read_text())
        if (not isinstance(source_map, dict) or not source_map or any(not isinstance(key, str)
                or not key or not isinstance(value, list) or not value for key, value in source_map.items())):
            raise ValueError('Saved SimpleMem provenance sidecar is empty or invalid')
        table = runtime / 'locomo_qa_memory.lance'
        if (not list((table / '_versions').glob('*.manifest')) or not list((table / 'data').glob('*.lance'))
                or not (table / '_indices/fts/meta.json').is_file()
                or not list((table / '_indices/fts').glob('*.idx'))):
            raise ValueError('Saved Lance/Tantivy memory is missing')
        for original, target in ((state, new_state), (runtime, new_runtime)):
            for item in original.rglob('*'):
                if item.is_symlink():
                    raise ValueError('Saved memory contains a symlink')
                if item.is_file():
                    checked = _pinned(item, receipt, hashes)
                    new = target / checked.relative_to(original)
                    if new in copies or not new.is_relative_to(destination):
                        raise ValueError('Saved memory destination escapes or overlaps')
                    copies[new] = checked
        lineage.append({'context_id': context, 'sample_id': list(dict.fromkeys(s for s, _ in expected))[context],
                        'source_agent': str(state), 'destination_agent': str(new_state),
                        'source_runtime': str(runtime), 'destination_runtime': str(new_runtime),
                        'table_name': 'locomo_qa_memory', 'entry_ids': sorted(source_map), 'run_id': RUNS['r17']})
    template = get_template(dataset['sub_dataset'], 'query', agent['agent_name'])
    validate_rows(rows, expected, template)
    actual = {row['qa_pair_id']: (row['context_id'], row['query_id']) for row in rows if row.get('status') == 'failed'}
    if actual != RETRIES or [row['query_id'] for row in rows] != list(range(1540)):
        raise ValueError('Only the three closed r17 failed QA may be retried')
    destination.mkdir(parents=True, exist_ok=False)
    for new, original in copies.items():
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, new)
        if digest(new) != hashes[str(original)]:
            raise ValueError('Saved memory copy changed')
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError('Original artifacts changed during preparation')
    result_path = destination / relative
    write_report(result_path, {'data': rows})
    return {'rows': 1540, 'retry_ids': sorted(RETRIES), 'skipped_seed_rows': 1537,
            'context_question_counts': [sum(sample == sid for sid, _ in expected)
                                        for sample in dict.fromkeys(sid for sid, _ in expected)],
            'results': str(result_path), 'source_sha256': hashes, 'memory_lineage': lineage,
            'copy_sha256': {str(path): digest(path) for path in copies},
            'seed_is_final_selection': False,
            'note': 'Skip seed only. Final selection remains old r13 848 first, r17 689 fills, then these three repairs.'}
