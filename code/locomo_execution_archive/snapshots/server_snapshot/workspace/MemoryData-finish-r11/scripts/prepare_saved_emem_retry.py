"""Assemble closed E-Mem shards for the native --retry_failed_queries path."""
import hashlib
import json
from pathlib import Path
import shutil


def validate_memory(storage: Path) -> None:
    """Reject missing native blocks before its loader could create empty files."""
    metadata = json.loads((storage / 'agents_metadata.json').read_text(encoding='utf-8'))
    if not isinstance(metadata, list) or not metadata:
        raise ValueError('Saved E-Mem metadata is empty or invalid')
    chunks_total = 0
    for item in metadata:
        target = storage / f"text_block_{item['block_id']}_{item['timestamp']}.json"
        if not target.resolve().is_relative_to(storage.resolve()):
            raise ValueError('Saved block path escapes memory directory')
        data = json.loads(target.read_text(encoding='utf-8'))
        chunks = data.get('chunks')
        if (not isinstance(chunks, list) or any(not isinstance(chunk.get('text'), str)
                or type(chunk.get('tokens')) is not int or chunk['tokens'] < 0 for chunk in chunks)):
            raise ValueError('Saved block chunks are invalid')
        if (data.get('block_id') != item['block_id'] or data.get('create_timestamp') != item['timestamp']
                or data.get('chunk_num') != len(chunks) or item.get('chunk_number') != len(chunks)
                or data.get('block_used') != sum(chunk['tokens'] for chunk in chunks)
                or item.get('block_used') != data.get('block_used')):
            raise ValueError('Saved block counts or identity disagree with metadata')
        chunks_total += len(chunks)
    if not chunks_total:
        raise ValueError('Saved E-Mem has no memory chunks')


def prepare_artifacts(source: Path, destination: Path, expected_ids: dict[str, tuple[int, int]]) -> dict:
    """Copy all saved memories and original result rows into a fresh artifact root."""
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Source and recovery artifacts must not overlap')
    rows, copies, hashes = [], {}, {}
    result_relative = None
    for index in range(10):
        artifacts = source / f'context_{index:02d}' / 'artifacts'
        results = list(artifacts.rglob('*_results.json'))
        ready = list((artifacts / 'agents').rglob('e_mem_ready.txt'))
        if len(results) != 1 or len(ready) != 1:
            raise ValueError(f'Context {index} lacks unique results or saved E-Mem memory')
        validate_memory(ready[0].parent / 'e_mem_text')
        relative = results[0].relative_to(artifacts)
        if result_relative is not None and relative != result_relative:
            raise ValueError('Shard result paths disagree')
        result_relative = relative
        data = json.loads(results[0].read_text(encoding='utf-8'))['data']
        if not data or any(row.get('context_id') != index for row in data):
            raise ValueError('Shard has incorrect original context IDs')
        rows.extend(data)
        hashes[str(results[0])] = hashlib.sha256(results[0].read_bytes()).hexdigest()
        for path in (artifacts / 'agents').rglob('*'):
            if path.is_file():
                target = path.relative_to(artifacts)
                if target in copies:
                    raise ValueError('Saved agent files overlap between contexts')
                copies[target] = path
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    ids = [row.get('qa_pair_id') for row in rows]
    if len(rows) != 1540 or len(set(ids)) != 1540 or set(ids) != set(expected_ids):
        raise ValueError('Recovery requires the exact 1540 original QA rows')
    if any((row.get('context_id'), row.get('query_id')) != expected_ids[row['qa_pair_id']] for row in rows):
        raise ValueError('Recovery has incorrect original global query IDs')
    retry_ids = sorted(row['qa_pair_id'] for row in rows if row.get('status') == 'failed')
    if not retry_ids:
        raise ValueError('There are no failed questions to recover')
    destination.mkdir(parents=True, exist_ok=False)
    for relative, path in copies.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != hashes[str(path)]:
            raise ValueError('Saved memory copy changed')
    result = destination / result_relative
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open('x', encoding='utf-8') as stream:
        json.dump({'data': rows}, stream, ensure_ascii=False)
    return {'rows': len(rows), 'retry_ids': retry_ids, 'results': str(result),
            'source_sha256': hashes}
