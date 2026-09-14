"""Read-only accounting for the two completed Qwen/MiniLM LoCoMo runs.

Only native generation and embedding calls contribute cost. The manifest binds
collected files, not the completeness of the original runtime instrumentation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

METHODS = ('seed', 'r40_fused_four_turn', 's_parent_single_2000',
           'recursive_source_rehearsal_v1')
RUNS = ('qwen35_baseline', 'recursive_v1')
N = 1540
MODEL = 'Qwen/Qwen3.5-9B'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
EMBED_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
DATA_HASH = 'cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993'


def digest(value: Any) -> str:
    """The existing source/refine.py protocol serializer (no trailing newline)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def ids_count(value: Any) -> int | None:
    return len(value) if isinstance(value, list) and all(integer(n) for n in value) else None


def generation_usage(response: Any, count: int) -> dict:
    """Recompute runtime_meter.generation_usage from serialized native outputs."""
    rows = response if isinstance(response, list) else []
    prompts, completions, finishes, generations = [], [], [], 0
    for row in rows:
        prompts.append(ids_count(row.get('prompt_token_ids')))
        candidates = row.get('outputs')
        candidates = candidates if isinstance(candidates, list) else []
        generations += len(candidates)
        values = [ids_count(c.get('token_ids')) for c in candidates]
        completions.append(sum(values) if values and all(n is not None for n in values) else None)
        finishes.extend(c.get('finish_reason') for c in candidates)
    prompt = sum(prompts) if prompts and all(n is not None for n in prompts) else None
    completion = sum(completions) if completions and all(n is not None for n in completions) else None
    complete = (prompt is not None and completion is not None and len(rows) == count
                and all(row.get('finished') is True for row in rows))
    return {'prompt_tokens': prompt if complete else None,
            'completion_tokens': completion if complete else None,
            'recorded_prompt_tokens': sum(n for n in prompts if n is not None),
            'recorded_completion_tokens': sum(n for n in completions if n is not None),
            'total_tokens': prompt + completion if complete else None,
            'request_count': count, 'response_count': len(rows), 'generation_count': generations,
            'finish_reasons': finishes, 'usage_complete': complete}


def embedding_usage(features: dict) -> dict:
    masks, ids = features['attention_mask'], features['input_ids']
    if not isinstance(masks, list) or not masks or not isinstance(masks[0], list):
        raise ValueError('Invalid embedding masks')
    width = len(masks[0])
    if (not 0 < width <= 256 or not isinstance(ids, list) or len(ids) != len(masks)
            or any(not isinstance(row, list) or len(row) != width
                   or any(type(n) is not int or n not in (0, 1) for n in row) for row in masks)
            or any(not isinstance(row, list) or len(row) != width or ids_count(row) is None for row in ids)):
        raise ValueError('Invalid embedding IDs/masks/window')
    counts = [sum(row) for row in masks]
    return {'input_ids': ids, 'attention_mask': masks, 'shape': [len(masks), width],
            'tokens_per_input': counts, 'input_tokens': sum(counts),
            'padded_tokens': len(masks) * width, 'usage_complete': True}


def empty_totals() -> dict:
    return dict(llm_prompt_tokens_known=0, llm_completion_tokens_known=0,
                embedding_tokens_known=0, embedding_padded_tokens_known=0,
                llm_inference_seconds_known=0.0, embedding_inference_seconds_known=0.0,
                native_generation_calls=0, native_generation_requests=0, native_embedding_calls=0,
                failed_native_calls=0, incomplete_usage_calls=0, incomplete_time_calls=0,
                native_metrics_present=0, native_metrics_absent=0)


def finish_totals(totals: dict, gaps: bool = False) -> dict:
    result = dict(totals)
    result['measurement_complete'] = not (gaps or totals['incomplete_usage_calls'] or totals['incomplete_time_calls'])
    known = totals['llm_prompt_tokens_known'] + totals['llm_completion_tokens_known']
    result['llm_tokens_known_lower_bound'] = known
    result['llm_tokens_M'] = known / 1e6 if result['measurement_complete'] else None
    result['embedding_tokens_M'] = totals['embedding_tokens_known'] / 1e6 if result['measurement_complete'] else None
    result['llm_tokens_M_known_lower_bound'] = known / 1e6
    result['embedding_tokens_M_known_lower_bound'] = totals['embedding_tokens_known'] / 1e6
    result['native_inference_seconds_known_lower_bound'] = (totals['llm_inference_seconds_known']
                                                           + totals['embedding_inference_seconds_known'])
    return result


class Evidence:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.gaps = []
        self.files = {}
        self.bad = set()
        manifest = read(self.root / 'evidence_manifest.json')
        for entry in manifest['files']:
            name = entry['path']
            path = self.root / name
            if (Path(name).is_absolute() or '\\' in name or '..' in Path(name).parts
                    or not path.resolve().is_relative_to(self.root) or name in self.files):
                raise ValueError('Unsafe/duplicate manifest path: ' + name)
            self.files[name] = entry
            if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                self.bad.add(name)
                self.gap(name, 'Manifest file missing, size changed, or SHA256 mismatch')
        for run in RUNS:
            for part in ('runtime', 'utility/runtime'):
                for path in (self.root / 'runs' / run / part).rglob('*.json'):
                    name = path.relative_to(self.root).as_posix()
                    if name not in self.files:
                        self.gap(name, 'Unmanifested runtime JSON excluded')

    def gap(self, path: str, reason: Any) -> None:
        self.gaps.append({'path': str(path), 'reason': str(reason)})

    def load(self, name: str) -> Any:
        if name not in self.files or name in self.bad:
            raise ValueError('Missing or invalid manifest binding: ' + name)
        return read(self.root / name)


def group_key(name: str, receipt: dict) -> tuple:
    parts = name.split('/')
    run = parts[1]
    index = parts.index('runtime')
    stage, pid, instance = parts[index + 1:index + 4]
    context = receipt.get('context', {})
    phase = context.get('phase', 'unattributed')
    method = context.get('method', METHODS[-1] if run == 'recursive_v1' else 'shared')
    if stage != 'evaluate':
        method = 'recursive_construction' if run == 'recursive_v1' else 'shared_ancestor'
    return (run, stage, phase, method), (run, stage, pid, instance), index


def validate_profile(profile: dict, baseline: dict, stage: str) -> None:
    role = 'scorer' if stage == 'score' else 'runtime'
    expected = {'role': role, 'model': baseline['model'], 'dtype': 'float16', 'quantization': None,
                'seed': 20260908 if role == 'scorer' else 20260907}
    if role == 'runtime':
        expected.update(embedding=baseline['embedding'], embedding_device='cpu',
                        embedding_dtype='float32', native_max_seq_length=256)
    if any(field not in profile or profile[field] != value for field, value in expected.items()):
        raise ValueError('Native profile differs from fixed model/embedding/seed/precision')


def aggregate_runtime(evidence: Evidence) -> tuple[dict, dict, dict]:
    baseline_protocol = evidence.load('runs/qwen35_baseline/protocol.json')
    groups = defaultdict(empty_totals)
    sequences = defaultdict(set)
    totals = empty_totals()
    records, identities, expected_files = {}, set(), set()
    operation_summary = Counter()
    coverage = defaultdict(Counter)
    for name in sorted(evidence.files):
        if not name.endswith('.receipt.json') or '/runtime/' not in name:
            continue
        try:
            receipt = evidence.load(name)
            key, session, index = group_key(name, receipt)
            validate_profile(receipt['profile'], baseline_protocol, key[1])
            parts = name.split('/')
            kind = receipt['kind']
            if kind not in ('generation', 'embedding', 'operation', 'encode_attempt'):
                raise ValueError('Unknown receipt kind')
            if (receipt['schema_version'] != 1 or parts[index + 4] != kind
                    or parts[-1] != f"{receipt['sequence']:08d}.receipt.json"
                    or receipt.get('status') not in ('success', 'error')):
                raise ValueError('Receipt kind/sequence/status/path mismatch')
            identity = (*session, receipt['sequence'])
            if identity in identities:
                raise ValueError('Duplicate meter invocation identity; excluded from sums')
            identities.add(identity)
            sequences[session].add(receipt['sequence'])
            request_name = name.replace('.receipt.json', '.request.json')
            expected_files.add(request_name)
            request = evidence.load(request_name)
            if sha(evidence.root / request_name) != receipt['request_sha256']:
                raise ValueError('Request digest mismatch')
            for field in ('schema_version', 'kind', 'sequence', 'profile', 'context', 'operation_id',
                          'operation_kind', 'encode_attempt_id'):
                if request.get(field) != receipt.get(field):
                    raise ValueError('Request/receipt field mismatch: ' + field)
            response = None
            response_name = name.replace('.receipt.json', '.response.json')
            if 'response_sha256' in receipt:
                expected_files.add(response_name)
                response = evidence.load(response_name)
                if sha(evidence.root / response_name) != receipt['response_sha256']:
                    raise ValueError('Response digest mismatch')
            if kind == 'generation':
                payload = request['payload']
                prompts = payload['args'][0] if payload.get('args') else payload['kwargs'].get('prompts')
                if not isinstance(prompts, list):
                    raise ValueError('Invalid generation request prompts')
                if response is not None:
                    computed = generation_usage(response, len(prompts))
                    if any(receipt.get(field) != value for field, value in computed.items()):
                        raise ValueError('Native generation token arithmetic/status mismatch')
                elif receipt['status'] == 'success' or receipt.get('usage_complete') is not False:
                    raise ValueError('Missing raw generation response')
                else:
                    computed = dict(recorded_prompt_tokens=0, recorded_completion_tokens=0,
                                    usage_complete=False, request_count=len(prompts))
            elif kind == 'embedding':
                computed = embedding_usage(request['payload']['features'])
                if any(receipt.get(field) != value for field, value in computed.items()):
                    raise ValueError('Native embedding token arithmetic mismatch')
            duration = receipt.get('inference_duration_s')
            if kind in ('generation', 'embedding') and duration is not None:
                if (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
                        or receipt.get('call_duration_s') != duration):
                    raise ValueError('Invalid native inference duration')
            records[identity] = (name, receipt, request, key, session)
            groups[key]  # Preserve observed operations even when every result was cached.
            if kind not in ('generation', 'embedding'):
                operation_summary[kind + '_receipts_excluded_from_native_sums'] += 1
                if kind == 'operation':
                    if key[1] == 'evaluate' and receipt['operation_kind'] == 'generation' and receipt['status'] == 'success':
                        users = request['payload']['inputs']['users']
                        if not isinstance(users, list) or not all(isinstance(user, str) for user in users):
                            raise ValueError('Invalid evaluation generation operation input list')
                        coverage[key[3]][receipt['context']['conversation']] += len(users)
                    operation_summary['explicit_cache_hit_proven'] += receipt.get('cache_hit_proven') is True
                    operation_summary['zero_invocation_operations_not_proven_cache_hits'] += (
                        receipt.get('zero_invocations_observed') is True and receipt.get('cache_hit_proven') is not True)
                continue
            for target in (totals, groups[key]):
                target['failed_native_calls'] += receipt['status'] == 'error'
                target['incomplete_usage_calls'] += computed['usage_complete'] is not True
                target['incomplete_time_calls'] += duration is None
                target['llm_inference_seconds_known' if kind == 'generation' else 'embedding_inference_seconds_known'] += duration or 0
                if kind == 'generation':
                    target['native_generation_calls'] += 1
                    target['native_generation_requests'] += computed['request_count']
                    target['llm_prompt_tokens_known'] += computed['recorded_prompt_tokens']
                    target['llm_completion_tokens_known'] += computed['recorded_completion_tokens']
                    if isinstance(response, list):
                        target['native_metrics_present'] += sum(row.get('metrics') is not None for row in response)
                        target['native_metrics_absent'] += sum(row.get('metrics') is None for row in response)
                else:
                    target['native_embedding_calls'] += 1
                    target['embedding_tokens_known'] += computed['input_tokens']
                    target['embedding_padded_tokens_known'] += computed['padded_tokens']
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            evidence.gap(name, exc)
    for session, values in sequences.items():
        if min(values) != 1 or len(values) != max(values):
            evidence.gap('/'.join(session), 'Missing meter sequence(s); sequence is global across kinds')
    for name in evidence.files:
        if '/runtime/' in name and name.endswith(('.request.json', '.response.json')) and name not in expected_files:
            evidence.gap(name, 'Orphan request/response without valid linked receipt')
    operations = {}
    children = defaultdict(Counter)
    attempts = set()
    for name, receipt, request, key, session in records.values():
        operation_id = (*session, receipt.get('operation_id'))
        if receipt['kind'] == 'operation':
            if operation_id in operations:
                evidence.gap(name, 'Duplicate operation ID')
            operations[operation_id] = (name, receipt)
        elif receipt['kind'] in ('generation', 'embedding'):
            children[operation_id][receipt['kind']] += 1
        if receipt['kind'] == 'encode_attempt':
            attempts.add((*session, receipt.get('encode_attempt_id')))
    for name, receipt, request, key, session in records.values():
        if receipt['kind'] != 'operation' and receipt.get('operation_id') is not None:
            if (*session, receipt['operation_id']) not in operations:
                evidence.gap(name, 'Native/attempt receipt has no parent operation')
        if receipt['kind'] == 'embedding' and receipt.get('encode_attempt_id') is not None:
            if (*session, receipt['encode_attempt_id']) not in attempts:
                evidence.gap(name, 'Embedding has no parent encode_attempt')
    for identity, (name, receipt) in operations.items():
        observed = {kind: children[identity][kind] for kind in ('generation', 'embedding')}
        if (receipt.get('actual_invocations') != observed
                or receipt.get('zero_invocations_observed') != (not any(observed.values()))):
            evidence.gap(name, 'Operation native invocation count mismatch')
    return totals, groups, {**operation_summary, 'evaluation_generation_input_coverage':
                            {method: dict(counts) for method, counts in coverage.items()}}


def prediction_summary(evidence: Evidence) -> dict:
    comparison = evidence.load('runs/recursive_v1/OFFICIAL_COMPARISON.json')
    baseline = evidence.load('runs/qwen35_baseline/BASELINE_COMPARISON.json')
    if set(comparison['prediction_hashes']) != set(METHODS):
        raise ValueError('Unexpected official comparison methods')
    protocols = {run: evidence.load(f'runs/{run}/protocol.json') for run in RUNS}
    parent, child = protocols[RUNS[0]], protocols[RUNS[1]]
    if (parent['config']['model'] != MODEL or parent['config']['embed_model'] != EMBED_MODEL
            or parent['config']['seed'] != 20260907 or parent['config']['dataset'] != 'locomo'
            or parent['model']['revision'] != MODEL_REVISION or parent['embedding']['revision'] != EMBED_REVISION
            or parent['dataset_sha256'] != DATA_HASH or child['dataset_sha256'] != DATA_HASH
            or comparison['dataset_sha256'] != DATA_HASH or baseline['dataset_sha256'] != DATA_HASH
            or child['baseline_protocol_sha256'] != digest(parent)
            or child['environment']['models'][MODEL] != parent['model']
            or child['environment']['models'][EMBED_MODEL] != parent['embedding']):
        raise ValueError('Fixed model/dataset/recursive lineage mismatch')
    results, population = {}, None
    for method in METHODS:
        run = RUNS[1] if method == METHODS[-1] else RUNS[0]
        name = f'runs/{run}/{method}.jsonl'
        if name not in evidence.files or name in evidence.bad:
            raise ValueError('Missing validated predictions: ' + name)
        if sha(evidence.root / name) != comparison['prediction_hashes'][method]:
            raise ValueError('Official comparison prediction hash mismatch: ' + method)
        rows = [json.loads(line, object_pairs_hook=unique_object) for line in
                (evidence.root / name).read_text(encoding='utf-8').splitlines() if line.strip()]
        ids = [row['question_id'] for row in rows]
        if len(rows) != N or len(set(ids)) != N or (population is not None and set(ids) != population):
            raise ValueError('Predictions must cover the same1540 unique question IDs')
        population = set(ids)
        if any(row['method'] != method or row['hypothesis'] != row['prediction']
               or not isinstance(row['prediction'], str) or not integer(row.get('input_tokens'))
               or not integer(row.get('output_tokens')) for row in rows):
            raise ValueError('Invalid prediction method, text or logical tokens')
        summary = comparison['methods'][method]
        if summary['n'] != N or not 0 <= summary['official_f1'] <= 1:
            raise ValueError('Invalid official F1 summary')
        if method != METHODS[-1] and (baseline['methods'][method]['n'] != N or
                                     baseline['methods'][method]['official_f1'] != summary['official_f1']):
            raise ValueError('Baseline/recursive official F1 disagreement')
        results[method] = {'run': run, 'questions': N, 'prediction_sha256': comparison['prediction_hashes'][method],
                           'official_f1_100': summary['official_f1'] * 100,
                           'logical_reader_input_tokens': sum(row['input_tokens'] for row in rows),
                           'logical_reader_output_tokens': sum(row['output_tokens'] for row in rows),
                           'empty_predictions': sum(not row['prediction'].strip() for row in rows),
                           'conversation_counts': dict(Counter(row['conversation_id'] for row in rows))}
    for run in RUNS:
        protocol = evidence.load(f'runs/{run}/protocol.json')
        lock = evidence.load(f'runs/{run}/memory_lock.json')
        if lock['protocol_sha256'] != digest(protocol):
            raise ValueError('Memory lock/protocol digest mismatch: ' + run)
    return results


def aggregate(root: Path) -> dict:
    evidence = Evidence(root)
    totals, groups, operations = aggregate_runtime(evidence)
    try:
        methods = prediction_summary(evidence)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        evidence.gap('predictions/protocols', exc)
        methods = {}
    for prefix in ('runs/qwen35_baseline/runtime/prepare/', 'runs/qwen35_baseline/utility/runtime/score/',
                   'runs/qwen35_baseline/runtime/evaluate/', 'runs/recursive_v1/runtime/construct/',
                   'runs/recursive_v1/runtime/evaluate/'):
        if not any(name.startswith(prefix) for name in evidence.files):
            evidence.gap(prefix, 'Required runtime stage evidence absent')
    for method, result in methods.items():
        if not any(key[0] == result['run'] and key[1] == 'evaluate' and key[3] == method for key in groups):
            evidence.gap(method, 'No evaluation runtime observations for method')
    for method, result in methods.items():
        if operations['evaluation_generation_input_coverage'].get(method) != result['conversation_counts']:
            evidence.gap(method, 'Evaluation generation operations do not cover1540 prediction inputs by conversation')
    grouped = []
    for (run, stage, phase, method), values in sorted(groups.items()):
        grouped.append(dict(run=run, stage=stage, phase=phase, method=method,
                            **finish_totals(values, bool(evidence.gaps))))
    for method, result in methods.items():
        values = empty_totals()
        for (run, stage, phase, group_method), group in groups.items():
            if stage == 'evaluate' and group_method == method and run == result['run']:
                for field, value in group.items():
                    values[field] += value
        measured = finish_totals(values, bool(evidence.gaps))
        result['evaluation_native'] = measured
        result['mean_native_inference_seconds_per_question'] = (
            measured['native_inference_seconds_known_lower_bound'] / N if measured['measurement_complete'] else None)
        result['mean_native_inference_seconds_per_question_known_lower_bound'] = (
            measured['native_inference_seconds_known_lower_bound'] / N)
        result['logical_reader_tokens_M'] = (result['logical_reader_input_tokens'] + result['logical_reader_output_tokens']) / 1e6
    return {'schema_version': 1, 'population': 'LoCoMo1540', 'evidence_manifest_sha256': sha(Path(root) / 'evidence_manifest.json'),
            'totals': finish_totals(totals, bool(evidence.gaps)), 'methods': methods,
            'stage_phase_method_groups': grouped, 'operations': operations, 'gaps': evidence.gaps,
            'interpretation': [
                'M means tokens / 1,000,000. LLM consumption is submitted native prompt plus generated tokens; it is not uncached GPU-prefill work.',
                'Embedding tokens are actual post-256 attention-mask nonpadding tokens, including failed forwards. Padding is separate.',
                'Every meter invocation is counted once. Retries with different invocation identities remain distinct cost even if inputs match.',
                'Only native generation/embedding inference durations are added; operation/encode_attempt durations are nested and excluded.',
                'Mean seconds/question = native evaluation LLM plus embedding call seconds /1540. This is amortized inference, not E2E latency.',
                'Per-row generation_batch_seconds repeats conversation-batch timing and is never averaged or summed.',
                'Baseline prepare is a shared execution pool; utility score is ancestor work for Refined and recursive descendants, not a Seed/r40 standalone requirement. Recursive construction is additional. No measured per-method build allocation is claimed.',
                'Evaluation amounts are measured incremental execution costs under the observed cache lifecycle and arm order, not cold-cache algorithm efficiency.',
                'Runtime phases are recorded context labels; recursive_round_complete can persist into subsequent work and does not define exact round costs.',
                'Logical prediction tokens can include cached results; they are separate from actual native calls. Zero-invocation operations do not prove cache hits.',
                'Official F1 is imported from the comparison bound to exact prediction hashes; this report does not rerun the scorer.',
                'Manifest verification establishes collected-file integrity. Memory payload and model weight integrity are not proven by this report.']}


def markdown(report: dict) -> str:
    lines = ['# Refinement usage (Qwen3.5-9B + MiniLM)', '',
             '| Method | F1 (0–100) | Native evaluation LLM (M) | Native evaluation embedding (M) | Native inference seconds/question | Logical reader tokens (M) |',
             '|---|---:|---:|---:|---:|---:|']
    def number(value):
        return 'unknown' if value is None else f'{value:.6f}'
    for method, row in report['methods'].items():
        values = [row['official_f1_100'], row['evaluation_native']['llm_tokens_M'],
                  row['evaluation_native']['embedding_tokens_M'], row['mean_native_inference_seconds_per_question'],
                  row['logical_reader_tokens_M']]
        lines.append('| ' + method + ' | ' + ' | '.join(number(v) for v in values) + ' |')
    lines += ['', '| Run / stage / phase / ownership | LLM known tokens (M) | Embedding known tokens (M) | Native seconds known | Complete |',
              '|---|---:|---:|---:|---|']
    for row in report['stage_phase_method_groups']:
        label = ' / '.join(row[field] for field in ('run', 'stage', 'phase', 'method'))
        lines.append('| ' + label + ' | ' + ' | '.join(number(row[field]) for field in
                     ('llm_tokens_M_known_lower_bound', 'embedding_tokens_M_known_lower_bound',
                      'native_inference_seconds_known_lower_bound')) + f" | {row['measurement_complete']} |")
    lines += ['', *['- ' + note for note in report['interpretation']], '',
              'Gaps: ' + str(len(report['gaps'])) + '. Known amounts are lower bounds whenever measurement is incomplete.']
    lines += ['- ' + item['path'] + ': ' + item['reason'] for item in report['gaps']]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_root', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='New output directory outside the evidence tree')
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.evidence_root.resolve()):
        parser.error('Output must be outside the read-only evidence tree')
    report = aggregate(args.evidence_root)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'USAGE_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (args.out / 'USAGE_REPORT.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps({'output': str(args.out), 'gaps': len(report['gaps']), 'complete': report['totals']['measurement_complete']}))


if __name__ == '__main__':
    main()
