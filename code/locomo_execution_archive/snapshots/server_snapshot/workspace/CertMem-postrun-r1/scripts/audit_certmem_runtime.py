"""Offline audit of closed v15 native artifacts; no model imports or source writes.

Every batch is counted once. Raw output token IDs (including length finishes and
failed known outputs) are the generation evidence; actual post-256 attention-mask
counts are encoder evidence. Missing proof/measurements never establish zero cost.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

RUN_ID = 'locomo-certmem-v15-20260908-r1'
MODEL = 'Qwen/Qwen3.5-9B'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBED_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
DATASET_SHA256 = 'cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993'
QA_TOTAL, CONV_TOTAL = 1540, 10
BASE = ('units_only', 'certified') + tuple(f'{k}_{b}' for b in (400, 1600, 4000) for k in ('uniform', 'ours', 'certified'))
POLICIES = {'full': BASE + ('raw_rag_400', 'raw_rag_1600', 'raw_rag_4000', 'full_raw'),
            'no_adaptive': BASE, 'no_residual': BASE}
REQUIRED_SOURCE = ('certmem/runtime.py', 'certmem/llm.py', 'certmem/config.py', 'certmem/locomo.py',
                   'scripts/run_system_v15.py', 'scripts/run_certmem_full.py', 'README.md', 'pyproject.toml')
CAPS = {'fact_extraction': (700,), 'memory_build': (4, 120, 512), 'query_reform': (60,), 'qa': (32,)}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), f'Expected object: {path}')
    return value


def integer(value: object) -> bool:
    return type(value) is int and value >= 0


def token_count(ids: object) -> int | None:
    return len(ids) if isinstance(ids, list) and all(integer(i) for i in ids) else None


def inventory(output: Path) -> dict:
    result = {}
    for path in output.rglob('*'):
        require(not path.is_symlink(), f'Symlink in snapshot: {path}')
        if path.is_file():
            result[path.relative_to(output).as_posix()] = digest(path)
    return result


def source_proof(manifest: dict, manifest_path: Path, expected_sha: str, output: Path) -> dict:
    require(digest(manifest_path) == expected_sha, 'Manifest hash mismatch')
    require(manifest.get('run_id') == RUN_ID and manifest.get('output') == str(output), 'Manifest identity mismatch')
    source, dataset = Path(manifest['source_root']), Path(manifest['dataset'])
    require(source.is_absolute() and dataset == source / 'data/locomo10.json', 'Dataset source path mismatch')
    pins = manifest['files_sha256']
    require(isinstance(pins, dict) and bool(pins), 'Missing source pins')
    required = {str(source / name) for name in REQUIRED_SOURCE} | {str(dataset)}
    required.update(str(p) for p in source.rglob('*.py') if 'tests' not in p.relative_to(source).parts)
    require(required.issubset(pins), 'Unpinned execution source')
    for name, value in pins.items():
        path = Path(name)
        require(path.is_absolute() and not path.is_symlink() and digest(path) == value, f'Source pin mismatch: {name}')
    if 'token_preflight' in manifest:
        pin = manifest['token_preflight']
        require(digest(Path(pin['path'])) == pin['sha256'], 'Token preflight pin mismatch')
    require(digest(dataset) == DATASET_SHA256, 'Original dataset hash mismatch')
    return read_object(output / 'receipt.json')


def context_bucket(context: dict, kind: str, questions: dict) -> str:
    require(isinstance(context, dict), 'Native context must be an object')
    phase, config, cid = context.get('phase'), context.get('config'), context.get('conv_id')
    require(isinstance(phase, str) and isinstance(cid, str) and (config is None or isinstance(config, str)),
            'Native attribution fields must be strings')
    require(cid in {key[0] for key in questions}, 'Foreign conversation attribution')
    if phase == 'fact_extraction' and kind == 'generation':
        require(config is None and 'items' not in context, 'Shared fact extraction must not be allocated to a config')
        return 'fact_extraction/shared'
    require(config in POLICIES and phase in (CAPS if kind == 'generation' else ('retrieval',)), 'Foreign phase/config attribution')
    return f'{phase}/{config}'


def generation_rows(request: dict, response: dict, receipt: dict, sequence: int,
                    paths: dict, questions: dict, identities: set) -> tuple[list, list]:
    rows, issues = [], []
    context = request.get('context') if isinstance(request.get('context'), dict) else {}
    prompts = request.get('prompts') if isinstance(request.get('prompts'), list) else []
    raw = response.get('responses') if isinstance(response.get('responses'), list) else []
    try:
        bucket = context_bucket(context, 'generation', questions)
    except (ValueError, TypeError) as exc:
        bucket = 'unattributed'
        issues.append(str(exc))
    items = context.get('items') if isinstance(context.get('items'), list) else []
    request_sha, response_sha = None, None
    try:
        request_sha = digest(paths['request']) if paths['request'].is_file() else None
        response_sha = digest(paths['response']) if paths['response'].is_file() else None
    except OSError as exc:
        issues.append(f'Native artifact changed or became unreadable: {exc}')
    finishes, candidate_count = [], 0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            issues.append('Non-object native response')
            continue
        outputs = item.get('outputs') if isinstance(item.get('outputs'), list) else []
        prompt_tokens = token_count(item.get('prompt_token_ids'))
        counts = [token_count(candidate.get('token_ids')) if isinstance(candidate, dict) else None for candidate in outputs]
        completion_tokens = sum(counts) if counts and all(n is not None for n in counts) else None
        reasons = [candidate.get('finish_reason') if isinstance(candidate, dict) else None for candidate in outputs]
        candidate_count += len(outputs)
        finishes.extend(reasons)
        rid, cached = item.get('request_id'), item.get('num_cached_tokens')
        entry = {'batch_sequence': sequence, 'response_index': index, 'request_id': rid,
            'bucket': bucket, 'context': {k: v for k, v in context.items() if k != 'items'},
            'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
            'known_completion_tokens': sum(n for n in counts if n is not None),
            'num_cached_tokens': cached, 'finish_reasons': reasons, 'generation_count': len(outputs),
            'request_sha256': request_sha, 'response_sha256': response_sha,
            'status': receipt.get('status'), 'qa_key': None, 'native_text': None}
        if (not isinstance(rid, str) or not rid or rid in identities or index >= len(prompts)
                or item.get('prompt') != (prompts[index] if index < len(prompts) else None)
                or item.get('finished') is not True or len(outputs) != 1
                or any(not isinstance(c, dict) or type(c.get('index')) is not int
                       or c['index'] != 0 or not isinstance(c.get('text'), str) for c in outputs)
                or any(reason not in ('stop', 'length') for reason in reasons)
                or prompt_tokens is None or completion_tokens is None):
            issues.append(f'Invalid native response identity/cardinality/tokens at {sequence}:{index}')
        if isinstance(rid, str):
            identities.add(rid)
        if cached is not None and (not integer(cached) or prompt_tokens is None or cached > prompt_tokens):
            issues.append('Invalid cached prompt token count')
        if context.get('phase') == 'qa':
            metadata = items[index] if index < len(items) and isinstance(items[index], dict) else {}
            ordinal, policy = metadata.get('question_ordinal'), metadata.get('policy')
            if (not integer(ordinal) or not all(isinstance(value, str) for value in
                    (context.get('conv_id'), context.get('config'), policy))
                    or (context.get('conv_id'), ordinal) not in questions
                    or policy not in POLICIES.get(context.get('config'), ())):
                issues.append('Invalid QA item attribution')
            else:
                entry['qa_key'] = (context['conv_id'], ordinal, context['config'], policy)
                entry['native_text'] = outputs[0].get('text') if outputs and isinstance(outputs[0], dict) else None
        rows.append(entry)
    try:
        require(receipt.get('type') == 'generation_batch' and receipt.get('batch_sequence') == sequence, 'Generation sequence/type mismatch')
        require(receipt.get('context') == context, 'Request/receipt context mismatch')
        require(request.get('model') == MODEL and request.get('model_revision') == REVISION, 'Generation model mismatch')
        sampling = request['sampling']
        require(isinstance(sampling, dict) and type(sampling.get('temperature')) in (int, float)
                and sampling['temperature'] == 0 and type(sampling.get('n')) is int and sampling['n'] == 1
                and sampling.get('max_tokens') in CAPS.get(context.get('phase'), ())
                and sampling.get('truncate_prompt_tokens') is None, 'Native sampling mismatch')
        require(all(isinstance(p, str) for p in prompts) and len(raw) == len(prompts), 'Request/response cardinality mismatch')
        require(context.get('phase') != 'qa' or len(items) == len(prompts), 'QA item cardinality mismatch')
        require(all(integer(receipt.get(k)) for k in ('request_count', 'response_count', 'generation_count'))
                and receipt.get('request_count') == len(prompts) and receipt.get('response_count') == len(raw)
                and receipt.get('generation_count') == candidate_count and receipt.get('finish_reasons') == finishes,
                'Native receipt counts mismatch')
        prompt_counts = [row['prompt_tokens'] for row in rows]
        completion_counts = [row['completion_tokens'] for row in rows]
        require(all(n is not None for n in prompt_counts + completion_counts), 'Actual token IDs missing')
        require(receipt.get('preflight_prompt_tokens') == prompt_counts, 'Preflight/actual prompt tokens mismatch')
        totals = {'prompt_tokens': sum(prompt_counts), 'completion_tokens': sum(completion_counts)}
        totals['total_tokens'] = sum(totals.values())
        require(all(integer(receipt.get(k)) and receipt[k] == n for k, n in totals.items()), 'Receipt token totals mismatch')
    except (ValueError, TypeError, KeyError) as exc:
        issues.append(str(exc))
    return rows, issues


def embedding_row(request: dict, receipt: dict, sequence: int, questions: dict) -> tuple[dict, list]:
    issues, counts = [], ('input_tokens', 'padded_tokens', 'tokens_per_input', 'shape')
    row = {'batch_sequence': sequence, 'bucket': 'unattributed', 'context': request.get('context'),
           'input_tokens': None, 'padded_tokens': None, 'status': receipt.get('status')}
    try:
        shape, per = request['shape'], request['tokens_per_input']
        require(isinstance(shape, list) and len(shape) == 2 and all(integer(n) and n > 0 for n in shape)
                and shape[1] <= 256 and isinstance(per, list) and len(per) == shape[0]
                and all(integer(n) and n <= shape[1] for n in per), 'Invalid native encoder mask shape')
        require(integer(request['input_tokens']) and request['input_tokens'] == sum(per)
                and integer(request['padded_tokens']) and request['padded_tokens'] == shape[0] * shape[1], 'Invalid native encoder token counts')
        row.update(input_tokens=request['input_tokens'], padded_tokens=request['padded_tokens'])
        row['bucket'] = context_bucket(request['context'], 'embedding', questions)
        require(request.get('model') == EMBED_MODEL and request.get('model_revision') == EMBED_REVISION
                and request.get('native_max_seq_length') == 256, 'Embedding model/window mismatch')
        require(all(receipt.get(k) == request[k] for k in counts), 'Embedding request/receipt counts mismatch')
        require(receipt.get('type') == 'embedding_forward' and receipt.get('batch_sequence') == sequence
                and receipt.get('context') == request.get('context'), 'Embedding receipt identity mismatch')
    except (ValueError, TypeError, KeyError) as exc:
        issues.append(str(exc))
    return row, issues


def summarize(generations: list, embeddings: list, complete: bool) -> dict:
    prompt = sum(r['prompt_tokens'] for r in generations if r['prompt_tokens'] is not None)
    completion = sum(r['known_completion_tokens'] for r in generations)
    totals = {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion}
    cached = [r['num_cached_tokens'] for r in generations]
    known_cached = sum(n for n in cached if integer(n))
    inputs = sum(r['input_tokens'] for r in embeddings if r['input_tokens'] is not None)
    return {'generation': {'reported_tokens': totals, 'exact_tokens': totals.copy() if complete else None,
        'response_batch_count': len({r['batch_sequence'] for r in generations}),
        'response_count': len(generations), 'generation_count': sum(r['generation_count'] for r in generations),
        'length_responses': sum(reason == 'length' for r in generations for reason in r['finish_reasons']),
        'cached_prompt_tokens': {'recorded': known_cached, 'exact': known_cached if complete and all(integer(n) for n in cached) else None},
        'cache_discount_applied': False},
        'embedding': {'forward_count': len(embeddings), 'recorded_input_tokens': inputs,
            'exact_input_tokens': inputs if complete else None,
            'recorded_padded_tokens': sum(r['padded_tokens'] for r in embeddings if r['padded_tokens'] is not None),
            'primary_measure': 'Actual native post-256 unpadded attention_mask tokens, including failed known forwards.'}}


def audit_runtime(output: Path, manifest_path: Path, expected_manifest_sha256: str,
                  closure_path: Path | None = None, expected_closure_sha256: str | None = None) -> dict:
    """Read-only accounting; absent closure leaves recorded subtotals, never exact totals."""
    issues, proof_issues, generations, embeddings, identities = [], [], [], [], set()
    questions, launcher, manifest, before, attempts = {}, {}, {}, {}, {}
    try:
        before = inventory(output)
        manifest = read_object(manifest_path)
        samples = json.loads(Path(manifest['dataset']).read_text(encoding='utf-8'))
        seen = set()
        for sample in samples:
            cid = sample['sample_id']
            require(isinstance(cid, str) and cid not in seen, 'Duplicate dataset conversation')
            seen.add(cid)
            require(isinstance(sample['qa'], list) and all(isinstance(q, dict) for q in sample['qa']),
                    'Dataset QA entries must be objects')
            qas = [q for q in sample['qa'] if q.get('category') != 5]
            require(all(type(q.get('category')) is int and q['category'] in (1, 2, 3, 4) for q in qas), 'Invalid QA category')
            questions.update({(cid, i): q for i, q in enumerate(qas)})
        require(len(seen) == CONV_TOTAL and len(questions) == QA_TOTAL, 'Dataset QA scope mismatch')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        proof_issues.append(str(exc))
    try:
        launcher = source_proof(manifest, manifest_path, expected_manifest_sha256, output)
        require(launcher.get('run_id') == RUN_ID and launcher.get('manifest_sha256') == expected_manifest_sha256
                and launcher.get('child_exit_code') == 0 and launcher.get('source_unchanged') is True
                and launcher.get('inference_complete') is True, 'Launcher completion/pins mismatch')
        require(closure_path is not None and expected_closure_sha256 is not None
                and digest(closure_path) == expected_closure_sha256, 'Missing or unpinned detached closure')
        closure = read_object(closure_path)
        require(closure.get('schema_version') == 1 and closure.get('run_id') == RUN_ID
                and closure.get('output') == str(output) and closure.get('supervisor_state') == 'EXITED'
                and closure.get('writer_pids') == [] and closure.get('child_exit_code') == 0,
                'Run closure/writer proof incomplete')
        require(closure.get('raw_source_sha256') == before, 'Closed output inventory mismatch')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        proof_issues.append(str(exc))
    for kind in ('generation', 'embedding'):
        folder, groups = output / 'runtime' / kind, defaultdict(dict)
        if not folder.is_dir():
            issues.append(f'Missing {kind} journal directory')
        for path in folder.iterdir() if folder.is_dir() else ():
            match = re.fullmatch(r'(\d{8})\.(request|response|receipt)\.json', path.name)
            if not match or not path.is_file():
                issues.append(f'Unclosed/unexpected runtime artifact: {path.name}')
                continue
            groups[int(match[1])][match[2]] = path
        if sorted(groups) != list(range(1, len(groups) + 1)) or not groups:
            issues.append(f'Non-contiguous/empty {kind} sequence')
        attempts[kind] = len(groups)
        for sequence, supplied in sorted(groups.items()):
            paths = {suffix: folder / f'{sequence:08d}.{suffix}.json' for suffix in ('request', 'response', 'receipt')}
            required = {'request', 'receipt', 'response'} if kind == 'generation' else {'request', 'receipt'}
            if set(supplied) != required:
                issues.append(f'Missing or foreign files in {kind}:{sequence}')
            objects = {}
            for suffix in required:
                try:
                    objects[suffix] = read_object(paths[suffix])
                except (OSError, ValueError) as exc:
                    objects[suffix] = {}
                    issues.append(str(exc))
            request, receipt = objects.get('request', {}), objects.get('receipt', {})
            try:
                require(receipt.get('request_hash') == receipt.get('request_sha256') == digest(paths['request']), 'Native request hash mismatch')
                if kind == 'generation':
                    require(receipt.get('response_sha256') == digest(paths['response']), 'Native response hash mismatch')
                require(receipt.get('status') == 'success' and receipt.get('usage_complete') is True, 'Failed/incomplete native call')
                duration = receipt.get('duration_s')
                require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0, 'Invalid native duration')
                if kind == 'generation':
                    inference = receipt.get('inference_duration_s')
                    require(type(inference) in (int, float) and math.isfinite(inference)
                            and 0 <= inference <= duration, 'Invalid actual inference duration')
            except (OSError, ValueError, TypeError) as exc:
                issues.append(f'{kind}:{sequence}: {exc}')
            if kind == 'generation':
                rows, errors = generation_rows(request, objects.get('response', {}), receipt, sequence, paths, questions, identities)
                generations.extend(rows)
            else:
                row, errors = embedding_row(request, receipt, sequence, questions)
                row['request_sha256'] = receipt.get('request_sha256')
                embeddings.append(row)
            issues.extend(f'{kind}:{sequence}: {error}' for error in errors)
    expected = {(cid, index, config, policy) for cid, index in questions for config, policies in POLICIES.items() for policy in policies}
    qa_rows = defaultdict(list)
    for row in generations:
        if row['qa_key'] is not None:
            qa_rows[row['qa_key']].append(row)
    qa_complete = set(qa_rows) == expected and all(len(rows) == 1 for rows in qa_rows.values())
    if not qa_complete:
        issues.append('QA response matrix missing/duplicate/foreign')
    csv_seen = set()
    try:
        with (output / 'items.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                key = (row['conv_id'], int(row['question_ordinal']), row['config'], row['policy'])
                require(key in expected and key not in csv_seen and len(qa_rows[key]) == 1, 'Invalid native CSV identity/refcount')
                qa, raw_text = questions[key[:2]], qa_rows[key][0]['native_text']
                require(isinstance(raw_text, str) and row['pred'] == raw_text.strip()
                        and row['question'] == qa['question'] and row['gold'] == str(qa.get('answer', ''))
                        and row['category'] == str(qa['category']), 'CSV/native response metadata mismatch')
                csv_seen.add(key)
        require(csv_seen == expected, 'Incomplete native CSV matrix')
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        qa_complete = False
        issues.append(str(exc))
    try:
        require(before == inventory(output), 'Output changed during audit')
        source_proof(manifest, manifest_path, expected_manifest_sha256, output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        proof_issues.append(str(exc))
    complete = not issues and not proof_issues and bool(generations) and bool(embeddings)
    groups = {key: summarize([r for r in generations if r['bucket'] == key], [r for r in embeddings if r['bucket'] == key], complete)
              for key in sorted({r['bucket'] for r in generations + embeddings})}
    policies = {}
    for config, names in POLICIES.items():
        for policy in names:
            rows = [row for key, entries in qa_rows.items() if key[2:] == (config, policy) for row in entries]
            policies[f'{config}/{policy}'] = {**summarize(rows, [], complete and len(rows) == QA_TOTAL),
                'response_count': len(rows), 'exclusive_gpu_energy_wh': None, 'exclusive_rental_usd': None,
                'causal_embedding_tokens': 0 if policy == 'full_raw' else None,
                'causal_construction_tokens': 0 if policy == 'full_raw' else None,
                'scope': 'Actual reader requests only; shared construction/retrieval not proportionally allocated.'}
    totals = summarize(generations, embeddings, complete)
    totals['generation'].update(recorded_batch_attempt_count=attempts['generation'],
                                native_generate_batch_count=attempts['generation'] if complete else None)
    return {'schema_version': 1, 'run_id': RUN_ID, 'complete': complete,
        'complete_scope': 'Closed native runtime token accounting and output attribution, not official scoring.',
        'benchmark_complete': False, 'closed_sources_complete': not proof_issues,
        'usage_complete': complete, 'issues': proof_issues + issues, **totals,
        'by_phase_config': groups, 'qa_by_policy': policies,
        'qa_coverage': {'complete': qa_complete, 'count': sum(map(len, qa_rows.values())), 'expected': QA_TOTAL * 37},
        'generation_ledger': [{k: v for k, v in row.items() if k != 'native_text'} for row in generations],
        'embedding_ledger': embeddings,
        'runtime_window_observations': launcher.get('runtime'),
        'exclusive_gpu_energy_wh': None, 'exclusive_rental_usd': None,
        'notes': ['Gross is the complete three-config, 37-reader-policy experiment, not one baseline cost.',
                  'Native in-process vLLM batches, individual responses and encoder forwards are distinct counts; none are HTTP API calls.',
                  'Shared fact extraction is counted once in fact_extraction/shared; cached tokens are not discounted.',
                  'Reported tokens are recorded subtotals, not exact totals if evidence is missing or invalid.',
                  'Native length and failed known output IDs remain counted; semantic discard quantities are not measured.',
                  'Naive full_raw has no causal memory construction or embedding; its shared device/rental costs cannot be allocated exclusively.']}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--closure', type=Path)
    parser.add_argument('--expected-closure-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    require(not args.output.resolve().is_relative_to(args.runtime_output.resolve()), 'Audit output must be outside original snapshot')
    source = Path(read_object(args.manifest)['source_root']).resolve()
    require(not args.output.resolve().is_relative_to(source), 'Audit output must be outside pinned source')
    result = audit_runtime(args.runtime_output, args.manifest, args.expected_manifest_sha256,
                           args.closure, args.expected_closure_sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
