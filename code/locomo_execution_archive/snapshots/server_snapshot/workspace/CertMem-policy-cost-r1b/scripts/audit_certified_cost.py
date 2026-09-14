"""Offline, trace-derived full/certified dependency cost; never a standalone rerun.

The fixed synchronous v15 recipe gives each nonempty full index two searches,
then an explicit query encoding, followed by nine other-policy encodings per QA.
All observed prefix forwards are shared index/sentence construction dependencies.
No model imports, rescoring, source mutations, or guessed encoder batch sizes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import ModuleType

MANIFEST_SHA256 = 'fd0aac2763441be3efcac3f88800d7c5d6251df8543346502eb7fcffebd9e944'
BASE_SHA256 = '5f79f9caf4a5eb35beda61d90350ba8a787da8b675067df1eb9a51e3056c335a'
QA_TOTAL, CONV_TOTAL = 1540, 10
TOKEN_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_base(path: Path) -> ModuleType:
    """Execute only the reviewed stdlib-only auditor bytes, without pycache writes."""
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == BASE_SHA256, 'Frozen base auditor hash mismatch')
    module = ModuleType('frozen_certmem_runtime_auditor')
    exec(compile(source, str(path), 'exec'), module.__dict__)  # noqa: S102 - exact reviewed SHA is checked above
    return module


def native_counts(base: ModuleType, manifest: dict, output: Path) -> dict:
    """Prove both indexes nonempty from the fixed loader and session write records."""
    samples = json.loads(Path(manifest['dataset']).read_text(encoding='utf-8'))
    require(isinstance(samples, list) and len(samples) == CONV_TOTAL, 'Original conversation scope mismatch')
    counts, sessions = {}, set()
    for sample in samples:
        require(isinstance(sample, dict), 'Dataset sample must be an object')
        cid, conversation, qas = sample['sample_id'], sample['conversation'], sample['qa']
        require(isinstance(cid, str) and cid and cid not in counts and isinstance(conversation, dict)
                and isinstance(qas, list) and all(isinstance(q, dict) for q in qas), 'Malformed dataset identity')
        selected = [q for q in qas if q.get('category') != 5]
        require(selected and all(type(q.get('category')) is int and q['category'] in (1, 2, 3, 4) for q in selected), 'Invalid original QA')
        counts[cid] = len(selected)
        number, raw_turn_count = 1, 0
        while f'session_{number}' in conversation:
            turns = conversation[f'session_{number}']
            require(isinstance(turns, list) and all(isinstance(t, dict) and isinstance(t.get('speaker'), str) for t in turns), 'Malformed native turns')
            raw_turn_count += len(turns)
            sessions.add((cid, number))
            number += 1
        require(raw_turn_count > 0, 'Raw-turn index is empty/unproven')
    require(sum(counts.values()) == QA_TOTAL, 'Original 1540 QA scope mismatch')
    seen, memory = set(), dict.fromkeys(counts, 0)
    with (output / 'write_stats.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            cid, config, session = row['conv_id'], row['config'], int(row['session'])
            require(config in ('full', 'no_adaptive', 'no_residual') and (cid, session) in sessions, 'Foreign write_stats identity')
            key = (cid, config, session)
            require(key not in seen, 'Duplicate write_stats session')
            seen.add(key)
            mem, raw = int(row['mem_tok']), int(row['raw_tok'])
            require(mem >= 0 and raw >= 0, 'Invalid write_stats token counts')
            if config == 'full':
                memory[cid] += mem
    expected = {(cid, config, session) for cid, session in sessions for config in ('full', 'no_adaptive', 'no_residual')}
    require(seen == expected and all(n > 0 for n in memory.values()), 'Full memory index/session evidence missing or empty')
    return counts


def select_embeddings(base: ModuleType, output: Path, report: dict, counts: dict) -> dict:
    """Select existing forward identities, preserving and reconciling all exclusions."""
    grouped = {cid: [] for cid in counts}
    ledger = [r for r in report['embedding_ledger'] if r['bucket'] == 'retrieval/full']
    sequence_seen = set()
    for row in ledger:
        seq, context = row['batch_sequence'], row['context']
        require(base.integer(seq) and seq > 0 and seq not in sequence_seen, 'Duplicate/invalid full forward sequence')
        sequence_seen.add(seq)
        require(isinstance(context, dict) and context == {'phase': 'retrieval', 'config': 'full', 'conv_id': context.get('conv_id')}
                and context['conv_id'] in counts, 'Foreign full forward context')
        request_path = output / f'runtime/embedding/{seq:08d}.request.json'
        receipt_path = output / f'runtime/embedding/{seq:08d}.receipt.json'
        request, receipt = base.read_object(request_path), base.read_object(receipt_path)
        request_hash = base.digest(request_path)
        require(request_hash == receipt.get('request_hash') == receipt.get('request_sha256') == row['request_sha256'], 'Embedding request SHA mismatch')
        require(request.get('context') == context and receipt.get('status') == 'success'
                and receipt.get('usage_complete') is True, 'Embedding context/status/usage mismatch')
        parsed, errors = base.embedding_row(request, receipt, seq, {(cid, 0): {} for cid in counts})
        require(not errors, 'Invalid embedding request/receipt: ' + '; '.join(errors))
        require(all(base.integer(receipt.get(k)) for k in ('input_tokens', 'padded_tokens'))
                and all(isinstance(receipt.get(k), list) and all(base.integer(n) for n in receipt[k])
                        for k in ('tokens_per_input', 'shape')), 'Receipt encoder counts must be integers, not booleans')
        require(all(n > 0 for n in request['tokens_per_input']), 'Nonpositive actual encoder mask counts')
        require(all(parsed[k] == row[k] for k in ('input_tokens', 'padded_tokens', 'status')), 'Base/full embedding ledger mismatch')
        grouped[context['conv_id']].append({'batch_sequence': seq, 'request_sha256': request_hash,
            'receipt_sha256': base.digest(receipt_path), 'context': context,
            **{k: request[k] for k in ('input_tokens', 'padded_tokens', 'tokens_per_input', 'shape')}})
    chosen, excluded, boundaries = [], [], {}
    for cid, rows in grouped.items():
        rows.sort(key=lambda r: r['batch_sequence'])
        ids = [r['batch_sequence'] for r in rows]
        prefix = len(rows) - 12 * counts[cid]
        require(prefix >= 2 and ids == list(range(ids[0], ids[-1] + 1)), 'Missing/noncontiguous full prefix or QA forwards')
        chosen.extend(rows[:prefix])
        for offset in range(prefix, len(rows), 12):
            group = rows[offset:offset + 12]
            require(len(group) == 12 and all(r['shape'][0] == 1 for r in group), 'QA suffix forwards must be singleton groups of 12')
            require(all(all(r[k] == group[0][k] for k in ('input_tokens', 'padded_tokens', 'tokens_per_input', 'shape')) for r in group), 'QA repeated-query mask counts differ')
            chosen.extend(group[:3])
            excluded.extend(group[3:])
        boundaries[cid] = {'qa_count': counts[cid], 'prefix_forward_count': prefix,
                           'first_batch_sequence': ids[0], 'qa_suffix_first_batch_sequence': ids[prefix], 'last_batch_sequence': ids[-1]}
    def totals(rows: list) -> dict:
        return {'forward_count': len(rows), 'input_tokens': sum(r['input_tokens'] for r in rows),
                'padded_tokens': sum(r['padded_tokens'] for r in rows)}
    selected_totals, excluded_totals = totals(chosen), totals(excluded)
    actual = report['by_phase_config']['retrieval/full']['embedding']
    require(selected_totals['forward_count'] + excluded_totals['forward_count'] == actual['forward_count']
            and selected_totals['input_tokens'] + excluded_totals['input_tokens'] == actual['exact_input_tokens']
            and selected_totals['padded_tokens'] + excluded_totals['padded_tokens'] == actual['recorded_padded_tokens'], 'Full embedding totals do not reconcile')
    return {'exact_input_tokens': selected_totals['input_tokens'], 'selected_totals': selected_totals,
            'excluded_totals': excluded_totals, 'selected': chosen, 'excluded': excluded, 'boundaries': boundaries,
            'primary_measure': 'Actual native post-256 unpadded attention-mask tokens; padded work is separate.'}


def generation_tokens(summary: dict) -> dict:
    tokens = summary['generation']['exact_tokens']
    require(isinstance(tokens, dict) and all(type(tokens.get(k)) is int and tokens[k] >= 0 for k in TOKEN_KEYS)
            and tokens['total_tokens'] == tokens['prompt_tokens'] + tokens['completion_tokens'], 'Missing/invalid exact generation usage')
    return {k: tokens[k] for k in TOKEN_KEYS}


def audit_certified_cost(output: Path, manifest_path: Path, expected_manifest_sha256: str,
                         closure_path: Path | None, expected_closure_sha256: str | None,
                         base_auditor_path: Path) -> dict:
    """Call the pinned base audit, then apply only the fixed original full recipe."""
    result = {'schema_version': 1, 'complete': False, 'issues': [],
        'scope': 'Observed shared-dependency token sum for original full/certified; trace-derived and NONADDITIVE across policies.',
        'standalone_actual_run': False, 'additive_across_policies': False, 'minimal_causal_cost': False,
        'exclusive_gpu_energy_wh': None, 'exclusive_rental_usd': None,
        'certified': {'generation': {'exact_tokens': None}, 'embedding': {'exact_input_tokens': None}},
        'naive': {'generation': {'exact_tokens': None}, 'causal_embedding_tokens': None, 'causal_construction_tokens': None},
        'limitations': ['Input text/callsite is not logged for embeddings: attribution is derived from frozen synchronous source order, not independent text proof.',
                       'All initial index/sentence forwards are shared dependencies, not a minimal standalone execution.',
                       'Shared fact extraction is counted once; generation length/known failures and cached tokens are not discounted.',
                       'The 37-policy three-config gross pipeline is not one baseline; GPU energy and rental cannot be allocated exclusively.']}
    try:
        base = load_base(base_auditor_path)
        require(expected_manifest_sha256 == MANIFEST_SHA256 and base.digest(manifest_path) == MANIFEST_SHA256, 'Original fixed manifest hash mismatch')
        report = base.audit_runtime(output, manifest_path, expected_manifest_sha256, closure_path, expected_closure_sha256)
        result['base_complete'] = report['complete']
        require(all(report.get(k) is True for k in ('complete', 'closed_sources_complete', 'usage_complete')), 'Closed complete base runtime audit required')
        require(report['qa_coverage'] == {'complete': True, 'count': QA_TOTAL * 37, 'expected': QA_TOTAL * 37}, 'Complete original 1540 x 37 QA matrix required')
        manifest = base.read_object(manifest_path)
        require(closure_path is not None and expected_closure_sha256 is not None
                and base.digest(closure_path) == expected_closure_sha256, 'Pinned closure required')
        closure = base.read_object(closure_path)
        before = base.inventory(output)
        require(closure.get('raw_source_sha256') == before, 'Closed snapshot changed before selection')
        source = Path(manifest['source_root'])
        command = [manifest['python'], '-u', str(source / 'scripts/run_system_v15.py'), '--comparison', '--dataset', 'locomo',
                   '--configs', 'full,no_adaptive,no_residual', '--budgets', '400,1600,4000', '--out', str(output)]
        require(base.read_object(output / 'receipt.json').get('command') == command, 'Native recipe CLI differs')
        counts = native_counts(base, manifest, output)
        embedding = select_embeddings(base, output, report, counts)
        components = {key: generation_tokens(report['by_phase_config'][key])
                      for key in ('fact_extraction/shared', 'memory_build/full', 'query_reform/full')}
        for policy in ('full/certified', 'full/full_raw'):
            require(report['qa_by_policy'][policy]['response_count'] == QA_TOTAL, 'Policy QA count mismatch')
        components['qa/full/certified'] = generation_tokens(report['qa_by_policy']['full/certified'])
        tokens = {k: sum(value[k] for value in components.values()) for k in TOKEN_KEYS}
        naive = generation_tokens(report['qa_by_policy']['full/full_raw'])
        require(base.inventory(output) == before and base.digest(closure_path) == expected_closure_sha256, 'Snapshot/closure changed during selection')
        base.source_proof(manifest, manifest_path, expected_manifest_sha256, output)
        result.update(complete=True, proof={'manifest_sha256': expected_manifest_sha256,
            'base_auditor_sha256': BASE_SHA256, 'closure_sha256': expected_closure_sha256,
            'inventory_unchanged': True, 'recipe': 'All prefix + first 3 of each native 12-forward full QA suffix'},
            certified={'generation': {'exact_tokens': tokens, 'components': components}, 'embedding': embedding},
            naive={'generation': {'exact_tokens': naive}, 'causal_embedding_tokens': 0, 'causal_construction_tokens': 0})
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        result['issues'].append(str(exc))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-output', 'manifest', 'base-auditor', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--closure', type=Path)
    parser.add_argument('--expected-closure-sha256')
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'Audit output already exists')
    require(not args.output.resolve().is_relative_to(args.runtime_output.resolve()), 'Audit output must be outside snapshot')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    require(not args.output.resolve().is_relative_to(Path(manifest['source_root']).resolve()), 'Audit output must be outside pinned source')
    result = audit_certified_cost(args.runtime_output, args.manifest, args.expected_manifest_sha256,
                                 args.closure, args.expected_closure_sha256, args.base_auditor)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
