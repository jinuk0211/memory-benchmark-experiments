"""Audit archived r1 first-six artifacts, never assert full benchmark/global closure.

No models or service controls. Exact usage means every recorded snapshot token is
verified against the pinned archive; it does not mean uninspectable OS processes
were proven unable to write the original output. Native receipts stay untouched.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from types import ModuleType

BASE_SHA256 = '5f79f9caf4a5eb35beda61d90350ba8a787da8b675067df1eb9a51e3056c335a'
SELECTOR_SHA256 = '87f3b984327140f32ef12594a36131f33f52cef50cfc8b65f48278cda696793a'
MANIFEST_SHA256 = 'fd0aac2763441be3efcac3f88800d7c5d6251df8543346502eb7fcffebd9e944'
ITEMS_SHA256 = '4bd01e08b31324ef6916bc450436cba032f2267727a063d8339bb906b6c7bfac'
SELECTED_CIDS = ('conv-26', 'conv-30', 'conv-41', 'conv-42', 'conv-43', 'conv-44')
SELECTED_QA = 885
EXPECTED_GROUPS = {'generation': 3123, 'embedding': 36516}
ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, csv.Error, tarfile.TarError)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_dependencies(base_path: Path, selector_path: Path) -> tuple[ModuleType, ModuleType]:
    """Only execute the exact reviewed stdlib-only helpers, without pycache writes."""
    modules = []
    for path, pin in ((base_path, BASE_SHA256), (selector_path, SELECTOR_SHA256)):
        source = path.read_bytes()
        require(hashlib.sha256(source).hexdigest() == pin, f'Frozen dependency hash mismatch: {path}')
        module = ModuleType(path.stem)
        exec(compile(source, str(path), 'exec'), module.__dict__)  # noqa: S102 - exact reviewed SHA gate
        modules.append(module)
    return modules[0], modules[1]


def archive_inventory(archive: Path, prefix: str, extra: dict) -> dict:
    """Stream member bytes without extraction; reject links, duplicates and extras."""
    found, seen, provenance = {}, set(), {}
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            name = member.name
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in name
                    and str(path) == name.rstrip('/'), 'Unsafe/noncanonical archive member')
            require(name not in seen, 'Duplicate archive member')
            seen.add(name)
            if member.isdir():
                require(name.rstrip('/') in (prefix, 'provenance') or name.startswith(prefix + '/'),
                        'Foreign archive directory')
                continue
            require(member.isfile(), 'Archive links/special members forbidden')
            with stream.extractfile(member) as data:
                digest = hashlib.file_digest(data, 'sha256').hexdigest()
            if name.startswith(prefix + '/'):
                found[name[len(prefix) + 1:]] = digest
            else:
                require(name in extra, 'Foreign archive file')
                provenance[name] = digest
    require(provenance == extra, 'Archive detached provenance mismatch')
    return found


def partial_dataset(base: ModuleType, manifest: dict, output: Path) -> tuple[dict, dict]:
    """Keep original 1540 identities while proving only the fixed six native builds."""
    samples = json.loads(Path(manifest['dataset']).read_text(encoding='utf-8'))
    require(isinstance(samples, list) and len(samples) == base.CONV_TOTAL, 'Original dataset conversation scope mismatch')
    all_questions, questions, counts, sessions, seen = {}, {}, {}, set(), set()
    for sample in samples:
        cid, qas = sample['sample_id'], sample['qa']
        require(isinstance(cid, str) and cid not in seen and isinstance(qas, list)
                and all(isinstance(q, dict) for q in qas), 'Malformed original QA identities')
        seen.add(cid)
        selected = [q for q in qas if q.get('category') != 5]
        require(all(type(q.get('category')) is int and q['category'] in (1, 2, 3, 4) for q in selected), 'Invalid QA category')
        all_questions.update({(cid, i): q for i, q in enumerate(selected)})
        if cid not in SELECTED_CIDS:
            continue
        questions.update({(cid, i): q for i, q in enumerate(selected)})
        counts[cid] = len(selected)
        conversation, number, raw_turns = sample['conversation'], 1, 0
        require(isinstance(conversation, dict), 'Malformed native conversation')
        while f'session_{number}' in conversation:
            turns = conversation[f'session_{number}']
            require(isinstance(turns, list) and all(isinstance(t, dict) and isinstance(t.get('speaker'), str) for t in turns),
                    'Malformed native raw turns')
            raw_turns += len(turns)
            sessions.add((cid, number))
            number += 1
        require(raw_turns > 0, 'Empty/unproven raw index')
    require(len(all_questions) == base.QA_TOTAL and tuple(counts) == SELECTED_CIDS
            and len(questions) == SELECTED_QA, 'Original/partial QA scope mismatch')
    expected = {(cid, config, session) for cid, session in sessions for config in base.POLICIES}
    found, memory = set(), dict.fromkeys(counts, 0)
    with (output / 'write_stats.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            key = (row['conv_id'], row['config'], int(row['session']))
            require(key in expected and key not in found, 'Foreign/duplicate native session')
            found.add(key)
            mem, raw = int(row['mem_tok']), int(row['raw_tok'])
            require(mem >= 0 and raw >= 0, 'Invalid native write token counts')
            if key[1] == 'full':
                memory[key[0]] += mem
    require(found == expected and all(n > 0 for n in memory.values()), 'Partial full/config session evidence missing or empty')
    return questions, counts


def validate_closure(base: ModuleType, closure: dict, launcher: dict, manifest: dict, output: Path,
                     before: dict, manifest_sha: str) -> None:
    require(closure.get('schema_version') == 1 and closure.get('closure_kind') == 'intentional-partial'
            and closure.get('run_id') == base.RUN_ID and closure.get('output') == str(output)
            and closure.get('supervisor_state') == 'EXITED' and closure.get('child_exit_code') == -15,
            'Intentional-partial closure identity/termination mismatch')
    require(closure.get('known_benchmark_writer_pids') == [] and 'writer_pids' in closure
            and closure['writer_pids'] is None and closure.get('global_writer_absence_proven') is False
            and isinstance(closure.get('uninspected_processes'), list), 'Known/global writer evidence missing or contradicted')
    require(closure.get('covered_conversations') == list(SELECTED_CIDS)
            and closure.get('captured_qa_per_policy') == SELECTED_QA
            and closure.get('reader_outputs') == SELECTED_QA * 37
            and closure.get('generation_groups') == EXPECTED_GROUPS['generation']
            and closure.get('embedding_groups') == EXPECTED_GROUPS['embedding']
            and closure.get('benchmark_complete') is False, 'Intentional-partial boundary mismatch')
    require(closure.get('raw_source_sha256') == before and closure.get('manifest_sha256') == manifest_sha,
            'Partial closure inventory/manifest mismatch')
    require(launcher.get('run_id') == base.RUN_ID and launcher.get('manifest_sha256') == manifest_sha
            and launcher.get('child_exit_code') == -15 and launcher.get('source_unchanged') is True
            and launcher.get('inference_complete') is False and launcher.get('errors') == [],
            'Launcher termination/source evidence mismatch')
    command = [manifest['python'], '-u', str(Path(manifest['source_root']) / 'scripts/run_system_v15.py'),
               '--comparison', '--dataset', 'locomo', '--configs', 'full,no_adaptive,no_residual',
               '--budgets', '400,1600,4000', '--out', str(output)]
    require(launcher.get('command') == command, 'Native recipe CLI mismatch')


def read_runtime(base: ModuleType, output: Path, questions: dict) -> tuple[list, list, list]:
    generations, embeddings, issues, identities = [], [], [], set()
    for kind in ('generation', 'embedding'):
        folder, groups = output / 'runtime' / kind, defaultdict(dict)
        try:
            if not folder.is_dir():
                issues.append(f'Missing {kind} journal')
            for path in folder.iterdir() if folder.is_dir() else ():
                match = re.fullmatch(r'(\d{8})\.(request|response|receipt)\.json', path.name)
                if not match or not path.is_file():
                    issues.append(f'Unclosed/foreign runtime artifact: {path.name}')
                    continue
                groups[int(match[1])][match[2]] = path
        except OSError as exc:
            issues.append(f'Unreadable {kind} journal: {exc}')
        if sorted(groups) != list(range(1, EXPECTED_GROUPS[kind] + 1)):
            issues.append(f'{kind} group count/continuity mismatch')
        required = {'request', 'response', 'receipt'} if kind == 'generation' else {'request', 'receipt'}
        for sequence, paths in sorted(groups.items()):
            if set(paths) != required:
                issues.append(f'Incomplete {kind} group {sequence}')
                continue
            try:
                objects = {k: base.read_object(path) for k, path in paths.items()}
                request, receipt = objects['request'], objects['receipt']
                try:
                    require(receipt.get('request_hash') == receipt.get('request_sha256') == base.digest(paths['request']), 'Request hash mismatch')
                    if kind == 'generation':
                        require(receipt.get('response_sha256') == base.digest(paths['response']), 'Response hash mismatch')
                    require(receipt.get('status') == 'success' and receipt.get('usage_complete') is True, 'Failed/incomplete native call')
                    duration = receipt.get('duration_s')
                    require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0, 'Invalid native duration')
                    if kind == 'generation':
                        inference = receipt.get('inference_duration_s')
                        require(type(inference) in (int, float) and math.isfinite(inference)
                                and 0 <= inference <= duration, 'Invalid native inference duration')
                except ERRORS as exc:
                    issues.append(f'{kind}:{sequence}: {exc}')
                if kind == 'generation':
                    rows, errors = base.generation_rows(request, objects['response'], receipt, sequence, paths, questions, identities)
                    generations.extend(rows)
                else:
                    row, errors = base.embedding_row(request, receipt, sequence, questions)
                    row['request_sha256'] = receipt.get('request_sha256')
                    if not (all(base.integer(receipt.get(k)) for k in ('input_tokens', 'padded_tokens'))
                            and all(isinstance(receipt.get(k), list) and all(base.integer(n) for n in receipt[k])
                                    for k in ('tokens_per_input', 'shape'))):
                        errors.append('Receipt embedding integer counts invalid')
                    embeddings.append(row)
                issues.extend(f'{kind}:{sequence}: {error}' for error in errors)
            except ERRORS as exc:
                issues.append(f'{kind}:{sequence}: {exc}')
    return generations, embeddings, issues


def qa_matrix(base: ModuleType, output: Path, questions: dict, generations: list) -> dict:
    expected = {(cid, ordinal, cfg, policy) for cid, ordinal in questions for cfg, policies in base.POLICIES.items() for policy in policies}
    rows = defaultdict(list)
    for row in generations:
        if row['qa_key'] is not None:
            rows[row['qa_key']].append(row)
    require(set(rows) == expected and len(expected) == SELECTED_QA * 37
            and all(len(values) == 1 for values in rows.values()), 'Partial raw QA matrix missing/duplicate/foreign')
    require(base.digest(output / 'items.csv') == ITEMS_SHA256, 'Fixed partial items hash mismatch')
    seen = set()
    with (output / 'items.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            key = (row['conv_id'], int(row['question_ordinal']), row['config'], row['policy'])
            require(key in expected and key not in seen, 'Foreign/duplicate CSV identity')
            qa, raw = questions[key[:2]], rows[key][0]['native_text']
            require(isinstance(raw, str) and row['pred'] == raw.strip() and row['question'] == qa['question']
                    and row['gold'] == str(qa.get('answer', '')) and row['category'] == str(qa['category']),
                    'CSV/raw/original metadata mismatch')
            seen.add(key)
    require(seen == expected, 'Partial CSV matrix incomplete')
    return rows


def audit_partial(output: Path, manifest_path: Path, manifest_sha: str, closure_path: Path, closure_sha: str,
                  archive: Path, archive_sha: str, base_path: Path, selector_path: Path) -> dict:
    base, selector = load_dependencies(base_path, selector_path)
    issues, manifest, launcher, closure, questions, counts, before, qa_rows = [], {}, {}, {}, {}, {}, {}, {}
    qa_complete = False
    try:
        require(manifest_sha == MANIFEST_SHA256 and base.digest(manifest_path) == MANIFEST_SHA256, 'Fixed manifest hash mismatch')
        before, manifest = base.inventory(output), base.read_object(manifest_path)
        launcher = base.source_proof(manifest, manifest_path, manifest_sha, output)
        require(base.digest(closure_path) == closure_sha and base.digest(archive) == archive_sha, 'Closure/archive hash mismatch')
        closure = base.read_object(closure_path)
        validate_closure(base, closure, launcher, manifest, output, before, manifest_sha)
        extras = {f'provenance/{closure_path.name}': closure_sha, f'provenance/{manifest_path.name}': manifest_sha}
        require(archive_inventory(archive, output.name, extras) == before, 'Archive/output inventory mismatch')
        questions, counts = partial_dataset(base, manifest, output)
    except ERRORS as exc:
        issues.append(str(exc))
    generations, embeddings, raw_issues = read_runtime(base, output, questions)
    issues.extend(raw_issues)
    try:
        qa_rows = qa_matrix(base, output, questions, generations)
        qa_complete = True
    except ERRORS as exc:
        issues.append(str(exc))
    valid = not issues and bool(generations) and bool(embeddings)
    groups = {key: base.summarize([r for r in generations if r['bucket'] == key],
                                 [r for r in embeddings if r['bucket'] == key], valid)
              for key in sorted({r['bucket'] for r in generations + embeddings})}
    policies = {f'{cfg}/{policy}': {**base.summarize([r for k, values in qa_rows.items() if k[2:] == (cfg, policy)
                                                    for r in values], [], valid), 'response_count': SELECTED_QA}
                for cfg, names in base.POLICIES.items() for policy in names} if qa_complete else {}
    selected = None
    if valid:
        try:
            report = {'embedding_ledger': embeddings, 'by_phase_config': groups}
            embedding = selector.select_embeddings(base, output, report, counts)
            components = {key: selector.generation_tokens(groups[key]) for key in
                          ('fact_extraction/shared', 'memory_build/full', 'query_reform/full')}
            components['qa/full/certified'] = selector.generation_tokens(policies['full/certified'])
            totals = {key: sum(value[key] for value in components.values()) for key in selector.TOKEN_KEYS}
            selected = {'scope': 'Recorded first-six primary-policy dependencies only; not full1540 baseline costs.',
                'qa_per_policy': SELECTED_QA, 'additive_across_policies': False, 'standalone_actual_run': False,
                'certified': {'generation': {'exact_tokens': totals, 'components': components}, 'embedding': embedding},
                'naive': {'generation': {'exact_tokens': selector.generation_tokens(policies['full/full_raw'])},
                          'causal_embedding_tokens': 0, 'causal_construction_tokens': 0}}
        except ERRORS as exc:
            issues.append(str(exc))
    try:
        require(base.inventory(output) == before and base.digest(closure_path) == closure_sha
                and base.digest(archive) == archive_sha, 'Snapshot/closure/archive changed during audit')
        base.source_proof(manifest, manifest_path, manifest_sha, output)
    except ERRORS as exc:
        issues.append(str(exc))
    valid = not issues and selected is not None
    if not valid:
        selected = None
        for value in [*groups.values(), *policies.values()]:
            value['generation']['exact_tokens'] = None
            value['embedding']['exact_input_tokens'] = None
            value['generation']['cached_prompt_tokens']['exact'] = None
    return {'schema_version': 1, 'run_id': base.RUN_ID, 'complete': False, 'benchmark_complete': False,
        'closed_sources_complete': False, 'recorded_snapshot_usage_complete': valid,
        'usage_scope': 'Exact tokens in the validated archived partial snapshot, not a global writer-closure proof.',
        'child_exit_code': launcher.get('child_exit_code'), 'issues': issues,
        'known_benchmark_writer_pids': closure.get('known_benchmark_writer_pids'),
        'uninspected_processes': closure.get('uninspected_processes'),
        'proof': {'manifest_sha256': manifest_sha, 'closure_sha256': closure_sha, 'archive_sha256': archive_sha,
                  'base_auditor_sha256': BASE_SHA256, 'selector_sha256': SELECTOR_SHA256,
                  'inventory_unchanged_and_archive_matched': valid},
        'qa_coverage': {'partial_matrix_complete': qa_complete, 'recorded': sum(r['qa_key'] is not None for r in generations),
            'partial_expected': SELECTED_QA * 37, 'original_expected': base.QA_TOTAL * 37,
            'qa_per_policy': SELECTED_QA, 'original_qa_per_policy': base.QA_TOTAL},
        'covered_conversations': list(SELECTED_CIDS), 'remaining_qa_per_policy': base.QA_TOTAL - SELECTED_QA,
        **base.summarize(generations, embeddings, valid), 'by_phase_config': groups, 'qa_by_policy': policies,
        'selected_primary_usage': selected,
        'generation_ledger': [{k: v for k, v in row.items() if k != 'native_text'} for row in generations],
        'embedding_ledger': embeddings, 'runtime_window_observations': launcher.get('runtime'),
        'exclusive_gpu_energy_wh': None, 'exclusive_rental_usd': None,
        'limitations': ['Global writer absence is not proven: container process inspection has an explicit blind spot.',
            'Exact snapshot totals include all 37 policies for the first six conversations, not a single complete baseline.',
            'Embedding attribution is derived from pinned synchronous source order, not independently logged text/callsite proof.',
            'Native summary.csv is absent at intentional interruption; launcher native_outputs.rows=0 is not actual CSV coverage.',
            'Length outputs and cached logical tokens are retained; no exclusive device energy/rental allocation.']}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'manifest', 'closure', 'archive', 'base-auditor', 'selector', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest', 'closure', 'archive'):
        parser.add_argument('--expected-' + name + '-sha256', required=True)
    args = parser.parse_args(argv)
    report = args.report.resolve()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    protected = (args.output.resolve(), Path(manifest['source_root']).resolve())
    require(not report.exists() and not any(report.is_relative_to(p) for p in protected),
            'Report must be fresh and outside the immutable source/snapshot')
    result = audit_partial(args.output, args.manifest, args.expected_manifest_sha256, args.closure,
                           args.expected_closure_sha256, args.archive, args.expected_archive_sha256,
                           args.base_auditor, args.selector)
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return 0 if result['recorded_snapshot_usage_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
