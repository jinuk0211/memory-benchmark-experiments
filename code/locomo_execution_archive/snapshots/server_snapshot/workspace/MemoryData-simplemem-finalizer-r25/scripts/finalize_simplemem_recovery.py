"""Offline r13/r17[/r21] SimpleMem composition; never runs inference or changes inputs.

Manifest v1: dataset/scorer/closure are original absolute POSIX artifact keys;
old/new contain root, plan, receipt keys. files maps each key to {path, sha256}
for an explicit local snapshot. Receipts use original_attempt_closed,
parallel_status, raw_source_sha256. A separate pinned closure proof must contain
run_ids, supervisor_states, writers (root -> []), and matching complete
files_sha256_before/after inventories. The caller obtains these on the server;
this offline tool cannot observe remote writers or authenticate a supplied pin.
Explicit mode='r13_r17_r21' additionally requires repair={root,plan,receipt}.
Its non-parallel receipt pins recovery_status, preparation_sha256, plan_path and
plan_sha256. The native 1540-row retry seed is skip evidence, NOT final selection.

Historical r13 semantic rejections were not instrumented. A valid score/token
audit can yield benchmark_complete=true, but complete/metrics_complete remain
false. Gross token totals already include every rejection; never add it again.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import locomo_server_queue as queue
from scripts.amem_failed_context_recovery_r15 import failed_contexts
from scripts.audit_simplemem_semantic_discards import audit_semantic_discards
from scripts.compose_simplemem_recovery import compose_predictions, partition_usage
from scripts.finalize_locomo_comparison import get_template, normalize_records
from scripts.prepare_saved_simplemem_retry import RETRIES, validate_rows
from scripts.response_delivery_audit import audit_delivery
from scripts.score_locomo_comparison import (
    build_report,
    expected_questions,
    read_jsonl,
    runtime_metrics,
    summarize_usage,
    validate_predictions,
    write_report,
)
from scripts.simplemem_saved_recovery_r21 import PRESERVED_FIELDS
from utils.artifact_paths import generate_agent_save_folder_path

RUNS = {'old': 'locomo-full-simplemem-validated-20260908-r13',
        'new': 'locomo-simplemem-failed-contexts-20260908-r17'}
INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]
REPAIR_RUN = 'locomo-simplemem-qa-repair-20260908-r21'


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file(manifest, key):
    entry = manifest['files'][key]
    path = Path(entry['path'])
    if (not path.is_absolute() or not PurePosixPath(key).is_absolute()
            or str(PurePosixPath(key)) != key or '..' in PurePosixPath(key).parts
            or digest(path) != entry['sha256']):
        raise ValueError(f'Unpinned, changed or noncanonical input: {key}')
    return path


def _json(manifest, key):
    return json.loads(_file(manifest, key).read_text(encoding='utf-8'))


def _plans(manifest, runs):
    plans, templates, configs = {}, [], []
    for name, run in runs.items():
        source = manifest[name]
        plan = plans[name] = _json(manifest, source['plan'])
        if (plan.get('run_id') != run or plan.get('dtype') != 'float16'
                or plan.get('model') != 'Qwen/Qwen3.5-9B'
                or str(PurePosixPath(plan['output']) / 'simplemem') != source['root']):
            raise ValueError('Wrong original run, model, precision or source root')
        items = [item for item in plan['methods'] if item['method'] == 'simplemem']
        if len(items) != 1:
            raise ValueError('Expected exactly one SimpleMem configuration')
        item = items[0]
        template_key = str(PurePosixPath(item['agent_config']).parents[1]
                           / 'benchmark/memoryagentbench/prompts/benchmark_templates.py')
        for key in (manifest['dataset'], manifest['scorer'], item['agent_config'],
                    item['dataset_config'], template_key):
            if plan['file_sha256'].get(key) != digest(_file(manifest, key)):
                raise ValueError('Execution plan does not pin a consumed config/data/scorer')
        if digest(_file(manifest, template_key)) != digest(sys.modules[get_template.__module__].__file__):
            raise ValueError('Imported query template differs from the execution source')
        agent = yaml.safe_load(_file(manifest, item['agent_config']).read_text(encoding='utf-8'))
        dataset = yaml.safe_load(_file(manifest, item['dataset_config']).read_text(encoding='utf-8'))
        templates.append(get_template(dataset['sub_dataset'], 'query', agent['agent_name']))
        configs.append((agent, dataset))
    keys = ('model', 'model_revision', 'dtype', 'embedding_path', 'dataset', 'scorer')
    if (any(not plans['old'].get(key) or plans['old'][key] != plan.get(key) for plan in plans.values() for key in keys)
            or plans['old']['dataset'] != manifest['dataset'] or plans['old']['scorer'] != manifest['scorer']
            or any(config != configs[0] for config in configs) or any(t != templates[0] for t in templates)):
        raise ValueError('Original/recovered model, embedding, data or configuration changed')
    return plans, templates[0]


def _inventory(manifest, name, run):
    source = manifest[name]
    root = source['root']
    receipt = _json(manifest, source['receipt'])
    inventory = receipt['raw_source_sha256']
    if receipt.get('method') != 'simplemem' or receipt.get('run_id') != run or not inventory:
        raise ValueError('Wrong source receipt identity or empty inventory')
    for key, pinned in inventory.items():
        if not key.startswith(root + '/') or digest(_file(manifest, key)) != pinned:
            raise ValueError('Source receipt inventory differs from the pinned snapshot')
    if {key for key in manifest['files'] if key.startswith(root + '/')} != set(inventory):
        raise ValueError('Source snapshot contains omitted inventory entries')
    return receipt, inventory


def _usage(manifest, root, run):
    rows = read_jsonl(_file(manifest, root + '/usage.jsonl'))
    if rows != (read_jsonl(_file(manifest, root + '/llm_usage.jsonl'))
                + read_jsonl(_file(manifest, root + '/embedding_usage.jsonl'))):
        raise ValueError('Combined ledger differs from the original LLM + embedding journals')
    if any(row.get('run_id') != run for row in rows):
        raise ValueError('A source journal contains foreign historical/diagnostic attribution')
    return rows


def _source(manifest, name, expected, three=False):
    source, run = manifest[name], RUNS[name]
    root = source['root']
    receipt, inventory = _inventory(manifest, name, run)
    status = _json(manifest, root + '/parallel_status.json')
    if status != receipt['parallel_status']:
        raise ValueError('Receipt status differs from raw terminal status')
    contexts = list(range(10)) if name == 'old' else INDICES
    if name == 'old':
        if failed_contexts(status) != INDICES:
            raise ValueError('Original failed-context selection changed')
    else:
        for event in ('start', 'end'):
            events = [row for row in status['events'] if row.get('event') == event]
            if (len(events) != 9 or any(type(row.get('context')) is not int for row in events)
                    or sorted(row['context'] for row in events) != INDICES
                    or event == 'end' and any(type(row.get('exit_code')) is not int
                        or (row['exit_code'] != 0) != (three and row['context'] in (4, 6)) for row in events)):
                raise ValueError('Recovered workers lack the exact terminal event set')
        if (status.get('active_contexts') != []
                or status.get('context_indices', INDICES if three else None) != INDICES):
            raise ValueError('Recovered worker scope is open or changed')
        if three:
            if (status.get('failed') is not True or status.get('complete') is not False
                    or status.get('selected_contexts_complete') is True
                    or any(root + '/' + key in inventory for key in ('selected_attempt.json', 'selected_results.json'))):
                raise ValueError('r17 must retain its actual failed-attempt status, without fabricated selection')
        else:
            attempt = _json(manifest, root + '/selected_attempt.json')
            if (status.get('selected_contexts_complete') is not True or status.get('failed') is not False
                    or attempt.get('run_id') != run or attempt.get('method') != 'simplemem'
                    or attempt.get('requires_composite') is not True or attempt.get('context_indices') != INDICES):
                raise ValueError('Recovered selected attempt is not complete')
    samples = list(dict.fromkeys(sample for sample, _ in expected))
    lookup = {(sample, str(qa.get('question_id') or f'{sample}_qa{index}')): (offset, index)
              for offset, ((sample, index), qa) in enumerate(expected.items())}
    rows, logs, log_sources, found, raw_paths = [], [], [], [], {}
    for context in contexts:
        prefix, sample = root + f'/context_{context:02d}', samples[context]
        logs.append(_file(manifest, prefix + '/worker.log').read_bytes().decode('utf-8'))
        log_sources.append({'run_id': run, 'sample_id': sample, 'path': prefix + '/worker.log'})
        results = sorted(key for key in inventory if key.startswith(prefix + '/') and key.endswith('_results.json'))
        if not results and name == 'old' and context in (3, 4, 6, 8):
            continue  # These four workers failed during memory construction, before QA.
        if len(results) != 1:
            raise ValueError('Missing or ambiguous original raw result shard')
        found.append(context)
        raw_paths[context] = results[0]
        records = _json(manifest, results[0])['data']
        global_ids = []
        for row in records:
            if row.get('status') not in (None, 'failed'):
                raise ValueError('Unknown native raw result status')
            metadata = row.get('eval_metadata') or {}
            qid = metadata.get('question_id')
            identity = lookup.get((sample, qid))
            if (identity is None or row.get('sample_id') != sample or metadata.get('sample_id') != sample
                    or metadata.get('qa_pair_id') != qid or row.get('qa_pair_id') != qid
                    or row.get('question_id', qid if row.get('status') == 'failed' else None) != qid
                    or type(row.get('context_id')) is not int or row['context_id'] != context
                    or type(row.get('query_id')) is not int or row['query_id'] != identity[0]):
                raise ValueError('Raw original/global/context QA identity changed')
            global_ids.append(row['query_id'])
        if global_ids != sorted(set(global_ids)):
            raise ValueError('Raw shard order or unique question IDs changed')
        expected_ids = [value[0] for (sid, _), value in lookup.items() if sid == sample]
        failed_shard = name == 'new' and three and context in (4, 6)
        if name == 'new' and global_ids != expected_ids:
            raise ValueError('Recovered original shard IDs or count changed')
        if failed_shard and prefix + '/completion.json' in inventory:
            raise ValueError('Failed r17 shard has a fabricated completion marker')
        if (name == 'new' and not failed_shard) or context == 5:
            marker = _json(manifest, prefix + '/completion.json')
            if (marker.get('context_index') != context or type(marker.get('context_index')) is not int
                    or marker.get('sample_id') != sample or type(marker.get('questions')) is not int
                    or marker['questions'] != len(expected_ids) or marker.get('result_path') != results[0]
                    or global_ids != expected_ids):
                raise ValueError('Completed shard marker/count/original IDs changed')
        rows.extend(records)
    if name == 'old' and (found != [0, 1, 2, 5, 7, 9] or len(rows) != 857):
        raise ValueError('Original saved 857-row scope changed')
    if name == 'new':
        if len(rows) != 1417 or (not three and _json(manifest, root + '/selected_results.json')['data'] != rows):
            raise ValueError('Recovered results differ from the complete original shards')
        if three and {r['qa_pair_id']: (r['context_id'], r['query_id']) for r in rows if r.get('status') == 'failed'} != RETRIES:
            raise ValueError('r17 must contain exactly the three original saved failures')
    return {'receipt': receipt, 'inventory': inventory, 'records': rows,
            'logs': logs, 'log_sources': log_sources, 'usage': _usage(manifest, root, run), 'raw_paths': raw_paths}


def _canonical(records, expected, template, directory):
    normalized = normalize_records(records, expected, template)
    # The existing canonical validator rejects status=failed before checking its
    # metadata. Remove ONLY this flag in a temporary validation copy, never source.
    validation = [{key: value for key, value in row.items() if key != 'status'} for row in normalized]
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        path = Path(staging) / 'metadata_validation_only.json'
        write_report(path, {'data': validation})
        rows = queue.canonical_predictions([path], expected)
    if validate_predictions(rows, expected)[1]:
        raise ValueError('Raw canonical metadata or duplicate identities failed validation')
    successful = [row for row, raw in zip(rows, records) if raw.get('status') != 'failed']
    if any(not row['prediction'].strip() for row in successful):
        raise ValueError('An allegedly successful raw prediction is empty')
    return successful


def _repair_source(manifest, plans, sources, expected, template):
    """Audit native skip/copy evidence; do not treat the retry seed as selected QA."""
    source, plan = manifest['repair'], plans['repair']
    root = source['root']
    receipt, inventory = _inventory(manifest, 'repair', REPAIR_RUN)
    status = _json(manifest, root + '/recovery_status.json')
    prep = _json(manifest, root + '/preparation.json')
    if (receipt.get('recovery_status') != status or receipt.get('plan_path') != source['plan']
            or receipt.get('plan_sha256') != digest(_file(manifest, source['plan']))
            or receipt.get('preparation_sha256') != digest(_file(manifest, root + '/preparation.json'))
            or status.get('run_id') != REPAIR_RUN or status.get('predictions_recovered') is not True
            or status.get('complete') is not False or status.get('requires_composite_audit') is not True
            or status.get('seed_is_final_selection') is not False or status.get('source_artifacts_unchanged') is not True
            or status.get('retry_ids') != sorted(RETRIES) or type(status.get('exit_code')) is not int or status['exit_code'] != 0
            or status.get('error') is not None or status.get('issues') != []
            or status.get('usage', {}).get('complete') is not True or status.get('delivery', {}).get('complete') is not True
            or type(status.get('new_memory_add_requests')) is not int or status['new_memory_add_requests'] != 0):
        raise ValueError('r21 lacks a consistent successful QA-only recovery receipt')
    samples = list(dict.fromkeys(sample for sample, _ in expected))
    probe = {'passed': True, 'skipped_queries': 1537, 'retry_global_ids': [681, 930, 993],
             'unchanged_inference_fields': list(PRESERVED_FIELDS)}
    if (prep.get('rows') != 1540 or prep.get('skipped_seed_rows') != 1537
            or prep.get('retry_ids') != sorted(RETRIES) or prep.get('seed_is_final_selection') is not False
            or prep.get('context_question_counts') != [sum(s == sample for s, _ in expected) for sample in samples]
            or prep.get('copied_tables_verified') is not True or prep.get('native_skip_probe') != probe):
        raise ValueError('r21 native skip/table preflight evidence is incomplete')
    required, copies = set(), {}
    for name, tag in (('old', 'r13'), ('new', 'r17')):
        previous = manifest[name]
        if plan['saved_simplemem'].get('source_' + tag) != previous['root']:
            raise ValueError('Saved-memory source run/root was remapped')
        required.add(previous['root'] + '/parallel_status.json')
        for kind in ('plan', 'receipt'):
            key = previous[kind]
            if (plan['saved_simplemem'].get(kind + '_' + tag) != key
                    or plan['file_sha256'].get(key) != digest(_file(manifest, key))):
                raise ValueError('r21 does not pin the actual original plan/receipt')
            required.add(key)
    required.update(sources['new']['raw_paths'].values())
    required.add(sources['old']['raw_paths'][5])
    item = next(item for item in plan['methods'] if item['method'] == 'simplemem')
    agent, dataset = [yaml.safe_load(_file(manifest, item[key]).read_text(encoding='utf-8'))
                      for key in ('agent_config', 'dataset_config')]
    if not isinstance(prep.get('memory_lineage'), list) or len(prep['memory_lineage']) != 2:
        raise ValueError('Expected exactly the two original saved memories')
    for context, lineage in zip((4, 6), prep['memory_lineage']):
        artifacts = [manifest['new']['root'] + f'/context_{context:02d}/artifacts', root + '/artifacts']
        paths = [PurePosixPath(generate_agent_save_folder_path(agent, dataset, context, base).replace('\\', '/')) for base in artifacts]
        # Hash the original POSIX path, not this offline machine's os.path.abspath.
        runtimes = [p.parent.parent / '_simplemem_runtime' / hashlib.sha256(str(p).encode()).hexdigest()[:16] for p in paths]
        sidecar = _json(manifest, str(paths[0] / 'simplemem_source_map.json'))
        if (not isinstance(sidecar, dict) or not sidecar or any(not isinstance(k, str) or not k
                or not isinstance(v, list) or not v for k, v in sidecar.items())
                or _file(manifest, str(paths[0] / 'simplemem_ready.txt')).read_text().strip() != 'ready'
                or lineage != {'context_id': context, 'sample_id': samples[context], 'run_id': RUNS['new'],
                    'source_agent': str(paths[0]), 'destination_agent': str(paths[1]),
                    'source_runtime': str(runtimes[0]), 'destination_runtime': str(runtimes[1]),
                    'table_name': 'locomo_qa_memory', 'entry_ids': sorted(sidecar)}):
            raise ValueError('Saved memory identity, native runtime rekey or source-map provenance changed')
        for original, target in ((paths[0], paths[1]), (runtimes[0], runtimes[1])):
            keys = [key for key in sources['new']['inventory'] if key.startswith(str(original) + '/')]
            if not keys:
                raise ValueError('Saved memory source inventory is empty')
            if original == runtimes[0]:
                table = str(original / 'locomo_qa_memory.lance')
                for directory, suffix in (('_versions', '.manifest'), ('data', '.lance'), ('_indices/fts', '.idx')):
                    if not any(key.startswith(table + '/' + directory + '/') and key.endswith(suffix) for key in keys):
                        raise ValueError('Saved Lance/Tantivy state is incomplete')
                if table + '/_indices/fts/meta.json' not in keys:
                    raise ValueError('Saved Tantivy metadata is absent')
            for key in keys:
                required.add(key)
                copied = str(target / PurePosixPath(key).relative_to(original))
                if copied in copies:
                    raise ValueError('Saved memory copies overlap')
                copies[copied] = sources['new']['inventory'][key]
    if (prep.get('source_sha256') != {key: digest(_file(manifest, key)) for key in required}
            or prep.get('copy_sha256') != copies):
        raise ValueError('Initial saved-memory copy/source hash evidence differs from original artifacts')
    relative = {str(PurePosixPath(key).relative_to(PurePosixPath(manifest[name]['root']) / f'context_{context:02d}' / 'artifacts'))
                for name in ('old', 'new') for context, key in sources[name]['raw_paths'].items() if name == 'new' or context == 5}
    if len(relative) != 1 or prep.get('results') != str(PurePosixPath(root) / 'artifacts' / next(iter(relative))):
        raise ValueError('Native post-retry result path changed')
    after = _json(manifest, prep['results'])['data']
    validate_rows(after, expected, template)
    if any(row.get('status') == 'failed' for row in after):
        raise ValueError('Native post-retry seed still contains failed QA')
    by_id = {row['qa_pair_id']: row for row in after}
    seed = sources['new']['records'] + [row for row in sources['old']['records'] if row['context_id'] == 5]
    if any(any(by_id[row['qa_pair_id']].get(key) != row.get(key) for key in PRESERVED_FIELDS)
           for row in seed if row['qa_pair_id'] not in RETRIES):
        raise ValueError('A skipped seed inference/input/retrieval field changed')
    recovered = _json(manifest, root + '/recovered_queries.json')['data']
    if (len(recovered) != 3 or {r.get('qa_pair_id') for r in recovered} != set(RETRIES)
            or any(row != by_id[row['qa_pair_id']] for row in recovered)):
        raise ValueError('Recovered queries differ from the exact three native post-retry results')
    return {'receipt': receipt, 'inventory': inventory, 'records': recovered,
            'logs': [_file(manifest, root + '/benchmark.log').read_bytes().decode('utf-8')],
            'log_sources': [{'run_id': REPAIR_RUN, 'sample_ids': [samples[4], samples[6]], 'path': root + '/benchmark.log'}],
            'usage': _usage(manifest, root, REPAIR_RUN), 'memory_lineage': prep['memory_lineage']}


def _usage_summary(rows):
    result = summarize_usage(rows, True)
    if any(row.get('usage_invalid') or row.get('usage_status') not in (None, 'reported') for row in rows):
        result['complete'] = False
    result['exact_tokens'] = result['reported_tokens'].copy() if result['complete'] else None
    return result


def finalize(manifest: dict, destination: str | Path) -> dict:
    """Validate explicitly pinned offline artifacts and create one fresh report directory."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Refusing to overwrite a composite evaluation')
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported manifest schema')
    mode = manifest.get('mode', 'r13_r17')
    three = mode == 'r13_r17_r21'
    if mode not in ('r13_r17', 'r13_r17_r21') or three != ('repair' in manifest):
        raise ValueError('Three-run recovery requires an explicit mode and repair source')
    runs = {**RUNS, **({'repair': REPAIR_RUN} if three else {})}
    for key in manifest['files']:
        path = _file(manifest, key).resolve()
        if path.is_relative_to(destination):
            raise ValueError('Destination overlaps a source artifact')
    for name in runs:
        status_file = '/recovery_status.json' if name == 'repair' else '/parallel_status.json'
        local_root = _file(manifest, manifest[name]['root'] + status_file).resolve().parent
        if destination.is_relative_to(local_root) or local_root.is_relative_to(destination):
            raise ValueError('Destination overlaps an original attempt directory')
    plans, template = _plans(manifest, runs)
    expected = expected_questions(_file(manifest, manifest['dataset']), 1540)
    sources = {name: _source(manifest, name, expected, three) for name in RUNS}
    if three:
        sources['repair'] = _repair_source(manifest, plans, sources, expected, template)
    inventory = {key: value for source in sources.values() for key, value in source['inventory'].items()}
    proof = _json(manifest, manifest['closure']) if manifest.get('closure') else {}
    closed = (proof.get('method') == 'simplemem' and proof.get('run_ids') == list(runs.values())
              and proof.get('supervisor_states') == dict.fromkeys(runs.values(), 'EXITED')
              and proof.get('writers') == {manifest[name]['root']: [] for name in runs}
              and proof.get('files_sha256_before') == inventory == proof.get('files_sha256_after')
              and all(source['receipt'].get('original_attempt_closed') is True for source in sources.values()))
    destination.mkdir(parents=True, exist_ok=False)
    candidates = {name: _canonical(source['records'], expected, template, destination)
                  for name, source in sources.items()}
    if [len(candidates[name]) for name in runs] != ([848, 1414, 3] if three else [848, 1417]):
        raise ValueError('Original/recovered successful candidate counts changed')
    repair_args = {'repair_rows': candidates['repair'], 'repair_run_id': REPAIR_RUN} if three else {}
    composed = compose_predictions(candidates['old'], candidates['new'], expected, *RUNS.values(), **repair_args)
    gross = [row for source in sources.values() for row in source['usage']]
    partition = partition_usage(gross, composed['provenance'], expected, *RUNS.values(),
                                **({'repair_run_id': REPAIR_RUN} if three else {}))
    summary = _usage_summary(gross)  # ALL numeric fields, including excluded attempts.
    coverage, coverage_errors = True, []
    for run in runs.values():
        selected_keys = {(item['sample'], item['index']) for item in composed['provenance'] if item['run_id'] == run}
        try:
            queue.audit_coverage([row for row in partition['selected'] if row['run_id'] == run],
                                 {key: qa for key, qa in expected.items() if key in selected_keys}, run, 'simplemem')
        except ValueError as error:
            coverage, coverage_errors = False, [*coverage_errors, str(error)]
    delivery = audit_delivery(partition['selected'])
    if not _usage_summary(partition['selected'])['complete']:
        delivery['discarded_attempt_total_tokens'] = None
    semantic = audit_semantic_discards(gross, [log for source in sources.values() for log in source['logs']])
    log_sources = [item for source in sources.values() for item in source['log_sources']]
    for index, event in enumerate(semantic['events']):
        origin = event['log_source'] = log_sources[event['log_index']]
        row = event['usage_row']
        if row and (row.get('run_id') != origin['run_id']
                    or row.get('sample_id') not in origin.get('sample_ids', [origin.get('sample_id')])):
            semantic['issues'].append({'event_index': index, 'error': 'log_source_attribution_mismatch'})
            semantic['complete'] = False
    semantic.update(recorded_events_complete=semantic['complete'], instrumentation_complete=False,
                    complete=False, exact_tokens=None,
                    historical_limit='r13 lacked native semantic-rejection events; recorded tokens are only a lower bound.')
    for filename, rows in (('predictions.jsonl', composed['predictions']), ('provenance.jsonl', composed['provenance']),
                           ('gross_usage.jsonl', gross), ('selected_usage.jsonl', partition['selected']),
                           ('excluded_usage.jsonl', partition['excluded'])):
        with (destination / filename).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    report = build_report(_file(manifest, manifest['dataset']), destination / 'predictions.jsonl',
                          _file(manifest, manifest['scorer']), destination / 'selected_usage.jsonl')
    report.update(method='simplemem', composite=True, run_ids=list(runs.values()),
        selected_questions={'old': 848, 'new': 689, 'repair': 3} if three else {'old': 848, 'new': 692},
        excluded_new_duplicate_questions=725,
        gross_attempt_usage=summary, excluded_usage=_usage_summary(partition['excluded']),
        usage_complete=summary['complete'], coverage_complete=coverage, coverage_errors=coverage_errors,
        response_delivery=delivery, delivery_complete=delivery['complete'],
        gross_response_delivery=audit_delivery(gross), semantic_discards=semantic,
        closed_sources_complete=closed, metrics_complete=False, complete=False,
        benchmark_complete=bool(report['predictions_complete'] and summary['complete'] and coverage
                                and delivery['complete'] and semantic['recorded_events_complete'] and closed),
        selection_policy=composed['policy'], exclusive_gpu_energy_wh=None,
        accounting_note='Gross = selected + excluded EXISTING rows, never added to each other again. '
            'Probe/history/diagnostic run IDs are rejected. Construction in mixed contexts includes both runs. '
            'Semantic discards and repairs are already in gross tokens. Missing usage is unknown, not zero. '
            'Per-attempt GPU/rental observations can overlap other jobs; no exclusive or proportional allocation, '
            'no sum of overlapping rental/energy observations, and no complete provider invoice is claimed.')
    report['attempt_runtime'] = [{**runtime_metrics([], _file(manifest, manifest[name]['root'] + '/telemetry.json'),
                                                   plans[name].get('hourly_rate_usd')), 'run_id': runs[name],
                                  'exclusive_attribution': False} for name in runs]
    if three:
        report['memory_reuse'] = {'construction_run_id': RUNS['new'], 'repair_run_id': REPAIR_RUN,
            'new_memory_add_requests': 0, 'lineage': sources['repair']['memory_lineage'],
            'note': 'Saved r17 construction is selected once from its original ledger. Copy hashes describe '
                    'initial byte copies; final mutable index state is separately snapshot-pinned. '
                    'r21 initialization time is saved-memory loading, not newly constructed memory.'}
    for key in manifest['files']:
        _file(manifest, key)  # Detect mutation while reading/scoring, including logs and receipts.
    report['raw_sources_sha256'] = {key: value['sha256'] for key, value in manifest['files'].items()}
    report['finalizer_sha256'] = digest(__file__)
    write_report(destination / 'report.json', report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if digest(args.manifest) != args.expected_manifest_sha256:
        raise ValueError('Offline manifest SHA256 mismatch')
    report = finalize(json.loads(args.manifest.read_text(encoding='utf-8')), args.output)
    print(json.dumps({key: report[key] for key in ('complete', 'benchmark_complete', 'metrics_complete', 'official_f1')}))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
