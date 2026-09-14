"""Recover only a closed r26 postrun import failure, then run the unchanged full branch.

The original probe exit 1 and raw files remain immutable. No probe inference is
repeated. A fresh r27 directory contains derived validation/postrun receipts only;
full inference, accounting and official scoring retain the native r26 run ID.
"""
import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

SOURCE_ROOT = '/workspace/MemoryData-amem-original-policy-r26'
RECOVERY_RUN_ID = 'locomo-amem-postrun-recovery-20260908-r27'
RECOVERY_OUTPUT = '/workspace/locomo-amem-postrun-recovery-r27'
PLAN_SHA = '4db3808753045e4754cdd4d092e2267f80c6cd584e65a9a198bd8081ade981f4'
CLOSURE_SHA = '1b6ebdbdc544c05addcd8e54839cb917c236ea61d8c57a4511add4c3e29459fe'
HELPER_SHA = 'fb186588bac538678fe4a566062943489ec349e1bf71de783310b1d5d1dcc8ae'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_pins(pins: dict) -> None:
    require(isinstance(pins, dict) and bool(pins), 'Missing file pins')
    for name, expected in pins.items():
        path = Path(name)
        require(path.is_absolute() and not path.is_symlink(), 'Unsafe pinned file path')
        if expected is None:
            require(not path.exists(), f'Expected absent package initializer: {name}')
        else:
            require(digest(path) == expected, f'File hash differs: {name}')


def inventory(root: Path) -> dict:
    require(root.is_dir() and not root.is_symlink(), 'Missing or linked probe root')
    result = {}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Probe inventory contains a symlink')
        if path.is_file():
            result[str(path)] = digest(path)
    return result


def verify_probe_inventory(closure: dict) -> None:
    require(inventory(Path(closure['probe_root'])) == closure['raw_source_sha256'],
            'Closed probe inventory changed')


def verify_prior_artifacts(closure: dict) -> None:
    verify_pins(read_json(Path(closure['probe_root']) / 'prior_artifact_sha256.json'))


def load_inputs(manifest_path: Path, manifest_sha: str) -> tuple:
    require(digest(manifest_path) == manifest_sha, 'Additive manifest hash differs')
    manifest = read_json(manifest_path)
    require(manifest.get('schema_version') == 1 and manifest.get('recovery_run_id') == RECOVERY_RUN_ID
            and manifest.get('source_root') == SOURCE_ROOT
            and manifest.get('recovery_output') == RECOVERY_OUTPUT, 'Wrong recovery scope')
    pins = manifest['file_sha256']
    require(set(pins) == {str(Path(__file__).resolve())}, 'Pin exactly the additive runner')
    verify_pins(pins)
    plan_ref, closure_ref = manifest['source_plan'], manifest['probe_closure']
    require(plan_ref['sha256'] == PLAN_SHA and closure_ref['sha256'] == CLOSURE_SHA,
            'Recovery must use the exact closed r26 attempt')
    verify_pins({plan_ref['path']: PLAN_SHA, closure_ref['path']: CLOSURE_SHA})
    plan, closure = read_json(plan_ref['path']), read_json(closure_ref['path'])
    require(plan['source_root'] == SOURCE_ROOT and closure['source_plan_path'] == plan_ref['path']
            and closure['source_plan_sha256'] == PLAN_SHA, 'Source plan mismatch')
    verify_pins(plan['file_sha256'])
    require(closure['helper_sha256'] == HELPER_SHA
            and closure['supplementary_package_files'].get(closure['helper_path']) == HELPER_SHA,
            'Wrong original supplementary helper')
    verify_pins(closure['supplementary_package_files'])
    verify_probe_inventory(closure)
    sys.path.insert(0, SOURCE_ROOT)
    driver = importlib.import_module('scripts.amem_original_policy_recovery_r26')
    require(Path(driver.__file__).resolve() == Path(SOURCE_ROOT) /
            'scripts/amem_original_policy_recovery_r26.py', 'Wrong imported r26 driver')
    driver.require_policy(plan)
    return manifest, plan, closure, driver


def validate_probe(plan: dict, closure: dict, driver: ModuleType, destination: Path) -> dict:
    """Reapply the frozen post-inference gates using a new normalized copy only."""
    require(closure.get('schema_version') == 1 and closure.get('closure_kind') == 'probe-postrun-import-failure'
            and closure.get('native_exit_code') == 1 and closure.get('supervisor_state') == 'EXITED'
            and closure.get('known_benchmark_writer_pids') == []
            and closure.get('original_failure_preserved') is True
            and closure.get('benchmark_complete') is False and closure.get('full_output_exists') is False
            and closure.get('qa_count') == 3 and closure.get('memory_turns') == 80
            and closure.get('run_id') == driver.PROBE_RUN_ID, 'Not the closed postrun-only failure')
    verify_probe_inventory(closure)
    out = Path(closure['output'])
    require(out == Path(closure['probe_root']) / 'a_mem', 'Wrong probe output')
    for stem in ('result', 'benchmark_log', 'recovery_status'):
        name = closure[stem + '_path']
        require(closure['raw_source_sha256'].get(name) == closure[stem + '_sha256'], 'Unpinned probe evidence')
    log = Path(closure['benchmark_log_path']).read_text(encoding='utf-8')
    require('Failed to import supplementary LongMemEval recall report generator.' in log
            and '_maybe_generate_post_run_reports(output_path, dataset_config)' in log
            and log.rstrip().endswith("ModuleNotFoundError: No module named 'evaluation'"),
            'Different native failure traceback')
    status = read_json(closure['recovery_status_path'])
    require(status.get('state') == 'failed' and status.get('complete') is False
            and status.get('error_type') == 'RuntimeError'
            and status.get('error') == 'a_mem probe exited with code 1', 'Wrong original exit record')
    probes = importlib.import_module('scripts.run_locomo_live_probes')
    source = read_json(plan['dataset'])
    data = read_json(out / 'test_dataset.json')
    require(data == probes.prefix_dataset(source, 80), 'Probe is not the original 80-turn prefix')
    turns = sum(len(v) for k, v in data[0]['conversation'].items() if re.fullmatch(r'session_\d+', k))
    require(turns == 80, 'Wrong native memory prefix length')
    item = next(item for item in plan['methods'] if item['method'] == 'a_mem')
    config = driver.yaml.safe_load(Path(item['dataset_config']).read_text())
    config.update(test_files=str(out / 'test_dataset.json'), max_test_samples=1)
    require(driver.yaml.safe_load((out / 'dataset.yaml').read_text()) == config, 'Probe dataset settings changed')
    expected = dict(list(driver.queue.expected_questions(plan['dataset'], 1540).items())[:3])
    raw = read_json(closure['result_path'])['data']
    require(len(raw) == 3, 'Expected exactly three saved QA')
    for ordinal, (row, ((sample, index), qa)) in enumerate(zip(raw, expected.items())):
        qid = str(qa.get('question_id') or f'{sample}_qa{index}')
        require(row.get('context_id') == 0 and type(row.get('query_id')) is int
                and row['query_id'] == ordinal and row.get('question_id') == qid
                and row.get('qa_pair_id') == qid and row.get('status') is None,
                'Saved QA identities or native status differ')
    agent = driver.yaml.safe_load(Path(item['agent_config']).read_text())
    records = driver.normalize_records(raw, expected,
        driver.get_template(config['sub_dataset'], 'query', agent['agent_name']))
    normalized = destination / 'normalized_probe_results.json'
    write_json(normalized, {'data': records})
    predictions = driver.queue.canonical_predictions([normalized], expected)
    valid, issues, missing = driver.validate_predictions(predictions, expected)
    llm = driver.read_jsonl(out / 'llm_usage.jsonl')
    embedding = driver.read_jsonl(out / 'embedding_usage.jsonl')
    rows = llm + embedding
    require(len(llm) == 163 and len(embedding) == 162, 'Closed probe call counts differ')
    events = []
    journals = list((out / 'semantic_outcomes').iterdir())
    require(bool(journals), 'Missing native event journals')
    for path in journals:
        require(path.is_file() and re.fullmatch(r'[0-9]+\.jsonl', path.name), 'Invalid native event journal')
        events.extend(driver.read_jsonl(path))
    usage = probes.summarize_usage(rows, True)
    driver.queue.audit_coverage(rows, expected, driver.PROBE_RUN_ID, 'a_mem')
    delivery = driver.audit_original_policy(rows, events, driver.PROBE_RUN_ID, expected)
    probes.check_gate(valid, issues, missing, usage, delivery)
    require(usage['reported_tokens']['total_tokens'] == 297045, 'Closed probe token totals differ')
    report = {'passed': True, 'diagnostic_only': True, 'qa_count': 3, 'memory_turns': turns,
        'original_native_exit_code': 1, 'original_failure_preserved': True, 'new_inference_calls': 0,
        'usage': delivery['usage'], 'original_policy_delivery_complete': delivery['original_policy_delivery_complete'],
        'coverage_complete': True, 'global_writer_absence_proven': closure['global_writer_absence_proven'],
        'uninspected_processes': closure['uninspected_processes'],
        'note': 'Saved inference gates passed; original process still exited 1. This does not assert global writer absence.'}
    write_json(destination / 'restored_probe_gate.json', report)
    verify_probe_inventory(closure)
    return report


POSTRUN_CODE = '''import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import main
import yaml
from evaluation.longmemeval.memoryagentbench_longmemeval_recall import supports_memoryagentbench_longmemeval_recall
cfg = yaml.safe_load(Path(sys.argv[3]).read_text())
applicable = supports_memoryagentbench_longmemeval_recall(cfg)
if applicable is not False:
    raise RuntimeError("Original supplementary helper is unexpectedly applicable")
postrun_result = main._maybe_generate_post_run_reports(sys.argv[4], cfg)
if postrun_result is not None:
    raise RuntimeError("Original supplementary postrun did not return None")
print("AMEM_NATIVE_POSTRUN_NOOP")
'''


def recover_postrun(plan: dict, closure: dict, destination: Path) -> dict:
    """Run the original postrun function on CPU; helper import path is child-only."""
    helper_root = str(Path(closure['helper_path']).parents[2])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([plan['source_root'], helper_root]))
    result = subprocess.run([plan['client_python'], '-B', '-c', POSTRUN_CODE, plan['source_root'],
        helper_root, str(Path(closure['output']) / 'dataset.yaml'), closure['result_path']],
        cwd=plan['source_root'], env=env, capture_output=True, text=True, timeout=120, check=False)
    for name, content in [('stdout', result.stdout), ('stderr', result.stderr)]:
        with (destination / f'postrun_{name}.log').open('x', encoding='utf-8') as stream:
            stream.write(content)
    require_marker = result.returncode == 0 and result.stdout.strip().endswith('AMEM_NATIVE_POSTRUN_NOOP')
    if not require_marker:
        raise RuntimeError(f'Native CPU postrun recovery failed: exit {result.returncode}')
    report = {'exit_code': result.returncode, 'new_inference_calls': 0,
              'helper_applicable': False, 'native_postrun_returned_none': True,
              'original_native_exit_code': 1, 'original_failure_preserved': True}
    write_json(destination / 'postrun_receipt.json', report)
    return report


def create_result_alias(output: Path) -> dict:
    """The frozen r26 finalizer expects selected_results; preserve native bytes."""
    source, target = output / 'parallel_results.json', output / 'selected_results.json'
    data = source.read_bytes()
    with target.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    require(digest(source) == digest(target), 'Derived result alias differs')
    return {'source': str(source), 'derived_alias': str(target), 'sha256': digest(source),
            'byte_identical': True, 'native_result_modified': False}


def run_recovery(manifest_path: Path, manifest_sha: str) -> int:
    manifest, plan, closure, driver = load_inputs(manifest_path, manifest_sha)
    destination, full = Path(manifest['recovery_output']), Path(plan['output'])
    if full.exists():
        raise FileExistsError(full)
    destination.mkdir(exist_ok=False)
    receipt = {'schema_version': 1, 'recovery_run_id': RECOVERY_RUN_ID, 'native_run_id': plan['run_id'],
        'manifest_sha256': manifest_sha, 'original_native_exit_code': 1, 'original_failure_preserved': True,
        'probe_inference_repeated': False, 'benchmark_complete': False, 'complete': False,
        'global_writer_absence_proven': closure['global_writer_absence_proven'],
        'uninspected_processes': closure['uninspected_processes']}
    try:
        names = (*driver.PREDECESSORS, 'locomo-amem-original-policy-r26')
        require(all(driver.queue.supervisor_state(name) == 'EXITED' for name in names), 'Predecessor is not EXITED')
        verify_prior_artifacts(closure)
        receipt['saved_probe'] = validate_probe(plan, closure, driver, destination)
        receipt['postrun'] = recover_postrun(plan, closure, destination)
        verify_probe_inventory(closure)
        driver.require_policy(plan)
        verify_prior_artifacts(closure)
        driver.require_free_ports()
        driver.queue.wait_vllm(plan)
        expected = driver.queue.expected_questions(plan['dataset'], 1540)
        full.mkdir(exist_ok=False)
        output = full / 'a_mem'
        encoder = [plan['server_python'], str(Path(plan['source_root']) / 'scripts/serve_minilm_embeddings.py'),
                   '--model-path', plan['embedding_path']]
        with driver.policy_environment(output), driver.queue.service(encoder, full / 'embedding_server.log') as process:
            driver.queue.wait_health('http://127.0.0.1:18081', process)
            driver.queue.run_method(plan, next(item for item in plan['methods'] if item['method'] == 'a_mem'),
                                    expected, context_indices=list(range(10)))
        receipt['derived_result_alias'] = create_result_alias(output)
        report = driver.finalize_attempt(plan, output, expected)
        driver.require_policy(plan)
        verify_probe_inventory(closure)
        verify_prior_artifacts(closure)
        if report.get('benchmark_complete') is not True:
            raise RuntimeError('Full native inference/accounting is incomplete')
        receipt.update(state='finished_native_policy_inference', benchmark_complete=True,
                       full_report_path=str(output / 'original_policy_report.json'),
                       note='Full r26 run completed; closed full artifact backup remains external. '
                            'Probe and all prior failed-attempt costs remain separate, never erased or duplicated.')
        write_json(destination / 'recovery_receipt.json', receipt)
        return 0
    except Exception as error:
        receipt.update(state='failed', error_type=type(error).__name__, error=str(error))
        write_json(destination / 'recovery_receipt.json', receipt)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args(argv)
    return run_recovery(args.manifest, args.manifest_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
