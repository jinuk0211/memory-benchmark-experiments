"""Pinned, foreground v15 launcher; does not change the native inference recipe.

This receipt proves process/output coverage, not final scoring or exact usage.
The native runtime owns raw token receipts; the later closed-artifact audit must
join those receipts. Missing measurements are never replaced with zero.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('/workspace/locomo-certmem-v15-20260908-r1')
PYTHON = '/venv/main/bin/python'
RUN_ID = 'locomo-certmem-v15-20260908-r1'
DATASET_SHA256 = 'cf50e013bb20551cba62f27a93f8310e70422ed31fff6010871031ac9e875993'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
HOURLY_RATE = 0.4068518518518518
BASE_POLICIES = ('units_only', 'certified') + tuple(
    f'{kind}_{budget}' for budget in (400, 1600, 4000) for kind in ('uniform', 'ours', 'certified'))
POLICIES = {'full': BASE_POLICIES + ('raw_rag_400', 'raw_rag_1600', 'raw_rag_4000', 'full_raw'),
            'no_adaptive': BASE_POLICIES, 'no_residual': BASE_POLICIES}


def digest(path: Path) -> str:
    """Hash exact bytes without normalizing prompts or line endings."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_pins(pins: dict) -> None:
    """Reject missing, changed, relative or symlinked pinned source files."""
    if not isinstance(pins, dict) or not pins:
        raise ValueError('Source pins are required')
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_absolute() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f'Source pin mismatch: {name}')


def preflight(manifest_path: Path, expected_hash: str) -> dict:
    """Validate all supplied evidence before creating output or a child process."""
    if digest(manifest_path) != expected_hash:
        raise ValueError('Manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('run_id') != RUN_ID
            or manifest.get('source_root') != str(ROOT) or manifest.get('output') != str(OUTPUT)
            or manifest.get('python') != PYTHON
            or manifest.get('dataset') != str(ROOT / 'data/locomo10.json')):
        raise ValueError('Unexpected fixed execution identity')
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError('Output must be fresh; no resume or overwrite')
    pins = manifest['files_sha256']
    check_pins(pins)
    required = {str(p) for p in ROOT.rglob('*.py') if 'tests' not in p.relative_to(ROOT).parts}
    required.update(str(ROOT / name) for name in ('README.md', 'pyproject.toml', 'data/locomo10.json'))
    if not required.issubset(pins):
        raise ValueError('Unpinned source or dataset')
    dataset = Path(manifest['dataset'])
    if digest(dataset) != DATASET_SHA256:
        raise ValueError('Not the original LoCoMo dataset')
    samples = json.loads(dataset.read_text(encoding='utf-8'))
    questions, counts = {}, {}
    for sample in samples:
        cid = sample['sample_id']
        if not isinstance(cid, str) or not cid or cid in counts:
            raise ValueError('Duplicate or invalid conversation identity')
        qas = sample['qa']
        if any(type(q.get('category')) is not int or q['category'] not in (1, 2, 3, 4, 5) for q in qas):
            raise ValueError('Unknown QA category')
        selected = [q for q in qas if q['category'] != 5]
        counts[cid] = len(selected)
        questions.update({(cid, i): q for i, q in enumerate(selected)})
    if len(counts) != 10 or len(questions) != 1540:
        raise ValueError('Expected ten conversations and 1540 category 1-4 QA')
    token_pin = manifest['token_preflight']
    token_path = Path(token_pin['path'])
    if not token_path.is_absolute() or digest(token_path) != token_pin['sha256']:
        raise ValueError('Token preflight hash mismatch')
    token = json.loads(token_path.read_text(encoding='utf-8'))
    fixed = {'schema_version': 1, 'dataset_sha256': DATASET_SHA256, 'model': 'Qwen/Qwen3.5-9B',
             'revision': REVISION, 'max_model_len': 49152, 'reader_max_tokens': 32,
             'qa_total': 1540, 'original_full_raw_omitted_qa': 453}
    if any(token.get(k) != v for k, v in fixed.items()) or token.get('all_fit') is not True:
        raise ValueError('Invalid pinned token preflight')
    seen, omitted = set(), set()
    for row in token['rows']:
        cid = row['conv_id']
        numbers = [row.get(k) for k in ('qa_count', 'raw_tokens', 'max_chat_prompt_tokens', 'max_total_tokens')]
        if (cid not in counts or cid in seen or any(type(n) is not int or n <= 0 for n in numbers)
                or row['qa_count'] != counts[cid]
                or row['max_total_tokens'] != row['max_chat_prompt_tokens'] + 32
                or row['max_total_tokens'] > 49152
                or row.get('original_full_raw_included') is not (row['raw_tokens'] < 39000)):
            raise ValueError('Invalid full-raw prompt coverage')
        seen.add(cid)
        if not row['original_full_raw_included']:
            omitted.add(cid)
    if (seen != set(counts) or omitted != {'conv-41', 'conv-43', 'conv-44'}
            or sum(counts[cid] for cid in omitted) != 453):
        raise ValueError('Incomplete token preflight or original guard mismatch')
    command = [PYTHON, '-u', str(ROOT / 'scripts/run_system_v15.py'), '--comparison', '--dataset', 'locomo',
               '--configs', 'full,no_adaptive,no_residual', '--budgets', '400,1600,4000', '--out', str(OUTPUT)]
    return {'manifest': manifest, 'questions': questions, 'command': command}


def audit_outputs(output: Path, questions: dict) -> dict:
    """Check the native 37-policy matrix and exact QA metadata, never rewrite it."""
    expected = {(cid, ordinal, config, policy) for cid, ordinal in questions
                for config, policies in POLICIES.items() for policy in policies}
    seen, count, raw_count, empty_count, issues, hashes = set(), 0, 0, 0, [], {}
    try:
        for name in ('items.csv', 'summary.csv', 'write_stats.csv'):
            path = output / name
            if path.stat().st_size == 0:
                raise ValueError(f'Empty native output: {name}')
            hashes[name] = digest(path)
        with (output / 'items.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                count += 1
                key = (row['conv_id'], int(row['question_ordinal']), row['config'], row['policy'])
                qa = questions.get(key[:2])
                if (key not in expected or key in seen or not isinstance(row['pred'], str) or qa is None
                        or row['question'] != qa['question'] or row['gold'] != str(qa.get('answer', ''))
                        or row['category'] != str(qa['category'])):
                    issues.append(f'Invalid or duplicate native output row {count}')
                seen.add(key)
                raw_count += row['policy'] == 'full_raw'
                empty_count += isinstance(row['pred'], str) and not row['pred'].strip()
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        issues.append(str(exc))
    if seen != expected:
        issues.append('Native reader matrix is incomplete')
    return {'complete': not issues, 'expected_rows': len(expected), 'rows': count,
            'full_raw_rows': raw_count, 'empty_predictions': empty_count, 'issues': issues, 'sha256': hashes}


def gpu_sample() -> dict | None:
    """Read one GPU; unavailable/ambiguous readings remain unknown."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,utilization.gpu,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=4)
        lines = result.stdout.strip().splitlines()
        if len(lines) != 1:
            return None
        values = [float(x.strip()) for x in lines[0].split(',')]
        if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values) or values[1] > 100:
            return None
        return dict(zip(('power_w', 'utilization_percent', 'memory_mib'), values))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def energy_wh(samples: list[dict]) -> float | None:
    """Trapezoidal sampled device energy, not exclusive experiment attribution."""
    if len(samples) < 2 or any(s['gpu'] is None for s in samples):
        return None
    return sum((b['elapsed_s'] - a['elapsed_s']) * (a['gpu']['power_w'] + b['gpu']['power_w']) / 7200
               for a, b in pairwise(samples))


def forward_signal(child: subprocess.Popen, signum: int) -> None:
    """Signal only the child process group created by this launcher."""
    try:
        if os.name == 'posix':
            os.killpg(child.pid, signum)
        else:
            child.send_signal(signum)
    except ProcessLookupError:
        pass


def run(manifest_path: Path, expected_hash: str) -> int:
    """Launch once, wait foreground, and retain failure/interrupt artifacts."""
    prepared = preflight(manifest_path, expected_hash)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started, started_utc = time.monotonic(), time.time()
    samples, errors, interrupted = [], [], []
    child, exit_code = None, None
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def on_signal(signum: int, _frame: object) -> None:
        interrupted.append(signum)
        if child is not None:
            forward_signal(child, signum)

    for sig in old_handlers:
        signal.signal(sig, on_signal)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
               VLLM_USE_FLASHINFER_SAMPLER='0', PYTHONUNBUFFERED='1')
    try:
        with (OUTPUT / 'benchmark.log').open('x', encoding='utf-8') as log, \
                (OUTPUT / 'gpu_samples.jsonl').open('x', encoding='utf-8') as journal:
            def sample() -> None:
                gpu = gpu_sample()
                value = {'elapsed_s': time.monotonic() - started, 'utc': time.time(), 'gpu': gpu}
                samples.append(value)
                journal.write(json.dumps(value) + '\n')
                journal.flush()
            sample()
            if interrupted:
                raise ValueError('Interrupted before child launch')
            child = subprocess.Popen(prepared['command'], cwd=str(ROOT), env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')
            if interrupted:
                forward_signal(child, interrupted[-1])
            while True:
                try:
                    exit_code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    sample()
            sample()
    except (OSError, ValueError) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        if child is not None and exit_code is None:
            forward_signal(child, signal.SIGTERM)
            exit_code = child.wait()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    wall = time.monotonic() - started
    try:
        check_pins(prepared['manifest']['files_sha256'])
        source_unchanged = True
    except (OSError, ValueError) as exc:
        source_unchanged = False
        errors.append(str(exc))
    outputs = audit_outputs(OUTPUT, prepared['questions'])
    inference_complete = exit_code == 0 and not errors and not interrupted and outputs['complete'] and source_unchanged
    receipt = {'schema_version': 1, 'run_id': RUN_ID, 'manifest_path': str(manifest_path),
        'manifest_sha256': expected_hash, 'command': prepared['command'], 'child_exit_code': exit_code,
        'signals_received': interrupted, 'errors': errors, 'source_unchanged': source_unchanged,
        'inference_complete': inference_complete, 'native_outputs': outputs, 'complete': False,
        'official_scoring_complete': False, 'usage_complete': None, 'exact_tokens': None,
        'runtime_receipts': str(OUTPUT / 'runtime'),
        'full_raw_policy': {'qa_total': 1540, 'original_guard_omitted_qa': 453,
            'comparison_guard_lifted': True, 'raw_input_truncated': False,
            'reason': 'Pinned chat-token preflight fits all original input plus native 32-token reader cap in 49152.'},
        'runtime': {'started_utc_epoch': started_utc, 'wall_seconds': wall,
            'hourly_rate_usd': HOURLY_RATE, 'estimated_rental_usd': wall / 3600 * HOURLY_RATE,
            'sampled_device_energy_wh': energy_wh(samples), 'gpu_sample_count': len(samples),
            'exclusive_gpu_energy_wh': None, 'cost_scope': 'Wall-window rental estimate, not an invoice.',
            'energy_scope': 'Sampled whole-device window; no exclusive allocation or overlap subtraction.'},
        'limitations': ['Closed runtime token/embedding receipt audit and secondary official scoring remain required.',
                        'Native generation length finishes are preserved by runtime, never relabeled stop.']}
    with (OUTPUT / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return 0 if inference_complete else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args(argv)
    return run(args.manifest, args.expected_manifest_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
