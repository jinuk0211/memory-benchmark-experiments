"""Verify one method's final-answer coverage by local evaluation timing sidecars."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


DIGEST_FORMAT = 'UTF-8 canonical JSON, sort_keys=True, ensure_ascii=False, no newline'


def canonical_digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(body).hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse(text: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject)


def read(path: Path) -> Any:
    return parse(path.read_text(encoding='utf-8'))


def aggregate_timing(predictions: Path, sidecar_root: Path, method: str) -> dict:
    """Return a mean only for complete, unambiguous coverage of supplied final rows.

    Failed calls are separate observed cost. Incomplete calls have unknown elapsed
    time. All successes are retained; no retry is selected using timestamps.
    """
    predictions, sidecar_root = Path(predictions), Path(sidecar_root)
    raw_predictions = predictions.read_bytes()
    rows = [parse(line) for line in raw_predictions.decode('utf-8').splitlines() if line.strip()]
    if not rows or not isinstance(method, str) or not method:
        raise ValueError('A nonempty method and final-answer population are required')
    by_id = {}
    for row in rows:
        qid = row.get('question_id')
        if (not isinstance(qid, str) or not qid or qid in by_id or row.get('method') != method
                or not isinstance(row.get('conversation_id'), str) or not row['conversation_id']
                or not isinstance(row.get('prediction'), str) or row.get('hypothesis') != row['prediction']):
            raise ValueError('Final rows contain duplicate/invalid IDs, method, conversation, or answer fields')
        by_id[qid] = row
    expected_by_conversation = {}
    for row in rows:
        expected_by_conversation.setdefault(row['conversation_id'], set()).add(row['question_id'])
    issues, attempts = [], []
    claimed, covered = Counter(), Counter()
    observed_success_seconds = observed_failure_seconds = matched_seconds = 0.0
    common_metadata = None
    ignored = 0
    for path in sidecar_root.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.tmp'):
            relative = path.relative_to(sidecar_root)
            if len(relative.parts) != 3 or path.name not in ('started.json', 'completed.json', 'error.json',
                                                            'started.tmp', 'completed.tmp', 'error.tmp'):
                issues.append({'directory': relative.as_posix(), 'reason': 'Orphan or unexpected timing receipt path'})
    directories = sorted(path for path in sidecar_root.glob('*/*') if path.is_dir())
    for directory in directories:
        relative = directory.relative_to(sidecar_root).as_posix()
        attempt = {'directory': relative, 'status': 'invalid', 'elapsed_seconds': None}
        try:
            started_path = directory / 'started.json'
            if not started_path.is_file():
                readable_methods = []
                for name in ('completed.json', 'error.json'):
                    if (directory / name).is_file():
                        readable_methods.append(read(directory / name).get('method'))
                if readable_methods and all(value != method and isinstance(value, str) for value in readable_methods):
                    ignored += 1
                    continue
                raise ValueError('Missing started receipt; terminal or partial attempt cannot be linked')
            started = read(started_path)
            for name in ('completed.json', 'error.json'):
                if (directory / name).is_file():
                    if read(directory / name).get('method') != started.get('method'):
                        raise ValueError('Started/terminal method mismatch; selected-method evidence cannot be ignored')
            if started.get('method') != method:
                if isinstance(started.get('method'), str):
                    ignored += 1
                    continue
                raise ValueError('Started receipt has no valid method')
            if (not directory.resolve().is_relative_to(sidecar_root.resolve())
                    or not re.fullmatch('[0-9a-f]{32}', directory.parent.name)
                    or not re.fullmatch('[0-9a-f]{32}', directory.name)
                    or started.get('session_id') != directory.parent.name
                    or started.get('invocation_id') != directory.name
                    or started.get('schema_version') != 1 or started.get('status') != 'started'
                    or started.get('clock') != 'time.perf_counter_ns'
                    or not isinstance(started.get('metadata'), dict)
                    or not isinstance(started.get('runtime'), dict)
                    or started.get('dataset') not in ('locomo', 'longmemeval')
                    or started.get('conversation_id') not in expected_by_conversation):
                raise ValueError('Invalid started receipt identity, metadata, clock, or conversation')
            metadata, runtime = started['metadata'], started['runtime']
            source_hashes = ('runner_sha256', 'evaluator_source_sha256', 'evaluator_function_sha256',
                             'launcher_sha256', 'recorder_sha256')
            if (any(not isinstance(metadata.get(key), str) or not re.fullmatch('[0-9a-f]{64}', metadata[key])
                    for key in source_hashes)
                    or any(not isinstance(metadata.get(key), str) or not metadata[key].strip()
                           for key in ('runner', 'evaluator_source'))
                    or any(not isinstance(runtime.get(key), str) or not runtime[key].strip()
                           for key in ('model', 'embed_model', 'out'))
                    or type(runtime.get('seed')) is not int or runtime['seed'] < 0):
                raise ValueError('Missing or invalid launcher/source hashes or runtime model/embedding/seed metadata')
            identity = {key: started.get(key) for key in
                        ('method', 'dataset', 'metadata', 'runtime', 'cache_lifecycle', 'timing_scope', 'clock')}
            if common_metadata is None:
                common_metadata = identity
            elif common_metadata != identity:
                issues.append({'directory': relative,
                               'reason': 'Inconsistent runtime/source/launcher/cache metadata across selected attempts'})
            attempt['conversation_id'] = started['conversation_id']
            terminals = [directory / name for name in ('completed.json', 'error.json') if (directory / name).is_file()]
            partials = list(directory.glob('*.tmp'))
            if not terminals:
                attempt['status'] = 'incomplete'
                raise ValueError('Started-only or partial attempt has unknown elapsed time')
            if len(terminals) != 1 or partials:
                raise ValueError('Conflicting or partial terminal records; attempt is not unambiguous')
            terminal_path = terminals[0]
            terminal = read(terminal_path)
            wanted_status = 'complete' if terminal_path.name == 'completed.json' else 'error'
            if (terminal.get('status') != wanted_status
                    or terminal.get('started_sha256') != hashlib.sha256(started_path.read_bytes()).hexdigest()
                    or any(key not in terminal or terminal[key] != value for key, value in started.items() if key != 'status')):
                raise ValueError('Terminal/started digest linkage or metadata mismatch')
            elapsed = terminal.get('elapsed_seconds')
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Elapsed seconds must be finite and nonnegative')
            attempt['elapsed_seconds'] = elapsed
            if wanted_status == 'error':
                if (not isinstance(terminal.get('exception_type'), str) or not terminal['exception_type']
                        or any(key in terminal for key in ('question_ids', 'question_count', 'returned_rows_sha256'))):
                    raise ValueError('Invalid error receipt; it cannot claim returned answer rows')
                attempt.update(status='error', exception_type=terminal['exception_type'])
                observed_failure_seconds += elapsed
            else:
                attempt['status'] = 'complete_unmatched'
                observed_success_seconds += elapsed
                ids = terminal.get('question_ids')
                if (not isinstance(ids, list) or not ids or any(not isinstance(qid, str) for qid in ids)
                        or len(ids) != len(set(ids)) or type(terminal.get('question_count')) is not int
                        or terminal['question_count'] != len(ids)):
                    raise ValueError('Invalid or duplicate returned question IDs/count')
                claimed.update(ids)
                attempt['question_ids'] = ids
                expected = expected_by_conversation[started['conversation_id']]
                if set(ids) != expected:
                    raise ValueError('Completed call does not cover exactly its final conversation rows')
                reconstructed = [by_id[qid] for qid in ids]
                if (terminal.get('returned_rows_digest_format') != DIGEST_FORMAT
                        or terminal.get('returned_rows_sha256') != canonical_digest(reconstructed)):
                    raise ValueError('Returned-row digest does not match original final JSONL rows in receipt order')
                attempt['status'] = 'complete_matched'
                covered.update(ids)
                matched_seconds += elapsed
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            issues.append({'directory': relative, 'reason': str(exc)})
        attempts.append(attempt)
    if not attempts:
        issues.append({'directory': '.', 'reason': 'No timing receipts for the selected method; elapsed time is unknown'})
    missing = sorted(set(by_id) - set(covered))
    overlap = sorted(qid for qid, count in claimed.items() if count > 1)
    if missing:
        issues.append({'directory': '.', 'reason': 'Missing completed timing coverage', 'question_ids': missing})
    if overlap:
        issues.append({'directory': '.', 'reason': 'Overlapping successful attempts; no retry was selected', 'question_ids': overlap})
    complete = not issues and set(covered) == set(by_id) and all(count == 1 for count in covered.values())
    unknown = sum(attempt['elapsed_seconds'] is None for attempt in attempts)
    return {
        'schema_version': 1, 'method': method, 'questions_in_final_file': len(rows),
        'final_answers_sha256': hashlib.sha256(raw_predictions).hexdigest(),
        'metadata': common_metadata, 'coverage_complete_and_unambiguous': complete,
        'verified_final_evaluation_seconds': matched_seconds if complete else None,
        'mean_final_evaluation_seconds_per_question': matched_seconds / len(rows) if complete else None,
        'observed_successful_attempt_seconds': observed_success_seconds,
        'observed_failed_attempt_seconds': observed_failure_seconds,
        'observed_all_terminal_attempt_seconds_lower_bound': observed_success_seconds + observed_failure_seconds,
        'attempts_with_unknown_elapsed': unknown, 'ignored_other_method_attempts': ignored,
        'attempts': attempts, 'issues': issues,
        'scope': [
            'Population is exactly the supplied final-answer JSONL; full benchmark completeness is not certified here.',
            'Mean = sum of uniquely matched completed evaluate_sample call seconds / final question count.',
            'The interval includes original embedding, retrieval, packing, generation and row assembly, and their existing cache reads.',
            'It excludes initialization, outer item-cache skip checks, caller item/final-file persistence, and timing sidecar I/O; it is not whole-job E2E latency.',
            'Returned rows are reconstructed in each receipt question-ID order, including original timing fields; no recomputed or normalized F1 rows are substituted.',
            'Failed calls are separate observed cost. Missing/partial calls have unknown elapsed time. Successful retries are never chosen by mtime or UUID.',
            'Observed costs depend on the retained cache lifecycle; this is not a cold-cache efficiency comparison.'],
    }


def markdown(report: dict) -> str:
    mean = report['mean_final_evaluation_seconds_per_question']
    rendered_mean = 'unknown' if mean is None else f'{mean:.6f}'
    lines = ['# Evaluation-function timing', '', f"Method: {report['method']}",
             f"Final-answer questions: {report['questions_in_final_file']}",
             f'Mean successful attributable evaluation-function seconds/question: {rendered_mean}',
             f"Observed successful-attempt seconds: {report['observed_successful_attempt_seconds']:.6f}",
             f"Observed failed-attempt seconds: {report['observed_failed_attempt_seconds']:.6f}",
             f"Attempts with unknown elapsed time: {report['attempts_with_unknown_elapsed']}", '',
             *['- ' + item for item in report['scope']], '', f"Issues: {len(report['issues'])}",
             *['- ' + item['directory'] + ': ' + item['reason'] for item in report['issues']]]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--sidecar-root', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--out', type=Path, required=True, help='New report directory outside sidecar evidence')
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.sidecar_root.resolve()):
        parser.error('Output must be outside the read-only timing sidecars')
    report = aggregate_timing(args.predictions, args.sidecar_root, args.method)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'EVALUATION_TIMING.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    (args.out / 'EVALUATION_TIMING.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps({'complete': report['coverage_complete_and_unambiguous'],
                      'mean_seconds_per_question': report['mean_final_evaluation_seconds_per_question']}))


if __name__ == '__main__':
    main()
