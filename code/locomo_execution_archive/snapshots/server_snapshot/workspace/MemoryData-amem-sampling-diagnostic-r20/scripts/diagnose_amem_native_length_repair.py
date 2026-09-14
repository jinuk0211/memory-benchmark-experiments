"""One metered diagnostic of native A-MEM correction; never a benchmark result."""
import argparse
import copy
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.parse import quote

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.metered_openai_proxy import MeteredProxy, metadata_id
from scripts.response_delivery_audit import audit_delivery
from scripts.score_locomo_comparison import read_jsonl, summarize_usage, write_report


def _native():
    path = Path(__file__).resolve().parents[1] / 'methods/a_mem/source/a_mem/evolution_repair.py'
    source = path.read_bytes()
    module = ModuleType('_diagnostic_native_evolution')
    exec(compile(source, str(path), 'exec'), module.__dict__)  # noqa: S102 - fixed local native helper
    module.os = SimpleNamespace(getenv={'BASELINE_STRICT_COMPARISON': '1'}.get)
    return module, hashlib.sha256(source).hexdigest()


def _format(count):
    string = {'type': 'string'}
    properties = {
        'should_evolve': {'type': 'boolean'},
        'actions': {'type': 'array', 'maxItems': 2,
                    'items': {'type': 'string', 'enum': ['strengthen', 'update_neighbor']}},
        'suggested_connections': {'type': 'array', 'maxItems': count, 'items': {'type': 'integer'}},
        'new_context_neighborhood': {'type': 'array', 'minItems': count, 'maxItems': count, 'items': string},
        'tags_to_update': {'type': 'array', 'items': string},
        'new_tags_neighborhood': {'type': 'array', 'minItems': count, 'maxItems': count,
                                  'items': {'type': 'array', 'items': string}},
    }
    return {'type': 'json_schema', 'json_schema': {'name': 'response', 'strict': True, 'schema': {
        'type': 'object', 'properties': properties, 'additionalProperties': False,
        'required': ['should_evolve', 'actions', 'suggested_connections', 'tags_to_update',
                     'new_context_neighborhood', 'new_tags_neighborhood']}}}


class _CorrectionReady(Exception):
    """Stop the native callback at the first corrected prompt, without inference."""


def build_correction(capture: dict) -> dict:
    """Reuse native correction on a known saved length failure; make no model call."""
    try:
        request, response = capture['request'], capture['response']
        messages, choices = request['messages'], response['choices']
        if (request['model'] != 'Qwen/Qwen3.5-9B' or request.get('stream', False) is not False
                or not metadata_id(capture.get('request_id')) or not metadata_id(response.get('id'))
                or response.get('model') != request['model']
                or not isinstance(messages, list) or len(messages) != 2
                or [m['role'] for m in messages] != ['system', 'user']
                or messages[0]['content'] != 'You must respond with a JSON object.'
                or not isinstance(messages[1]['content'], str)
                or not messages[1]['content'].lstrip().startswith(
                    'You are an AI memory evolution agent responsible for managing and evolving a knowledge base.')
                or not isinstance(choices, list) or len(choices) != 1
                or choices[0]['finish_reason'] != 'length'):
            raise ValueError('Capture is not the native nonstream Qwen A-MEM length failure')
        prompt = messages[1]['content']
        indices = [int(index) for index in re.findall(r'(?m)^\s*memory index:(\d+)\t', prompt)]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('Missing or duplicate ordered global neighbor indices')
        if json.dumps(request['response_format'], sort_keys=True) != json.dumps(_format(len(indices)), sort_keys=True):
            raise ValueError('Capture schema differs from the exact native six-field neighborhood contract')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed saved A-MEM capture') from error
    native, native_hash = _native()
    saved_failure = RuntimeError('Reused saved length failure, not a new model call')
    saved_failure.status_code = 502
    saved_failure.body = {'type': 'comparison_incomplete_output',
                          'finish_reasons': [choices[0]['finish_reason']]}
    calls = []

    def completion(current_prompt, **_kwargs):
        calls.append(current_prompt)
        if len(calls) == 1:
            raise saved_failure
        raise _CorrectionReady(current_prompt)

    try:
        native.evolution_completion(completion, indices, prompt)
    except _CorrectionReady as ready:
        corrected = ready.args[0]
    else:
        raise RuntimeError('Native helper did not yield its first correction')
    if len(calls) != 2 or calls[0] != prompt or not corrected.startswith(prompt + '\n\n'):
        raise RuntimeError('Native correction did not preserve the exact original user prefix')
    result = copy.deepcopy(request)
    result['messages'][1]['content'] = corrected
    return {'request': result, 'proof': {
        'indices': indices, 'native_helper_sha256': native_hash,
        'saved_request_id': capture.get('request_id'), 'saved_response_id': response.get('id'),
        'saved_finish_reasons': saved_failure.body['finish_reasons'],
        'original_user_prefix_preserved': True, 'seeded_failure_reused': True,
        'seeded_failure_new_calls': 0}}


def build_diagnostic_request(capture: dict, sampling_profile: str = 'original') -> dict:
    """Apply an explicit diagnostic-only profile without changing the native task."""
    if sampling_profile not in ('original', 'qwen35-instruct'):
        raise ValueError('Unknown diagnostic sampling profile')
    result = build_correction(capture)
    request, proof = result['request'], result['proof']
    settings = {}
    if sampling_profile == 'qwen35-instruct':
        if ((request.get('chat_template_kwargs') or {}).get('enable_thinking') is not False
                or request.get('enable_thinking', False) is not False):
            raise ValueError('Instruct profile requires captured non-thinking mode')
        settings = {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0,
                    'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'seed': 0}
        proof['sampling_source'] = (
            'https://huggingface.co/Qwen/Qwen3.5-9B/blob/'
            'c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md#best-practices')
        proof['sampling_note'] = (
            'Six official non-thinking general-task parameters plus preselected diagnostic seed 0. '
            'Not a performance guarantee, full benchmark, or unchanged comparison protocol. '
            'Previously absent keys are not evidence of effective server defaults.')
    proof['sampling_profile'] = sampling_profile
    proof['protocol_changed_from_original'] = bool(settings)
    proof['sampling_changes'] = {
        key: {'previous_present': key in request, 'previous': request.get(key), 'effective': value}
        for key, value in settings.items()}
    request.update(settings)
    return result


def run_diagnostic(payload: Path, output: Path, expected_payload_sha256: str, run_id: str,
                   sampling_profile: str = 'original') -> dict:
    """One direct HTTP call via an ephemeral loopback meter; retain failed evidence."""
    payload, output = Path(payload), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Diagnostic output already exists')
    if not metadata_id(run_id):
        raise ValueError('An explicit valid diagnostic run ID is required')
    raw = payload.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_payload_sha256:
        raise ValueError('Saved payload SHA256 mismatch')
    correction = build_diagnostic_request(json.loads(raw), sampling_profile)
    native, native_hash = _native()
    if native_hash != correction['proof']['native_helper_sha256']:
        raise ValueError('Native helper changed during preflight')
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / 'request.json', correction['request'])
    write_report(output / 'correction_proof.json', correction['proof'])
    started = time.monotonic()
    receipt = {'diagnostic_only': True, 'full_benchmark_complete': False, 'passed': False,
               'run_id': run_id, 'payload_sha256': expected_payload_sha256,
               'seeded_failure_new_calls': 0, 'gpu_energy_wh': None,
               'runtime_note': 'Runtime overlaps SimpleMem; GPU energy is unmeasured, not zero.',
               'http_status': None, 'error': None, 'sampling_profile': sampling_profile,
               'protocol_changed_from_original': correction['proof']['protocol_changed_from_original']}
    try:
        with MeteredProxy(('127.0.0.1', 0), 'http://127.0.0.1:18080/v1', output / 'usage.jsonl',
                          run_id, 'a_mem', timeout=1200, comparison_policy=False) as server:
            thread = threading.Thread(target=server.serve_forever,
                                      kwargs={'poll_interval': 0.05}, daemon=True)
            thread.start()
            try:
                route = (f'http://127.0.0.1:{server.server_port}/meter/{quote(run_id, safe="")}'
                         '/a_mem/memory_add/conv-26/-/v1/chat/completions')
                with httpx.Client(timeout=1230, trust_env=False) as client:
                    response = client.post(route, json=correction['request'])
                    receipt['http_status'] = response.status_code
                    receipt['response_headers'] = dict(response.headers)
                    (output / 'response.body').write_bytes(response.content)
                    response.raise_for_status()
                    data = response.json()
                    if (len(data.get('choices', [])) != 1
                            or data['choices'][0].get('finish_reason') != 'stop'):
                        raise ValueError('Diagnostic did not finish with exactly one stop response')
                    native._validate_response(data['choices'][0]['message']['content'], correction['proof']['indices'])
            finally:
                server.shutdown()
                thread.join()
        # MeteredProxy.__exit__ drains all in-flight handlers and closes the journal.
    except Exception as error:  # noqa: BLE001 - retain a failed diagnostic receipt, never retry
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
    journal, journal_raw = output / 'usage.jsonl', None
    try:
        journal_raw = journal.read_bytes() if journal.exists() else None
        rows = read_jsonl(journal) if journal_raw is not None else []
    except (ValueError, OSError) as error:
        rows = []
        receipt['journal_error'] = {'type': type(error).__name__, 'message': str(error)}
    receipt['usage'] = summarize_usage(rows, True)
    receipt['delivery'] = audit_delivery(rows)
    receipt['actual_requests'] = len(rows) if rows else None
    receipt['exact_total_tokens'] = (receipt['usage']['reported_tokens']['total_tokens']
                                      if len(rows) == 1 and receipt['usage']['complete'] else None)
    receipt['passed'] = (receipt['error'] is None and len(rows) == 1
                         and receipt['usage']['complete'] and receipt['delivery']['complete']
                         and all(row.get('run_id') == run_id and row.get('method') == 'a_mem'
                                 and row.get('phase') == 'memory_add' and row.get('sample_id') == 'conv-26'
                                 for row in rows))
    receipt['wall_seconds'] = time.monotonic() - started
    receipt['usage_sha256'] = hashlib.sha256(journal_raw).hexdigest() if journal_raw is not None else None
    write_report(output / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-payload-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--sampling-profile', choices=['original', 'qwen35-instruct'], default='original')
    args = parser.parse_args()
    result = run_diagnostic(args.payload, args.output, args.expected_payload_sha256, args.run_id,
                            args.sampling_profile)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
