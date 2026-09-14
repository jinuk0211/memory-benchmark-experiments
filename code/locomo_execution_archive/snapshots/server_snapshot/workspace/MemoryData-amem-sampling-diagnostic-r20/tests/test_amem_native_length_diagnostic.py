"""Saved native length failure seeds correction without inventing model work."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def capture():
    indices = [1, 0, 7, 46, 48]
    string = {'type': 'string'}
    properties = {
        'should_evolve': {'type': 'boolean'},
        'actions': {'type': 'array', 'maxItems': 2,
                    'items': {'type': 'string', 'enum': ['strengthen', 'update_neighbor']}},
        'suggested_connections': {'type': 'array', 'maxItems': 5,
                                  'items': {'type': 'integer'}},
        'new_context_neighborhood': {'type': 'array', 'minItems': 5, 'maxItems': 5,
                                     'items': string},
        'tags_to_update': {'type': 'array', 'items': string},
        'new_tags_neighborhood': {'type': 'array', 'minItems': 5, 'maxItems': 5,
                                  'items': {'type': 'array', 'items': string}},
    }
    required = ['should_evolve', 'actions', 'suggested_connections', 'tags_to_update',
                'new_context_neighborhood', 'new_tags_neighborhood']
    prompt = ('\n You are an AI memory evolution agent responsible for managing and evolving '
              'a knowledge base. Original target and all repeated facts.\n' + ''.join(
                  f' memory index:{i}\t memory content: original {i}, original {i}\n'
                  for i in indices))
    return {'request_id': 'saved-request', 'request': {
        'model': 'Qwen/Qwen3.5-9B', 'temperature': 0, 'enable_thinking': False,
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'system', 'content': 'You must respond with a JSON object.'},
                     {'role': 'user', 'content': prompt}],
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'response', 'strict': True, 'schema': {'type': 'object',
            'properties': properties, 'required': required, 'additionalProperties': False}}},
    }, 'response': {'id': 'saved-response', 'model': 'Qwen/Qwen3.5-9B', 'choices': [
        {'finish_reason': 'length', 'message': {'role': 'assistant', 'content': '{"partial":'}}]}}


def target():
    return importlib.import_module('scripts.diagnose_amem_native_length_repair')


def test_native_builder_changes_only_user_suffix_and_preserves_environment(capture):
    before, environment = copy.deepcopy(capture), dict(os.environ)
    result = target().build_correction(capture)
    request, proof = result['request'], result['proof']
    original_prompt = capture['request']['messages'][1]['content']
    assert request['messages'][1]['content'].startswith(original_prompt + '\n\nCorrection required:')
    assert 'global indices [1, 0, 7, 46, 48]' in request['messages'][1]['content']
    assert proof['indices'] == [1, 0, 7, 46, 48]
    assert proof['seeded_failure_reused'] is True and proof['seeded_failure_new_calls'] == 0
    assert proof['original_user_prefix_preserved'] is True
    native = Path(__file__).resolve().parents[1] / 'methods/a_mem/source/a_mem/evolution_repair.py'
    assert proof['native_helper_sha256'] == hashlib.sha256(native.read_bytes()).hexdigest()
    request['messages'][1]['content'] = original_prompt
    assert request == capture['request'] and capture == before and dict(os.environ) == environment


@pytest.mark.parametrize('profile', ['original', 'qwen35-instruct'])
def test_sampling_profile_preserves_all_input_schema_and_records_changes(capture, profile):
    module = target()
    before = copy.deepcopy(capture)
    native = module.build_correction(capture)
    actual = module.build_diagnostic_request(capture, profile)
    settings = ({'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0,
                 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'seed': 0}
                if profile == 'qwen35-instruct' else {})
    expected = copy.deepcopy(native['request'])
    expected.update(settings)
    assert actual['request'] == expected and capture == before
    assert actual['proof']['sampling_profile'] == profile
    assert actual['proof']['sampling_changes'] == {
        key: {'previous_present': key in native['request'],
              'previous': native['request'].get(key), 'effective': value}
        for key, value in settings.items()}
    assert actual['proof']['protocol_changed_from_original'] is bool(settings)
    if settings:
        assert 'c202236235762e1c871ad0ccb60c8ee5ba337b9a' in actual['proof']['sampling_source']
    assert actual['proof']['original_user_prefix_preserved'] is True
    assert actual['proof']['seeded_failure_new_calls'] == 0


def test_unknown_sampling_profile_rejected_before_model_or_artifact(tmp_path, capture, monkeypatch):
    module = target()
    monkeypatch.setattr(module, 'MeteredProxy', lambda *_a, **_k: pytest.fail('No server'))
    path, digest = payload_file(tmp_path, capture)
    with pytest.raises(ValueError, match='sampling profile'):
        module.run_diagnostic(path, tmp_path / 'out', digest, 'diag', sampling_profile='guess')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('mode', [None, True, 'false'])
def test_instruct_profile_rejects_missing_or_nonfalse_thinking_flag(capture, mode):
    capture['request']['chat_template_kwargs'] = {} if mode is None else {'enable_thinking': mode}
    with pytest.raises(ValueError, match='non-thinking'):
        target().build_diagnostic_request(capture, 'qwen35-instruct')


def test_instruct_profile_rejects_conflicting_thinking_flag(capture):
    capture['request']['enable_thinking'] = True
    with pytest.raises(ValueError, match='non-thinking'):
        target().build_diagnostic_request(capture, 'qwen35-instruct')


@pytest.mark.parametrize('case', ['model', 'stream', 'messages', 'roles', 'system', 'prompt',
                                 'finish', 'choices', 'duplicate_indices', 'missing_indices',
                                 'bounds', 'extra_field', 'field_type', 'required', 'strict',
                                 'missing_request', 'not_object', 'missing_request_id',
                                 'missing_response_id', 'response_model'])
def test_invalid_capture_rejected_before_model_io(capture, monkeypatch, case):
    module = target()
    monkeypatch.setattr(module, 'MeteredProxy', lambda *_a, **_k: pytest.fail('No server'))
    request = capture['request']
    if case == 'model':
        request['model'] = 'other-model'
    elif case == 'stream':
        request['stream'] = True
    elif case == 'messages':
        request['messages'].append(copy.deepcopy(request['messages'][1]))
    elif case == 'roles':
        request['messages'][1]['role'] = 'system'
    elif case == 'system':
        request['messages'][0]['content'] = 'Different system'
    elif case == 'prompt':
        request['messages'][1]['content'] = 'Unrelated task'
    elif case == 'finish':
        capture['response']['choices'][0]['finish_reason'] = 'stop'
    elif case == 'choices':
        capture['response']['choices'].append(copy.deepcopy(capture['response']['choices'][0]))
    elif case == 'duplicate_indices':
        request['messages'][1]['content'] += 'memory index:1\t duplicate\n'
    elif case == 'missing_indices':
        request['messages'][1]['content'] = request['messages'][1]['content'].replace('memory index:', 'index:')
    elif case == 'bounds':
        request['response_format']['json_schema']['schema']['properties']['new_tags_neighborhood']['minItems'] = 4
    elif case == 'extra_field':
        request['response_format']['json_schema']['schema']['properties']['invented'] = {'type': 'string'}
    elif case == 'field_type':
        request['response_format']['json_schema']['schema']['properties']['should_evolve']['type'] = 'string'
    elif case == 'required':
        request['response_format']['json_schema']['schema']['required'].pop()
    elif case == 'strict':
        request['response_format']['json_schema']['strict'] = False
    elif case == 'missing_request':
        capture.pop('request')
    elif case == 'missing_request_id':
        capture.pop('request_id')
    elif case == 'missing_response_id':
        capture['response'].pop('id')
    elif case == 'response_model':
        capture['response']['model'] = 'foreign'
    else:
        capture = []
    with pytest.raises(ValueError):
        module.build_correction(capture)


def payload_file(tmp_path, capture):
    path = tmp_path / 'saved.json'
    raw = json.dumps(capture, ensure_ascii=False).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('case', ['hash', 'existing_output', 'run_id', 'bad_capture'])
def test_runner_preflight_blocks_before_any_server_or_new_artifact(tmp_path, capture, monkeypatch, case):
    module = target()
    monkeypatch.setattr(module, 'MeteredProxy', lambda *_a, **_k: pytest.fail('No server'))
    if case == 'bad_capture':
        capture['request']['model'] = 'foreign'
    path, digest = payload_file(tmp_path, capture)
    output = tmp_path / 'diagnostic'
    if case == 'existing_output':
        output.mkdir()
        (output / 'original.txt').write_text('preserve')
    with pytest.raises((ValueError, FileExistsError)):
        module.run_diagnostic(path, output, '0' * 64 if case == 'hash' else digest,
                              '' if case == 'run_id' else 'diagnostic-run')
    if case == 'existing_output':
        assert (output / 'original.txt').read_text() == 'preserve'
        assert len(list(output.iterdir())) == 1
    else:
        assert not output.exists()


@pytest.mark.parametrize('profile', ['original', 'qwen35-instruct'])
@pytest.mark.parametrize('case', ['valid', 'length', 'invalid_native', 'http500', 'invalid_json',
                                 'missing_usage', 'transport_timeout', 'journal_error', 'no_journal_rows'])
def test_runner_exactly_one_metered_mocked_call_and_retains_failures(tmp_path, capture, monkeypatch, case, profile):
    module = target()
    original_server = module.MeteredProxy
    calls = []
    native = {'should_evolve': True, 'actions': ['strengthen', 'update_neighbor'],
              'suggested_connections': [1, 48], 'tags_to_update': ['same', 'same'],
              'new_context_neighborhood': ['context'] * 5,
              'new_tags_neighborhood': [['same', 'same'] for _ in range(5)]}
    if case == 'invalid_native':
        native['suggested_connections'] = [999]
    response = {'id': 'new-response', 'model': 'Qwen/Qwen3.5-9B', 'choices': [
        {'finish_reason': 'length' if case == 'length' else 'stop',
         'message': {'content': json.dumps(native), 'role': 'assistant'}}],
        'usage': None if case == 'missing_usage' else {
            'prompt_tokens': 100, 'completion_tokens': 40, 'total_tokens': 140}}
    raw = b'invalid upstream body' if case == 'invalid_json' else json.dumps(response).encode()

    def upstream(request):
        calls.append((str(request.url), json.loads(request.content)))
        if case == 'transport_timeout':
            raise httpx.ReadTimeout('Synthetic unknown-usage timeout', request=request)
        return httpx.Response(500 if case == 'http500' else 200, content=raw,
                              headers={'content-type': 'application/json'})

    def metered(*args, **kwargs):
        assert args[0] == ('127.0.0.1', 0) and args[1] == 'http://127.0.0.1:18080/v1'
        assert kwargs['comparison_policy'] is False
        server = original_server(*args, **kwargs)
        server.client.close()
        server.client = httpx.Client(transport=httpx.MockTransport(upstream))
        if case == 'journal_error':
            original_record = server.record

            def record(row):
                original_record(row)
                server.journal.write('{"partial":')
                server.journal.flush()

            server.record = record
        elif case == 'no_journal_rows':
            server.record = lambda _row: None
        return server

    monkeypatch.setattr(module, 'MeteredProxy', metered)
    path, digest = payload_file(tmp_path, capture)
    output = tmp_path / 'diagnostic'
    receipt = module.run_diagnostic(path, output, digest, 'diagnostic-run', sampling_profile=profile)
    assert len(calls) == 1 and calls[0][0] == 'http://127.0.0.1:18080/v1/chat/completions'
    assert calls[0][1] == module.build_diagnostic_request(capture, profile)['request']
    assert receipt['sampling_profile'] == profile
    assert receipt['protocol_changed_from_original'] is (profile != 'original')
    assert json.loads((output / 'correction_proof.json').read_text())['sampling_profile'] == profile
    if case == 'transport_timeout':
        assert json.loads((output / 'response.body').read_bytes())['error']['message'] == 'Upstream transport error'
    else:
        assert (output / 'response.body').read_bytes() == raw
    assert json.loads((output / 'request.json').read_text()) == calls[0][1]
    assert receipt == json.loads((output / 'receipt.json').read_text())
    if case in ('journal_error', 'no_journal_rows'):
        journal = (output / 'usage.jsonl').read_bytes()
        assert journal.endswith(b'{"partial":') if case == 'journal_error' else journal == b''
        assert receipt['passed'] is False and receipt['exact_total_tokens'] is None
        assert receipt['usage_sha256'] == hashlib.sha256(journal).hexdigest()
        return
    rows = [json.loads(line) for line in (output / 'usage.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['run_id'] == 'diagnostic-run'
    assert rows[0]['method'] == 'a_mem' and rows[0]['phase'] == 'memory_add'
    assert rows[0]['sample_id'] == 'conv-26'
    assert receipt['passed'] is (case == 'valid')
    assert receipt['diagnostic_only'] is True and receipt['full_benchmark_complete'] is False
    assert receipt['gpu_energy_wh'] is None and 'SimpleMem' in receipt['runtime_note']
    assert receipt['seeded_failure_new_calls'] == 0
    assert receipt['exact_total_tokens'] == (None if case in ('missing_usage', 'invalid_json', 'transport_timeout') else 140)
