"""r26 is opt-in original-policy code preparation; tests never call a GPU/service."""
import importlib
import json
from contextlib import contextmanager

import pytest
from test_amem_original_policy_audit import POLICY, RUN, native, row


def save(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(json.dumps(item) for item in value) + '\n') if lines else json.dumps(value), encoding='utf-8')


def target():
    return importlib.import_module('scripts.amem_original_policy_recovery_r26')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    module = target()
    full, probe, prior = [tmp_path / name for name in ('full', 'probe', 'prior')]
    previous = tuple(tmp_path / f'previous-{i}' for i in range(6))
    for directory in (prior, *previous):
        directory.mkdir()
        save(directory / 'retained.json', {'passed': False})
    save(prior / 'parallel_status.json', {'active_contexts': [], 'failed': True, 'events': [
        *[{'event': 'start', 'context': c} for c in range(10)],
        *[{'event': 'end', 'context': c, 'exit_code': 1} for c in range(10)]]})
    agent = tmp_path / 'agent.json'
    save(agent, {'agent_name': 'Agentic_memory_a_mem', 'model': 'Qwen/Qwen3.5-9B', 'temperature': 0.0,
        'a_mem_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2', 'qwen3_disable_thinking': True})
    plan = {'run_id': RUN, 'amem_failure_policy': POLICY, 'output': str(full), 'dtype': 'float16',
        'model': 'Qwen/Qwen3.5-9B', 'model_revision': 'same-revision', 'embedding_path': 'same-minilm',
        'server_python': 'server', 'dataset': 'original1540', 'context_workers': 2, 'file_sha256': {},
        'methods': [{'method': name, 'agent_config': str(agent)} for name in module.queue.METHODS],
        'skip_methods': {name: 'Prior result preserved; r26 is A-MEM only' for name in module.queue.METHODS if name != 'a_mem'}}
    plan_path = tmp_path / 'plan.json'
    save(plan_path, plan)
    for name, value in (('PLAN_PATH', plan_path), ('FULL_OUTPUT', full), ('PROBE_OUTPUT', probe),
                        ('PRIOR_OUTPUT', prior), ('OTHER_PRIOR_OUTPUTS', previous)):
        monkeypatch.setattr(module, name, value)
    calls, count = [], {}

    def state(name):
        count[name] = count.get(name, 0) + 1
        return 'RUNNING' if count[name] == 1 else 'EXITED'

    monkeypatch.setattr(module.queue, 'supervisor_state', state)
    monkeypatch.setattr(module.time, 'sleep', lambda _: calls.append('wait'))
    monkeypatch.setattr(module, 'running_writers', lambda _: [])
    monkeypatch.setattr(module.queue, 'validate_plan', lambda _: calls.append('pins'))
    monkeypatch.setattr(module, 'require_free_ports', lambda: calls.append('ports'))
    monkeypatch.setattr(module.queue, 'wait_vllm', lambda _: calls.append('fp16'))
    expected = {(f'conv-{c}', q): {} for c in range(10) for q in range(154)}
    monkeypatch.setattr(module.queue, 'expected_questions', lambda *_: expected)
    monkeypatch.setattr(module.queue, 'wait_health', lambda *_: None)

    @contextmanager
    def service(*_args):
        calls.append('encoder')
        yield object()

    monkeypatch.setattr(module.queue, 'service', service)

    def smoke(probe_plan, method, turns, *, delivery_auditor):
        assert all(count[name] >= 2 for name in module.PREDECESSORS)
        assert turns == 80 and method == 'a_mem' and callable(delivery_auditor)
        assert module.os.environ['AMEM_ORIGINAL_FAILURE_POLICY'] == '1'
        assert module.os.environ['AMEM_SEMANTIC_JOURNAL_DIR'] == str(probe / 'a_mem/semantic_outcomes')
        calls.append('smoke')
        return {'passed': True, 'qa_count': 3, 'memory_turns': 80,
                'response_delivery': {'original_policy_delivery_complete': True}}

    monkeypatch.setattr(module, 'run_probe', smoke)

    def full_run(full_plan, item, wanted, *, context_indices):
        assert 'smoke' in calls and context_indices == list(range(10)) and len(wanted) == 1540
        assert module.os.environ['AMEM_SEMANTIC_JOURNAL_DIR'] == str(full / 'a_mem/semantic_outcomes')
        calls.append('full')
        return {'requires_composite': True, 'complete': False}

    monkeypatch.setattr(module.queue, 'run_method', full_run)
    monkeypatch.setattr(module, 'finalize_attempt', lambda *_: {'benchmark_complete': True, 'inference_complete': True})
    return module, plan, calls, prior, previous, full, probe


def test_only_after_real_smoke_runs_all10_and_keeps_failed_prior_diagnostics(setup, monkeypatch):
    module, plan, calls, prior, previous, _, probe = setup
    before = {path: path.read_bytes() for directory in (prior, *previous) for path in directory.rglob('*') if path.is_file()}
    monkeypatch.setenv('AMEM_SEMANTIC_JOURNAL_DIR', 'previous-value')
    assert module.main() == 0
    assert calls.index('wait') < calls.index('smoke') < calls.index('full')
    assert module.os.environ['AMEM_SEMANTIC_JOURNAL_DIR'] == 'previous-value'
    assert all(path.read_bytes() == value for path, value in before.items())
    result = json.loads((probe / 'recovery_status.json').read_text())
    assert result['state'] == 'finished_native_policy_inference' and result['complete'] is False
    assert json.loads(module.PLAN_PATH.read_text()) == plan


@pytest.mark.parametrize('failure', ['policy', 'output', 'dtype', 'skips', 'scope', 'prior_qa',
    'live_writer', 'bad_state', 'existing_output', 'smoke', 'semantic_smoke', 'mutated_prior', 'final_audit'])
def test_r26_gate_never_runs_full_without_valid_inputs_and_native_smoke(setup, monkeypatch, failure):
    module, plan, calls, prior, previous, full, _ = setup
    if failure in ('policy', 'output', 'dtype', 'skips'):
        key, value = {'policy': ('amem_failure_policy', 'other'), 'output': ('output', str(full / 'foreign')),
                      'dtype': ('dtype', 'int8'), 'skips': ('skip_methods', {})}[failure]
        plan[key] = value
        save(module.PLAN_PATH, plan)
    elif failure == 'scope':
        status = json.loads((prior / 'parallel_status.json').read_text())
        status['events'][-1]['exit_code'] = 0
        save(prior / 'parallel_status.json', status)
    elif failure == 'prior_qa':
        save(prior / 'prior_results.json', {'data': [{'output': 'already succeeded'}]})
    elif failure == 'live_writer':
        monkeypatch.setattr(module, 'running_writers', lambda _: [123])
    elif failure == 'bad_state':
        monkeypatch.setattr(module.queue, 'supervisor_state', lambda _: 'FATAL')
    elif failure == 'existing_output':
        full.mkdir()
    elif failure in ('smoke', 'semantic_smoke'):
        monkeypatch.setattr(module, 'run_probe', lambda *_args, **_kwargs: {'passed': failure != 'smoke',
            'qa_count': 3, 'memory_turns': 80, 'response_delivery': {'original_policy_delivery_complete': False}})
    elif failure == 'mutated_prior':
        original = module.run_probe

        def mutate(*args, **kwargs):
            report = original(*args, **kwargs)
            save(previous[0] / 'retained.json', {'changed': True})
            return report

        monkeypatch.setattr(module, 'run_probe', mutate)
    else:
        monkeypatch.setattr(module, 'finalize_attempt', lambda *_: {'benchmark_complete': False})
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        module.main()
    if failure != 'final_audit':
        assert 'full' not in calls


@pytest.fixture
def full_artifacts(tmp_path):
    module = target()
    output = tmp_path / 'a_mem'
    dataset = [{'sample_id': f'conv-{c}', 'qa': [{'question': f'Q{q}', 'answer': 'answer',
                'category': 1, 'evidence': ['D1:1']} for q in range(154)]} for c in range(10)]
    data_path, agent_path, config_path, scorer_path = [tmp_path / name for name in ('dataset.json', 'agent.json', 'config.json', 'scorer.py')]
    save(data_path, dataset)
    save(agent_path, {'agent_name': 'Agentic_memory_a_mem'})
    save(config_path, {'sub_dataset': 'locomo_qa'})
    scorer_path.write_text('def eval_question_answering(rows, eval_key, metric):\n    return [float(r[eval_key] == r["answer"]) for r in rows], [], []\n')
    plan = {'run_id': RUN, 'dataset': str(data_path), 'scorer': str(scorer_path), 'file_sha256': {},
            'methods': [{'method': 'a_mem', 'agent_config': str(agent_path), 'dataset_config': str(config_path)}]}
    expected = module.queue.expected_questions(data_path, 1540)
    template = module.get_template('locomo_qa', 'query', 'Agentic_memory_a_mem')
    raw, llm, embeddings, events = [], [], [], []
    for c in range(10):
        prefix = output / f'context_{c:02d}'
        shard = []
        call, event = native()
        call.update(request_id=f'meta-{c}', response_id=f'response-meta-{c}', sample_id=f'conv-{c}')
        event.update(response_id=call['response_id'], sample_id=call['sample_id'])
        llm.append(call)
        events.append(event)
        for q, qa in enumerate(dataset[c]['qa']):
            qid, sample = f'conv-{c}_qa{q}', f'conv-{c}'
            shard.append({'query': template.format(question=qa['question']), 'answer': qa['answer'], 'output': 'answer',
                'sample_id': sample, 'qa_pair_id': qid, 'question_id': qid, 'context_id': c, 'query_id': 154*c+q,
                'eval_metadata': {'dataset': 'locomo_qa', 'sample_id': sample, 'question_id': qid,
                                 'qa_pair_id': qid, 'category': 1, 'evidence': qa['evidence']}})
            llm.append(dict(row(f'qa-{c}-{q}'), sample_id=sample, question_id=q))
            embeddings.append(dict(row(f'embed-{c}-{q}', kind='embedding'), sample_id=sample, question_id=q))
        result_path = prefix / 'artifacts/native_results.json'
        save(result_path, {'data': shard})
        save(prefix / 'completion.json', {'context_index': c, 'sample_id': f'conv-{c}', 'questions': 154, 'result_path': str(result_path)})
        raw.extend(shard)
    status = {'active_contexts': [], 'failed': False, 'selected_contexts_complete': True, 'context_indices': list(range(10)),
        'events': [*[{'event': 'start', 'context': c} for c in range(10)],
                   *[{'event': 'end', 'context': c, 'exit_code': 0} for c in range(10)]]}
    save(output / 'parallel_status.json', status)
    save(output / 'selected_results.json', {'data': raw})
    for name, values in (('llm_usage.jsonl', llm), ('embedding_usage.jsonl', embeddings), ('usage.jsonl', llm + embeddings),
                          ('semantic_outcomes/123.jsonl', events)):
        save(output / name, values, lines=True)
    save(output / 'telemetry.json', {'wall_seconds': 10})
    return module, plan, output, expected


def test_full1540_native_policy_score_separates_strict_generation_and_real_tokens(full_artifacts):
    module, plan, output, expected = full_artifacts
    report = module.finalize_attempt(plan, output, expected)
    assert report['benchmark_complete'] is True and report['inference_complete'] is True
    assert report['usage_complete'] is True and report['original_policy_delivery_complete'] is True
    assert report['strict_all_generation_delivery']['complete'] is False
    assert report['official_f1'] == 1 and report['evaluated'] == 1540
    assert report['native_policy_audit']['native_length_responses'] == 10
    assert report['native_policy_audit']['usage']['exact_tokens']['total_tokens'] == 43980
    assert report['complete'] is False


@pytest.mark.parametrize('failure', ['missing_embedding', 'corrupt_event', 'missing_events', 'duplicate_event',
    'empty_prediction', 'metadata', 'global_id', 'missing_shard', 'wrong_combined'])
def test_fullrun_accounting_or_prediction_failure_is_preserved_and_not_completed(full_artifacts, failure):
    module, plan, output, expected = full_artifacts
    if failure == 'missing_embedding':
        (output / 'embedding_usage.jsonl').unlink()
    elif failure in ('corrupt_event', 'missing_events', 'duplicate_event'):
        path = output / 'semantic_outcomes/123.jsonl'
        if failure == 'missing_events':
            path.unlink()
        elif failure == 'corrupt_event':
            path.write_text('{bad-json\n')
        else:
            events = module.read_jsonl(path)
            save(path, events + [events[0]], lines=True)
    elif failure == 'missing_shard':
        (output / 'context_00/artifacts/native_results.json').unlink()
    elif failure == 'wrong_combined':
        save(output / 'usage.jsonl', [], lines=True)
    else:
        path = output / 'selected_results.json'
        raw = json.loads(path.read_text())
        raw['data'][0][{'empty_prediction': 'output', 'metadata': 'sample_id', 'global_id': 'query_id'}[failure]] = '' if failure == 'empty_prediction' else 'foreign'
        save(path, raw)
    if failure in ('empty_prediction', 'metadata', 'global_id', 'missing_shard'):
        with pytest.raises((ValueError, FileNotFoundError)):
            module.finalize_attempt(plan, output, expected)
    else:
        report = module.finalize_attempt(plan, output, expected)
        assert report['benchmark_complete'] is False
        if failure in ('missing_embedding', 'wrong_combined'):
            assert report['native_policy_audit']['usage']['exact_tokens'] is None
    assert (output / 'llm_usage.jsonl').exists()


def test_supervisor_is_explicitly_opt_in_and_never_restarts_or_powers_off():
    module = target()
    text = (module.ROOT / 'scripts/amem_original_policy_recovery_r26.conf').read_text()
    assert 'autostart=false' in text and 'autorestart=false' in text
    assert 'AMEM_ORIGINAL_FAILURE_POLICY="1"' in text
    assert 'locomo-amem-original-policy-r26' in text and 'shutdown' not in text
