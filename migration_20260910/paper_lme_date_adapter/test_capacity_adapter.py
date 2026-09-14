"""CPU behavior checks for capacity-only execution changes and frozen provenance."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_refined_runtime import receipts

pytest_plugins = ('test_refined_runtime',)

ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize('stage,capacity', [('prepare', 65536), ('evaluate', 8192)])
def test_engine_capacity_boundary_and_rejection_before_native_calls(setup, stage, capacity):
    b = setup
    b.args.stage = stage
    rt = b.t.Runtime(b.args, b.env)
    assert b.engines[0].kwargs['max_model_len'] == capacity
    if stage == 'prepare':
        assert b.engines[0].kwargs['enable_chunked_prefill'] is True
    else:
        assert 'enable_chunked_prefill' not in b.engines[0].kwargs
    assert b.engines[0].kwargs['max_num_batched_tokens'] == 8192
    assert b.engines[0].kwargs['max_num_seqs'] == 24
    system, reserve = 'system', 96
    template = rt.tok.apply_chat_template(
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': ''}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    user = 'x' * (capacity - reserve - rt.ntok(template))
    assert rt.generate(system, [user], max_tokens=reserve) == ['answer']
    assert len(b.engines[0].calls) == 1
    native = receipts(b, 'generation')[0]
    assert native['prompt_tokens'] + reserve == capacity
    assert native['profile']['context_capacity'] == capacity
    assert native['usage_complete'] is True
    with pytest.raises(ValueError, match='refusing silent truncation'):
        rt.generate(system, [user + 'x'], max_tokens=reserve)
    assert len(b.engines[0].calls) == 1
    failures = [r for r in receipts(b, 'operation') if r['status'] == 'error']
    assert len(failures) == 1 and failures[0]['operation_kind'] == 'generation'


def test_writer_accepts_original_8192_overflow_but_reader_rejects(setup):
    b = setup
    writer = b.t.Runtime(b.args, b.env)
    assert writer.generate('system', ['x' * 9000], max_tokens=512) == ['answer']
    # Each run starts with a fresh output/cache namespace. Capacity is not a cache key.
    b.args.out = str(b.out / 'fresh-reader')
    b.args.stage = 'evaluate'
    reader = b.t.Runtime(b.args, b.env)
    with pytest.raises(ValueError, match='refusing silent truncation'):
        reader.generate('system', ['x' * 9000], max_tokens=512)
    assert not b.engines[1].calls


def test_scorer_engine_and_native_guard_remain_8192(setup, monkeypatch):
    b = setup
    b.args.stage = 'score'
    scorer = b.t.Scorer(b.args, b.env, b.out)
    engine = b.engines[0]
    assert engine.kwargs['max_model_len'] == 8192
    assert engine.kwargs['max_num_batched_tokens'] == 512
    assert engine.kwargs['seed'] == 20260908
    module = sys.modules['run_evidence_utility']
    def request(tokenizer, user, answer):
        length = 8191 if user == 'fits' else 8192
        return {'prompt_token_ids': [1] * length}, length - 1, [1]
    monkeypatch.setattr(module, 'answer_token_request', request)
    scorer.score([{'user': 'fits', 'answer': 'answer'}])
    assert len(engine.calls) == 1
    assert receipts(b, 'generation')[0]['prompt_tokens'] == 8191
    with pytest.raises(ValueError, match='NLL prompt exceeds'):
        scorer.score([{'user': 'overflows', 'answer': 'answer'}])
    assert len(engine.calls) == 1


def test_generation_override_preserves_frozen_algorithm_except_capacity():
    source = (ROOT / 'source/refine.py').read_text(encoding='utf-8')
    begin = source.index('    def generate(self, system, users, max_tokens=512):')
    end = source.index('    def encode(self, texts, query=False):')
    expected = source[begin:end]
    expected = expected.replace('def generate(', 'def _generate(', 1)
    expected = expected.replace('key = digest(', 'key = core.digest(')
    expected = expected.replace('                save(paths[j],', '                core.save(paths[j],')
    expected = expected.replace(' + max_tokens > 8192:', ' + max_tokens > self.context_capacity:')
    adapted = (ROOT / 'transfer_runtime.py').read_text(encoding='utf-8')
    begin = adapted.index('    def _generate(self, system, users, max_tokens=512):')
    end = adapted.index('    def encode(self, texts:')
    assert adapted[begin:end] == expected


def test_cache_batching_sampling_and_exception_identity_survive_override(setup):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    users = [f'query {index}' for index in range(25)]
    assert rt.generate('system', users, 96) == ['answer'] * 25
    assert [len(call[0]) for call in b.engines[0].calls] == [24, 1]
    assert all(call[1].temperature == 0 and call[1].max_tokens == 96 for call in b.engines[0].calls)
    for user in users:
        key = b.t.core.digest([rt.model_meta, 20260907, 'system', user, 96, False])
        assert (rt.cache / 'generations' / (key + '.json')).exists()
    assert rt.generate('system', users, 96) == ['answer'] * 25
    assert len(b.engines[0].calls) == 2
    original = RuntimeError('synthetic native failure')
    b.engines[0].error = original
    with pytest.raises(RuntimeError) as caught:
        rt.generate('system', ['new non-cached question'], 96)
    assert caught.value is original
    failures = [r for r in receipts(b, 'generation') if r['status'] == 'error']
    assert len(failures) == 1 and failures[0]['usage_complete'] is False


def test_plan_records_actual_adapter_hash_and_explicit_role_capacities(setup, monkeypatch):
    b = setup
    monkeypatch.setitem(sys.modules, 'transfer_runtime', b.t)
    spec = importlib.util.spec_from_file_location('isolated_paper_runner', ROOT / 'run_transfer.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda name: 'cpu-test-stub')
    data, environment, output = b.out / 'data.json', b.out / 'environment.json', b.out / 'new-run'
    data.write_text(json.dumps([{'sample_id': 'source-only-test', 'qa': [], 'conversation': {
        'session_1_date_time': '2026-09-10',
        'session_1': [{'speaker': 'user', 'text': 'A source history.', 'dia_id': 'D1:1'}],
    }}]), encoding='utf-8')
    environment.write_text(json.dumps(b.env), encoding='utf-8')
    argv = [str(ROOT / 'run_transfer.py'), 'plan', '--dataset', 'locomo', '--data', str(data),
            '--out', str(output), '--environment', str(environment)]
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['context_capacity'] == {'writer': 65536, 'reader': 8192, 'scorer': 8192}
    assert protocol['execution_profile'] == b.t.EXECUTION_PROFILE
    assert protocol['read_budget'] == 2048
    assert protocol['max_answer_tokens'] == 96
    assert protocol['extra_storage_budget'] == 2000
    assert protocol['methods'] == ['seed', 'r40_fused_four_turn', 's_parent_single_2000']
    assert protocol['evaluation_methods'] == protocol['methods']
    assert protocol['target_selection'] is False and protocol['history_truncation'] is False
    assert protocol['config']['model'] == 'Qwen/Qwen3.5-9B'
    assert protocol['config']['embed_model'] == 'sentence-transformers/all-MiniLM-L6-v2'
    assert protocol['source_hashes']['transfer_runtime.py'] == hashlib.sha256((ROOT / 'transfer_runtime.py').read_bytes()).hexdigest()
    assert not b.engines and not b.encoders
    runner.main()  # Identical locked protocol can resume without changing any fields.


def test_frozen_source_and_plans_have_unchanged_copy_hashes():
    provenance = json.loads((ROOT / 'COPIED_SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
    frozen = [r for r in provenance['files'] if r['path'].startswith('source/')]
    assert len(frozen) == 15  # 11 transitive algorithm modules and 4 frozen plans.
    for entry in frozen:
        assert entry['copied_unchanged'] is True
        assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['original_sha256']
    # Every local import in the copied execution closure resolves in the clone.
    present = {p.stem for p in (ROOT / 'source').glob('*.py')}
    assert {'refine', 'portable_parent', 'evidence_utility', 'run_evidence_utility'} <= present
    for path in [ROOT / 'run_transfer.py', ROOT / 'transfer_runtime.py']:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert '/workspace/recursive_minilm_20260909' not in node.value
                assert 'D:/MemoryData/experiments' not in node.value
