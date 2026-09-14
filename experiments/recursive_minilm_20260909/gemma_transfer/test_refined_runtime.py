"""CPU mocks exercise the original cached methods; no real models or services."""
import dataclasses
import importlib.util
import json
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parent
QWEN = 'google/gemma-4-E4B-it'
MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
QREV = 'ee0ef6023621cff504d758262d4e04895a5af4a2'
EREV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) % 30 + 1 for c in text]

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = '|'.join(m['content'] for m in messages) + '|assistant:'
        return self.encode(text) if tokenize else text


class OOM(Exception):
    pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    engines, encoders, cleared = [], [], []
    class LLM:
        def __init__(self, **kwargs):
            self.kwargs, self.calls, self.error, self.malformed = kwargs, [], None, False
            engines.append(self)
        def get_tokenizer(self):
            return FakeTokenizer()
        def generate(self, prompts, sampling, **kwargs):
            self.calls.append((prompts, sampling, kwargs))
            if self.error:
                raise self.error
            result = []
            for i, prompt in enumerate(prompts):
                ids = prompt['prompt_token_ids'] if isinstance(prompt, dict) else FakeTokenizer().encode(prompt)
                candidate = SimpleNamespace(index=0, text=' answer ', token_ids=[10] if sampling.max_tokens == 1 else [10, 11],
                                            finish_reason='length', logprobs=[{10: SimpleNamespace(logprob=-.2, rank=1)}])
                result.append(SimpleNamespace(request_id=f'{len(self.calls)}-{i}', prompt=prompt, prompt_token_ids=ids,
                    outputs=[candidate], num_cached_tokens=1, finished=True,
                    prompt_logprobs=[{n: SimpleNamespace(logprob=-.2, rank=1, decoded_token='x')} for n in ids]))
            if self.malformed:
                result[0].outputs = []
            return result
    class Encoder:
        max_seq_length = 256
        def __init__(self, path, device):
            self.path, self.device, self.dtype, self.calls, self.forwards, self.fail_at = path, device, None, [], 0, None
            encoders.append(self)
        def float(self):
            self.dtype = 'float32'
            return self
        def forward(self, features):
            self.forwards += 1
            if self.forwards == self.fail_at:
                raise OOM('synthetic forward OOM')
            return {'sentence_embedding': np.array([[int(n), 1, 2] for n in features['attention_mask'].sum(axis=1)])}
        def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
            self.calls.append((list(texts), batch_size, normalize_embeddings, show_progress_bar))
            result = []
            for start in range(0, len(texts), batch_size):
                counts = [min(len(t) + 2, 256) for t in texts[start:start + batch_size]]
                width = max(counts)
                features = {'input_ids': np.array([[1] * n + [0] * (width - n) for n in counts]),
                            'attention_mask': np.array([[1] * n + [0] * (width - n) for n in counts])}
                result.append(self.forward(features)['sentence_embedding'])
            return np.vstack(result)
    monkeypatch.syspath_prepend(str(ROOT / 'source'))
    monkeypatch.syspath_prepend(str(ROOT))
    for name in ('refine', 'run_evidence_utility', 'evidence_utility', 'runtime_meter', 'chat_tokenizer_compat'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda n: None, OutOfMemoryError=OOM,
        cuda=SimpleNamespace(empty_cache=lambda: cleared.append(True))))
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Encoder))
    spec = importlib.util.spec_from_file_location('isolated_runtime', ROOT / 'transfer_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(out=str(tmp_path), stage='prepare', model=QWEN, embed_model=MINILM, seed=20260907, embed_batch_size=4)
    model_dir = tmp_path / 'gemma_model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_bytes((ROOT / 'model_config.json').read_bytes())
    environment = {'models': {QWEN: {'path': str(model_dir), 'revision': QREV}, MINILM: {'path': '/pinned/minilm', 'revision': EREV}}}
    return SimpleNamespace(t=module, args=args, env=environment, engines=engines, encoders=encoders,
                           out=tmp_path, cleared=cleared)


def receipts(b, kind):
    return [json.loads(p.read_text()) for p in sorted(b.out.rglob('*.receipt.json'))
            if json.loads(p.read_text())['kind'] == kind]


def test_fixed_model_initialization_and_native_seeds(setup):
    b = setup
    b.t.Runtime(b.args, b.env)
    assert b.engines[0].kwargs['dtype'] == 'bfloat16'
    assert b.engines[0].kwargs['quantization'] is None
    assert b.engines[0].kwargs['seed'] == 20260907 and b.engines[0].kwargs['enable_prefix_caching'] is True
    assert b.encoders[0].device == 'cpu' and b.encoders[0].dtype == 'float32'
    b.args.stage = 'score'
    b.t.Scorer(b.args, b.env, b.out)
    assert b.engines[1].kwargs['dtype'] == 'bfloat16' and b.engines[1].kwargs['seed'] == 20260908
    assert b.engines[1].kwargs['enable_prefix_caching'] is False


def test_native_generation_duplicate_work_cache_and_length_recorded(setup):
    b = setup
    b.t.set_runtime_context('construction', conversation='conv-26')
    rt = b.t.Runtime(b.args, b.env)
    assert rt.generate('system', ['same', 'same'], max_tokens=96) == ['answer', 'answer']
    assert rt.generate('system', ['same'], max_tokens=96) == ['answer']
    assert len(b.engines[0].calls) == 1 and len(b.engines[0].calls[0][0]) == 2
    calls = receipts(b, 'generation')
    assert len(calls) == 1 and calls[0]['response_count'] == 2 and calls[0]['completion_tokens'] == 4
    assert calls[0]['finish_reasons'] == ['length', 'length']
    assert calls[0]['context']['conversation'] == 'conv-26'
    operations = [r for r in receipts(b, 'operation') if r['operation_kind'] == 'generation']
    assert operations[-1]['zero_invocations_observed'] is True
    assert operations[-1]['cache_hit_proven'] is False
    raw = json.loads(next(b.out.rglob('generation/*.response.json')).read_text())
    assert raw[0]['outputs'][0]['token_ids'] == [10, 11] and raw[0]['prompt_logprobs']


def test_minilm_input_adapter_preserves_native_cache_and_order(setup):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    texts = ['abc', 'longer']
    result = rt.encode(texts, query=True)
    assert b.encoders[0].calls[0] == (texts, 4, True, False)
    np.testing.assert_array_equal(result, [[5, 1, 2], [8, 1, 2]])
    np.testing.assert_array_equal(rt.encode(texts, query=True), result)
    assert len(b.encoders[0].calls) == 1 and texts == ['abc', 'longer']
    assert receipts(b, 'embedding')[0]['input_tokens'] == 13
    assert receipts(b, 'embedding')[0]['padded_tokens'] == 16
    prefix = 'Instruct: Retrieve relevant conversation memories to answer the question.\nQuery: '
    rt.encode([prefix + 'literal document'], query=False)
    assert b.encoders[0].calls[-1][0] == [prefix + 'literal document']


def test_native_oom_backoff_counts_failed_and_replayed_forwards(setup):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    b.encoders[0].fail_at = 2
    result = rt.encode(['abc'] * 5)
    assert result.shape == (5, 3)
    assert [call[1] for call in b.encoders[0].calls] == [4, 2]
    rows = receipts(b, 'embedding')
    assert len(rows) == 5 and sum(r['input_tokens'] for r in rows) == 50
    assert [r['status'] for r in rows].count('error') == 1
    assert b.cleared == [True]


def test_native_nll_score_and_control_generation_metered_separately(setup):
    b = setup
    b.args.stage = 'score'
    b.t.set_runtime_context('scoring_source', probe_id='p1')
    scorer = b.t.Scorer(b.args, b.env, b.out)
    jobs = [{'user': 'original source question', 'answer': 'answer', 'source_ids': ['D1:2'], 'context_tokens': 5}]
    scorer.score(jobs)
    assert jobs[0]['score']['answer_tokens'] == 6
    scored = receipts(b, 'generation')[0]
    assert scored['operation_kind'] == 'source_utility_nll' and scored['completion_tokens'] == 1
    assert scored['prompt_tokens'] == len(b.engines[0].calls[0][0][0]['prompt_token_ids'])
    assert b.engines[0].calls[0][1].prompt_logprobs == 0
    result = scorer.generate({'full': jobs[0]})
    assert result['full']['text'] == 'answer'
    assert receipts(b, 'generation')[-1]['operation_kind'] == 'source_utility_generation'
    scorer.score(jobs)
    assert len(b.engines[0].calls) == 2


def test_raw_results_survive_native_consumer_failure(setup):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    b.engines[0].malformed = True
    with pytest.raises(IndexError):
        rt.generate('system', ['question'])
    assert next(b.out.rglob('generation/*.response.json')).is_file()
    assert [r for r in receipts(b, 'operation') if r['operation_kind'] == 'generation'][-1]['status'] == 'error'


def test_model_exception_keeps_unknown_usage_and_reraises(setup):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    b.engines[0].error = RuntimeError('synthetic engine failure')
    with pytest.raises(RuntimeError, match='synthetic engine failure'):
        rt.generate('system', ['question'])
    row = receipts(b, 'generation')[0]
    assert row['status'] == 'error' and row['prompt_tokens'] is None and row['completion_tokens'] is None


@pytest.mark.parametrize('which', ['model', 'embedding'])
def test_revision_metadata_mismatch_before_model_load(setup, which):
    b = setup
    b.env['models'][QWEN if which == 'model' else MINILM]['revision'] = 'wrong'
    with pytest.raises(ValueError, match='revision'):
        b.t.Runtime(b.args, b.env)
    assert not b.engines and not b.encoders


def test_missing_response_is_recorded_lowerbound_not_exact_batch_usage(setup):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    original = b.engines[0].generate
    b.engines[0].generate = lambda *args, **kwargs: original(*args, **kwargs)[:1]
    assert rt.generate('system', ['one', 'two']) == ['answer', None]
    row = receipts(b, 'generation')[0]
    assert row['usage_complete'] is False and row['total_tokens'] is None
    assert row['request_count'] == 2 and row['response_count'] == 1
    assert row['recorded_completion_tokens'] == 2


def test_inference_duration_excludes_artifact_writes(setup):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    rt.generate('system', ['question'])
    row = receipts(b, 'generation')[0]
    assert 0 <= row['inference_duration_s'] <= row['duration_s']


def test_first_engine_config_recorded_before_initialization(setup):
    b = setup
    b.t.Runtime(b.args, b.env)
    requests = [json.loads(p.read_text()) for p in b.out.rglob('operation/*.request.json')]
    recorded = next(r for r in requests if r['operation_kind'] == 'llm_initialization')['payload']['inputs']
    assert recorded == b.engines[0].kwargs


def test_wrong_native_embedding_window_has_final_failure_receipt(setup, monkeypatch):
    b = setup
    monkeypatch.setattr(sys.modules['sentence_transformers'].SentenceTransformer, 'max_seq_length', 512)
    with pytest.raises(ValueError, match='256'):
        b.t.Runtime(b.args, b.env)
    assert receipts(b, 'operation')[-1]['status'] == 'error'
    assert not b.engines


@pytest.mark.parametrize('suffix', ['request.json', 'response.json', 'receipt.json'])
def test_generation_logging_failure_never_delivers_success(setup, monkeypatch, suffix):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    original = rt.meter.write
    def fail(path, value):
        if path.parent.name == 'generation' and path.name.endswith(suffix):
            raise OSError('synthetic journal unavailable')
        return original(path, value)
    monkeypatch.setattr(rt.meter, 'write', fail)
    with pytest.raises(OSError, match='journal unavailable'):
        rt.generate('system', ['question'])
    assert len(b.engines[0].calls) == (0 if suffix == 'request.json' else 1)
    assert receipts(b, 'operation')[-1]['status'] == 'error'


def test_native_cache_save_failure_keeps_metered_raw_and_tokens(setup, monkeypatch):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    def fail(*args):
        raise OSError('synthetic native cache failure')
    monkeypatch.setattr(b.t.core, 'save', fail)
    with pytest.raises(OSError, match='native cache failure'):
        rt.generate('system', ['question'])
    assert receipts(b, 'generation')[0]['usage_complete'] is True
    assert next(b.out.rglob('generation/*.response.json')).is_file()
    assert receipts(b, 'operation')[-1]['status'] == 'error'


def test_nll_consumer_failure_retains_prompt_logprobs_and_stub_tokens(setup, monkeypatch):
    b = setup
    b.args.stage = 'score'
    scorer = b.t.Scorer(b.args, b.env, b.out)
    module = sys.modules['run_evidence_utility']
    monkeypatch.setattr(module, 'extract_answer_logprob', lambda *args: (_ for _ in ()).throw(ValueError('bad target logprob')))
    with pytest.raises(ValueError, match='bad target logprob'):
        scorer.score([{'user': 'question', 'answer': 'answer'}])
    assert receipts(b, 'generation')[0]['completion_tokens'] == 1
    assert next(b.out.rglob('generation/*.response.json')).is_file()
    assert receipts(b, 'operation')[-1]['status'] == 'error'


def test_final_oom_and_preflight_failure_are_not_unrecorded(setup):
    b = setup
    b.args.embed_batch_size = 1
    rt = b.t.Runtime(b.args, b.env)
    b.encoders[0].fail_at = 1
    with pytest.raises(OOM):
        rt.encode(['abc'])
    assert receipts(b, 'embedding')[0]['input_tokens'] == 5
    assert receipts(b, 'embedding')[0]['status'] == 'error'
    with pytest.raises(ValueError, match='refusing silent truncation'):
        rt.generate('system', ['x' * 70000])
    assert not b.engines[0].calls


def test_fresh_instance_namespaces_and_context_replacement(setup):
    b = setup
    b.t.set_runtime_context('first', probe_id='old')
    first = b.t.Runtime(b.args, b.env)
    b.t.set_runtime_context('second', conversation='new')
    second = b.t.Runtime(b.args, b.env)
    assert first.meter.root != second.meter.root
    second.generate('system', ['question'])
    assert receipts(b, 'generation')[0]['context'] == {'phase': 'second', 'conversation': 'new'}


def test_native_struct_serialization_and_failclosed_unsupported_request(setup):
    b = setup
    meter_module = sys.modules['runtime_meter']
    @dataclasses.dataclass
    class Logprob:
        logprob: float
        rank: int
    class Mode(Enum):
        FINAL = 1
    class Params:
        __slots__ = ('output_kind', 'temperature')
        __struct_fields__ = ('temperature', 'output_kind')
        def __init__(self):
            self.temperature, self.output_kind = 0.0, Mode.FINAL
    assert meter_module.jsonable(Params()) == {'temperature': 0.0, 'output_kind': 1}
    assert meter_module.jsonable(Logprob(-.2, 1)) == {'logprob': -.2, 'rank': 1}
    assert meter_module.jsonable(float('-inf')) == {'nonfinite_float': '-inf'}
    assert meter_module.jsonable(b.out) == str(b.out)
    rt = b.t.Runtime(b.args, b.env)
    with pytest.raises(ValueError, match='Unsupported native logging'):
        rt.generate('system', [object()])
    assert not b.engines[0].calls


def test_tensor_masks_and_invalid_native_masks_fail_before_forward(setup):
    meter_module = sys.modules['runtime_meter']
    class Tensor:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return self.value
    assert meter_module.embedding_usage({'input_ids': Tensor([[1, 2]]), 'attention_mask': Tensor([[1, 1]])})['input_tokens'] == 2
    for mask in (None, [], [[2]], [[1] * 257]):
        with pytest.raises(ValueError, match='mask|MiniLM'):
            meter_module.embedding_usage({'input_ids': [[1]], 'attention_mask': mask})


def test_meter_refuses_overwrite_bad_stage_and_changed_prefix(setup):
    b = setup
    meter_module = sys.modules['runtime_meter']
    with pytest.raises(ValueError, match='stage'):
        meter_module.Meter(b.out, '../foreign', {})
    with pytest.raises(ValueError, match='phase'):
        b.t.set_runtime_context('')
    rt = b.t.Runtime(b.args, b.env)
    path = rt.meter.root / 'exclusive.json'
    rt.meter.write(path, {'first': True})
    with pytest.raises(FileExistsError):
        rt.meter.write(path, {'second': True})
    assert json.loads(path.read_text()) == {'first': True}
    rt.embed.query = True
    with pytest.raises(ValueError, match='prefix'):
        rt.embed.encode(['unprepared query'], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
    assert not b.encoders[0].calls


@pytest.mark.parametrize('attribute,value', [('model', 'Qwen/other'), ('seed', 20260908)])
def test_wrong_native_model_or_runtime_seed_refused_before_loading(setup, attribute, value):
    b = setup
    setattr(b.args, attribute, value)
    with pytest.raises(ValueError):
        b.t.Runtime(b.args, b.env)
    assert not b.engines and not b.encoders


def test_unsupported_batch_contract_is_not_sent_to_engine(setup):
    b = setup
    rt = b.t.Runtime(b.args, b.env)
    with pytest.raises(TypeError, match='batched prompt'):
        rt.llm.generate('not a batch', SimpleNamespace(max_tokens=1))
    assert not b.engines[0].calls


@pytest.mark.parametrize('role', ['reader', 'nll'])
@pytest.mark.parametrize('stop_ids', [set(), {9, 3}, frozenset({9, 3})])
def test_real_samplingparams_stop_token_set_contract(setup, monkeypatch, role, stop_ids):
    b = setup
    # The actual vLLM0.28 struct has this set-valued field for reader and NLL.
    # Other fields here are the exact fields consumed by our CPU fake engine.
    class SamplingParams:
        __slots__ = ('_all_stop_token_ids', 'max_tokens', 'prompt_logprobs', 'temperature')
        __struct_fields__ = __slots__
        def __init__(self, temperature, max_tokens, prompt_logprobs=None):
            self.temperature, self.max_tokens, self.prompt_logprobs = temperature, max_tokens, prompt_logprobs
            self._all_stop_token_ids = stop_ids
    monkeypatch.setattr(sys.modules['vllm'], 'SamplingParams', SamplingParams)
    if role == 'reader':
        rt = b.t.Runtime(b.args, b.env)
        assert rt.generate('system', ['question'], max_tokens=96) == ['answer']
    else:
        b.args.stage = 'score'
        scorer = b.t.Scorer(b.args, b.env, b.out)
        jobs = [{'user': 'question', 'answer': 'answer'}]
        scorer.score(jobs)
        assert jobs[0]['score']['answer_tokens'] == 6
    assert len(b.engines[0].calls) == 1
    assert b.engines[0].calls[0][1]._all_stop_token_ids is stop_ids
    request = json.loads(next(b.out.rglob('generation/*.request.json')).read_text())
    encoded = request['payload']['args'][1]['_all_stop_token_ids']
    assert encoded == {'native_type': type(stop_ids).__name__, 'items': sorted(stop_ids)}
    assert receipts(b, 'generation')[0]['usage_complete'] is True


def test_set_serialization_is_deterministic_and_unknown_members_still_fail(setup):
    meter_module = sys.modules['runtime_meter']
    left = {frozenset({3, 9}), 'native', 4}
    right = set()
    for item in reversed(list(left)):
        right.add(item)
    assert json.dumps(meter_module.jsonable(left), sort_keys=True) == json.dumps(meter_module.jsonable(right), sort_keys=True)
    with pytest.raises(ValueError, match='Unsupported native logging'):
        meter_module.jsonable({object()})


def test_gemma_writer_accepts_long_sessions_but_reader_keeps8192(setup):
    b = setup
    writer = b.t.Runtime(b.args, b.env)
    assert writer.context_capacity == 65536
    assert writer.generate('system', ['x' * 8500], max_tokens=96) == ['answer']
    b.args.stage = 'evaluate'
    reader = b.t.Runtime(b.args, b.env)
    assert reader.context_capacity == 8192
    overhead = len(FakeTokenizer().apply_chat_template([
        {'role': 'system', 'content': 'system'}, {'role': 'user', 'content': ''}]))
    boundary = 8192 - 96 - overhead
    assert reader.generate('system', ['y' * boundary], max_tokens=96) == ['answer']
    with pytest.raises(ValueError, match='refusing silent truncation'):
        reader.generate('system', ['y' * (boundary + 1)], max_tokens=96)
