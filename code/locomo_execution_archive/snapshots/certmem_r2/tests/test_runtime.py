"""Comparison runtime meters real native work without changing reader behavior."""
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from certmem.config import Config


def runtime():
    return importlib.import_module('certmem.runtime')


class Tokens:
    def __init__(self, count):
        self.input_ids = list(range(count))


class Tokenizer:
    def __call__(self, text, **_kwargs):
        return Tokens(len(text.split()))

    def apply_chat_template(self, messages, **kwargs):
        return '|'.join(item['content'] for item in messages) + ('<think>' if kwargs['enable_thinking'] else '|assistant')


@pytest.fixture
def engine_factory(tmp_path, monkeypatch):
    calls = {'models': [], 'tokenizers': [], 'generate': []}
    tokenizer = Tokenizer()
    fake_vllm, fake_transformers = ModuleType('vllm'), ModuleType('transformers')

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self, **kwargs):
            calls['models'].append(kwargs)
            self.error = None
            self.outputs = None

        def generate(self, prompts, params, **kwargs):
            calls['generate'].append((copy.deepcopy(prompts), copy.deepcopy(params.__dict__), kwargs))
            if self.error:
                raise self.error
            return self.outputs if self.outputs is not None else [SimpleNamespace(
                request_id=f'engine-{i}', prompt=prompt, prompt_token_ids=[10, 11, 12], finished=True,
                outputs=[SimpleNamespace(index=0, text=' answer ', token_ids=[20, 21],
                                         finish_reason='length', stop_reason=None, cumulative_logprob=-0.5)],
                metrics=SimpleNamespace(time_in_queue=0.01, first_token_time=1.0))
                for i, prompt in enumerate(prompts)]

    def tokenizer_factory(model, **kwargs):
        calls['tokenizers'].append((model, kwargs))
        return tokenizer

    fake_vllm.LLM, fake_vllm.SamplingParams = LLM, Params
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer_factory)
    monkeypatch.setitem(sys.modules, 'vllm', fake_vllm)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    sys.modules.pop('certmem.llm', None)
    module = importlib.import_module('certmem.llm')

    def make(comparison=True, out=None):
        cfg = runtime().comparison_config(Config(), out or tmp_path / 'run') if comparison else Config()
        return module.Engine(cfg), cfg, calls

    yield make
    sys.modules.pop('certmem.llm', None)


def read_all(root, kind, suffix):
    return [json.loads(path.read_text()) for path in sorted((root / 'runtime' / kind).glob(f'*.{suffix}.json'))]


def test_exact_comparison_config_without_mutating_native_defaults(tmp_path):
    original = Config()
    before = copy.deepcopy(original)
    cfg = runtime().comparison_config(original, tmp_path)
    assert original == before and cfg is not original
    assert cfg.model == 'Qwen/Qwen3.5-9B' and cfg.dtype == 'half'
    assert cfg.model_revision == 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
    assert cfg.embed_model == 'sentence-transformers/all-MiniLM-L6-v2'
    assert cfg.embed_revision == '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
    assert cfg.max_model_len == 49152 and cfg.gpu_memory_utilization == 0.92
    assert cfg.max_num_seqs == 48 and cfg.max_num_batched_tokens == 8192
    assert cfg.embedding_device == 'cpu'


def test_native_default_engine_invocation_and_outputs_unchanged(engine_factory):
    engine, _, calls = engine_factory(False)
    assert calls['models'] == [{'model': 'Qwen/Qwen3-8B', 'max_model_len': 16384,
                               'gpu_memory_utilization': 0.88, 'dtype': 'bfloat16', 'seed': 0}]
    assert calls['tokenizers'] == [('Qwen/Qwen3-8B', {})]
    engine.set_context(phase='reader')
    assert engine.generate('system', ['original'], max_tokens=32) == ['answer']
    assert calls['generate'][0][1] == {'temperature': 0.0, 'max_tokens': 32}


def test_native_comparison_parameters_and_raw_length_accounting(engine_factory):
    engine, cfg, calls = engine_factory()
    model = calls['models'][0]
    assert model['dtype'] == 'half' and model['quantization'] is None
    assert model['revision'] == cfg.model_revision and model['tokenizer_revision'] == cfg.model_revision
    assert model['language_model_only'] is True and model['generation_config'] == 'vllm'
    assert model['max_num_seqs'] == 48 and model['max_num_batched_tokens'] == 8192
    assert model['enable_prefix_caching'] is True
    engine.set_context(phase='reader', conv_id='conv-26', config='full')
    assert engine.generate('system', ['original question'], max_tokens=32) == ['answer']
    root = Path(cfg.comparison_output_dir)
    response, = read_all(root, 'generation', 'response')
    receipt, = read_all(root, 'generation', 'receipt')
    assert response['responses'][0]['prompt_token_ids'] == [10, 11, 12]
    assert response['responses'][0]['outputs'][0]['text'] == ' answer '
    assert response['responses'][0]['outputs'][0]['token_ids'] == [20, 21]
    assert receipt['prompt_tokens'] == 3 and receipt['completion_tokens'] == 2 and receipt['total_tokens'] == 5
    assert receipt['usage_complete'] is True and receipt['finish_reasons'] == ['length']
    assert receipt['context'] == {'phase': 'reader', 'conv_id': 'conv-26', 'config': 'full'}
    assert receipt['duration_s'] >= 0 and receipt['request_sha256'] and receipt['response_sha256']


def test_repeated_native_requests_are_new_work_not_cache(engine_factory):
    engine, cfg, calls = engine_factory()
    for _ in range(2):
        engine.generate('same system', ['same prompt'])
    assert len(calls['generate']) == 2
    rows = read_all(Path(cfg.comparison_output_dir), 'generation', 'receipt')
    assert len(rows) == 2 and rows[0]['batch_sequence'] != rows[1]['batch_sequence']
    assert rows[0]['request_hash'] == rows[1]['request_hash']


def test_precreated_top_output_allowed_but_existing_telemetry_refuses_reuse(engine_factory, tmp_path):
    out = tmp_path / 'precreated'
    out.mkdir()
    (out / 'manifest.json').write_text('{}')
    engine_factory(out=out)
    with pytest.raises(FileExistsError):
        engine_factory(out=out)


def test_inference_error_receipt_has_unknown_not_zero_tokens(engine_factory):
    engine, cfg, _ = engine_factory()
    engine.llm.error = RuntimeError('synthetic engine failure')
    with pytest.raises(RuntimeError, match='synthetic'):
        engine.generate('system', ['question'])
    row, = read_all(Path(cfg.comparison_output_dir), 'generation', 'receipt')
    assert row['status'] == 'error' and row['usage_complete'] is False
    assert row['prompt_tokens'] is None and row['completion_tokens'] is None


def test_unknown_actual_tokens_fail_after_preserving_raw_response(engine_factory):
    engine, cfg, _ = engine_factory()
    engine.llm.outputs = [SimpleNamespace(request_id='raw', prompt='q', prompt_token_ids=None,
        outputs=[SimpleNamespace(text='answer', token_ids=[1], finish_reason='stop')])]
    with pytest.raises(ValueError, match='token'):
        engine.generate('system', ['question'])
    assert read_all(Path(cfg.comparison_output_dir), 'generation', 'response')
    row, = read_all(Path(cfg.comparison_output_dir), 'generation', 'receipt')
    assert row['usage_complete'] is False and row['prompt_tokens'] is None


def test_preflight_oversize_refuses_before_any_generation(engine_factory):
    engine, cfg, calls = engine_factory()
    with pytest.raises(ValueError, match='context'):
        engine.generate('system', ['word ' * cfg.max_model_len], max_tokens=32)
    assert not calls['generate']
    row, = read_all(Path(cfg.comparison_output_dir), 'generation', 'receipt')
    assert row['status'] == 'rejected_preflight'


def test_thinking_trim_and_original_native_budget_unchanged(engine_factory):
    engine, cfg, _ = engine_factory()
    engine.llm.outputs = [SimpleNamespace(request_id='think', prompt_token_ids=[1], outputs=[
        SimpleNamespace(text='<think> reason </think> answer ', token_ids=list(range(7)), finish_reason='stop')])]
    assert engine.generate('system', ['question'], max_tokens=32, think=True) == ['answer']
    assert engine.last_think_tokens == [6]
    request, = read_all(Path(cfg.comparison_output_dir), 'generation', 'request')
    assert request['sampling']['max_tokens'] == 768


def test_unmetered_answer_logprob_explicitly_disabled_comparison(engine_factory):
    engine, _, calls = engine_factory()
    with pytest.raises(NotImplementedError):
        engine.answer_logprob('system', ['question'], ['gold'])
    assert not calls['generate']


class Mask:
    shape = (2, 3)

    def sum(self, dim=None):
        return SimpleNamespace(item=lambda: 5, tolist=lambda: [2, 3])

    def numel(self):
        return 6


@pytest.fixture
def embeddings(monkeypatch):
    calls = []

    class AutoModel:
        def __init__(self):
            self.forward_calls = []
            self.output = SimpleNamespace(last_hidden_state=SimpleNamespace(shape=(2, 3, 384)))

        def forward(self, input_ids=None, attention_mask=None, fail=False, **kwargs):
            self.forward_calls.append((input_ids, attention_mask, fail, kwargs))
            if fail:
                raise RuntimeError('embedding forward failed')
            return self.output

        def __call__(self, *args, **kwargs):
            if hasattr(self, 'pre'):
                self.pre(self, args, kwargs)
            output = None
            try:
                output = self.forward(*args, **kwargs)
                return output
            finally:
                if hasattr(self, 'post'):
                    self.post(self, args, kwargs, output)

        def register_forward_pre_hook(self, hook, **kwargs):
            assert kwargs == {'with_kwargs': True}
            self.pre = hook

        def register_forward_hook(self, hook, **kwargs):
            assert kwargs == {'with_kwargs': True, 'always_call': True}
            self.post = hook

    class Embedder:
        max_seq_length = 256

        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))
            self.auto = AutoModel()

        def __getitem__(self, _index):
            return SimpleNamespace(auto_model=self.auto)

        def encode(self, _texts, fail=False):
            # ST 6.0.1 calls the bound forward method, not Module.__call__.
            self.auto.forward(attention_mask=Mask(), fail=fail)
            return [[1.0], [2.0]]

    module = ModuleType('sentence_transformers')
    module.SentenceTransformer = Embedder
    monkeypatch.setitem(sys.modules, 'sentence_transformers', module)
    return Embedder, calls


def test_embedding_actual_forward_tokens_padding_errors_and_shared_context(engine_factory, embeddings):
    engine, cfg, _ = engine_factory()
    emb = runtime().load_embedding(cfg)
    _, calls = embeddings
    assert calls == [(cfg.embed_model, {'device': 'cpu', 'revision': cfg.embed_revision, 'local_files_only': True})]
    engine.set_context(phase='retrieval', conv_id='conv-26')
    assert emb.encode(['a', 'b']) == [[1.0], [2.0]]
    with pytest.raises(RuntimeError):
        emb.encode(['a', 'b'], fail=True)
    rows = read_all(Path(cfg.comparison_output_dir), 'embedding', 'receipt')
    assert len(rows) == 2 and [row['status'] for row in rows] == ['success', 'error']
    assert all(row['input_tokens'] == 5 and row['padded_tokens'] == 6 for row in rows)
    assert all(row['context'] == {'phase': 'retrieval', 'conv_id': 'conv-26'} for row in rows)
    assert rows[0]['usage_complete'] is True and rows[1]['usage_complete'] is False


def test_native_embedding_default_unchanged_and_comparison_wrong_cap_rejected(embeddings, tmp_path):
    embedder, calls = embeddings
    assert runtime().load_embedding(Config()).max_seq_length == 256
    assert calls == [('Qwen/Qwen3-Embedding-0.6B', {'device': 'cuda'})]
    embedder.max_seq_length = 128
    cfg = runtime().comparison_config(Config(), tmp_path)
    with pytest.raises(ValueError, match='256'):
        runtime().load_embedding(cfg)


def test_raw_file_hashes_bind_receipt_and_context_is_snapshot(engine_factory):
    engine, cfg, _ = engine_factory()
    items = [{'policy': 'full_raw', 'question_ordinal': 0}]
    engine.set_context(phase='qa', items=items)
    items[0]['question_ordinal'] = 999
    engine.generate('system', ['question'])
    root = Path(cfg.comparison_output_dir) / 'runtime' / 'generation'
    receipt = json.loads((root / '00000001.receipt.json').read_text())
    assert receipt['context']['items'][0]['question_ordinal'] == 0
    for kind in ('request', 'response'):
        assert receipt[f'{kind}_sha256'] == hashlib.sha256((root / f'00000001.{kind}.json').read_bytes()).hexdigest()
    assert not list(root.glob('*.partial'))


def test_multiple_candidate_actual_tokens_and_cached_tokens_preserved(engine_factory):
    engine, cfg, _ = engine_factory()
    engine.llm.outputs = [SimpleNamespace(request_id='multiple', prompt_token_ids=[1, 2],
        num_cached_tokens=1, outputs=[
            SimpleNamespace(text='first', token_ids=[3], finish_reason='stop'),
            SimpleNamespace(text='second', token_ids=[4, 5], finish_reason='length')])]
    assert engine.generate('system', ['question']) == ['first']
    root = Path(cfg.comparison_output_dir)
    row, = read_all(root, 'generation', 'receipt')
    assert row['prompt_tokens'] == 2 and row['completion_tokens'] == 3
    assert row['generation_count'] == 2 and row['finish_reasons'] == ['stop', 'length']
    response, = read_all(root, 'generation', 'response')
    assert response['responses'][0]['num_cached_tokens'] == 1


def test_partial_native_batch_records_known_tokens_but_rejects_success(engine_factory):
    engine, cfg, _ = engine_factory()
    engine.llm.outputs = [SimpleNamespace(request_id='partial', prompt_token_ids=[1],
        outputs=[SimpleNamespace(text='first', token_ids=[2, 3], finish_reason='stop')])]
    with pytest.raises(ValueError, match='incomplete'):
        engine.generate('system', ['first question', 'second question'])
    row, = read_all(Path(cfg.comparison_output_dir), 'generation', 'receipt')
    assert row['prompt_tokens'] == 1 and row['completion_tokens'] == 2
    assert row['usage_complete'] is False and row['status'] == 'error'


def test_unprepared_comparison_embedding_fails_before_load(embeddings):
    _, calls = embeddings
    with pytest.raises(ValueError, match='comparison_config'):
        runtime().load_embedding(Config(comparison_enabled=True))
    assert calls == []


def test_receipt_write_failure_propagates_with_raw_response_preserved(engine_factory, monkeypatch):
    engine, cfg, _ = engine_factory()
    write = engine.runtime._write

    def fail_receipt(kind, sequence, suffix, data):
        if suffix == 'receipt':
            raise OSError('journal disk full')
        return write(kind, sequence, suffix, data)

    monkeypatch.setattr(engine.runtime, '_write', fail_receipt)
    with pytest.raises(OSError, match='disk full'):
        engine.generate('system', ['question'])
    assert read_all(Path(cfg.comparison_output_dir), 'generation', 'response')
    assert not read_all(Path(cfg.comparison_output_dir), 'generation', 'receipt')


@pytest.mark.parametrize('direct', [True, False])
def test_embedding_direct_forward_and_module_call_preserve_arguments_and_output(engine_factory, embeddings, direct):
    _, cfg, _ = engine_factory()
    emb = runtime().load_embedding(cfg)
    ids, mask, marker = object(), Mask(), object()
    call = emb.auto.forward if direct else emb.auto
    assert call(ids, mask, extra=marker) is emb.auto.output
    assert emb.auto.forward_calls == [(ids, mask, False, {'extra': marker})]
    row, = read_all(Path(cfg.comparison_output_dir), 'embedding', 'receipt')
    assert row['status'] == 'success' and row['input_tokens'] == 5 and row['padded_tokens'] == 6


@pytest.mark.parametrize('suffix', ['request', 'receipt'])
def test_embedding_journal_failure_propagates_and_prewrite_prevents_forward(engine_factory, embeddings, monkeypatch, suffix):
    engine, cfg, _ = engine_factory()
    emb = runtime().load_embedding(cfg)
    write = engine.runtime._write

    def fail_write(kind, sequence, selected_suffix, data):
        if selected_suffix == suffix:
            raise OSError('embedding disk full')
        return write(kind, sequence, selected_suffix, data)

    monkeypatch.setattr(engine.runtime, '_write', fail_write)
    with pytest.raises(OSError, match='embedding disk full'):
        emb.auto.forward(attention_mask=Mask())
    assert len(emb.auto.forward_calls) == (0 if suffix == 'request' else 1)
    assert not read_all(Path(cfg.comparison_output_dir), 'embedding', 'receipt')


def test_embedding_reinstrumentation_rejected_without_double_count(engine_factory, embeddings):
    engine, cfg, _ = engine_factory()
    emb = runtime().load_embedding(cfg)
    with pytest.raises(ValueError, match='already instrumented'):
        engine.runtime.instrument_embedding(emb)
    emb.encode(['x'])
    assert len(read_all(Path(cfg.comparison_output_dir), 'embedding', 'receipt')) == 1
