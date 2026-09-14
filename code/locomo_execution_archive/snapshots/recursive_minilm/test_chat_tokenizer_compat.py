"""CPU regressions for the TF5 dictionary default and frozen NLL token spans."""
from collections import UserDict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
from evidence_utility import answer_token_request, extract_answer_logprob
from chat_tokenizer_compat import ListChatTokenizer


class FakeBatchEncoding(UserDict):
    """Mapping whose iteration reproduces Transformers BatchEncoding keys."""


class NewDefaultTokenizer:
    name_or_path = 'synthetic-tokenizer'

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append(('encode', text, add_special_tokens))
        return [201, 202]

    def apply_chat_template(self, messages, *, tokenize=True, return_dict=True, **kwargs):
        self.calls.append(('chat', messages, tokenize, return_dict, kwargs))
        if not tokenize:
            return '<synthetic rendered chat>'
        if return_dict:
            return FakeBatchEncoding({'input_ids': [101, 102, 103], 'attention_mask': [1, 1, 1]})
        return [101, 102, 103]


class ChatTokenizerCompatibilityTests(unittest.TestCase):
    def test_frozen_nll_request_uses_integer_prefix_and_exact_answer_span(self):
        raw = NewDefaultTokenizer()
        broken_request, broken_start, _ = answer_token_request(raw, 'question', 'answer')
        self.assertEqual(broken_request['prompt_token_ids'][:broken_start], ['input_ids', 'attention_mask'])

        request, start, target = answer_token_request(ListChatTokenizer(raw), 'question', 'answer')
        self.assertEqual(request, {'prompt_token_ids': [101, 102, 103, 201, 202]})
        self.assertTrue(all(isinstance(token_id, int) for token_id in request['prompt_token_ids']))
        self.assertEqual(start, 3)
        self.assertEqual(target, [201, 202])
        self.assertEqual(request['prompt_token_ids'][start:], target)
        output = SimpleNamespace(
            prompt_token_ids=request['prompt_token_ids'],
            prompt_logprobs=[None, None, None,
                             {201: SimpleNamespace(logprob=-0.2)},
                             {202: SimpleNamespace(logprob=-0.4)}],
        )
        result = extract_answer_logprob(output, start, target)
        self.assertEqual(result['answer_tokens'], 2)
        self.assertAlmostEqual(result['mean_logprob'], -0.3)
        self.assertEqual(result['token_logprobs'], [-0.2, -0.4])

    def test_explicit_dictionary_request_is_preserved(self):
        tokenizer = ListChatTokenizer(NewDefaultTokenizer())
        result = tokenizer.apply_chat_template([], tokenize=True, return_dict=True)
        self.assertIsInstance(result, FakeBatchEncoding)
        self.assertEqual(result['input_ids'], [101, 102, 103])
        self.assertEqual(tokenizer.apply_chat_template([], return_dict=False), [101, 102, 103])

    def test_render_encode_and_attributes_are_unchanged(self):
        raw = NewDefaultTokenizer()
        tokenizer = ListChatTokenizer(raw)
        messages = [{'role': 'user', 'content': 'question'}]
        self.assertEqual(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ), raw.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))
        self.assertEqual(tokenizer.encode('answer', add_special_tokens=False), [201, 202])
        self.assertEqual(raw.calls[-1], ('encode', 'answer', False))
        self.assertEqual(tokenizer.name_or_path, raw.name_or_path)
        self.assertEqual(raw.calls[0][4], {'add_generation_prompt': True, 'enable_thinking': False})


if __name__ == '__main__':
    unittest.main()
