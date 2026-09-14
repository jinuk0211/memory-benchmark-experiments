"""Behavioral checks for recursion, budget limits and source-only boundaries."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import recursive_memory as recursive


def rows(fit, audit=1.0):
    result = {str(i): {'score': score, 'split': 'probe_fit'} for i, score in enumerate(fit)}
    if audit is not None:
        result['audit'] = {'score': audit, 'split': 'probe_audit'}
    return result


class RecursiveMemoryTest(unittest.TestCase):
    def test_rejects_fit_only_improvement_when_audit_regresses(self):
        self.assertFalse(recursive.accept(rows([0]), rows([1], .9)))
        self.assertFalse(recursive.accept(rows([1]), rows([1])))
        self.assertFalse(recursive.accept(rows([0], None), rows([1], None)))
        self.assertTrue(recursive.accept(rows([0]), rows([1])))

    def test_allocation_counts_index_storage_and_preserves_input(self):
        ntok = lambda value: len(value.split())
        groups = [[{'id': 'a', 'gain': 1, 'cost': 0,
                    'unit': {'text': 'word ' * 1998, 'index_text': 'four index query words',
                             'source_probe_id': 'a'}}]]
        memory, report = recursive.allocation(groups, {'a': 1.0}, ntok)
        self.assertEqual(memory, [])
        self.assertEqual(report['actual_tokens'], 0)
        self.assertEqual(groups[0][0]['cost'], 0)

    def test_longer_rewrite_survives_when_it_retrieves_evidence_earlier(self):
        import numpy as np
        class Runtime:
            ntok = staticmethod(lambda value: len(value.split()))
            def encode(self, texts, query=False):
                return np.asarray([[1.0, 0.0] if query or 'distinct event location' in text
                                   else ([-1.0, 0.0] if text == 'cue' else [0.0, 1.0])
                                   for text in texts])
        rt = Runtime()
        parent = [{'text': f'distractor record {i}', 'sources': [f'D1:{i + 1}'], 'session': 1}
                  for i in range(8)]
        groups = [[{'id': name, 'gain': 1.0, 'cost': 0,
                    'unit': {'text': 'literal evidence', 'index_text': cue, 'sources': ['D1:20'],
                             'source_probe_id': 'fit', 'session': 1}}
                   for name, cue in (('original', 'cue'), ('rewrite', 'distinct event location'))]]
        recursive.score_index_routes(rt, parent, groups, [{'id': 'fit', 'question': 'distinct event location'}])
        original, rewritten = groups[0]
        self.assertGreater(rewritten['gain'], original['gain'])
        selected, report = recursive.allocation(groups, {'fit': 1.0}, rt.ntok)
        self.assertEqual(report['selected_options'], ['rewrite'])
        self.assertEqual(selected[0]['index_text'], 'distinct event location')

    def test_audit_probes_cannot_create_options(self):
        with self.assertRaises(ValueError):
            recursive.option_pool(None, [], [], [{'split': 'probe_audit'}])

    def test_round_two_uses_accepted_memory_and_rejected_round_is_preserved(self):
        class Runtime:
            ntok = staticmethod(lambda value: len(value.split()))
            def __init__(self):
                self.prompts = []
            def generate(self, system, prompts, max_tokens):
                self.prompts.extend(prompts)
                return ['revised cue'] * len(prompts)
        rt = Runtime()
        base = [{'text': 'base', 'sources': ['D1:1'], 'session': 1}]
        initial = base + [{'text': 'initial', 'sources': ['D1:1'], 'session': 1}]
        probes = [{'id': name, 'question': 'source question', 'answer': 'source answer',
                   'split': 'probe_fit', 'recorded_date': '1 January 2026',
                   'candidate_context_ids': ['D1:1']} for name in ('A', 'B')]
        audit = [{'id': 'audit', 'split': 'probe_audit'}]
        groups = [[{'id': name, 'kind': 'single', 'gain': 1, 'cost': 1002,
                    'unit': {'text': name + ' word' * 1000, 'index_text': 'cue',
                             'sources': ['D1:1'], 'session': 1, 'source_probe_id': name}}]
                  for name in ('A', 'B')]
        observed = []
        def rehearse(runtime, memory, tasks):
            selected = {unit.get('source_probe_id') for unit in memory}
            observed.append(selected)
            scores = (1, .5) if 'A' in selected else ((0, 1) if 'B' in selected else (0, .5))
            return {name: {'score': score, 'split': 'probe_fit', 'context': 'retrieved evidence'}
                    for name, score in zip(('A', 'B'), scores)} | {
                        'audit': {'score': 1.0, 'split': 'probe_audit', 'context': 'audit evidence'}}
        saved = []
        sessions = [{'turns': [{'id': 'D1:1', 'text': 'source text'}]}]
        with patch.object(recursive, 'option_pool', return_value=groups), patch.object(recursive, 'rehearse', side_effect=rehearse), patch.object(recursive, 'score_index_routes'):
            final, history = recursive.refine_memory(rt, sessions, base, initial, probes, audit,
                lambda round_id, memory, record: saved.append((round_id, record['accepted'])))
        self.assertEqual(saved, [(1, True), (2, False), (3, False)])
        self.assertEqual(history[1]['cumulative_residual']['A'], 1.0)
        self.assertEqual(history[2]['cumulative_residual']['A'], 1.0)
        self.assertEqual(final[-1]['source_probe_id'], 'A')
        self.assertEqual(len(observed), 4)
        self.assertTrue(rt.prompts)
        self.assertTrue(all(record['actual_extra_tokens'] <= 2000 for record in history))
        self.assertEqual(initial[-1]['text'], 'initial')


if __name__ == '__main__':
    unittest.main()
