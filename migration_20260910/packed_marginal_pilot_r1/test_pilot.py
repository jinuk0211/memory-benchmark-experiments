"""CPU-only semantic guards for the source-only feasibility pilot."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pilot


def job(lp: float, *, context: str = 'c', pid: str = 'p', answer: str = 'a') -> dict:
    return {'context': context, 'probe_id': pid, 'question': 'QA', 'answer': answer,
            'score': {'mean_logprob': lp}}


def row(pid: str, cid: str = 'c') -> dict:
    return {'id': pid, 'conv_id': cid, 'split': 'probe_fit',
            'analysis': {'source_dependent': True},
            'generations': {'full': {'source_answer_f1': 1.0, 'source_ids': ['s']}},
            'subsets': [{'kind': 'empty', 'source_ids': [], 'score': {'mean_logprob': -3.}},
                        {'kind': 'full', 'source_ids': ['s'], 'score': {'mean_logprob': -1.}}]}


class Guards(unittest.TestCase):
    def test_hash_selection_is_input_order_independent(self) -> None:
        rows = [row(str(i), cid) for cid in ('a', 'b') for i in range(20)]
        for r in rows:
            r['id'] = r['conv_id'] + r['id']
        audit = [{'id': f'{cid}-audit-{i}', 'conv_id': cid, 'split': 'probe_audit'}
                 for cid in ('a', 'b') for i in range(5)]
        selected = pilot.select('locomo', rows, audit)
        self.assertEqual(selected, pilot.select('locomo', list(reversed(rows)), list(reversed(audit))))
        self.assertEqual(tuple(map(len, selected[1:])), (16, 4, 4))

    def test_original_admission_is_source_fit_only(self) -> None:
        r = row('p')
        self.assertTrue(pilot.admissible(r))
        r['split'] = 'probe_audit'
        self.assertFalse(pilot.admissible(r))
        r['split'] = 'probe_fit'
        r['generations']['full']['source_answer_f1'] = .79
        self.assertFalse(pilot.admissible(r))

    def test_zero_and_negative_packed_gain(self) -> None:
        self.assertEqual(pilot.value({'context': 'same'}, {'context': 'same'}, {}, {}), 0.)
        self.assertEqual(pilot.value(job(-3, context='new'), job(-1), job(-1), job(-3)), 0.)
        self.assertEqual(pilot.value(job(-1, context='new'), job(-3), job(-2.95), job(-3)), 0.)
        self.assertEqual(pilot.value(job(-1, context='new'), job(-3), job(-1), job(-3)), 2.)

    def test_nonfinite_any_scored_control_is_rejected(self) -> None:
        values = [job(-1, context='new'), job(-3), job(-1), job(-3)]
        for i in range(4):
            for bad in (float('nan'), float('inf'), float('-inf')):
                altered = copy.deepcopy(values)
                altered[i]['score']['mean_logprob'] = bad
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    pilot.value(*altered)

    def test_macro_gate_rejects_changed_reference_or_denominator(self) -> None:
        with self.assertRaisesRegex(ValueError, 'identities'):
            pilot.paired_delta([job(-1)], [job(-1, answer='changed')])
        with self.assertRaisesRegex(ValueError, 'identities'):
            pilot.paired_delta([job(-1)], [])
        self.assertFalse(pilot.gate(([job(-1)], [job(-2)]), ([job(-1)], [job(-1)]), (1., 1.)))
        self.assertFalse(pilot.gate(([job(-2)], [job(-1)]), ([job(-1)], [job(-2)]), (1., 1.)))
        self.assertTrue(pilot.gate(([job(-2)], [job(-1)]), ([job(-1)], [job(-1)]), (1., 1.)))

    def test_answer_count_and_f1_guards(self) -> None:
        scorer = SimpleNamespace(generate=lambda controls: {'wrong': {'source_answer_f1': 1.}})
        with patch.object(pilot, 'generation_guard'):
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                pilot.answer_scores(scorer, [job(-1)])
            scorer.generate = lambda controls: {'0': {'source_answer_f1': float('nan')}}
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                pilot.answer_scores(scorer, [job(-1)])

    def test_generation_cap_includes_actual_chat_template(self) -> None:
        scorer = SimpleNamespace(tok=SimpleNamespace(apply_chat_template=lambda *a, **k: 'rendered'),
                                 ntok=lambda prompt: 8097)
        with patch.dict('sys.modules', {'refine': SimpleNamespace(READER='reader')}):
            with self.assertRaisesRegex(ValueError, '8192'):
                pilot.generation_guard(scorer, [{'user': 'query'}])
            scorer.ntok = lambda prompt: 8096
            pilot.generation_guard(scorer, [{'user': 'query'}])

    def test_insufficient_views_does_not_initialize_gpu(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp)
            pilot.lock(out / 'views' / 'a.json', {})
            p = {'adapter_root': 'unused', 'v2_root': 'unused', 'histories': {'a': {}}}
            with patch.object(pilot, 'verified', return_value=p), \
                 patch.object(pilot, 'support', return_value=(SimpleNamespace(), {})), \
                 patch.object(pilot, 'checked_views', return_value={'accepted_count': 0}):
                pilot.score(out)
            self.assertEqual(pilot.read(out / 'CALIBRATION_STATUS.json')['status'], 'insufficient_evidence')
            self.assertFalse((out / 'COMPLETE.json').exists())
            self.assertFalse((out / 'memory_lock.json').exists())

    def test_immutable_artifact_and_protocol_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp)
            pilot.lock(out / 'protocol.json', {'first': True})
            pilot.lock(out / 'PLAN_LOCK.json', {'protocol_sha256': pilot.sha(out / 'protocol.json')})
            with self.assertRaisesRegex(ValueError, 'Frozen artifact'):
                pilot.lock(out / 'protocol.json', {'first': False})
            (out / 'protocol.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                pilot.verified(out)

    def test_existing_helper_is_identical_except_zero_return(self) -> None:
        root = Path(__file__).parent.parent
        approved_import = Path(os.environ.get('PACKED_PILOT_IMPORT_SOURCE', root / 'seed_parent_ablation/import_source.py'))
        v2 = Path(os.environ.get('PACKED_PILOT_V2_ROOT', root.parent / 'experiments/recursive_minilm_20260909'))
        self.assertEqual(approved_import.read_bytes(),
                         Path(__file__).with_name('import_source.py').read_bytes())
        old = (v2 / 'source_views_v2.py').read_text()
        removed = "    if not records:\n        raise ValueError('No source-supported paired views; never fall back to the indexed question')\n"
        self.assertEqual(Path(__file__).with_name('source_views_v2_pilot.py').read_text(), old.replace(removed, ''))


if __name__ == '__main__':
    unittest.main(verbosity=2)
