"""Failures that could produce artificial v2 improvements or target leakage."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
import recursive_memory as base
import recursive_memory_v2 as v2
import source_views_v2 as views
import run_recursive_v2 as runner

Q0 = 'What place did Dana use for weekly yoga practice?'
QA = 'Where would Dana attend her regular yoga sessions?'
QB = 'Which venue hosted the yoga classes Dana attended?'
REWRITE = 'At what location did Dana take weekly yoga classes?'
ROW = {'id': 'fit', 'conv_id': 'conv', 'question': Q0, 'answer': 'Riverside studio',
       'split': 'probe_fit', 'recorded_date': '1 January 2026',
       'candidate_context_ids': ['D1:1'], 'source_ids': ['D1:1']}
SESSIONS = [{'turns': [{'id': 'D1:1', 'text': 'Dana practiced yoga at Riverside studio.'}]}]
PARENT = [{'text': 'General dialogue context', 'sources': ['D1:1'], 'session': 1}]
AUDIT = [{'id': 'audit', 'conv_id': 'conv', 'question': 'Who practiced yoga?', 'answer': 'Dana', 'split': 'probe_audit'}]


def bundle():
    manifest = {'policy': views.VIEW_POLICY, 'input_sha256': core.digest([ROW]),
                'stored_cues_sha256': core.digest([]), 'records': [{'id': 'fit', 'original': copy.deepcopy(ROW),
                'fit_question': QA, 'diagnostic_question': QB}]}
    return {'manifest': manifest, 'manifest_sha256': core.digest(manifest)}


class Runtime:
    ntok = staticmethod(lambda text: len(text.split()))
    def __init__(self, rewrite=REWRITE):
        self.calls = []
        self.rewrite = rewrite
    def generate(self, system, users, max_tokens):
        self.calls.append((system, users))
        if system == views.WRITER:
            return [json.dumps({'question_1': QA, 'question_2': QB}) for _ in users]
        if system in (views.QUESTION_CHECK, views.VERIFIER):
            return ['EQUIVALENT'] * len(users)
        if system == base.REINDEX:
            return [self.rewrite] * len(users)
        return ['Riverside studio'] * len(users)


class SourceViewsTest(unittest.TestCase):
    def test_source_calibration_keeps_original_answer_and_fixed_roles(self):
        rt = Runtime()
        result = views.prepare_views(rt, SESSIONS, [ROW], PARENT)
        fit = views.fit_rows(result, [ROW])
        self.assertEqual(fit[0]['answer'], ROW['answer'])
        self.assertNotEqual(fit[0]['question'], Q0)
        self.assertEqual(result['accepted_count'], 1)
        self.assertEqual(result['manifest']['records'][0]['original'], ROW)
        self.assertTrue(all('General dialogue context' not in prompt for _, users in rt.calls for prompt in users))

    def test_missing_generation_cannot_reduce_denominator(self):
        rt = Runtime()
        with patch.object(rt, 'generate', return_value=[]), self.assertRaisesRegex(ValueError, 'Missing'):
            views.prepare_views(rt, SESSIONS, [ROW], PARENT)

    def test_changed_scope_or_unknown_fidelity_rejects_pair(self):
        for rejected_system, verdict in ((views.QUESTION_CHECK, 'DIFFERENT'), (views.VERIFIER, 'UNKNOWN')):
            rt = Runtime()
            original = rt.generate
            def generate(system, users, max_tokens):
                return [verdict] * len(users) if system == rejected_system else original(system, users, max_tokens)
            with patch.object(rt, 'generate', side_effect=generate), self.assertRaisesRegex(ValueError, 'No source-supported'):
                views.prepare_views(rt, SESSIONS, [ROW], PARENT)

    def test_json_arrays_and_duplicate_fields_are_not_valid_views(self):
        for value in ('["question_1", "question_2"]', '{"question_1":"a","question_1":"b","question_2":"c"}'):
            rt = Runtime()
            with patch.object(rt, 'generate', return_value=[value]), self.assertRaises(ValueError):
                views.prepare_views(rt, SESSIONS, [ROW], PARENT)

    def test_question_copy_inside_longer_index_is_detected(self):
        self.assertTrue(views.copies_view('Find evidence: ' + QA + ' Please.', {views.fingerprint(QA)}))
        self.assertFalse(views.copies_view(Q0, {views.fingerprint(QA)}))
        with self.assertRaisesRegex(ValueError, 'No source-supported'):
            views.prepare_views(Runtime(), SESSIONS, [ROW], [{'text': 'Find evidence: ' + QA}])

    def test_wrapping_original_question_does_not_count_as_independent_view(self):
        rt = Runtime()
        generated = json.dumps({'question_1': 'According to the conversation, ' + Q0, 'question_2': QB})
        with patch.object(rt, 'generate', return_value=[generated]), self.assertRaisesRegex(ValueError, 'No source-supported'):
            views.prepare_views(rt, SESSIONS, [ROW], PARENT)

    def test_manifest_rejects_changed_question_answer_or_partition(self):
        for field in ('question', 'answer', 'split'):
            changed = copy.deepcopy(ROW)
            changed[field] = 'changed'
            with self.assertRaises(ValueError):
                views.fit_rows(bundle(), [changed])
        corrupted = bundle()
        corrupted['manifest']['records'][0]['fit_question'] = 'changed'
        with self.assertRaises(ValueError):
            views.fit_rows(corrupted, [ROW])


class RecursiveV2Test(unittest.TestCase):
    def test_different_denominators_cannot_be_accepted(self):
        expected = {'fit': 'f', 'audit': 'a'}
        old = {'fit': {'score': 0, 'split': 'probe_fit', 'task_sha256': 'f'},
               'audit': {'score': 1, 'split': 'probe_audit', 'task_sha256': 'a'}}
        new = copy.deepcopy(old)
        new['other'] = new.pop('fit')
        with self.assertRaises(ValueError):
            v2.accept(old, new, expected)
        new = copy.deepcopy(old)
        new['fit']['score'] = 1
        new['audit']['score'] = .9
        self.assertFalse(v2.accept(old, new, expected))
        new['audit']['score'] = 1
        self.assertTrue(v2.accept(old, new, expected))

    def test_missing_or_changed_rehearsal_tasks_fail_before_scoring(self):
        probes = [ROW] + AUDIT
        expected = v2.contract(probes)
        with patch.object(base, 'rehearse') as call, self.assertRaises(ValueError):
            v2.rehearse(Runtime(), PARENT, probes[:1], expected)
        call.assert_not_called()
        with patch.object(base, 'rehearse', return_value={}), self.assertRaises(ValueError):
            v2.rehearse(Runtime(), PARENT, probes, expected)

    def run_refinement(self, rewrite=REWRITE):
        rt = Runtime(rewrite)
        groups = [[{'id': 'fit:literal', 'gain': 1.0, 'cost': 0,
                    'unit': {'text': 'Dana practiced yoga at Riverside studio.', 'index_text': Q0,
                             'source_probe_id': 'fit', 'sources': ['D1:1'], 'session': 1}}]]
        def options(runtime, sessions, parent, rows):
            self.assertEqual(rows, [ROW])
            return copy.deepcopy(groups)
        def rank(runtime, parent, candidate_groups, probes):
            self.assertEqual([r['question'] for r in probes], [QA])
            for group in candidate_groups:
                for option in group:
                    option['gain'] = 1 if option['unit']['index_text'] == REWRITE else .01
        def score(runtime, memory, probes):
            rewritten = any(u.get('index_text') == REWRITE for u in memory)
            self.assertEqual(probes[0]['question'], QA)
            return {'fit': {'score': .9 if rewritten else .2, 'split': 'probe_fit', 'context': 'retrieved source'},
                    'audit': {'score': 1.0, 'split': 'probe_audit', 'context': 'audit context'}}
        with patch.object(base, 'option_pool', side_effect=options), patch.object(base, 'score_index_routes', side_effect=rank), patch.object(base, 'rehearse', side_effect=score):
            final, histories = v2.refine_memory(rt, SESSIONS, PARENT, PARENT, [ROW], AUDIT,
                                               bundle(), lambda *args: None)
        return rt, final, histories

    def test_rewrites_use_q0_and_can_improve_qa_with_fixed_reader_acceptance(self):
        rt, final, history = self.run_refinement()
        self.assertEqual([r['accepted'] for r in history], [False, True, False])
        self.assertEqual(final[-1]['index_text'], REWRITE)
        prompts = [p for system, users in rt.calls if system == base.REINDEX for p in users]
        self.assertTrue(prompts)
        self.assertTrue(all(Q0 in p and QA not in p and QB not in p for p in prompts))
        self.assertEqual(history[0]['rehearsal_contract'], history[2]['rehearsal_contract'])
        self.assertTrue(all(r['actual_extra_tokens'] <= 2000 for r in history))

    def test_rewriter_cannot_copy_fit_or_post_lock_question(self):
        for query in (QA, QB, 'Evidence lookup: ' + QA):
            _, final, history = self.run_refinement(query)
            self.assertEqual(final, PARENT)
            self.assertFalse(any(r['accepted'] for r in history))
            self.assertFalse(history[1]['rewrite_checks'][0]['allowed'])

    def test_longer_cue_gets_better_actual_rank_on_distinct_query(self):
        import numpy as np
        class Encoder(Runtime):
            def encode(self, texts, query=False):
                return np.asarray([[1., 0.] if query or t == REWRITE else ([-1., 0.] if t == Q0 else [0., 1.]) for t in texts])
        groups = [[{'id': name, 'gain': 1., 'cost': 0, 'unit': {'text': 'Riverside studio evidence',
                   'index_text': cue, 'source_probe_id': 'fit', 'sources': ['D1:1'], 'session': 1}}
                   for name, cue in (('original', Q0), ('rewrite', REWRITE))]]
        parent = [{'text': 'distractor ' + str(i), 'sources': ['D1:1'], 'session': 1} for i in range(10)]
        rt = Encoder()
        base.score_index_routes(rt, parent, groups, [{'id': 'fit', 'question': QA}])
        self.assertNotIn(QA, [o['unit']['index_text'] for o in groups[0]])
        self.assertGreater(groups[0][1]['gain'], groups[0][0]['gain'])
        _, report = base.allocation(groups, {'fit': 1.0}, rt.ntok)
        self.assertEqual(report['selected_options'], ['rewrite'])

    def test_construct_blocks_benchmark_access_and_validates_partial_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'run'
            environment = root / 'environment.json'
            environment.write_text('{}', encoding='utf8')
            benchmark = root / 'benchmark.json'
            benchmark.write_text('benchmark QA must not be read', encoding='utf8')
            parent_protocol = {'config': {'model': 'fixed', 'embed_model': 'fixed', 'seed': 20260907,
                              'embed_batch_size': 8}, 'dataset_sha256': 'unused during construction'}
            baseline = root / 'baseline'
            core.save(baseline / 'memory_lock.json', {})
            starting = {name: {'conv': PARENT} for name in ('r40_fused_four_turn', 's_parent_single_2000')}
            def refine(rt, sessions, parent, initial, utility, audit, source_views, save_round):
                history = []
                for round_id in range(1, 4):
                    record = {'round': round_id, 'accepted': False, 'before': {}, 'after': {},
                              'accepted_memory_sha256': core.digest(PARENT)}
                    save_round(round_id, PARENT, record)
                    history.append(record)
                return PARENT, history
            def diagnostic_rehearsal(rt, memory, probes):
                self.assertTrue((out / 'memory_lock.json').exists())
                self.assertEqual(probes[0]['question'], QB)
                return {'fit': {'score': 1, 'split': 'probe_view_audit'}}
            original_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path == benchmark:
                    raise AssertionError('Construction tried to open benchmark QA')
                return original_open(path, *args, **kwargs)
            argv = ['run_recursive_v2.py', 'construct', '--baseline', str(baseline), '--out', str(out),
                    '--environment', str(environment), '--data', str(benchmark)]
            with patch.object(sys, 'argv', argv), patch.object(Path, 'open', guarded_open), \
                 patch.object(runner.v1, 'verify_baseline', return_value=(parent_protocol, {'conv': SESSIONS}, starting)), \
                 patch.object(runner, 'source_inputs', return_value=([ROW], [ROW] + AUDIT, {})), \
                 patch.object(runner, 'Runtime', return_value=Runtime()), \
                 patch.object(runner, 'prepare_views', return_value=bundle()), \
                 patch.object(runner, 'refine_memory', side_effect=refine) as construct, \
                 patch.object(runner.rehearsal, 'rehearse', side_effect=diagnostic_rehearsal):
                runner.main()
                self.assertEqual(construct.call_count, 1)
                runner.main()
                self.assertEqual(construct.call_count, 1)
                self.assertTrue((out / 'complete/conv.json').exists())
                core.save(out / 'memories/conv.json', [{'text': 'unverified replacement'}])
                with self.assertRaisesRegex(ValueError, 'final memory differs'):
                    runner.main()
                self.assertEqual(construct.call_count, 1)

    def test_diagnostic_requires_final_memory_and_view_lock_before_inference(self):
        rt = Runtime()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            runner.diagnostic(rt, Path(tmp), {}, {'conv': PARENT}, {'conv': SESSIONS}, {'conv': bundle()})
        self.assertFalse(rt.calls)


if __name__ == '__main__':
    unittest.main()
