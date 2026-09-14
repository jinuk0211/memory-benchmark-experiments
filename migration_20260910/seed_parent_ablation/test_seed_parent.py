"""Behavioral checks for source imports, immutable construction, and reader identity."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'source')]
import run_transfer as runner
import import_source as source
import refine as core
from portable_parent import augment
from budgeted_evidence import storage_cost


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


class ImportFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'origin'
        self.sample = {'sample_id': 'conv-1', 'conversation': {
            'session_1_date_time': '1:00 pm on 8 May, 2023',
            'session_1': [{'dia_id': 'D1:1', 'speaker': 'Alice', 'text': 'I own a blue bicycle.'}]},
            'qa': [{'question': 'held-out question', 'answer': 'held-out answer', 'category': 1}]}
        self.sessions = {'conv-1': core.session_data(self.sample)}
        self.probe = {'id': 'conv-1:source_probe:0', 'conv_id': 'conv-1', 'split': 'probe_fit',
                      'question': 'What does Alice own?', 'answer': 'blue bicycle',
                      'candidate_context_ids': ['D1:1'], 'source_ids': ['D1:1']}
        self.pool = {'records': [self.probe]}
        self.protocol = {'config': {'dataset': 'locomo'}, 'methods': list(source.ORIGINAL_METHODS),
                         'selection': {'selected_ids': ['conv-1']}, 'target_selection': False,
                         'history_truncation': False}
        self.memory = [{'text': 'Alice owns a blue bicycle.', 'sources': ['D1:1'], 'kind': 'fact'}]
        files = {'protocol.json': self.protocol, 'source_sessions.json': self.sessions,
                 'source_generations.json': {'records': []}, 'source_probes.json': self.pool,
                 'utility_selection.json': {'ids': [self.probe['id']], 'source_pool_sha256': core.digest(self.pool)},
                 'utility/items/conv-1_source_probe_0.json': {**self.probe, 'skip': 'fixture'}}
        memories = {m: {'conv-1': self.memory} for m in source.ORIGINAL_METHODS}
        files['memory_lock.json'] = {'protocol_sha256': core.digest(self.protocol),
            'source_sha256': core.digest(self.sessions), 'memories_sha256': core.digest(memories),
            'benchmark_questions_used': False}
        for method in source.ORIGINAL_METHODS:
            files[f'memories/{method}/conv-1.json'] = self.memory
        for name, value in files.items():
            save(self.root / name, value)
        self.pin = patch.dict(source.PINNED_PROTOCOLS, {'locomo': source.sha(self.root / 'protocol.json')})
        self.pool_patch = patch.object(source, 'split_probe_pool', return_value=self.pool)
        self.pin.start()
        self.pool_mock = self.pool_patch.start()

    def tearDown(self):
        self.pool_patch.stop()
        self.pin.stop()
        self.temp.cleanup()

    def test_import_keeps_source_only_and_rechecks_origin_memory_lock(self):
        out = Path(self.temp.name) / 'new'
        # Unrelated outcome/cache files exist, but the explicit import must exclude them.
        save(self.root / 'items/seed/conv-1.json', [{'gold': 'held-out answer'}])
        save(self.root / 'cache/private.json', {'text': 'held-out answer'})
        lineage = source.snapshot(self.root, out, 'locomo')
        self.assertEqual(len(lineage['selected_probe_ids']), 1)
        self.assertEqual(lineage['cost_policy'], source.COST_POLICY)
        self.assertFalse((out / 'imported_source/items').exists())
        self.assertFalse((out / 'imported_source/cache').exists())
        source.verify_snapshot(out / 'imported_source', lineage)

    def test_import_rejects_wrong_original_protocol(self):
        save(self.root / 'protocol.json', {**self.protocol, 'target_selection': True})
        with self.assertRaisesRegex(ValueError, 'protocol'):
            source.imported_files(self.root, 'locomo')

    def test_import_rejects_changed_seed_under_original_lock(self):
        save(self.root / 'memories/seed/conv-1.json', [{**self.memory[0], 'text': 'changed'}])
        with self.assertRaisesRegex(ValueError, 'memory lock'):
            source.imported_files(self.root, 'locomo')

    def test_import_rejects_foreign_probe_utility(self):
        save(self.root / 'utility/items/conv-1_source_probe_0.json', {**self.probe, 'question': 'external QA', 'skip': 'fixture'})
        with self.assertRaisesRegex(ValueError, 'source-fit'):
            source.imported_files(self.root, 'locomo')

    def test_import_rejects_incomplete_utility(self):
        save(self.root / 'utility/items/conv-1_source_probe_0.json', self.probe)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            source.imported_files(self.root, 'locomo')

    def test_import_rejects_missing_or_extra_selected_ids(self):
        save(self.root / 'utility_selection.json', {'ids': [], 'source_pool_sha256': core.digest(self.pool)})
        with self.assertRaisesRegex(ValueError, 'Selected utility IDs'):
            source.imported_files(self.root, 'locomo')

    def test_import_rejects_unreproducible_source_pool(self):
        self.pool_mock.return_value = {'records': []}
        with self.assertRaisesRegex(ValueError, 'cannot be reproduced'):
            source.imported_files(self.root, 'locomo')

    def test_import_rejects_audit_instead_of_fit(self):
        pool = deepcopy(self.pool)
        pool['records'][0]['split'] = 'probe_audit'
        save(self.root / 'source_probes.json', pool)
        self.pool_mock.return_value = pool
        with self.assertRaisesRegex(ValueError, 'source-fit'):
            source.imported_files(self.root, 'locomo')

    def test_snapshot_rejects_changed_bytes_and_extra_files(self):
        out = Path(self.temp.name) / 'new'
        lineage = source.snapshot(self.root, out, 'locomo')
        root = out / 'imported_source'
        save(root / 'unexpected.json', {})
        with self.assertRaisesRegex(ValueError, 'inventory'):
            source.verify_snapshot(root, lineage)
        (root / 'unexpected.json').unlink()
        (root / 'source_sessions.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            source.verify_snapshot(root, lineage)

    def test_path_escape_and_noncanonical_names_rejected(self):
        for name in ('../origin/protocol.json', '/protocol.json', 'a\\b', 'a//b', 'C:/x'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                source.safe_file(self.root, name)

    def test_full_source_history_check_rejects_omission(self):
        runner.verify_histories([self.sample], self.sessions)
        altered = deepcopy(self.sessions)
        altered['conv-1'][0]['turns'][0]['body'] = 'truncated'
        with self.assertRaisesRegex(ValueError, 'Full canonical'):
            runner.verify_histories([self.sample], altered)

    def test_seed_parent_unchanged_with_no_admissible_options(self):
        parent = deepcopy(self.memory)
        rt = SimpleNamespace(ntok=lambda text: len(text.split()))
        memory, details, _ = augment(rt, self.sessions['conv-1'], parent, [])
        self.assertEqual(parent, self.memory)
        self.assertEqual(memory, self.memory)
        self.assertEqual(details['storage_tokens'], details['parent_tokens'])

    def test_parent_augmentation_respects_key_plus_payload_budget(self):
        rt = SimpleNamespace(ntok=len)
        parent = deepcopy(self.memory)
        key = 'k' * 1980
        unit = {'text': parent[0]['text'], 'sources': ['D1:1'], 'index_text': key}
        option = {'id': 'p:single:D1:1', 'kind': 'single', 'unit': unit,
                  'cost': storage_cost(unit, rt.ntok), 'gain': 1.0}
        import parent_evidence
        with patch('portable_parent.make_options', return_value=([[option]], {option['id']: option})):
            memory, details, _ = augment(rt, self.sessions['conv-1'], parent, [])
        self.assertEqual(memory, parent)  # Payload plus key exceeds 2000 even though key alone fits.
        self.assertEqual(details['storage_tokens'], details['parent_tokens'])
        self.assertIsNotNone(parent_evidence.construct)

    def test_frozen_reader_source_and_timing_sources_identical(self):
        self.assertEqual(inspect.getsource(runner.evaluate_sample), inspect.getsource(runner.approved.evaluate_sample))
        self.assertEqual(Path(inspect.getsourcefile(runner.evaluate_sample)).resolve(), ROOT / 'run_transfer.py')

    def test_construct_cli_rejects_target_dataset(self):
        with patch.object(sys, 'argv', ['run_transfer.py', 'construct', '--out', 'x', '--tokenizer', 'tok', '--data', 'qa.json']):
            with self.assertRaises(SystemExit):
                runner.main()

    def test_construct_does_not_read_target_samples(self):
        self.assertNotIn('samples_for(', inspect.getsource(runner.construct))
        self.assertNotIn('args.data', inspect.getsource(runner.construct))

    def test_tokenizer_pin_rejects_changed_file(self):
        root = Path(self.temp.name) / 'tok'
        root.mkdir()
        (root / 'config.json').write_text('wrong', encoding='utf-8')
        with patch.dict(source.TOKENIZER_FILES, {'config.json': '0' * 64}, clear=True):
            with self.assertRaisesRegex(ValueError, 'tokenizer file changed'):
                source.verify_tokenizer(root)


if __name__ == '__main__':
    unittest.main()

