"""Offline regression tests for fixed-population LoCoMo reporting."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_locomo as report
from transfer_data import paired_summary

ROOT = Path(__file__).resolve().parent.parent
POPULATIONS = ROOT / 'locomo_transfer/evaluation_populations.json'
DATA = ROOT / 'locomo_transfer/locomo3_unchanged_samples.json'


class LoCoMoReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.population_file = self.run / 'populations.json'
        self.population_file.write_bytes(POPULATIONS.read_bytes())
        (self.run / DATA.name).write_bytes(DATA.read_bytes())
        self.population = json.loads(self.population_file.read_text(encoding='utf-8'))
        self.protocol = {
            'config': {'dataset': 'locomo', 'model': 'Qwen/Qwen3.5-9B', 'data': str(DATA)},
            'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
            'model_transfer': 'all writer, source likelihood scorer, and reader roles use selected model',
        }
        self.write('protocol.json', self.protocol)
        self.write('status.json', {'phase': 'generation_complete', 'questions': 507})
        self.rows = {
            method: [
                {'question_id': row['id'], 'conversation_id': row['conv_id'],
                 'category': row['category'], 'method': method, 'prediction': 'synthetic prediction',
                 'status': 'ok', 'official_f1': float(method == report.METHODS[2])}
                for row in self.population['generated_population']
            ] for method in report.METHODS
        }
        self.save_rows()
        # Exercise real validation and cluster accounting with fewer resamples.
        original = paired_summary
        self.summary_patch = patch.object(
            report, 'paired_summary',
            side_effect=lambda *args, **kwargs: original(*args, **kwargs, n_bootstrap=20),
        )
        self.summary_patch.start()
        self.addCleanup(self.summary_patch.stop)

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value), encoding='utf-8')

    def save_rows(self):
        for method, rows in self.rows.items():
            (self.run / (method + '.jsonl')).write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8',
            )

    def summarize(self):
        return report.summarize(self.run, self.population_file)

    def test_all_declared_populations_and_empty_generation_denominator(self):
        empty_id = self.population['primary_question_population'][0]['id']
        for row in self.rows[report.METHODS[2]]:
            if row['question_id'] == empty_id:
                row['prediction'] = ''
                row['status'] = 'generation_empty'
        self.save_rows()
        result = self.summarize()
        for name, count in zip(report.POPULATIONS, (507, 377, 100)):
            for comparison in result['results'][name].values():
                self.assertEqual(comparison['n'], count)
                self.assertEqual(comparison['cluster_count'], 3)
        primary = result['results']['primary_question_population']['r40_fused_four_turn']
        self.assertEqual(primary['generation_failures_counted_zero']['refined'], 1)
        self.assertAlmostEqual(primary['refined'], 376 / 377)
        self.assertEqual(result['population_sha256'], hashlib.sha256(self.population_file.read_bytes()).hexdigest())

    def test_missing_same_id_from_every_arm_is_rejected(self):
        for rows in self.rows.values():
            rows.pop()
        self.save_rows()
        with self.assertRaises(ValueError):
            self.summarize()

    def test_duplicate_id_is_rejected(self):
        self.rows[report.METHODS[2]].append(copy.deepcopy(self.rows[report.METHODS[2]][0]))
        self.save_rows()
        with self.assertRaises(ValueError):
            self.summarize()

    def test_same_wrong_cluster_or_category_in_all_arms_is_rejected(self):
        original = copy.deepcopy(self.rows)
        for field, value in (('conversation_id', 'wrong-history'), ('category', 4)):
            with self.subTest(field=field):
                self.rows = copy.deepcopy(original)
                for rows in self.rows.values():
                    rows[0][field] = value
                self.save_rows()
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_dataset_hash_mismatch_is_rejected(self):
        self.protocol['dataset_sha256'] = '0' * 64
        self.write('protocol.json', self.protocol)
        with self.assertRaises(ValueError):
            self.summarize()

    def test_unfinished_generation_is_rejected(self):
        self.write('status.json', {'phase': 'evaluating', 'questions': 507})
        with self.assertRaises(ValueError):
            self.summarize()

    def test_outcome_selected_population_is_rejected(self):
        self.population['population_selection_used_target_outcomes'] = True
        self.population_file.write_text(json.dumps(self.population), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.summarize()


if __name__ == '__main__':
    unittest.main()

