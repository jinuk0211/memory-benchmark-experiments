"""Full canonical QA identity and official scoring contracts; CPU only."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import score_refined_full as scorer


def fixture():
    data, rows = [], []
    for number in range(10):
        sample = {'sample_id': f'conv-{number}', 'qa': []}
        for index in range(154):
            sample['qa'].append({'question': 'Repeated question', 'answer': str(index),
                                 'category': index % 4 + 1, 'evidence': ['D1:1']})
        sample['qa'].insert(3, {'question': 'Excluded', 'answer': '', 'category': 5})
        data.append(sample)
        for index, qa in enumerate(sample['qa']):
            if qa['category'] == 5:
                continue
            rows.append({'question_id': f'{sample["sample_id"]}:{index}',
                         'conversation_id': sample['sample_id'], 'method': 's_parent_single_2000',
                         'question': qa['question'], 'gold': str(qa['answer']),
                         'category': qa['category'], 'prediction': 'answer',
                         'hypothesis': 'answer', 'official_f1': 0.5})
    return data, rows


class RefinedOfficialTests(unittest.TestCase):
    def setUp(self):
        self.data, self.rows = fixture()

    def report(self, rows=None, score_fn=None):
        return scorer.build_report(self.data, self.rows if rows is None else rows,
                                   score_fn or (lambda records: [0.5] * len(records)))

    def test_exact_1540_preserves_duplicate_question_text_and_original_indices(self):
        seen = []
        def scores(records):
            seen.extend(records)
            return [0.5] * len(records)
        result = self.report(score_fn=scores)
        self.assertTrue(result['predictions_complete'])
        self.assertEqual(result['qa_count'], 1540)
        self.assertEqual(result['official_f1'], 0.5)
        self.assertEqual(len(seen), 1540)
        self.assertEqual(len({(r['sample'], r['index']) for r in seen}), 1540)
        self.assertEqual(seen[3]['index'], 4)
        self.assertEqual(seen[0]['evidence'], ['D1:1'])

    def test_incomplete_duplicate_and_extra_rows_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]],
                     self.rows[:-1] + [self.rows[0]]):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                self.report(rows)

    def test_metadata_and_wrong_method_fail_before_scoring(self):
        for key, value in [('method', 'seed'), ('method', 'r40_fused_four_turn'),
                           ('conversation_id', 'other'), ('question_id', 'conv-0:3'),
                           ('question', 'different'), ('gold', 'changed'),
                           ('category', 5), ('prediction', None), ('hypothesis', 'different')]:
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.report(rows, lambda _: self.fail('Scored invalid records'))

    def test_empty_native_answer_stays_in_denominator(self):
        self.rows[0].update(prediction='', hypothesis='')
        result = self.report()
        self.assertEqual(result['qa_count'], 1540)
        self.assertEqual(result['empty_predictions'], 1)

    def test_official_scorer_invalid_output_rejected(self):
        for values in ([0.5] * 1539, [math.nan] * 1540, [1.1] * 1540):
            with self.subTest(first=values[0]), self.assertRaises(ValueError):
                self.report(score_fn=lambda _, v=values: v)

    def test_wrong_population_rejected(self):
        self.data.pop()
        with self.assertRaises(ValueError):
            self.report()

    def test_cli_scores_exact_input_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'data.json'
            predictions = root / 'predictions.jsonl'
            official = root / 'official.py'
            output = root / 'official_scores.json'
            dataset.write_text(json.dumps(self.data), encoding='utf-8')
            predictions.write_text('\n'.join(json.dumps(row) for row in self.rows), encoding='utf-8')
            official.write_text('def eval_question_answering(records, eval_key, metric):\n'
                                '    assert eval_key == "prediction" and metric == "f1"\n'
                                '    return [0.25] * len(records), None, None\n', encoding='utf-8')
            args = ['--dataset', str(dataset), '--predictions', str(predictions),
                    '--scorer', str(official), '--output', str(output)]
            with patch.object(scorer, 'DATASET_SHA256', hashlib.sha256(dataset.read_bytes()).hexdigest()), \
                    patch.object(scorer, 'SCORER_SHA256', hashlib.sha256(official.read_bytes()).hexdigest()):
                self.assertEqual(scorer.main(args), 0)
                report = json.loads(output.read_text())
                self.assertEqual(report['official_f1'], 0.25)
                self.assertEqual(report['sources_sha256']['predictions'], hashlib.sha256(predictions.read_bytes()).hexdigest())
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    scorer.main(args)
            output.unlink()
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                scorer.main(args)
            self.assertFalse(output.exists())

    def test_cli_rejects_sources_that_change_during_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, predictions, official, output = [root / name for name in ('data.json', 'rows.jsonl', 'official.py', 'scores.json')]
            dataset.write_text(json.dumps(self.data), encoding='utf-8')
            predictions.write_text('\n'.join(json.dumps(row) for row in self.rows), encoding='utf-8')
            official.write_text('from pathlib import Path\n'
                                'def eval_question_answering(records, eval_key, metric):\n'
                                f'    Path({str(predictions)!r}).write_text("changed", encoding="utf-8")\n'
                                '    return [0.25] * len(records), None, None\n', encoding='utf-8')
            args = ['--dataset', str(dataset), '--predictions', str(predictions),
                    '--scorer', str(official), '--output', str(output)]
            with patch.object(scorer, 'DATASET_SHA256', hashlib.sha256(dataset.read_bytes()).hexdigest()), \
                    patch.object(scorer, 'SCORER_SHA256', hashlib.sha256(official.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'changed during'):
                    scorer.main(args)
            self.assertFalse(output.exists())

    def test_cli_preserves_unicode_line_characters_in_lf_and_crlf_jsonl(self):
        prediction = 'first\u2028second\u2029third\u0085fourth'
        self.rows[0].update(prediction=prediction, hypothesis=prediction)
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=repr(newline)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset, predictions, official, output = [root / name for name in
                    ('data.json', 'rows.jsonl', 'official.py', 'scores.json')]
                dataset.write_text(json.dumps(self.data), encoding='utf-8')
                original = (newline.join(json.dumps(row, ensure_ascii=False) for row in self.rows)
                            + newline).encode('utf-8')
                predictions.write_bytes(original)
                for character in ('\u2028', '\u2029', '\u0085'):
                    self.assertIn(character.encode('utf-8'), original)
                official.write_text('def eval_question_answering(records, eval_key, metric):\n'
                                    '    assert len(records) == 1540\n'
                                    '    assert eval_key == "prediction" and metric == "f1"\n'
                                    f'    assert records[0]["prediction"] == {prediction!r}\n'
                                    '    return [0.25] * len(records), None, None\n', encoding='utf-8')
                args = ['--dataset', str(dataset), '--predictions', str(predictions),
                        '--scorer', str(official), '--output', str(output)]
                with patch.object(scorer, 'DATASET_SHA256', hashlib.sha256(dataset.read_bytes()).hexdigest()), \
                        patch.object(scorer, 'SCORER_SHA256', hashlib.sha256(official.read_bytes()).hexdigest()):
                    self.assertEqual(scorer.main(args), 0)
                report = json.loads(output.read_text())
                self.assertEqual(report['qa_count'], 1540)
                self.assertEqual(report['official_f1'], 0.25)
                self.assertEqual(report['sources_sha256']['predictions'], hashlib.sha256(original).hexdigest())
                self.assertEqual(predictions.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
