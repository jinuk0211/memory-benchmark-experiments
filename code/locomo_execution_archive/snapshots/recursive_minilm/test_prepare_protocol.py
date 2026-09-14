import unittest
from prepare_protocol import select_ids, TYPES


class TransferSelectionTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{'question_id': f'{kind}_{i}', 'question_type': kind}
                     for kind in TYPES for i in range(8)]
        self.rows += [{'question_id': f'{kind}_{i}_abs', 'question_type': kind}
                      for kind in TYPES for i in range(2)]

    def test_order_invariant_exact_types_and_disjoint_clusters(self):
        selected = select_ids(self.rows, [])
        self.assertEqual(selected, select_ids(list(reversed(self.rows)), []))
        self.assertEqual(len(selected), 18)
        self.assertEqual(len({qid.removesuffix('_abs') for qid in selected}), 18)
        kinds = {row['question_id']: row['question_type'] for row in self.rows}
        for kind in TYPES:
            self.assertEqual(sum(kinds[qid] == kind for qid in selected), 3)
        self.assertEqual(sum(qid.endswith('_abs') for qid in selected), 6)

    def test_previous_cluster_excluded_including_alternate_form(self):
        excluded = [TYPES[0] + '_0', TYPES[1] + '_1_abs']
        selected = select_ids(self.rows, excluded)
        self.assertFalse({qid.removesuffix('_abs') for qid in selected} &
                         {qid.removesuffix('_abs') for qid in excluded})

    def test_no_abstention_type_uses_three_regular_questions(self):
        rows = [row for row in self.rows if not row['question_id'].endswith('_abs')]
        self.assertEqual(len(select_ids(rows, [])), 18)

    def test_duplicate_and_insufficient_clusters_fail(self):
        with self.assertRaises(ValueError):
            select_ids(self.rows + [self.rows[0]], [])
        with self.assertRaises(ValueError):
            select_ids([row for row in self.rows if row['question_type'] != TYPES[0]], [])


if __name__ == '__main__':
    unittest.main()
