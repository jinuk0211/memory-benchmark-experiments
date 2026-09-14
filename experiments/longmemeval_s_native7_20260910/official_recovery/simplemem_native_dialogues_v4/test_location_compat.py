import json
import unittest
from test_runner import NativeTests
import runner
import location_compat

class CompatibilityTests(unittest.TestCase):
    setUp = NativeTests.setUp

    def install(self):
        self.audit = self.root / 'audit.jsonl'
        location_compat.install(self.builder, runner.HERE / 'upstream/simplemem/core/memory_builder.py', self.audit)

    def parse(self, location):
        return self.builder._parse_llm_response(json.dumps([{'lossless_restatement':'test fact','location':location}]), [1])[0]

    def test_string_and_null_remain_unchanged(self):
        self.install()
        self.assertEqual(self.parse('room').location, 'room')
        self.assertIsNone(self.parse(None).location)
        self.assertEqual(self.audit.read_text(), '')

    def test_list_keeps_order_and_duplicates_and_raw_audit(self):
        self.install()
        self.assertEqual(self.parse(['room','park','room']).location, 'room, park, room')
        audit = json.loads(self.audit.read_text())
        self.assertEqual(audit['original_location'], ['room','park','room'])
        self.assertEqual(json.loads(audit['raw_response'])[0]['location'], ['room','park','room'])

    def test_invalid_list_remains_failure(self):
        self.install()
        with self.assertRaises(ValueError): self.parse(['room', 5])

    def test_empty_list_does_not_invent_location(self):
        self.install()
        self.assertEqual(self.parse([]).location, '')

    def test_original_fields_still_validated(self):
        self.install()
        with self.assertRaises(Exception):
            self.builder._parse_llm_response('[{"location":["room"]}]', [1])

    def test_pinned_upstream_and_overlay_are_bound(self):
        self.assertIn('location_compat.py', runner.source_hashes())
        self.assertIn('Compatibility v4', runner.POLICY['normalization'])
