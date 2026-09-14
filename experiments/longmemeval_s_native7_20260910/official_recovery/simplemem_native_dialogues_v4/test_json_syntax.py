import json
import unittest
from test_location_compat import CompatibilityTests
from json_syntax_compat import repair_locations

class SyntaxTests(unittest.TestCase):
    setUp=CompatibilityTests.setUp
    install=CompatibilityTests.install

    def test_multiple_location_values_become_lossless_array(self):
        original='[{"lossless_restatement":"fact","location":"Austin", "Dallas", "Austin","topic":"places"}]'
        repaired,changes=repair_locations(original)
        self.assertEqual(json.loads(repaired)[0]['location'],['Austin','Dallas','Austin'])
        self.assertEqual(len(changes),1)
        self.install()
        entries=self.builder._parse_llm_response(original,[1,2])
        self.assertEqual(entries[0].location,'Austin, Dallas, Austin')
        audit=json.loads((self.root/'json_syntax_compat_audit.jsonl').read_text())
        self.assertEqual(audit['raw_response'],original)

    def test_next_key_is_not_swallowed_and_valid_arrays_unchanged(self):
        text='[{"lossless_restatement":"fact","location":"Austin","topic":"unchanged"}]'
        self.assertEqual(repair_locations(text),(text,[]))
        self.install()
        self.assertEqual(self.builder._parse_llm_response(text,[1])[0].topic,'unchanged')

    def test_quoted_schema_words_and_escaped_quotes_are_preserved(self):
        value='quoted "location": "a", "b" and https://example.com'
        original=json.dumps([{'lossless_restatement':value,'location':['A','B']}])
        self.assertEqual(repair_locations(original),(original,[]))

    def test_full_array_with_trailing_fence_does_not_drop_entries(self):
        self.install()
        raw='[{"lossless_restatement":"one"},{"lossless_restatement":"two"}]\n```'
        self.assertEqual(len(self.builder._parse_llm_response(raw,[1])),2)

    def test_unrelated_invalid_json_or_numeric_location_is_not_guessed(self):
        self.install()
        for raw in ('[{"lossless_restatement":"fact","location":"a",5}]',
                    '[{"lossless_restatement":"fact", "topic" "missing colon"}]'):
            with self.assertRaises(ValueError): self.builder._parse_llm_response(raw,[1])

    def test_second_array_is_not_silently_discarded(self):
        self.install()
        with self.assertRaises(ValueError):
            self.builder._parse_llm_response('[{"lossless_restatement":"one"}] [{"lossless_restatement":"two"}]',[1])
