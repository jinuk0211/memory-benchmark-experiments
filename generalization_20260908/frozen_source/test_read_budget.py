import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import read_budget
import refine as core


class ReaderTokenTests(unittest.TestCase):
    def test_served_prompt_and_output_tokens_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            rt=SimpleNamespace(model_meta={'model':'toy'},args=SimpleNamespace(seed=1),cache=Path(folder))
            row={'id':'toy','context':'source body','question':'toy question','prediction':'answer','read_tokens':2}
            user='Conversation memory:\nsource body\n\nQuestion: toy question\nAnswer:'
            key=core.digest([rt.model_meta,1,core.READER,user,96,False])
            core.save(rt.cache/'generations'/f'{key}.json',{'text':'answer','input_tokens':31,'output_tokens':3})
            with patch.object(read_budget,'indexed_evaluate',return_value=[row]):
                result=read_budget.evaluate(rt,{},[{'split':'dev'}],{},5,'trial',Path(folder))
            self.assertEqual(result[0]['reader_total_tokens'],34)
            self.assertEqual(result[0]['reader_input_tokens'],31)
            self.assertEqual(result[0]['read_budget'],5)
            self.assertEqual(json.loads((Path(folder)/'trial_dev_items.json').read_text())[0],result[0])

    def test_audit_cannot_select_a_budget(self):
        with self.assertRaises(ValueError):read_budget.assess(None,{}, {},[{'split':'probe_audit'}],'x',None,1024,'q0')


if __name__=='__main__':unittest.main()
