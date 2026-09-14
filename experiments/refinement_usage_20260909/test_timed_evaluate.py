"""CPU-only launcher checks using a fake experiment project."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import timed_evaluate

ROOT = Path(__file__).resolve().parent
FAKE_API = """
from pathlib import Path
import json
import sys
from types import SimpleNamespace

CALLS = []
ROW = {'question_id': 'conv-1:0', 'prediction': 'answer', 'generation_batch_seconds': 123.0}

def evaluate_sample(rt, units, sample, method, dataset):
    CALLS.append((rt, units, sample, method, dataset))
    if sample.get('raise_error'):
        raise LookupError('original evaluator error')
    return [ROW]

def main():
    rt = SimpleNamespace(args=SimpleNamespace(out=Path('relative_run'), seed=7, model='fake', embed_model='fake'), model_meta={'name': 'fake'}, embed_meta={'name': 'fake'})
    result = evaluate_sample(rt, ['unit'], {'sample_id': 'conv-1'}, 'method', 'locomo')
    assert result[0] is ROW and str(rt.args.out) == 'relative_run'
    Path('observed.json').write_text(json.dumps({'row': result[0], 'argv': sys.argv, 'cwd': str(Path.cwd()), 'calls': len(CALLS)}))
"""
FAKE_RECURSIVE = """
import run_transfer as baseline_api

def main():
    baseline_api.main()
"""


class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        (self.project/'run_transfer.py').write_text(FAKE_API, encoding='utf-8')
        (self.project/'run_recursive_v2.py').write_text(FAKE_RECURSIVE, encoding='utf-8')

    def invoke(self, runner: str, stage: str = 'evaluate') -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(ROOT/'timed_evaluate.py'), '--runner',
                               str(self.project/runner), '--', stage], cwd=self.project,
                              capture_output=True, text=True, timeout=15, check=False)

    def test_recursive_hook_preserves_rows_relative_out_and_cli(self) -> None:
        result = self.invoke('run_recursive_v2.py')
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads((self.project/'observed.json').read_text())
        self.assertEqual(observed['calls'], 1)
        self.assertEqual(observed['row']['generation_batch_seconds'], 123.0)
        self.assertEqual(observed['argv'], [str(self.project/'run_recursive_v2.py'), 'evaluate'])
        self.assertEqual(Path(observed['cwd']), self.project)
        self.assertEqual(len(list((self.project/'relative_run/evaluation_timing').rglob('completed.json'))), 1)

    def test_baseline_module_global_is_wrapped(self) -> None:
        result = self.invoke('run_transfer.py')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((self.project/'relative_run/evaluation_timing').rglob('completed.json'))), 1)

    def test_baseline_failure_keeps_existing_diagnostic_marker(self) -> None:
        api = self.project/'run_transfer.py'
        api.write_text(FAKE_API + '\ndef main():\n    raise ValueError("baseline failure")\n', encoding='utf-8')
        result = self.invoke('run_transfer.py')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TRANSFER_FAILED ValueError('baseline failure')", result.stdout)

    def test_spawn_child_does_not_reenter_launcher_or_duplicate_evaluation(self) -> None:
        api = self.project/'run_transfer.py'
        api.write_text(FAKE_API + '\ndef child_probe():\n    Path("child_seen.txt").write_text("ok")\n', encoding='utf-8')
        (self.project/'run_recursive_v2.py').write_text(
            'import multiprocessing\nimport run_transfer as baseline_api\n'
            'def main():\n'
            '    child = multiprocessing.get_context("spawn").Process(target=baseline_api.child_probe)\n'
            '    child.start()\n'
            '    try:\n'
            '        child.join(8)\n'
            '        assert child.exitcode == 0\n'
            '    finally:\n'
            '        if child.is_alive():\n'
            '            child.terminate()\n'
            '            child.join(3)\n'
            '    baseline_api.main()\n', encoding='utf-8')
        result = self.invoke('run_recursive_v2.py')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.project/'child_seen.txt').read_text(), 'ok')
        self.assertEqual(json.loads((self.project/'observed.json').read_text())['calls'], 1)
        self.assertEqual(len(list((self.project/'relative_run/evaluation_timing').rglob('completed.json'))), 1)

    def test_construction_is_rejected_before_runner_import(self) -> None:
        (self.project/'run_recursive_v2.py').write_text("raise RuntimeError('must not import')", encoding='utf-8')
        result = self.invoke('run_recursive_v2.py', 'construct')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('only the evaluate stage', result.stderr)
        self.assertNotIn('must not import', result.stderr)

    def test_different_project_module_is_rejected(self) -> None:
        loaded = ModuleType('run_transfer')
        loaded.__file__ = str(self.project/'wrong_project/run_transfer.py')
        with patch.dict(sys.modules, {'run_transfer': loaded}):
            with self.assertRaisesRegex(ValueError, 'fresh process'):
                timed_evaluate.run_timed(self.project/'run_recursive_v2.py', ['evaluate'])

    def test_function_argv_path_restored_when_main_fails(self) -> None:
        api_path = self.project/'run_transfer.py'
        spec = importlib.util.spec_from_file_location('run_transfer', api_path)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        original, argv, path = api.evaluate_sample, sys.argv, sys.path[:]
        (self.project/'run_recursive_v2.py').write_text(
            'import run_transfer as baseline_api\ndef main():\n    raise RuntimeError("main failure")\n', encoding='utf-8')
        with patch.dict(sys.modules, {'run_transfer': api}):
            with self.assertRaisesRegex(RuntimeError, 'main failure'):
                timed_evaluate.run_timed(self.project/'run_recursive_v2.py', ['evaluate'])
            self.assertIs(api.evaluate_sample, original)
            self.assertIs(sys.argv, argv)
            self.assertEqual(sys.path, path)


if __name__ == '__main__':
    unittest.main()
