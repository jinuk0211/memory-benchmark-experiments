"""Execute the sequence-selection shell block and validate tuner integration."""
import configparser
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which('bash')


class SupervisorConfigTests(unittest.TestCase):
    def test_tuner_stays_manual_and_preserves_native_worker_process_groups(self):
        config = configparser.ConfigParser(interpolation=None)
        config.read(HERE / 'quad-tune-lightmem.conf', encoding='utf-8')
        self.assertEqual(config.sections(), ['program:quad-tune-lightmem'])
        program = config['program:quad-tune-lightmem']
        self.assertFalse(program.getboolean('autostart'))
        self.assertEqual(program['autorestart'], 'unexpected')
        self.assertEqual(program.getint('startretries'), 1)
        self.assertEqual(program.getint('startsecs'), 2)
        self.assertFalse(program.getboolean('stopasgroup'))
        self.assertFalse(program.getboolean('killasgroup'))
        arguments = shlex.split(program['command'])
        self.assertEqual(arguments[-2:], [
            '/usr/bin/python3', '/workspace/quad_3090_20260911/tune_lightmem.py'])
        self.assertEqual(arguments[:-2], [
            '/usr/bin/env', '-u', 'CONTAINER_API_KEY', '-u', 'OPENROUTER_API_KEY',
            '-u', 'HF_TOKEN', '-u', 'HUGGING_FACE_HUB_TOKEN'])
        self.assertEqual(program['directory'], '/workspace/quad_3090_20260911')
        self.assertEqual(program['stdout_logfile'],
                         '/workspace/quad_3090_20260911/quad-tune-lightmem.log')


@unittest.skipUnless(BASH, 'Bash is required for launcher validation')
class LauncherSequenceTests(unittest.TestCase):
    def selection(self, directory, port):
        source = (HERE / 'serve_qwen.sh').read_text(encoding='utf-8')
        block = source[source.index('PORT="${2:?Port required}"'):source.index('TASK_ROOT=')]
        block = block.replace('/workspace/quad_3090_20260911/seqs_${PORT}.txt',
                              '$TUNE_TEST_SEQS/seqs_${PORT}.txt')
        script = 'set -euo pipefail\n' + block + '\nprintf \'%s\\n\' "$MAX_SEQS"\n'
        environment = {**os.environ, 'TUNE_TEST_SEQS': directory.as_posix()}
        return subprocess.run([BASH, '-c', script, 'launcher-selection-fixture', 'GPU-fixture', str(port)],
                              text=True, capture_output=True, timeout=10, env=environment)

    def test_full_launcher_has_valid_bash_syntax(self):
        subprocess.run([BASH, '-n', str(HERE / 'serve_qwen.sh')], check=True, timeout=10)

    def test_only_four_named_ports_read_their_sequence_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for port in (18081, 18091, 18101, 18111, 18112):
                (directory / f'seqs_{port}.txt').write_text('2\n')
            for port in (18081, 18091, 18101, 18111):
                with self.subTest(port=port):
                    result = self.selection(directory, port)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), '2')
            result = self.selection(directory, 18112)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), '1')

    def test_missing_file_defaults_to_one_and_invalid_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = self.selection(directory, 18101)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), '1')
            for value in ('0\n', '3\n', '2\nextra\n', ''):
                with self.subTest(value=value):
                    (directory / 'seqs_18101.txt').write_text(value)
                    result = self.selection(directory, 18101)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
