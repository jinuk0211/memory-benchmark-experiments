"""Tests for zero-result initialization and immutable input verification."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import prepare_fresh as fresh
import verify_fresh as verifier

LOCAL = Path(__file__).resolve().parent
BASE_AUTH = json.loads((LOCAL / 'FRESH_RUN.json').read_text())


def archive(path, entries):
    with tarfile.open(path, 'w:gz') as output:
        for name, value in entries:
            info = tarfile.TarInfo(name)
            if isinstance(value, bytes):
                info.size = len(value)
                output.addfile(info, io.BytesIO(value))
            else:
                info.type = tarfile.SYMTYPE
                info.linkname = value
                output.addfile(info)


class FreshTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.root = self.directory / 'root'
        self.state = self.directory / 'state'
        self.root.mkdir()
        self.state.mkdir()
        fresh.atomic(self.state / 'FRESH_RUN.json', BASE_AUTH)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self):
        filename = 'runs/lightmem/protocol.json'
        content = b'{"frozen": true}'
        target = self.state / 'fixture.tgz'
        archive(target, [(filename, content), ('runs/lightmem/e47becba/prediction.json', b'old result'),
                         ('../escape.txt', b'must not extract')])
        entries = {filename: (len(content), hashlib.sha256(content).hexdigest())}
        return {'fixture.tgz': (fresh.sha(target), entries)}

    def test_selective_initialization_never_extracts_old_results(self):
        with patch.object(fresh, 'ARCHIVES', self.fixture()):
            result = fresh.prepare(self.root, self.state)
            self.assertEqual(result['initial_counts'], {'simplemem': 0, 'lightmem': 0})
            self.assertFalse(result['past_results_reused'])
            self.assertFalse((self.root / 'runs/lightmem/e47becba').exists())
            self.assertFalse((self.directory / 'escape.txt').exists())

    def test_existing_history_refused_without_initialization_marker(self):
        target = self.root / 'runs/simplemem_native_dialogues_v4/histories/old/attempt_0001'
        target.mkdir(parents=True)
        with patch.object(fresh, 'ARCHIVES', self.fixture()):
            with self.assertRaisesRegex(ValueError, 'history'):
                fresh.prepare(self.root, self.state)
        self.assertFalse((self.state / 'fresh_initialized.json').exists())

    def test_idempotent_call_preserves_new_results_and_marker_bytes(self):
        with patch.object(fresh, 'ARCHIVES', self.fixture()):
            fresh.prepare(self.root, self.state)
            marker = (self.state / 'fresh_initialized.json').read_bytes()
            target = self.root / 'runs/lightmem/new_question/prediction.json'
            target.parent.mkdir()
            target.write_text('new generation')
            fresh.prepare(self.root, self.state)
            self.assertEqual(target.read_text(), 'new generation')
            self.assertEqual((self.state / 'fresh_initialized.json').read_bytes(), marker)

    def test_changed_authorization_refused_after_initialization(self):
        with patch.object(fresh, 'ARCHIVES', self.fixture()):
            fresh.prepare(self.root, self.state)
            modified = dict(BASE_AUTH, authorization='Different authorization bytes')
            fresh.atomic(self.state / 'FRESH_RUN.json', modified)
            with self.assertRaisesRegex(ValueError, 'identity'):
                fresh.prepare(self.root, self.state)

    def test_hash_mismatch_does_not_overwrite_existing_input(self):
        content = b'pinned'
        source = self.state / 'one.tgz'
        archive(source, [('input.json', content)])
        target = self.root / 'input.json'
        target.write_bytes(b'existing')
        with self.assertRaisesRegex(ValueError, 'Existing input differs'):
            fresh.extract_selected(source, fresh.sha(source),
                                   {'input.json': (len(content), hashlib.sha256(content).hexdigest())}, self.root)
        self.assertEqual(target.read_bytes(), b'existing')

    def test_symlink_selected_member_refused(self):
        source = self.state / 'link.tgz'
        archive(source, [('input.json', '/outside')])
        with self.assertRaisesRegex(ValueError, 'regular'):
            fresh.extract_selected(source, fresh.sha(source), {'input.json': (0, 'x')}, self.root)

    def test_duplicate_selected_member_refused(self):
        source = self.state / 'duplicate.tgz'
        archive(source, [('input.json', b'x'), ('input.json', b'x')])
        with self.assertRaisesRegex(ValueError, 'one regular'):
            fresh.extract_selected(source, fresh.sha(source), {'input.json': (1, 'x')}, self.root)

    def test_all_three_frozen_requirement_lists_parse_exactly(self):
        expected_counts = {'inference': 190, 'lightmem': 73, 'simplemem': 57}
        for label, _, digest in verifier.ENVIRONMENTS.values():
            path = LOCAL / ('bootstrap_' + label + '_requirements.txt')
            self.assertEqual(fresh.sha(path), digest)
            packages = verifier.expected_packages(path.read_text())
            self.assertEqual(len(packages), expected_counts[label])
        self.assertEqual(verifier.expected_packages(
            'torch @ https://download.pytorch.org/whl/torch-2.8.0%2Bcpu-cp311-cp311-linux_x86_64.whl')['torch'], '2.8.0+cpu')


if __name__ == '__main__':
    unittest.main()
