"""Offline safety checks for verified wheel transfer; no network or installers."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fetch_runtime_wheels as fetch


class Response:
    def __init__(self, status, content_range, chunks):
        self.status_code = status
        self.headers = {'Content-Range': content_range}
        self.chunks = chunks
        self.body_read = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        self.body_read = True
        yield from self.chunks


class WheelTransferTests(unittest.TestCase):
    def test_bad_status_or_range_never_reads_body_or_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            for status, header in ((200, 'bytes 0-3/4'), (206, 'bytes 1-4/4')):
                with self.subTest(status=status, header=header):
                    destination = Path(directory) / 'part'
                    response = Response(status, header, [b'abcd'])
                    with patch.object(fetch.requests, 'get', return_value=response), patch.object(fetch.time, 'sleep'):
                        with self.assertRaisesRegex(ValueError, 'Invalid range'):
                            fetch.fetch_range(('https://example.invalid/wheel', destination, 0, 3, 4))
                    self.assertFalse(response.body_read)
                    self.assertFalse(destination.exists())

    def test_short_and_oversized_bodies_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            for body, message in ((b'ab', 'Short'), (b'abcde', 'Oversized')):
                with self.subTest(body=body):
                    destination = Path(directory) / 'part'
                    response = Response(206, 'bytes 0-3/4', [body])
                    with patch.object(fetch.requests, 'get', return_value=response), patch.object(fetch.time, 'sleep'):
                        with self.assertRaisesRegex(ValueError, message):
                            fetch.fetch_range(('https://example.invalid/wheel', destination, 0, 3, 4))
                    self.assertFalse(destination.exists())

    def test_valid_range_publishes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'part'
            response = Response(206, 'bytes 0-3/4', [b'ab', b'cd'])
            with patch.object(fetch.requests, 'get', return_value=response):
                fetch.fetch_range(('https://example.invalid/wheel', destination, 0, 3, 4))
            self.assertEqual(destination.read_bytes(), b'abcd')
            self.assertFalse(destination.with_suffix('.tmp').exists())

    def fixture(self, root):
        payload = b'verified wheel payload'
        entry = {'filename': 'sample-1-py3-none-any.whl', 'size': len(payload),
                 'sha256': hashlib.sha256(payload).hexdigest(), 'url': 'https://example.invalid/wheel'}
        (root / 'cuda_wheels.json').write_text(json.dumps([entry]))
        candidate = root / 'pip-source.whl'
        candidate.write_bytes(payload)
        return payload, entry, candidate

    def test_preserve_only_rechecks_copy_and_never_deletes_source(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload, entry, candidate = self.fixture(root)
                original_copy = fetch.shutil.copyfile

                def copy(source, target):
                    if corrupt:
                        Path(target).write_bytes(b'corrupt')
                        return str(target)
                    return original_copy(source, target)

                with patch.object(fetch, '__file__', str(root / 'fetch_runtime_wheels.py')), patch('sys.argv', ['fetch', '--preserve-only']), patch.object(Path, 'glob', return_value=[candidate]), patch.object(fetch.shutil, 'copyfile', side_effect=copy), patch.object(fetch.requests, 'get') as network:
                    if corrupt:
                        with self.assertRaisesRegex(ValueError, 'Preserved wheel'):
                            fetch.main()
                    else:
                        fetch.main()
                destination = root / 'wheelhouse' / entry['filename']
                self.assertEqual(candidate.read_bytes(), payload)
                self.assertEqual(destination.exists(), not corrupt)
                if not corrupt:
                    self.assertEqual(destination.read_bytes(), payload)
                self.assertFalse((root / 'runtime_wheels_ready.json').exists())
                network.assert_not_called()

    def test_assembly_requires_hash_before_publication_and_receipt(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload, entry, _ = self.fixture(root)

                def range_stub(job):
                    _, destination, start, end, _ = job
                    destination.write_bytes((b'x' * len(payload) if corrupt else payload)[start:end + 1])

                with patch.object(fetch, '__file__', str(root / 'fetch_runtime_wheels.py')), patch('sys.argv', ['fetch']), patch.object(Path, 'glob', return_value=[]), patch.object(fetch, 'fetch_range', side_effect=range_stub):
                    if corrupt:
                        with self.assertRaisesRegex(ValueError, 'Assembled wheel'):
                            fetch.main()
                    else:
                        fetch.main()
                destination = root / 'wheelhouse' / entry['filename']
                receipt = root / 'runtime_wheels_ready.json'
                self.assertEqual(destination.exists(), not corrupt)
                self.assertEqual(receipt.exists(), not corrupt)
                if not corrupt:
                    self.assertEqual(destination.read_bytes(), payload)
                    self.assertEqual(json.loads(receipt.read_text())['verified_wheels'], [entry['filename']])


if __name__ == '__main__':
    unittest.main()
