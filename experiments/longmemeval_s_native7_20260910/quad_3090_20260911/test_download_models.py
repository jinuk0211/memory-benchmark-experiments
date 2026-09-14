"""Behavioral tests for resume, data integrity and credential boundaries."""

import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import download_models as downloader

PAYLOAD = b'actual pinned model bytes\x00\xff' * 200


class Server:
    def __init__(self, behavior):
        self.behavior = behavior
        self.ranges = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.ranges.append(self.headers.get('Range'))
                owner.behavior(self, len(owner.ranges))

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def open(self, url, token, offset, timeout):
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        return urllib.request.urlopen(urllib.request.Request(
            f'http://127.0.0.1:{self.server.server_port}/file', headers=headers), timeout=timeout)


def respond(handler, payload, offset=0, status=None, range_start=None):
    handler.send_response(status or (206 if offset else 200))
    if offset or status == 206:
        start = offset if range_start is None else range_start
        handler.send_header('Content-Range', f'bytes {start}-{len(payload)-1}/{len(payload)}')
    handler.send_header('Content-Length', str(len(payload) - offset))
    handler.end_headers()
    handler.wfile.write(payload[offset:])


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.item = {'target': self.root / 'model.bin', 'bytes': len(PAYLOAD),
                     'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'model': 'test',
                     'name': 'model.bin', 'url': 'https://huggingface.co/test/model'}
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.temporary.cleanup()

    def download(self, server, attempts=2):
        return downloader.download_file(self.item, 'hf_test', attempts=attempts,
                                        opener=server.open, retry_delay=0, reserve_bytes=0)

    def test_dropped_connection_resumes_exact_retained_bytes(self):
        def serve(handler, number):
            if number == 1:
                handler.send_response(200)
                handler.send_header('Content-Length', str(len(PAYLOAD)))
                handler.end_headers()
                handler.wfile.write(PAYLOAD[:700])
                handler.wfile.flush()
                handler.close_connection = True
            else:
                respond(handler, PAYLOAD, offset=700)

        with Server(serve) as server:
            self.download(server)
        self.assertEqual(server.ranges, [None, 'bytes=700-'])
        self.assertEqual(self.item['target'].read_bytes(), PAYLOAD)

    def test_ignored_range_replaces_partial_instead_of_appending(self):
        (self.root / 'model.bin.partial').write_bytes(PAYLOAD[:900])
        with Server(lambda handler, number: respond(handler, PAYLOAD)) as server:
            self.download(server)
        self.assertEqual(server.ranges, ['bytes=900-'])
        self.assertEqual(self.item['target'].read_bytes(), PAYLOAD)

    def test_checksum_mismatch_never_promotes_and_can_retry_cleanly(self):
        bad = b'!' * len(PAYLOAD)
        with Server(lambda handler, number: respond(handler, bad if number == 1 else PAYLOAD)) as server:
            self.download(server)
        self.assertEqual(server.ranges, [None, None])
        self.assertEqual(self.item['target'].read_bytes(), PAYLOAD)

    def test_terminal_hash_mismatch_does_not_replace_existing_file(self):
        self.item['target'].write_bytes(b'old artifact')
        with Server(lambda handler, number: respond(handler, b'!' * len(PAYLOAD))) as server:
            with self.assertRaises(RuntimeError):
                self.download(server, attempts=1)
        self.assertEqual(self.item['target'].read_bytes(), b'old artifact')
        self.assertFalse((self.root / 'model.bin.partial').exists())

    def test_bad_content_range_preserves_partial_without_appending(self):
        partial = self.root / 'model.bin.partial'
        partial.write_bytes(PAYLOAD[:700])
        with Server(lambda handler, number: respond(handler, PAYLOAD, 700, range_start=600)) as server:
            with self.assertRaises(RuntimeError):
                self.download(server, attempts=1)
        self.assertEqual(partial.read_bytes(), PAYLOAD[:700])
        self.assertFalse(self.item['target'].exists())

    def test_verified_file_is_reused_without_network(self):
        self.item['target'].write_bytes(PAYLOAD)
        def fail(*args):
            raise AssertionError('Unexpected network call')
        result = downloader.download_file(self.item, 'hf_test', opener=fail)
        self.assertTrue(result['reused'])

    def test_complete_partial_is_verified_and_promoted_without_network(self):
        (self.root / 'model.bin.partial').write_bytes(PAYLOAD)
        def fail(*args):
            raise AssertionError('Unexpected network call')
        result = downloader.download_file(self.item, 'hf_test', opener=fail)
        self.assertTrue(result['reused'])
        self.assertEqual(self.item['target'].read_bytes(), PAYLOAD)


class AuthenticationTests(unittest.TestCase):
    def test_bearer_is_not_forwarded_to_cdn(self):
        requests = []
        sentinel = object()
        class Opener:
            def open(self, request, timeout):
                requests.append(request)
                if len(requests) == 1:
                    raise urllib.error.HTTPError(request.full_url, 302, 'redirect',
                                                 {'Location': 'https://cas-bridge.xethub.hf.co/file?signature=secret'}, None)
                return sentinel
        with patch.object(downloader.urllib.request, 'build_opener', return_value=Opener()):
            result = downloader.open_download('https://huggingface.co/org/model/resolve/commit/file', 'hf_test', 100)
        self.assertIs(result, sentinel)
        self.assertEqual(requests[0].get_header('Authorization'), 'Bearer hf_test')
        self.assertIsNone(requests[1].get_header('Authorization'))
        self.assertEqual(requests[1].get_header('Range'), 'bytes=100-')

    def test_unapproved_redirect_is_rejected_before_request(self):
        class Opener:
            calls = 0
            def open(self, request, timeout):
                self.calls += 1
                raise urllib.error.HTTPError(request.full_url, 302, 'redirect',
                                             {'Location': 'https://untrusted.example/file'}, None)
        opener = Opener()
        with patch.object(downloader.urllib.request, 'build_opener', return_value=opener):
            with self.assertRaises(ValueError):
                downloader.open_download('https://huggingface.co/org/model/resolve/commit/file', 'hf_test', 0)
        self.assertEqual(opener.calls, 1)

    def test_actual_manifest_has_all_thirty_pinned_files(self):
        manifest = json.loads((Path(__file__).resolve().parent.parent / 'model_integrity.json').read_text())
        files = downloader.inventory(manifest)
        self.assertEqual(len(files), 30)
        self.assertEqual({item['model'] for item in files}, {'qwen', 'minilm', 'compressor'})
        self.assertGreater(sum(item['bytes'] for item in files), 20_000_000_000)


if __name__ == '__main__':
    unittest.main()
