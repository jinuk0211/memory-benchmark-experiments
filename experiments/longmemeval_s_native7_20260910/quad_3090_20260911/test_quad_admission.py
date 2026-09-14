"""Real local HTTP/SSE admission tests; no GPU or external services."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

import httpx
import openai

import quad_admission as admission
import quad_config as config
import quad_metered_proxy as proxy
from test_quad_queues import Fixture


class AdmissionTests(Fixture):
    def setUp(self):
        super().setUp()
        self.starts, self.requests, self.active, self.peak = [], {}, 0, 0
        self.changed = threading.Condition()
        self.release = {str(index): threading.Event() for index in range(20)}
        self.errors = set()
        self.truncated = set()
        self.trailing = {}
        fixture = self

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                body = json.dumps({'object': 'list', 'data': [{'id': config.MODEL}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                ordinal = self.headers['X-Test-Ordinal']
                with fixture.changed:
                    fixture.requests[ordinal] = (raw, dict(self.headers), self.path)
                    fixture.starts.append(ordinal)
                    fixture.active += 1
                    fixture.peak = max(fixture.peak, fixture.active)
                    fixture.changed.notify_all()
                try:
                    if not fixture.release[ordinal].wait(5):
                        raise TimeoutError('Test upstream was not released')
                    if ordinal in fixture.errors:
                        body = json.dumps({'error': {'message': 'synthetic upstream failure', 'type': 'server_error'},
                                           'usage': {'prompt_tokens': 2, 'completion_tokens': 0, 'total_tokens': 2}}).encode()
                        self.send_response(503)
                        self.send_header('Content-Type', 'application/json')
                    else:
                        chunk = {'id': 'synthetic-response', 'model': config.MODEL,
                                 'choices': [{'index': 0, 'delta': {'content': 'ok'}, 'finish_reason': 'stop'}],
                                 'usage': {'prompt_tokens': 4, 'completion_tokens': 1, 'total_tokens': 5}}
                        body = b'data: ' + json.dumps(chunk).encode() + b'\n\n'
                        if ordinal not in fixture.truncated:
                            body += b'data: [DONE]\n\n'
                        body += fixture.trailing.get(ordinal, b'')
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass  # The transport-timeout fixture intentionally closes this socket.
                finally:
                    with fixture.changed:
                        fixture.active -= 1
                        fixture.changed.notify_all()

        self.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        source = Path(__file__).parents[1] / 'fast_native2_20260911/dual_gpu_20260911/dual_metered_proxy.py'
        self.assertEqual(config.sha(source), proxy.METER_SHA)
        self.meter = config.load_module('admission_test_frozen_meter', source)
        self.routes(ids=['a']).begin('a', 'sm0', 'first', 1)

        def route(attribution, default):
            _, record = proxy.select_upstream(attribution, default)
            return 'http://127.0.0.1:' + str(self.upstream.server_port) + '/v1', record

        self.meter.select_upstream = route
        patch.dict('os.environ', {'METER_ACTIVITY_PATH': ''}).start()
        self.journal = self.directory / 'admission_usage.jsonl'
        self.server = self.meter.MeteredProxy(('127.0.0.1', 0), 'http://unused/v1', self.journal,
                                               'unassigned', 'unassigned', 2, comparison_policy=False)
        admission.install_admission(self.server, self.meter, heartbeat_seconds=0.02, queue_wait_seconds=1)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port) + '/v1'
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.addCleanup(self.cleanup_servers)

    def cleanup_servers(self):
        for event in self.release.values():
            event.set()
        self.pool.shutdown(wait=True)
        self.server.shutdown()
        self.server.server_close()
        self.upstream.shutdown()
        self.upstream.server_close()

    def headers(self, ordinal, canonical=False, lane='sm0'):
        run_id = self.deployment['protocols']['simplemem'] if canonical else 'quad_probe_' + lane + '_' + ordinal
        return {'Content-Type': 'application/json', 'X-Meter-Method': 'simplemem' if canonical else 'runtime_probe',
                'X-Meter-Run-Id': run_id, 'X-Meter-Sample-Id': 'a' if canonical else run_id,
                'X-Meter-Question-Id': 'a' if canonical else run_id,
                'X-Test-Ordinal': ordinal, 'Authorization': 'Bearer EMPTY', 'Connection': 'close'}

    def request(self, ordinal, canonical=False, timeout=2, lane='sm0'):
        payload = {'model': config.MODEL, 'stream': True, 'messages': [{'role': 'user', 'content': 'synthetic'}],
                   'temperature': 0.37, 'top_p': 0.81, 'max_tokens': 27}
        with urlopen(Request(self.url + '/chat/completions?fixture=1', json.dumps(payload).encode(),
                             self.headers(ordinal, canonical, lane)), timeout=timeout) as response:
            return response.status, response.read()

    def sdk_request(self, ordinal, timeout=0.15):
        with openai.OpenAI(api_key='EMPTY', base_url=self.url, max_retries=0,
                          timeout=httpx.Timeout(3, read=timeout)) as client:
            with client.chat.completions.create(model=config.MODEL, stream=True,
                    messages=[{'role': 'user', 'content': 'synthetic'}], temperature=0.37, top_p=0.81,
                    max_tokens=27, extra_headers=self.headers(ordinal)) as stream:
                return ''.join(item.choices[0].delta.content or '' for item in stream if item.choices)

    def wait_for(self, predicate, seconds=3):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail('Timed out waiting for local test state')

    def records(self):
        return [json.loads(line) for line in self.journal.read_text().splitlines()]

    def occupy_two(self):
        futures = []
        for ordinal in ('0', '1'):
            futures.append(self.pool.submit(self.request, ordinal))
            self.wait_for(lambda: ordinal in self.starts)
        return futures

    def test_installed_handler_get_and_post_keepalive_never_leaks_sse_headers(self):
        self.server.upstream = 'http://127.0.0.1:' + str(self.upstream.server_port) + '/v1'
        self.release['0'].set()
        with closing(HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)) as connection:
            connection.connect()
            original_socket = connection.sock
            for path, expected in [('/health', 200), ('/v1/models', 200), ('/unknown', 404)]:
                connection.request('GET', path)
                response = connection.getresponse()
                body = json.loads(response.read())
                self.assertEqual(response.status, expected)
                self.assertIn('application/json', response.headers['Content-Type'])
                if path == '/health':
                    self.assertEqual(body['status'], 'ok')
                    self.assertEqual(body['in_flight'], 0)
            headers = self.headers('0')
            headers.pop('Connection')
            payload = {'model': config.MODEL, 'stream': True,
                       'messages': [{'role': 'user', 'content': 'synthetic'}], 'max_tokens': 27}
            connection.request('POST', '/v1/chat/completions', json.dumps(payload), headers)
            response = connection.getresponse()
            body = response.read()
            self.assertIn('text/event-stream', response.headers['Content-Type'])
            self.assertIn(b'"usage":', body)
            self.assertIn(b'[DONE]', body)
            connection.request('GET', '/health')
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn('application/json', response.headers['Content-Type'])
            self.assertEqual(json.loads(response.read())['status'], 'ok')
            self.assertIs(connection.sock, original_socket)

    def test_five_fifo_capacity_two_heartbeats_and_exact_prepared_bytes(self):
        futures = self.occupy_two()
        for ordinal in ('2', '3', '4'):
            self.release[ordinal].set()
            futures.append(self.pool.submit(self.request, ordinal, True, 0.15))
            expected = int(ordinal) - 1
            self.wait_for(lambda: len(self.server.simplemem_admission['sm0'].waiting) == expected)
        # The queued clients' read timeout expires twice unless comments reach them.
        time.sleep(0.35)
        self.assertEqual(self.starts, ['0', '1'])
        self.release['0'].set()
        self.wait_for(lambda: len(self.starts) == 5)
        self.release['1'].set()
        results = [future.result(timeout=3) for future in futures]
        self.assertEqual(self.starts, ['0', '1', '2', '3', '4'])
        self.assertEqual(self.peak, 2)
        for status, body in results:
            self.assertEqual(status, 200)
            self.assertIn(b': waiting for inference slot\n\n', body)
            self.assertIn(b'data: [DONE]\n\n', body)
        bodies = [self.requests[str(index)][0] for index in range(5)]
        self.assertEqual(len(set(bodies)), 1)
        prepared = json.loads(bodies[0])
        self.assertEqual(prepared['temperature'], 0.37)
        self.assertEqual(prepared['top_p'], 0.81)
        self.assertEqual(prepared['max_tokens'], 27)
        self.assertEqual(prepared['stream_options'], {'include_usage': True})
        for ordinal, (_, headers, path) in self.requests.items():
            self.assertEqual(headers['X-Test-Ordinal'], ordinal)
            self.assertEqual(headers['Authorization'], 'Bearer EMPTY')
            self.assertFalse(any(key.lower().startswith('x-meter-') for key in headers))
            self.assertTrue(path.endswith('?fixture=1'))
        self.wait_for(lambda: len(self.records()) == 5)
        rows = self.records()
        self.assertTrue(all(row['success'] and row['http_status'] == 200 for row in rows))
        self.assertTrue(all(row['client_http_status'] == 200 and row['stream_complete'] for row in rows))
        self.assertTrue(all(row['usage']['total_tokens'] == 5 for row in rows))
        self.assertTrue(any(row['admission_wait_seconds'] >= 0.3 for row in rows))
        self.assertEqual(sorted(row['admission_sequence'] for row in rows), [1, 2, 3, 4, 5])
        self.assertTrue(all(row['admission_active_at_start'] <= 2 for row in rows))
        self.assertTrue(all(row['upstream_started_at'] <= row['upstream_completed_at'] for row in rows))
        self.assertTrue(all(row['upstream_ttfb_seconds'] >= 0 and row['upstream_seconds'] >= 0 for row in rows))

    def test_actual_openai_sdk_ignores_queue_comments_and_finishes_under_short_read_timeout(self):
        first = self.occupy_two()
        self.release['2'].set()
        queued = self.pool.submit(self.sdk_request, '2')
        self.wait_for(lambda: len(self.server.simplemem_admission['sm0'].waiting) == 1)
        time.sleep(0.35)
        self.release['0'].set()
        self.assertEqual(queued.result(timeout=3), 'ok')
        self.release['1'].set()
        for future in first:
            future.result(timeout=3)

    def test_queued_disconnect_removes_waiter_and_never_calls_upstream(self):
        first = self.occupy_two()
        connection = socket.create_connection(('127.0.0.1', self.server.server_port), timeout=1)
        payload = json.dumps({'model': config.MODEL, 'stream': True, 'messages': []}).encode()
        headers = self.headers('2')
        lines = ['POST /v1/chat/completions HTTP/1.1', 'Host: localhost', 'Content-Length: ' + str(len(payload))]
        lines.extend(key + ': ' + value for key, value in headers.items())
        connection.sendall(('\r\n'.join(lines) + '\r\n\r\n').encode() + payload)
        self.assertIn(b'200', connection.recv(4096))
        self.wait_for(lambda: len(self.server.simplemem_admission['sm0'].waiting) == 1)
        connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        self.wait_for(lambda: not self.server.simplemem_admission['sm0'].waiting)
        self.wait_for(lambda: bool(self.records()))
        row = self.records()[0]
        self.assertEqual(row['run_id'], 'quad_probe_sm0_2')
        self.assertTrue(row['client_disconnected'])
        self.assertFalse(row['success'])
        self.assertIsNone(row['http_status'])
        self.assertIsNone(row['upstream_seconds'])
        self.assertNotIn('2', self.starts)
        for ordinal in ('0', '1'):
            self.release[ordinal].set()
        for future in first:
            future.result(timeout=3)
        self.wait_for(lambda: self.server.simplemem_admission['sm0'].active == 0)

    def test_upstream_http_error_after_early200_is_sdk_error_and_failed_record(self):
        self.errors.add('0')
        self.release['0'].set()
        with self.assertRaises(openai.APIError):
            self.sdk_request('0', timeout=1)
        self.wait_for(lambda: len(self.records()) == 1)
        row = self.records()[0]
        self.assertFalse(row['success'])
        self.assertEqual(row['http_status'], 503)
        self.assertEqual(row['client_http_status'], 200)
        self.assertEqual(row['error_class'], 'UpstreamHTTPError')
        self.assertEqual(row['usage']['total_tokens'], 2)
        self.wait_for(lambda: self.server.simplemem_admission['sm0'].active == 0)

    def test_truncated_stream_raises_in_sdk_instead_of_returning_partial_answer(self):
        self.truncated.add('0')
        self.release['0'].set()
        with self.assertRaises(openai.APIError):
            self.sdk_request('0', timeout=1)
        self.wait_for(lambda: len(self.records()) == 1)
        row = self.records()[0]
        self.assertFalse(row['success'])
        self.assertFalse(row['stream_complete'])
        self.assertEqual(row['error_type'], 'incomplete_stream')
        self.assertEqual(row['usage']['total_tokens'], 5)
        self.wait_for(lambda: self.server.simplemem_admission['sm0'].active == 0)

    def test_unfinished_sse_field_or_comment_cannot_swallow_truncation_error(self):
        for ordinal, tail in [('0', b'da'), ('1', b': stalled'), ('2', b'data: {')]:
            with self.subTest(tail=tail):
                self.truncated.add(ordinal)
                self.trailing[ordinal] = tail
                self.release[ordinal].set()
                expected = json.JSONDecodeError if tail == b'data: {' else openai.APIError
                with self.assertRaises(expected):
                    self.sdk_request(ordinal, timeout=1)
        self.wait_for(lambda: len(self.records()) == 3)
        self.assertTrue(all(not row['success'] for row in self.records()))
        self.assertTrue(all(row['error_type'] == 'incomplete_stream' for row in self.records()))

    def test_expired_history_budget_never_calls_upstream_or_reports_success(self):
        with patch.object(admission, 'remaining_history_seconds', return_value=-1):
            status, body = self.request('0', canonical=True)
        self.assertEqual(status, 200)
        self.assertIn(b'"type": "admission_timeout"', body)
        self.assertNotIn(b'[DONE]', body)
        self.wait_for(lambda: len(self.records()) == 1)
        self.assertFalse(self.records()[0]['success'])
        self.assertIsNone(self.records()[0]['http_status'])
        self.assertEqual(self.starts, [])

    def test_wait_limit_removes_ticket_and_preserves_false_success(self):
        first = self.occupy_two()
        status, body = self.request('2')
        self.assertEqual(status, 200)
        self.assertIn(b'admission_timeout', body)
        self.assertNotIn('2', self.starts)
        self.wait_for(lambda: len(self.records()) == 1)
        row = self.records()[0]
        self.assertEqual(row['error_type'], 'admission_timeout')
        self.assertGreaterEqual(row['admission_wait_seconds'], 0.95)
        self.assertFalse(row['success'])
        self.assertFalse(self.server.simplemem_admission['sm0'].waiting)
        for ordinal in ('0', '1'):
            self.release[ordinal].set()
        for future in first:
            future.result(timeout=3)

    def test_upstream_transport_timeout_after_headers_is_failed_sse(self):
        # The client timeout is longer; the unchanged upstream httpx timeout wins.
        with self.assertRaises(openai.APIError):
            self.sdk_request('0', timeout=3)
        self.wait_for(lambda: len(self.records()) == 1)
        row = self.records()[0]
        self.assertEqual(row['error_type'], 'upstream_transport_error')
        self.assertEqual(row['error_class'], 'ReadTimeout')
        self.assertEqual(row['client_http_status'], 200)
        self.assertIsNone(row['http_status'])
        self.assertFalse(row['success'])
        self.assertEqual(self.server.simplemem_admission['sm0'].active, 0)
        self.release['0'].set()

    def test_lanes_have_independent_capacity(self):
        first = self.occupy_two()
        third = self.pool.submit(self.request, '2', False, 2, 'sm1')
        self.wait_for(lambda: '2' in self.starts)
        self.assertEqual(self.server.simplemem_admission['sm0'].active, 2)
        self.assertEqual(self.server.simplemem_admission['sm1'].active, 1)
        for ordinal in ('0', '1', '2'):
            self.release[ordinal].set()
        for future in first + [third]:
            future.result(timeout=3)

    def test_configuration_cannot_extend_queue_or_heartbeat_bounds(self):
        for heartbeat, wait in [(16, 3600), (15, 4801), (0, 3600), (15, 0)]:
            with self.subTest(heartbeat=heartbeat, wait=wait), self.assertRaises(ValueError):
                admission.install_admission(self.server, self.meter, heartbeat_seconds=heartbeat, queue_wait_seconds=wait)

    def test_lightmem_lane_stream_bypasses_admission_byte_path(self):
        self.release['0'].set()
        status, body = self.request('0', lane='lm0')
        self.assertEqual(status, 200)
        self.assertNotIn(b': waiting', body)
        self.wait_for(lambda: len(self.records()) == 1)
        self.assertNotIn('admission_wait_seconds', self.records()[0])
        self.assertEqual(self.server.simplemem_admission['sm0'].active, 0)


if __name__ == '__main__':
    unittest.main()