"""Attribution and pass-through checks for fresh four-GPU metering."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
import unittest

import quad_config as config
import quad_metered_proxy as proxy
from test_quad_queues import Fixture


class RoutingChecks(Fixture):
    def attribution(self, method='simplemem', sample='a'):
        return {'method': method, 'sample_id': sample, 'question_id': sample,
                'run_id': self.deployment['protocols'][method] if method == 'simplemem' else 'native7_20260910'}

    def test_routes_each_method_to_its_two_exact_gpus(self):
        for method, names in [('simplemem', ['sm0', 'sm1']), ('lightmem', ['lm0', 'lm1'])]:
            routes = self.routes(method=method)
            for qid, lane in zip(('a', 'b'), names):
                routes.begin(qid, lane, qid, 1)
                endpoint, record = proxy.select_upstream(self.attribution(method, qid))
                self.assertEqual(endpoint, self.deployment['lanes'][lane]['api_base'])
                self.assertEqual(record['inference_gpu_uuid'], self.deployment['lanes'][lane]['gpu_uuid'])
                self.assertTrue(record['canonical_benchmark'])
                self.assertEqual(record['execution_run_id'], 'fresh_test')

    def test_missing_assignment_or_dispatch_never_falls_back(self):
        self.routes().assign('a', 'sm0')
        for qid in ('a', 'b'):
            with self.subTest(qid=qid), self.assertRaises(ValueError):
                proxy.select_upstream(self.attribution(sample=qid), 'http://fallback')

    def test_conflicting_question_protocol_and_route_identity_rejected(self):
        routes = self.routes()
        routes.begin('a', 'sm0', 'first', 1)
        attribution = self.attribution()
        for key, value in [('question_id', 'b'), ('run_id', 'different_protocol'), ('method', 'unknown')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                proxy.select_upstream({**attribution, key: value})
        manifest = config.read(routes.path)
        manifest['histories']['a']['gpu_uuid'] = 'wrong'
        config.atomic(routes.path, manifest)
        with self.assertRaises(ValueError):
            proxy.select_upstream(attribution)

    def test_probe_all_four_lanes_before_ready_is_excluded_from_benchmark(self):
        (self.directory / 'READY.json').unlink()
        for lane in self.deployment['lanes']:
            probe_id = 'quad_probe_' + lane + '_unique123'
            endpoint, record = proxy.select_upstream({'method': 'runtime_probe', 'run_id': probe_id,
                                                       'sample_id': probe_id, 'question_id': probe_id})
            self.assertEqual(endpoint, self.deployment['lanes'][lane]['api_base'])
            self.assertFalse(record['canonical_benchmark'])
        with self.assertRaises(ValueError):
            proxy.select_upstream(self.attribution())

    def test_probe_must_have_exact_safe_lane_and_matching_id(self):
        for run_id, sample in [('quad_probe_sm2_unknown', 'quad_probe_sm2_unknown'),
                               ('quad_probe_sm0_good', 'other'), ('anything', 'anything')]:
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                proxy.select_upstream({'method': 'runtime_probe', 'run_id': run_id, 'sample_id': sample})


class MeterIntegration(Fixture):
    def test_frozen_meter_preserves_payload_and_routes_and_journals(self):
        self.check_meter(False)

    def test_central_simplemem_sse_preserves_sampling_and_records_usage(self):
        self.check_meter(True)

    def check_meter(self, streaming):
        source = __import__('pathlib').Path(__file__).parents[1] / 'fast_native2_20260911/dual_gpu_20260911/dual_metered_proxy.py'
        self.assertEqual(config.sha(source), proxy.METER_SHA)
        meter = config.load_module('test_quad_frozen_meter', source)
        seen = []

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body = json.dumps({'id': 'synthetic-response', 'model': config.MODEL,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 4, 'completion_tokens': 1, 'total_tokens': 5}}).encode()
                if seen[-1].get('stream'):
                    body = b'data: ' + body + b'\n\ndata: [DONE]\n\n'
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream' if seen[-1].get('stream') else 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        self.routes().begin('a', 'sm1', 'first', 1)
        original = proxy.select_upstream

        def test_route(attribution, default):
            endpoint, record = original(attribution, default)
            self.assertEqual(endpoint, self.deployment['lanes']['sm1']['api_base'])
            return 'http://127.0.0.1:' + str(upstream.server_port) + '/v1', record

        meter.select_upstream = test_route
        journal = self.directory / 'usage.jsonl'
        with patch.dict('os.environ', {'METER_ACTIVITY_PATH': ''}):
            server = meter.MeteredProxy(('127.0.0.1', 0), 'http://unused/v1', journal, 'unassigned', 'unassigned', 10,
                                       comparison_policy=False)
            proxy.install_admission(server, meter)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            try:
                payload = {'model': config.MODEL, 'messages': [{'role': 'user', 'content': 'synthetic fixture'}],
                           'temperature': 0.37, 'top_p': 0.8, 'max_tokens': 2000, 'stream': streaming}
                headers = {'Content-Type': 'application/json', 'X-Meter-Method': 'simplemem',
                           'X-Meter-Run-Id': self.deployment['protocols']['simplemem'],
                           'X-Meter-Sample-Id': 'a', 'X-Meter-Question-Id': 'a'}
                url = 'http://127.0.0.1:' + str(server.server_port) + '/v1/chat/completions'
                with urlopen(Request(url, json.dumps(payload).encode(), headers), timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    response.read()
                headers['X-Meter-Question-Id'] = 'conflicting'
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url, json.dumps(payload).encode(), headers), timeout=10)
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]['messages'], payload['messages'])
        self.assertEqual(seen[0]['temperature'], 0.37)
        self.assertEqual(seen[0]['stream'], streaming)
        if streaming:
            self.assertTrue(seen[0]['stream_options']['include_usage'])
        self.assertEqual(seen[0]['top_p'], 0.8)
        self.assertEqual(seen[0]['max_tokens'], 2000)
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        successful = [row for row in rows if row['success']]
        self.assertEqual(len(successful), 1)
        self.assertEqual(successful[0]['inference_lane'], 'sm1')
        self.assertEqual(successful[0]['usage']['total_tokens'], 5)
        if streaming:
            self.assertTrue(successful[0]['stream_complete'])
        self.assertNotIn('synthetic fixture', journal.read_text())


if __name__ == '__main__':
    unittest.main()