"""Isolated loopback smoke; synthetic upstream, temporary journal, no production ports."""
from contextlib import closing
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import quad_admission as admission
import quad_config as cfg

PENDING = cfg.STATE / 'quad_metered_proxy.admission_pending.py'
PINS = {'quad_admission.py': '1a785ac42c264b75acdf6b8c996b5e6df4c45e7710fd3ad28d7e12437a723423',
        'quad_metered_proxy.py': '700b043d38268e64a9114c3d6bf1e7b82d88b5510ad24a1b5e4cf1eaa8b0ac90',
        'canary_admission.py': '5e98177c743dded53920f7acf2044c6c4b193ee34177395c3d67f9c244b1dd6e'}


def identity():
    if os.environ.get('CONTAINER_ID') != cfg.INSTANCE:
        raise ValueError('Wrong instance')
    for name, expected in PINS.items():
        path = PENDING if name == 'quad_metered_proxy.py' else cfg.STATE / name
        if cfg.sha(path) != expected:
            raise ValueError('Staged smoke code differs')
    frozen = cfg.ROOT / 'fast_native2_20260911/dual_gpu_20260911/dual_metered_proxy.py'
    frozen_sha = '8ef745e6df482e02df494422dcbf96eaf56cb67d278656d4f14699b1e429cb93'
    if cfg.sha(frozen) != frozen_sha:
        raise ValueError('Frozen meter changed')
    return {**cfg.fresh_identity(), 'deployment_sha256': cfg.DEPLOYMENT_SHA,
            'smoke_code_sha256': dict(PINS), 'frozen_meter_sha256': frozen_sha,
            'candidate_wrapper_path': str(PENDING), 'smoke_script_sha256': cfg.sha(Path(__file__))}


def run_smoke():
    if cfg.sha(PENDING) != PINS['quad_metered_proxy.py']:
        raise ValueError('Pending candidate wrapper SHA differs')
    frozen = cfg.ROOT / 'fast_native2_20260911/dual_gpu_20260911/dual_metered_proxy.py'
    if cfg.sha(frozen) != '8ef745e6df482e02df494422dcbf96eaf56cb67d278656d4f14699b1e429cb93':
        raise ValueError('Frozen meter changed')
    proxy = cfg.load_module('isolated_pending_wrapper', PENDING)
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            self.reply({'object': 'list', 'data': [{'id': cfg.MODEL}]})
        def reply(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if payload.get('stream') is not True:
                raise ValueError('Smoke payload is not streaming')
            chunk = {'id': 'isolated-smoke', 'model': cfg.MODEL,
                     'choices': [{'index': 0, 'delta': {'content': 'synthetic'}, 'finish_reason': 'length'}],
                     'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
            body = b'data: ' + json.dumps(chunk).encode() + b'\n\ndata: [DONE]\n\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with tempfile.TemporaryDirectory(prefix='admission-smoke-') as temporary:
        os.environ['METER_ACTIVITY_PATH'] = str(Path(temporary) / 'activity.json')
        meter = cfg.load_module('isolated_smoke_meter', frozen)
        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        base = 'http://127.0.0.1:' + str(upstream.server_port) + '/v1'
        meter.select_upstream = lambda attribution, default: (base, proxy.select_upstream(attribution, default)[1])
        server = meter.MeteredProxy(('127.0.0.1', 0), base, Path(temporary) / 'usage.jsonl',
                                   'unassigned', 'unassigned', 3, comparison_policy=False)
        admission.install_admission(server, meter)
        for service in (upstream, server):
            threading.Thread(target=service.serve_forever, daemon=True).start()
        try:
            checks = []
            with closing(HTTPConnection('127.0.0.1', server.server_port, timeout=5)) as client:
                client.connect()
                connection = client.sock
                for path, status in [('/health', 200), ('/v1/models', 200), ('/unknown', 404), ('stream', 200), ('/health', 200)]:
                    if path == 'stream':
                        sample = 'quad_probe_sm0_smoke_' + uuid.uuid4().hex
                        headers = {'Content-Type': 'application/json', 'X-Meter-Method': 'runtime_probe',
                                   'X-Meter-Run-Id': sample, 'X-Meter-Sample-Id': sample,
                                   'X-Meter-Question-Id': sample, 'X-Meter-Phase': 'isolated_smoke'}
                        client.request('POST', '/v1/chat/completions', json.dumps({'model': cfg.MODEL,
                            'stream': True, 'messages': [{'role': 'user', 'content': 'synthetic smoke'}],
                            'max_tokens': 1}), headers)
                    else:
                        client.request('GET', path)
                    response = client.getresponse()
                    raw = response.read()
                    if response.status != status or client.sock is not connection:
                        raise ValueError('Smoke status or keep-alive connection differs')
                    if path == 'stream':
                        if 'text/event-stream' not in response.headers.get('Content-Type', '') or b'[DONE]' not in raw or b'"usage"' not in raw:
                            raise ValueError('Smoke SSE usage/DONE is missing')
                    else:
                        body = json.loads(raw)
                        if 'application/json' not in response.headers.get('Content-Type', ''):
                            raise ValueError('SSE state leaked into GET')
                        if path == '/health' and (body.get('status') != 'ok' or body.get('in_flight') != 0):
                            raise ValueError('Smoke meter health is not idle/healthy')
                        if path == '/v1/models' and body.get('data', [{}])[0].get('id') != cfg.MODEL:
                            raise ValueError('Smoke model JSON differs')
                    checks.append({'path': path, 'http_status': status, 'response_sha256': hashlib.sha256(raw).hexdigest()})
            return {'checks': checks, 'same_keepalive_connection': True, 'synthetic_upstream': True,
                    'production_ports_used': False, 'temporary_journal_and_activity': True}
        finally:
            for service in (server, upstream):
                service.shutdown()
                service.server_close()


def main():
    receipt = cfg.STATE / 'admission_smoke_attempt_2.json'
    state = {'status': 'admission_smoke_running', 'instance_id': cfg.INSTANCE, 'started_at': time.time()}
    try:
        initial = identity()
        state.update(initial, **run_smoke())
        if identity() != initial:
            raise ValueError('Smoke code changed while running')
        state['status'] = 'admission_smoke_verified'
    except Exception as error:
        state.update(status='admission_smoke_failed', error_class=type(error).__name__)
    state['completed_at'] = time.time()
    cfg.atomic(receipt, state)
    print(json.dumps({'status': state['status'], 'receipt': str(receipt)}), flush=True)
    return 0 if state['status'] == 'admission_smoke_verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
