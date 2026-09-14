"""FIFO admission and queue-only SSE heartbeats around the frozen native meter."""
from collections import deque
from contextlib import contextmanager
import json
import math
import select
import socket
import threading
import time
from urllib.parse import urlsplit

import quad_config as config


class AdmissionExpired(ValueError):
    pass


class LaneAdmission:
    def __init__(self):
        self.changed = threading.Condition()
        self.waiting = deque()
        self.active = 0
        self.sequence = 0

    @contextmanager
    def acquire(self, deadline, heartbeat, interval):
        ticket = object()
        admitted = False
        with self.changed:
            self.waiting.append(ticket)
        try:
            while True:
                heartbeat()
                with self.changed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AdmissionExpired('History admission deadline exceeded')
                    if self.waiting[0] is ticket and self.active < 2:
                        self.waiting.popleft()
                        self.active += 1
                        self.sequence += 1
                        grant = {'admission_sequence': self.sequence, 'admission_active_at_start': self.active}
                        admitted = True
                        self.changed.notify_all()
                        break
                    self.changed.wait(min(interval, remaining))
            yield grant
        finally:
            with self.changed:
                if admitted:
                    self.active -= 1
                else:
                    self.waiting.remove(ticket)
                self.changed.notify_all()


def remaining_history_seconds(attribution):
    manifest = config.read(config.STATE / 'routes_simplemem.json')
    dispatch = manifest['dispatches'][attribution['sample_id']][-1]
    started = dispatch['started_at']
    if not isinstance(started, (int, float)) or not math.isfinite(started):
        raise ValueError('Invalid history dispatch time')
    return config.HISTORY_TIMEOUT - (time.time() - started) - 15


def install_admission(server, meter, *, heartbeat_seconds=15, queue_wait_seconds=3600):
    """Install before serve_forever; all other request paths retain the base handler."""
    if (not 0 < heartbeat_seconds <= 15 or not 0 < queue_wait_seconds <= 4800):
        raise ValueError('Admission heartbeat/wait bounds exceeded')
    context = threading.local()
    lanes = {name: LaneAdmission() for name in ('sm0', 'sm1')}
    server.simplemem_admission = lanes
    original_stream, original_route = server.client.stream, meter.select_upstream

    class Handler(meter.ProxyHandler):
        def handle_one_request(self):
            self.admission_metrics = None
            self.early_headers = False
            return super().handle_one_request()

        def do_POST(self):
            self.admission_started = time.monotonic()
            context.handler = self
            try:
                super().do_POST()
            finally:
                context.handler = None

        def _route(self):
            path, attribution = super()._route()
            self.admission_attribution = attribution
            return path, attribution

        def chunk(self, data):
            self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
            self.wfile.flush()

        def heartbeat(self):
            readable, _, _ = select.select([self.connection], [], [], 0)
            if readable and not self.connection.recv(1, socket.MSG_PEEK):
                raise BrokenPipeError('Queued client disconnected')
            self.chunk(b': waiting for inference slot\n\n')

        def _response_headers(self, response, length=None):
            if not self.early_headers:
                super()._response_headers(response, length)

        def _json(self, status, payload):
            if not self.early_headers:
                return super()._json(status, payload)
            kind = (self.admission_metrics or {}).get('admission_error', 'upstream_transport_error')
            self.close_connection = True
            self.chunk(b'data: ' + json.dumps({'error': {'message': 'Inference request failed',
                                                       'type': kind, 'code': status}}).encode() + b'\n\n')
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()

        def _record_attempt(self, record, started):
            if self.admission_metrics is not None:
                record.update(self.admission_metrics)
                record['client_http_status'] = 200 if self.early_headers else record.get('client_http_status')
                if record.get('admission_error'):
                    record['error_type'] = record['admission_error']
            super()._record_attempt(record, started)

    def route(attribution, default, *args):
        result = original_route(attribution, default, *args)
        handler = getattr(context, 'handler', None)
        if handler is not None:
            handler.admission_routing = result[1]
        return result

    class Response:
        def __init__(self, response, handler, upstream_started):
            self.response, self.handler = response, handler
            self.upstream_started = upstream_started
            self.status_code = response.status_code
            self.headers = {'content-type': 'text/event-stream'}

        def iter_bytes(self):
            for chunk in self.response.iter_bytes():
                if chunk and self.handler.admission_metrics['ttfb_seconds'] is None:
                    self.handler.admission_metrics['ttfb_seconds'] = time.monotonic() - self.handler.admission_started
                    self.handler.admission_metrics['upstream_ttfb_seconds'] = time.monotonic() - self.upstream_started
                yield chunk

    class ErrorResponse(Response):
        def iter_bytes(self):
            # Early SSE headers cannot be replaced with an upstream JSON error status.
            raw = self.response.read()
            metrics = self.handler.admission_metrics
            metrics['ttfb_seconds'] = time.monotonic() - self.handler.admission_started
            metrics['upstream_ttfb_seconds'] = time.monotonic() - self.upstream_started
            metrics['error_class'] = 'UpstreamHTTPError' if self.status_code >= 300 else 'UnexpectedContentType'
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            if not isinstance(payload.get('error'), dict):
                payload['error'] = {'message': 'Upstream did not return an SSE response',
                                    'type': 'upstream_http_error', 'code': self.status_code}
            yield b'data: ' + json.dumps(payload).encode() + b'\n\n'


    @contextmanager
    def stream(method, url, **kwargs):
        handler = getattr(context, 'handler', None)
        attribution = getattr(handler, 'admission_attribution', {})
        try:
            payload = json.loads(kwargs.get('content', b''))
        except (ValueError, TypeError):
            payload = None
        routing = getattr(handler, 'admission_routing', {})
        admitted_path = (handler is not None and method == 'POST'
                         and urlsplit(url).path.endswith('/chat/completions')
                         and routing.get('inference_lane') in lanes
                         and isinstance(payload, dict) and payload.get('stream') is True)
        if not admitted_path:
            with original_stream(method, url, **kwargs) as response:
                yield response
            return
        metrics = handler.admission_metrics = {'queue_wait_seconds': 0.0, 'upstream_seconds': None,
                                               'ttfb_seconds': None, 'upstream_ttfb_seconds': None, 'admission_wait_seconds': 0.0,
                                               'upstream_started_at': None, 'upstream_completed_at': None}
        waiting = time.monotonic()
        remaining = remaining_history_seconds(attribution) if routing['canonical_benchmark'] else queue_wait_seconds
        deadline = waiting + min(queue_wait_seconds, remaining)
        handler.send_response(200)
        handler.send_header('Content-Type', 'text/event-stream')
        handler.send_header('Transfer-Encoding', 'chunked')
        handler.send_header('Cache-Control', 'no-cache')
        handler.end_headers()
        handler.wfile.flush()
        handler.early_headers = True
        upstream_started = None
        try:
            with lanes[routing['inference_lane']].acquire(deadline, handler.heartbeat, heartbeat_seconds) as grant:
                metrics.update(grant)
                metrics['queue_wait_seconds'] = time.monotonic() - waiting
                metrics['admission_wait_seconds'] = metrics['queue_wait_seconds']
                upstream_started = time.monotonic()
                metrics['upstream_started_at'] = time.time()
                try:
                    # Exact prepared body/headers and the existing httpx timeout pass through.
                    with original_stream(method, url, **kwargs) as response:
                        wrapped = Response if 'text/event-stream' in response.headers.get('content-type', '') else ErrorResponse
                        if response.status_code >= 300:
                            metrics['error_class'] = 'UpstreamHTTPError'
                        yield wrapped(response, handler, upstream_started)
                finally:
                    # Record the interval before releasing the slot, not after journal fsync.
                    metrics['upstream_completed_at'] = time.time()
                    metrics['upstream_seconds'] = time.monotonic() - upstream_started
        except AdmissionExpired:
            metrics['admission_error'] = 'admission_timeout'
            raise
        finally:
            if upstream_started is None:
                metrics['queue_wait_seconds'] = time.monotonic() - waiting
                metrics['admission_wait_seconds'] = metrics['queue_wait_seconds']

    server.RequestHandlerClass = Handler
    server.client.stream = stream
    meter.select_upstream = route