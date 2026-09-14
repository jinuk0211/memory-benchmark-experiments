"""Pinned four-GPU identity, READY gate, and single-writer per-method routing."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/longmemeval_s_native7_20260910')
STATE = Path('/workspace/quad_3090_20260911')
INSTANCE = '50577514'
DEPLOYMENT_SHA = '46f9747b2ab2ab5d780f46485c02c0a37041af36edad0f1c065b9c47fc42e7ac'
MODEL = 'Qwen/Qwen3.5-9B'
MAX_ATTEMPTS = 2
HISTORY_TIMEOUT = 5400
DATASET_SHA = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deployment(path=None):
    path = path or Path(__file__).with_name('deployment.json')
    if sha(path) != DEPLOYMENT_SHA:
        raise ValueError('Deployment manifest changed')
    return read(path)


def fresh_identity(directory=None):
    directory = directory or STATE
    marker = read(directory / 'FRESH_RUN.json')
    if (marker.get('status') != 'fresh_run_authorized' or marker.get('instance_id') != INSTANCE
            or marker.get('dataset_sha256') != DATASET_SHA
            or marker.get('methods') != {'simplemem': 500, 'lightmem': 500}
            or marker.get('past_results_reused') is not False or marker.get('fresh_generation') is not True
            or marker.get('canonical_population') != 500 or not isinstance(marker.get('run_id'), str)
            or not marker['run_id'].strip()):
        raise ValueError('Expected explicit fresh 500-per-method authorization')
    marker_sha = sha(directory / 'FRESH_RUN.json')
    initialized = read(directory / 'fresh_initialized.json')
    if (initialized.get('status') != 'fresh_initialized' or initialized.get('instance_id') != INSTANCE
            or initialized.get('fresh_run_sha256') != marker_sha):
        raise ValueError('Fresh initialization proof differs')
    return {'fresh_run_sha256': marker_sha, 'execution_run_id': marker['run_id'],
            'fresh_initialized_sha256': sha(directory / 'fresh_initialized.json')}


def require_dispatch(receipt, method, qid, identity=None):
    identity = identity or fresh_identity()
    if (receipt.get('deployment_sha256') != DEPLOYMENT_SHA or receipt.get('method') != method
            or receipt.get('question_id') != qid
            or any(receipt.get(key) != value for key, value in identity.items())):
        raise ValueError('Attempt is not from this authorized fresh run')
    return receipt


def require_ready(config=None, directory=None):
    config = config or deployment()
    directory = directory or STATE
    identity = fresh_identity(directory)
    gate = read(directory / 'READY.json')
    if (os.environ.get('CONTAINER_ID') != INSTANCE or gate.get('instance_id') != INSTANCE
            or gate.get('status') != 'runtime_verified' or gate.get('deployment_sha256') != DEPLOYMENT_SHA
            or not isinstance(gate.get('model_proof'), dict) or not gate['model_proof']
            or gate.get('protocol_proof') != config['protocols']
            or gate.get('fresh_run_sha256') != identity['fresh_run_sha256']):
        raise ValueError('Expected genuine READY for this deployment')
    for lane, binding in config['lanes'].items():
        replica = gate.get('replicas', {}).get(lane, {})
        if (replica.get('status') != 'inference_verified' or replica.get('model') != MODEL
                or replica.get('max_model_len') != 65536
                or any(replica.get(key) != binding[key] for key in ('gpu_uuid', 'api_base'))):
            raise ValueError('Replica runtime proof differs: ' + lane)
    return gate


def child_environment(binding, method):
    result = {key: value for key, value in os.environ.items()
              if not any(word in key.upper() for word in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
    result.update(CONTAINER_ID=INSTANCE, CUDA_VISIBLE_DEVICES=binding['gpu_uuid'] if method == 'lightmem' else '',
                  HF_HOME='/workspace/.hf_home', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  MKL_NUM_THREADS='4', OPENAI_API_KEY='EMPTY', PYTHONUNBUFFERED='1',
                  NO_PROXY='localhost,127.0.0.1,::1', no_proxy='localhost,127.0.0.1,::1',
                  NLTK_DATA='/root/.cache/longmemeval_s_native7_20260910/nltk_data',
                  PYTHONPATH=str(ROOT) + ':' + str(ROOT / 'fast_native2_20260911'))
    return result


class Routes:
    def __init__(self, method, ids, evidence, config=None, directory=None):
        self.config, self.method = config or deployment(), method
        self.directory = directory or STATE
        self.ids = set(ids)
        self.fresh = fresh_identity(self.directory)
        self.path = self.directory / ('routes_' + method + '.json')
        marker = self.directory / ('routes_' + method + '.initialized.json')
        self.identity = {'schema': 'quad-history-routes-v1', 'deployment_sha256': DEPLOYMENT_SHA,
                         'instance_id': INSTANCE, 'method': method,
                         'protocol_sha256': self.config['protocols'][method], **self.fresh}
        if marker.exists() and (read(marker) != self.identity or not self.path.exists()):
            raise ValueError('Initialized routing state is missing or changed')
        self.data = read(self.path) if self.path.exists() else {**self.identity, 'histories': {}, 'dispatches': {}}
        if any(self.data.get(key) != value for key, value in self.identity.items()):
            raise ValueError('Route identity differs')
        if set(self.data['histories']) - set(ids) or set(self.data['dispatches']) - set(ids):
            raise ValueError('Routes contain noncanonical histories')
        for qid, route in self.data['histories'].items():
            self.validate_route(route)
        for qid, dispatches in self.data['dispatches'].items():
            if (qid not in self.data['histories'] or not 1 <= len(dispatches) <= MAX_ATTEMPTS
                    or [item['ordinal'] for item in dispatches] != list(range(1, len(dispatches) + 1))
                    or len({item['id'] for item in dispatches}) != len(dispatches)):
                raise ValueError('Invalid persisted dispatch budget')
        for qid, path in evidence:
            receipt = require_dispatch(read(path), method, qid, self.fresh)
            if (receipt.get('question_id') != qid or receipt.get('route') != self.data['histories'].get(qid)
                    or receipt.get('dispatch') not in self.data['dispatches'].get(qid, [])):
                raise ValueError('Durable dispatch evidence lost its route or differs')
        atomic(self.path, self.data)
        if not marker.exists():
            atomic(marker, self.identity)

    def validate_route(self, route):
        lane = route.get('lane')
        binding = self.config['lanes'].get(lane, {})
        if (binding.get('method') != self.method or route.get('instance_id') != INSTANCE
                or any(route.get(key) != binding.get(key) for key in ('gpu_uuid', 'gpu_index', 'api_base'))):
            raise ValueError('Invalid lane binding')
        return binding

    def lane(self, qid):
        return self.data['histories'].get(qid, {}).get('lane')

    def assign(self, qid, lane):
        if qid not in self.ids:
            raise ValueError('Noncanonical history cannot be dispatched')
        route = self.data['histories'].get(qid)
        if route is not None and route['lane'] != lane:
            raise ValueError('An assigned history cannot move lanes')
        if route is None:
            binding = self.config['lanes'][lane]
            route = {'lane': lane, 'instance_id': INSTANCE, 'assigned_at': time.time(),
                     **{key: binding[key] for key in ('gpu_uuid', 'gpu_index', 'api_base')}}
            self.validate_route(route)
            self.data['histories'][qid] = route
            atomic(self.path, self.data)
        return route

    def receipt(self, qid, dispatch):
        return {'deployment_sha256': DEPLOYMENT_SHA, 'method': self.method, **self.fresh,
                'question_id': qid, 'route': self.data['histories'][qid], 'dispatch': dispatch}

    def receipts(self, qid):
        return [self.receipt(qid, item) for item in self.data['dispatches'].get(qid, [])]

    def begin(self, qid, lane, token, ordinal):
        self.assign(qid, lane)
        previous = self.data['dispatches'].setdefault(qid, [])
        if not 1 <= ordinal <= MAX_ATTEMPTS or ordinal != len(previous) + 1:
            raise ValueError('Dispatch ordinal duplicates or exceeds retry budget')
        dispatch = {'id': token, 'ordinal': ordinal, 'started_at': time.time()}
        previous.append(dispatch)
        atomic(self.path, self.data)
        return self.receipt(qid, dispatch)
