"""Fixed four-replica routing over the unchanged content-free comparison meter."""
import os
import re

import quad_config as config
from quad_admission import install_admission

METER_SHA = '8ef745e6df482e02df494422dcbf96eaf56cb67d278656d4f14699b1e429cb93'
METER_PATH = config.ROOT / 'fast_native2_20260911/dual_gpu_20260911/dual_metered_proxy.py'


def select_upstream(attribution, _default=None, _route_path=None):
    try:
        deployment = config.deployment()
        identity = config.fresh_identity()
        method = attribution.get('method')
        sample, question = attribution.get('sample_id'), attribution.get('question_id')
        run_id = attribution.get('run_id')
        if not isinstance(sample, str) or not sample or question not in (None, '', sample):
            raise ValueError('Conflicting or missing history attribution')
        if method == 'runtime_probe':
            match = re.fullmatch(r'quad_probe_(sm[01]|lm[01])_[A-Za-z0-9_-]+', run_id or '')
            if not match or sample != run_id:
                raise ValueError('Synthetic probe must identify its exact replica')
            lane, canonical = match.group(1), False
            protocol = None
        else:
            config.require_ready(deployment)
            if method not in deployment['protocols']:
                raise ValueError('Unknown canonical method')
            protocol = deployment['protocols'][method]
            expected_run = protocol if method == 'simplemem' else 'native7_20260910'
            if run_id != expected_run:
                raise ValueError('Canonical metering protocol/run mismatch')
            routes = config.read(config.STATE / ('routes_' + method + '.json'))
            expected = {'schema': 'quad-history-routes-v1', 'deployment_sha256': config.DEPLOYMENT_SHA,
                        'instance_id': config.INSTANCE, 'method': method, 'protocol_sha256': protocol, **identity}
            if any(routes.get(key) != value for key, value in expected.items()):
                raise ValueError('Routing manifest belongs to another run')
            route = routes['histories'].get(sample)
            if route is None or not routes['dispatches'].get(sample):
                raise ValueError('History has no persisted dispatch')
            lane = route['lane']
            binding = deployment['lanes'][lane]
            if (binding['method'] != method or route.get('instance_id') != config.INSTANCE
                    or any(route.get(key) != binding[key] for key in ('gpu_uuid', 'gpu_index', 'api_base'))):
                raise ValueError('History lane identity differs')
            canonical = True
        binding = deployment['lanes'][lane]
        return binding['api_base'], {
            'inference_lane': lane, 'inference_instance_id': config.INSTANCE,
            'inference_gpu_uuid': binding['gpu_uuid'], 'inference_gpu_index': binding['gpu_index'],
            'routing_protocol_sha256': protocol, 'deployment_sha256': config.DEPLOYMENT_SHA,
            'canonical_benchmark': canonical, **identity}
    except (OSError, KeyError, TypeError) as error:
        raise ValueError('Required routing proof is unavailable') from error


def main():
    if os.environ.get('CONTAINER_ID') != config.INSTANCE:
        raise ValueError('Wrong instance')
    deployment = config.deployment()
    config.fresh_identity()
    if config.sha(METER_PATH) != METER_SHA:
        raise ValueError('Frozen comparison meter source changed')
    meter = config.load_module('quad_existing_meter', METER_PATH)
    meter.select_upstream = select_upstream
    os.environ['METER_INSTANCE_ID'] = config.INSTANCE
    os.environ['METER_ACTIVITY_PATH'] = str(config.STATE / 'inference_activity.json')
    with meter.MeteredProxy(('127.0.0.1', 18083), deployment['lanes']['sm0']['api_base'],
            config.STATE / 'request_usage.jsonl', 'unassigned', 'unassigned', 600,
            comparison_policy=False) as server:
        install_admission(server, meter)
        print('Four-replica metered API listening on 127.0.0.1:18083', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()