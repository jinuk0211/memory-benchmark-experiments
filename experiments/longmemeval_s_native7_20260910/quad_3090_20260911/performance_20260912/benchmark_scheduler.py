"""Measure coordinator overhead with two busy workers; no inference is invoked."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import quad_config as config
import quad_scheduler


def measure(module, method, directory):
    counts = {'attempt_scans': 0, 'publications': 0}
    def attempts(qid):
        counts['attempt_scans'] += 1
        return len(list((directory / qid).glob('attempt_*')))
    def publish(*_):
        counts['publications'] += 1
    adapter = SimpleNamespace(method=method, ids=[str(i) for i in range(500)],
                              completed={}, failures={}, attempts=attempts, publish=publish)
    routes = SimpleNamespace(config=config.deployment(), fresh={}, lane=lambda _: None)
    scheduler = module.Scheduler(adapter, routes, 0)
    scheduler.active = {lane: {'qid': str(index), 'ordinal': 1, 'process': Mock(pid=index+1)}
                        for index, lane in enumerate(scheduler.lanes)}
    clock = [0]
    def sleep(seconds):
        clock[0] += seconds
        if clock[0] >= 60:
            scheduler.stop()
    def reap(*_):
        if scheduler.stopping:
            scheduler.active.clear()
    begin = time.perf_counter()
    with patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]), \
            patch.object(module.time, 'sleep', side_effect=sleep), \
            patch.object(module, 'model_ready', return_value=True), \
            patch.object(config, 'require_ready'), patch.object(config, 'atomic'), \
            patch.object(module.shutil, 'disk_usage', return_value=Mock(free=10**12)), \
            patch.object(scheduler, 'reap', side_effect=reap):
        scheduler.execute()
    counts['coordinator_wall_seconds'] = time.perf_counter() - begin
    counts['unchanged_poll_publications'] = counts['publications'] - 1
    return counts


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('scheduler_before', folder / 'quad_scheduler.before.py')
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    report = {'scope': 'simulated 60 seconds, 500 IDs, two busy workers; no GPU throughput claim',
              'methods': {}}
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for i in range(500):
            (directory / str(i)).mkdir()
        for method in ('simplemem', 'lightmem'):
            report['methods'][method] = {label: measure(module, method, directory)
                                        for label, module in [('before', before), ('after', quad_scheduler)]}
    (folder / 'scheduler_benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
