"""Two CUDA-bound LightMem lanes using unchanged native runner and merge validators."""
import argparse
from pathlib import Path
import signal
import sys
import time

import quad_config as config
from quad_scheduler import Scheduler


class LightMem:
    method = 'lightmem'

    def __init__(self, root, plan, vendor):
        self.root, self.state, self.vendor = root, root / 'fast_native2_20260911', vendor
        self.queue = config.load_module('quad_existing_lightmem', self.state / 'lightmem_queue.py')
        self.args = argparse.Namespace(plan=plan, protocol_sha256=config.deployment()['protocols'][self.method])
        self.ids, self.rows, self.protocol, self.command, self.original = self.queue.preflight(self.args)
        self.support = {str(root / name): config.sha(root / name)
                        for name in ('export_official.py', 'score_diagnostic_f1.py', 'run_all.py')}
        self.lanes = self.state / 'lightmem_lanes'
        for index in range(2):
            (self.lanes / ('lane_' + str(index))).mkdir(parents=True, exist_ok=True)
        self.journal_path = self.state / 'lightmem_attempts.json'
        self.journal = config.read(self.journal_path) if self.journal_path.exists() else {}
        if set(self.journal) - set(self.ids):
            raise ValueError('Attempt journal contains noncanonical IDs')
        self.fresh = config.fresh_identity()
        self.routes = config.Routes(self.method, self.ids, [])
        self.completed, self.failures, self.evidence = {}, {}, []
        for qid, entry in self.journal.items():
            for previous in entry.get('history', []) + [entry]:
                config.require_dispatch(previous['quad_route'], self.method, qid, self.fresh)
        for qid in self.ids:
            self.reconcile(qid)
            targets = list(self.lanes.glob('lane_*/' + qid))
            candidates = [target for target in targets if (target / 'prediction.json').exists()]
            if len(candidates) > 1:
                raise ValueError('Duplicate LightMem completion')
            for target in targets:
                receipts = list(target.glob('quad_dispatch_*.json'))
                if not receipts and any(target.iterdir()):
                    raise ValueError('Existing LightMem history lacks fresh-run dispatch evidence')
                for path in receipts:
                    config.require_dispatch(config.read(path), self.method, qid, self.fresh)
                    self.evidence.append((qid, path))
            if candidates:
                self.validate(candidates[0], qid)
                attempt = candidates[0] / config.read(candidates[0] / 'prediction.json')['attempt']
                config.require_dispatch(config.read(attempt / 'quad_route.json'), self.method, qid, self.fresh)
                self.completed[qid] = candidates[0]
            elif self.attempts(qid):
                self.failures[qid] = self.journal.get(qid, {'question_id': qid, 'reason': 'Existing incomplete attempt preserved'})
        self.config_path = self.state / 'lightmem_queue_config.json'
        expected = {'command': self.command, 'protocol_sha256': self.args.protocol_sha256,
                    'canonical_ids': self.ids, 'maximum_attempts_per_question': 2, 'support_sha256': self.support}
        if self.config_path.exists() and config.read(self.config_path) != expected:
            raise ValueError('Saved LightMem queue configuration differs')
        config.atomic(self.config_path, expected)

    def reconcile(self, qid):
        receipts = self.routes.receipts(qid)
        if not receipts:
            return
        index = int(receipts[0]['route']['lane'][-1])
        target = self.lanes / ('lane_' + str(index)) / qid
        target.mkdir(parents=True, exist_ok=True)
        for receipt in receipts:
            path = target / ('quad_dispatch_' + str(receipt['dispatch']['ordinal']) + '.json')
            if path.exists() and config.read(path) != receipt:
                raise ValueError('LightMem dispatch evidence differs from manifest')
            if not path.exists():
                config.atomic(path, receipt)
        if self.journal.get(qid, {}).get('attempt', 0) < len(receipts):
            previous = self.journal.get(qid)
            history = previous.get('history', []) + [previous] if previous else []
            self.journal[qid] = {'lane': index, 'attempt': len(receipts), 'status': 'interrupted_dispatch',
                                 'quad_route': receipts[-1], 'history': history}
            config.atomic(self.journal_path, self.journal)
        self.bind_attempts(qid, target)

    def bind_attempts(self, qid, target):
        receipts = self.routes.receipts(qid)
        used = set()
        for attempt in sorted(target.glob('attempt_*')):
            if not attempt.is_dir():
                continue
            started = int(attempt.name.removeprefix('attempt_')) / 1e9
            eligible = [receipt for receipt in receipts if receipt['dispatch']['started_at'] <= started]
            if not eligible:
                raise ValueError('Native LightMem attempt predates this fresh dispatch')
            receipt = eligible[-1]
            ordinal = receipt['dispatch']['ordinal']
            if ordinal in used:
                raise ValueError('Multiple native attempts share one dispatch')
            used.add(ordinal)
            proof = attempt / 'quad_route.json'
            if proof.exists() and config.read(proof) != receipt:
                raise ValueError('Native LightMem attempt route differs')
            if not proof.exists():
                config.atomic(proof, receipt)

    def attempts(self, qid):
        physical = sum(1 for path in self.lanes.glob('lane_*/' + qid + '/attempt_*') if path.is_dir())
        return max(physical, self.journal.get(qid, {}).get('attempt', 0))

    def validate(self, target, qid):
        result = self.queue.validate_prediction(target, self.rows[qid], self.protocol)
        attempt = target / config.read(target / 'prediction.json')['attempt']
        if (attempt / 'failure.json').exists() or not (attempt / 'qdrant').is_dir():
            raise ValueError('LightMem completion has failure or lacks Qdrant evidence')
        return result

    def prepare(self, qid, lane, _route, routes):
        index = int(lane[-1])
        run_dir = self.lanes / ('lane_' + str(index))
        target = run_dir / qid
        target.mkdir(exist_ok=True)
        ordinal = self.attempts(qid) + 1
        token = 'lightmem_' + qid + '_' + str(ordinal)
        receipt = routes.begin(qid, lane, token, ordinal)
        config.atomic(target / ('quad_dispatch_' + str(ordinal) + '.json'), receipt)
        previous = self.journal.get(qid)
        history = (previous.get('history', []) + [{key: value for key, value in previous.items() if key != 'history'}]) if previous else []
        log = self.state / ('lightmem_' + qid + '_' + str(time.time_ns()) + '.log')
        self.journal[qid] = {'lane': index, 'status': 'dispatched', 'log': str(log), 'time': time.time(),
                             'attempt': ordinal, 'history': history, 'quad_route': receipt}
        config.atomic(self.journal_path, self.journal)
        ids_file = self.state / ('lightmem_lane_' + str(index) + '_ids.json')
        config.atomic(ids_file, [qid])
        argv = list(self.command)
        argv[argv.index('--run-dir') + 1] = str(run_dir)
        argv[argv.index('--api-base') + 1] = (
            'http://127.0.0.1:18083/meter/native7_20260910/lightmem/native/' + qid + '/' + qid + '/v1')
        argv.extend(['--ids-file', str(ids_file)])
        return {'ordinal': ordinal, 'target': target, 'log': log, 'argv': argv, 'route_receipt': receipt}

    def finish(self, job, code, process_code):
        qid, target = job['qid'], job['target']
        entry = self.journal[qid]
        entry.update(returncode=code, process_returncode=process_code, completed_at=time.time())
        self.bind_attempts(qid, target)
        try:
            self.validate(target, qid)
            self.completed[qid] = target
            self.failures.pop(qid, None)
            entry['status'] = 'generated'
        except (OSError, ValueError, KeyError) as error:
            entry.update(status='failed', error_type=type(error).__name__)
            self.failures[qid] = dict(entry)
        config.atomic(self.journal_path, self.journal)

    def publish(self, active, phase):
        complete = len(self.completed) == 500 and not self.failures and not active
        config.atomic(self.state / 'lightmem_status.json', {
            'method': self.method, 'planned': 500, 'population': 500, 'generated': len(self.completed),
            'failed': len(self.failures), 'failed_ids': list(self.failures),
            'inflight': [job['qid'] for job in active.values()],
            'pending': 500 - len(self.completed) - len(active),
            'status': 'generation_complete' if complete else 'incomplete' if phase == 'finished' else phase,
            'generation_complete': complete, 'official_judge_pending': True, 'time': time.time()})

    def aggregate(self):
        self.queue.preflight(self.args)
        if any(config.sha(Path(path)) != expected for path, expected in self.support.items()):
            raise ValueError('Frozen export/scoring support changed')
        destination = self.original.parent / 'lightmem_fast_native2'
        self.queue.aggregate(self.ids, self.rows, self.protocol, self.completed, destination)
        self.queue.finish_outputs(self.ids, self.command, self.vendor, destination, self.state)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=config.ROOT / 'fast_native2_20260911/lightmem_plan.json')
    parser.add_argument('--vendor', type=Path, default=config.ROOT / 'official_longmemeval')
    args = parser.parse_args()
    config.require_ready()
    sys.path.insert(0, str(config.ROOT))
    with (config.ROOT / 'fast_native2_20260911/lightmem_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        adapter = LightMem(config.ROOT, args.plan, args.vendor)
        routes = config.Routes(adapter.method, adapter.ids, adapter.evidence)
        adapter.routes = routes
        runner = Scheduler(adapter, routes, lock.fileno())
        signal.signal(signal.SIGTERM, runner.stop)
        signal.signal(signal.SIGINT, runner.stop)
        if not runner.execute():
            return 1
        adapter.aggregate()
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
