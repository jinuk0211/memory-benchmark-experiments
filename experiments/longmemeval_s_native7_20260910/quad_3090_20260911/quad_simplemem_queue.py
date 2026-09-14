"""Two SimpleMem lanes; unchanged v4 workers and one canonical publisher."""
import argparse
import hashlib
from pathlib import Path
import signal
import sys

import quad_config as config
from quad_scheduler import Scheduler


class SimpleMem:
    method = 'simplemem'

    def __init__(self, root, runtime_receipt):
        self.root, self.state = root, root / 'fast_native2_20260911'
        self.run = root / 'runs/simplemem_native_dialogues_v4'
        self.runner = root / 'official_recovery/simplemem_native_dialogues_v4/runner.py'
        self.queue = config.load_module('quad_existing_simplemem', self.state / 'simplemem_queue.py')
        if config.sha(self.runner) != self.queue.RUNNER_SHA:
            raise ValueError('Pinned native runner changed')
        receipt = config.read(runtime_receipt)
        if (receipt.get('status') != 'runtime_verified' or receipt.get('candidate') != str(self.runner.parent)
                or receipt['integrity']['runtime_files']['official_recovery/simplemem_native_dialogues_v4/runner.py'] != self.queue.RUNNER_SHA):
            raise ValueError('Original SimpleMem runtime receipt changed')
        if self.queue.live_native_workers(self.runner, self.run):
            raise ValueError('An existing native SimpleMem worker must drain first')
        dataset = root / 'longmemeval_s_cleaned.json'
        if config.sha(dataset) != self.queue.DATA_SHA:
            raise ValueError('Canonical dataset changed')
        self.rows = {row['question_id']: row for row in config.read(dataset)}
        self.ids = list(self.rows)
        if len(self.ids) != 500:
            raise ValueError('Expected canonical 500')
        self.native = config.load_module('quad_native_simplemem', self.runner)
        self.protocol = config.read(self.run / 'protocol.json')
        runtime = self.protocol['runtime']
        embedding = Path(runtime['embedding_model'])
        expected_runtime = {'method': 'simplemem', 'api_base': 'http://127.0.0.1:18083/v1', 'model': self.native.MODEL,
                            'embedding_model': str(embedding), 'embedding_revision': self.native.MINILM_REVISION}
        expected = {'dataset_sha256': self.queue.DATA_SHA, 'population_ids': self.ids, 'runtime': expected_runtime,
                    'source_files_sha256': self.native.source_hashes(), 'policy': self.native.POLICY,
                    'embedding_config_sha256': {name: self.native.sha(embedding / name) for name in
                        ('config.json', 'modules.json', 'sentence_bert_config.json')}}
        self.protocol_hash = config.deployment()['protocols'][self.method]
        if self.protocol != expected or self.native.digest(self.protocol) != self.protocol_hash:
            raise ValueError('Frozen SimpleMem protocol/source mismatch')
        self.histories = {qid: self.run / 'histories' / hashlib.sha256(qid.encode()).hexdigest()[:24] for qid in self.ids}
        self.identities = {qid: {'protocol_sha256': self.protocol_hash,
            'source_sha256': self.native.digest(self.native.source_only(row)),
            'query_sha256': self.native.digest({key: row[key] for key in ('question_id', 'question', 'question_date')})}
            for qid, row in self.rows.items()}
        self.fresh = config.fresh_identity()
        self.routes = config.Routes(self.method, self.ids, [])
        self.completed, self.failures = {}, {}
        for qid in self.ids:
            self.reconcile(qid)
            for attempt in self.histories[qid].glob('attempt_*'):
                config.require_dispatch(config.read(attempt / 'quad_route.json'), self.method, qid, self.fresh)
            prediction = self.verified(qid)
            if prediction is not None:
                self.completed[qid] = prediction
            elif self.attempts(qid):
                self.failures[qid] = {'question_id': qid, 'attempts': self.attempts(qid),
                                      'reason': 'Current fresh run incomplete/failed native attempts preserved'}
        self.evidence = [(qid, path) for qid in self.ids for path in self.histories[qid].glob('attempt_*/quad_route.json')]

    def reconcile(self, qid):
        for receipt in self.routes.receipts(qid):
            ordinal = receipt['dispatch']['ordinal']
            attempt = self.histories[qid] / f'attempt_{ordinal:04d}'
            if receipt['dispatch']['id'] != str(attempt):
                raise ValueError('SimpleMem reservation path differs')
            proof = attempt / 'quad_route.json'
            if not proof.exists():
                if attempt.exists() and any(path.name != 'quad_route.json.tmp' for path in attempt.iterdir()):
                    raise ValueError('Existing native artifacts lost their prelaunch route receipt')
                attempt.mkdir(parents=True, exist_ok=True)
                config.atomic(proof, receipt)
            elif config.read(proof) != receipt:
                raise ValueError('SimpleMem attempt differs from its reserved dispatch')

    def attempts(self, qid):
        return len([path for path in self.histories[qid].glob('attempt_*') if path.is_dir()])

    def verified(self, qid):
        value = self.native.verified(self.histories[qid], self.identities[qid])
        if value is not None:
            for receipt in self.histories[qid].glob('attempt_*/completion.json'):
                if config.read(receipt).get('identity') == self.identities[qid] and (receipt.parent / 'failure.json').exists():
                    raise ValueError('A sealed completion has a failure marker')
        return value

    def prepare(self, qid, lane, _route, routes):
        ordinal = self.attempts(qid) + 1
        expected = self.histories[qid] / f'attempt_{ordinal:04d}'
        route_receipt = routes.begin(qid, lane, str(expected), ordinal)
        attempt = self.native.next_attempt(self.histories[qid])
        if attempt != expected:
            raise ValueError('Native attempt allocation differs from reservation')
        config.atomic(attempt / 'quad_route.json', route_receipt)
        row = self.rows[qid]
        for name, value in (
            ('source.json', self.native.source_only(row)),
            ('query.json', {key: row[key] for key in ('question_id', 'question', 'question_date')}),
            ('worker.json', {'identity': self.identities[qid], 'runtime': self.protocol['runtime'],
                             'source_files_sha256': self.protocol['source_files_sha256']})):
            self.native.save_json(attempt / name, value)
        return {'ordinal': ordinal, 'attempt': attempt, 'log': attempt / 'console.log',
                'argv': [sys.executable, str(self.runner), '--worker-dir', str(attempt)]}

    def finish(self, job, code, process_code):
        qid, attempt = job['qid'], job['attempt']
        result = {'question_id': qid, 'attempt': str(attempt), 'attempts': self.attempts(qid),
                  'exit_code': code, 'process_returncode': process_code, 'completed_at': config.time.time()}
        config.atomic(attempt / 'quad_coordinator_result.json', result)
        prediction = self.verified(qid)
        if prediction is not None:
            self.completed[qid] = prediction
            self.failures.pop(qid, None)
        else:
            self.failures[qid] = result
            if not (attempt / 'failure.json').exists():
                config.atomic(attempt / 'failure.json', result)

    def publish(self, active, phase):
        qids = [job['qid'] for job in active.values()]
        self.queue.publish(self.run, self.state, self.ids, self.completed, self.failures,
                           active=qids[0] if qids else None, phase=phase)


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-receipt', type=Path,
                        default=config.ROOT / 'fast_native2_20260911/simplemem_v4_runtime.json')
    args = parser.parse_args()
    config.require_ready()
    sys.path.insert(0, str(config.ROOT))
    state = config.ROOT / 'fast_native2_20260911'
    with (state / 'simplemem_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        adapter = SimpleMem(config.ROOT, args.runtime_receipt)
        routes = config.Routes(adapter.method, adapter.ids, adapter.evidence)
        adapter.routes = routes
        runner = Scheduler(adapter, routes, lock.fileno())
        signal.signal(signal.SIGTERM, runner.stop)
        signal.signal(signal.SIGINT, runner.stop)
        return 0 if runner.execute() else 1


if __name__ == '__main__':
    raise SystemExit(main())
