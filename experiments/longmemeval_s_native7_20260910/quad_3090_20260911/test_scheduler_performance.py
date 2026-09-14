"""Behavior regressions for queue polling without redundant filesystem work."""
from unittest.mock import Mock, patch

from test_quad_queues import Adapter, Fixture
from quad_scheduler import Scheduler


class PollingTests(Fixture):
    def setUp(self):
        super().setUp()
        self.adapter = Adapter(self.directory, ids=tuple(str(i) for i in range(500)))
        self.scheduler = Scheduler(self.adapter, self.routes(self.adapter.ids), 77)

    def test_candidate_reads_each_unfinished_attempt_only_once(self):
        self.adapter.completed['0'] = True
        self.adapter.counts.update({str(i): 1 for i in range(1, 400)})
        with patch.object(self.adapter, 'attempts', wraps=self.adapter.attempts) as attempts:
            self.assertEqual(self.scheduler.candidate('sm0'), '400')
        self.assertEqual(attempts.call_count, 499)

    def test_busy_lanes_are_reaped_without_scanning_pending_directories(self):
        self.scheduler.active = {'sm0': {'qid': '0', 'ordinal': 1},
                                 'sm1': {'qid': '1', 'ordinal': 1}}
        self.scheduler.next_health = float('inf')
        self.scheduler.health = dict.fromkeys(self.scheduler.lanes, True)
        def finish_drain(*_):
            if self.scheduler.stopping:
                self.scheduler.active.clear()
        with patch.object(self.scheduler, 'candidate') as candidate, \
                patch.object(self.scheduler, 'reap', side_effect=finish_drain), \
                patch.object(self.scheduler, 'publish'), \
                patch('quad_scheduler.shutil.disk_usage', return_value=Mock(free=10**12)), \
                patch('quad_scheduler.time.sleep', side_effect=self.scheduler.stop):
            self.scheduler.execute()
        candidate.assert_not_called()

    def test_unchanged_outputs_publish_every_30_seconds_and_changes_immediately(self):
        self.adapter.publish = Mock()
        with patch('quad_scheduler.time.monotonic') as clock:
            clock.return_value = 0
            self.scheduler.publish('running')
            for second in range(2, 30, 2):
                clock.return_value = second
                self.scheduler.publish('running')
            self.adapter.publish.assert_called_once()
            clock.return_value = 30
            self.scheduler.publish('running')
            self.assertEqual(self.adapter.publish.call_count, 2)
            clock.return_value = 31
            self.adapter.completed['0'] = True
            self.scheduler.publish('running')
            self.scheduler.disabled.add('sm1')
            self.scheduler.publish('running')
            self.scheduler.publish('draining')
            self.scheduler.publish('finished')
            self.assertEqual(self.adapter.publish.call_count, 6)

    def test_failed_publication_is_retried_without_waiting(self):
        self.adapter.publish = Mock(side_effect=[OSError('disk full'), None])
        with self.assertRaises(OSError):
            self.scheduler.publish('running')
        self.scheduler.publish('running')
        self.assertEqual(self.adapter.publish.call_count, 2)
