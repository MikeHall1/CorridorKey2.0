"""Analytic export milestone clocks, coverage, races and bounded failure."""
import json
import threading
import unittest
from unittest.mock import Mock

from corridorkey.output_telemetry import OutputTelemetry


class Clock:
    def __init__(self, value=100):
        self.value = value
    def __call__(self):
        return self.value


class OutputTelemetryTests(unittest.TestCase):
    def test_exact_multioutput_nearest_rank_and_written_before_submitted(self):
        clock = Clock()
        telemetry = OutputTelemetry({'alpha', 'fg'}, 3, native_completion=True, origin=100, clock=clock)
        for index in range(3):
            clock.value = 101 + index * 10
            telemetry.render_ready((index,))
            # Both native consumers may finish before queue.put returns.
            telemetry.written('alpha', index, clock.value + 1)
            telemetry.written('fg', index, clock.value + 3)
            clock.value += 4
            telemetry.submitted('alpha', (index,))
            clock.value += 1
            telemetry.submitted('fg', (index,))
        report = telemetry.finish()
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['all_native_completed_frames'], 3)
        self.assertEqual(report['pending_frames'], 0)
        self.assertEqual(report['native_write_latency']['p50_seconds'], 3)
        self.assertEqual(report['native_write_latency']['p95_seconds'], 3)
        self.assertEqual(report['queue_accept_latency']['p99_seconds'], 5)
        self.assertEqual(report['completion_spacing']['count'], 2)
        self.assertEqual(report['completion_spacing']['p50_seconds'], 10)
        self.assertEqual(report['per_output']['fg']['submitted_frames'], 3)
        json.dumps(report, allow_nan=False)
        self.assertEqual(telemetry.finish(), report)

    def test_thread_callbacks_can_arrive_out_of_global_time_order(self):
        clock = Clock(101)
        telemetry = OutputTelemetry(('alpha', 'fg'), 1, native_completion=True, origin=100, clock=clock)
        telemetry.render_ready([0])
        # Different writer threads captured different instants before taking lock.
        late = threading.Thread(target=lambda: telemetry.written('fg', 0, 104))
        early = threading.Thread(target=lambda: telemetry.written('alpha', 0, 102))
        late.start()
        late.join(timeout=1)
        early.start()
        early.join(timeout=1)
        clock.value = 105
        telemetry.submitted('alpha', [0])
        telemetry.submitted('fg', [0])
        self.assertIsNone(telemetry.finish()['failure'])
        self.assertEqual(telemetry.snapshot()['native_write_latency']['p50_seconds'], 3)

    def test_missing_output_and_failed_finalization_are_partial(self):
        clock = Clock(101)
        telemetry = OutputTelemetry(['alpha', 'fg'], 2, native_completion=True, origin=100, clock=clock)
        telemetry.render_ready([0])
        telemetry.submitted('alpha', [0])
        telemetry.submitted('fg', [0])
        telemetry.written('alpha', 0, 102)
        report = telemetry.finish()
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['missing_native_frames'], 2)
        self.assertEqual(report['per_output']['alpha']['native_completed_frames'], 1)
        self.assertEqual(report['per_output']['fg']['native_completed_frames'], 0)
        self.assertIsNone(report['native_write_latency']['p99_seconds'])

    def test_no_truncated_window_quantiles_on_sample_or_pending_limit(self):
        for kwargs in ({'max_frames': 1}, {'max_inflight_frames': 1}):
            clock = Clock(101)
            telemetry = OutputTelemetry(['alpha'], 2, native_completion=True, origin=100, clock=clock, **kwargs)
            telemetry.render_ready([0])
            telemetry.render_ready([1])
            report = telemetry.finish()
            self.assertNotEqual(report['status'], 'complete')
            self.assertIsNotNone(report['failure'])
            self.assertIsNone(report['render_ready_offsets']['p99_seconds'])
            self.assertLessEqual(report['registered_frames'], 1)
            self.assertLess(len(json.dumps(report)), 10000)

    def test_invalid_events_and_clocks_do_not_raise_or_claim_zero_latency(self):
        for event in ('duplicate', 'unknown', 'negative', 'nonfinite', 'backward', 'pre_ready', 'late'):
            with self.subTest(event=event):
                clock = Clock(101)
                telemetry = OutputTelemetry(['alpha'], 1, native_completion=True, origin=100, clock=clock)
                telemetry.render_ready([0])
                if event == 'duplicate':
                    telemetry.render_ready([0])
                elif event == 'unknown':
                    telemetry.written('other', 0, 102)
                elif event == 'negative':
                    telemetry.written('alpha', -1, 102)
                elif event == 'nonfinite':
                    telemetry.written('alpha', 0, float('nan'))
                elif event == 'backward':
                    clock.value = 100
                    telemetry.submitted('alpha', [0])
                elif event == 'pre_ready':
                    telemetry.written('alpha', 0, 100.5)
                else:
                    telemetry.finish()
                    telemetry.written('alpha', 0, 102)
                report = telemetry.finish()
                self.assertIsNotNone(report['failure'])
                self.assertIsNone(report['native_write_latency']['p50_seconds'])
                self.assertNotEqual(report['status'], 'complete')
        telemetry = OutputTelemetry(['alpha'], 1, native_completion=True, origin=100,
                                    clock=Mock(side_effect=RuntimeError('clock denied')))
        telemetry.render_ready([0])
        self.assertIsNotNone(telemetry.finish()['failure'])

    def test_snapshot_is_detached_and_never_calls_clock(self):
        clock = Clock(101)
        telemetry = OutputTelemetry(['alpha'], 1, native_completion=False, origin=100, clock=clock)
        telemetry.render_ready([0])
        telemetry.submitted('alpha', [0])
        telemetry.clock = Mock(side_effect=AssertionError('no clocks in snapshot/finish'))
        report = telemetry.finish()
        self.assertEqual(report['queue_accept_latency']['count'], 1)
        self.assertEqual(report['native_write_latency']['status'], 'unavailable')
        report['per_output']['alpha']['submitted_frames'] = -1
        self.assertEqual(telemetry.snapshot()['per_output']['alpha']['submitted_frames'], 1)

    def test_constructor_is_strict_and_failure_hook_is_observation_only(self):
        for args in ({'outputs': 'alpha'}, {'expected_frames': True}, {'origin': float('inf')},
                     {'native_completion': 1}, {'max_frames': 0}):
            settings = {'outputs': ['alpha'], 'expected_frames': 1, 'native_completion': True, 'origin': 0}
            settings.update(args)
            with self.assertRaises(ValueError):
                OutputTelemetry(**settings)
        telemetry = OutputTelemetry(['alpha'], 1, native_completion=True, origin=0)
        telemetry.observation_failed('alpha: hook unavailable')
        self.assertIn('hook unavailable', telemetry.finish()['failure'])
