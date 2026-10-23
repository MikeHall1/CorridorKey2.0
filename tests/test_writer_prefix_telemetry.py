"""Recovered files never acquire fabricated queue or native completion times."""
import json
import unittest

from corridorkey.output_telemetry import OutputTelemetry


class WriterPrefixTelemetryTests(unittest.TestCase):
    def test_suffix_population_and_full_render_coverage(self):
        telemetry = OutputTelemetry(('alpha', 'fg'), 4, native_completion=True,
                                    origin=100, clock=lambda: 101, restored_prefix_frames=2)
        telemetry.render_ready((0, 1, 2, 3))
        for role in ('alpha', 'fg'):
            telemetry.written(role, 2, 102)
            telemetry.written(role, 3, 103)
            telemetry.submitted(role, (2, 3))
        report = telemetry.finish()
        self.assertEqual(report['status'], 'partial')
        self.assertIsNone(report['failure'])
        self.assertEqual(report['pending_frames'], 0)
        self.assertEqual(report['render_ready_offsets']['count'], 4)
        self.assertEqual(report['render_ready_offsets']['expected_count'], 4)
        for key in ('native_write_latency', 'queue_accept_latency', 'native_completion_offsets'):
            self.assertEqual(report[key]['count'], 2)
            self.assertEqual(report[key]['expected_count'], 2)
            self.assertEqual(report[key]['status'], 'complete')
        self.assertEqual(report['missing_native_frames'], 0)
        self.assertEqual(report['per_output']['alpha']['native_completed_frames'], 2)
        json.dumps(report, allow_nan=False)

    def test_all_restored_has_no_native_percentiles(self):
        telemetry = OutputTelemetry(('alpha',), 3, native_completion=True,
                                    origin=100, clock=lambda: 101, restored_prefix_frames=3)
        telemetry.render_ready((0, 1, 2))
        report = telemetry.finish()
        self.assertEqual(report['status'], 'partial')
        self.assertIsNone(report['failure'])
        self.assertEqual(report['registered_frames'], 3)
        self.assertEqual(report['pending_frames'], 0)
        self.assertEqual(report['all_native_completed_frames'], 0)
        self.assertEqual(report['native_write_latency']['expected_count'], 0)
        self.assertIsNone(report['native_write_latency']['p95_seconds'])

    def test_copied_prefix_callback_is_rejected(self):
        telemetry = OutputTelemetry(('alpha',), 2, native_completion=True,
                                    origin=100, clock=lambda: 101, restored_prefix_frames=1)
        telemetry.render_ready((0, 1))
        telemetry.written('alpha', 0, 102)
        report = telemetry.finish()
        self.assertIsNotNone(report['failure'])
        self.assertEqual(report['all_native_completed_frames'], 0)
        self.assertIsNone(report['native_write_latency']['p95_seconds'])

    def test_invalid_restored_coverage(self):
        for count, native in ((True, True), (-1, True), (4, True), (1, False)):
            with self.subTest(count=count, native=native), self.assertRaises(ValueError):
                OutputTelemetry(('alpha',), 3, native_completion=native, origin=100,
                                restored_prefix_frames=count)
