"""Actual native EXR hook ordering, bytes and error isolation."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HAS_STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'cv2'))


@unittest.skipUnless(HAS_STACK, 'requires native CPU EXR writer stack')
class WriterTelemetryTests(unittest.TestCase):
    def test_native_callback_after_each_success_and_default_no_clock_calls(self):
        import numpy as np

        from corridorkey.video_io import ExrSequenceWriter
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            default_clock = Mock(side_effect=AssertionError('default writer must not observe clock'))
            frames = np.full((3, 2, 2, 1), .4, np.float32)
            writer = ExrSequenceWriter(str(root/'default'), 24, (2, 2), channels=1,
                                       dtype='float', clock=default_clock)
            writer.write_many(frames)
            writer.release()
            default_clock.assert_not_called()
            received = []
            def observed(index, timestamp):
                self.assertTrue((root/'observed'/f'frame_{index:06d}.exr').is_file())
                received.append((index, timestamp))
            writer = ExrSequenceWriter(str(root/'observed'), 24, (2, 2), channels=1,
                                       dtype='float', clock=Mock(side_effect=[10., 20., 30.]), on_frame_written=observed)
            writer.write_many(frames)
            writer.release()
            self.assertEqual(received, [(0, 10.), (1, 20.), (2, 30.)])
            self.assertIsNone(writer.observation_error)
            for a, b in zip(sorted((root/'default').glob('*.exr')), sorted((root/'observed').glob('*.exr'))):
                self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertFalse(writer.worker.thread.is_alive())

    def test_clock_and_callback_failure_preserve_written_outputs(self):
        import numpy as np

        from corridorkey.video_io import ExrSequenceWriter
        for cause in ('clock', 'callback', 'nonfinite', 'interrupt'):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as folder:
                callback = (Mock(side_effect=KeyboardInterrupt('observer interrupted')) if cause == 'interrupt' else
                            Mock(side_effect=RuntimeError('observer callback failed')) if cause == 'callback' else Mock())
                clock = (Mock(side_effect=RuntimeError('clock failed')) if cause == 'clock' else
                         Mock(return_value=float('nan') if cause == 'nonfinite' else 10.))
                writer = ExrSequenceWriter(folder, 24, (2, 2), channels=1, dtype='float',
                                           on_frame_written=callback, clock=clock)
                writer.write_many(np.full((2, 2, 2, 1), .4, np.float32))
                writer.release()
                self.assertEqual(len(list(Path(folder).glob('*.exr'))), 2)
                self.assertIsNotNone(writer.observation_error)
                assert writer.observation_error is not None
                self.assertLessEqual(len(writer.observation_error), 512)
                self.assertFalse(writer.worker.thread.is_alive())
                self.assertEqual(clock.call_count, 1)

    def test_no_ack_for_failed_native_write_and_primary_error_survives(self):
        import numpy as np

        from corridorkey.video_io import ExrSequenceWriter
        with tempfile.TemporaryDirectory() as folder:
            callback = Mock(side_effect=RuntimeError('observer must not replace writer'))
            writer = ExrSequenceWriter(folder, 24, (2, 2), channels=1, on_frame_written=callback)
            with patch('corridorkey.video_io.cv2.imwrite', return_value=False):
                writer.write_many(np.zeros((1, 2, 2, 1), np.float32))
                with self.assertRaisesRegex(RuntimeError, 'cv2.imwrite failed'):
                    writer.release()
            callback.assert_not_called()
            self.assertIsNone(writer.observation_error)
            self.assertFalse(writer.worker.thread.is_alive())

    def test_primary_native_error_after_observer_failure_is_preserved(self):
        import numpy as np

        from corridorkey.video_io import ExrSequenceWriter, cv2
        with tempfile.TemporaryDirectory() as folder:
            original = cv2.imwrite
            calls = 0
            def fail_second(*args):
                nonlocal calls
                calls += 1
                return original(*args) if calls == 1 else False
            writer = ExrSequenceWriter(folder, 24, (2, 2), channels=1,
                on_frame_written=Mock(side_effect=RuntimeError('separate observer failure')))
            with patch('corridorkey.video_io.cv2.imwrite', side_effect=fail_second):
                writer.write_many(np.zeros((2, 2, 2, 1), np.float32))
                with self.assertRaisesRegex(RuntimeError, 'cv2.imwrite failed'):
                    writer.release()
            assert writer.observation_error is not None
            self.assertIn('separate observer failure', writer.observation_error)
            self.assertEqual(len(list(Path(folder).glob('*.exr'))), 1)
            self.assertFalse(writer.worker.thread.is_alive())

    def test_actual_writer_can_complete_before_queue_acceptance_event(self):
        import numpy as np

        from corridorkey.output_telemetry import OutputTelemetry
        from corridorkey.video_io import ExrSequenceWriter
        with tempfile.TemporaryDirectory() as folder:
            telemetry = OutputTelemetry(['alpha'], 1, native_completion=True, origin=0, clock=lambda: 1.)
            telemetry.render_ready([0])
            writer = ExrSequenceWriter(folder, 24, (2, 2), channels=1, dtype='float', clock=lambda: 2.,
                                       on_frame_written=lambda index, at: telemetry.written('alpha', index, at))
            writer.write_many(np.zeros((1, 2, 2, 1), np.float32))
            writer.release()
            telemetry.clock = lambda: 3.
            telemetry.submitted('alpha', [0])
            report = telemetry.finish()
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['native_write_latency']['p50_seconds'], 1.)
            self.assertEqual(report['queue_accept_latency']['p50_seconds'], 2.)
