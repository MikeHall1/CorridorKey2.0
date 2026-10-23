"""Terminal phase context survives missing progress delivery and cleanup.

These are injected loader/runner fixtures, not actual neural inference.
"""
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.control import CancellationToken, CancelledError, JobControl
from corridorkey.session import InferenceSession, JobRequest


class FailureCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.png'
        self.source.write_bytes(b'local fixture; never decoded')
        self.job = JobRequest.create(self.source, self.root / 'outputs', InferenceConfig(output_format='exr'))

    @staticmethod
    def last_record(session: InferenceSession):
        assert session.last_result is not None
        return session.last_result.to_dict()

    def test_checkpoint_is_retained_if_cancelled_before_progress_delivery(self):
        token = CancellationToken()
        progress = Mock()
        control = JobControl(token, progress)
        control.checkpoint('ready', 1, 1)
        earlier = control.last_checkpoint
        token.cancel()
        with self.assertRaises(CancelledError):
            control.checkpoint('stage2_frames', 2, 5)
        self.assertEqual(progress.call_count, 1)
        assert earlier is not None and control.last_checkpoint is not None
        self.assertEqual(earlier.stage, 'ready')
        self.assertEqual(control.last_checkpoint.stage, 'stage2_frames')
        attribute = 'stage'
        with self.assertRaises(FrozenInstanceError):
            setattr(control.last_checkpoint, attribute, 'rewritten')

    def test_loader_failure_records_phase_without_any_progress_consumer(self):
        def load():
            raise RuntimeError('fixture incompatible weights')

        with InferenceSession(loader=load, runner=Mock(), disposer=Mock()) as session:
            with self.assertRaisesRegex(RuntimeError, 'incompatible weights'):
                session.run(self.job)
            record = self.last_record(session)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['failure_checkpoint'], {'stage': 'model_load', 'completed': 0, 'total': 1})
        self.assertEqual(record['diagnostics']['last_checkpoint'], record['failure_checkpoint'])
        self.assertIsNone(record['end_to_end_fps'])
        self.assertEqual(self.source.read_bytes(), b'local fixture; never decoded')

    def test_encoder_preflight_failure_does_not_load_model(self):
        loader = Mock()
        job = JobRequest.create(self.source, self.root / 'outputs', writer=WriterConfig(ffmpeg_bin='missing-fixture'))
        with InferenceSession(loader=loader, runner=Mock(), disposer=Mock()) as session, \
                patch('shutil.which', return_value=None):
            with self.assertRaises(FileNotFoundError):
                session.run(job)
            record = self.last_record(session)
        loader.assert_not_called()
        self.assertEqual(record['failure_checkpoint']['stage'], 'encoder_preflight')
        self.assertFalse((self.root / 'outputs').exists())

    def test_reused_model_failure_reports_processing_not_old_model_load(self):
        count = 0

        def run(model, job, config, writer, diagnostics):
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError('fixture runner failed before its own checkpoint')

        disposer = Mock()
        with InferenceSession(loader=object, runner=run, disposer=disposer) as session:
            first = session.run(self.job).to_dict()
            with self.assertRaises(ValueError):
                session.run(self.job)
            record = self.last_record(session)
        self.assertTrue(record['model_reused'])
        self.assertEqual(record['failure_checkpoint'], {'stage': 'processing', 'completed': 0, 'total': None})
        self.assertNotIn('failure_checkpoint', first)
        disposer.assert_called_once()

    def test_failure_boundary_is_copied_before_cleanup_mutates_last_checkpoint(self):
        controls = []

        def run(model, job, config, writer, diagnostics):
            controls.append(diagnostics.control)
            diagnostics.control.checkpoint('stage2_frames', 2, 5)
            raise ValueError('fixture native operation error')

        def dispose(model):
            controls[0].checkpoint('cleanup', 1, 1)

        with InferenceSession(loader=object, runner=run, disposer=dispose) as session:
            with self.assertRaisesRegex(ValueError, 'native operation'):
                session.run(self.job)
            record = self.last_record(session)
        self.assertEqual(record['failure_checkpoint'], {'stage': 'stage2_frames', 'completed': 2, 'total': 5})
        self.assertEqual(record['diagnostics']['last_checkpoint']['stage'], 'cleanup')

    def test_precancelled_validation_and_postload_cancel_are_explicit(self):
        token = CancellationToken()
        token.cancel()
        with InferenceSession(loader=Mock(), runner=Mock(), disposer=Mock()) as session:
            with self.assertRaises(CancelledError):
                session.run(self.job, cancellation=token)
            record = self.last_record(session)
        self.assertEqual(record['failure_checkpoint']['stage'], 'validate')
        self.assertIsNone(record['requested_settings'])
        token = CancellationToken()
        runner, disposer = Mock(), Mock()

        def progress(event):
            if event.stage == 'processing':
                token.cancel('fixture stop after load')

        with InferenceSession(loader=object, runner=runner, disposer=disposer) as session:
            with self.assertRaises(CancelledError):
                session.run(self.job, cancellation=token, progress=progress)
            record = self.last_record(session)
        self.assertEqual(record['status'], 'cancelled')
        self.assertEqual(record['failure_checkpoint']['stage'], 'processing')
        runner.assert_not_called()
        disposer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
