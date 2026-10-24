"""Real Qt/owned CPU-fixture previews; no pretrained quality/speed assertions."""
import hashlib
import importlib.util
import json
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import test_desktop_window as fixtures


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt is unavailable')
class ProgressiveDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DesktopWindowTests.setUpClass()

    def setUp(self):
        self.fixture = fixtures.DesktopWindowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window, self.root, self.app = self.fixture.window, self.fixture.root, self.fixture.app
        self.panel = self.window.progressive

    def until(self, predicate, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            self.panel.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail(f'Progressive condition not reached: {self.panel.status.text()}')

    def configure(self, *, gate=None):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest

        source = self.root / 'frames'
        source.mkdir()
        for index in range(4):
            self.assertTrue(cv2.imwrite(str(source / f'{index:03d}.png'), np.full((24, 32, 3), index * 30, np.uint8)))
        controls: dict[str, object] = {'export_fg': True}
        if gate is not None:
            controls['gate'] = str(gate)
        (source / '.worker-control.json').write_text(json.dumps(controls))
        config = InferenceConfig(output_format='exr', outputs={'alpha', 'fg'}, auto_vram=False,
                                 start_frame=0, num_frames=4, frame_dir_fps=24.)
        job = JobRequest.create(source, self.root / 'exports', config)
        self.window.populate_job(job, frame=1)
        scratch = self.root / 'preview-scratch'
        scratch.mkdir()
        self.panel.scratch.setText(str(scratch))
        self.panel.idle.setChecked(False)
        self.panel.draft.setChecked(True)
        return job, scratch, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.resolve().iterdir()}

    def stop(self):
        self.panel.enabled.setChecked(False)
        self.until(lambda: not self.panel.busy)

    def test_open_is_disabled_and_performs_no_io_or_model_work(self):
        with patch('corridorkey.desktop_progressive.prepare_preview') as prepare, patch.object(self.window.bridge, 'send') as send:
            from PySide6.QtWidgets import QPushButton
            button = next(button for button in self.window.findChildren(QPushButton) if button.text() == 'Progressive preview…')
            button.click()
            self.app.processEvents()
            self.assertTrue(self.panel.isVisible())
            self.panel.poll()
            self.assertFalse(self.panel.enabled.isChecked())
            self.assertFalse(self.panel.busy)
            self.assertIsNone(self.panel.recipe)
            prepare.assert_not_called()
            send.assert_not_called()
        self.assertFalse(self.window.jobs)

    def test_actual_draft_final_replacement_preserves_form_source_exports_and_queue(self):
        from corridorkey.worker_protocol import job_to_wire

        job, scratch, before = self.configure()
        original = job_to_wire(self.window.job())
        self.panel.enabled.setChecked(True)
        self.panel.begin()
        self.until(lambda: self.panel.scheduler is not None and self.panel.scheduler.snapshot(1).status == 'final')
        scheduler = self.panel.scheduler
        assert scheduler is not None
        state = scheduler.snapshot(1)
        self.assertIsNotNone(state.result)
        result = state.result
        assert result is not None
        self.assertEqual(result.request.source_frame, 1)
        self.assertEqual([stage.name for stage in scheduler.stages], ['draft', 'final'])
        self.assertEqual(json.loads(scheduler.stages[0].effective_settings_json)['inference_settings']['model_scale'], .5)
        self.assertEqual(json.loads(scheduler.stages[-1].effective_settings_json)['inference_settings']['model_scale'], 1.)
        self.assertEqual(set(result.artifact.payload), {128})
        artifact_metadata = json.loads(result.artifact.metadata_json)
        self.assertIn('full captured timeline', artifact_metadata['temporal_context'])
        self.assertIs(artifact_metadata['execution_observations']['neural_inference'], False)
        self.assertGreaterEqual(artifact_metadata['execution_observations']['fixture_loads'], 1)
        self.assertFalse(self.window.jobs)
        self.assertEqual(job_to_wire(self.window.job()), original)
        self.assertFalse((self.root / 'exports').exists())
        self.stop()
        self.assertEqual(list(scratch.iterdir()), [])
        self.assertEqual({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(job.input_path).iterdir()}, before)

    def test_rapid_seek_cancels_old_owned_request_and_only_latest_frame_finishes(self):
        gate = self.root / 'allow-fixture'
        self.configure(gate=gate)
        self.panel.enabled.setChecked(True)
        self.panel.begin()
        self.until(lambda: self.panel.worker_id is not None)
        old_request = self.panel.request
        assert old_request is not None
        self.window.frame.setValue(2)
        self.panel.poll()
        self.assertTrue(self.panel.cancel_sent)
        gate.write_text('allow')
        self.until(lambda: self.panel.scheduler is not None and self.panel.scheduler.snapshot(2).status == 'final')
        scheduler = self.panel.scheduler
        assert scheduler is not None
        snapshot = scheduler.snapshot(2)
        assert snapshot.result is not None
        self.assertEqual(snapshot.result.request.source_frame, 2)
        self.assertGreater(snapshot.stale_completions, 0)
        self.assertNotEqual(snapshot.result.request.identity, old_request.identity)
        self.stop()
        self.assertFalse(self.window.jobs)

    def test_settings_change_disables_without_replacing_captured_export_recipe(self):
        self.configure()
        self.panel.enabled.setChecked(True)
        self.panel.begin()
        self.until(lambda: self.panel.scheduler is not None and self.panel.scheduler.snapshot(1).status == 'final')
        self.panel._display()
        self.assertFalse(self.panel.image.pixmap().isNull())
        self.window.update_settings({'stage2_batch': 2})
        self.panel.poll()
        self.assertFalse(self.panel.enabled.isChecked())
        self.assertIn('Stale preview hidden', self.panel.image.text())
        self.assertEqual(self.panel.image.toolTip(), '')
        self.until(lambda: not self.panel.busy)
        self.assertEqual(self.window.job().configurations()[0].stage2_batch, 2)
        self.assertFalse(self.window.jobs)

    def test_cancelled_late_capture_cannot_restart_after_checkbox_reenabled(self):
        from corridorkey.preview_execution import prepare_preview

        entered, release = threading.Event(), threading.Event()
        self.configure()

        def late(*args, **kwargs):
            recipe = prepare_preview(*args, **kwargs)
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture release')
            return recipe  # Deliberately returns after its final cancellation fence.

        with patch('corridorkey.desktop_progressive.prepare_preview', new=late):
            self.panel.enabled.setChecked(True)
            self.panel.begin()
            self.assertTrue(entered.wait(3))
            self.panel.enabled.setChecked(False)
            self.panel.enabled.setChecked(True)
            release.set()
            self.until(lambda: not self.panel.io_busy)
            self.assertIsNone(self.panel.recipe)
            self.assertIsNone(self.panel.scheduler)
            self.assertIsNone(self.panel.worker_id)
        self.stop()
        self.assertFalse(self.window.jobs)

    def test_invalid_editor_during_late_capture_is_rejected_without_timer_exception(self):
        from corridorkey.preview_execution import prepare_preview

        entered, release = threading.Event(), threading.Event()
        self.configure()

        def late(*args, **kwargs):
            recipe = prepare_preview(*args, **kwargs)
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture release')
            return recipe

        with patch('corridorkey.desktop_progressive.prepare_preview', new=late):
            self.panel.enabled.setChecked(True)
            self.panel.begin()
            self.assertTrue(entered.wait(3))
            self.window.config.setPlainText('{invalid current settings')
            release.set()
            self.until(lambda: not self.panel.io_busy)
            self.assertIn('current settings are invalid', self.panel.status.text())
            self.assertFalse(self.panel.enabled.isChecked())
            self.assertIsNone(self.panel.recipe)
            self.assertIsNone(self.panel.worker_id)
            self.assertEqual(self.window.config.toPlainText(), '{invalid current settings')
        self.assertFalse(self.window.jobs)

    def test_cleanup_failure_keeps_owner_and_retry_releases_actual_private_root(self):
        from corridorkey.preview_execution import PreviewWorkspace

        _, scratch, _ = self.configure()
        actual_cleanup = PreviewWorkspace.cleanup
        failures = []

        def fail_once(workspace, *args, **kwargs):
            if not failures:
                failures.append(workspace.root)
                raise OSError('fixture denied cleanup')
            return actual_cleanup(workspace, *args, **kwargs)

        with patch.object(PreviewWorkspace, 'cleanup', new=fail_once):
            self.panel.enabled.setChecked(True)
            self.panel.begin()
            self.until(lambda: self.panel.failure is not None and 'cleanup NOT VERIFIED' in self.panel.failure)
            self.assertTrue(self.panel.busy)
            self.assertIsNotNone(self.panel.prepared)
            self.assertTrue(failures[0].exists())
            self.assertIn(str(failures[0]), self.panel.status.text())
            self.panel.retry_cleanup()
            self.until(lambda: not self.panel.busy)
        self.assertEqual(list(scratch.iterdir()), [])
        self.assertFalse(self.window.jobs)

    def test_parent_close_retains_capture_owner_until_cancel_boundary(self):
        entered, release = threading.Event(), threading.Event()
        self.configure()

        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture release')
            kwargs['check']()
            raise AssertionError('Cancelled capture cannot publish')

        with patch('corridorkey.desktop_progressive.prepare_preview', new=blocked):
            self.panel.enabled.setChecked(True)
            self.panel.begin()
            self.assertTrue(entered.wait(3))
            self.window.close()
            self.assertTrue(self.window.isVisible())
            self.assertTrue(self.panel.io_busy)
            token = self.panel.io_token
            assert token is not None
            self.assertTrue(token.cancelled)
            release.set()
            self.until(lambda: not self.window.isVisible() and not self.panel.busy)
        self.assertFalse(self.window.jobs)


if __name__ == '__main__':
    unittest.main()
