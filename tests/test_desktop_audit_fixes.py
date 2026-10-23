"""Audit boundary regressions with real offscreen Qt and local CPU fixtures."""
import importlib.util
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_desktop_progressive as progressive_fixtures


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt is unavailable')
class ProgressiveAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        progressive_fixtures.ProgressiveDesktopTests.setUpClass()

    def setUp(self):
        self.fixture = progressive_fixtures.ProgressiveDesktopTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.window, self.root, self.app = self.fixture.window, self.fixture.root, self.fixture.app
        self.panel = self.fixture.panel

    def configure(self, **kwargs):
        return self.fixture.configure(**kwargs)

    def until(self, predicate):
        self.fixture.until(predicate)

    def stop(self):
        self.fixture.stop()

    def test_escape_disables_and_cancels_owned_preview(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        gate = self.root / 'allow-fixture'
        self.configure(gate=gate)
        self.panel.enabled.setChecked(True)
        self.panel.begin()
        self.until(lambda: self.panel.worker_id is not None)
        self.panel.show()
        QTest.keyClick(self.panel, Qt.Key.Key_Escape)
        self.assertFalse(self.panel.enabled.isChecked())
        self.assertTrue(self.panel.cancel_sent)
        gate.write_text('allow')
        self.until(lambda: not self.panel.busy)

    def test_playback_focus_and_pause_follow_accepted_absolute_source_frame(self):
        self.configure()
        self.window.update_settings({'frame_dir_fps': 2.})
        self.panel.enabled.setChecked(True)
        self.panel.begin()
        self.until(lambda: self.panel.scheduler is not None)
        self.window.view.setCurrentText('Source')
        self.window.playback.play()
        self.until(lambda: self.window.playback.displayed_frame == 2)
        self.panel.poll()
        self.assertEqual(self.window.frame.value(), 1)
        self.assertEqual(self.panel.desired_frame, 2, self.panel.status.text())
        assert self.panel.scheduler is not None
        self.assertEqual(self.panel.scheduler._frame, 2)
        self.assertTrue(self.panel.scheduler._playing)
        self.window.playback.pause()
        self.panel.poll()
        self.assertEqual(self.panel.desired_frame, 2, self.panel.status.text())
        self.assertFalse(self.panel.scheduler._playing)
        self.stop()

    def test_play_state_change_retires_late_preparation_without_disabling(self):
        from corridorkey.preview_execution import PreviewRecipe

        self.configure()
        entered, release = threading.Event(), threading.Event()
        prepare = PreviewRecipe.job_for
        self.addCleanup(release.set)

        def late(recipe, *args, **kwargs):
            result = prepare(recipe, *args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture release')
            return result

        with patch.object(PreviewRecipe, 'job_for', new=late):
            self.panel.enabled.setChecked(True)
            self.panel.begin()
            self.until(entered.is_set)
            with patch.object(self.window.playback, 'progressive_focus', return_value=(1, True)):
                self.panel.poll()
                release.set()
                self.until(lambda: not self.panel.io_busy)
                self.assertIsNone(self.panel.failure, self.panel.status.text())
                self.assertTrue(self.panel.enabled.isChecked())
        self.stop()

    def test_trimmed_export_local_zero_maps_to_source_start(self):
        from corridorkey.desktop_comparison import RenderView

        job, _, _ = self.configure()
        self.window.update_settings({'start_frame': 2, 'num_frames': 2})
        self.window.frame.setValue(0)
        self.window.view.setCurrentText('Alpha')
        view = RenderView(job.input_path, 2, 2, 24, 32, 24., (), True, False, 'ignore')
        with patch.object(self.window, 'selected', return_value='completed'), patch.object(self.window, 'selected_render_view', return_value=view):
            self.panel.enabled.setChecked(True)
            self.panel.begin()
            self.until(lambda: self.panel.scheduler is not None or self.panel.failure is not None)
            self.assertIsNone(self.panel.failure)
            self.assertEqual(self.panel.desired_frame, 2, self.panel.status.text())
            assert self.panel.scheduler is not None
            self.assertEqual(self.panel.scheduler._frame, 2)
            self.stop()

    def test_initial_focus_outside_captured_range_is_visible_not_callback_exception(self):
        self.configure()
        self.window.update_settings({'start_frame': 2, 'num_frames': 2})
        self.window.frame.setValue(0)
        self.panel.enabled.setChecked(True)
        self.panel.begin()
        assert self.panel.io_thread is not None
        self.panel.io_thread.join(5)
        self.panel.poll()
        self.assertFalse(self.panel.enabled.isChecked())
        self.assertIsNone(self.panel.recipe)
        self.assertIn('range', self.panel.status.text())
        self.assertIsNone(self.panel.worker_id)

    def test_disabled_cancel_delivery_failure_is_visible_and_explicitly_retryable(self):
        bridge = SimpleNamespace(send=Mock(side_effect=[queue.Full('capacity'), 'cancel-request']), done=threading.Event())
        self.panel.enabled.setChecked(True)
        self.panel.worker_id = 'owned-preview'
        self.addCleanup(setattr, self.panel, 'worker_id', None)
        with patch.object(self.window, 'bridge', bridge):
            self.panel.enabled.setChecked(False)
            for _ in range(3):
                self.panel.poll()
                self.panel.invalidate('Repeated close or settings invalidation')
            self.assertEqual(bridge.send.call_count, 1)
            self.assertIn('cancel', self.panel.status.text().lower())
            self.assertIn('capacity', self.panel.status.text())
            self.assertEqual(self.panel.worker_id, 'owned-preview')
            self.assertFalse(self.panel.cancel_sent)
            self.panel.retry_cleanup()
            self.assertTrue(self.panel.cancel_sent)
            self.assertEqual(self.panel.worker_id, 'owned-preview')
            self.assertEqual(bridge.send.call_count, 2)
        self.panel.worker_id = None

    def test_failed_cancel_retry_remains_visible_and_owned(self):
        bridge = SimpleNamespace(send=Mock(side_effect=RuntimeError('worker unavailable')), done=threading.Event())
        self.panel.enabled.setChecked(True)
        self.panel.worker_id = 'owned-preview'
        self.addCleanup(setattr, self.panel, 'worker_id', None)
        with patch.object(self.window, 'bridge', bridge):
            self.panel.enabled.setChecked(False)
            self.panel.retry_cleanup()
            self.panel.poll()
            self.assertIn('worker unavailable', self.panel.status.text())
            self.assertIn('Ownership retained', self.panel.status.text())
            self.assertEqual(self.panel.worker_id, 'owned-preview')
            self.assertFalse(self.panel.cancel_sent)
            self.assertEqual(bridge.send.call_count, 2)

    def test_incompatible_export_source_is_refused(self):
        from corridorkey.desktop_comparison import RenderView

        self.configure()
        self.window.view.setCurrentText('Alpha')
        view = RenderView(str(self.root / 'other'), 0, 4, 24, 32, 24., (), True, False, 'ignore')
        with patch.object(self.window, 'selected', return_value='completed'), patch.object(self.window, 'selected_render_view', return_value=view):
            self.panel.enabled.setChecked(True)
            self.panel.begin()
            self.assertIn('source', self.panel.status.text().lower())
            self.assertFalse(self.panel.busy)
            self.assertIsNone(self.panel.recipe)


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt is unavailable')
class DialogEscapeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        progressive_fixtures.ProgressiveDesktopTests.setUpClass()

    def test_escape_capture_retains_render_owner_and_requests_cancel(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        from corridorkey.desktop_ofx_capture import OfxCaptureDialog
        from tests.qt_cleanup import dispose_widget

        dialog = OfxCaptureDialog('{}', '{}')
        self.addCleanup(dispose_widget, dialog)
        cancellations = []
        dialog.cancel_requested.connect(cancellations.append)
        dialog.mark_submitted('owned-capture')
        dialog.show()
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.assertTrue(dialog.closing)
        self.assertTrue(dialog.isVisible())
        self.assertTrue(dialog.rendering)
        self.assertEqual(cancellations, ['owned-capture'])
        dialog.observe({'kind': 'cancelled', 'request_id': 'owned-capture', 'payload': {}})
        dialog.poll()
        self.assertFalse(dialog.isVisible())

    def test_escape_capture_io_cancels_but_retains_thread_until_retirement(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        from corridorkey.desktop_ofx_capture import OfxCaptureDialog
        from tests.qt_cleanup import dispose_widget

        entered, release = threading.Event(), threading.Event()
        dialog = OfxCaptureDialog('{}', '{}')
        self.addCleanup(dispose_widget, dialog)

        def operation(control):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture release')
            control.check()
            return 'late metadata'

        dialog._start('Inspect', operation)
        owned = dialog.operation
        assert owned is not None
        try:
            self.assertTrue(entered.wait(2))
            dialog.show()
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
            self.assertTrue(dialog.isVisible())
            self.assertIs(dialog.operation, owned)
            self.assertTrue(owned.cancellation.cancelled)
        finally:
            release.set()
            owned.thread.join(5)
            dialog.poll()
        self.assertIsNone(dialog.operation)
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.details.toPlainText(), '')

    def test_escape_background_requests_existing_preparation_cancel(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        from corridorkey.desktop_background import BackgroundDialog
        from tests.qt_cleanup import dispose_widget

        dialog = BackgroundDialog()
        self.addCleanup(dispose_widget, dialog)
        cancellations = []
        dialog.cancel_requested.connect(lambda: cancellations.append(True))
        dialog.set_waiting(True, 'Waiting')
        dialog.show()
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.assertEqual(cancellations, [True])
        self.assertFalse(dialog.isVisible())

    def test_escape_dirty_correction_preserves_visible_draft(self):
        import test_desktop_corrections as correction_fixtures
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        fixture = correction_fixtures.DesktopCorrectionTests()
        fixture.setUpClass()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        editor = fixture.editor
        QTest.mouseClick(editor.canvas, Qt.MouseButton.LeftButton, pos=editor.canvas.rect().center())
        self.assertTrue(editor.canvas.dirty)
        QTest.keyClick(editor, Qt.Key.Key_Escape)
        self.assertTrue(editor.isVisible())
        self.assertFalse(editor.closing)
        self.assertTrue(editor.canvas.dirty)
        self.assertIn('Commit or Clear', editor.status.text())
