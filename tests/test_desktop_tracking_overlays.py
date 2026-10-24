"""Actual editor/Qt spatial diagnostics; heuristics are not visual-quality proof."""
import importlib.util
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt binding unavailable')
class DesktopTrackingOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['overlay-test', '-platform', 'offscreen'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.originals = {}
        self.make_dialog()

    def make_dialog(self, *, failed_motion=False):
        from corridorkey.desktop_corrections import CorrectionDialog
        source = self.root / ('failed-motion' if failed_motion else 'source')
        source.mkdir()
        rng = np.random.default_rng(12)
        image = rng.integers(20, 180, (48, 64, 3), dtype=np.uint8)
        for index in range(5):
            pixels = np.full_like(image, 250) if failed_motion and index >= 2 else image
            self.assertTrue(cv2.imwrite(str(source / f'{index}.png'), pixels))
        self.originals.update({p: p.read_bytes() for p in source.iterdir()})
        job = JobRequest.create(source, self.root / 'exports', InferenceConfig(output_format='exr', start_frame=1, num_frames=3, frame_dir_fps=30.))
        self.dialog = CorrectionDialog(job, self.root)
        self.dialog.show()
        self.until(lambda: self.dialog.pending is None and self.dialog.displayed_frame == 0)
        mask = np.zeros((48, 64), np.uint8)
        mask[10:30, 20:40] = 255
        self.dialog.request('brush', mask=mask, kind='keep', frame=0, strength=.5, direction='both', span=[0, 1])
        self.until(lambda: self.dialog.pending is None and len(self.dialog.entries) == 1)

    def retire_dialog(self):
        self.dialog.canvas.clear_draft()
        self.dialog.close()
        self.dialog.service.thread.join(10)
        self.dialog.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.dialog)

    def tearDown(self):
        self.retire_dialog()
        self.assertEqual({p: p.read_bytes() for p in self.originals}, self.originals)
        self.assertFalse((self.root / 'exports').exists())
        self.temp.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.dialog.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail(self.dialog.status.text())

    def snapshot(self):
        snapshot = self.dialog.snapshot
        assert snapshot is not None
        return snapshot

    def capture(self, enabled=True):
        self.dialog.capture_tracking_diagnostics.setChecked(enabled)
        self.dialog.add_tracking_anchor()
        self.dialog.start.setValue(0)
        self.dialog.end.setValue(3)
        self.dialog.direction.setCurrentText('forward')
        self.dialog.preview_tracking()
        self.until(lambda: self.dialog.pending is None)
        assert self.dialog.snapshot is not None
        preview = self.snapshot()['tracking_preview']
        self.assertIsNotNone(preview, self.dialog.status.text())
        return preview

    def inspect(self, frame=1):
        self.assertTrue(self.dialog.inspect_tracking_diagnostic(frame))
        self.until(lambda: self.dialog.pending is None)

    def source_bytes(self):
        from PySide6.QtGui import QImage
        source = self.dialog.canvas.source
        assert source is not None
        image = source.toImage().convertToFormat(QImage.Format.Format_RGB888)
        return bytes(image.constBits())

    def test_default_no_maps_and_deliberate_capture_preserves_history_and_acceptance(self):
        self.assertFalse(self.dialog.capture_tracking_diagnostics.isChecked())
        preview = self.capture(False)
        self.assertFalse(preview['diagnostics']['captured'])
        before = self.snapshot()['identity']
        self.inspect()
        self.assertIn('diagnostic', self.dialog.status.text().lower())
        self.assertEqual(self.snapshot()['identity'], before)
        self.assertFalse(self.snapshot()['tracking_preview']['inspected'])
        self.capture(True)
        self.inspect()
        self.assertEqual(self.dialog.displayed_role, 'diagnostic')
        self.assertFalse(self.dialog.canvas.painting)
        self.assertFalse(self.dialog.canvas.click_mode)
        self.assertIsNone(self.dialog.canvas.saved_overlay)
        self.assertEqual(self.dialog.displayed_frame, 1)
        self.assertIn('source start 1', self.dialog.frame_note.text())
        self.assertIn('heuristic', self.dialog.frame_note.text())
        self.assertFalse(self.snapshot()['tracking_preview']['inspected'])
        self.assertEqual(self.snapshot()['identity'], before)
        self.dialog.request('track_accept')
        self.until(lambda: self.dialog.pending is None)
        self.assertIn('inspected', self.dialog.status.text())
        self.assertNotIn('tracks', self.snapshot()['document'])

    def test_dirty_draft_survives_navigation_mode_hide_and_correction_requests(self):
        self.capture()
        self.dialog.canvas._stroke((12, 15))
        draft = self.dialog.canvas.draft
        assert draft is not None
        before = draft.copy()
        frame, start, end = self.dialog.frame.value(), self.dialog.start.value(), self.dialog.end.value()
        self.assertFalse(self.dialog.inspect_tracking_diagnostic(2))
        self.dialog.tracking_diagnostic_events.setCurrentIndex(2)
        self.dialog.navigate_tracking_diagnostic()
        self.dialog.load_frame()
        self.dialog.correct_tracking_diagnostic()
        self.dialog.use_tracking_diagnostic_range()
        self.assertIsNone(self.dialog.pending)
        self.assertTrue(self.dialog.canvas.dirty)
        draft = self.dialog.canvas.draft
        assert draft is not None
        np.testing.assert_array_equal(draft, before)
        self.assertEqual((self.dialog.frame.value(), self.dialog.start.value(), self.dialog.end.value()), (frame, start, end))

    def test_new_correction_reloads_real_source_and_invalidates_maps_only_after_commit(self):
        self.capture()
        self.inspect(1)
        self.dialog.start.setValue(1)
        self.dialog.end.setValue(2)
        self.dialog.use_tracking_diagnostic_range()
        self.assertEqual((self.dialog.start.value(), self.dialog.end.value()), (0, 3))
        self.dialog.correct_tracking_diagnostic()
        self.assertFalse(self.dialog.canvas.painting)
        self.until(lambda: self.dialog.pending is None and self.dialog.displayed_role == 'authored')
        self.assertEqual(self.dialog.displayed_frame, 1)
        self.assertTrue(self.dialog.canvas.painting)
        self.assertEqual((self.dialog.start.value(), self.dialog.end.value()), (1, 2))
        self.dialog.canvas._stroke((25, 20))
        self.dialog.commit()
        self.until(lambda: self.dialog.pending is None and len(self.dialog.entries) == 2)
        self.assertIsNone(self.snapshot()['tracking_preview'])
        self.assertEqual(self.dialog.tracking_diagnostic_events.count(), 0)
        self.inspect(1)
        self.assertNotEqual(self.dialog.displayed_role, 'diagnostic')
        self.assertIn('preview', self.dialog.status.text().lower())

    def test_stopped_frame_has_actual_maps_without_a_mask_and_opacity_affects_display_only(self):
        self.retire_dialog()
        self.make_dialog(failed_motion=True)
        preview = self.capture()
        self.assertFalse(preview['complete'])
        event = next(event for event in preview['diagnostics']['frame_events'] if event['frame'] == 1)
        self.assertTrue(event['available'])
        self.assertEqual(event['source_frame'], 2)
        self.assertEqual(event['status'], 'anchor_required')
        self.assertNotIn(1, [row['frame'] for row in preview['frames'] if row['status'] in ('preview', 'approximation')])
        self.dialog.frame.setValue(1)
        self.dialog.load_frame()
        self.until(lambda: self.dialog.pending is None)
        real_source = self.source_bytes()
        self.dialog.tracking_diagnostic_kind.setCurrentIndex(self.dialog.tracking_diagnostic_kind.findData('uncertainty'))
        self.dialog.tracking_diagnostic_opacity.setValue(0)
        self.inspect()
        self.assertEqual(self.source_bytes(), real_source)
        self.dialog.tracking_diagnostic_opacity.setValue(.75)
        self.inspect()
        self.assertNotEqual(self.source_bytes(), real_source)
        self.assertFalse(self.snapshot()['tracking_preview']['inspected'])
        self.assertIn('not calibrated', self.dialog.frame_note.text())
        self.assertFalse(self.dialog.canvas.painting)

    def test_cancellation_after_completed_view_before_delivery_drops_late_pixels(self):
        from corridorkey.correction_editor import CorrectionEditor
        self.capture()
        entered, release = threading.Event(), threading.Event()
        original = CorrectionEditor.tracking_overlay_view
        old_pixels = self.source_bytes()

        def late(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture release')
            return result

        with patch.object(CorrectionEditor, 'tracking_overlay_view', new=late):
            self.dialog.inspect_tracking_diagnostic(1)
            try:
                self.assertTrue(entered.wait(3))
                self.dialog.cancel_operation()
            finally:
                release.set()
            self.until(lambda: self.dialog.pending is None)
        self.assertEqual(self.dialog.displayed_role, 'authored')
        self.assertEqual(self.source_bytes(), old_pixels)
        self.assertIn('cancelled', self.dialog.status.text().lower())
        self.assertFalse(self.snapshot()['tracking_preview']['inspected'])

    def test_existing_selector_displays_actual_two_anchor_disagreement(self):
        mask = np.zeros((48, 64), np.uint8)
        mask[10:30, 22:42] = 255
        self.dialog.request('brush', mask=mask, kind='keep', frame=2, strength=.5, direction='both', span=[0, 3])
        self.until(lambda: self.dialog.pending is None and len(self.dialog.entries) == 2)
        self.dialog.tracking_anchors = [entry['id'] for entry in self.dialog.entries]
        self.dialog.capture_tracking_diagnostics.setChecked(True)
        self.dialog.start.setValue(0)
        self.dialog.end.setValue(3)
        self.dialog.direction.setCurrentText('both')
        self.dialog.preview_tracking()
        self.until(lambda: self.dialog.pending is None)
        self.assertIn('disagreement', self.snapshot()['tracking_preview']['diagnostics']['kinds'])
        index = self.dialog.tracking_diagnostic_kind.findData('disagreement')
        self.assertGreaterEqual(index, 0)
        self.dialog.tracking_diagnostic_kind.setCurrentIndex(index)
        self.inspect(1)
        self.assertEqual(self.dialog.displayed_role, 'diagnostic')
        self.assertIn('Warped-anchor', self.dialog.tracking_diagnostic_kind.currentText())
        self.assertIn('Warped-anchor coverage difference', self.dialog.frame_note.text())
        self.assertFalse(self.snapshot()['tracking_preview']['inspected'])


if __name__ == '__main__':
    unittest.main()
