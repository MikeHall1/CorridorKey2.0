"""Real Qt/editor-owner tracking flow with local tiny image sequences."""
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

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest


@unittest.skipUnless(importlib.util.find_spec('PySide6') is not None, 'Optional Qt binding unavailable')
class DesktopTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['tracking-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_corrections import CorrectionDialog
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(4):
            cv2.imwrite(str(self.source / f'{frame}.png'), np.full((24, 32, 3), 100, np.uint8))
        self.job = JobRequest.create(self.source, self.root / 'render', InferenceConfig(output_format='exr'))
        self.dialog = CorrectionDialog(self.job, self.root)
        self.dialog.show()
        self.until(lambda: self.dialog.displayed_frame == 0 and self.dialog.pending is None)
        self.dialog.request('brush', mask=np.full((24, 32), 255, np.uint8), kind='keep', frame=0,
                            strength=.5, direction='both', span=[0, 1])
        self.until(lambda: len(self.dialog.entries) == 1 and self.dialog.pending is None)

    def tearDown(self):
        self.dialog.canvas.clear_draft()
        self.dialog.close()
        self.dialog.service.thread.join(10)
        self.dialog.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.dialog)
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

    def test_preview_inspect_readonly_failure_discard_and_explicit_new_mask(self):
        self.dialog.add_tracking_anchor()
        self.dialog.start.setValue(0)
        self.dialog.end.setValue(4)
        self.dialog.direction.setCurrentText('forward')
        self.dialog.preview_tracking()
        self.until(lambda: self.dialog.pending is None)
        assert self.dialog.snapshot is not None
        self.assertTrue(self.dialog.snapshot['tracking_preview']['complete'])
        self.assertNotIn('tracks', self.dialog.snapshot['document'])
        self.dialog.frame.setValue(2)
        self.dialog.inspect_tracking()
        self.until(lambda: self.dialog.pending is None and self.dialog.displayed_role == 'tracking')
        self.assertFalse(self.dialog.canvas.painting)
        self.dialog.select()
        self.assertFalse(self.dialog.canvas.painting)
        with patch('corridorkey.correction_tracking_editor.accept_tracking', side_effect=ValueError('fixture failed accept')):
            self.dialog.request('track_accept')
            self.until(lambda: self.dialog.pending is None)
        self.assertIn('fixture failed accept', self.dialog.status.text())
        self.assertFalse(self.dialog.canvas.painting)
        self.dialog.request('track_discard')
        self.until(lambda: self.dialog.pending is None)
        self.assertFalse(self.dialog.canvas.painting)
        self.dialog.commit()
        self.assertIn('read-only', self.dialog.status.text())
        self.dialog.new_mask()
        self.assertTrue(self.dialog.canvas.painting)

    def test_cancellation_retires_operation_and_owner_accepts_next_request(self):
        entered = threading.Event()

        def blocked(*args, control, **kwargs):
            entered.set()
            while True:
                control.check()
                time.sleep(.002)

        self.dialog.end.setValue(4)
        self.dialog.direction.setCurrentText('forward')
        with patch('corridorkey.correction_tracking_editor.track_corrections', side_effect=blocked):
            self.dialog.preview_tracking()
            self.until(entered.is_set)
            self.dialog.cancel_operation()
            self.until(lambda: self.dialog.pending is None)
        self.assertIn('cancelled', self.dialog.status.text())
        self.assertFalse(list(self.root.glob('corrections-*/tracking-preview-*')))
        self.dialog.load_frame()
        self.until(lambda: self.dialog.pending is None)
        self.assertFalse(self.dialog.service.done.is_set())
        self.assertEqual(self.dialog.displayed_role, 'authored')

    def test_failed_operation_keeps_original_error_and_shows_cleanup_ownership(self):
        self.dialog.end.setValue(4)
        self.dialog.direction.setCurrentText('forward')
        with patch('corridorkey.correction_tracking_editor.track_corrections', side_effect=ValueError('fixture solver failure')), \
                patch('corridorkey.correction_tracking_editor.shutil.rmtree', side_effect=OSError('fixture private cleanup failure')):
            self.dialog.preview_tracking()
            self.until(lambda: self.dialog.pending is None)
        assert self.dialog.snapshot is not None
        self.assertEqual(len(self.dialog.snapshot['tracking_cleanup_pending']), 1)
        self.assertIn('fixture solver failure', self.dialog.status.text())
        self.assertIn('cleanup pending', self.dialog.status.text())
        self.assertIn('Private preview cleanup pending', self.dialog.timeline.text())
        self.assertIn('fixture private cleanup failure', self.dialog.timeline.toolTip())
        self.dialog.preview_tracking()
        self.until(lambda: self.dialog.pending is None)
        self.assertFalse(self.dialog.snapshot['tracking_cleanup_pending'])


if __name__ == '__main__':
    unittest.main()
