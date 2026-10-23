"""Qt subject clicks with only the process-class boundary substituted.

The real editor/service, native PNG source, guidance session/provenance and
ordinary Keep import run; the provider is an explicit test fixture, never SAM2.
"""
import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from corridorkey.config import InferenceConfig
from corridorkey.control import JobControl
from corridorkey.session import JobRequest
from corridorkey.subject_guidance import (
    ExclusiveResourceLease,
    GuidanceSession,
    GuidanceUnavailable,
)
from tests import test_subject_guidance


class _FixtureProcess:
    instances: ClassVar[list["_FixtureProcess"]] = []

    def __init__(self, config, checkpoint, *, workspace):
        self.config, self.checkpoint, self.workspace = config, checkpoint, workspace
        self.provider = test_subject_guidance.FixtureProvider()
        self.closed = False
        self.frame = None
        self.clears = 0
        self.session = GuidanceSession(self.provider, ExclusiveResourceLease('cpu', 'qt-fixture', lambda: not self.closed))
        self.instances.append(self)

    def predict(self, frame, clicks, *, check):
        if self.frame is None or frame.identity != self.frame.identity:
            self.session.bind(frame)
            self.frame = frame
        return self.session.predict(self.session.request(clicks), control=JobControl(check_hook=check))

    def clear(self):
        self.clears += 1
        self.session.clear()

    def close(self):
        self.session.close()
        self.closed = True


@unittest.skipUnless(importlib.util.find_spec('PySide6') is not None, 'Optional Qt binding unavailable')
class SubjectCanvasMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['subject-canvas-test', '-platform', 'offscreen'])

    def test_scaled_letterbox_clicks_are_native_and_separate_from_default_brush(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        from corridorkey.correction_canvas import CorrectionCanvas, image_rect
        from tests.qt_cleanup import dispose_widget
        canvas = CorrectionCanvas()
        canvas.resize(640, 240)
        pixels = np.full((48, 64, 3), 100, np.uint8)
        canvas.set_frame(pixels, 64, 48, None)
        canvas.show()
        self.app.processEvents()
        clicks = []
        canvas.native_clicked.connect(lambda x, y: clicks.append((x, y)))
        left, top, width, height = image_rect(canvas.width(), canvas.height(), 64, 48)
        position = QPoint(round(left + 16.5 * width / 64), round(top + 12.5 * height / 48))
        try:
            self.assertFalse(canvas.click_mode)
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, position)
            self.assertTrue(canvas.dirty)
            self.assertEqual(clicks, [])
            canvas.clear_draft()
            canvas.click_mode = True
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(1, 120))
            self.assertEqual(clicks, [])
            QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, position)
            self.assertEqual(clicks, [(16, 12)])
            self.assertFalse(canvas.dirty)
            draft = canvas.draft
            assert draft is not None
            self.assertFalse(np.any(draft))
            np.testing.assert_array_equal(pixels, 100)
        finally:
            dispose_widget(canvas)


@unittest.skipUnless(importlib.util.find_spec('PySide6') is not None, 'Optional Qt binding unavailable')
class DesktopSubjectGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['subject-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_corrections import CorrectionDialog
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        image = np.full((48, 64, 3), (20, 150, 30), np.uint8)
        image[16:32, 24:40] = (30, 30, 210)
        for frame in range(4):
            cv2.imwrite(str(self.source / f'{frame}.png'), image)
        self.before = {path: path.read_bytes() for path in self.source.iterdir()}
        self.config, self.checkpoint = self.root / 'local.yaml', self.root / 'local.pt'
        self.config.write_text('test-only configuration; no SAM2 execution')
        self.checkpoint.write_bytes(b'test-only checkpoint; no model execution')
        _FixtureProcess.instances.clear()
        self.process_patch = patch('corridorkey.subject_guidance_editor.SubjectGuidanceProcess', _FixtureProcess)
        self.process_patch.start()
        self.job = JobRequest.create(self.source, self.root / 'render', InferenceConfig(output_format='exr', start_frame=1))
        self.dialog = CorrectionDialog(self.job, self.root)
        self.dialog.show()
        self.until(lambda: self.dialog.pending is None and self.dialog.displayed_frame == 0)
        self.dialog.subject_toggle.setChecked(True)
        self.dialog.subject_config.setText(str(self.config))
        self.dialog.subject_checkpoint.setText(str(self.checkpoint))

    def tearDown(self):
        for owner in _FixtureProcess.instances:
            owner.provider.release.set()
        self.dialog.canvas.clear_draft()
        self.dialog.close()
        self.dialog.service.thread.join(10)
        self.dialog.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.dialog)
        self.process_patch.stop()
        self.assertTrue(all(owner.closed for owner in _FixtureProcess.instances))
        self.assertEqual({path: path.read_bytes() for path in self.before}, self.before)
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

    def prepare(self):
        self.dialog.prepare_subject()
        self.until(lambda: self.dialog.pending is None)
        self.assertEqual(self.dialog.displayed_role, 'subject_source', self.dialog.status.text())

    def add(self, x, y, positive=True):
        self.dialog.subject_positive.setCurrentIndex(self.dialog.subject_positive.findData(positive))
        self.dialog.subject_x.setValue(x)
        self.dialog.subject_y.setValue(y)
        self.dialog.subject_add.click()

    def click_canvas(self, x, y, positive=True):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        from corridorkey.correction_canvas import image_rect
        self.dialog.subject_positive.setCurrentIndex(self.dialog.subject_positive.findData(positive))
        self.dialog.subject_click_mode.setChecked(True)
        canvas = self.dialog.canvas
        left, top, width, height = image_rect(canvas.width(), canvas.height(), canvas.native_width, canvas.native_height)
        position = QPoint(round(left + (x + .5) * width / canvas.native_width), round(top + (y + .5) * height / canvas.native_height))
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, position)

    def predict(self):
        self.dialog.predict_subject()
        self.until(lambda: self.dialog.pending is None)

    def inspect(self):
        self.dialog.subject_inspect.click()
        self.until(lambda: self.dialog.pending is None)

    def test_explicit_prediction_inspection_keep_acceptance_and_no_automatic_models(self):
        self.assertEqual(_FixtureProcess.instances, [])
        self.assertFalse(self.dialog.subject_click_mode.isChecked())
        self.prepare()
        owner = _FixtureProcess.instances[0]
        self.assertEqual(owner.provider.embeddings, 0)
        assert self.dialog.snapshot is not None
        preview = self.dialog.snapshot['subject_preview']
        self.assertEqual((preview['frame'], preview['source_frame']), (0, 1))
        self.assertFalse(self.dialog.canvas.painting)
        self.click_canvas(10, 8)
        self.add(20, 16, False)
        self.assertEqual(self.dialog.subject_clicks, [{'x': 10, 'y': 8, 'positive': True}, {'x': 20, 'y': 16, 'positive': False}])
        self.assertFalse(self.dialog.canvas.dirty)
        self.predict()
        self.assertEqual(owner.provider.embeddings, 1)
        self.assertFalse(self.dialog.entries)
        self.assertFalse(self.dialog.subject_accept.isEnabled())
        self.assertIsNone(self.dialog.canvas.saved_overlay)
        self.inspect()
        self.assertEqual(self.dialog.displayed_role, 'subject')
        self.assertTrue(self.dialog.subject_accept.isEnabled())
        overlay = self.dialog.canvas.saved_overlay
        assert overlay is not None
        self.assertEqual(overlay[8, 10], 255)
        self.assertIn('fixture', self.dialog.subject_note.text())
        self.dialog.select()
        self.assertFalse(self.dialog.canvas.painting)
        self.add(12, 10)
        self.assertFalse(self.dialog.subject_accept.isEnabled())
        self.assertFalse(self.dialog.subject_inspect.isEnabled())
        self.predict()
        self.assertIsNone(self.dialog.canvas.saved_overlay)
        self.inspect()
        self.dialog.subject_accept.click()
        self.until(lambda: self.dialog.pending is None and len(self.dialog.entries) == 1)
        self.assertEqual(self.dialog.entries[0]['kind'], 'keep')
        self.assertIn('guidance_id', self.dialog.entries[0])
        self.assertEqual(self.dialog.entries[0]['range'], [0, 1])
        self.assertTrue(owner.closed)

    def test_click_undo_clear_release_and_newmask_reload(self):
        self.prepare()
        self.add(10, 8)
        self.add(20, 16, False)
        self.dialog.subject_undo.click()
        self.assertEqual(self.dialog.subject_clicks, [{'x': 10, 'y': 8, 'positive': True}])
        self.predict()
        self.inspect()
        self.dialog.subject_clear.click()
        self.until(lambda: self.dialog.pending is None)
        self.assertFalse(self.dialog.subject_accept.isEnabled())
        self.assertIsNone(self.dialog.canvas.saved_overlay)
        self.assertEqual(self.dialog.subject_clicks, [])
        self.assertEqual(_FixtureProcess.instances[0].clears, 1)
        self.assertFalse(_FixtureProcess.instances[0].closed)
        self.add(12, 10)
        self.predict()
        self.inspect()
        self.dialog.subject_discard.click()
        self.until(lambda: self.dialog.pending is None)
        self.assertTrue(_FixtureProcess.instances[0].closed)
        self.assertFalse(self.dialog.subject_predict.isEnabled())
        self.assertFalse(self.dialog.canvas.painting)
        self.dialog.new_mask()
        self.assertFalse(self.dialog.canvas.painting)
        self.until(lambda: self.dialog.pending is None)
        self.assertEqual(self.dialog.displayed_role, 'authored')
        self.assertTrue(self.dialog.canvas.painting)

    def test_dirty_guard_actionable_dependency_error_and_cancelled_prediction(self):
        self.dialog.canvas.dirty = True
        self.dialog.prepare_subject()
        self.assertIn('Commit or Clear draft', self.dialog.status.text())
        self.assertEqual(_FixtureProcess.instances, [])
        self.dialog.canvas.clear_draft()
        self.prepare()
        self.add(10, 8)
        owner = _FixtureProcess.instances[0]
        with patch.object(owner, 'predict', side_effect=GuidanceUnavailable('Install optional SAM2 dependencies and select compatible local weights; no download performed.')):
            self.predict()
        self.assertIn('optional SAM2 dependencies', self.dialog.status.text())
        self.assertFalse(self.dialog.subject_accept.isEnabled())
        self.assertFalse(self.dialog.canvas.painting)
        owner.provider.block = True
        self.dialog.predict_subject()
        self.until(owner.provider.entered.is_set)
        self.dialog.cancel_operation()
        self.assertIsNotNone(self.dialog.pending)
        self.assertFalse(self.dialog.request('subject_predict', clicks=self.dialog.subject_clicks))
        self.assertFalse(self.dialog.canvas.click_mode)
        owner.provider.release.set()
        self.until(lambda: self.dialog.pending is None)
        self.assertIn('cancelled', self.dialog.status.text().lower())
        self.assertFalse(self.dialog.subject_accept.isEnabled())
        self.assertFalse(self.dialog.entries)
