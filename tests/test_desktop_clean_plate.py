"""Local real-raster Qt clean-plate previews remain explicit and read-only."""
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
class DesktopCleanPlateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['plate-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_corrections import CorrectionDialog
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        plate = np.full((48, 64, 3), (40, 150, 55), np.uint8)
        image = plate.copy()
        image[16:32, 24:40] = (40, 40, 210)
        for frame in range(4):
            cv2.imwrite(str(self.source / f'{frame}.png'), image)
        self.reference = self.root / 'plate.png'
        cv2.imwrite(str(self.reference), plate)
        self.samples = self.root / 'samples.png'
        samples = np.full((48, 64), 255, np.uint8)
        samples[12:36, 20:44] = 0
        cv2.imwrite(str(self.samples), samples)
        self.before = {p: p.read_bytes() for p in self.source.iterdir()}
        self.before[self.reference] = self.reference.read_bytes()
        self.before[self.samples] = self.samples.read_bytes()
        self.job = JobRequest.create(self.source, self.root / 'render', InferenceConfig(output_format='exr', start_frame=1))
        self.dialog = CorrectionDialog(self.job, self.root)
        self.dialog.show()
        self.dialog.plate_toggle.setChecked(True)
        self.until(lambda: self.dialog.displayed_frame == 0 and self.dialog.pending is None)
        self.dialog.plate_reference_path.setText(str(self.reference))
        self.dialog.plate_samples_path.setText(str(self.samples))
        self.dialog.plate_profile.setCurrentIndex(self.dialog.plate_profile.findData('srgb'))

    def tearDown(self):
        self.dialog.canvas.clear_draft()
        self.dialog.close()
        self.dialog.service.thread.join(10)
        self.dialog.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.dialog)
        self.assertEqual({p: p.read_bytes() for p in self.before}, self.before)
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

    def preview(self):
        self.dialog.preview_plate()
        self.until(lambda: self.dialog.pending is None)
        assert self.dialog.snapshot is not None
        self.assertIsNotNone(self.dialog.snapshot.get('plate_preview'), self.dialog.status.text())

    def inspect(self, mode):
        self.dialog.plate_mode.setCurrentIndex(self.dialog.plate_mode.findData(mode))
        self.dialog.inspect_plate()
        self.until(lambda: self.dialog.pending is None)

    def test_explicit_preview_guidance_inspection_acceptance_and_immutable_inputs(self):
        self.preview()
        assert self.dialog.snapshot is not None
        preview = self.dialog.snapshot['plate_preview']
        self.assertEqual((preview['frame'], preview['source_frame'], preview['reference_frame']), (0, 1, 0))
        self.assertTrue(preview['usable'])
        self.assertFalse(preview['inspected'])
        self.assertFalse(self.dialog.entries)
        self.assertFalse(self.dialog.plate_accept.isEnabled())
        self.assertFalse(self.dialog.plate_thumbnail.pixmap().isNull())
        self.inspect('residual')
        self.assertFalse(self.dialog.snapshot['plate_preview']['inspected'])
        self.assertFalse(self.dialog.canvas.painting)
        self.inspect('guidance')
        self.assertEqual(self.dialog.canvas.overlay_kind, 'trimap')
        self.assertTrue(self.dialog.plate_accept.isEnabled())
        self.dialog.select()
        self.assertFalse(self.dialog.canvas.painting)
        self.dialog.commit()
        self.assertIn('read-only', self.dialog.status.text())
        self.dialog.plate_accept.click()
        self.until(lambda: self.dialog.pending is None and len(self.dialog.entries) == 1)
        self.assertEqual(self.dialog.entries[0]['kind'], 'trimap')
        self.assertIn('guidance_id', self.dialog.entries[0])
        self.assertTrue(self.dialog.detach_guidance_button.isEnabled())
        self.dialog.detach_guidance_button.click()
        self.until(lambda: self.dialog.pending is None)
        self.assertNotIn('guidance_id', self.dialog.entries[0])
        self.assertFalse(self.dialog.detach_guidance_button.isEnabled())
        self.assertFalse(self.dialog.canvas.painting)
        self.dialog.new_mask()
        self.assertTrue(self.dialog.canvas.painting)

    def test_defaults_dirty_guard_and_captured_settings_are_deliberate(self):
        self.assertFalse(self.dialog.plate_whole_frame.isChecked())
        self.dialog.plate_samples_path.clear()
        self.dialog.preview_plate()
        self.assertIn('known-background samples', self.dialog.status.text())
        self.dialog.plate_samples_path.setText(str(self.samples))
        self.dialog.canvas.dirty = True
        self.dialog.preview_plate()
        self.assertIn('Commit or Clear draft', self.dialog.status.text())
        self.assertIsNone(self.dialog.pending)
        self.dialog.canvas.clear_draft()
        self.preview()
        self.dialog.plate_bg.setValue(.03)
        self.inspect('guidance')
        self.assertIn('BG 0.04', self.dialog.plate_note.text())
        self.assertIn('current controls prepare a new preview', self.dialog.plate_note.text())
        self.dialog.request('plate_discard')
        self.until(lambda: self.dialog.pending is None)
        self.assertFalse(self.dialog.plate_accept.isEnabled())
        self.assertFalse(self.dialog.canvas.painting)

    def test_new_mask_reloads_source_after_every_plate_inspection_role(self):
        self.preview()
        for mode in ('reference', 'screen_field', 'residual', 'guidance'):
            with self.subTest(mode=mode):
                self.inspect(mode)
                self.assertFalse(self.dialog.canvas.painting)
                self.dialog.new_mask()
                self.assertFalse(self.dialog.canvas.painting)
                self.until(lambda: self.dialog.pending is None)
                self.assertTrue(self.dialog.canvas.painting)
                self.assertEqual(self.dialog.displayed_role, 'authored')
                self.assertEqual(self.dialog.displayed_frame, 0)
                assert self.dialog.canvas.source is not None
                self.assertEqual(self.dialog.canvas.source.toImage().pixelColor(32, 24).getRgb(), (210, 40, 40, 255))

    def test_unusable_alignment_and_failed_accept_preserve_readonly_preview(self):
        self.dialog.plate_reference_frame.setValue(99999)
        self.dialog.preview_plate()
        self.until(lambda: self.dialog.pending is None)
        self.assertIn('Error', self.dialog.status.text())
        self.assertFalse(self.dialog.plate_accept.isEnabled())
        self.dialog.plate_reference_frame.setValue(0)
        self.preview()
        self.inspect('guidance')
        with patch('corridorkey.clean_plate_editor.accept_plate', side_effect=ValueError('fixture failed acceptance')):
            self.dialog.request('plate_accept')
            self.until(lambda: self.dialog.pending is None)
        self.assertIn('fixture failed acceptance', self.dialog.status.text())
        self.assertFalse(self.dialog.canvas.painting)
        self.assertFalse(self.dialog.entries)

    def test_explicit_linear_exr_sequence_index_and_native_views(self):
        from corridorkey.color_utils import srgb_to_linear
        sequence = self.root / 'reference-sequence'
        sequence.mkdir()
        encoded_bgr = cv2.imread(str(self.reference), cv2.IMREAD_COLOR).astype(np.float32) / 255
        linear_bgr = srgb_to_linear(encoded_bgr).astype(np.float32)
        cv2.imwrite(str(sequence / 'plate2.exr'), np.full_like(linear_bgr, .01))
        cv2.imwrite(str(sequence / 'plate10.exr'), linear_bgr)
        for path in sequence.iterdir():
            self.before[path] = path.read_bytes()
        self.dialog.plate_reference_path.setText(str(sequence))
        self.dialog.plate_reference_frame.setValue(1)
        self.dialog.plate_profile.setCurrentIndex(self.dialog.plate_profile.findData('linear_bt709_d65'))
        self.preview()
        assert self.dialog.snapshot is not None
        captured = self.dialog.snapshot['plate_preview']
        self.assertEqual(captured['reference_frame'], 1)
        self.assertTrue(captured['usable'])
        self.assertEqual(captured['request']['mapping']['mode'], 'sequence')
        for mode in ('screen_field', 'reference', 'residual'):
            self.inspect(mode)
            self.assertFalse(self.dialog.snapshot['plate_preview']['inspected'])
            self.assertIsNone(self.dialog.canvas.saved_overlay)
            self.assertFalse(self.dialog.canvas.painting)
        self.inspect('guidance')
        self.assertTrue(self.dialog.snapshot['plate_preview']['inspected'])

    def test_poor_alignment_candidate_explains_refusal_without_enabling_acceptance(self):
        self.dialog.plate_alignment.setCurrentIndex(self.dialog.plate_alignment.findData('estimate_translation'))
        self.assertFalse(self.dialog.plate_dx.isEnabled())
        self.preview()
        assert self.dialog.snapshot is not None
        self.assertFalse(self.dialog.snapshot['plate_preview']['usable'])
        self.assertIn('Insufficient texture', self.dialog.plate_note.text())
        self.inspect('guidance')
        self.assertFalse(self.dialog.plate_accept.isEnabled())
        self.assertFalse(self.dialog.canvas.painting)
        self.dialog.request('plate_accept')
        self.until(lambda: self.dialog.pending is None)
        self.assertFalse(self.dialog.entries)

    def test_reference_colour_must_be_deliberately_declared(self):
        self.dialog.plate_profile.setCurrentIndex(-1)
        self.dialog.preview_plate()
        self.assertIn('Declare reference colour explicitly', self.dialog.status.text())
        self.assertIsNone(self.dialog.pending)

    def test_cancelled_preview_releases_owned_service_without_authored_changes(self):
        entered = threading.Event()

        def blocked(*args, check, **kwargs):
            entered.set()
            while True:
                check()
                time.sleep(.002)

        with patch('corridorkey.clean_plate_editor.assist_clean_plate', side_effect=blocked):
            self.dialog.preview_plate()
            self.until(entered.is_set)
            self.dialog.cancel_operation()
            self.until(lambda: self.dialog.pending is None)
        self.assertIn('cancelled', self.dialog.status.text())
        self.assertFalse(self.dialog.entries)
        self.assertFalse(self.dialog.service.done.is_set())
        self.preview()
        self.assertTrue(self.dialog.plate_inspect.isEnabled())
