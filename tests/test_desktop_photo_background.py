"""Actual Qt/PNG/EXR photo previews; completed records are explicit CPU fixtures."""
import copy
import importlib.util
import threading
import unittest
from unittest.mock import patch

import test_desktop_playback as playback_fixtures


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt binding unavailable')
class DesktopPhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        playback_fixtures.DesktopPlaybackTests.setUpClass()

    def setUp(self):
        self.fixture = playback_fixtures.DesktopPlaybackTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window, self.app = self.fixture.window, self.fixture.app
        self.root = self.fixture.root.resolve()
        self.until = self.fixture.until
        self.photo = self.root / 'photo.png'
        self.write_photo()
        self.originals = {path: path.read_bytes() for folder in (self.fixture.source, self.root / 'a', self.root / 'b')
                          for path in folder.rglob('*') if path.is_file()}
        self.addCleanup(self.assert_originals)

    def assert_originals(self):
        for path, raw in self.originals.items():
            self.assertEqual(path.read_bytes(), raw)

    def write_photo(self, value=(51, 102, 153)):
        import cv2
        import numpy as np
        self.assertTrue(cv2.imwrite(str(self.photo), np.full((2, 3, 3), value, np.uint8)))

    def panel(self):
        self.window.open_background()
        panel = self.window.background_dialog
        assert panel is not None
        return panel

    def prepare(self):
        panel = self.panel()
        panel.path.setText(str(self.photo))
        panel.prepare.click()
        self.until(lambda: self.window.background_image is not None and self.window.pixmap is not None)
        self.assertFalse(panel.waiting)
        return panel

    def rgb(self, *, reference=False):
        pixmap = self.window.reference if reference else self.window.pixmap
        assert pixmap is not None
        colour = pixmap.toImage().pixelColor(0, 0)
        return colour.red(), colour.green(), colour.blue()

    def expected(self, alpha):
        import numpy as np

        from corridorkey.color_utils import linear_to_srgb, srgb_to_linear
        background = srgb_to_linear(np.array([153, 102, 51], np.float32) / 255)
        return tuple(np.rint(np.clip(linear_to_srgb(.25 * alpha + background * (1-alpha)), 0, 1)*255).astype(np.uint8))

    def test_open_no_io_then_actual_photo_composite_and_captured_snapshot_preserve_jobs(self):
        from corridorkey.worker_protocol import job_to_wire
        before, jobs, bridge = job_to_wire(self.window.job()), copy.deepcopy(self.window.jobs), self.window.bridge
        with patch('corridorkey.desktop_loader.prepare_background', side_effect=AssertionError('opening must not read photo')):
            panel = self.panel()
            self.assertIsNone(self.window.background_serial)
            self.assertIsNone(self.window.background_image)
            self.assertIn('no image I/O', panel.status.text())
        self.prepare()
        image = self.window.background_image
        assert image is not None
        self.assertEqual(self.window.view.currentText(), 'Photo composite')
        self.assertEqual(self.rgb(), self.expected(.7))
        self.assertIn(image.identity[:16], self.window.preview_description)
        self.assertIn('Re-prepare explicitly', panel.status.text())
        old_pixel = self.rgb()
        self.write_photo((0, 255, 0))
        self.window.request_preview()
        self.until(lambda: self.window.pixmap is not None)
        self.assertEqual(self.rgb(), old_pixel)
        self.assertIs(self.window.background_image, image)
        self.assertEqual(job_to_wire(self.window.job()), before)
        self.assertEqual(self.window.jobs, jobs)
        self.assertIs(self.window.bridge, bridge)

    def test_same_photo_source_aligned_pair_and_playback_keep_offsets(self):
        self.prepare()
        self.window.table.selectRow(0)
        self.window.synchronize_reference()
        self.window.table.selectRow(1)
        self.window.frame.setValue(3)
        self.window.request_preview()
        self.until(lambda: self.window.pixmap is not None and self.window.reference is not None)
        self.assertEqual(self.rgb(reference=True), self.expected(.3))
        self.assertEqual(self.rgb(), self.expected(.8))
        self.assertIn('export frame 2', self.window.reference_description)
        self.assertIn('decoded frame 1', self.window.preview_description)
        self.window.playback.play()
        self.until(lambda: self.window.playback.displayed_frame is not None)
        engine = self.window.playback.engine
        assert engine is not None
        self.assertIs(engine.background_image, self.window.background_image)
        self.window.playback.pause()
        self.window.pin_reference()
        photo = self.window.background_image
        assert photo is not None
        self.assertIn(photo.identity[:16], self.window.reference_note.text())

    def test_pin_pending_playback_keeps_metadata_of_visible_pixels(self):
        self.prepare()
        self.window.playback.play()
        self.until(lambda: self.window.playback.displayed_frame is not None)
        visible = self.window.playback.displayed_frame
        engine = self.window.playback.engine
        assert engine is not None and visible is not None
        target = visible + 1
        self.assertLess(target, engine.end)
        old_pixel = self.rgb()
        engine.seek(target)
        self.window.playback.tick()  # Submit, but do not process its terminal pixels.
        self.assertIsNotNone(self.window.playback.pending_serial)
        self.window.playback.pause()
        self.window.pin_reference()
        self.assertEqual(self.rgb(reference=True), old_pixel)
        self.assertIn(f'source frame {visible} ·', self.window.reference_note.text())
        self.assertNotIn(f'source frame {target} ·', self.window.reference_note.text())
        self.assertIn(f'source frame {visible} ·', self.window.reference_note.toolTip())

    def test_explicit_clear_uses_checkerboard_and_static_a_keeps_captured_photo_identity(self):
        from PySide6.QtWidgets import QPushButton
        panel = self.prepare()
        image = self.window.background_image
        assert image is not None
        self.window.pin_reference()
        reference = self.rgb(reference=True)
        next(button for button in panel.findChildren(QPushButton) if button.text() == 'Use checkerboard').click()
        self.until(lambda: self.window.pixmap is not None)
        self.assertIsNone(self.window.background_image)
        self.assertEqual(self.window.view.currentText(), 'Checker composite')
        self.assertEqual(self.rgb(reference=True), reference)
        self.assertIn(image.identity[:16], self.window.reference_note.text())

    def test_cancel_late_preparation_hides_stale_photo_and_never_publishes(self):
        from PySide6.QtWidgets import QPushButton

        from corridorkey.preview_background import prepare_background
        panel = self.prepare()
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            result = prepare_background(*args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture release')
            return result  # Deliberately ignore late cancellation; loader must fence it.
        with patch('corridorkey.desktop_loader.prepare_background', side_effect=blocked):
            panel.prepare.click()
            try:
                self.assertTrue(entered.wait(3))
                self.assertIsNone(self.window.pixmap)
                self.assertIsNone(self.window.background_image)
                next(button for button in panel.findChildren(QPushButton) if button.text() == 'Cancel preparation').click()
                self.assertFalse(panel.waiting)
                self.assertIsNone(self.window.background_serial)
            finally:
                release.set()
            terminal = self.window.loader.results.get(timeout=5)
            self.window.loader.results.put_nowait(terminal)
            self.window.poll()
        self.assertIsNone(self.window.background_image)
        self.assertIsNone(self.window.pixmap)

    def test_photo_terminal_retires_superseded_actual_playback_decode(self):
        from corridorkey.desktop_preview import preview
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture decode release')
            return preview(*args, **kwargs)
        with patch('corridorkey.desktop_loader.preview', side_effect=blocked):
            self.window.playback.play()
            try:
                self.assertTrue(entered.wait(3))
                original = self.window.playback.pending_serial
                self.assertIsNotNone(original)
                panel = self.panel()
                panel.path.setText(str(self.photo))
                panel.prepare.click()
                self.assertTrue(panel.waiting)
                self.assertEqual(self.window.playback.pending_serial, original)
            finally:
                release.set()
            self.until(lambda: self.window.background_image is not None and self.window.pixmap is not None)
        self.assertIsNone(self.window.playback.pending_serial)
        engine = self.window.playback.engine
        assert engine is not None
        self.assertIsNone(engine.pending)

    def test_close_waits_for_actual_existing_loader_preparation_owner(self):
        from corridorkey.preview_background import prepare_background
        panel = self.panel()
        panel.path.setText(str(self.photo))
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            result = prepare_background(*args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture close release')
            return result
        with patch('corridorkey.desktop_loader.prepare_background', side_effect=blocked):
            panel.prepare.click()
            try:
                self.assertTrue(entered.wait(3))
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(self.window.loader.thread.is_alive())
            finally:
                release.set()
            self.until(lambda: not self.window.isVisible())
        self.assertFalse(self.window.loader.thread.is_alive())
        self.assertIsNone(self.window.background_image)


if __name__ == '__main__':
    unittest.main()
