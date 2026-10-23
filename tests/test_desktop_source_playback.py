"""Actual source-only PNG/EXR playback, without a render job or model."""
import importlib.util
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional requirements-desktop.txt is not installed')
class DesktopSourcePlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['source-playback-test', '-platform', 'offscreen'])

    def setUp(self):
        import cv2
        import numpy as np

        from corridorkey.desktop_window import DesktopWindow
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for index in range(7):
            self.assertTrue(cv2.imwrite(str(self.source / f'{index:03}.png'), np.full((2, 3, 3), 20 + index * 20, np.uint8)))
        self.original = {path: path.read_bytes() for path in self.source.iterdir()}
        self.window = DesktopWindow()
        self.window.show()
        self.window.timer.stop()
        self.controls = self.window.playback
        self.window.source.setText(str(self.source))
        self.assertTrue(self.window.update_settings({'start_frame': 2, 'num_frames': 3, 'frame_dir_fps': 10.}))
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.bridge.thread.join(5)
        self.window.loader.thread.join(5)
        self.window.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.window)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail('Source playback condition not reached: ' + self.controls.status.text())

    def clock(self):
        assert self.controls.engine is not None
        clock = [time.monotonic()]
        self.controls.engine.clock = lambda: clock[0]
        return clock

    def test_no_render_needed_and_exact_paths_avoid_per_frame_directory_scans(self):
        from corridorkey.preview_sequence import prepare_sequence
        self.assertFalse(self.window.jobs)
        self.assertEqual(self.window.source.text(), str(self.source))
        with patch('corridorkey.desktop_loader.prepare_sequence', wraps=prepare_sequence) as prepare, \
             patch('corridorkey.desktop_preview.frame_paths', side_effect=AssertionError('per-frame source rescan')):
            self.assertFalse(prepare.called)
            self.controls.play()
            self.until(lambda: self.controls.displayed_frame == 2)
            self.assertEqual(prepare.call_count, 1)
            assert self.window.pixmap is not None
            self.assertEqual(self.window.pixmap.toImage().pixelColor(0, 0).red(), 60)
            clock = self.clock()
            engine = self.controls.engine
            assert engine is not None
            engine.seek(2)
            clock[0] += .15
            self.until(lambda: self.controls.displayed_frame == 3)
            self.controls.pause()
            self.assertEqual(engine.frame, 3)
            prepared = self.controls.source_sequence
            self.controls.play()
            self.until(lambda: self.controls.pending_serial is None)
            self.assertIs(self.controls.source_sequence, prepared)
            self.assertEqual(self.controls.displayed_frame, 3)
            self.assertEqual(prepare.call_count, 1)
        self.assertFalse(self.window.jobs)
        self.assertEqual({path: path.read_bytes() for path in self.source.iterdir()}, self.original)
        self.assertIn('frozen path/stat', self.window.preview_note.text())

    def test_loop_offsets_and_settings_change_reprepare_without_touching_render_settings(self):
        self.controls.in_frame.setText('2')
        self.controls.out_frame.setText('4')
        self.controls.loop.setChecked(True)
        before = self.window.config.toPlainText()
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        clock = self.clock()
        engine = self.controls.engine
        assert engine is not None
        engine.seek(2)
        clock[0] += .15
        self.until(lambda: self.controls.displayed_frame == 3)
        clock[0] += .10
        self.until(lambda: self.controls.displayed_frame == 2)
        self.assertEqual(self.window.config.toPlainText(), before)
        self.assertTrue(self.window.update_settings({'frame_dir_fps': 12.}))
        self.assertFalse(engine.playing)
        self.assertIsNone(self.controls.source_sequence)
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        assert self.controls.engine is not None
        self.assertEqual(self.controls.engine.view.fps, 12.)
        assert self.controls.displayed_seconds is not None
        self.assertAlmostEqual(self.controls.displayed_seconds, 2 / 12)

    def test_late_source_preparation_error_cannot_stop_new_source_generation(self):
        import cv2
        import numpy as np

        from corridorkey.preview_sequence import prepare_sequence
        other = self.root / 'other'
        other.mkdir()
        for index in range(5):
            self.assertTrue(cv2.imwrite(str(other / f'{index:03}.png'), np.full((2, 3, 3), 190, np.uint8)))
        entered, release = threading.Event(), threading.Event()
        first = [True]

        def held_prepare(job, *, check):
            if first[0]:
                first[0] = False
                entered.set()
                release.wait(5)
                raise ValueError('obsolete native preparation failure')
            return prepare_sequence(job, check=check)

        with patch('corridorkey.desktop_loader.prepare_sequence', side_effect=held_prepare):
            try:
                self.controls.play()
                self.assertTrue(entered.wait(2))
                self.window.source.setText(str(other))
                self.controls.play()
            finally:
                release.set()
            self.until(lambda: self.controls.displayed_frame == 2)
        assert self.controls.engine is not None and self.window.pixmap is not None
        self.assertEqual(self.controls.engine.view.source, str(other.resolve()))
        self.assertEqual(self.window.pixmap.toImage().pixelColor(0, 0).red(), 190)
        self.assertTrue(self.controls.engine.playing)

    def test_pause_and_close_cancel_preparation_and_wait_for_owned_native_decode(self):
        from corridorkey.preview_sequence import prepare_sequence
        entered, release = threading.Event(), threading.Event()

        def held_prepare(job, *, check):
            entered.set()
            release.wait(5)
            return prepare_sequence(job, check=check)

        with patch('corridorkey.desktop_loader.prepare_sequence', side_effect=held_prepare):
            try:
                self.controls.play()
                self.assertTrue(entered.wait(2))
                self.controls.pause()
                self.assertFalse(self.controls.preparing)
                self.window.close()
                self.assertTrue(self.window.isVisible())
            finally:
                release.set()
            self.until(lambda: not self.window.isVisible())
        self.assertIsNone(self.controls.source_sequence)
        self.assertIsNone(self.controls.displayed_frame)
        self.assertEqual({path: path.read_bytes() for path in self.source.iterdir()}, self.original)

    def test_video_refuses_without_submitting_source_preparation(self):
        source = self.root / 'video.mp4'
        source.write_bytes(b'local fixture, no encoded media')
        self.window.source.setText(str(source))
        with patch.object(self.window.loader, 'submit') as submit:
            self.controls.play()
            self.assertFalse(submit.called)
        self.assertIn('video/VFR', self.controls.status.text())

    def test_linear_exr_and_source_primary_declaration_use_existing_display_conversion(self):
        import cv2
        import numpy as np

        from corridorkey.desktop_preview import preview
        source = self.root / 'exrs'
        source.mkdir()
        for index in range(4):
            pixels = np.broadcast_to(np.array([.05, .2, .7], np.float32), (2, 3, 3)).copy()
            self.assertTrue(cv2.imwrite(str(source / f'{index:03}.exr'), pixels))
        self.window.source.setText(str(source))
        self.assertTrue(self.window.update_settings({'frame_dir_linear': True, 'source_colour_profile': 'linear_bt2020_d65'}))
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        expected = preview(str(source), 2, linear=True, source_colour_profile='linear_bt2020_d65')
        assert self.window.pixmap is not None
        colour = self.window.pixmap.toImage().pixelColor(0, 0)
        self.assertEqual((colour.red(), colour.green(), colour.blue()), tuple(int(v) for v in expected.pixels[0, 0]))
        legacy = preview(str(source), 2, linear=True)
        self.assertFalse(np.array_equal(expected.pixels, legacy.pixels))
        self.assertIn('linear_bt2020_d65', self.window.preview_note.text())

    def test_explicit_refresh_reinspects_new_membership_without_changing_settings(self):
        import cv2
        import numpy as np
        self.assertTrue(self.window.update_settings({'num_frames': -1}))
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        prepared = self.controls.source_sequence
        assert prepared is not None
        self.controls.pause()
        self.assertTrue(cv2.imwrite(str(self.source / '007.png'), np.full((2, 3, 3), 210, np.uint8)))
        before = self.window.config.toPlainText()
        self.controls.refresh_source()
        self.assertIsNone(self.controls.source_sequence)
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        fresh = self.controls.source_sequence
        assert fresh is not None
        self.assertEqual(len(prepared.index.paths), 7)
        self.assertEqual(len(fresh.index.paths), 8)
        self.assertNotEqual(prepared.index.identity, fresh.index.identity)
        self.assertEqual(self.window.config.toPlainText(), before)
        self.assertFalse(self.window.jobs)
