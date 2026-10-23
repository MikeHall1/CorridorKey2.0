"""Actual Qt export index lifecycle and no per-frame directory rescans."""
import importlib.util
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import test_desktop_playback as playback_fixtures


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt binding unavailable')
class DesktopExportIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        playback_fixtures.DesktopPlaybackTests.setUpClass()

    def setUp(self):
        self.fixture = playback_fixtures.DesktopPlaybackTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window, self.controls = self.fixture.window, self.fixture.controls
        self.until = self.fixture.until
        self.window.view.setCurrentText('Checker composite')

    def play(self):
        self.controls.play()
        self.until(lambda: not self.controls.preparing and self.controls.displayed_frame is not None)
        self.assertIsNotNone(self.controls.export_indexes)

    def advance(self):
        engine = self.controls.engine
        assert engine is not None and self.controls.displayed_frame is not None
        frame = self.controls.displayed_frame + 1
        self.assertLess(frame, engine.end)
        engine.seek(frame)
        self.controls.tick()
        self.until(lambda: self.controls.displayed_frame == frame)
        return frame

    def test_actual_fg_alpha_playback_reuses_prepared_paths_without_per_frame_scan(self):
        import os

        from corridorkey.desktop_preview import preview
        original_scan = os.scandir
        scans = []
        def scan(path):
            scans.append(str(Path(path).resolve()))
            return original_scan(path)
        with patch('corridorkey.preview_sequence.os.scandir', side_effect=scan):
            self.play()
            before = list(scans)
            self.assertTrue(before)
            with patch('corridorkey.desktop_preview.frame_paths', side_effect=AssertionError('per-frame directory scan')):
                frame = self.advance()
            self.assertEqual(scans, before)
        view = self.window.selected_render_view()
        legacy = preview(**view.options('Checker composite', frame))
        pixmap = self.window.pixmap
        assert pixmap is not None
        actual = pixmap.toImage().pixelColor(0, 0)
        self.assertEqual((actual.red(), actual.green(), actual.blue()), tuple(legacy.pixels[0, 0]))
        self.assertIn('index', self.window.preview_note.text().lower())

    def test_added_export_frames_are_frozen_until_explicit_refresh_then_count_refuses(self):
        import cv2
        import numpy as np
        self.play()
        view = self.window.selected_render_view()
        for kind, path in view.outputs:
            shape = (2, 3) if kind == 'alpha' else (2, 3, 3)
            self.assertTrue(cv2.imwrite(str(Path(path) / '999.exr'), np.full(shape, .4, np.float32)))
        self.advance()  # New membership does not silently alter the frozen sequence.
        self.controls.refresh_source()
        self.assertIsNone(self.controls.export_indexes)
        self.controls.play()
        self.until(lambda: not self.controls.preparing)
        self.assertIsNone(self.controls.export_indexes)
        self.assertIn('frame coverage differs', self.controls.status.text().lower())

    def test_changed_leaf_pauses_without_reindex_until_refresh(self):
        import cv2
        import numpy as np
        self.play()
        engine = self.controls.engine
        assert engine is not None and self.controls.displayed_frame is not None
        target = self.controls.displayed_frame + 1
        alpha = Path(dict(self.window.selected_render_view().outputs)['alpha']) / f'{target-engine.view.start:03}.exr'
        self.assertTrue(cv2.imwrite(str(alpha), np.full((2, 3), .1, np.float32)))
        snapshot = self.controls.export_indexes
        engine.seek(target)
        self.controls.tick()
        self.until(lambda: 'Refresh' in self.controls.status.text())
        self.assertFalse(engine.playing)
        self.assertIs(self.controls.export_indexes, snapshot)
        self.controls.play()
        self.until(lambda: 'Refresh' in self.controls.status.text())
        self.assertIs(self.controls.export_indexes, snapshot)
        self.controls.refresh_source()
        self.controls.play()
        self.until(lambda: not self.controls.preparing and self.controls.export_indexes is not None)
        self.assertIsNot(self.controls.export_indexes, snapshot)

    def test_late_preparation_after_mode_change_never_starts_old_playback(self):
        from corridorkey.preview_exports import prepare_export_indexes
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            result = prepare_export_indexes(*args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture index release')
            return result
        with patch('corridorkey.desktop_loader.prepare_export_indexes', side_effect=blocked):
            self.controls.play()
            try:
                self.assertTrue(entered.wait(3))
                self.window.view.setCurrentText('Alpha')
                self.assertFalse(self.controls.preparing)
            finally:
                release.set()
            terminal = self.window.loader.results.get(timeout=5)
            self.window.loader.results.put_nowait(terminal)
            self.window.poll()
        self.assertIsNone(self.controls.export_indexes)
        engine = self.controls.engine
        assert engine is not None
        self.assertFalse(engine.playing)

    def test_source_aligned_a_b_uses_frozen_separate_local_offsets(self):
        from corridorkey.desktop_preview import preview_pair
        self.window.table.selectRow(0)
        self.window.view.setCurrentText('Alpha')
        self.window.synchronize_reference()
        self.window.table.selectRow(1)
        self.window.frame.setValue(3)
        self.window.request_preview()
        self.until(lambda: self.window.pixmap is not None and self.window.reference is not None)
        self.controls.play()
        self.until(lambda: not self.controls.preparing and self.controls.displayed_frame is not None)
        frame = self.controls.displayed_frame
        assert frame is not None
        reference = self.window.moving_reference
        assert reference is not None
        b = self.window.selected_render_view()
        legacy = preview_pair(reference.options('Alpha', frame), b.options('Alpha', frame))
        self.assertAlmostEqual(self.fixture.pixel(reference=True), int(legacy.a.pixels[0, 0, 0]), delta=0)
        self.assertAlmostEqual(self.fixture.pixel(), int(legacy.b.pixels[0, 0, 0]), delta=0)
        self.assertEqual(frame-reference.start, frame-1)
        self.assertEqual(frame-b.start, frame-2)

    def test_parent_close_waits_for_existing_index_preparation_owner(self):
        from corridorkey.preview_exports import prepare_export_indexes
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            result = prepare_export_indexes(*args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture close release')
            return result
        with patch('corridorkey.desktop_loader.prepare_export_indexes', side_effect=blocked):
            self.controls.play()
            try:
                self.assertTrue(entered.wait(3))
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(self.window.loader.thread.is_alive())
            finally:
                release.set()
            self.until(lambda: not self.window.isVisible())
        self.assertFalse(self.window.loader.thread.is_alive())
        self.assertIsNone(self.controls.export_indexes)


if __name__ == '__main__':
    unittest.main()
