"""Real local sequence/EXR decoder playback checks; no model inference."""
import importlib.util
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional requirements-desktop.txt is not installed')
class DesktopPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['playback-ui-test', '-platform', 'offscreen'])

    def setUp(self):
        import cv2
        import numpy as np

        from corridorkey.desktop_window import DesktopWindow
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(7):
            self.assertTrue(cv2.imwrite(str(self.source / f'{frame:03}.png'), np.full((2, 3, 3), 20 + frame * 20, np.uint8)))
        self.originals = {path: path.read_bytes() for path in self.source.iterdir()}
        self.window = DesktopWindow()
        self.window.show()
        self.window.timer.stop()  # tests drive the existing poll deterministically
        self.controls = self.window.playback
        self.add_render('a', 1, [.1, .2, .3, .4, .5])
        self.add_render('b', 2, [.7, .8, .9])
        self.window.populate_job(self.window.jobs['b']['job'])
        self.window.refresh_table()
        self.window.table.selectRow(1)
        self.window.view.setCurrentText('Alpha')
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

    def add_render(self, name, start, values):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest
        output = self.root / name
        alpha, fg = output / 'alpha', output / 'fg'
        alpha.mkdir(parents=True)
        fg.mkdir()
        for index, value in enumerate(values):
            self.assertTrue(cv2.imwrite(str(alpha / f'{index:03}.exr'), np.full((2, 3), value, np.float32)))
            self.assertTrue(cv2.imwrite(str(fg / f'{index:03}.exr'), np.full((2, 3, 3), .25, np.float32)))
        config = InferenceConfig(output_format='exr', outputs={'alpha', 'fg'}, frame_dir_fps=10., start_frame=start, num_frames=len(values))
        job = JobRequest.create(self.source, output, config)
        record: dict[str, Any] = {'status': 'complete', 'input': job.input_path, 'diagnostics': {
            'source': {'start_frame': start, 'num_frames': len(values), 'native_hw': [2, 3], 'native_fps': 10.},
            'frames_submitted': len(values), 'outputs': {'alpha': str(alpha), 'fg': str(fg)}}}
        self.window.jobs[name] = {'job': job, 'status': 'result', 'event': {'kind': 'result', 'payload': {'record': record}}}

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail('Playback condition not reached: ' + self.controls.status.text())

    def pixel(self, *, reference=False):
        pixmap = self.window.reference if reference else self.window.pixmap
        assert pixmap is not None
        return pixmap.toImage().pixelColor(0, 0).red()

    def clock(self):
        assert self.controls.engine is not None
        value = [time.monotonic()]
        self.controls.engine.clock = lambda: value[0]
        return value

    def test_real_exports_play_pause_manual_seek_and_source_bytes_preserved(self):
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        assert self.controls.displayed_seconds is not None
        self.assertAlmostEqual(self.controls.displayed_seconds, .2)
        self.assertAlmostEqual(self.pixel(), 178, delta=1)
        self.assertIn('Displayed source frame 2', self.controls.status.text())
        self.window.frame.setValue(1)  # existing export-local still navigation
        assert self.controls.engine is not None
        assert self.controls.engine is not None
        self.assertFalse(self.controls.engine.playing)
        self.window.request_preview()
        self.until(lambda: self.window.pixmap is not None and 'frame 1' in self.window.preview_description)
        self.assertAlmostEqual(self.pixel(), 204, delta=1)
        self.assertEqual(self.window.frame.value(), 1)
        self.assertEqual({path: path.read_bytes() for path in self.source.iterdir()}, self.originals)

    def test_declared_source_playback_and_loop_range_are_source_coordinates(self):
        self.window.view.setCurrentText('Source')
        self.controls.in_frame.setText('2')
        self.controls.out_frame.setText('4')
        self.controls.loop.setChecked(True)
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        clock = self.clock()
        engine = self.controls.engine
        assert engine is not None
        engine.seek(2)
        clock[0] += .15
        self.until(lambda: self.controls.displayed_frame == 3)
        self.assertEqual(self.pixel(), 80)
        clock[0] += .10
        self.until(lambda: self.controls.displayed_frame == 2)
        assert self.controls.displayed_seconds is not None
        self.assertAlmostEqual(self.controls.displayed_seconds, .2)
        self.controls.pause()
        self.assertFalse(engine.playing)

    def test_synchronized_real_exr_pair_uses_distinct_export_offsets(self):
        self.window.table.selectRow(0)
        self.window.synchronize_reference()
        self.window.table.selectRow(1)
        self.controls.in_frame.setText('3')
        self.controls.out_frame.setText('5')
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 3)
        self.assertAlmostEqual(self.pixel(reference=True), 76, delta=1)
        self.assertAlmostEqual(self.pixel(), 204, delta=1)
        self.assertIn('source frame 3', self.window.reference_description)
        self.controls.pause()
        self.window.clear_reference()
        assert self.controls.engine is not None
        self.assertFalse(self.controls.engine.playing)

    def test_native_inflight_decode_is_fenced_and_manual_success_retires_owner(self):
        from corridorkey.desktop_preview import preview
        entered, release = threading.Event(), threading.Event()

        def gated_preview(**settings):
            entered.set()
            release.wait(5)
            return preview(**settings)

        with patch('corridorkey.desktop_loader.preview', side_effect=gated_preview):
            try:
                self.controls.play()
                self.assertTrue(entered.wait(2))
                pending = self.controls.pending_serial
                self.window.frame.setValue(1)
                self.window.request_preview()
                assert self.controls.engine is not None
                self.assertFalse(self.controls.engine.playing)
                self.assertEqual(self.controls.pending_serial, pending)
                self.assertLessEqual(self.window.loader.requests.qsize(), 1)
                time.sleep(.02)
            finally:
                release.set()
            self.until(lambda: self.controls.pending_serial is None and self.window.pixmap is not None)
        self.assertIsNone(self.controls.displayed_frame)
        self.assertAlmostEqual(self.pixel(), 204, delta=1)
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 3)

    def test_obsolete_decode_error_retires_without_stopping_latest_playback(self):
        from corridorkey.desktop_preview import preview
        entered, release = threading.Event(), threading.Event()

        def gated_preview(**settings):
            if settings.get('coverage'):
                entered.set()
                release.wait(5)
                raise ValueError('obsolete Alpha decode failed')
            return preview(**settings)

        with patch('corridorkey.desktop_loader.preview', side_effect=gated_preview):
            try:
                self.controls.play()
                self.assertTrue(entered.wait(2))
                self.window.view.setCurrentText('Source')
                self.controls.play()
                self.clock()
                assert self.controls.engine is not None
                self.assertTrue(self.controls.engine.playing)
            finally:
                release.set()
            self.until(lambda: self.controls.displayed_frame == 2)
        self.assertTrue(self.controls.engine.playing)
        self.assertEqual(self.controls.engine.stale_completions, 1)
        self.assertNotIn('obsolete Alpha', self.controls.status.text())
        self.assertAlmostEqual(self.pixel(), 60, delta=1)

    def test_current_decode_error_still_stops_and_reports(self):
        with patch('corridorkey.desktop_loader.preview', side_effect=ValueError('current decode failure')):
            self.controls.play()
            self.until(lambda: self.controls.pending_serial is None)
        assert self.controls.engine is not None
        self.assertFalse(self.controls.engine.playing)
        self.assertIn('current decode failure', self.controls.status.text())

    def test_bad_timing_video_and_nonoverlap_refuse_without_decode(self):
        from dataclasses import replace
        state = self.window.jobs['b']
        source_record = state['event']['payload']['record']['diagnostics']['source']
        source_record['native_fps'] = 24.
        with patch.object(self.window.loader, 'submit') as submit:
            self.controls.play()
            self.assertIn('timing differs', self.controls.status.text())
            self.assertFalse(submit.called)
        source_record['native_fps'] = 10.
        state['job'] = replace(state['job'], input_path=str(self.root / 'video.mp4'))
        state['event']['payload']['record']['input'] = state['job'].input_path
        with patch.object(self.window.loader, 'submit') as submit:
            self.controls.play()
            self.assertIn('Video/VFR', self.controls.status.text())
            self.assertFalse(submit.called)

    def test_composite_speed_selection_and_invalid_loop_range(self):
        self.window.view.setCurrentText('Checker composite')
        self.controls.in_frame.setText('0')
        self.controls.play()
        self.assertIn('Loop range', self.controls.status.text())
        self.controls.in_frame.clear()
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        self.assertIsNotNone(self.window.pixmap)
        self.controls.speed.setCurrentIndex(3)
        assert self.controls.engine is not None
        self.assertEqual(self.controls.engine.speed, 2.)
        self.window.table.selectRow(0)
        assert self.controls.engine is not None
        self.assertFalse(self.controls.engine.playing)
        self.assertIn('dropped opportunities', self.controls.status.text())

    def test_same_selection_record_refresh_keeps_playing_and_range_end_displays(self):
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        self.window.refresh_table()
        self.window.show_record()
        engine = self.controls.engine
        assert engine is not None
        self.assertTrue(engine.playing)
        clock = self.clock()
        engine.seek(2)
        clock[0] += .35
        self.controls.tick()
        self.assertFalse(engine.playing)
        self.window.show_record()  # unrelated job events must not fence final pixels
        self.until(lambda: self.controls.displayed_frame == 4)
        self.assertAlmostEqual(self.pixel(), 230, delta=1)
        self.assertIsNone(self.controls.pending_serial)

    def advance_to_three_and_pause(self):
        self.controls.play()
        self.until(lambda: self.controls.displayed_frame == 2)
        clock = self.clock()
        engine = self.controls.engine
        assert engine is not None
        engine.seek(2)
        clock[0] += .15
        self.until(lambda: self.controls.displayed_frame == 3)
        self.controls.pause()
        self.assertEqual(engine.frame, 3)
        self.assertEqual(self.window.frame.value(), 0)
        return clock

    def test_pause_resume_keeps_cursor_and_repeated_play_keeps_pending_request(self):
        self.advance_to_three_and_pause()
        self.window.refresh_table()
        self.window.show_record()  # unrelated job updates preserve paused cursor
        self.controls.play()
        request = self.controls.pending_request
        assert request is not None
        self.assertEqual(request.source_frame, 3)
        serial = self.controls.pending_serial
        self.controls.play()
        self.assertIs(self.controls.pending_request, request)
        self.assertEqual(self.controls.pending_serial, serial)
        self.until(lambda: self.controls.pending_serial is None)
        self.assertAlmostEqual(self.pixel(), 204, delta=1)
        self.assertEqual(self.controls.displayed_frame, 3)

    def test_explicit_preview_at_unchanged_still_index_supersedes_paused_cursor(self):
        self.advance_to_three_and_pause()
        self.window.request_preview()  # unchanged local index0 deliberately shows source2
        self.until(lambda: self.window.pixmap is not None)
        self.assertAlmostEqual(self.pixel(), 178, delta=1)
        self.assertIsNone(self.controls.displayed_frame)
        self.controls.play()
        request = self.controls.pending_request
        assert request is not None
        self.assertEqual(request.source_frame, 2)
        self.until(lambda: self.controls.pending_serial is None)
        self.assertEqual(self.controls.displayed_frame, 2)

    def test_explicit_range_and_view_changes_supersede_paused_cursor(self):
        self.advance_to_three_and_pause()
        self.controls.in_frame.setText('4')
        self.controls.out_frame.setText('5')
        self.controls.play()
        request = self.controls.pending_request
        assert request is not None
        self.assertEqual(request.source_frame, 4)
        self.until(lambda: self.controls.pending_serial is None)
        self.controls.in_frame.clear()
        self.controls.out_frame.clear()
        self.window.view.setCurrentText('Source')
        self.controls.play()
        request = self.controls.pending_request
        assert request is not None
        self.assertEqual(request.source_frame, 2)  # Source0 clamps to selected range In2
        self.until(lambda: self.controls.pending_serial is None)

    def test_pause_while_buffering_freezes_last_displayed_frame(self):
        from corridorkey.desktop_preview import preview
        clock = self.advance_to_three_and_pause()
        self.controls.play()
        self.until(lambda: self.controls.pending_serial is None)
        entered, release = threading.Event(), threading.Event()

        def gated_preview(**settings):
            entered.set()
            release.wait(5)
            return preview(**settings)

        with patch('corridorkey.desktop_loader.preview', side_effect=gated_preview):
            try:
                clock[0] += .15
                self.controls.tick()
                self.assertTrue(entered.wait(2))
                clock[0] += 10.
                self.controls.pause()
                assert self.controls.engine is not None
                self.assertEqual(self.controls.engine.frame, 3)
                self.assertEqual(self.controls.displayed_frame, 3)
            finally:
                release.set()
            self.until(lambda: self.controls.pending_serial is None)
        self.assertEqual(self.controls.displayed_frame, 3)
        self.controls.play()
        assert self.controls.pending_request is not None
        self.assertEqual(self.controls.pending_request.source_frame, 3)
        self.until(lambda: self.controls.pending_serial is None)
