"""Real preview recovery owns only its old message, never newer job errors."""
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class ComparisonStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['comparison-status-test', '-platform', 'offscreen'])

    def setUp(self):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.session import JobRequest
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'unchanged source identity fixture; only native exports are decoded')
        self.releases = []
        self.window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable,
            str(Path(__file__).with_name('worker_fixture.py'))])))
        self.window.set_source(str(self.source))
        self.window.output.setText(str(self.root))
        for name, start, values in [('A', 10, [.1, .2, .3, .4, .5]), ('B', 12, [.7, .8, .9])]:
            output = self.root / name
            alpha = output / 'alpha'
            alpha.mkdir(parents=True)
            for frame, value in enumerate(values):
                self.assertTrue(cv2.imwrite(str(alpha / f'alpha_{frame:06}.exr'), np.full((2, 3), value, np.float32)))
            job = JobRequest.create(self.source, output, InferenceConfig(output_format='exr', outputs={'alpha'}))
            record = {'status': 'complete', 'input': job.input_path, 'diagnostics': {
                'source': {'start_frame': start, 'num_frames': len(values), 'native_hw': [2, 3], 'native_fps': 24.},
                'frames_submitted': len(values), 'outputs': {'alpha': str(alpha)}}}
            self.window.jobs[name] = {'job': job, 'status': 'result', 'event': {'kind': 'result', 'payload': {'record': record}}}
        self.originals = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.job_snapshots = {key: state['job'] for key, state in self.window.jobs.items()}
        self.window.show()
        self.window.refresh_table()
        self.window.table.selectRow(0)
        self.window.view.setCurrentText('Alpha')
        self.window.synchronize_reference()
        self.until(lambda: self.window.pixmap is not None)
        self.window.table.selectRow(1)
        self.assertIsNone(self.window.pixmap)
        self.assertIn('common source frame in [12, 15)', self.window.statusBar().currentMessage())
        self.old_error = self.window.statusBar().currentMessage()
        self.assertEqual(self.window.details.toPlainText(), self.old_error)

    def tearDown(self):
        from tests.qt_cleanup import dispose_widget
        for release in self.releases:
            release.set()
        self.window.close()
        self.window.bridge.thread.join(5)
        self.window.loader.thread.join(5)
        self.assertFalse(self.window.bridge.thread.is_alive())
        self.assertFalse(self.window.loader.thread.is_alive())
        self.window.poll()
        for path, content in self.originals.items():
            self.assertEqual(path.read_bytes(), content)
        for key, job in self.job_snapshots.items():
            self.assertEqual(self.window.jobs[key]['job'], job)
            self.assertEqual(self.window.jobs[key]['status'], 'result')
        self.assertEqual(self.window.bridge.commands.qsize(), 0)
        dispose_widget(self.window)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail(self.window.statusBar().currentMessage())

    @contextmanager
    def delayed_valid_preview(self):
        from corridorkey.desktop_preview import preview_pair
        started, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def gated(*args, **kwargs):
            started.set()
            release.wait(5)
            return preview_pair(*args, **kwargs)

        with patch('corridorkey.desktop_loader.preview_pair', side_effect=gated):
            self.window.frame.setValue(12)
            self.assertTrue(started.wait(2))
            yield
            release.set()
            self.until(lambda: self.window.pixmap is not None and self.window.reference is not None)

    def test_valid_shared_preview_replaces_own_error_and_restores_selected_record(self):
        self.window.frame.setValue(12)
        self.until(lambda: self.window.pixmap is not None and self.window.reference is not None)
        self.assertEqual(self.window.statusBar().currentMessage(), 'Synchronized comparison ready · source frame 12')
        self.assertEqual(json.loads(self.window.details.toPlainText()), self.window.jobs['B']['event'])
        assert self.window.reference is not None and self.window.pixmap is not None
        self.assertEqual(self.window.reference.toImage().pixelColor(0, 0).red(), 76)
        self.assertEqual(self.window.pixmap.toImage().pixelColor(0, 0).red(), 178)
        self.assertIn('export frame 2', self.window.reference_description)
        self.assertIn('decoded frame 0', self.window.preview_description)
        self.assertEqual(len(self.window.jobs), 2)

    def test_newer_unrelated_error_survives_late_successful_pair(self):
        message = 'Unrelated worker failed: encoder disk error'
        with self.delayed_valid_preview():
            self.window.error(message)
        self.assertEqual(self.window.statusBar().currentMessage(), message)
        self.assertEqual(self.window.details.toPlainText(), message)
        self.assertIsNotNone(self.window.pixmap)

    def test_actual_bridge_job_error_and_new_record_details_survive_late_pair(self):
        self.window.jobs['other'] = {'job': replace(self.window.jobs['B']['job'], output_dir=str(self.root / 'other')),
                                     'status': 'accepted', 'event': None}
        self.window.refresh_table()
        with self.delayed_valid_preview():
            self.window.bridge._publish({'request_id': 'other', 'kind': 'failed', 'payload': {
                'error': {'type': 'OSError', 'message': 'Independent encoder disk error'}}})
            self.window.poll()
            expected_status = self.window.statusBar().currentMessage()
            expected_details = self.window.details.toPlainText()
            self.assertIn('Independent encoder disk error', expected_status)
            self.assertEqual(self.window.jobs['other']['status'], 'failed')
        self.assertEqual(self.window.statusBar().currentMessage(), expected_status)
        self.assertEqual(self.window.details.toPlainText(), expected_details)
        self.assertIsNotNone(self.window.pixmap)
