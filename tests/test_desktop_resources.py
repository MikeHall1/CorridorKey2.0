"""Actual worker snapshots and responsive filesystem I/O ownership."""
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.qt_cleanup import dispose_widget

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class DesktopResourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['resources-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_window import DesktopWindow
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name).resolve()
        self.window = DesktopWindow()
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.bridge.thread.join(10)
        if self.window.resources.io_thread is not None:
            self.window.resources.io_thread.join(10)
        self.window.poll()
        self.app.processEvents()
        dispose_widget(self.window)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Resource condition not reached: ' + self.window.resources.status.text())

    def test_actual_worker_and_disk_snapshots_create_no_job_output_or_cuda_context(self):
        panel = self.window.resources
        self.window.request_resources()
        self.until(lambda: bool(panel.worker_view.toPlainText()))
        data = json.loads(panel.worker_view.toPlainText())
        self.assertNotEqual(data['payload']['resources']['pid'], os.getpid())
        self.assertIsNone(data['payload']['resources']['gpu']['allocated_bytes'])
        self.assertIsNone(data['payload']['active'])
        self.assertFalse(self.window.jobs)
        self.assertIsNone(self.window.resource_request)
        self.assertIn('lifetime peak RSS', panel.status.text())
        target = self.root / 'not-created' / 'output'
        panel.path.setText(str(target))
        panel.sample_disk()
        self.until(lambda: not panel.io_busy and bool(panel.disk_view.toPlainText()))
        self.assertEqual(json.loads(panel.disk_view.toPlainText())['measured_at'], str(self.root))
        self.assertFalse(target.parent.exists())
        panel.observe({'kind': 'lost', 'payload': {'error': 'fixture' * 20000}})
        self.assertLessEqual(len(panel.status.text()), 4096)
        self.assertLess(len(panel.worker_view.toPlainText()), 66000)

    def test_close_waits_for_disk_io_and_rejects_new_worker_snapshot(self):
        from corridorkey.resource_snapshot import disk_snapshot
        entered, release = threading.Event(), threading.Event()

        def slow_snapshot(path):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture gate')
            return disk_snapshot(path)

        with patch('corridorkey.desktop_resources.disk_snapshot', side_effect=slow_snapshot):
            try:
                self.window.resources.path.setText(str(self.root))
                self.window.resources.sample_disk()
                self.assertTrue(entered.wait(5))
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.window.request_resources()
                self.assertIn('shutdown', self.window.resources.status.text())
                self.assertIsNone(self.window.resource_request)
            finally:
                release.set()
            self.until(lambda: not self.window.isVisible())
        self.assertFalse(self.window.jobs)
