"""Fresh-review regressions for preview shutdown and explicit correction selection."""
import importlib.util
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class DesktopOwnershipReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['desktop-ownership-review', '-platform', 'offscreen'])

    def setUp(self):
        import cv2
        import numpy as np
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.assertTrue(cv2.imwrite(str(self.source / '0.png'), np.zeros((2, 3, 3), np.uint8)))
        self.original = (self.source / '0.png').read_bytes()

    def until(self, poll, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            self.app.processEvents()
            poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail('Owned UI operation did not reach its expected state')

    def test_close_waits_for_actual_preview_decoder_before_stopping_timer(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_preview import preview
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        from tests.qt_cleanup import dispose_widget
        window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable,
            str(Path(__file__).with_name('worker_fixture.py'))])))
        entered, release = threading.Event(), threading.Event()
        def gated(**settings):
            entered.set()
            if not release.wait(8):
                raise TimeoutError('preview regression gate')
            return preview(**settings)
        try:
            window.show()
            window.set_source(str(self.source))
            with patch('corridorkey.desktop_loader.preview', side_effect=gated):
                window.request_preview()
                self.assertTrue(entered.wait(5))
                window.close()
                self.until(window.poll, window.bridge.done.is_set)
                # Repeat the close path after the independent bridge owner ends.
                window.close()
                self.assertTrue(window.isVisible())
                self.assertTrue(window.timer.isActive())
                self.assertTrue(window.loader.thread.is_alive())
                self.assertFalse(release.is_set())
                self.assertIn('preview', window.statusBar().currentMessage())
                release.set()
                self.until(window.poll, lambda: not window.isVisible())
            window.bridge.thread.join(5)
            window.loader.thread.join(5)
            self.assertFalse(window.loader.thread.is_alive())
            self.assertFalse(window.bridge.thread.is_alive())
            self.assertFalse(window.timer.isActive())
            self.assertEqual((self.source / '0.png').read_bytes(), self.original)
            self.assertFalse(window.jobs)
        finally:
            release.set()
            window.close()
            window.bridge.thread.join(5)
            window.loader.thread.join(5)
            window.poll()
            dispose_widget(window)

    def test_explicit_new_worker_blocks_uncertain_cleanup_but_allows_clean_crash(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        from tests.qt_cleanup import dispose_widget
        window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable,
            str(Path(__file__).with_name('worker_fixture.py'))])))
        try:
            window.bridge.stop()
            window.bridge.thread.join(5)
            self.assertTrue(window.bridge.done.is_set())
            old = window.bridge
            old.failure = 'Fixture crash; child reaped and I/O owners stopped'
            self.assertIsNone(old.cleanup_error)
            window.new_worker()
            self.assertIsNot(window.bridge, old)
            replacement = window.bridge
            replacement.stop()
            replacement.thread.join(5)
            self.assertTrue(replacement.done.is_set())
            replacement.failure = 'Fixture crash'
            replacement.cleanup_error = 'Owned child cleanup could not be verified'
            window.new_worker()
            self.assertIs(window.bridge, replacement)
            self.assertIn('cleanup is unverified', window.statusBar().currentMessage())
            self.assertFalse(window.jobs)
        finally:
            window.close()
            window.bridge.thread.join(5)
            window.loader.thread.join(5)
            window.poll()
            dispose_widget(window)

    def test_deselected_correction_cannot_be_updated_deleted_or_reordered(self):
        from corridorkey.desktop_corrections import CorrectionDialog
        from corridorkey.session import JobRequest
        from tests.qt_cleanup import dispose_widget
        dialog = CorrectionDialog(JobRequest.create(self.source, self.root / 'render'), self.root)
        try:
            dialog.show()
            self.until(dialog.poll, lambda: dialog.snapshot is not None and dialog.pending is None)
            self.assertTrue(dialog.request('reset', frame=0))
            self.until(dialog.poll, lambda: len(dialog.entries) == 1 and dialog.pending is None)
            selected = dialog.selected()
            assert selected is not None and dialog.snapshot is not None
            original_id, original_identity = selected['id'], dialog.snapshot['identity']
            dialog.table.clearSelection()
            self.app.processEvents()
            self.assertEqual(dialog.table.currentRow(), 0)
            self.assertFalse(dialog.table.selectionModel().selectedRows())
            self.assertIsNone(dialog.selected())
            self.assertTrue(all(not button.isEnabled() for button in dialog.selection_buttons))
            dialog.remove()
            dialog.move_entry(1)
            dialog.edit()
            self.assertIsNone(dialog.pending)
            self.assertEqual(dialog.snapshot['identity'], original_identity)
            self.assertEqual(dialog.entries[0]['id'], original_id)
            dialog.table.selectRow(0)
            self.assertTrue(all(button.isEnabled() for button in dialog.selection_buttons))
            dialog.remove()
            self.until(dialog.poll, lambda: not dialog.entries and dialog.pending is None)
            self.assertEqual((self.source / '0.png').read_bytes(), self.original)
        finally:
            dialog.close()
            dialog.service.thread.join(5)
            self.assertFalse(dialog.service.thread.is_alive())
            dialog.poll()
            dispose_widget(dialog)


class DesktopBridgeCleanupReviewTests(unittest.TestCase):
    def test_failed_factory_preserves_typed_cleanup_uncertainty_and_stops_admission(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.worker_client import WorkerCleanupError, WorkerLost
        for uncertain in (True, False):
            with self.subTest(uncertain=uncertain):
                invoked = threading.Event()
                attempts = []
                error = (WorkerCleanupError('Startup owned cleanup NOT VERIFIED') if uncertain else
                         WorkerLost('Startup child cleanly reaped'))
                def factory(attempts=attempts, invoked=invoked, error=error):
                    attempts.append(1)
                    invoked.set()
                    raise error
                bridge = DesktopBridge(factory)
                try:
                    bridge.send('status', {})
                    self.assertTrue(invoked.wait(5))
                    if uncertain:
                        self.assertTrue(bridge.done.wait(5))
                        assert isinstance(error, WorkerCleanupError)
                        self.assertEqual(bridge.cleanup_error, error.cleanup_error)
                        with self.assertRaises(RuntimeError):
                            bridge.send('status', {})
                    else:
                        bridge.stop()
                    bridge.thread.join(5)
                    self.assertFalse(bridge.thread.is_alive())
                    self.assertTrue(bridge.done.is_set())
                    self.assertEqual(len(attempts), 1)
                    if not uncertain:
                        self.assertIsNone(bridge.cleanup_error)
                finally:
                    bridge.stop()
                    bridge.thread.join(5)

    def test_bridge_keeps_cleanup_uncertainty_separate_from_connection_failure(self):
        from types import SimpleNamespace
        from typing import Any

        from corridorkey.desktop_bridge import DesktopBridge
        for mode in ('clean_crash', 'uncertain', 'close_exception', 'live_thread'):
            with self.subTest(mode=mode):
                entered, release = threading.Event(), threading.Event()
                class Client:
                    process = SimpleNamespace(pid=1)
                    generation = 'explicit-test-client'
                    failure = 'Fixture connection failed'
                    reader = writer = None
                    def __init__(self, release=release, mode=mode, entered=entered):
                        self.mode, self.entered = mode, entered
                        self.cleanup_error = 'Pinned child not gone' if mode == 'uncertain' else None
                        self.log_reader = (threading.Thread(target=release.wait, name='review-test-io-owner')
                                           if mode == 'live_thread' else None)
                        if self.log_reader is not None:
                            self.log_reader.start()
                    def send(self, *args, **kwargs):
                        self.entered.set()
                    def next_event(self, timeout=0):
                        raise TimeoutError()
                    def close(self):
                        if self.mode == 'close_exception':
                            raise OSError('fixture close could not verify owners')
                        return False
                client: Any = Client()
                bridge = DesktopBridge(lambda client=client: client)
                try:
                    bridge.send('status', {})
                    self.assertTrue(entered.wait(5))
                    bridge.stop()
                    bridge.thread.join(5)
                    self.assertFalse(bridge.thread.is_alive())
                    self.assertTrue(bridge.done.is_set())
                    self.assertIsNotNone(bridge.failure)
                    if mode == 'clean_crash':
                        self.assertIsNone(bridge.cleanup_error)
                    else:
                        self.assertIsNotNone(bridge.cleanup_error)
                finally:
                    release.set()
                    bridge.stop()
                    bridge.thread.join(5)
                    if client.log_reader is not None:
                        client.log_reader.join(5)
                        self.assertFalse(client.log_reader.is_alive())
