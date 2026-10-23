"""Actual history transfer UI, explicit settings load and owned close."""
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt binding unavailable')
class DesktopQueueTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['queue-transfer-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_queue import QueuePanel
        from corridorkey.queue_store import QueueStore
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_to_wire
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name).resolve()
        self.source = self.root / 'source.json'
        self.source.write_text('{}')
        self.batch = self.root / 'batch.json'
        self.batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
            {'input': str(self.source), 'output_dir': str(self.root / 'outputs'), 'settings': {'output_format': 'exr'}}]}))
        self.history = QueueStore.create(self.root)
        store = QueueStore(self.history)
        try:
            store.start_session()
            store.prepare('render', 'render', {'job': job_to_wire(JobRequest.create(self.source, self.root / 'output'))})
            self.recovery_payload = store.prepare('recover', 'recover', {'batch': str(self.batch), 'workspace': str(self.root / 'cache'), 'max_cache_bytes': 0})
        finally:
            store.close()
        QueueStore.seal(self.history)
        self.destination = self.root / 'destination'
        self.destination.mkdir()
        self.owner = QueuePanel()
        self.owner.show()
        self.owner.run_io('open', self.history)
        self.until(lambda: not self.owner.io_busy)
        self.owner.open_transfer()
        self.panel = self.owner.transfer_panel
        self.panel.destination_parent.setText(str(self.destination))
        self.activations, self.renders, self.recoveries = [], [], []
        self.owner.activate_requested.connect(self.activations.append)
        self.owner.load_requested.connect(self.renders.append)
        self.owner.recovery_requested.connect(self.recoveries.append)

    def tearDown(self):
        self.owner.close()
        if self.panel.task is not None:
            self.panel.task.thread.join(5)
        if self.owner.io_thread is not None:
            self.owner.io_thread.join(5)
        self.owner.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.owner)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.owner.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail('History transfer condition not reached')

    def finish(self):
        self.until(lambda: not self.owner.io_busy)

    def test_actual_export_import_explicit_use_and_relocated_recovery_load(self):
        from corridorkey.queue_store import QueueStore
        before = {str(path.relative_to(self.history)): path.read_bytes() for path in self.history.rglob('*') if path.is_file()}
        self.panel.start()
        self.finish()
        archive = Path(self.panel.result_path.text())
        self.assertEqual(archive.parent, self.destination)
        self.assertIn('Export complete', self.panel.status.text())
        self.assertIn('Preserved requests: 2', self.panel.details.toPlainText())
        self.assertFalse(self.panel.use_button.isEnabled())
        self.panel.operation.setCurrentIndex(1)
        self.panel.source_path.setText(str(archive))
        self.panel.start()
        self.finish()
        imported = Path(self.panel.result_path.text())
        self.assertTrue(self.panel.use_button.isEnabled())
        self.assertEqual(self.owner.root, self.history)
        self.panel.use_button.click()
        self.finish()
        self.assertEqual(self.owner.root, imported)
        self.assertTrue(self.owner.archived)
        self.assertFalse(self.owner.use_button.isEnabled())
        self.assertEqual(self.activations, [])
        self.assertEqual(self.renders, [])
        self.assertEqual(self.recoveries, [])
        row = next(index for index, value in enumerate(self.owner.rows) if value['id'] == 'recover')
        self.owner.table.selectRow(row)
        self.owner.load_recovery()
        self.finish()
        self.assertEqual(len(self.recoveries), 1)
        relocated = Path(self.recoveries[0]['batch'])
        self.assertTrue(relocated.is_relative_to(imported))
        self.assertEqual(relocated.read_bytes(), Path(self.recovery_payload['batch']).read_bytes())
        details = json.loads(self.owner.details.toPlainText())
        self.assertEqual(details['request']['payload']['batch'], self.recovery_payload['batch'])
        original = QueueStore.inspect(self.history, request_id='recover')
        saved = QueueStore.inspect(imported, request_id='recover')
        self.assertEqual(saved['request'], original['request'])
        row = next(index for index, value in enumerate(self.owner.rows) if value['id'] == 'render')
        self.owner.table.selectRow(row)
        self.owner.load_selected()
        self.finish()
        self.assertEqual(len(self.renders), 1)
        self.assertEqual(self.renders[0].input_path, str(self.source))
        self.assertEqual(self.activations, [])
        self.assertEqual(before, {str(path.relative_to(self.history)): path.read_bytes() for path in self.history.rglob('*') if path.is_file()})

    def test_parent_close_waits_unexpected_transfer_exit_and_excludes_history_io(self):
        entered, release = threading.Event(), threading.Event()
        names = []

        def unexpected(source, parent, *, check=None):
            names.append(threading.current_thread().name)
            entered.set()
            release.wait(5)
            raise SystemExit('unexpected export exit')

        with patch('corridorkey.queue_transfer.export_history', side_effect=unexpected):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            assert self.panel.task is not None
            task = self.panel.task
            previous_io = self.owner.io_thread
            self.owner.refresh()
            self.assertIs(self.owner.io_thread, previous_io)
            self.panel.start()
            self.assertIs(self.panel.task, task)
            self.owner.close()
            self.assertTrue(self.owner.isVisible())
            self.assertTrue(task.cancellation.cancelled)
            release.set()
            self.until(lambda: not self.owner.isVisible())
        self.assertFalse(task.thread.is_alive())
        self.assertTrue(task.done.is_set())
        self.assertEqual(names, ['queue-history-transfer'])
        self.assertIn('SystemExit', self.panel.status.text())
        self.assertEqual(task.results.maxsize, 1)

    def test_partial_cancelled_path_and_unknown_counts_remain_visible(self):
        entered, release = threading.Event(), threading.Event()
        partial = self.destination / 'partial'

        def partial_export(source, parent, *, check=None):
            partial.mkdir()
            entered.set()
            release.wait(5)
            return {'status': 'cancelled', 'destination': str(partial), 'usable': False,
                    'copied_bytes': None, 'copied_files': 1, 'request_count': None, 'error': 'cancel during copy'}

        with patch('corridorkey.queue_transfer.export_history', side_effect=partial_export):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            self.panel.cancel()
            release.set()
            self.finish()
        self.assertEqual(self.panel.result_path.text(), str(partial))
        self.assertIn('Export cancelled', self.panel.status.text())
        self.assertIn('Copied bytes: Unknown', self.panel.details.toPlainText())
        self.assertIn('Partial destination retained', self.panel.details.toPlainText())
        self.assertFalse(self.panel.use_button.isEnabled())

    def test_late_commit_survives_cancel_but_inflight_edit_cannot_enable_use(self):
        self.panel.operation.setCurrentIndex(1)
        self.panel.source_path.setText(str(self.history))
        entered, release = threading.Event(), threading.Event()
        captured = []

        def committed(source, parent, *, check=None):
            captured.append((source, parent))
            entered.set()
            release.wait(5)
            return {'status': 'complete', 'destination': str(parent / 'committed'), 'usable': True,
                    'copied_bytes': 9223372036854775807, 'copied_files': 1, 'request_count': 2}

        with patch('corridorkey.queue_transfer.import_history', side_effect=committed):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            self.panel.cancel()
            self.panel.source_path.setText(str(self.root))
            release.set()
            self.finish()
        self.assertEqual(captured, [(self.history, self.destination)])
        self.assertIn('Import complete', self.panel.status.text())
        self.assertIn('Copied bytes: 9223372036854775807', self.panel.details.toPlainText())
        self.assertFalse(self.panel.use_button.isEnabled())

    def test_history_worker_system_exit_delivers_terminal_and_blocks_child(self):
        entered, release = threading.Event(), threading.Event()

        def unexpected(*args, **kwargs):
            entered.set()
            release.wait(5)
            raise SystemExit('unexpected history exit')

        with patch('corridorkey.desktop_queue.QueueStore.inspect', side_effect=unexpected):
            self.owner.refresh()
            self.assertTrue(entered.wait(2))
            self.panel.start()
            self.assertIsNone(self.panel.task)
            self.assertIn('Wait for history', self.panel.status.text())
            release.set()
            self.finish()
        self.assertIn('SystemExit', self.owner.status.text())
