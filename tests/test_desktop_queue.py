import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
FIXTURE = Path(__file__).with_name('worker_fixture.py')


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt binding unavailable')
class DesktopQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['queue-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(FIXTURE)])))
        self.window.show()
        self.window.queue_panel.show()

    def tearDown(self):
        self.window.close()
        self.window.bridge.thread.join(10)
        thread = self.window.queue_panel.io_thread
        if thread is not None:
            thread.join(5)
        transfer = self.window.queue_panel.transfer_panel.task
        if transfer is not None:
            transfer.thread.join(5)
        self.window.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
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
        self.fail('Queue condition not reached: ' + self.window.queue_panel.status.text())

    def test_durable_create_render_history_and_load_settings_do_not_replay(self):
        from corridorkey.queue_store import QueueStore
        panel = self.window.queue_panel
        panel.run_io('create', self.root)
        self.until(lambda: self.window.bridge.journal_workspace is not None and 'Durable queue:' in self.window.queue_label.text())
        workspace = self.window.bridge.journal_workspace
        assert workspace is not None
        source = self.root / 'source.json'
        source.write_text('{}')
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root))
        self.window.update_settings({'output_format': 'exr'})
        self.window.enqueue()
        request_id = next(iter(self.window.jobs))
        self.until(lambda: self.window.jobs[request_id]['status'] == 'result')
        self.assertEqual(QueueStore.inspect(workspace, request_id=request_id)['status'], 'result')
        panel.refresh()
        self.until(lambda: panel.table.rowCount() == 1 and not panel.io_busy)
        cell = panel.table.item(0, 2)
        assert cell is not None
        self.assertIn('unverified', cell.text())
        self.window.clear_finished()
        panel.table.selectRow(0)
        panel.load_selected()
        self.until(lambda: not panel.io_busy)
        self.assertFalse(self.window.jobs)
        self.assertEqual(self.window.source.text(), str(source.resolve()))
        self.assertEqual(self.window.output.text(), str(self.root.resolve()))
        self.assertEqual(len(QueueStore.inspect(workspace)['rows']), 1)

    def test_deselected_history_does_not_load_old_current_row(self):
        from corridorkey.queue_store import QueueStore
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_to_wire
        root = QueueStore.create(self.root)
        store = QueueStore(root)
        try:
            store.start_session()
            store.prepare('history', 'render', {'job': job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))})
        finally:
            store.close()
        panel = self.window.queue_panel
        panel.run_io('open', root)
        self.until(lambda: not panel.io_busy and len(panel.rows) == 1)
        panel.table.selectRow(0)
        self.assertEqual(panel.selected(), 'history')
        panel.table.clearSelection()
        self.assertIsNone(panel.selected())
        original_thread = panel.io_thread
        for action in (panel.inspect_selected, panel.load_selected, panel.load_recovery):
            action()
        self.assertIs(panel.io_thread, original_thread)
        self.assertFalse(self.window.jobs)

    def test_open_unresolved_history_is_read_only_until_explicit_activation(self):
        from corridorkey.queue_store import QueueStore
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_to_wire
        root = QueueStore.create(self.root)
        store = QueueStore(root)
        try:
            store.start_session()
            store.prepare('pending', 'render', {'job': job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))})
        finally:
            store.close()
        self.window.queue_panel.run_io('open', root)
        self.until(lambda: self.window.queue_panel.table.rowCount() == 1)
        cell = self.window.queue_panel.table.item(0, 2)
        assert cell is not None
        self.assertIn('unresolved', cell.text())
        self.assertIsNone(self.window.bridge.journal_workspace)
        self.assertFalse(self.window.jobs)
        self.assertTrue(self.window.bridge.commands.empty())

    def test_restart_does_not_replace_owner_activated_during_poll(self):
        self.window.bridge.stop()
        self.window.bridge.thread.join(10)
        self.assertTrue(self.window.bridge.done.is_set())
        panel = self.window.queue_panel
        panel.run_io('create', self.root)
        assert panel.io_thread is not None
        panel.io_thread.join(10)
        self.assertFalse(panel.results.empty())
        self.window.new_worker()
        self.until(lambda: 'Durable queue:' in self.window.queue_label.text())
        self.assertIsNone(self.window.bridge.failure)
        self.assertFalse(self.window.bridge.done.is_set())

    def test_competing_queue_owner_is_reported_without_guessing_it_stopped(self):
        from corridorkey.queue_store import QueueStore
        root = QueueStore.create(self.root)
        owner = QueueStore(root)
        try:
            owner.start_session()
            self.window.activate_queue(root)
            self.until(lambda: self.window.bridge.journal_workspace == root and self.window.bridge.done.is_set())
            self.window.poll()
            self.assertIsNotNone(self.window.bridge.failure)
            self.assertIn('unavailable', self.window.queue_label.text())
            self.assertFalse(self.window.jobs)
        finally:
            owner.close()

    def test_read_only_older_newer_refresh_and_old_record_inspection_do_not_submit(self):
        from corridorkey.queue_store import QueueStore
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_to_wire
        root = QueueStore.create(self.root)
        store = QueueStore(root)
        try:
            store.start_session()
            wire = job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))
            for index in range(260):
                store.prepare(f'history{index}', 'render', {'job': wire})
        finally:
            store.close()
        panel = self.window.queue_panel
        panel.run_io('open', root)
        self.until(lambda: not panel.io_busy and len(panel.rows) == 128)
        self.assertEqual(panel.rows[0]['id'], 'history259')
        panel.older()
        self.until(lambda: not panel.io_busy and len(panel.cursors) == 2)
        self.assertEqual(panel.rows[0]['id'], 'history131')
        panel.older()
        self.until(lambda: not panel.io_busy and len(panel.cursors) == 3)
        self.assertEqual(len(panel.rows), 4)
        self.assertFalse(panel.older_button.isEnabled())
        panel.table.selectRow(3)
        panel.inspect_selected()
        self.until(lambda: not panel.io_busy and 'history0' in panel.details.toPlainText())
        panel.newer()
        self.until(lambda: not panel.io_busy and len(panel.cursors) == 2)
        self.assertEqual(panel.rows[0]['id'], 'history131')
        panel.refresh()
        self.until(lambda: not panel.io_busy and len(panel.cursors) == 1)
        self.assertEqual(panel.rows[0]['id'], 'history259')
        self.assertFalse(panel.newer_button.isEnabled())
        self.assertIsNone(self.window.bridge.journal_workspace)
        self.assertFalse(self.window.jobs)
        self.assertTrue(self.window.bridge.commands.empty())

    def test_archive_requires_stopped_owner_and_keeps_readonly_inspection(self):
        from corridorkey.queue_store import QueueStore
        panel = self.window.queue_panel
        panel.run_io('create', self.root)
        self.until(lambda: not panel.io_busy and 'Durable queue:' in self.window.queue_label.text())
        root = panel.root
        assert root is not None
        bridge = self.window.bridge
        panel.seal_history()
        self.until(lambda: not panel.io_busy)
        self.assertFalse(panel.archived)
        self.assertFalse((root / 'queue-archive.json').exists())
        self.assertFalse(bridge.stopping.is_set())
        self.window.stop_worker()
        self.until(lambda: bridge.done.is_set())
        panel.seal_history()
        self.until(lambda: not panel.io_busy and panel.archived)
        self.assertIsNotNone(QueueStore.inspect(root)['archive'])
        self.assertFalse(panel.use_button.isEnabled())
        panel.activate()
        self.assertIn('cannot accept', panel.status.text())
        self.assertIs(self.window.bridge, bridge)
        self.assertFalse(self.window.jobs)
        self.assertIn('stopped', self.window.queue_label.text())

    def test_archive_and_fresh_queue_preserve_old_history_and_activate_no_jobs(self):
        from corridorkey.queue_store import QueueStore
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_to_wire
        root = QueueStore.create(self.root)
        store = QueueStore(root)
        try:
            store.start_session()
            store.prepare('preserved', 'render', {'job': job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))})
        finally:
            store.close()
        panel = self.window.queue_panel
        panel.run_io('open', root)
        self.until(lambda: not panel.io_busy and len(panel.rows) == 1)
        panel.rotate_history()
        self.until(lambda: not panel.io_busy and self.window.bridge.journal_workspace is not None and 'Durable queue:' in self.window.queue_label.text())
        fresh = self.window.bridge.journal_workspace
        self.assertNotEqual(fresh, root)
        self.assertEqual(panel.last_archive, root)
        self.assertEqual(panel.rows, [])
        self.assertFalse(self.window.jobs)
        self.assertTrue(self.window.bridge.commands.empty())
        panel.open_last_archive()
        self.until(lambda: not panel.io_busy and panel.archived)
        self.assertEqual(panel.rows[0]['id'], 'preserved')
        self.assertFalse(panel.use_button.isEnabled())
        self.assertEqual(self.window.bridge.journal_workspace, fresh)
        self.assertEqual(QueueStore.inspect(root, request_id='preserved')['status'], 'prepared')

    def test_fresh_queue_failure_reports_committed_archive_without_activation(self):
        from unittest.mock import patch

        from corridorkey.queue_store import QueueStore
        root = QueueStore.create(self.root)
        panel = self.window.queue_panel
        panel.run_io('open', root)
        self.until(lambda: not panel.io_busy and panel.root == root)
        bridge = self.window.bridge
        with patch('corridorkey.desktop_queue.QueueStore.create', side_effect=OSError('fixture creation failure')):
            panel.rotate_history()
            self.until(lambda: not panel.io_busy)
        self.assertIsNotNone(QueueStore.inspect(root)['archive'])
        self.assertTrue(panel.archived)
        self.assertIn('History is archived', panel.status.text())
        self.assertFalse(panel.use_button.isEnabled())
        self.assertIs(self.window.bridge, bridge)
        self.assertIsNone(bridge.journal_workspace)
        self.assertFalse(self.window.jobs)

    def test_queue_change_respects_selected_assets_before_stopping_current_owner(self):
        from corridorkey.asset_inventory import AssetSelection
        from corridorkey.queue_store import QueueStore
        self.window.asset_selection = AssetSelection(self.root)
        root = QueueStore.create(self.root)
        bridge = self.window.bridge
        self.window.activate_queue(root)
        self.assertIsNone(self.window.pending_queue_root)
        self.assertFalse(bridge.stopping.is_set())
        self.assertIs(self.window.bridge, bridge)
        self.assertIn('separate', self.window.queue_panel.status.text())
