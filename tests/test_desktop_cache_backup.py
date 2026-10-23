"""Actual Qt backup/restore workflow and parent-owned terminal cleanup."""
import importlib.util
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional requirements-desktop.txt is not installed')
class DesktopCacheBackupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['backup-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_cache import CacheMaintenancePanel
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_journal import AttemptJournal
        from corridorkey.recovery_store import BundleStore

        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = self.root / 'original.exr'
        self.source.write_bytes(b'local explicit fixture output' * 50)
        self.cache = self.root / 'cache'
        self.identity = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'backup-ui'}))
        with BundleStore(self.cache, 100000) as store:
            store.publish(self.identity, {'alpha.exr': self.source}, {'status': 'complete'})
            AttemptJournal(store, [{'fixture': 'backup-ui'}])
        self.destination = self.root / 'destination'
        self.destination.mkdir()
        self.owner = CacheMaintenancePanel()
        self.owner.set_workspace(str(self.cache))
        self.owner.show()
        self.owner.open_backup()
        self.panel = self.owner.backup_panel
        self.panel.destination_parent.setText(str(self.destination))
        self.app.processEvents()

    def tearDown(self):
        self.owner.close()
        if self.panel.task is not None:
            self.panel.task.thread.join(5)
        if self.owner.task is not None:
            self.owner.task.thread.join(5)
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
            time.sleep(0.005)
        self.fail('Backup UI condition not reached')

    def finish(self):
        self.until(lambda: not self.owner.io_busy)

    def test_actual_backup_restore_and_explicit_use_without_submission(self):
        from corridorkey.recovery_store import BundleStore

        original = {str(path.relative_to(self.cache)): path.read_bytes() for path in self.cache.rglob('*') if path.is_file()}
        self.assertIsNone(self.panel.task)
        self.assertFalse(self.panel.use_button.isEnabled())
        self.panel.start()
        self.finish()
        backup = Path(self.panel.result_path.text())
        self.assertEqual(backup.parent, self.destination.resolve())
        self.assertTrue(backup.is_dir())
        self.assertIn('Backup complete', self.panel.status.text())
        self.assertIn('Preserved journals: 1', self.panel.details.toPlainText())
        self.assertFalse(self.panel.use_button.isEnabled())
        self.panel.operation.setCurrentIndex(1)
        self.panel.source_path.setText(str(backup))
        self.panel.start()
        self.finish()
        restored = Path(self.panel.result_path.text())
        self.assertEqual(restored.parent, self.destination.resolve())
        self.assertNotEqual(restored, backup)
        self.assertTrue(self.panel.use_button.isEnabled())
        self.assertEqual(self.owner.root_path.text(), str(self.cache))
        self.assertIsNone(self.owner.task)
        self.panel.use_button.click()
        self.assertEqual(self.owner.root_path.text(), str(restored))
        self.assertIsNone(self.owner.task)
        self.assertIn('No recovery or rendering submitted', self.panel.status.text())
        with BundleStore(restored, 100000) as store:
            self.assertIsNotNone(store.lookup(self.identity))
        self.assertEqual(original, {str(path.relative_to(self.cache)): path.read_bytes() for path in self.cache.rglob('*') if path.is_file()})
        self.assertEqual(self.source.read_bytes(), b'local explicit fixture output' * 50)

    def test_parent_close_waits_background_system_exit_and_excludes_maintenance(self):
        entered, release = threading.Event(), threading.Event()
        names = []

        def unexpected(source, parent, *, check=None):
            names.append(threading.current_thread().name)
            entered.set()
            release.wait(5)
            raise SystemExit('unexpected backup exit')

        with patch('corridorkey.cache_backup.backup_workspace', side_effect=unexpected):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            assert self.panel.task is not None
            task = self.panel.task
            self.panel.start()
            self.assertIs(self.panel.task, task)
            self.owner.start('inventory')
            self.assertIsNone(self.owner.task)
            self.owner.close()
            self.assertTrue(self.owner.isVisible())
            self.assertTrue(self.owner.io_busy)
            self.assertTrue(task.cancellation.cancelled)
            release.set()
            self.until(lambda: not self.owner.isVisible())
        self.assertFalse(task.thread.is_alive())
        self.assertTrue(task.done.is_set())
        self.assertEqual(names, ['cache-backup-restore'])
        self.assertIn('SystemExit', self.panel.status.text())
        self.assertEqual(task.results.maxsize, 1)

    def test_cancelled_partial_destination_remains_visible_unusable(self):
        entered, release = threading.Event(), threading.Event()
        partial = self.destination / 'partial-fixture'

        def cancelled(source, parent, *, check=None):
            partial.mkdir()
            (partial / 'copied.bin').write_bytes(b'partial')
            entered.set()
            release.wait(5)
            return {'status': 'cancelled', 'destination': str(partial), 'usable': False,
                    'copied_bytes': 7, 'copied_files': 1, 'entry_count': None,
                    'journal_count': None, 'report_count': None, 'error': 'cancel during copy'}

        with patch('corridorkey.cache_backup.backup_workspace', side_effect=cancelled):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            self.panel.cancel()
            release.set()
            self.finish()
        self.assertEqual(self.panel.result_path.text(), str(partial))
        self.assertIn('Backup cancelled', self.panel.status.text())
        self.assertIn('Copied bytes: 7', self.panel.details.toPlainText())
        self.assertIn('Partial destination retained', self.panel.details.toPlainText())
        self.assertIn('Sealed bundles: Unknown', self.panel.details.toPlainText())
        self.assertFalse(self.panel.use_button.isEnabled())
        self.assertTrue((partial / 'copied.bin').exists())

    def test_committed_restore_survives_late_cancel_and_edit_invalidates_use(self):
        self.panel.operation.setCurrentIndex(1)
        self.panel.source_path.setText(str(self.cache))
        entered, release = threading.Event(), threading.Event()
        restored = self.destination / 'committed-fixture'

        def committed(source, parent, *, check=None):
            restored.mkdir()
            entered.set()
            release.wait(5)
            return {'status': 'complete', 'destination': str(restored), 'usable': True,
                    'copied_bytes': 9223372036854775807, 'copied_files': 1,
                    'entry_count': 1, 'journal_count': 0, 'report_count': 0, 'error': None}

        with patch('corridorkey.cache_backup.restore_backup', side_effect=committed):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            self.panel.cancel()
            release.set()
            self.finish()
        self.assertIn('Restore complete', self.panel.status.text())
        self.assertIn('Copied bytes: 9223372036854775807', self.panel.details.toPlainText())
        self.assertTrue(self.panel.use_button.isEnabled())
        self.panel.destination_parent.setText(str(self.root))
        self.assertFalse(self.panel.use_button.isEnabled())
        self.panel.use_restored()
        self.assertEqual(self.owner.root_path.text(), str(self.cache))

    def test_inflight_edit_does_not_retarget_copy_or_enable_use(self):
        self.panel.operation.setCurrentIndex(1)
        self.panel.source_path.setText(str(self.cache))
        entered, release = threading.Event(), threading.Event()
        seen = []

        def restore(source, parent, *, check=None):
            seen.append((source, parent))
            entered.set()
            release.wait(5)
            return {'status': 'complete', 'destination': str(parent / 'restored-fixture'),
                    'usable': True, 'copied_files': 0, 'copied_bytes': 0}

        with patch('corridorkey.cache_backup.restore_backup', side_effect=restore):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            self.panel.source_path.setText(str(self.root))
            release.set()
            self.finish()
        self.assertEqual(seen, [(self.cache, self.destination)])
        self.assertFalse(self.panel.use_button.isEnabled())
        self.assertEqual(self.panel.result_path.text(), str(self.destination / 'restored-fixture'))

    def test_symlink_source_refuses_before_destination_creation(self):
        alias = self.root / 'cache-link'
        alias.symlink_to(self.cache, target_is_directory=True)
        self.panel.source_path.setText(str(alias))
        self.panel.start()
        self.finish()
        self.assertIn('Backup failed', self.panel.status.text())
        self.assertEqual(self.panel.result_path.text(), '')
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertTrue(alias.is_symlink())
