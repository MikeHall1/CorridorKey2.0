"""Actual Qt cache maintenance, immutable previews and owned shutdown."""
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
class CachePresentationTests(unittest.TestCase):
    def test_human_summary_preserves_exact_bytes_and_marks_partial_unknown(self):
        from corridorkey.desktop_cache import report_summary

        exact = 9223372036854775807
        preview = report_summary('preview', {'target_bytes': exact, 'before_bytes': exact,
                                            'after_bytes': exact, 'remove_count': 0, 'remove_bytes': 0,
                                            'token': 'internal-snapshot-secret', 'owner': 'internal-owner'})
        self.assertIn(f'Retain at most: {exact} logical bytes', preview)
        self.assertIn('Proposed removal: 0 bundles · 0 bytes', preview)
        self.assertNotIn('internal-', preview)
        result = report_summary('apply', {'status': 'failed', 'removed': ['first'],
                                         'removed_bytes': None, 'remaining_sealed_bytes': None,
                                         'remaining_entry_bytes': 123, 'remaining_entry_count': 2,
                                         'partial_entry': 'second', 'error': 'interrupted deletion'})
        self.assertIn('Committed removals: 1 bundles · Unknown bytes', result)
        self.assertIn('Retained sealed bytes: Unknown', result)
        self.assertIn('including partial bundles: 123', result)
        self.assertIn('Partially removed bundle: second', result)
        self.assertIn('Error: interrupted deletion', result)
        self.assertNotIn('None', result)
        inventory = report_summary('inventory', {'entry_count': True, 'sealed_bytes': -1})
        self.assertIn('Sealed bundles: Unknown', inventory)
        self.assertIn('Preserved reports:', inventory)


@unittest.skipUnless(QT_AVAILABLE, 'Optional requirements-desktop.txt is not installed')
class DesktopCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['cache-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_cache import CacheMaintenancePanel
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore

        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.exr'
        self.source.write_bytes(b'original fixture bytes' * 20)
        self.cache = self.root / 'cache'
        with BundleStore(self.cache, 100000) as store:
            for name in ('one', 'two'):
                identity = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': name}))
                store.publish(identity, {'alpha.exr': self.source}, {'status': 'complete'})
        self.panel = CacheMaintenancePanel()
        self.panel.set_workspace(str(self.cache))
        self.panel.show()
        self.app.processEvents()

    def tearDown(self):
        self.panel.close()
        if self.panel.task is not None:
            self.panel.task.thread.join(5)
        self.panel.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.panel)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.panel.poll()
            if predicate():
                return
            time.sleep(0.005)
        self.fail('Cache panel condition not reached')

    def finish(self):
        self.until(lambda: not self.panel.io_busy)

    def test_real_readonly_inventory_preview_and_explicit_apply_preserve_source(self):
        before = {str(path.relative_to(self.cache)): path.read_bytes() for path in self.cache.rglob('*') if path.is_file()}
        self.assertIsNone(self.panel.task)
        self.assertFalse(self.panel.apply_button.isEnabled())
        self.panel.start('inventory')
        self.finish()
        self.assertEqual(self.panel.table.rowCount(), 2)
        self.assertIn('2 sealed bundles', self.panel.status.text())
        self.assertEqual(before, {str(path.relative_to(self.cache)): path.read_bytes() for path in self.cache.rglob('*') if path.is_file()})
        self.panel.target.setText('0')
        self.panel.start('preview')
        self.finish()
        self.assertEqual(len(list((self.cache / 'entries').iterdir())), 2)
        self.assertTrue(self.panel.apply_button.isEnabled())
        self.assertIn('Proposed removal: 2 bundles', self.panel.details.toPlainText())
        self.panel.start('apply')
        self.finish()
        self.assertEqual(list((self.cache / 'entries').iterdir()), [])
        self.assertIn('Apply complete: 2 bundles removed', self.panel.status.text())
        self.assertEqual(self.source.read_bytes(), b'original fixture bytes' * 20)
        self.assertFalse(self.panel.apply_button.isEnabled())

    def test_changed_target_or_workspace_invalidates_exact_preview(self):
        self.panel.start('preview')
        self.finish()
        self.assertIsNotNone(self.panel.plan)
        self.panel.target.setText('0')
        self.assertIsNone(self.panel.plan)
        self.panel.start('apply')
        self.assertIn('Preview the exact trim', self.panel.status.text())
        self.assertEqual(len(list((self.cache / 'entries').iterdir())), 2)
        self.panel.start('preview')
        self.finish()
        self.panel.set_workspace(str(self.root))
        self.assertIsNone(self.panel.plan)
        self.panel.target.setText('-1')
        self.panel.start('preview')
        self.assertIn('decimal integer', self.panel.status.text())

    def test_close_waits_unexpected_worker_exit_and_io_is_background_owned(self):
        entered, release = threading.Event(), threading.Event()
        worker_names = []

        def broken_inventory(root, *, check=None):
            worker_names.append(threading.current_thread().name)
            entered.set()
            release.wait(5)
            raise SystemExit('injected unexpected worker exit')

        with patch('corridorkey.cache_maintenance.inventory', side_effect=broken_inventory):
            self.panel.start('inventory')
            self.assertTrue(entered.wait(2))
            assert self.panel.task is not None
            original_task = self.panel.task
            self.panel.start('preview')
            self.assertIs(self.panel.task, original_task)
            self.panel.close()
            self.assertTrue(self.panel.isVisible())
            self.assertTrue(self.panel.io_busy)
            self.assertTrue(original_task.cancellation.cancelled)
            release.set()
            self.until(lambda: not self.panel.isVisible())
        self.assertTrue(original_task.done.is_set())
        self.assertFalse(original_task.thread.is_alive())
        self.assertIn('SystemExit', self.panel.status.text())
        self.assertEqual(worker_names, ['cache-maintenance'])
        self.assertEqual(original_task.results.maxsize, 1)

    def test_partial_apply_report_remains_authoritative_after_cancel(self):
        self.panel.target.setText('0')
        self.panel.start('preview')
        self.finish()
        entered, release = threading.Event(), threading.Event()

        def partial_apply(root, plan, *, check=None):
            entered.set()
            release.wait(5)
            return {'status': 'cancelled', 'workspace': str(root), 'removed': ['committed-key'],
                    'removed_bytes': 123, 'remaining_sealed_bytes': 456,
                    'partial_entry': None, 'error': 'cancel after first committed removal'}

        with patch('corridorkey.cache_maintenance.apply_trim', side_effect=partial_apply):
            self.panel.start('apply')
            self.assertTrue(entered.wait(2))
            self.panel.cancel()
            release.set()
            self.finish()
        self.assertIn('Apply cancelled: 1 bundles removed, 123 bytes removed; 456', self.panel.status.text())
        self.assertIn('cancel after first committed removal', self.panel.status.text())
        self.assertIsNone(self.panel.plan)

    def test_programmatic_edit_during_io_cannot_reenable_stale_preview(self):
        from corridorkey.cache_maintenance import plan_trim
        entered, release = threading.Event(), threading.Event()

        def delayed_preview(root, target_bytes, *, check=None):
            report = plan_trim(root, target_bytes, check=check)
            entered.set()
            release.wait(5)
            return report

        with patch('corridorkey.cache_maintenance.plan_trim', side_effect=delayed_preview):
            self.panel.start('preview')
            self.assertTrue(entered.wait(2))
            self.panel.target.setText('0')
            release.set()
            self.finish()
        self.assertIsNone(self.panel.plan)
        self.assertIn('Preview invalidated', self.panel.status.text())

    def test_busy_workspace_and_stale_apply_refuse_without_deleting(self):
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore

        with BundleStore(self.cache, 100000):
            self.panel.start('inventory')
            self.finish()
            self.assertIn('failed:', self.panel.status.text())
            self.assertIn('busy', self.panel.status.text())
        self.panel.target.setText('0')
        self.panel.start('preview')
        self.finish()
        with BundleStore(self.cache, 100000) as store:
            identity = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'third'}))
            store.publish(identity, {'alpha.exr': self.source}, {'status': 'complete'})
        self.panel.start('apply')
        self.finish()
        self.assertIn('failed:', self.panel.status.text())
        self.assertEqual(len(list((self.cache / 'entries').iterdir())), 3)
        self.assertFalse(self.panel.apply_button.isEnabled())

    def test_all_candidates_are_accessible_in_bounded_pages(self):
        report = {'entries': [{'key': f'{index:064x}', 'bytes': index, 'mtime_ns': index}
                              for index in range(205)], 'entry_count': 205, 'sealed_bytes': 20910}
        with patch('corridorkey.cache_maintenance.inventory', return_value=report):
            self.panel.start('inventory')
            self.finish()
        self.assertEqual(self.panel.table.rowCount(), 100)
        first = self.panel.table.item(0, 0)
        assert first is not None
        self.assertEqual(first.text(), f'{0:064x}')
        self.panel.next_button.click()
        second = self.panel.table.item(0, 0)
        assert second is not None
        self.assertEqual(second.text(), f'{100:064x}')
        self.panel.next_button.click()
        self.assertEqual(self.panel.table.rowCount(), 5)
        last = self.panel.table.item(4, 0)
        assert last is not None
        self.assertEqual(last.text(), f'{204:064x}')
        self.assertFalse(self.panel.next_button.isEnabled())

    def test_symlink_root_is_not_normalized_past_core_admission(self):
        alias = self.root / 'cache-link'
        alias.symlink_to(self.cache, target_is_directory=True)
        self.panel.set_workspace(str(alias))
        self.panel.start('inventory')
        self.finish()
        self.assertIn('failed:', self.panel.status.text())
        self.assertEqual(self.panel.table.rowCount(), 0)
        self.assertEqual(len(list((self.cache / 'entries').iterdir())), 2)
        self.assertTrue(alias.is_symlink())
