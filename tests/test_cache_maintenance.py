"""Retention exercises real owned bundles; source exports/journals stay local."""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey import cache_maintenance as maintenance
from corridorkey.control import CancelledError
from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recovery_store import BundleStore


class CacheMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.workspace = self.root / 'cache'
        self.source = self.root / 'original.exr'
        self.source.write_bytes(b'original source export' * 100)
        self.keys = []
        with BundleStore(self.workspace, 100000) as store:
            self.owner = store.owner
            for index in range(3):
                identity = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': index}))
                store.publish(identity, {'alpha/000001.exr': self.source}, {'status': 'complete'})
                self.keys.append(identity.key)
        # Deterministic lookup-recency ordering, not lexical content-key order.
        for index, key in enumerate(self.keys):
            os.utime(self.workspace / 'entries' / key, ns=(1000000000 + index, 1000000000 + index))
        journals = self.workspace / 'journals'
        journals.mkdir()
        (journals / ('a' * 64 + '.json')).write_bytes(b'preserved journal')
        self.stage = self.workspace / 'staging' / f'entry-{self.owner}-crashfixture'
        self.stage.mkdir()
        (self.stage / '.record-incomplete').write_bytes(b'preserved partial record')

    def fingerprint(self):
        return {str(path.relative_to(self.root)): (path.stat().st_mtime_ns, path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
                for path in [self.root, *self.root.rglob('*')]}

    def test_inventory_and_preview_are_read_only_and_do_not_prune(self):
        before = self.fingerprint()
        current = maintenance.inventory(self.workspace)
        preview = maintenance.plan_trim(self.workspace, 0)
        self.assertEqual(self.fingerprint(), before)
        self.assertEqual(current['entry_count'], 3)
        self.assertEqual(current['staging_bytes'], len(b'preserved partial record'))
        self.assertEqual(current['journal_bytes'], len(b'preserved journal'))
        self.assertEqual(preview['remove_count'], 3)
        self.assertEqual([item['key'] for item in preview['remove']], self.keys)
        self.assertEqual(preview['after_bytes'], 0)
        self.assertIn('not measured', current['allocation'])

    def test_trim_real_bundles_to_budget_preserves_exports_journals_and_staging(self):
        current = maintenance.inventory(self.workspace)
        oldest = next(item for item in current['entries'] if item['key'] == self.keys[0])
        target = current['sealed_bytes'] - oldest['bytes']
        preview = maintenance.plan_trim(self.workspace, target)
        source = self.source.read_bytes()
        result = maintenance.apply_trim(self.workspace, preview)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['removed'], self.keys[:1])
        self.assertEqual(result['remaining_sealed_bytes'], target)
        self.assertEqual(maintenance.inventory(self.workspace)['sealed_bytes'], target)
        self.assertEqual(self.source.read_bytes(), source)
        self.assertTrue((self.stage / '.record-incomplete').exists())
        self.assertEqual((self.workspace / 'journals' / ('a' * 64 + '.json')).read_bytes(), b'preserved journal')

    def test_missing_workspace_or_lock_never_creates_anything(self):
        missing = self.root / 'missing'
        with self.assertRaises(FileNotFoundError):
            maintenance.inventory(missing)
        self.assertFalse(missing.exists())
        (self.workspace / 'lock.sqlite').unlink()
        before = self.fingerprint()
        with self.assertRaises(FileNotFoundError):
            maintenance.inventory(self.workspace)
        self.assertEqual(self.fingerprint(), before)

    def test_busy_real_writer_refuses_inventory_preview_and_apply(self):
        preview = maintenance.plan_trim(self.workspace, 0)
        connection = sqlite3.connect(self.workspace / 'lock.sqlite', timeout=0)
        self.addCleanup(connection.close)
        connection.execute('BEGIN EXCLUSIVE')
        for operation in [lambda: maintenance.inventory(self.workspace),
                          lambda: maintenance.plan_trim(self.workspace, 0),
                          lambda: maintenance.apply_trim(self.workspace, preview)]:
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                operation()
        connection.rollback()
        self.assertEqual(maintenance.inventory(self.workspace)['entry_count'], 3)

    def test_readonly_inspection_holds_exclusive_lock_without_writing(self):
        observed = []

        def check():
            if observed:
                return
            connection = sqlite3.connect(self.workspace / 'lock.sqlite', timeout=0)
            try:
                connection.execute('BEGIN EXCLUSIVE')
            except sqlite3.OperationalError:
                observed.append(True)
            else:
                connection.rollback()
            finally:
                connection.close()

        before = self.fingerprint()
        maintenance.inventory(self.workspace, check=check)
        self.assertEqual(observed, [True])
        self.assertEqual(self.fingerprint(), before)

    def test_wal_lock_refuses_without_creating_auxiliary_files(self):
        connection = sqlite3.connect(self.workspace / 'lock.sqlite')
        connection.execute('PRAGMA journal_mode=WAL')
        connection.close()
        before = self.fingerprint()
        with self.assertRaisesRegex(ValueError, 'WAL unsupported'):
            maintenance.inventory(self.workspace)
        self.assertEqual(self.fingerprint(), before)

    def test_stale_or_edited_preview_never_deletes(self):
        preview = maintenance.plan_trim(self.workspace, 0)
        preview['remove'][0]['key'] = self.keys[1]
        before = self.fingerprint()
        with self.assertRaisesRegex(ValueError, 'stale or changed'):
            maintenance.apply_trim(self.workspace, preview)
        self.assertEqual(self.fingerprint(), before)
        preview = maintenance.plan_trim(self.workspace, 0)
        os.utime(self.workspace / 'entries' / self.keys[0], None)
        before = self.fingerprint()
        with self.assertRaisesRegex(ValueError, 'stale or changed'):
            maintenance.apply_trim(self.workspace, preview)
        self.assertEqual(self.fingerprint(), before)

    def test_unowned_unexpected_symlink_and_hardlink_contents_fail_before_deletion(self):
        cases = ['unknown', 'symlink', 'hardlink', 'directory', 'ownership']
        for case in cases:
            with self.subTest(case=case):
                original = b''
                entry = self.workspace / 'entries' / self.keys[2]
                target = entry / 'payload-999999.bin'
                if case == 'unknown':
                    target = entry / 'user-footage.exr'
                    target.write_bytes(b'preserve')
                elif case == 'symlink':
                    target.symlink_to(self.source)
                elif case == 'hardlink':
                    os.link(self.source, target)
                elif case == 'directory':
                    target.mkdir()
                else:
                    target = entry / 'owner.json'
                    original = target.read_bytes()
                    data = json.loads(original)
                    data['owner'] = '0' * 32
                    target.write_text(json.dumps(data))
                before = self.fingerprint()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    maintenance.plan_trim(self.workspace, 0)
                self.assertEqual(self.fingerprint(), before)
                if case == 'directory':
                    target.rmdir()
                elif case == 'ownership':
                    target.write_bytes(original)
                else:
                    target.unlink()

    def test_metadata_and_entry_bounds_refuse_without_mutation(self):
        before = self.fingerprint()
        with patch.object(maintenance, 'MAX_ENTRIES', 2), self.assertRaisesRegex(ValueError, 'bounded'):
            maintenance.inventory(self.workspace)
        with patch.object(maintenance, 'MAX_RECORD_BYTES', 16), self.assertRaisesRegex(ValueError, 'bounded'):
            maintenance.inventory(self.workspace)
        self.assertEqual(self.fingerprint(), before)

    def test_cancellation_after_one_committed_entry_reports_truth(self):
        preview = maintenance.plan_trim(self.workspace, 0)

        def check():
            if not (self.workspace / 'entries' / self.keys[0]).exists():
                raise CancelledError('fixture cancelled after first commit')

        result = maintenance.apply_trim(self.workspace, preview, check=check)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['removed'], self.keys[:1])
        self.assertEqual(result['removed_bytes'], preview['remove'][0]['bytes'])
        self.assertEqual(result['remaining_sealed_bytes'], maintenance.inventory(self.workspace)['sealed_bytes'])
        self.assertIsNone(result['partial_entry'])

    def test_unexpected_check_failure_keeps_committed_removal_report(self):
        preview = maintenance.plan_trim(self.workspace, 0)

        def check():
            if not (self.workspace / 'entries' / self.keys[0]).exists():
                raise RuntimeError('fixture callback failure after first commit')

        with self.assertLogs('corridorkey.cache_maintenance', level='ERROR'):
            result = maintenance.apply_trim(self.workspace, preview, check=check)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['removed'], self.keys[:1])
        self.assertEqual(result['removed_bytes'], preview['remove'][0]['bytes'])
        self.assertEqual(result['remaining_entry_bytes'], maintenance.inventory(self.workspace)['sealed_bytes'])

    def test_actual_recovery_report_layout_is_inspected_preserved_and_bound_to_preview(self):
        reports = self.workspace / 'reports'
        reports.mkdir()
        report = reports / ('run-' + 'b' * 32 + '.json')
        report.write_text(json.dumps({'schema_version': 1, 'kind': 'recoverable_exr_batch', 'status': 'complete'}))
        before = self.fingerprint()
        current = maintenance.inventory(self.workspace)
        self.assertEqual(current['report_count'], 1)
        self.assertEqual(current['report_bytes'], report.stat().st_size)
        preview = maintenance.plan_trim(self.workspace, 0)
        self.assertEqual(self.fingerprint(), before)
        report.write_text(report.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'stale or changed'):
            maintenance.apply_trim(self.workspace, preview)
        preview = maintenance.plan_trim(self.workspace, 0)
        self.assertEqual(maintenance.apply_trim(self.workspace, preview)['status'], 'complete')
        self.assertTrue(report.exists())
        self.assertEqual(maintenance.inventory(self.workspace)['report_count'], 1)

    def test_unlink_failure_reports_actual_partial_bytes_and_does_not_continue(self):
        preview = maintenance.plan_trim(self.workspace, 0)
        original = Path.unlink

        def failed(path, *args, **kwargs):
            if path.parent.name == self.keys[0] and path.name == 'complete.json':
                raise PermissionError('fixture deletion denied')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'unlink', failed):
            result = maintenance.apply_trim(self.workspace, preview)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['removed'], [])
        self.assertEqual(result['removed_bytes'], self.source.stat().st_size)
        self.assertEqual(result['partial_entry'], self.keys[0])
        self.assertIsNone(result['remaining_sealed_bytes'])
        actual = sum(path.stat().st_size for path in (self.workspace / 'entries').rglob('*') if path.is_file())
        self.assertEqual(result['remaining_entry_bytes'], actual)
        self.assertTrue((self.workspace / 'entries' / self.keys[1] / 'complete.json').exists())
        self.assertTrue(self.source.exists())

    def test_interrupt_after_actual_unlink_reports_committed_bytes_without_rollback(self):
        preview = maintenance.plan_trim(self.workspace, 0)
        original = Path.unlink

        def interrupted(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.parent.name == self.keys[0] and path.name == 'payload-000000.bin':
                raise KeyboardInterrupt('fixture interrupt after OS committed unlink')
            return result

        with patch.object(Path, 'unlink', interrupted), self.assertLogs('corridorkey.cache_maintenance', level='ERROR'):
            result = maintenance.apply_trim(self.workspace, preview)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['removed_bytes'], self.source.stat().st_size)
        self.assertEqual(result['removed'], [])
        self.assertEqual(result['partial_entry'], self.keys[0])
        actual = sum(path.stat().st_size for path in (self.workspace / 'entries').rglob('*') if path.is_file())
        self.assertEqual(result['remaining_entry_bytes'], actual)
        self.assertIsNone(result['remaining_sealed_bytes'])

    def test_interrupt_after_actual_rmdir_counts_the_completed_removal(self):
        preview = maintenance.plan_trim(self.workspace, 0)
        original = Path.rmdir

        def interrupted(path):
            original(path)
            if path.name == self.keys[0]:
                raise KeyboardInterrupt('fixture interrupt after OS committed rmdir')

        with patch.object(Path, 'rmdir', interrupted), self.assertLogs('corridorkey.cache_maintenance', level='ERROR'):
            result = maintenance.apply_trim(self.workspace, preview)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['removed'], self.keys[:1])
        self.assertEqual(result['removed_bytes'], preview['remove'][0]['bytes'])
        self.assertEqual(result['remaining_entry_count'], 2)

    def test_noop_plan_and_late_cancellation_after_last_commit(self):
        current = maintenance.inventory(self.workspace)
        preview = maintenance.plan_trim(self.workspace, current['sealed_bytes'])
        self.assertEqual(maintenance.apply_trim(self.workspace, preview)['removed'], [])
        preview = maintenance.plan_trim(self.workspace, 0)

        def check():
            if not any((self.workspace / 'entries').iterdir()):
                raise CancelledError('late cancellation must not hide committed success')

        result = maintenance.apply_trim(self.workspace, preview, check=check)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['removed'], self.keys)
        self.assertEqual(result['remaining_sealed_bytes'], 0)

    def test_import_is_stdlib_only_no_background_or_model_loading(self):
        code = "import sys,threading; import corridorkey.cache_maintenance; print('torch' in sys.modules,'numpy' in sys.modules,threading.active_count())"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(result.stdout.strip(), 'False False 1')

    def test_cli_plan_and_apply_use_real_preview_without_creating_other_paths(self):
        command = [sys.executable, '-m', 'corridorkey.cache_maintenance']
        inspected = subprocess.run([*command, 'inspect', '--workspace', str(self.workspace)],
                                   capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(json.loads(inspected.stdout)['entry_count'], 3)
        planned = subprocess.run([*command, 'plan', '--workspace', str(self.workspace), '--target-bytes', '0'],
                                 capture_output=True, text=True, timeout=10, check=True)
        plan_path = self.root / 'reviewed-plan.json'
        plan_path.write_text(planned.stdout)
        applied = subprocess.run([*command, 'apply', '--workspace', str(self.workspace), '--plan', str(plan_path)],
                                 capture_output=True, text=True, timeout=10, check=True)
        result = json.loads(applied.stdout)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['remaining_entry_count'], 0)
        self.assertEqual(result['removed'], self.keys)
        self.assertTrue(self.source.exists())


if __name__ == '__main__':
    unittest.main()
