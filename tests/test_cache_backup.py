import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

from corridorkey import cache_backup as transfer
from corridorkey import cache_maintenance as maintenance
from corridorkey.control import CancelledError
from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recovery_store import BundleStore
from tests import test_recovery_runner as recovery_fixtures


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.exr'
        self.source.write_bytes(b'preserve original export' * 500)
        self.workspace = self.root / 'workspace'
        self.identity = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': True}))
        with BundleStore(self.workspace, 100000) as store:
            self.owner = store.owner
            store.publish(self.identity, {'alpha/one.exr': self.source}, {'status': 'complete'})
        self.entry = self.workspace / 'entries' / self.identity.key
        os.utime(self.entry, ns=(1000000001, 1000000001))
        stage = self.workspace / 'staging' / f'entry-{self.owner}-partial'
        stage.mkdir()
        (stage / '.record-incomplete').write_bytes(b'{partial preserved bytes')
        for directory, name in [('journals', 'a' * 64 + '.json'), ('reports', 'run-' + 'b' * 32 + '.json')]:
            path = self.workspace / directory
            path.mkdir()
            (path / name).write_bytes(b'historical absolute references remain unchanged')
        self.parent = self.root / 'destinations'
        self.parent.mkdir()

    @staticmethod
    def fingerprint(root):
        return {str(path.relative_to(root)): (path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
                for path in [root, *root.rglob('*')]}

    def backup(self):
        result = transfer.backup_workspace(self.workspace, self.parent)
        self.assertEqual(result['status'], 'complete', result)
        self.assertTrue(result['usable'])
        return Path(result['destination'])

    def test_roundtrip_exact_bytes_recency_and_fresh_lock_without_source_changes(self):
        before = self.fingerprint(self.workspace)
        source = self.source.read_bytes()
        backup = self.backup()
        backup_before = self.fingerprint(backup)
        result = transfer.restore_backup(backup, self.parent)
        self.assertEqual(result['status'], 'complete', result)
        restored = Path(result['destination'])
        for name, value in before.items():
            if name != 'lock.sqlite':
                self.assertEqual(self.fingerprint(restored)[name], value, name)
        self.assertNotEqual((restored / 'lock.sqlite').stat().st_ino, (self.workspace / 'lock.sqlite').stat().st_ino)
        self.assertEqual(self.fingerprint(self.workspace), before)
        self.assertEqual(self.fingerprint(backup), backup_before)
        self.assertEqual(self.source.read_bytes(), source)
        self.assertEqual(maintenance.inventory(restored)['entry_count'], 1)
        self.assertEqual(transfer.inspect_backup(backup)['status'], 'complete')

    def test_windows_relative_keys_backup_restore_preserve_posix_manifest_and_real_files(self):
        relative_to = Path.relative_to

        def windows_relative_to(path, *args, **kwargs):
            return PureWindowsPath(relative_to(path, *args, **kwargs).as_posix())

        before = self.fingerprint(self.workspace)
        with patch.object(Path, 'relative_to', windows_relative_to):
            backup = self.backup()
            payload = json.loads((backup / 'backup.json').read_bytes())['payload']
            self.assertEqual(payload['schema_version'], 1)
            self.assertIn(f'entries/{self.identity.key}', [item['name'] for item in payload['directories']])
            self.assertIn(f'entries/{self.identity.key}/payload-000000.bin', [item['name'] for item in payload['files']])
            self.assertTrue(all('\\' not in item['name'] for group in ('directories', 'files') for item in payload[group]))
            self.assertEqual(transfer.inspect_backup(backup)['status'], 'complete')
            result = transfer.restore_backup(backup, self.parent)
            self.assertEqual(result['status'], 'complete', result)
            self.assertTrue(result['usable'])
            self.assertEqual(maintenance.inventory(result['destination'])['entry_count'], 1)
        restored = self.fingerprint(Path(result['destination']))
        for name, value in before.items():
            if name != 'lock.sqlite':
                self.assertEqual(restored[name], value, name)
        self.assertEqual(self.fingerprint(self.workspace), before)
        self.assertEqual(self.source.read_bytes(), b'preserve original export' * 500)

    def test_busy_source_and_missing_parent_do_not_create_destination(self):
        connection = sqlite3.connect(self.workspace / 'lock.sqlite', timeout=0)
        connection.execute('BEGIN EXCLUSIVE')
        try:
            result = transfer.backup_workspace(self.workspace, self.parent)
        finally:
            connection.rollback()
            connection.close()
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['destination'])
        self.assertEqual(list(self.parent.iterdir()), [])
        result = transfer.backup_workspace(self.workspace, self.root / 'missing-parent')
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((self.root / 'missing-parent').exists())

    def test_parent_inside_source_is_refused_and_each_run_uses_new_destination(self):
        result = transfer.backup_workspace(self.workspace, self.workspace / 'staging')
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['destination'])
        first = self.backup()
        second = self.backup()
        self.assertNotEqual(first, second)

    def test_corrupt_sealed_source_payload_is_not_blessed_as_backup(self):
        payload = self.entry / 'payload-000000.bin'
        raw = payload.read_bytes()
        payload.write_bytes(b'x' * len(raw))
        result = transfer.backup_workspace(self.workspace, self.parent)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['usable'])
        self.assertFalse((Path(result['destination']) / 'backup.json').exists())
        self.assertIn('original completion record', result['error'])

    def test_corrupt_missing_extra_symlink_and_hardlink_backup_refuse_before_restore_destination(self):
        for kind in ('corrupt', 'missing', 'extra', 'symlink', 'hardlink'):
            with self.subTest(kind=kind):
                backup = self.backup()
                envelope = json.loads((backup / 'backup.json').read_text())
                blob = backup / envelope['payload']['files'][0]['blob']
                if kind == 'corrupt':
                    blob.write_bytes(b'x' * blob.stat().st_size)
                elif kind == 'missing':
                    blob.unlink()
                elif kind == 'extra':
                    (backup / 'unknown.bin').write_bytes(b'preserve')
                else:
                    blob.unlink()
                    if kind == 'symlink':
                        blob.symlink_to(self.source)
                    else:
                        os.link(self.source, blob)
                result = transfer.restore_backup(backup, self.parent)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['usable'])
                self.assertIsNone(result['destination'])

    def test_untrusted_manifest_paths_duplicate_keys_and_bad_ownership_are_refused(self):
        for kind in ('traversal', 'absolute', 'duplicate', 'ownership'):
            with self.subTest(kind=kind):
                backup = self.backup()
                marker = backup / 'backup.json'
                envelope = json.loads(marker.read_text())
                if kind in ('traversal', 'absolute'):
                    envelope['payload']['files'][0]['name'] = '../outside' if kind == 'traversal' else '/outside'
                elif kind == 'ownership':
                    envelope['payload']['owner'] = '0' * 32
                if kind == 'duplicate':
                    marker.write_text('{"payload":{},' + marker.read_text()[1:])
                else:
                    envelope['sha256'] = maintenance._digest(envelope['payload'])
                    marker.write_bytes(maintenance._canonical(envelope))
                result = transfer.restore_backup(backup, self.parent)
                self.assertFalse(result['usable'])
                self.assertIsNone(result['destination'])
        self.assertFalse((self.root / 'outside').exists())

    def test_cancelled_partial_backup_never_seals_and_reports_retained_destination(self):
        def check():
            destinations = list(self.parent.glob('recovery-backup-*'))
            if destinations and any(destinations[0].glob('blob-*')):
                raise CancelledError('fixture cancellation during copy')

        result = transfer.backup_workspace(self.workspace, self.parent, check=check)
        self.assertEqual(result['status'], 'cancelled')
        self.assertFalse(result['usable'])
        partial = Path(result['destination'])
        self.assertTrue(partial.is_dir())
        self.assertFalse((partial / 'backup.json').exists())
        self.assertEqual(result['copied_bytes'], sum(path.stat().st_size for path in partial.iterdir()))

    def test_cancelled_restore_never_publishes_workspace_marker(self):
        backup = self.backup()

        def check():
            destinations = list(self.parent.glob('recovery-restored-*'))
            if destinations and (destinations[0] / 'lock.sqlite').exists():
                raise CancelledError('fixture cancellation during restore')

        result = transfer.restore_backup(backup, self.parent, check=check)
        self.assertEqual(result['status'], 'cancelled')
        partial = Path(result['destination'])
        self.assertFalse((partial / 'workspace.json').exists())
        self.assertFalse(result['usable'])
        with self.assertRaisesRegex(ValueError, 'empty or already owned'):
            BundleStore(partial, 100000)

    def test_snapshot_changes_during_backup_and_restore_refuse_seal(self):
        original = transfer._copy
        changed = []

        def copying(source, destination, expected, result, check):
            digest = original(source, destination, expected, result, check)
            if not changed:
                os.utime(self.entry, None)
                changed.append(True)
            return digest

        with patch.object(transfer, '_copy', copying):
            result = transfer.backup_workspace(self.workspace, self.parent)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('snapshot changed', result['error'])
        backup = self.backup()
        changed.clear()

        def copying_restore(source, destination, expected, result, check):
            digest = original(source, destination, expected, result, check)
            if not changed:
                os.utime(backup, None)
                changed.append(True)
            return digest

        with patch.object(transfer, '_copy', copying_restore):
            result = transfer.restore_backup(backup, self.parent)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((Path(result['destination']) / 'workspace.json').exists())

    def test_disk_failure_and_interrupt_report_real_partial_bytes(self):
        original = transfer._copy
        for exception in (OSError('fixture disk full'), KeyboardInterrupt('fixture interrupt')):
            with self.subTest(exception=type(exception).__name__):
                def copying(source, destination, expected, result, check, failure=exception):
                    original(source, destination, expected, result, check)
                    raise failure

                with patch.object(transfer, '_copy', copying):
                    result = transfer.backup_workspace(self.workspace, self.parent)
                self.assertFalse(result['usable'])
                partial = Path(result['destination'])
                self.assertFalse((partial / 'backup.json').exists())
                self.assertEqual(result['copied_bytes'], sum(path.stat().st_size for path in partial.iterdir()))

    def test_growing_source_cannot_write_or_hash_beyond_admitted_size(self):
        source = self.root / 'growing.bin'
        source.write_bytes(b'admitted bytes')
        expected = {'bytes': source.stat().st_size, 'mtime_ns': source.stat().st_mtime_ns}
        changed = []

        def grow():
            if not changed:
                with source.open('ab') as stream:
                    stream.write(b'growth beyond the admitted snapshot' * 100000)
                changed.append(True)

        destination = self.root / 'bounded.bin'
        result = {'copied_bytes': 0, 'copied_files': 0}
        with self.assertRaisesRegex(ValueError, 'grew beyond'):
            transfer._copy(source, destination, expected, result, grow)
        self.assertEqual(destination.stat().st_size, expected['bytes'])
        self.assertEqual(result['copied_bytes'], expected['bytes'])
        source.write_bytes(b'admitted bytes')
        changed.clear()
        with self.assertRaisesRegex(ValueError, 'grew beyond'):
            transfer._hash(source, grow, expected_bytes=expected['bytes'])

    def test_committed_backup_and_restore_remain_usable_after_finalization_failure(self):
        original = transfer._seal

        def seal_then_fail(path, raw):
            original(path, raw)
            raise PermissionError('fixture finalization failure after full marker commit')

        with patch.object(transfer, '_seal', seal_then_fail):
            result = transfer.backup_workspace(self.workspace, self.parent)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['usable'])
        backup = Path(result['destination'])
        self.assertEqual(transfer.inspect_backup(backup)['status'], 'complete')
        self.assertTrue(all(path.stat().st_nlink == 1 for path in backup.iterdir()))
        with patch.object(transfer, '_seal', seal_then_fail):
            result = transfer.restore_backup(backup, self.parent)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['usable'])
        self.assertEqual(maintenance.inventory(result['destination'])['entry_count'], 1)

    def test_late_cancellation_after_seal_does_not_hide_commit(self):
        original = transfer._seal

        def seal_then_cancel(path, raw):
            original(path, raw)
            raise CancelledError('fixture late cancellation')

        with patch.object(transfer, '_seal', seal_then_cancel):
            result = transfer.backup_workspace(self.workspace, self.parent)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['usable'])

    def test_valid_json_prefix_disk_failure_never_publishes_restore_ownership_marker(self):
        ownership = self.workspace / 'workspace.json'
        ownership.write_bytes(ownership.read_bytes() + b' ' * 4096)
        backup = self.backup()
        original = Path.open

        class PartialWriter:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def write(self, raw):
                self.stream.write(raw[:raw.index(b'}') + 1])
                self.stream.flush()
                raise OSError('fixture disk failure after complete JSON prefix')

        def opened(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            return PartialWriter(stream) if path.name.startswith('.seal-') else stream

        with patch.object(Path, 'open', opened):
            result = transfer.restore_backup(backup, self.parent)
        self.assertEqual(result['status'], 'failed')
        partial = Path(result['destination'])
        self.assertFalse((partial / 'workspace.json').exists())
        stage = next(partial.glob('.seal-*'))
        self.assertEqual(json.loads(stage.read_bytes())['owner'], self.owner)
        with self.assertRaisesRegex(ValueError, 'empty or already owned'):
            BundleStore(partial, 100000)

    @unittest.skipUnless(recovery_fixtures.HAS_STACK, 'requires actual CPU EXR controller adapters')
    def test_known_original_export_destinations_are_protected_for_backup_and_restore(self):
        fixture = recovery_fixtures.RecoveryRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        report = fixture.run_jobs()
        export = Path(report['jobs'][0]['outputs']['alpha'])
        before = self.fingerprint(export)
        rejected = transfer.backup_workspace(fixture.workspace, export)
        self.assertEqual(rejected['status'], 'failed')
        self.assertIsNone(rejected['destination'])
        self.assertEqual(self.fingerprint(export), before)
        # Common project ancestor remains usable: the generated child is a sibling.
        backup = transfer.backup_workspace(fixture.workspace, fixture.root)
        self.assertEqual(backup['status'], 'complete', backup)
        rejected = transfer.restore_backup(backup['destination'], export)
        self.assertEqual(rejected['status'], 'failed')
        self.assertIsNone(rejected['destination'])
        self.assertEqual(self.fingerprint(export), before)

    @unittest.skipUnless(recovery_fixtures.HAS_STACK, 'requires actual CPU EXR controller adapters')
    def test_actual_controller_workspace_roundtrip_and_resume_without_new_neural_execution(self):
        fixture = recovery_fixtures.RecoveryRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        report = fixture.run_jobs()
        backup = transfer.backup_workspace(fixture.workspace, self.parent)
        self.assertEqual(backup['status'], 'complete', backup)
        restored = transfer.restore_backup(backup['destination'], self.parent)
        self.assertEqual(restored['status'], 'complete', restored)
        fixture.workspace = Path(restored['destination'])
        resumed = fixture.run_jobs(resume=True)
        self.assertEqual([job['mode'] for job in resumed['jobs']], ['verified_existing', 'verified_existing'])
        self.assertEqual(fixture.loads, 1)
        self.assertEqual(len(fixture.renders), 1)
        self.assertEqual(Path(report['journal']).read_bytes(), Path(resumed['journal']).read_bytes())
        corrupt = Path(resumed['jobs'][1]['outputs']['alpha']) / 'alpha_000000.exr'
        corrupt.write_bytes(b'fixture corrupt existing output; original directory retained')
        recovered = fixture.run_jobs(resume=True)
        self.assertEqual(recovered['jobs'][1]['mode'], 'verified_cache_restore')
        self.assertEqual(len(fixture.renders), 1)
        self.assertEqual(fixture.loads, 1)
        self.assertEqual(corrupt.read_bytes(), b'fixture corrupt existing output; original directory retained')

    def test_cli_and_import_have_no_model_loading_or_background_threads(self):
        code = "import sys,threading; import corridorkey.cache_backup; print('torch' in sys.modules,'numpy' in sys.modules,threading.active_count())"
        imported = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(imported.stdout.strip(), 'False False 1')
        command = [sys.executable, '-m', 'corridorkey.cache_backup']
        created = subprocess.run([*command, 'backup', '--workspace', str(self.workspace), '--output-parent', str(self.parent)],
                                 capture_output=True, text=True, check=True, timeout=10)
        backup = json.loads(created.stdout)['destination']
        inspected = subprocess.run([*command, 'inspect', '--backup', backup], capture_output=True, text=True, check=True, timeout=10)
        self.assertTrue(json.loads(inspected.stdout)['usable'])
        restored = subprocess.run([*command, 'restore', '--backup', backup, '--output-parent', str(self.parent)],
                                  capture_output=True, text=True, check=True, timeout=10)
        self.assertTrue(json.loads(restored.stdout)['usable'])


if __name__ == '__main__':
    unittest.main()
