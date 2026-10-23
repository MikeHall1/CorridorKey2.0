import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from corridorkey import queue_transfer as transfer
from corridorkey.batch import load_jobs
from corridorkey.control import CancelledError
from corridorkey.queue_store import QueueStore
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire


class QueueTransferTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.json'
        self.source.write_text('{}')
        self.batch = self.root / 'batch.json'
        self.batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
            {'input': str(self.source), 'output_dir': str(self.root / 'render-output'), 'settings': {'output_format': 'exr'}}]}))
        self.queue = QueueStore.create(self.root)
        self.store = QueueStore(self.queue)
        self.store.start_session()
        self.store.prepare('render', 'render', {'job': job_to_wire(JobRequest.create(self.source, self.root / 'single-output'))})
        self.store.observe({'kind': 'result', 'request_id': 'render', 'payload': {'record': {'status': 'complete'}}})
        self.store.prepare('recover', 'recover', {'batch': str(self.batch), 'workspace': str(self.root / 'recovery-cache'), 'max_cache_bytes': 0})
        self.store.observe({'kind': 'accepted', 'request_id': 'recover', 'payload': {}})
        self.store.finish(True, None)
        self.store.close()
        QueueStore.seal(self.queue)
        self.parent = self.root / 'transfers'
        self.parent.mkdir()

    @staticmethod
    def fingerprint(root):
        return {str(path.relative_to(root)): (path.stat().st_mtime_ns, path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)
                for path in [root, *root.rglob('*')]}

    def export(self):
        result = transfer.export_history(self.queue, self.parent)
        self.assertEqual(result['status'], 'complete', result)
        self.assertTrue(result['usable'])
        return Path(result['destination'])

    def test_real_mixed_history_moves_and_loads_fresh_queue_without_modifying_original_records(self):
        before = self.fingerprint(self.queue)
        original = QueueStore.inspect(self.queue, request_id='recover')
        exported = self.export()
        self.assertEqual(self.fingerprint(self.queue), before)
        self.assertEqual((self.queue / 'owner-lock.sqlite').stat().st_size, 0)
        export_before = self.fingerprint(exported)
        imported = transfer.import_history(exported, self.parent)
        self.assertEqual(imported['status'], 'complete', imported)
        imported_root = Path(imported['destination'])
        self.assertEqual((imported_root / 'queue.sqlite').read_bytes(), (self.queue / 'queue.sqlite').read_bytes())
        self.assertEqual(self.fingerprint(exported), export_before)
        shutil.rmtree(self.queue)
        snapshot = QueueStore.inspect(imported_root)
        self.assertEqual(len(snapshot['rows']), 2)
        self.assertIsNotNone(snapshot['archive'])
        detail = QueueStore.inspect(imported_root, request_id='recover')
        self.assertEqual(detail['request'], original['request'])
        self.assertEqual(detail['event'], original['event'])
        self.assertEqual(detail['request_hash'], original['request_hash'])
        local = Path(detail['load_payload']['batch'])
        self.assertEqual(local, imported_root / 'batches/recover.json')
        self.assertTrue(local.is_file())
        self.assertEqual(load_jobs(local)[0][0].input_path, str(self.source.resolve()))
        with self.assertRaisesRegex(ValueError, 'read-only'):
            QueueStore(imported_root)
        fresh = QueueStore.create(self.parent)
        active = QueueStore(fresh)
        try:
            active.start_session()
            prepared = active.prepare('fresh-recovery', 'recover', detail['load_payload'])
        finally:
            active.close()
        self.assertEqual(Path(prepared['batch']), fresh / 'batches/fresh-recovery.json')
        self.assertEqual(detail['request'], original['request'])

    def test_reexport_import_preserves_original_path_after_all_previous_roots_removed(self):
        original = QueueStore.inspect(self.queue, request_id='recover')['request']
        first = self.export()
        first_root = Path(transfer.import_history(first, self.parent)['destination'])
        second = transfer.export_history(first_root, self.parent)
        self.assertEqual(second['status'], 'complete', second)
        second_root = Path(transfer.import_history(second['destination'], self.parent)['destination'])
        shutil.rmtree(self.queue)
        shutil.rmtree(first)
        shutil.rmtree(first_root)
        shutil.rmtree(second['destination'])
        detail = QueueStore.inspect(second_root, request_id='recover')
        self.assertEqual(detail['request'], original)
        self.assertTrue(Path(detail['load_payload']['batch']).is_file())

    def test_all_pages_and_details_are_exported_not_only_first128(self):
        queue = QueueStore.create(self.root)
        store = QueueStore(queue)
        try:
            store.start_session()
            for index in range(130):
                store.prepare(f'job-{index}', 'render', {'job': job_to_wire(JobRequest.create(self.source, self.root / 'out'))})
        finally:
            store.close()
        QueueStore.seal(queue)
        exported = transfer.export_history(queue, self.parent)
        self.assertEqual(exported['request_count'], 130)
        imported = transfer.import_history(exported['destination'], self.parent)
        root = Path(imported['destination'])
        first = QueueStore.inspect(root)
        second = QueueStore.inspect(root, before=first['next_before'])
        self.assertEqual(len(first['rows']) + len(second['rows']), 130)
        self.assertEqual(QueueStore.inspect(root, request_id='job-0')['status'], 'prepared')

    def test_active_busy_missing_wal_and_unknown_contents_refuse_without_creating(self):
        active_root = QueueStore.create(self.root)
        refused = transfer.export_history(active_root, self.parent)
        self.assertEqual(refused['status'], 'failed')
        self.assertIsNone(refused['destination'])
        connection = sqlite3.connect(self.queue / 'owner-lock.sqlite')
        connection.execute('BEGIN EXCLUSIVE')
        try:
            refused = transfer.export_history(self.queue, self.parent)
        finally:
            connection.rollback()
            connection.close()
        self.assertEqual(refused['status'], 'failed')
        self.assertIsNone(refused['destination'])
        unknown = self.queue / 'unknown.bin'
        unknown.write_bytes(b'preserve')
        self.assertFalse(transfer.export_history(self.queue, self.parent)['usable'])
        unknown.unlink()
        with sqlite3.connect(self.queue / 'queue.sqlite') as db:
            db.execute('PRAGMA journal_mode=WAL')
        self.assertFalse(transfer.export_history(self.queue, self.parent)['usable'])
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_imported_history_refuses_write_when_archive_removed_and_requires_local_batch(self):
        imported = transfer.import_history(self.export(), self.parent)
        root = Path(imported['destination'])
        (root / 'queue-archive.json').unlink()
        with self.assertRaisesRegex(ValueError, 'read-only'):
            QueueStore(root)
        with self.assertRaisesRegex(ValueError, 'relocation'):
            QueueStore.inspect(root)

    def test_local_snapshot_tamper_symlink_directory_and_request_hash_tamper_refuse(self):
        for kind in ('snapshot', 'batch-directory', 'hash'):
            with self.subTest(kind=kind):
                imported = transfer.import_history(self.export(), self.parent)
                root = Path(imported['destination'])
                if kind == 'snapshot':
                    (root / 'batches/recover.json').write_bytes(b'{}')
                elif kind == 'batch-directory':
                    shutil.rmtree(root / 'batches')
                    (root / 'batches').symlink_to(self.queue / 'batches', target_is_directory=True)
                else:
                    with sqlite3.connect(root / 'queue.sqlite') as db:
                        db.execute("UPDATE requests SET request='{}' WHERE id='recover'")
                with self.assertRaises(ValueError):
                    QueueStore.inspect(root, request_id='recover')

    def test_imported_root_batch_parent_and_leaf_reparse_attributes_are_refused(self):
        imported = transfer.import_history(self.export(), self.parent)
        root = Path(imported['destination'])
        original = Path.lstat
        for target in (root, root / 'batches', root / 'batches/recover.json'):
            with self.subTest(target=target.name):
                def attributes(path, *, marked=target):
                    info = original(path)
                    if path == marked:
                        return SimpleNamespace(st_mode=info.st_mode, st_nlink=info.st_nlink,
                                               st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
                    return info

                with patch.object(Path, 'lstat', attributes), self.assertRaisesRegex(ValueError, 'reparse'):
                    QueueStore.inspect(root, request_id='recover')

    def test_original_destination_paths_are_protected_before_mkdir(self):
        output = self.root / 'single-output'
        output.mkdir()
        before = self.fingerprint(output)
        rejected = transfer.export_history(self.queue, output)
        self.assertIsNone(rejected['destination'])
        self.assertEqual(self.fingerprint(output), before)
        exported = self.export()
        rejected = transfer.import_history(exported, output)
        self.assertIsNone(rejected['destination'])
        self.assertEqual(self.fingerprint(output), before)

    def test_corrupt_extra_missing_link_foreign_and_path_tamper_refuse_before_import(self):
        for kind in ('corrupt', 'extra', 'missing', 'symlink', 'hardlink', 'foreign', 'path'):
            with self.subTest(kind=kind):
                exported = self.export()
                manifest = exported / 'queue-export.json'
                envelope = json.loads(manifest.read_text())
                blob = exported / envelope['payload']['files'][0]['blob']
                if kind == 'corrupt':
                    blob.write_bytes(b'x' * blob.stat().st_size)
                elif kind == 'extra':
                    (exported / 'unknown.bin').write_bytes(b'preserved')
                elif kind == 'missing':
                    blob.unlink()
                elif kind in ('symlink', 'hardlink'):
                    blob.unlink()
                    if kind == 'symlink':
                        blob.symlink_to(self.source)
                    else:
                        os.link(self.source, blob)
                else:
                    if kind == 'foreign':
                        envelope['payload']['path_flavour'] = 'windows' if transfer.FLAVOUR == 'posix' else 'posix'
                    else:
                        envelope['payload']['files'][0]['name'] = '../outside'
                    envelope['sha256'] = transfer.maintenance._digest(envelope['payload'])
                    manifest.write_bytes(transfer.maintenance._canonical(envelope))
                result = transfer.import_history(exported, self.parent)
                self.assertFalse(result['usable'])
                self.assertIsNone(result['destination'])

    def test_cancel_disk_failure_and_late_commit_are_truthful(self):
        original = transfer.copying._copy

        def copying(source, destination, expected, result, check):
            original(source, destination, expected, result, check)
            raise CancelledError('fixture interrupted after one copied file')

        with patch.object(transfer.copying, '_copy', copying):
            cancelled = transfer.export_history(self.queue, self.parent)
        self.assertEqual(cancelled['status'], 'cancelled')
        partial = Path(cancelled['destination'])
        self.assertFalse((partial / 'queue-export.json').exists())
        self.assertEqual(cancelled['copied_bytes'], sum(path.stat().st_size for path in partial.iterdir()))
        exported = self.export()
        with patch.object(transfer.copying, '_copy', side_effect=OSError('fixture disk full')):
            failed = transfer.import_history(exported, self.parent)
        self.assertFalse(failed['usable'])
        partial = Path(failed['destination'])
        self.assertFalse((partial / 'queue-owner.json').exists())
        with self.assertRaises((ValueError, FileNotFoundError)):
            QueueStore(partial)
        seal = transfer.copying._seal

        def sealed(path, raw):
            seal(path, raw)
            if path.name == 'queue-owner.json':
                raise KeyboardInterrupt('fixture interrupt after atomic commit')

        with patch.object(transfer.copying, '_seal', sealed):
            committed = transfer.import_history(exported, self.parent)
        self.assertTrue(committed['usable'], committed)
        self.assertEqual(committed['status'], 'complete')
        self.assertEqual(QueueStore.inspect(Path(committed['destination']))['rows'][0]['id'], 'recover')

    def test_cli_roundtrip(self):
        command = [sys.executable, '-m', 'corridorkey.queue_transfer']
        exported = subprocess.run([*command, 'export', '--workspace', str(self.queue), '--output-parent', str(self.parent)],
                                  capture_output=True, text=True, timeout=10, check=True)
        path = json.loads(exported.stdout)['destination']
        inspected = subprocess.run([*command, 'inspect', '--archive', path], capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(json.loads(inspected.stdout)['request_count'], 2)
        imported = subprocess.run([*command, 'import', '--archive', path, '--output-parent', str(self.parent)],
                                  capture_output=True, text=True, timeout=10, check=True)
        self.assertTrue(json.loads(imported.stdout)['usable'])


if __name__ == '__main__':
    unittest.main()
