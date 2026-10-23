"""Sealed managed history stays readable and never becomes a writable queue."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.queue_store import QueueStore
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire


class QueueArchiveTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = QueueStore.create(Path(self.folder.name))
        self.store = QueueStore(self.root)
        self.store.start_session()
        self.store.prepare('uncertain', 'render', {'job': job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))})

    def tearDown(self):
        if self.store is not None:
            self.store.close()
        self.folder.cleanup()

    def stop(self):
        assert self.store is not None
        self.store.finish(True, None)
        self.store.close()
        self.store = None

    def test_live_owner_refused_then_seal_preserves_database_and_unresolved_history(self):
        with self.assertRaises(sqlite3.OperationalError):
            QueueStore.seal(self.root)
        self.assertFalse((self.root / 'queue-archive.json').exists())
        self.stop()
        before = hashlib.sha256((self.root / 'queue.sqlite').read_bytes()).hexdigest()
        marker = QueueStore.seal(self.root)
        self.assertEqual(QueueStore.inspect(self.root)['archive'], marker)
        self.assertEqual(QueueStore.inspect(self.root, request_id='uncertain')['status'], 'prepared')
        self.assertEqual(hashlib.sha256((self.root / 'queue.sqlite').read_bytes()).hexdigest(), before)
        for operation in (QueueStore, QueueStore.seal):
            with self.assertRaisesRegex(ValueError, 'archived'):
                operation(self.root)
        self.assertFalse(list(self.root.glob('.archive-*')))

    def test_publication_failure_keeps_history_writable_and_does_not_remove_user_files(self):
        self.stop()
        important = self.root / 'important'
        important.write_bytes(b'keep')
        with patch('corridorkey.queue_store.os.link', side_effect=OSError('fixture link failure')), self.assertRaises(OSError):
            QueueStore.seal(self.root)
        self.assertFalse((self.root / 'queue-archive.json').exists())
        self.assertFalse(list(self.root.glob('.archive-*')))
        self.assertEqual(important.read_bytes(), b'keep')
        writer = QueueStore(self.root)
        writer.close()

    def test_archive_checked_after_lock_and_invalid_markers_fail_closed(self):
        self.stop()
        connect = sqlite3.connect
        triggered = False

        def interleaved(path, *args, **kwargs):
            nonlocal triggered
            if path == self.root / 'owner-lock.sqlite' and not triggered:
                triggered = True
                with patch('corridorkey.queue_store.sqlite3.connect', connect):
                    QueueStore.seal(self.root)
            return connect(path, *args, **kwargs)

        with patch('corridorkey.queue_store.sqlite3.connect', side_effect=interleaved), self.assertRaisesRegex(ValueError, 'archived'):
            QueueStore(self.root)
        marker = self.root / 'queue-archive.json'
        data = json.loads(marker.read_text())
        data['queue_id'] = 'another_queue'
        marker.write_text(json.dumps(data))
        for operation in (QueueStore, QueueStore.inspect):
            with self.assertRaisesRegex(ValueError, 'archive marker'):
                operation(self.root)

    def test_existing_unrecognized_archive_file_never_overwritten(self):
        self.stop()
        marker = self.root / 'queue-archive.json'
        marker.write_bytes(b'user-owned-content')
        with self.assertRaises(ValueError):
            QueueStore.seal(self.root)
        self.assertEqual(marker.read_bytes(), b'user-owned-content')

    def test_recovery_snapshots_remain_verifiable_and_explicitly_load_into_fresh_queue(self):
        parent = Path(self.folder.name).resolve()
        source = parent / 'source.json'
        source.write_text('{}')
        batch = parent / 'batch.json'
        batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
            {'input': str(source), 'output_dir': str(parent / 'output'), 'settings': {'output_format': 'exr'}}]}))
        assert self.store is not None
        payload = self.store.prepare('recover', 'recover', {'batch': str(batch), 'workspace': str(parent / 'cache'), 'max_cache_bytes': 0})
        original = Path(payload['batch']).read_bytes()
        self.stop()
        QueueStore.seal(self.root)
        batch.write_text('{"not":"the frozen job"}')
        saved = QueueStore.inspect(self.root, request_id='recover')
        self.assertEqual(Path(saved['request']['payload']['batch']).read_bytes(), original)
        fresh_root = QueueStore.create(parent)
        fresh = QueueStore(fresh_root)
        try:
            fresh.start_session()
            copied = fresh.prepare('explicit_resume', 'recover', {**saved['request']['payload'], 'resume': True})
            self.assertEqual(Path(copied['batch']).read_bytes(), original)
        finally:
            fresh.close()
        self.assertEqual(Path(payload['batch']).read_bytes(), original)
        self.assertFalse((parent / 'output').exists())

    def test_process_exit_after_atomic_publication_keeps_complete_readonly_archive(self):
        import subprocess
        import sys
        self.stop()
        code = '''import os,sys
from pathlib import Path
from corridorkey.queue_store import QueueStore
link=os.link
def publish(source,target):
    link(source,target)
    os._exit(82)
os.link=publish
QueueStore.seal(Path(sys.argv[1]))
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.root)], capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 82)
        self.assertIsNotNone(QueueStore.inspect(self.root)['archive'])
        with self.assertRaisesRegex(ValueError, 'archived'):
            QueueStore(self.root)
        self.assertEqual(QueueStore.inspect(self.root, request_id='uncertain')['status'], 'prepared')

    def test_publication_never_replaces_a_concurrently_created_marker(self):
        self.stop()
        import os
        link = os.link
        marker = self.root / 'queue-archive.json'

        def racing_marker(source, target):
            marker.write_bytes(b'concurrent user content')
            link(source, target)

        with patch('corridorkey.queue_store.os.link', side_effect=racing_marker), self.assertRaises(FileExistsError):
            QueueStore.seal(self.root)
        self.assertEqual(marker.read_bytes(), b'concurrent user content')
        self.assertFalse(list(self.root.glob('.archive-*')))
