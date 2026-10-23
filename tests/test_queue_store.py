import sqlite3
import tempfile
import unittest
from pathlib import Path

from corridorkey.config import InferenceConfig
from corridorkey.queue_store import QueueStore
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire


class QueueStoreTests(unittest.TestCase):
    def test_single_owner_and_reopened_history_never_infers_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = QueueStore.create(Path(folder))
            store = QueueStore(root)
            try:
                store.start_session()
                with self.assertRaises(sqlite3.OperationalError):
                    QueueStore(root)
                job = JobRequest.create('/tmp/source', '/tmp/output', InferenceConfig(output_format='exr'))
                store.prepare('job', 'render', {'job': job_to_wire(job)})
                store.connected(12345, 'a' * 32)
                store.phase('job', 'dispatching')
                store.observe({'kind': 'accepted', 'request_id': 'job', 'payload': {}})
            finally:
                store.close()
            snapshot = QueueStore.inspect(root)
            self.assertEqual(snapshot['rows'][0]['status'], 'accepted')
            self.assertEqual(snapshot['rows'][0]['closed'], 0)
            self.assertIn('NOT VERIFIED', snapshot['scope'])
            restored = QueueStore(root)
            restored.close()

    def test_terminal_records_are_preserved_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = QueueStore.create(Path(folder))
            store = QueueStore(root)
            try:
                store.start_session()
                store.prepare('job', 'render', {'job': job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))})
                store.observe({'kind': 'result', 'request_id': 'job', 'payload': {'record': {'status': 'complete'}}})
                store.observe({'kind': 'progress', 'request_id': 'job', 'payload': {}})
                store.finish(True, None)
            finally:
                store.close()
            saved = QueueStore.inspect(root, request_id='job')
            self.assertEqual(saved['status'], 'result')
            self.assertEqual(saved['event']['payload']['record']['status'], 'complete')
            with sqlite3.connect(root / 'queue.sqlite') as db:
                db.execute("UPDATE requests SET event='{}' WHERE id='job'")
            with self.assertRaisesRegex(ValueError, 'checksum'):
                QueueStore.inspect(root, request_id='job')

    def test_unowned_workspace_does_not_modify_user_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'important'
            source.write_bytes(b'keep')
            with self.assertRaises(ValueError):
                QueueStore(root)
            self.assertEqual(source.read_bytes(), b'keep')
            self.assertFalse((root / 'queue.sqlite').exists())

    def test_terminal_phase_cannot_be_replaced_or_metadata_tampered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = QueueStore.create(Path(folder))
            store = QueueStore(root)
            try:
                store.start_session()
                store.prepare('job', 'render', {'job': job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))})
                store.observe({'kind': 'result', 'request_id': 'job', 'payload': {}})
                with self.assertRaises(ValueError):
                    store.phase('job', 'sent')
            finally:
                store.close()
            with sqlite3.connect(root / 'queue.sqlite') as db:
                db.execute("UPDATE requests SET status='failed' WHERE id='job'")
            with self.assertRaisesRegex(ValueError, 'metadata'):
                QueueStore.inspect(root, request_id='job')

    def test_marker_raw_limit_and_actual_process_crash_release_lock(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as folder:
            root = QueueStore.create(Path(folder))
            code = '''import os,sys
from pathlib import Path
from corridorkey.queue_store import QueueStore
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire
s=QueueStore(Path(sys.argv[1]));s.start_session()
s.prepare('crashed','render',{'job':job_to_wire(JobRequest.create('/tmp/source','/tmp/output'))})
s.observe({'kind':'accepted','request_id':'crashed','payload':{}})
os._exit(79)
'''
            process = subprocess.run([sys.executable, '-c', code, str(root)], check=False, timeout=10, capture_output=True)
            self.assertEqual(process.returncode, 79)
            history = QueueStore.inspect(root)
            self.assertEqual(history['rows'][0]['status'], 'accepted')
            self.assertEqual(history['rows'][0]['closed'], 0)
            store = QueueStore(root)
            store.close()
            marker = root / 'queue-owner.json'
            with marker.open('ab') as stream:
                stream.write(b'\n' * (4 * 1024 * 1024))
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                QueueStore.inspect(root)
