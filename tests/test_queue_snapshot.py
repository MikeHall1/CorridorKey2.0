import json
import tempfile
import unittest
from pathlib import Path

from corridorkey.batch import load_jobs
from corridorkey.queue_store import QueueStore
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire


class QueueSnapshotTests(unittest.TestCase):
    def test_batch_snapshot_checksum_and_relative_path_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'source.json').write_text('{}')
            batch = root / 'batch.json'
            batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch',
                      'jobs': [{'input': 'source.json', 'output_dir': 'out', 'settings': {'output_format': 'exr'}}]}))
            workspace = QueueStore.create(root)
            store = QueueStore(workspace)
            try:
                store.start_session()
                payload = store.prepare('recover', 'recover', {'batch': str(batch), 'workspace': str(root / 'cache'), 'max_cache_bytes': 0})
            finally:
                store.close()
            snapshot = Path(payload['batch'])
            jobs, _ = load_jobs(snapshot)
            self.assertEqual(jobs[0].input_path, str((root / 'source.json').resolve()))
            snapshot.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'snapshot checksum'):
                QueueStore.inspect(workspace, request_id='recover')

    def test_queue_source_separation_and_strict_headless_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = QueueStore.create(root)
            store = QueueStore(workspace)
            try:
                store.start_session()
                job = JobRequest.create(root, root / 'out')
                with self.assertRaisesRegex(ValueError, 'separate'):
                    store.prepare('nested', 'render', {'job': job_to_wire(job)})
            finally:
                store.close()
            source = root / 'source'
            source.touch()
            batch = root / 'batch.json'
            for raw in ['{"schema_version":true,"kind":"inference_batch","jobs":[]}',
                        '{"schema_version":1,"schema_version":1,"kind":"inference_batch","jobs":[]}']:
                batch.write_text(raw)
                with self.assertRaises(ValueError):
                    load_jobs(batch)
