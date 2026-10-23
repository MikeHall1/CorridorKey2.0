"""Bounded read-only history paging stays stable while new work is admitted."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from corridorkey.queue_store import QueueStore
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire


class QueuePagingTests(unittest.TestCase):
    def test_pages_cover_all_records_with_equal_times_and_concurrent_insertions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = QueueStore.create(Path(folder))
            store = QueueStore(root)
            try:
                store.start_session()
                wire = job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))
                for index in range(270):
                    store.prepare(f'job{index}', 'render', {'job': wire})
                assert store.db is not None
                with store.db:
                    store.db.execute('UPDATE requests SET created=1')
                page = QueueStore.inspect(root)
                self.assertEqual(len(page['rows']), 128)
                found = [row['id'] for row in page['rows']]
                self.assertEqual(found[0], 'job269')
                store.prepare('newer', 'render', {'job': wire})
                while page['next_before'] is not None:
                    page = QueueStore.inspect(root, before=page['next_before'])
                    self.assertLessEqual(len(page['rows']), 128)
                    found.extend(row['id'] for row in page['rows'])
                self.assertEqual(len(found), 270)
                self.assertEqual(len(set(found)), 270)
                self.assertEqual(found[-1], 'job0')
                self.assertEqual(QueueStore.inspect(root)['rows'][0]['id'], 'newer')
                self.assertEqual(QueueStore.inspect(root, request_id='job0')['status'], 'prepared')
            finally:
                store.close()
            with sqlite3.connect(root / 'queue.sqlite') as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM requests').fetchone()[0], 271)

    def test_invalid_cursors_and_detail_combination_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = QueueStore.create(Path(folder))
            for cursor in (True, 0, -1, 2**63, '128', 1.5):
                with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                    QueueStore.inspect(root, before=cast(Any, cursor))
            with self.assertRaises(ValueError):
                QueueStore.inspect(root, request_id='a', before=1)
            empty = QueueStore.inspect(root)
            self.assertEqual(empty['rows'], [])
            self.assertIsNone(empty['next_before'])
