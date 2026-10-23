"""SQLite admission precedes Python TEXT materialization for queue details."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.queue_store import QueueStore
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import (
    MAX_FRAME_BYTES,
    encode_frame,
    job_from_wire,
    job_to_wire,
)


class QueueDetailBoundsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.parent = Path(self.folder.name).resolve()
        self.root = QueueStore.create(self.parent)
        self.job = JobRequest.create(self.parent / 'source.mp4', self.parent / 'output',
                                     InferenceConfig(temp_dir=self.parent / 'temp'))
        store = QueueStore(self.root)
        try:
            store.start_session()
            store.prepare('job', 'render', {'job': job_to_wire(self.job)})
        finally:
            store.close()
        self.original = (self.root / 'queue.sqlite').read_bytes()

    def update(self, field, value):
        self.assertIn(field, {'request', 'event', 'request_hash', 'event_hash', 'status', 'session', 'created'})
        with closing(sqlite3.connect(self.root / 'queue.sqlite')) as db, db:
            db.execute(f'UPDATE requests SET {field}=? WHERE id=?', (value, 'job'))

    @contextmanager
    def admitted_text_only(self, *, read_only=True):
        connect = sqlite3.connect
        seen = []
        def bounded_text(raw):
            self.assertLessEqual(len(raw), MAX_FRAME_BYTES, 'Oversized SQLite TEXT reached the Python adapter')
            seen.append(len(raw))
            return raw.decode('utf-8')
        def read_only_connect(*args, **kwargs):
            if read_only:
                self.assertTrue(kwargs.get('uri'))
                self.assertIn('mode=ro', str(args[0]))
            db = connect(*args, **kwargs)
            db.text_factory = bounded_text
            return db
        with patch('corridorkey.queue_store.sqlite3.connect', side_effect=read_only_connect):
            yield seen

    def test_oversized_request_and_event_never_cross_text_adapter(self):
        for field in ('request', 'event'):
            with self.subTest(field=field):
                self.update(field, 'x' * (MAX_FRAME_BYTES + 1))
                before = (self.root / 'queue.sqlite').read_bytes()
                with self.admitted_text_only(), self.assertRaisesRegex(ValueError, 'size'):
                    QueueStore.inspect(self.root, request_id='job')
                self.assertEqual((self.root / 'queue.sqlite').read_bytes(), before)
                self.update(field, None if field == 'event' else json.dumps({}))

    def test_utf8_bytes_not_characters_are_admitted_and_blobs_are_refused(self):
        for value in ('é' * (MAX_FRAME_BYTES // 2 + 1), b'x' * 32):
            with self.subTest(kind=type(value).__name__):
                self.update('request', value)
                with self.admitted_text_only(), self.assertRaisesRegex(ValueError, 'size|type'):
                    QueueStore.inspect(self.root, request_id='job')

    def test_multibyte_event_exact_byte_boundary_and_one_character_over(self):
        event: dict[str, Any] = {'kind': 'result', 'request_id': 'job', 'payload': {'note': ''}}
        overhead = len(json.dumps(event, ensure_ascii=False, separators=(',', ':')).encode())
        available = MAX_FRAME_BYTES - 1 - overhead
        note = 'é' * (available // 2) + ('x' if available % 2 else '')
        event['payload']['note'] = note
        text = json.dumps(event, ensure_ascii=False, separators=(',', ':'))
        self.assertEqual(len(text.encode()), MAX_FRAME_BYTES - 1)
        self.update('status', 'result')
        self.update('event', text)
        self.update('event_hash', hashlib.sha256(text.encode()).hexdigest())
        with self.admitted_text_only():
            record = QueueStore.inspect(self.root, request_id='job')
        self.assertEqual(record['event']['payload']['note'], note)
        event['payload']['note'] += 'é'
        self.update('event', json.dumps(event, ensure_ascii=False, separators=(',', ':')))
        with self.admitted_text_only(), self.assertRaisesRegex(ValueError, 'size'):
            QueueStore.inspect(self.root, request_id='job')

    def test_utf16_history_refused_before_reading_payload_columns(self):
        # SQLite supports UTF-16 databases, but app-owned queue writers always
        # create UTF-8. Clone only this tiny fixture with a different encoding.
        connect = sqlite3.connect
        with closing(connect(self.root / 'queue.sqlite')) as db:
            dump = '\n'.join(db.iterdump())
        replacement = self.root / 'utf16.sqlite'
        with closing(connect(replacement)) as db:
            db.execute("PRAGMA encoding='UTF-16le'")
            db.executescript(dump)
        replacement.replace(self.root / 'queue.sqlite')
        before = (self.root / 'queue.sqlite').read_bytes()
        payload_reads = []
        def authorize(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ and table == 'requests' and column in {'request', 'event'}:
                payload_reads.append(column)
            return sqlite3.SQLITE_OK
        def read_only_connect(*args, **kwargs):
            db = connect(*args, **kwargs)
            db.set_authorizer(authorize)
            return db
        with patch('corridorkey.queue_store.sqlite3.connect', side_effect=read_only_connect):
            for request_id in (None, 'job'):
                with self.subTest(request_id=request_id), self.assertRaisesRegex(ValueError, 'UTF-8'):
                    QueueStore.inspect(self.root, request_id=request_id)
        self.assertFalse(payload_reads)
        self.assertEqual((self.root / 'queue.sqlite').read_bytes(), before)

    def test_oversized_hash_and_other_metadata_never_cross_adapter(self):
        for field in ('request_hash', 'session', 'created'):
            with self.subTest(field=field):
                # Replace only the chosen malformed column in an otherwise
                # legitimate row so payload handling cannot hide the failure.
                (self.root / 'queue.sqlite').write_bytes(self.original)
                self.update(field, 'x' * (MAX_FRAME_BYTES + 1))
                with self.admitted_text_only(), self.assertRaisesRegex(ValueError, 'size|type'):
                    QueueStore.inspect(self.root, request_id='job')

    def test_history_status_and_owner_metadata_admitted_before_python_text(self):
        self.update('status', 'x' * (MAX_FRAME_BYTES * 2))
        before = (self.root / 'queue.sqlite').read_bytes()
        with self.admitted_text_only(), self.assertRaisesRegex(ValueError, 'size'):
            QueueStore.inspect(self.root)
        self.assertEqual((self.root / 'queue.sqlite').read_bytes(), before)
        (self.root / 'queue.sqlite').write_bytes(self.original)
        with closing(sqlite3.connect(self.root / 'queue.sqlite')) as db, db:
            db.execute('UPDATE metadata SET queue_id=?', ('x' * (MAX_FRAME_BYTES * 2),))
        before = (self.root / 'queue.sqlite').read_bytes()
        for writer in (False, True):
            with self.subTest(writer=writer), self.admitted_text_only(read_only=not writer), \
                    self.assertRaisesRegex(ValueError, 'size'):
                if writer:
                    with closing(QueueStore(self.root)):
                        pass
                else:
                    QueueStore.inspect(self.root)
            self.assertEqual((self.root / 'queue.sqlite').read_bytes(), before)

    def test_duplicate_metadata_fetch_is_bounded_to_two_rows(self):
        with closing(sqlite3.connect(self.root / 'queue.sqlite')) as db, db:
            queue_id = db.execute('SELECT queue_id FROM metadata').fetchone()[0]
            db.executemany('INSERT INTO metadata VALUES (?,1)', [(queue_id,)] * 1000)
        before = (self.root / 'queue.sqlite').read_bytes()
        with self.admitted_text_only() as seen, self.assertRaisesRegex(ValueError, 'ownership/version'):
            QueueStore.inspect(self.root)
        # Count actual ownership values delivered by the SQLite text adapter,
        # not a particular SQL spelling or an estimate of peak memory.
        self.assertEqual(sum(size == len(queue_id.encode()) for size in seen), 2)
        self.assertEqual((self.root / 'queue.sqlite').read_bytes(), before)

    def test_history_paging_keeps_128_rows_and_exact_cursor_coverage(self):
        store = QueueStore(self.root)
        try:
            store.start_session()
            for index in range(130):
                store.prepare(f'job{index:03}', 'render', {'job': job_to_wire(self.job)})
        finally:
            store.close()
        before = (self.root / 'queue.sqlite').read_bytes()
        first = QueueStore.inspect(self.root)
        second = QueueStore.inspect(self.root, before=first['next_before'])
        self.assertEqual(len(first['rows']), 128)
        self.assertEqual(len(second['rows']), 3)
        ids = [row['id'] for page in (first, second) for row in page['rows']]
        self.assertEqual(len(set(ids)), 131)
        self.assertEqual(set(ids), {'job', *(f'job{index:03}' for index in range(130))})
        self.assertIsNone(second['next_before'])
        self.assertTrue(all(not key.startswith('_') and key != 'ordinal'
                            for page in (first, second) for row in page['rows'] for key in row))
        self.assertEqual((self.root / 'queue.sqlite').read_bytes(), before)

    def test_nullable_event_and_checksum_validation_remain_read_only(self):
        before = (self.root / 'queue.sqlite').read_bytes()
        with self.admitted_text_only():
            record = QueueStore.inspect(self.root, request_id='job')
        self.assertIsNone(record['event'])
        self.assertEqual(record['status'], 'prepared')
        self.assertFalse(any(key.startswith('_') for key in record))
        self.assertEqual(job_from_wire(record['request']['payload']['job']), self.job)
        self.assertEqual((self.root / 'queue.sqlite').read_bytes(), before)
        self.update('request_hash', '0' * 64)
        with self.admitted_text_only(), self.assertRaisesRegex(ValueError, 'checksum'):
            QueueStore.inspect(self.root, request_id='job')

    def test_individually_near_limit_payloads_in_one_large_row_load_settings(self):
        # A valid strict worker request can contain a large codec string; this
        # test loads it as settings only and never launches an encoder/job.
        cfg, _ = self.job.configurations()
        cfg.temp_dir = cfg.temp_dir.resolve()
        codec = 'x' * (MAX_FRAME_BYTES - 20000)
        job = JobRequest.create(self.job.input_path, self.job.output_dir, cfg, WriterConfig(codec=codec))
        request: dict[str, Any] = {'schema_version': 1, 'request_id': 'job', 'action': 'render',
                   'payload': {'job': job_to_wire(job)}}
        request_text = encode_frame(request).decode().rstrip('\n')
        codec += 'x' * (MAX_FRAME_BYTES - 1 - len(request_text.encode()))
        job = JobRequest.create(self.job.input_path, self.job.output_dir, cfg, WriterConfig(codec=codec))
        request['payload']['job'] = job_to_wire(job)
        request_text = encode_frame(request).decode().rstrip('\n')
        event: dict[str, Any] = {'kind': 'result', 'request_id': 'job', 'payload': {'record': {'status': 'complete', 'note': ''}}}
        overhead = len(encode_frame(event)) - 1
        event['payload']['record']['note'] = 'y' * (MAX_FRAME_BYTES - 1 - overhead)
        event_text = encode_frame(event).decode().rstrip('\n')
        self.assertEqual(len(request_text.encode()), MAX_FRAME_BYTES - 1)
        self.assertEqual(len(event_text.encode()), MAX_FRAME_BYTES - 1)
        self.assertGreater(len(request_text.encode()) + len(event_text.encode()), MAX_FRAME_BYTES)
        with closing(sqlite3.connect(self.root / 'queue.sqlite')) as db, db:
            db.execute('UPDATE requests SET request=?,request_hash=?,status=?,event=?,event_hash=? WHERE id=?',
                       (request_text, hashlib.sha256(request_text.encode()).hexdigest(), 'result', event_text,
                        hashlib.sha256(event_text.encode()).hexdigest(), 'job'))
        before = (self.root / 'queue.sqlite').read_bytes()
        with self.admitted_text_only() as seen:
            record = QueueStore.inspect(self.root, request_id='job')
        self.assertGreater(sum(size > MAX_FRAME_BYTES // 2 for size in seen), 1)
        loaded = job_from_wire(record['request']['payload']['job'])
        self.assertEqual(loaded, job)
        self.assertEqual(loaded.configurations()[1].codec, codec)
        self.assertEqual(record['event']['payload']['record']['status'], 'complete')
        self.assertEqual((self.root / 'queue.sqlite').read_bytes(), before)
