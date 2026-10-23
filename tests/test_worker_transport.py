import io
import os
import queue
import threading
import unittest

from corridorkey.worker_protocol import parse_event
from corridorkey.worker_transport import Outbox, _read_input


class TransportTests(unittest.TestCase):
    def test_progress_coalesces_and_terminal_records_have_contiguous_sequences(self):
        stream = io.BytesIO()
        box = Outbox(stream)
        box.emit('ready', None, {'capacity': 4})
        for frame in range(10000):
            box.emit('progress', 'job', {'stage': 'frames', 'completed': frame, 'total': 10000})
        self.assertEqual(len(box.progress), 1)
        self.assertLess(box.bytes_pending, 2000)
        box.emit('result', 'job', {'record': {'status': 'complete'}})
        self.assertEqual(box.progress, {})
        box.start()
        self.assertTrue(box.close())
        events = [parse_event(line) for line in stream.getvalue().splitlines(keepends=True)]
        self.assertEqual([event['kind'] for event in events], ['ready', 'result'])
        self.assertEqual([event['sequence'] for event in events], [0, 1])

    def test_partial_writes_preserve_frames_and_disconnect_is_explicit(self):
        class Partial(io.BytesIO):
            def write(self, value):
                return super().write(value[:3])

        stream = Partial()
        box = Outbox(stream)
        box.emit('ready', None, {'note': 'Ελλάδα'})
        box.start()
        self.assertTrue(box.close())
        self.assertEqual(parse_event(stream.getvalue())['payload']['note'], 'Ελλάδα')
        stream.close()
        broken = Outbox(stream)
        broken.emit('ready', None, {})
        broken.start()
        self.assertTrue(broken.failed.wait(2))
        self.assertFalse(broken.close())

    def test_critical_backpressure_fails_without_unbounded_growth(self):
        box = Outbox(io.BytesIO(), max_events=2)
        for name in ('a', 'b'):
            box.emit('accepted', name, {})
        with self.assertRaises(BufferError):
            box.emit('result', 'c', {})
        self.assertEqual(len(box.critical), 2)
        self.assertTrue(box.failed.is_set())
        box.start()
        self.assertFalse(box.close())

    def test_fragmented_unicode_and_truncated_input_use_actual_pipes(self):
        for data, expected in (('{"text":"Ελλάδα"}\n'.encode(), 'frame'), (b'{}', 'error')):
            with self.subTest(expected=expected):
                read_fd, write_fd = os.pipe()
                messages = queue.Queue(maxsize=8)
                stop = threading.Event()
                reader = threading.Thread(target=_read_input, args=(read_fd, messages, stop))
                reader.start()
                try:
                    for byte in data:
                        os.write(write_fd, bytes([byte]))
                    os.close(write_fd)
                    write_fd = None
                    kind, value = messages.get(timeout=2)
                    self.assertEqual(kind, expected)
                    if kind == 'frame':
                        self.assertEqual(value, data)
                    self.assertEqual(messages.get(timeout=2), ('eof', None))
                finally:
                    stop.set()
                    if write_fd is not None:
                        os.close(write_fd)
                    reader.join(2)
                    os.close(read_fd)
