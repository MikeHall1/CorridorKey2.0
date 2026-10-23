"""Native consumer acknowledgements remain separate from durable checkpoints."""
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from corridorkey.control import CancellationToken, CancelledError
from corridorkey.lifecycle import WriterWorker

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
STACK = all(importlib.util.find_spec(name) for name in ('cv2', 'numpy', 'torch'))


class WriterBarrierTests(unittest.TestCase):
    def close_worker(self, worker, release=None):
        if release is not None:
            release.set()
        try:
            worker.close()
        except (RuntimeError, CancelledError, TimeoutError):
            pass
        worker.thread.join(1)
        self.assertFalse(worker.thread.is_alive())

    def test_acknowledges_only_after_prior_consumption_and_keeps_writer_open(self):
        entered, release, waiting, done = (threading.Event() for _ in range(4))
        consumed, errors = [], []
        def consume(item):
            entered.set()
            release.wait(2)
            consumed.append(item)
        worker = WriterWorker(consume, timeout=1)
        self.addCleanup(self.close_worker, worker, release)
        worker.submit('first')
        self.assertTrue(entered.wait(1))
        def flush():
            waiting.set()
            try:
                worker.flush_barrier()
            except (RuntimeError, CancelledError, TimeoutError) as exc:
                errors.append(exc)
            finally:
                done.set()
        thread = threading.Thread(target=flush)
        thread.start()
        self.assertTrue(waiting.wait(1))
        self.assertFalse(done.wait(.03))
        self.assertEqual(consumed, [])
        release.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        worker.submit('second')
        worker.flush_barrier()
        self.assertEqual(consumed, ['first', 'second'])
        self.assertFalse(worker.closed)

    def test_consumer_failure_cannot_become_successful_queue_drain(self):
        error = RuntimeError('native storage failure')
        def consume(item):
            raise error
        worker = WriterWorker(consume, timeout=1)
        self.addCleanup(self.close_worker, worker)
        worker.submit('fail')
        with self.assertRaises(RuntimeError) as raised:
            worker.flush_barrier()
        self.assertIs(raised.exception, error)

    def test_cancellation_during_native_call_does_not_claim_acknowledgement(self):
        entered, release, waiting = (threading.Event() for _ in range(3))
        token, errors = CancellationToken(), []
        def consume(item):
            entered.set()
            release.wait(2)
        worker = WriterWorker(consume, check=token.check, timeout=1)
        self.addCleanup(self.close_worker, worker, release)
        worker.submit('pending')
        self.assertTrue(entered.wait(1))
        def flush():
            waiting.set()
            try:
                worker.flush_barrier()
            except (RuntimeError, CancelledError, TimeoutError) as exc:
                errors.append(exc)
        thread = threading.Thread(target=flush)
        thread.start()
        self.assertTrue(waiting.wait(1))
        token.cancel('cancel barrier')
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], CancelledError)
        self.assertTrue(worker.stopped.is_set())

    def test_stalled_consumer_times_out_and_closed_worker_refuses_barrier(self):
        entered, release = threading.Event(), threading.Event()
        def consume(item):
            entered.set()
            release.wait(2)
        worker = WriterWorker(consume, timeout=.04)
        self.addCleanup(self.close_worker, worker, release)
        worker.submit('pending')
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(TimeoutError, 'barrier timed out'):
            worker.flush_barrier()
        healthy = WriterWorker(lambda item: None)
        healthy.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            healthy.flush_barrier()


@unittest.skipUnless(STACK, 'Requires actual CPU EXR writer')
class ExrBarrierTests(unittest.TestCase):
    def test_native_suffix_index_and_observations_remain_separate_from_barrier(self):
        import cv2
        import numpy as np

        from corridorkey.video_io import ExrSequenceWriter
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'fg_000000.exr'
            original.write_bytes(b'preserved prefix fixture')
            observed, clocks = [], []
            def clock():
                clocks.append(True)
                return 1.0
            writer = ExrSequenceWriter(folder, 24, (2, 2), dtype='float', frame_name='fg', start_index=1,
                                       on_frame_written=lambda index, stamp: observed.append((index, stamp)), clock=clock)
            try:
                writer.write_many(np.full((2, 2, 2, 3), .25, np.float32))
                self.assertEqual(writer.flush_barrier(), 3)
                self.assertEqual(observed, [(1, 1.), (2, 1.)])
                self.assertEqual(clocks, [True, True])
                self.assertEqual(writer.flush_barrier(), 3)
                self.assertEqual(clocks, [True, True])
                np.testing.assert_array_equal(cv2.imread(str(root / 'fg_000002.exr'), cv2.IMREAD_UNCHANGED),
                                              np.full((2, 2, 3), .25, np.float32))
                self.assertEqual(original.read_bytes(), b'preserved prefix fixture')
            finally:
                writer.release()
            self.assertFalse(writer.worker.thread.is_alive())

    def test_invalid_start_index_refused_before_directory_creation(self):
        from corridorkey.video_io import ExrSequenceWriter
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'not-created'
            for value in (-1, True, 1.5, 2**63):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    ExrSequenceWriter(str(target), 24, (2, 2), start_index=json.loads(json.dumps(value)))
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
