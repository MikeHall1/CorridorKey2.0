import threading
import time
import unittest
from unittest.mock import Mock

from corridorkey.control import CancellationToken, CancelledError, JobControl
from corridorkey.lifecycle import WriterWorker


class ControlTests(unittest.TestCase):
    def test_progress_is_immutable_and_can_cancel_at_checkpoint(self):
        token = CancellationToken()
        events = []

        def progress(event):
            events.append(event)
            if event.completed == 1:
                token.cancel("user pressed stop")

        control = JobControl(token, progress)
        control.checkpoint("frames", 0, 5)
        with self.assertRaisesRegex(CancelledError, "user pressed stop"):
            control.checkpoint("frames", 1, 5)
        self.assertEqual([e.completed for e in events], [0, 1])
        token.cancel("later reason must not replace first")
        with self.assertRaisesRegex(CancelledError, "user pressed stop"):
            control.check()

    def test_cancelled_queue_unblocks_producer_and_aborts_consumer(self):
        entered, resume = threading.Event(), threading.Event()
        token = CancellationToken()

        def consume(item):
            entered.set()
            if not resume.wait(3):
                raise TimeoutError("coordination failed")

        worker = WriterWorker(consume, maxsize=1, check=token.check, abort_consumer=resume.set)
        worker.submit(0)
        self.assertTrue(entered.wait(2))
        worker.submit(1)
        token.cancel()
        with self.assertRaises(CancelledError):
            worker.submit(2)
        with self.assertRaises(CancelledError):
            worker.close()
        self.assertFalse(worker.thread.is_alive())
        self.assertEqual(worker.queue.qsize(), 0)

    def test_stalled_consumer_has_bounded_shutdown(self):
        entered, resume = threading.Event(), threading.Event()

        def consume(item):
            entered.set()
            resume.wait(3)

        worker = WriterWorker(consume, timeout=0.05, abort_consumer=resume.set)
        worker.submit(0)
        self.assertTrue(entered.wait(2))
        with self.assertRaises(TimeoutError):
            worker.close()
        self.assertFalse(worker.thread.is_alive())

    def test_invalid_timeout_never_starts_worker(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                WriterWorker(lambda value: None, timeout=value)
