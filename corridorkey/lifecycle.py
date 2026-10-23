"""Bounded writer transport and cleanup, independent of codecs and models.

Single producer; consumers must return/raise or provide an abort hook. External
encoders supply a termination hook. Native in-process calls cannot be killed.
"""
import queue
import threading
import time
import math


class _WriterBarrier:
    def __init__(self):
        self.done = threading.Event()


class WriterWorker:
    def __init__(self, consume, maxsize=2, *, check=None, timeout=60.0, abort_consumer=None):
        if maxsize < 1:
            raise ValueError("writer queue maxsize must be positive")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("writer timeout must be finite and positive")
        self.queue = queue.Queue(maxsize=maxsize)
        self.consume = consume
        self.error = None
        self.closed = False
        self.check = check or (lambda: None)
        self.timeout = timeout
        self.abort_consumer = abort_consumer
        self.stopped = threading.Event()
        self._abort_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopped.is_set():
                try:
                    item = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if item is None:
                        return
                    if isinstance(item, _WriterBarrier):
                        item.done.set()
                    else:
                        self.consume(item)
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self.error = exc

    def _put(self, item):
        started = time.monotonic()
        while True:
            self.check()
            if self.error is not None:
                raise self.error
            if not self.thread.is_alive():
                raise RuntimeError("writer worker stopped")
            if time.monotonic() - started >= self.timeout:
                raise TimeoutError("writer queue timed out; encoder/storage may be stalled")
            try:
                self.queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def submit(self, item):
        if self.closed:
            raise RuntimeError("writer is closed")
        if item is None:
            raise ValueError("None is reserved for writer shutdown")
        try:
            self._put(item)
        except BaseException as exc:
            self.abort(exc)
            raise

    def flush_barrier(self):
        """Wait for earlier consumer calls; this is not fsync/encoder completion.

        The single producer retains ownership. Queue drainage after abort is
        never an acknowledgement that queued items were consumed successfully.
        """
        if self.closed:
            raise RuntimeError('writer is closed')
        barrier = _WriterBarrier()
        started = time.monotonic()
        try:
            self._put(barrier)
            while True:
                self.check()
                if self.error is not None:
                    raise self.error
                if barrier.done.is_set():
                    return
                if not self.thread.is_alive():
                    raise RuntimeError('writer stopped before barrier acknowledgement')
                if time.monotonic() - started >= self.timeout:
                    raise TimeoutError('writer barrier timed out; native consumer may be stalled')
                barrier.done.wait(.05)
        except BaseException as exc:
            self.abort(exc)
            raise

    def abort(self, reason):
        with self._abort_lock:
            if self.stopped.is_set():
                return
            self.stopped.set()
            if self.error is None:
                self.error = reason
            if self.abort_consumer is not None:
                cleanup_all([self.abort_consumer], primary_error=reason)
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._put(None)
                started = time.monotonic()
                while self.thread.is_alive():
                    self.check()
                    if time.monotonic() - started >= self.timeout:
                        raise TimeoutError("writer finalization timed out")
                    self.thread.join(timeout=0.05)
            except BaseException as exc:
                self.abort(exc)
                self.thread.join(timeout=min(1.0, self.timeout))
                if self.thread.is_alive():
                    exc.add_note("native writer did not stop; partial output may remain")
                raise
        if self.error is not None:
            raise self.error


def terminate_process(process, grace=1.0):
    """Terminate/reap an owned encoder; never wait indefinitely."""
    import subprocess
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=grace)


def cleanup_all(callbacks, primary_error=None):
    """Attempt all cleanup; retain the processing exception as the primary error."""
    errors = []
    for callback in callbacks:
        try:
            callback()
        except BaseException as exc:
            errors.append(exc)
    if primary_error is not None:
        for exc in errors:
            primary_error.add_note(f"Cleanup also failed: {exc!r}")
    elif errors:
        for exc in errors[1:]:
            errors[0].add_note(f"Cleanup also failed: {exc!r}")
        raise errors[0]
