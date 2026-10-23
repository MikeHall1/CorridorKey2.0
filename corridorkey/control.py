"""Thread-safe cooperative cancellation and interface-independent progress."""
import threading
from dataclasses import dataclass


class CancelledError(RuntimeError):
    pass


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = "processing cancelled"

    def cancel(self, reason="processing cancelled"):
        with self._lock:
            if not self._event.is_set():
                self._reason = str(reason)
                self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self):
        if self.cancelled:
            raise CancelledError(self._reason)


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    completed: int
    total: int | None


class JobControl:
    def __init__(self, cancellation=None, progress=None, *, check_hook=None):
        self.cancellation = cancellation if cancellation is not None else CancellationToken()
        self.progress = progress
        self.check_hook = check_hook
        self.last_checkpoint: ProgressEvent | None = None

    def check(self):
        self.cancellation.check()
        if self.check_hook is not None:
            self.check_hook()

    def checkpoint(self, stage, completed=0, total=None):
        # Retain the attempted boundary even when cancellation, resource checks
        # or progress delivery fail before the consumer receives an event.
        self.last_checkpoint = ProgressEvent(stage, completed, total)
        self.check()
        if self.progress is not None:
            self.progress(self.last_checkpoint)
        self.check()
