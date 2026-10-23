"""Cooperative resident-memory stop threshold; never an allocator hard cap."""
from __future__ import annotations

import copy
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from .control import CancelledError


class ResourceLimitError(RuntimeError):
    pass


class ResourceMeasurementError(RuntimeError):
    pass


class MemoryObserver(Protocol):
    def sample(self, check: Callable[[], None] | None = None) -> dict[str, Any]: ...


class ProcessTreeRssGuard:
    def __init__(self, limit_bytes: int, observer: MemoryObserver | None = None, *,
                 cancel_check: Callable[[], None] | None = None,
                 clock: Callable[[], float] = time.monotonic, interval: float = 0.25):
        if type(limit_bytes) is not int or not 0 <= limit_bytes <= 2**63 - 1:
            raise ValueError('process tree RSS threshold must be 0..2^63-1 bytes')
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
            raise ValueError('memory sampling interval must be finite and positive')
        self.limit_bytes = limit_bytes
        self.observer = observer
        self.cancel_check = cancel_check or (lambda: None)
        self.clock, self.interval = clock, float(interval)
        self.lock = threading.Lock()
        self.sample_lock = threading.Lock()
        self.last_time: float | None = None
        self.last_sample: dict[str, Any] | None = None
        self.peak: int | None = None
        self.samples = 0
        self.failure: tuple[type[ResourceLimitError | ResourceMeasurementError], str] | None = None

    def _raise_failure(self):
        if self.failure is not None:
            error, message = self.failure
            raise error(message)

    def check(self, *, force: bool = False):
        self.cancel_check()
        with self.sample_lock:
            self.cancel_check()
            with self.lock:
                self._raise_failure()
                last_time = self.last_time
            try:
                now = self.clock()
                if not math.isfinite(now):
                    raise ValueError('memory sampling clock is nonfinite')
                if not force and last_time is not None and 0 <= now - last_time < self.interval:
                    return
                if self.observer is None:
                    # Default-off jobs never construct or import this native observer.
                    from .process_memory import ProcessTreeMemory

                    self.observer = ProcessTreeMemory()
                sample = self.observer.sample(check=self.cancel_check)
                raw = json.dumps(sample, allow_nan=False, separators=(',', ':'))
                if len(raw.encode('utf-8')) > 1024 * 1024:
                    raise ValueError('memory observation exceeds the 1 MiB diagnostic limit')
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise TypeError('memory observer did not return a report object')
                sampled_at = self.clock()
                if not math.isfinite(sampled_at):
                    raise ValueError('memory sampling clock is nonfinite')
                with self.lock:
                    self.last_sample = data
                    self.samples += 1
                    self.last_time = sampled_at
                value = data.get('current_rss_bytes')
                if data.get('status') != 'available' or type(value) is not int or value < 0:
                    reason = str(data.get('reason') or 'complete current tree RSS is unavailable')[:2048]
                    raise ResourceMeasurementError(f'Process-tree RSS threshold could not be measured: {reason}')
                with self.lock:
                    self.peak = max(value, self.peak or 0)
                    if value > self.limit_bytes:
                        message = (f'Observed worker-tree RSS {value} bytes exceeds stop threshold {self.limit_bytes} bytes; '
                                   'cooperative stop at a safe checkpoint, not a hard RAM cap')
                        self.failure = ResourceLimitError, message
            except CancelledError:
                raise
            except Exception as exc:
                logging.getLogger(__name__).exception('Process-tree RSS measurement failed')
                message = f'{type(exc).__name__}: {exc}'[:4096]
                with self.lock:
                    if self.last_sample is None:
                        self.last_sample = {'status': 'unavailable', 'current_rss_bytes': None,
                                            'provider': 'unavailable', 'reason': message}
                    self.failure = ResourceMeasurementError, message
            with self.lock:
                self._raise_failure()

    def snapshot(self) -> dict[str, Any]:
        """Cached JSON only; safe during cleanup and never invokes the observer."""
        with self.lock:
            failure = {'type': self.failure[0].__name__, 'message': self.failure[1]} if self.failure is not None else None
            return {'limit_bytes': self.limit_bytes, 'sample_interval_seconds': self.interval,
                    'samples': self.samples, 'observed_peak_rss_bytes': self.peak,
                    'last_sample': copy.deepcopy(self.last_sample), 'failure': failure,
                    'scope': 'Sampled sum of current resident memory for the observed worker tree. Shared pages may count more than once; transient children/peaks can be missed. Cooperative failure, not a hard RAM cap, whole-machine usage or GPU memory limit.'}
