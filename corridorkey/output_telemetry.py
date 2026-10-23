"""Bounded export milestones, not neural service time or playback latency."""
from __future__ import annotations

import copy
import math
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from typing import Any


@dataclass
class _Frame:
    ready: float
    submitted: int = 0
    written: int = 0
    last_submit: float = 0.0
    last_write: float = 0.0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _error(error):
    with suppress(BaseException):
        return f'{type(error).__name__}: {error}'[:512]
    return 'Observation error message unavailable'


class ObservationBoundary:
    """Contain any observer exception through context-manager exit, without I/O.

    The handler only updates owned diagnostic state. This boundary deliberately
    contains interrupts from observer callbacks as well as ordinary exceptions.
    """
    def __init__(self, on_error):
        self.on_error = on_error

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.on_error(error)
            return True
        return False


def observation_error_text(error):
    return _error(error)


def _distribution(values, expected, *, reason=None):
    result = {'status': 'unavailable', 'count': len(values), 'expected_count': expected,
        'min_seconds': None, 'max_seconds': None, 'p50_seconds': None,
        'p95_seconds': None, 'p99_seconds': None,
        'method': 'exact nearest-rank ceil(p*N), no interpolation', 'reason': reason}
    if reason is not None or not values:
        result['reason'] = reason or 'No observed events'
        return result
    ordered = sorted(values)
    result.update(status='complete' if len(values) == expected else 'partial',
        min_seconds=ordered[0], max_seconds=ordered[-1],
        p50_seconds=ordered[max(0, math.ceil(.50 * len(ordered)) - 1)],
        p95_seconds=ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
        p99_seconds=ordered[max(0, math.ceil(.99 * len(ordered)) - 1)],
        reason=None if len(values) == expected else 'Observed completed subset only; not a whole-job distribution')
    return result


class OutputTelemetry:
    """Register before queue submission; native callbacks may arrive before submit returns.

    Event methods contain observation failures. Finish is called by the owner after
    writer finalization; it never joins threads. Snapshot is cached JSON only.
    """
    def __init__(self, outputs, expected_frames, *, native_completion, origin,
                 clock=time.perf_counter, max_frames=100000, max_inflight_frames=4096,
                 restored_prefix_frames=0):
        if (not isinstance(outputs, (list, tuple, set, frozenset)) or not 1 <= len(outputs) <= 16
                or any(not isinstance(name, str) or not 1 <= len(name) <= 64 for name in outputs)
                or len(set(outputs)) != len(outputs) or type(expected_frames) is not int or not 1 <= expected_frames <= 2**63 - 1
                or type(native_completion) is not bool or not _finite(origin)
                or type(max_frames) is not int or not 1 <= max_frames <= 1000000
                or type(max_inflight_frames) is not int or not 1 <= max_inflight_frames <= 65536
                or type(restored_prefix_frames) is not int or not 0 <= restored_prefix_frames <= expected_frames
                or restored_prefix_frames > 0 and not native_completion):
            raise ValueError('Invalid output telemetry contract/bounds')
        self.outputs = tuple(sorted(outputs))
        self.bits = {name: 1 << index for index, name in enumerate(self.outputs)}
        self.all_bits = (1 << len(self.outputs)) - 1
        self.expected = expected_frames
        self.restored = restored_prefix_frames
        self.write_expected = expected_frames - restored_prefix_frames
        self.native = native_completion
        self.origin, self.clock = float(origin), clock
        self.max_frames, self.max_inflight = max_frames, max_inflight_frames
        self.lock = threading.Lock()
        self.pending = {}
        self.seen = set()
        self.ready_offsets = []
        self.queue_latencies = []
        self.native_latencies = []
        self.native_offsets = []
        self.submitted_counts = dict.fromkeys(self.outputs, 0)
        self.written_counts = dict.fromkeys(self.outputs, 0)
        self.last_written: dict[str, tuple[int, float] | None] = dict.fromkeys(self.outputs, None)
        self.last_submitted: dict[str, float | None] = dict.fromkeys(self.outputs, None)
        self.last_ready: float | None = None
        self.last_index = -1
        self.failure: str | None = None
        self.finished = False
        if expected_frames > max_frames:
            self.failure = 'Expected frame population exceeds exact sample limit; no truncated-window percentiles'
        self.cached = self._report(finished=False)

    def _report(self, *, finished) -> dict[str, Any]:
        reason = self.failure or (None if finished else 'Collection has not finished')
        ready = _distribution(self.ready_offsets, self.expected, reason=reason)
        queues = _distribution(self.queue_latencies, self.write_expected, reason=reason)
        native_reason = reason or (None if self.native else 'Native per-frame completion unavailable for this output format')
        native = _distribution(self.native_latencies, self.write_expected, reason=native_reason)
        offsets = _distribution([value for _, value in self.native_offsets], self.write_expected, reason=native_reason)
        spacings = []
        if finished and native_reason is None:
            ordered = sorted(self.native_offsets)
            spacings = [later[1] - earlier[1] for earlier, later in pairwise(ordered)
                        if later[0] == earlier[0] + 1]
            if any(value < 0 for value in spacings):
                native_reason = 'Nonmonotonic all-output completion order'
                offsets = _distribution([], self.write_expected, reason=native_reason)
                native = _distribution([], self.write_expected, reason=native_reason)
        spacing = _distribution(spacings, max(0, self.write_expected - 1), reason=native_reason)
        complete = (finished and self.failure is None and len(self.ready_offsets) == self.expected
                    and len(self.queue_latencies) == self.write_expected
                    and self.native and len(self.native_latencies) == self.write_expected and native_reason is None
                    and self.restored == 0)
        observed = bool(self.ready_offsets or self.queue_latencies or self.native_latencies)
        return {'enabled': True, 'status': 'complete' if complete else 'partial' if observed else 'unavailable',
            'finished': finished, 'expected_frames': self.expected, 'registered_frames': len(self.seen),
            'restored_prefix_frames': self.restored, 'expected_newly_written_frames': self.write_expected,
            'write_distribution_population': 'Newly written suffix only; copied prefix has no queue/native timestamps' if self.restored else 'All selected output frames',
            'all_queues_accepted_frames': len(self.queue_latencies),
            'all_native_completed_frames': len(self.native_latencies),
            'missing_registered_frames': self.expected - len(self.seen),
            'missing_queue_frames': self.write_expected - len(self.queue_latencies),
            'missing_native_frames': self.write_expected - len(self.native_latencies) if self.native else None,
            'pending_frames': len(self.pending), 'max_frames': self.max_frames,
            'max_inflight_frames': self.max_inflight,
            'per_output': {name: {'submitted_frames': self.submitted_counts[name],
                'native_completed_frames': self.written_counts[name]} for name in self.outputs},
            'failure': self.failure, 'native_completion': self.native, 'origin': 'processing_start',
            'frame_index_space': 'output-relative',
            'source_alignment_note': 'selected source range remains in parent diagnostics',
            'render_ready_offsets': ready, 'queue_accept_latency': queues,
            'native_write_latency': native, 'native_completion_offsets': offsets, 'completion_spacing': spacing,
            'scope': 'Render-ready is completed batch boundary (or QuickKey rendered frame). Queue acceptance is not encoding. EXR native completion is cv2.imwrite return, not fsync durability. No source-arrival, per-frame neural, presented-frame or playback latency claim.'}

    def _failure(self, error):
        with self.lock:
            if self.failure is None:
                self.failure = _error(error)
            self.cached['failure'] = self.failure
            self.cached['status'] = 'partial' if self.seen else 'unavailable'
            counts = {'render_ready_offsets': len(self.ready_offsets), 'queue_accept_latency': len(self.queue_latencies),
                      'native_write_latency': len(self.native_latencies), 'native_completion_offsets': len(self.native_offsets),
                      'completion_spacing': max(0, len(self.native_offsets) - 1)}
            for key, count in counts.items():
                expected = (max(0, self.write_expected - 1) if key == 'completion_spacing'
                            else self.expected if key == 'render_ready_offsets' else self.write_expected)
                self.cached[key] = _distribution([], expected, reason=self.failure)
                self.cached[key]['count'] = count

    def observation_failed(self, reason):
        """Accept a bounded writer-hook diagnostic without replacing job errors."""
        with ObservationBoundary(self._failure):
            self._failure(ValueError(str(reason)[:512]))


    def _ids(self, frames):
        if not isinstance(frames, (list, tuple, range)) or not 1 <= len(frames) <= self.max_inflight:
            raise ValueError('Frame event sequence exceeds bounded in-flight admission')
        if any(type(index) is not int or not 0 <= index < self.expected for index in frames):
            raise ValueError('Frame index is outside selected output timeline')
        if len(set(frames)) != len(frames):
            raise ValueError('Duplicate frame indices in one event')
        return frames

    def _timestamp(self, value):
        if not _finite(value) or value < self.origin:
            raise ValueError('Nonfinite or pre-origin observation clock')
        return value - self.origin

    def render_ready(self, frame_ids):
        with ObservationBoundary(self._failure):
            if self.failure is not None:
                return
            now = self._timestamp(self.clock())
            ids = self._ids(frame_ids)
            with self.lock:
                if self.finished:
                    raise ValueError('Render event after telemetry finish')
                if self.last_ready is not None and now < self.last_ready:
                    raise ValueError('Backward render observation clock')
                if len(self.pending) + sum(index >= self.restored for index in ids) > self.max_inflight or len(self.seen) + len(ids) > self.max_frames:
                    raise ValueError('Telemetry exact sample/in-flight capacity exceeded')
                if any(index in self.seen for index in ids) or any(index <= self.last_index for index in ids):
                    raise ValueError('Duplicate/backward render frame event')
                if list(ids) != sorted(ids):
                    raise ValueError('Nonchronological render frame event')
                for index in ids:
                    if index >= self.restored:
                        self.pending[index] = _Frame(now)
                    self.seen.add(index)
                    self.ready_offsets.append(now)
                self.last_ready, self.last_index = now, ids[-1]
                self._cache_counts()

    def submitted(self, output, frame_ids):
        with ObservationBoundary(self._failure):
            if self.failure is not None:
                return
            now = self._timestamp(self.clock())
            ids = self._ids(frame_ids)
            with self.lock:
                if self.finished or output not in self.bits:
                    raise ValueError('Unknown output or queue event after finish')
                prior = self.last_submitted[output]
                if prior is not None and now < prior:
                    raise ValueError('Backward queue observation clock')
                bit = self.bits[output]
                for index in ids:
                    frame = self.pending.get(index)
                    if frame is None or frame.submitted & bit or now < frame.ready:
                        raise ValueError('Unknown/duplicate/pre-render queue acceptance')
                for index in ids:
                    frame = self.pending[index]
                    frame.submitted |= bit
                    frame.last_submit = max(frame.last_submit, now)
                    self.submitted_counts[output] += 1
                    if frame.submitted == self.all_bits:
                        self.queue_latencies.append(frame.last_submit - frame.ready)
                    self._retire(index, frame)
                self.last_submitted[output] = now
                self._cache_counts()

    def written(self, output, index, at_seconds):
        with ObservationBoundary(self._failure):
            if self.failure is not None:
                return
            self._ids((index,))
            now = self._timestamp(at_seconds)
            with self.lock:
                if self.finished or not self.native or output not in self.bits:
                    raise ValueError('Unknown/unsupported native event or event after finish')
                prior = self.last_written[output]
                if prior is not None and (now < prior[1] or index <= prior[0]):
                    raise ValueError('Backward/duplicate native completion event')
                frame = self.pending.get(index)
                bit = self.bits[output]
                if frame is None or frame.written & bit or now < frame.ready:
                    raise ValueError('Unknown/duplicate/pre-render native completion')
                frame.written |= bit
                frame.last_write = max(frame.last_write, now)
                self.written_counts[output] += 1
                if frame.written == self.all_bits:
                    self.native_latencies.append(frame.last_write - frame.ready)
                    self.native_offsets.append((index, frame.last_write))
                self.last_written[output] = (index, now)
                self._retire(index, frame)
                self._cache_counts()

    def _retire(self, index, frame):
        if frame.submitted == self.all_bits and (not self.native or frame.written == self.all_bits):
            del self.pending[index]

    def _cache_counts(self):
        # Constant-size cached summary; no sorting, callbacks or clocks under lock.
        self.cached.update(registered_frames=len(self.seen), all_queues_accepted_frames=len(self.queue_latencies),
            all_native_completed_frames=len(self.native_latencies), pending_frames=len(self.pending),
            missing_registered_frames=self.expected-len(self.seen), missing_queue_frames=self.write_expected-len(self.queue_latencies),
            missing_native_frames=self.write_expected-len(self.native_latencies) if self.native else None)
        self.cached['per_output'] = {name: {'submitted_frames': self.submitted_counts[name],
            'native_completed_frames': self.written_counts[name]} for name in self.outputs}

    def finish(self):
        with ObservationBoundary(self._failure):
            with self.lock:
                if self.finished:
                    return copy.deepcopy(self.cached)
                self.finished = True
            # Called only after owner finalization; expensive exact sorts never
            # hold the event/snapshot lock. Unexpected late events become failure.
            report = self._report(finished=True)
            with self.lock:
                self.cached = report
                failure = self.failure
            if failure is not None:
                self._failure(ValueError(failure))
            return self.snapshot()
        return self.snapshot()

    def snapshot(self):
        with suppress(BaseException), self.lock:
            return copy.deepcopy(self.cached)
        return {'enabled': True, 'status': 'unavailable', 'failure': 'Telemetry snapshot unavailable'}
