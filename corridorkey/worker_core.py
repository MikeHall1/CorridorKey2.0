"""Sequential session owner with a bounded waiting queue and concurrent controls."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from .control import CancellationToken, CancelledError, ProgressEvent
from .diagnostics import json_value
from .resource_snapshot import process_snapshot
from .session import InferenceSession
from .worker_protocol import Request, job_from_wire

logger = logging.getLogger(__name__)

Sink = Callable[[str, str | None, dict[str, Any]], None]


def _execution_interval(started, finished):
    return {'started_monotonic_ns': started, 'finished_monotonic_ns': finished,
            'clock_implementation': time.get_clock_info('monotonic').implementation,
            'pid': os.getpid(),
            'scope': 'Local engine.run invocation, including load/preflight/render or recovery; '
                     'excludes broker delivery. Overlap is not GPU-kernel concurrency.'}


class Engine(Protocol):
    def run(self, request: Request, cancellation: CancellationToken,
            progress: Callable[[ProgressEvent], None]) -> dict[str, Any]: ...
    def failure_record(self) -> dict[str, Any] | None: ...
    def close(self) -> None: ...


class SharedEngine:
    """Normal jobs share one lazy model; recovery requests own fresh batch sessions.

    Closing the normal session before recovery avoids relabelling resident old
    weights with newly fingerprinted assets. Recovery already owns batch lifetime.
    """
    def __init__(self, *, weights=None, package_root=None, asset_root=None):
        self.weights = weights
        self.root = Path(package_root or Path(__file__).resolve().parents[1]).resolve()
        self.asset_root = Path(asset_root).resolve() if asset_root is not None else None
        self.session: InferenceSession | None = None
        self.record = None

    def run(self, request, cancellation, progress):
        self.record = None
        payload = request.payload()
        if request.action == 'render':
            if self.session is None:
                self.session = InferenceSession(weights=self.weights, package_root=self.root, asset_root=self.asset_root)
            try:
                return self.session.run(job_from_wire(payload['job']), cancellation=cancellation, progress=progress).to_dict()
            except BaseException:
                if self.session.last_result is not None:
                    self.record = self.session.last_result.to_dict()
                raise
        self.close()
        from .batch import load_jobs
        from .recovery import cuda_environment, run_recoverable
        from .runtime_paths import RuntimePaths

        jobs, weights = load_jobs(payload['batch'])
        weights = weights if weights is not None else self.weights
        if any(config.output_format != 'exr' or config.auto_vram for job in jobs for config, _ in (job.configurations(),)):
            raise ValueError('recovery requires EXR output and explicit scheduling (auto_vram=false)')
        paths = RuntimePaths.from_package(self.root, weights, asset_root=self.asset_root)
        paths.validate()
        assets = {name: getattr(paths, name) for name in
                  ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir')}
        self.record = {'recovery_workspace': payload['workspace']}
        report = run_recoverable(
            jobs, workspace=payload['workspace'], max_cache_bytes=payload['max_cache_bytes'], assets=assets,
            environment=cuda_environment(), implementation_root=self.root, weights=weights,
            session_factory=partial(InferenceSession, asset_root=self.asset_root),
            cancellation=cancellation, progress=progress, resume=payload['resume'],
            durable_foundations=payload['durable_foundations'], durable_rvm=payload['durable_rvm'],
            durable_hann=payload['durable_hann'], durable_stage2=payload['durable_stage2'],
            durable_writer_prefix=payload['durable_writer_prefix'])
        return {'status': report['status'], 'report_path': report['report_path'], 'journal': report['journal'],
                'job_count': len(report['jobs']), 'session_scope': 'fresh_recovery_batch',
                'note': 'Full records and historical replay provenance are in the local report.'}

    def failure_record(self):
        return self.record

    def close(self):
        if self.session is not None:
            session, self.session = self.session, None
            session.close()


@dataclass(frozen=True)
class Pending:
    request: Request
    cancellation: CancellationToken


class WorkerCore:
    def __init__(self, engine: Engine, emit: Sink, *, capacity=4, max_requests=4096):
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError('worker waiting capacity must be1..64')
        if type(max_requests) is not int or max_requests < capacity + 1:
            raise ValueError('request history limit must exceed waiting capacity')
        self.engine = engine
        self.emit = emit
        self.capacity = capacity
        self.max_requests = max_requests
        self.condition = threading.Condition()
        self.pending: deque[Pending] = deque()
        self.active: Pending | None = None
        self.seen: set[str] = set()
        self.closing = False
        self.done = threading.Event()
        self.fatal: BaseException | None = None
        self.thread = threading.Thread(target=self._loop, name='engine-owner', daemon=True)
        self.thread.start()

    def dispatch(self, request: Request):
        payload = request.payload()
        with self.condition:
            if request.action in ('render', 'recover'):
                code = ('duplicate_id' if request.request_id in self.seen else
                        'stopping' if self.closing else 'restart_required' if len(self.seen) >= self.max_requests else
                        'queue_full' if len(self.pending) >= self.capacity else None)
                if code is not None:
                    self.emit('rejected', request.request_id, {'code': code})
                    return
                self.seen.add(request.request_id)
                self.pending.append(Pending(request, CancellationToken()))
                self.emit('accepted', request.request_id, {'position': len(self.pending), 'action': request.action})
                self.condition.notify_all()
            elif request.action == 'cancel':
                target = payload['target']
                reason = payload.get('reason', 'cancelled by client')
                state = 'unknown_or_terminal'
                for item in list(self.pending):
                    if item.request.request_id == target:
                        self.pending.remove(item)
                        item.cancellation.cancel(reason)
                        self.emit('cancelled', target, {'error': {'type': 'CancelledError', 'message': reason}, 'record': None})
                        state = 'cancelled_before_start'
                        break
                if self.active is not None and self.active.request.request_id == target:
                    self.active.cancellation.cancel(reason)
                    state = 'cancelling'
                self.emit('status', request.request_id, {'target': target, 'state': state})
            elif request.action == 'status':
                self.emit('status', request.request_id, {
                    'active': self.active.request.request_id if self.active is not None else None,
                    'pending': [item.request.request_id for item in self.pending], 'capacity': self.capacity,
                    'closing': self.closing, 'restart_required': len(self.seen) >= self.max_requests,
                    'resources': process_snapshot()})
            elif request.action == 'shutdown':
                self.emit('stopping', request.request_id, {'drain': payload.get('drain', False)})
                self.stop(drain=payload.get('drain', False))

    def stop(self, *, drain=False):
        with self.condition:
            self.closing = True
            if not drain:
                for item in self.pending:
                    item.cancellation.cancel('worker stopping')
                if self.active is not None:
                    self.active.cancellation.cancel('worker stopping')
            self.condition.notify_all()

    def finish(self, timeout=5.0):
        self.thread.join(timeout)
        return self.done.is_set() and self.fatal is None

    def _progress(self, request_id, event):
        with self.condition:
            if self.active is not None and self.active.request.request_id == request_id:
                self.emit('progress', request_id, json_value({'stage': event.stage, 'completed': event.completed, 'total': event.total}))

    def _terminal(self, kind, request_id, payload):
        with self.condition:
            if self.active is not None and self.active.request.request_id == request_id:
                self.active = None
            self.emit(kind, request_id, payload)

    def _loop(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: bool(self.pending) or self.closing)
                    if not self.pending:
                        break
                    item = self.pending.popleft()
                    self.active = item
                request_id = item.request.request_id
                try:
                    item.cancellation.check()
                    self.emit('started', request_id, {'action': item.request.action})
                    execution_started = time.monotonic_ns()
                    try:
                        result = self.engine.run(item.request, item.cancellation,
                                                 partial(self._progress, request_id))
                    except BaseException as exc:
                        execution = _execution_interval(execution_started, time.monotonic_ns())
                        logger.exception('engine request %s failed', request_id)
                        kind = 'cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed'
                        self._terminal(kind, request_id, {'error': {'type': type(exc).__name__, 'message': str(exc)},
                                                     'record': self.engine.failure_record(),
                                                     'execution_interval': execution})
                    else:
                        execution = _execution_interval(execution_started, time.monotonic_ns())
                        try:
                            self._terminal('result', request_id, {'record': result,
                                                                 'execution_interval': execution})
                        except ValueError as exc:
                            self._terminal('failed', request_id, {'error': {'type': 'ResultEncodingError', 'message': str(exc)},
                                                             'outputs_may_be_complete': True, 'record': None,
                                                             'execution_interval': execution})
                except CancelledError as exc:
                    self._terminal('cancelled', request_id, {'error': {'type': type(exc).__name__, 'message': str(exc)}, 'record': None})
                finally:
                    with self.condition:
                        self.active = None
        except BaseException as exc:
            logger.exception('worker owner failed')
            self.fatal = exc
            self.stop()
        finally:
            try:
                self.engine.close()
                self.emit('stopped', None, {'clean': self.fatal is None})
            except BaseException as exc:
                logger.exception('worker cleanup failed')
                self.fatal = self.fatal or exc
            self.done.set()
