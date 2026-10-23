"""Supervised local worker process, bounded event buffers, explicit restart only."""
from __future__ import annotations

import logging
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stress_containment import StressContainment

from .app_process import module_command
from .worker_protocol import MAX_FRAME_BYTES, job_to_wire, parse_event, request_frame


class WorkerLost(RuntimeError):
    pass


class WorkerCleanupError(WorkerLost):
    """Startup failed before a caller could acquire the uncertain owner."""

    def __init__(self, cleanup_error: str):
        self.cleanup_error = cleanup_error
        super().__init__('Worker startup cleanup NOT VERIFIED: ' + cleanup_error)


class WorkerClient:
    def __init__(self, *, capacity=4, weights=None, command=None, startup_timeout=10.0, shutdown_timeout=5.0, asset_root=None):
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError('worker capacity must be1..64')
        for duration in (startup_timeout, shutdown_timeout):
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                raise ValueError('worker lifecycle timeouts must be finite positive numbers')
        self.capacity, self.weights = capacity, weights
        self.command = list(command) if command is not None else module_command('corridorkey.worker', 'serve', '--capacity', str(capacity))
        if command is None and weights is not None:
            self.command += ['--weights', str(Path(weights).resolve())]
        if command is None and asset_root is not None:
            self.command += ['--asset-root', str(Path(asset_root).resolve())]
        self.shutdown_timeout = shutdown_timeout
        self.condition = threading.Condition()
        self.write_lock = threading.Lock()
        self.kill_lock = threading.Lock()
        self.events = deque()
        self.progress = {}
        self.buffered_bytes = 0
        self.outstanding: set[str] = set()
        self.seen: set[str] = set()
        self.logs = deque(maxlen=64)
        self.generation = None
        self.sequence = -1
        self.failure: str | None = None
        self.eof = False
        self.expected_close = False
        self.clean_stop = False
        self.cleanup_error = None
        self.containment: StressContainment | None = None
        self.closed = False
        self.outgoing: queue.Queue[bytes | None] = queue.Queue(maxsize=8)
        self.writer = threading.Thread(target=self._write, name='worker-requests', daemon=True)
        self.reader = threading.Thread(target=self._read, name='worker-events', daemon=True)
        self.log_reader = threading.Thread(target=self._read_logs, name='worker-logs', daemon=True)
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        cwd=Path(__file__).resolve().parents[1], start_new_session=os.name != 'nt',
                                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        try:
            self.writer.start()
            self.reader.start()
            self.log_reader.start()
            ready = self.next_event(startup_timeout)
            if ready['kind'] != 'ready':
                raise WorkerLost('worker did not begin with ready handshake')
            value = ready['payload'].get('capacity')
            if type(value) is not int or not 1 <= value <= 64:
                raise WorkerLost('invalid worker capacity handshake')
            self.capacity = value
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_exc:
                logging.getLogger(__name__).exception('Worker startup cleanup failed')
                self.cleanup_error = (f'Worker startup close failed: '
                                      f'{type(cleanup_exc).__name__}: {cleanup_exc}')[:4096]
            if self.cleanup_error is not None:
                raise WorkerCleanupError(self.cleanup_error) from exc
            raise

    def send(self, action, payload, *, request_id=None):
        request_id = request_id or uuid.uuid4().hex
        frame = request_frame(request_id, action, payload)
        with self.condition:
            if self.closed or self.failure is not None or self.eof:
                raise WorkerLost(self.failure or 'worker is closed')
            if action in ('render', 'preview', 'recover'):
                if request_id in self.seen:
                    raise ValueError('request id already used in this generation')
                if len(self.outstanding) >= self.capacity + 1:
                    raise BufferError('client outstanding request limit reached')
                if len(self.seen) >= 4096:
                    raise RuntimeError('worker request history exhausted; explicit restart required')
                self.seen.add(request_id)
                self.outstanding.add(request_id)
        try:
            with self.write_lock:
                self.outgoing.put_nowait(frame)
        except queue.Full as exc:
            with self.condition:
                if action in ('render', 'preview', 'recover'):
                    self.outstanding.discard(request_id)
                    self.seen.discard(request_id)
            raise BufferError('client outbound queue is full') from exc
        return request_id

    def _write(self):
        try:
            assert self.process.stdin is not None
            while True:
                frame = self.outgoing.get()
                if frame is None:
                    return
                view = memoryview(frame)
                while view:
                    count = os.write(self.process.stdin.fileno(), view)
                    if count <= 0:
                        raise BrokenPipeError('worker input stopped accepting bytes')
                    view = view[count:]
        except (OSError, ValueError) as exc:
            with self.condition:
                unexpected = not self.expected_close or bool(self.outstanding)
            if unexpected:
                self._fail(f'worker input disconnected: {exc}')

    def submit(self, job, *, request_id=None):
        return self.send('render', {'job': job_to_wire(job)}, request_id=request_id)

    def submit_preview(self, job, *, request_id=None):
        """Run an explicit preview job through the same sequential model owner."""
        return self.send('preview', {'job': job_to_wire(job)}, request_id=request_id)

    def cancel(self, target, reason='cancelled by client'):
        return self.send('cancel', {'target': target, 'reason': reason})

    def next_event(self, timeout=10.0):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('observation timeout must be finite and nonnegative')
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.events and not self.progress:
                if self.failure is not None:
                    raise WorkerLost(self.failure)
                if self.eof:
                    raise EOFError('worker output closed')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('worker event observation timed out; process may still be running')
                self.condition.wait(remaining)
            if self.events:
                event, size = self.events.popleft()
                self.buffered_bytes -= size
                return event
            key = next(iter(self.progress))
            return self.progress.pop(key)

    def _deliver(self, event, size):
        with self.condition:
            kind, request_id = event['kind'], event['request_id']
            if kind == 'progress':
                if request_id in self.outstanding:
                    self.progress[request_id] = event
            else:
                if len(self.events) >= 128 or self.buffered_bytes + size > 32 * 1024 * 1024:
                    raise BufferError('client critical event buffer exhausted')
                if (kind in ('result', 'failed', 'cancelled', 'rejected') and
                        not (kind == 'rejected' and event['payload'].get('code') == 'duplicate_id')):
                    self.outstanding.discard(request_id)
                    self.progress.pop(request_id, None)
                if kind == 'stopped':
                    self.clean_stop = event['payload'].get('clean') is True
                self.events.append((event, size))
                self.buffered_bytes += size
            self.condition.notify_all()

    def _read(self):
        try:
            assert self.process.stdout is not None
            while True:
                frame = self.process.stdout.readline(MAX_FRAME_BYTES + 1)
                if not frame:
                    break
                event = parse_event(frame)
                if self.generation is None:
                    if event['kind'] != 'ready' or event['sequence'] != 0:
                        raise ValueError('missing initial worker handshake')
                    self.generation = event['generation']
                if event['generation'] != self.generation or event['sequence'] != self.sequence + 1:
                    raise ValueError('worker generation/sequence mismatch')
                self.sequence = event['sequence']
                self._deliver(event, len(frame))
            with self.condition:
                unexpected = bool(self.outstanding) or not (self.expected_close or self.clean_stop)
            if unexpected:
                self._fail('worker exited without completing its requests')
        except (OSError, ValueError, TypeError, BufferError) as exc:
            self._fail(f'worker protocol failed: {exc}')
        finally:
            with self.condition:
                self.eof = True
                self.condition.notify_all()

    def _read_logs(self):
        assert self.process.stderr is not None
        try:
            while True:
                data = os.read(self.process.stderr.fileno(), 4096)
                if not data:
                    return
                with self.condition:
                    self.logs.append(data)
        except OSError:
            return

    @property
    def log_tail(self):
        with self.condition:
            return b''.join(self.logs).decode('utf-8', errors='replace')

    def _fail(self, message):
        with self.condition:
            if self.failure is not None:
                return
            self.failure = message
            for request_id in sorted(self.outstanding):
                self.events.append(({'kind': 'lost', 'request_id': request_id, 'generation': self.generation,
                                     'payload': {'error': message, 'outputs_may_be_partial': True}}, 0))
            self.outstanding.clear()
            self.progress.clear()
            self.condition.notify_all()
        self._terminate_owned()

    def _terminate_owned(self):
        # Cooperative shutdown has its own deadline. Force the whole live owned
        # group at once: killing a leader first can lose descendant ownership.
        with self.kill_lock:
            self._capture_descendants()
            if self.containment is not None:
                self.containment.terminate_descendants()
            if os.name == 'nt':
                if self.process.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                                   check=False, capture_output=True, timeout=3)
            else:
                try:
                    if self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    self.cleanup_error = f'owned process-group cleanup could not be verified: {exc}'
                    self.clean_stop = False
                    if self.process.poll() is None:
                        self.process.kill()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

    def _capture_descendants(self):
        # The existing close/failure owner samples before signalling or waiting;
        # no observer thread or per-frame native queries are introduced.
        if os.name == 'nt' or self.process.poll() is not None:
            return
        if self.containment is None:
            from .stress_containment import StressContainment

            process = self.process
            self.containment = StressContainment(process.pid, root_exited=lambda: process.poll() is not None)
        self.containment.sample(force=True)

    def _verify_descendant_cleanup(self):
        try:
            if self.containment is not None:
                self.containment.terminate_descendants()
                self.containment.verify()
            elif os.name != 'nt':
                try:
                    os.killpg(self.process.pid, 0)
                except ProcessLookupError:
                    return
                raise RuntimeError('Former worker group remains present without living leader or birth-pinned authority')
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.cleanup_error = f'owned descendant cleanup NOT VERIFIED: {exc}'
            self.clean_stop = False

    def poll(self):
        """Observe/reap the owned Popen under the root-signal authority lock."""
        with self.kill_lock:
            return self.process.poll()

    def restart(self):
        if self.poll() is None:
            raise RuntimeError('cannot restart a worker confirmed still running')
        self.close()
        if self.cleanup_error is not None:
            raise RuntimeError('cannot restart while owned worker cleanup is unverified: ' + self.cleanup_error)
        return WorkerClient(capacity=self.capacity, weights=self.weights, command=self.command,
                            shutdown_timeout=self.shutdown_timeout)

    def close(self):
        if self.closed:
            return self.failure is None and self.cleanup_error is None and self.clean_stop and self.process.returncode == 0
        self.expected_close = True
        with self.kill_lock:
            self._capture_descendants()
        if self.poll() is None and self.failure is None:
            try:
                self.send('shutdown', {'drain': False})
            except (OSError, WorkerLost, ValueError, BufferError):
                pass
        try:
            # Reaping releases numeric PID/group authority. Keep it atomic
            # with the failure owner's poll -> group-signal critical section.
            with self.kill_lock:
                self.process.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            self._terminate_owned()
        with self.kill_lock:
            self._verify_descendant_cleanup()
        try:
            self.outgoing.put_nowait(None)
        except queue.Full:
            pass
        for owner in (self.writer, self.reader, self.log_reader):
            if owner.ident is not None:
                owner.join(2)
        if self.reader.is_alive() or self.writer.is_alive() or self.log_reader.is_alive():
            self.clean_stop = False
            self.cleanup_error = self.cleanup_error or 'Worker I/O owner remains live after bounded cleanup'
        while True:
            try:
                self.outgoing.get_nowait()
            except queue.Empty:
                break
        for pipe, owner in ((self.process.stdin, self.writer), (self.process.stdout, self.reader),
                            (self.process.stderr, self.log_reader)):
            # Buffered close can wait forever on a live reader's internal lock.
            # Retain that owner's handle and explicit uncertainty instead.
            if owner.is_alive():
                continue
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    self.clean_stop = False
                    self.cleanup_error = self.cleanup_error or 'Worker pipe cleanup failed'
        self.closed = True
        return self.failure is None and self.cleanup_error is None and self.clean_stop and self.process.returncode == 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def run_worker_batch(manifest, report_path, *, capacity=4, workspace=None, max_cache_bytes=0,
                     resume=False, durable_foundations=False, durable_rvm=False, durable_hann=False, durable_stage2=False,
                     durable_writer_prefix=False,
                     client_factory: Callable[..., WorkerClient] = WorkerClient):
    """Working sequential CLI client; GUI callers can use the bounded queue API."""
    from .batch import load_jobs
    from .diagnostics import write_json_atomic
    from .lifecycle import cleanup_all
    from .output_paths import prepare_report_destination, validate_batch_output_paths

    report = {'schema_version': 1, 'kind': 'worker_batch', 'status': 'running', 'jobs': [],
              'note': 'Local child process; normal jobs share a model, recovery requests own fresh batch sessions. '
                      'No automatic retry; lost work may have partial output.'}
    client = None
    started = time.perf_counter()
    jobs, weights = (), None
    report_admitted = False
    report['report_write'] = {'status': 'not_written', 'reason': 'Protected job inventory and report destination not admitted.'}

    admit_report = None

    def persist():
        nonlocal report_admitted
        if not report_admitted:
            return
        try:
            assert admit_report is not None
            admit_report()
            report['report_write'] = {'status': 'written'}
            write_json_atomic(report_path, report)
        except BaseException as exc:
            report_admitted = False
            report['report_write'] = {'status': 'not_written', 'reason': f'{type(exc).__name__}: {exc}'}
            raise

    def record(event):
        status = 'complete' if event['kind'] == 'result' else event['kind']
        report['jobs'].append({'request_id': event['request_id'], 'status': status, **event['payload']})
        if status != 'complete':
            report['status'] = 'cancelled' if status == 'cancelled' else 'failed'

    try:
        loaded_jobs, weights = load_jobs(manifest)
        jobs = tuple(loaded_jobs)
        validate_batch_output_paths(jobs, weights=weights, manifest_path=manifest)
        admit_report = prepare_report_destination(
            report_path, jobs, manifest_path=manifest, weights=weights,
            protected_directories=() if workspace is None else (workspace,))
        report_admitted = True
        persist()
        client = client_factory(capacity=capacity, weights=weights)
        report['worker'] = {'generation': client.generation, 'pid': client.process.pid, 'capacity': client.capacity}
        persist()
        requests = ([('recover', {'batch': str(Path(manifest).resolve()), 'workspace': str(Path(workspace).resolve()),
                                  'max_cache_bytes': max_cache_bytes, 'resume': resume,
                                  'durable_foundations': durable_foundations, 'durable_rvm': durable_rvm,
                                  'durable_hann': durable_hann, 'durable_stage2': durable_stage2,
                                  'durable_writer_prefix': durable_writer_prefix})] if workspace is not None else
                    [('render', {'job': job_to_wire(job)}) for job in jobs])
        last_progress_write = 0.0
        for action, payload in requests:
            request_id = client.send(action, payload)
            while True:
                try:
                    event = client.next_event(timeout=60)
                except TimeoutError:
                    report['observation'] = {'state': 'waiting_on_same_worker', 'pid': client.process.pid,
                                             'returncode': client.poll()}
                    persist()
                    continue
                if event['request_id'] == request_id and event['kind'] in ('result', 'failed', 'cancelled', 'rejected', 'lost'):
                    record(event)
                    persist()
                    break
                if event['kind'] == 'progress':
                    report['progress'] = event
                    now = time.monotonic()
                    if now - last_progress_write >= 0.25:
                        persist()
                        last_progress_write = now
            if report['status'] != 'running':
                break
        if report['status'] == 'running':
            report['status'] = 'complete'
    except KeyboardInterrupt:
        report['status'] = 'cancelled'
        if client is not None:
            with client.condition:
                pending = tuple(client.outstanding)
            for request_id in pending:
                try:
                    client.cancel(request_id, 'batch client interrupted')
                except WorkerLost:
                    break
    except (OSError, ValueError, TypeError, WorkerLost, TimeoutError) as exc:
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if client is not None:
            try:
                report['worker_clean_stop'] = client.close()
                if client.cleanup_error is not None:
                    report['cleanup_error'] = client.cleanup_error
                if not report['worker_clean_stop'] and report['status'] == 'complete':
                    report['status'] = 'failed'
                    report['cleanup_error'] = client.cleanup_error or 'worker shutdown was not verified clean'
                known = {item['request_id'] for item in report['jobs']}
                while True:
                    try:
                        event = client.next_event(0)
                    except (EOFError, TimeoutError, WorkerLost):
                        break
                    if event['kind'] in ('result', 'failed', 'cancelled', 'rejected', 'lost') and event['request_id'] not in known:
                        record(event)
                        known.add(event['request_id'])
                report['worker_log_tail'] = client.log_tail
            except (OSError, RuntimeError) as exc:
                report['status'] = 'failed'
                report['cleanup_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        report['operation_seconds'] = time.perf_counter() - started
        cleanup_all([persist], primary_error=sys.exception())
    return report
