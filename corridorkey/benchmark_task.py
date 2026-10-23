"""Owned benchmark CLI process, asynchronous observation and bounded log tail."""
from __future__ import annotations

import logging
import math
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .stress_containment import StressContainment

from .app_process import module_command
from .benchmark_report import read_report, report_view
from .lifecycle import cleanup_all
from .local_json import read_object

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTask:
    def __init__(self, manifest: Path, output_parent: Path, *, grace_seconds: float = 5,
                 command_prefix: Sequence[str] | None = None):
        if type(grace_seconds) not in (int, float) or not math.isfinite(grace_seconds) or grace_seconds <= 0:
            raise ValueError('Benchmark cancellation grace must be finite and positive.')
        self.manifest = manifest.resolve()
        self.directory = output_parent.resolve() / f'benchmark-{uuid.uuid4().hex}'
        self.report_path = self.directory / 'report.json'
        self.select_runner = command_prefix is None
        self.command = list(command_prefix) if command_prefix is not None else module_command('corridorkey.benchmark')
        self.command += ['--manifest', str(self.manifest), '--output', str(self.report_path)]
        self.grace_seconds = grace_seconds
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.logs: deque[str] = deque(maxlen=64)
        self.state: dict[str, Any] = {'status': 'starting', 'pid': None, 'returncode': None, 'error': None,
                                     'report_path': str(self.report_path), 'report_available': False,
                                     'report_view': '', 'forced': False, 'cleanup_error': None, 'revision': 0}
        self.process: subprocess.Popen[bytes] | None = None
        self.reader: threading.Thread | None = None
        self.containment: StressContainment | None = None
        self.stress_mode = False
        self.thread = threading.Thread(target=self._run, name='desktop-benchmark', daemon=True)
        self.thread.start()

    def cancel(self) -> None:
        self.cancelled.set()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {**self.state, 'log_tail': ''.join(self.logs)}

    def _update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)
            self.state['revision'] += 1

    def _read_logs(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while data := os.read(self.process.stdout.fileno(), 4096):
                with self.lock:
                    self.logs.append(data.decode('utf-8', errors='replace'))
                    self.state['revision'] += 1
        except OSError as exc:
            self._update(cleanup_error=f'Benchmark log read failed: {exc}')

    def _signal(self, *, force: bool) -> None:
        assert self.process is not None
        if force and self.containment is not None:
            self.containment.terminate_descendants()
        # Native descendant sampling can reap this Popen between the loop's
        # liveness test and this signal. A reaped PID is no group authority.
        if self.process.poll() is not None:
            return
        if os.name != 'nt':
            try:
                os.killpg(self.process.pid, signal.SIGKILL if force else signal.SIGINT)
            except ProcessLookupError:
                pass
        elif force:
            result = subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                                    capture_output=True, timeout=5, check=False)
            if result.returncode and self.process.poll() is None:
                self.process.kill()
        elif self.process.poll() is None:
            self.process.send_signal(signal.CTRL_BREAK_EVENT)

    def _cleanup(self) -> None:
        if self.process is None:
            return
        process = self.process

        def reap():
            if process.poll() is None:
                self._signal(force=True)
                process.wait(timeout=5)

        def contain():
            if self.containment is not None:
                try:
                    self.containment.terminate_descendants()
                    self.containment.verify()
                finally:
                    self._update(process_containment=self.containment.snapshot())
                    if self.stress_mode:
                        self._update(stress_containment=self.containment.snapshot())
                return
            if self.stress_mode:
                raise RuntimeError('Stress observer was not initialized; descendant cleanup NOT VERIFIED')
            # Descendants can retain stdout or survive a failed/crashed leader.
            if os.name != 'nt':
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Former benchmark group remains present without living leader or birth-pinned authority')

        def join_logs():
            if self.reader is not None:
                self.reader.join(2)
                if self.reader.is_alive():
                    raise RuntimeError('Benchmark log reader remains live; descendant cleanup unverified.')

        def close_logs():
            if process.stdout is not None and (self.reader is None or not self.reader.is_alive()):
                process.stdout.close()

        cleanup_all([reap, contain, join_logs, close_logs])

    def _run(self) -> None:
        try:
            if self.cancelled.is_set():
                self._update(status='cancelled')
                return
            if not self.manifest.is_file() or not self.directory.parent.is_dir():
                raise ValueError('Select an existing benchmark manifest and output parent directory.')
            manifest, _ = read_object(self.manifest, max_bytes=4 * 1024 * 1024, regular_file=True)
            self.stress_mode = manifest.get('kind') == 'production_stress'
            if self.select_runner:
                kind = manifest.get('kind')
                module = ({'inference_pair': 'corridorkey.benchmark_pair',
                           'production_stress': 'corridorkey.stress_suite'}.get(kind, 'corridorkey.benchmark')
                          if isinstance(kind, str) else 'corridorkey.benchmark')
                self.command = module_command(module, '--manifest', str(self.manifest), '--output', str(self.report_path))
            self.directory.mkdir(exist_ok=False)
            self.process = subprocess.Popen(self.command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            start_new_session=os.name != 'nt',
                                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            self._update(status='running', pid=self.process.pid)
            if self.stress_mode or (os.name != 'nt' and self.process.poll() is None):
                from .stress_containment import StressContainment

                process = self.process
                self.containment = StressContainment(process.pid, root_exited=lambda: process.poll() is not None)
                self.containment.sample(force=True)
                self._update(**{'stress_containment' if self.stress_mode else 'process_containment': self.containment.snapshot()})
            self.reader = threading.Thread(target=self._read_logs, name='benchmark-logs', daemon=True)
            self.reader.start()
            cancel_deadline: float | None = None
            while self.process.poll() is None:
                if self.containment is not None:
                    sampled_at = self.containment.last_sample
                    self.containment.sample()
                    if self.containment.last_sample != sampled_at:
                        self._update(**{'stress_containment' if self.stress_mode else 'process_containment': self.containment.snapshot()})
                if self.cancelled.is_set() and cancel_deadline is None:
                    self._update(status='cancelling')
                    self._signal(force=False)
                    cancel_deadline = time.monotonic() + self.grace_seconds
                elif cancel_deadline is not None and time.monotonic() >= cancel_deadline:
                    self._update(forced=True)
                    self._signal(force=True)
                    self.process.wait(timeout=5)
                    break
                self.done.wait(0.02)
            returncode = self.process.wait()
            self._update(returncode=returncode)
            if self.report_path.is_file():
                report, _ = read_report(self.report_path, regular_file=True)
                report_status = report.get('status') or report.get('inference', {}).get('status')
                status = ('cancelled' if report_status == 'cancelled' or returncode == 130 else
                          'complete' if returncode == 0 and report['summary']['success'] else 'failed')
                if (report['kind'] == 'production_stress' and returncode == 2 and report_status == 'complete'
                        and not report['summary']['success'] and report['summary']['not_verified_cases'] > 0
                        and report['summary']['failed_cases'] == report['summary']['cancelled_cases'] == 0
                        and report['summary']['cleanup_verified'] is True):
                    status = 'unavailable'
                self._update(status=status, report_available=True, report_view=report_view(report))
            else:
                self._update(status='cancelled' if self.cancelled.is_set() else 'failed',
                             error='Process exited without a validated report; outputs may be partial.')
        except Exception as exc:
            logging.getLogger(__name__).exception('Benchmark process failed')
            self._update(status='failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                self._cleanup()
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                logging.getLogger(__name__).exception('Benchmark cleanup could not be verified')
                self._update(cleanup_error=str(exc), status='failed')
            finally:
                if self.state['cleanup_error'] is not None:
                    self._update(status='failed')
                self.done.set()
