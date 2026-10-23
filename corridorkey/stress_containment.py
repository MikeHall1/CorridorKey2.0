"""Bounded birth-pinned stress descendants, sampled by the existing task owner.

This is observation-based containment, not a kernel job object: short-lived
unobserved forks and native query latency remain outside its guarantees.
"""
from __future__ import annotations

import os
import signal
import time
from typing import Any

from .process_memory import ProcessInfo, ProcessTreeMemory, _valid


class StressContainment:
    def __init__(self, root_pid: int, *, observer=None, root_exited=None, clock=time.monotonic, interval=.05):
        self.root_pid = root_pid
        self.root_exited = root_exited or (lambda: False)
        self.samples = 0
        self.complete_samples = 0
        self.terminal_exit_races = 0
        self.observer = observer if observer is not None else ProcessTreeMemory(root_pid=root_pid)
        self.clock, self.interval = clock, interval
        self.last_sample: float | None = None
        self.pins: dict[int, dict[str, Any]] = {}
        self.failure: str | None = None
        self.retired_count = 0

    def _fail(self, reason):
        self.failure = self.failure or str(reason)[:2048]

    def _read(self, pid):
        with self.observer.lock:
            info = self.observer.provider.read(pid)
        return None if info is None else _valid(info, pid)

    def _retire_confirmed_exits(self):
        if os.name == 'nt':
            return  # Direct Windows provider verification remains unsupported.
        for pid, pinned in tuple(self.pins.items()):
            if pid == self.root_pid:
                continue
            try:
                current = self._read(pid)
                if current is None or current.birth_identity != pinned['birth_identity']:
                    del self.pins[pid]
                    self.retired_count += 1
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                # Unknown is still owned/uncertain, never silently retired.
                self._fail(f'Pinned identity retirement unverified: {type(exc).__name__}: {exc}')

    def sample(self, *, force=False):
        if self.root_exited():
            if self.samples == 0:
                self._fail('Stress root exited before initial descendant discovery')
            return
        now = self.clock()
        if not force and self.last_sample is not None and now - self.last_sample < self.interval:
            return
        self.last_sample = now
        try:
            self._retire_confirmed_exits()
            data = self.observer.sample()
            self.samples += 1
            for row in data['processes']:
                pid = row['pid']
                birth = row['birth_identity']
                _valid(ProcessInfo(pid, row['parent_pid'], birth, row['rss_bytes']), pid)
                prior = self.pins.get(pid)
                if prior is not None and prior['birth_identity'] != birth:
                    # The original identity ended. A newly discovered process
                    # may be admitted only through the current observer tree.
                    self.pins.pop(pid)
                    self.retired_count += 1
                if len(self.pins) >= 256 and pid not in self.pins:
                    raise ValueError('Stress containment exceeds 256 tracked identities')
                depth = self.observer.depths.get(pid, 0)
                self.pins[pid] = {**row, 'depth': depth}
            if data['status'] == 'available':
                self.complete_samples += 1
            elif (data.get('reason_code') == 'root_exited' and self.complete_samples > 0
                  and self.root_exited()):
                # The Popen owner confirms normal disappearance after an earlier
                # complete discovery. RSS stays unavailable; existing/new pins
                # above still undergo termination and final group/exit checks.
                # Reuse, native denial/deadline and partial child discovery never
                # enter this branch and remain permanently fail-closed.
                self.terminal_exit_races += 1
            else:
                self._fail(str(data.get('reason') or 'incomplete native process discovery'))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self._fail(f'{type(exc).__name__}: {exc}')

    def terminate_descendants(self):
        if os.name == 'nt':
            self._fail('Windows birth-pinned stress descendant termination is NOT VERIFIED')
            return
        for pid, pinned in sorted(self.pins.items(), key=lambda item: item[1]['depth'], reverse=True):
            if pid == self.root_pid:
                continue
            try:
                current = self._read(pid)
                if current is None or current.birth_identity != pinned['birth_identity']:
                    continue
                group = os.getpgid(pid)
                again = self._read(pid)
                if again is None or again.birth_identity != pinned['birth_identity']:
                    continue
                if group == pid:
                    os.killpg(group, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                self._fail(f'Descendant termination unverified: {type(exc).__name__}: {exc}')

    def verify(self, *, timeout=2.0):
        if not self.root_exited():
            self._fail('The owned CLI has not been confirmed terminal by its Popen owner')
        if os.name == 'nt':
            self._fail('Windows birth-pinned stress descendant verification is NOT VERIFIED')
            raise RuntimeError('Stress cleanup NOT VERIFIED: ' + str(self.failure))
        deadline = self.clock() + timeout
        live = []
        while True:
            live = []
            for pid, pinned in self.pins.items():
                if pid == self.root_pid:
                    continue
                try:
                    current = self._read(pid)
                    if current is not None and current.birth_identity == pinned['birth_identity']:
                        live.append(pid)
                except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                    self._fail(f'Descendant exit verification failed: {type(exc).__name__}: {exc}')
            if not live or self.clock() >= deadline:
                break
            time.sleep(.01)
        if live:
            self._fail(f'Tracked stress descendants remain live: {live}')
        if os.name != 'nt':
            try:
                os.killpg(self.root_pid, 0)
            except ProcessLookupError:
                pass
            except OSError as exc:
                self._fail(f'Former stress process-group query failed: {exc}')
            else:
                # No living leader authority remains after Popen reaping.
                # Never signal a possibly reused process-group identifier.
                self._fail('Former CLI process group remains present; ownership cannot be safely authorized')
        if self.failure is not None:
            raise RuntimeError('Stress cleanup NOT VERIFIED: ' + self.failure)

    def snapshot(self):
        return {'tracked_processes': [dict(row) for row in self.pins.values()], 'failure': self.failure,
                'retired_process_count': self.retired_count, 'complete_samples': self.complete_samples,
                'terminal_exit_race_samples': self.terminal_exit_races,
                'scope': 'Birth-pinned observed descendants only; unobserved short-lived forks and native query latency remain outside guarantees.'}
