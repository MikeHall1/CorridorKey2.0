"""Bounded cooperative current-RSS observation of a pinned process family.

Samples are not atomic snapshots, hard RAM limits or unique physical memory:
shared resident pages can be counted in multiple processes. Native OS calls
cannot be interrupted by our deadline. Short-lived/unobserved reparented children
can escape discovery. No process is adopted from a persisted PID file.
"""
from __future__ import annotations

import ctypes
import errno
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .control import CancelledError

MAX_SYSTEM_ENTRIES = 65536
MAX_STAT_BYTES = 65536


class MeasurementError(RuntimeError):
    def __init__(self, message: str, *, reason_code: str = 'measurement_unavailable'):
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    parent_pid: int
    birth_identity: str
    rss_bytes: int


def _valid(record, pid):
    if (not isinstance(record, ProcessInfo) or record.pid != pid
            or type(record.pid) is not int or not 0 < record.pid < 2**31
            or type(record.parent_pid) is not int or not 0 <= record.parent_pid < 2**31
            or type(record.rss_bytes) is not int or not 0 <= record.rss_bytes < 2**63
            or not isinstance(record.birth_identity, str) or not 0 < len(record.birth_identity) <= 128):
        raise MeasurementError("Invalid process measurement")
    return record


class _Linux:
    name = 'linux-proc-current-rss (target-platform validation pending)'

    def __init__(self, max_nodes):
        self.page_size = os.sysconf('SC_PAGE_SIZE')
        self.parents = {}

    def read(self, pid):
        try:
            with (Path('/proc') / str(pid) / 'stat').open('rb') as stream:
                data = stream.read(MAX_STAT_BYTES + 1)
        except (FileNotFoundError, ProcessLookupError):
            return None
        if len(data) > MAX_STAT_BYTES:
            raise MeasurementError('Process stat exceeds read bound')
        end = data.rfind(b')')
        fields = data[end + 2:].split()
        if end < 0 or len(fields) < 22:
            raise MeasurementError('Malformed process stat')
        # Fields after comm begin at field3(state); ppid4/starttime22/rss24.
        return ProcessInfo(pid, int(fields[1]), fields[19].decode('ascii'), int(fields[21]) * self.page_size)

    def begin(self, tick):
        self.parents = {}
        with os.scandir('/proc') as entries:
            for count, entry in enumerate(entries):
                tick()
                if count >= MAX_SYSTEM_ENTRIES:
                    raise MeasurementError('System process enumeration bound exceeded')
                if entry.name.isascii() and entry.name.isdecimal():
                    pid = int(entry.name)
                    info = self.read(pid)
                    if info is not None:
                        _valid(info, pid)
                        self.parents.setdefault(info.parent_pid, []).append(pid)

    def children(self, pid, tick):
        tick()
        return self.parents.get(pid, ())


class _BsdInfo(ctypes.Structure):
    # macOS SDK sys/proc_info.h proc_bsdinfo; MAXCOMLEN is16.
    _fields_ = [(name, ctypes.c_uint32) for name in (
        'flags', 'status', 'xstatus', 'pid', 'ppid', 'uid', 'gid', 'ruid', 'rgid',
        'svuid', 'svgid', 'reserved')] + [
        ('comm', ctypes.c_char * 16), ('name', ctypes.c_char * 32)] + [
        (name, ctypes.c_uint32) for name in ('nfiles', 'pgid', 'pjobc', 'tdev', 'tpgid')] + [
        ('nice', ctypes.c_int32), ('start_sec', ctypes.c_uint64), ('start_usec', ctypes.c_uint64)]


class _TaskInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        'virtual_size', 'resident_size', 'total_user', 'total_system', 'threads_user', 'threads_system')] + [
        (f'counter_{i}', ctypes.c_int32) for i in range(12)]


class _TaskAll(ctypes.Structure):
    _fields_ = [('bsd', _BsdInfo), ('task', _TaskInfo)]


class _Mac:
    name = 'macos-libproc-taskallinfo-current-rss'

    def __init__(self, max_nodes):
        self.lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_pidinfo.restype = ctypes.c_int
        self.lib.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_listchildpids.restype = ctypes.c_int
        self.capacity = max_nodes + 1

    def read(self, pid):
        result = _TaskAll()
        ctypes.set_errno(0)
        size = self.lib.proc_pidinfo(pid, 2, 0, ctypes.byref(result), ctypes.sizeof(result))
        if size != ctypes.sizeof(result):
            code = ctypes.get_errno()
            if size <= 0 and code in (errno.ESRCH, errno.ENOENT):
                return None
            raise OSError(code, 'Incomplete libproc process measurement')
        if result.bsd.pid != pid:
            raise MeasurementError('libproc PID identity mismatch')
        return ProcessInfo(pid, result.bsd.ppid, f'{result.bsd.start_sec}:{result.bsd.start_usec}',
                           result.task.resident_size)

    def begin(self, tick):
        tick()

    def children(self, pid, tick):
        tick()
        array = (ctypes.c_int * self.capacity)()
        ctypes.set_errno(0)
        count = self.lib.proc_listchildpids(pid, array, ctypes.sizeof(array))
        tick()
        if count < 0 or (count == 0 and ctypes.get_errno() != 0):
            raise OSError(ctypes.get_errno(), 'libproc child enumeration failed')
        # proc_listchildpids returns PID count (not bytes); a full buffer may truncate.
        if count >= self.capacity:
            raise MeasurementError('Child enumeration bound exceeded')
        return tuple(array[i] for i in range(count) if array[i] > 0)


class _Windows:
    name = 'windows-toolhelp-working-set (target-platform validation pending)'

    def __init__(self, max_nodes):
        from ctypes import wintypes as w
        self.w = w
        windows = vars(ctypes)
        self.last_error = windows['get_last_error']
        self.kernel = windows['WinDLL']('kernel32', use_last_error=True)
        self.psapi = windows['WinDLL']('psapi', use_last_error=True)

        class Entry(ctypes.Structure):
            _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD), ('pid', w.DWORD),
                        ('heap', ctypes.c_size_t), ('module', w.DWORD), ('threads', w.DWORD),
                        ('ppid', w.DWORD), ('priority', w.LONG), ('flags', w.DWORD), ('exe', w.WCHAR * 260)]

        class Memory(ctypes.Structure):
            _fields_ = [('cb', w.DWORD), ('faults', w.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak', 'working', 'paged_peak', 'paged',
                    'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]

        self.Entry, self.Memory = Entry, Memory
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.CloseHandle.restype = w.BOOL
        self.kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
        self.kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
        for name in ('Process32FirstW', 'Process32NextW'):
            fn = getattr(self.kernel, name)
            fn.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
            fn.restype = w.BOOL
        self.kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = w.BOOL
        self.kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        self.kernel.GetExitCodeProcess.restype = w.BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(Memory), w.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = w.BOOL
        self.parents = {}
        self.ppids = {}
        self.tick = lambda: None

    def begin(self, tick):
        self.tick = tick
        self.parents, self.ppids = {}, {}
        snapshot = self.kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise OSError(self.last_error(), 'Toolhelp snapshot unavailable')
        try:
            entry = self.Entry()
            entry.dwSize = ctypes.sizeof(entry)
            found = self.kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            count = 0
            while found:
                tick()
                count += 1
                if count > MAX_SYSTEM_ENTRIES:
                    raise MeasurementError('System process enumeration bound exceeded')
                self.parents.setdefault(entry.ppid, []).append(entry.pid)
                self.ppids[entry.pid] = entry.ppid
                found = self.kernel.Process32NextW(snapshot, ctypes.byref(entry))
            if self.last_error() != 18:  # ERROR_NO_MORE_FILES only.
                raise OSError(self.last_error(), 'Incomplete Toolhelp enumeration')
        finally:
            self.kernel.CloseHandle(snapshot)

    def children(self, pid, tick):
        tick()
        return self.parents.get(pid, ())

    def read(self, pid):
        handle = self.kernel.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            code = self.last_error()
            if code == 87:  # ERROR_INVALID_PARAMETER: no such PID.
                return None
            raise OSError(code, 'Process query unavailable')
        try:
            times = [self.w.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                raise OSError(self.last_error(), 'Process creation identity unavailable')
            memory = self.Memory()
            memory.cb = ctypes.sizeof(memory)
            if not self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), ctypes.sizeof(memory)):
                raise OSError(self.last_error(), 'Current working set unavailable')
            birth = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            prior_parent = self.ppids.get(pid)
            # Toolhelp rows lack birth identity. A fresh snapshot after opening
            # the process detects stale PPID/reused PID associations; this open
            # handle pins the queried birth, and exit is checked after refresh.
            self.begin(self.tick)
            current_parent = self.ppids.get(pid)
            exit_code = self.w.DWORD()
            if not self.kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise OSError(self.last_error(), 'Process exit state unavailable')
            if exit_code.value != 259:
                return None
            if current_parent is None or (prior_parent is not None and current_parent != prior_parent):
                raise MeasurementError('Toolhelp parent changed during live query')
            return ProcessInfo(pid, current_parent, str(birth), memory.working)
        finally:
            self.kernel.CloseHandle(handle)


def _native(max_nodes):
    if sys.platform == 'darwin':
        return _Mac(max_nodes)
    if sys.platform.startswith('linux'):
        return _Linux(max_nodes)
    if sys.platform == 'win32':
        return _Windows(max_nodes)
    raise MeasurementError('Current process-tree RSS provider is unsupported')


class ProcessTreeMemory:
    """Observe only this root identity and descendants discovered during this life.

    Injectable providers implement read(pid)->ProcessInfo|None, begin(tick), and
    children(pid,tick)->iterable[int]. None means confirmed exit, not permission
    denial. Deadlines are checked between native calls, never a hard OS timeout.
    """
    def __init__(self, root_pid=None, *, max_nodes=256, max_depth=32, sample_timeout=0.25,
                 provider=None, clock=time.monotonic):
        root_pid = os.getpid() if root_pid is None else root_pid
        if (type(root_pid) is not int or not 0 < root_pid < 2**31
                or type(max_nodes) is not int or not 1 <= max_nodes <= 4096
                or type(max_depth) is not int or not 0 <= max_depth <= 128
                or isinstance(sample_timeout, bool) or not isinstance(sample_timeout, (int, float))
                or not 0 < sample_timeout <= 60):
            raise ValueError('Invalid process observer bounds')
        self.root_pid, self.max_nodes, self.max_depth = root_pid, max_nodes, max_depth
        self.timeout, self.clock = sample_timeout, clock
        self.provider: Any = provider
        self.known = {}
        self.depths = {}
        self.root_identity = None
        self.initial_error = None
        self.initial_reason_code = 'measurement_unavailable'
        self.lock = threading.Lock()
        try:
            self.provider = _native(max_nodes) if provider is None else provider
            info = self.provider.read(root_pid)
            if info is None:
                raise MeasurementError('Root process has exited', reason_code='root_exited')
            self.root_identity = _valid(info, root_pid).birth_identity
            self.known[root_pid] = self.root_identity
            self.depths[root_pid] = 0
        except (OSError, ValueError, TypeError, AttributeError, OverflowError, MeasurementError) as exc:
            self.initial_error = f'{type(exc).__name__}: {exc}'[:256]
            self.initial_reason_code = getattr(exc, 'reason_code', 'measurement_unavailable')

    def sample(self, *, check=None) -> dict[str, Any]:
        with self.lock:
            return self._sample(check)

    def _sample(self, check):
        started = self.clock()
        records = {}
        reason = None
        reason_code = None

        def tick():
            if check is not None:
                check()
            if self.clock() - started > self.timeout:
                raise MeasurementError('Cooperative sample deadline exceeded')

        def read(pid):
            tick()
            info = self.provider.read(pid)
            tick()
            return None if info is None else _valid(info, pid)

        try:
            tick()
            if self.initial_error is not None:
                raise MeasurementError(self.initial_error, reason_code=self.initial_reason_code)
            self.provider.begin(tick)
            tick()
            # Known descendants remain owned across reparenting, until exit/reuse.
            pending = []
            for pid, identity in tuple(self.known.items()):
                info = read(pid)
                if info is None or info.birth_identity != identity:
                    if pid == self.root_pid:
                        raise MeasurementError('Root process exited' if info is None else 'Root process identity changed',
                                               reason_code='root_exited' if info is None else 'root_identity_changed')
                    del self.known[pid]
                    self.depths.pop(pid, None)
                else:
                    pending.append((info, self.depths[pid]))
            while pending:
                info, depth = pending.pop()
                if info.pid in records:
                    continue
                if len(records) >= self.max_nodes:
                    raise MeasurementError('Owned process count bound exceeded')
                records[info.pid] = info
                for index, pid in enumerate(self.provider.children(info.pid, tick)):
                    tick()
                    if index >= self.max_nodes:
                        raise MeasurementError('Child enumeration bound exceeded')
                    if type(pid) is not int or not 0 < pid < 2**31:
                        raise MeasurementError('Invalid child PID')
                    child = read(pid)
                    if child is None:
                        continue
                    parent = read(info.pid)
                    if parent is None or parent.birth_identity != info.birth_identity:
                        raise MeasurementError('Parent identity changed during discovery')
                    if child.parent_pid != info.pid:
                        raise MeasurementError('Child parent changed during discovery')
                    existing = self.known.get(pid)
                    if existing is not None and existing != child.birth_identity:
                        raise MeasurementError('Child PID reused during discovery')
                    if pid not in self.known:
                        if depth >= self.max_depth or len(self.known) >= self.max_nodes:
                            raise MeasurementError('Owned tree depth or node bound exceeded')
                        self.known[pid] = child.birth_identity
                        self.depths[pid] = depth + 1
                        pending.append((child, depth + 1))
            # Revalidate after discovery. Exits are valid; reused PIDs are excluded.
            for pid, prior in tuple(records.items()):
                latest = read(pid)
                if latest is None or latest.birth_identity != prior.birth_identity:
                    if pid == self.root_pid:
                        raise MeasurementError('Root process exited before sample completion' if latest is None else 'Root identity changed before sample completion',
                                               reason_code='root_exited' if latest is None else 'root_identity_changed')
                    records.pop(pid)
                    self.known.pop(pid, None)
                    self.depths.pop(pid, None)
                else:
                    records[pid] = latest
            tick()
        except CancelledError:
            raise
        except (OSError, ValueError, TypeError, AttributeError, OverflowError, MeasurementError) as exc:
            reason = f'{type(exc).__name__}: {exc}'[:256]
            reason_code = getattr(exc, 'reason_code', 'measurement_unavailable')
        observed = sum(item.rss_bytes for item in records.values())
        return {'status': 'available' if reason is None else 'unavailable',
                'current_rss_bytes': observed if reason is None else None,
                'observed_rss_bytes': observed, 'process_count': len(records),
                'sampled_monotonic': started, 'duration_seconds': max(0.0, self.clock() - started),
                'provider': str(getattr(self.provider, 'name', 'injected'))[:128], 'reason': reason,
                'reason_code': reason_code,
                'processes': [{'pid': i.pid, 'birth_identity': i.birth_identity,
                               'parent_pid': i.parent_pid, 'rss_bytes': i.rss_bytes}
                              for i in sorted(records.values(), key=lambda i: i.pid)],
                'scope': 'Sampled sum of current root and observed descendant RSS; shared pages may count multiple times. Cooperative stop observation, not a hard RAM cap or atomic snapshot; native calls and unobserved short-lived children remain outside guarantees.'}
