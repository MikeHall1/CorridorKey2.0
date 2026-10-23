"""Pinned process family sampling, failure completeness and owned native evidence."""
import json
import statistics
import subprocess
import sys
import time
import unittest
from typing import Any
from unittest.mock import patch

from corridorkey.control import CancellationToken, CancelledError
from corridorkey.process_memory import ProcessInfo, ProcessTreeMemory


def info(pid, parent=0, birth=None, rss=100):
    return ProcessInfo(pid, parent, str(pid) if birth is None else birth, rss)


class Provider:
    name = 'deterministic-fixture'

    def __init__(self, records):
        self.records = {record.pid: record for record in records}
        self.read_hook: Any = None
        self.children_hook: Any = None

    def read(self, pid):
        if self.read_hook is not None:
            self.read_hook(pid)
        return self.records.get(pid)

    def begin(self, tick):
        tick()

    def children(self, pid, tick):
        tick()
        if self.children_hook is not None:
            return self.children_hook(pid)
        return [record.pid for record in self.records.values() if record.parent_pid == pid]


class ProcessMemoryTests(unittest.TestCase):
    def test_complete_current_rss_and_reparented_child_descendants(self):
        provider = Provider([info(1), info(2, 1, rss=200)])
        observer = ProcessTreeMemory(1, provider=provider)
        first = observer.sample()
        self.assertEqual(first['current_rss_bytes'], 300)
        provider.records[2] = info(2, 99, rss=50)
        provider.records[3] = info(3, 2, rss=70)
        second = observer.sample()
        self.assertEqual(second['current_rss_bytes'], 220)
        self.assertEqual([p['pid'] for p in second['processes']], [1, 2, 3])
        self.assertLess(second['current_rss_bytes'], first['current_rss_bytes'])
        json.dumps(second, allow_nan=False)

    def test_confirmed_exit_and_pid_reuse_exclude_unrelated_process(self):
        provider = Provider([info(1), info(2, 1)])
        observer = ProcessTreeMemory(1, provider=provider)
        observer.sample()
        provider.records[2] = info(2, 99, birth='replacement', rss=99999)
        result = observer.sample()
        self.assertEqual(result['current_rss_bytes'], 100)
        self.assertNotIn(2, observer.known)
        provider.records[1] = info(1, birth='reused-root')
        result = observer.sample()
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['current_rss_bytes'])

    def test_denied_child_never_returns_partial_as_total(self):
        provider = Provider([info(1), info(2, 1)])
        observer = ProcessTreeMemory(1, provider=provider)
        def denied(pid):
            if pid == 2:
                raise PermissionError('query denied')
        provider.read_hook = denied
        result = observer.sample()
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['current_rss_bytes'])
        self.assertEqual(result['observed_rss_bytes'], 100)

    def test_constructor_failure_is_representable_and_bounded(self):
        provider = Provider([])
        observer = ProcessTreeMemory(1, provider=provider)
        self.assertEqual(observer.sample()['status'], 'unavailable')
        self.assertIsNone(observer.sample()['current_rss_bytes'])
        with patch('corridorkey.process_memory._native', side_effect=OSError('x' * 10000)):
            result = ProcessTreeMemory().sample()
        self.assertLessEqual(len(result['reason']), 256)
        self.assertLess(len(json.dumps(result)), 1500)

    def test_depth_node_deadline_and_enumeration_bounds(self):
        provider = Provider([info(1), info(2, 1), info(3, 2)])
        for kwargs in ({'max_depth': 1}, {'max_nodes': 2}):
            result = ProcessTreeMemory(1, provider=provider, **kwargs).sample()
            self.assertEqual(result['status'], 'unavailable')
            self.assertIsNone(result['current_rss_bytes'])
            self.assertLessEqual(result['process_count'], 2)
        clock = iter([0, 0, 2, 2, 2]).__next__
        self.assertEqual(ProcessTreeMemory(1, provider=provider, clock=clock).sample()['status'], 'unavailable')
        provider.children_hook = lambda pid: iter([2] * 100) if pid == 1 else ()
        self.assertEqual(ProcessTreeMemory(1, provider=provider, max_nodes=2).sample()['status'], 'unavailable')
        # Depth stays bounded across samples, even when already-known nodes seed traversal.
        provider.children_hook = None
        provider.records.pop(3)
        observer = ProcessTreeMemory(1, provider=provider, max_depth=1)
        self.assertEqual(observer.sample()['status'], 'available')
        provider.records[3] = info(3, 2)
        self.assertEqual(observer.sample()['status'], 'unavailable')

    def test_exit_and_parent_identity_races(self):
        provider = Provider([info(1), info(2, 1)])
        observer = ProcessTreeMemory(1, provider=provider)
        provider.children_hook = lambda pid: (2,) if pid == 1 else ()
        provider.records.pop(2)
        self.assertEqual(observer.sample()['current_rss_bytes'], 100)
        provider.records[2] = info(2, 9)
        self.assertEqual(observer.sample()['status'], 'unavailable')
        provider.records[2] = info(2, 1)
        def replace_parent(pid):
            if pid == 2:
                provider.records[1] = info(1, birth='new')
        provider.read_hook = replace_parent
        self.assertEqual(observer.sample()['status'], 'unavailable')

    def test_cancellation_and_invalid_values(self):
        provider = Provider([info(1)])
        observer = ProcessTreeMemory(1, provider=provider)
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(CancelledError):
            observer.sample(check=token.check)
        invalid_cases: list[dict[str, Any]] = [{'max_nodes': False}, {'max_nodes': 0}, {'max_depth': -1},
                       {'sample_timeout': float('nan')}, {'sample_timeout': float('inf')}]
        for kwargs in invalid_cases:
            with self.assertRaises(ValueError):
                ProcessTreeMemory(1, provider=provider, **kwargs)
        provider.records[1] = info(1, rss=-1)
        self.assertEqual(observer.sample()['status'], 'unavailable')

    def test_linux_bounded_stat_parser_and_missing_vs_denied(self):
        import io

        from corridorkey.process_memory import MAX_STAT_BYTES, _Linux
        # A command containing ')' and spaces must not shift field indices.
        fields = ['S', '1'] + ['0'] * 17 + ['12345', '0', '7']
        data = ('22 (worker ) name) ' + ' '.join(fields)).encode()
        provider = _Linux(4)
        with patch('pathlib.Path.open', return_value=io.BytesIO(data)):
            record = provider.read(22)
        self.assertEqual(record, ProcessInfo(22, 1, '12345', 7 * provider.page_size))
        with patch('pathlib.Path.open', side_effect=FileNotFoundError):
            self.assertIsNone(provider.read(22))
        with patch('pathlib.Path.open', side_effect=PermissionError), self.assertRaises(PermissionError):
            provider.read(22)
        with (patch('pathlib.Path.open', return_value=io.BytesIO(b'x' * (MAX_STAT_BYTES + 1))),
              self.assertRaises(RuntimeError)):
            provider.read(22)

    @unittest.skipUnless(sys.platform == 'darwin', 'actual libproc evidence requires macOS')
    def test_actual_owned_allocated_child_and_observer_overhead(self):
        observer = ProcessTreeMemory()
        baseline = observer.sample()
        self.assertEqual(baseline['status'], 'available', baseline)
        child = subprocess.Popen([sys.executable, '-u', '-c',
            "import sys; data=bytearray(32*1024*1024); print('ready',flush=True); sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert child.stdout is not None
            self.assertEqual(child.stdout.readline(), b'ready\n')
            result = observer.sample()
            self.assertEqual(result['status'], 'available', result)
            owned = next(item for item in result['processes'] if item['pid'] == child.pid)
            self.assertGreater(owned['rss_bytes'], 24 * 1024 * 1024)
            self.assertGreater(result['current_rss_bytes'], baseline['current_rss_bytes'] + 24 * 1024 * 1024)
            timings = []
            for _ in range(20):
                started = time.perf_counter()
                self.assertEqual(observer.sample()['status'], 'available')
                timings.append(time.perf_counter() - started)
            print(f'Owned libproc observer: n=20 median={statistics.median(timings):.6f}s max={max(timings):.6f}s child_rss={owned["rss_bytes"]}')
        finally:
            if child.stdin is not None:
                child.stdin.close()
            child.wait(timeout=5)
            for stream in (child.stdout, child.stderr):
                if stream is not None:
                    stream.close()
        self.assertIsNotNone(child.returncode)
        self.assertEqual(observer.sample()['process_count'], baseline['process_count'])
        self.assertNotIn(child.pid, observer.known)
