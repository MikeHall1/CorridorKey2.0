"""Pinned identity safety and actual escaped stress-worker force cleanup."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from corridorkey.benchmark_task import BenchmarkTask
from corridorkey.process_memory import ProcessInfo
from corridorkey.stress_containment import StressContainment


class Provider:
    def __init__(self):
        self.records = {100: ProcessInfo(100, 1, 'root', 10), 101: ProcessInfo(101, 100, 'owned', 10)}

    def read(self, pid):
        return self.records.get(pid)


class Observer:
    def __init__(self):
        self.lock = threading.Lock()
        self.provider = Provider()
        self.depths = {100: 0, 101: 1}
        self.status = 'available'

    def sample(self):
        return {'status': self.status, 'reason': 'Injected denied process discovery',
                'processes': [{'pid': row.pid, 'parent_pid': row.parent_pid, 'birth_identity': row.birth_identity,
                               'rss_bytes': row.rss_bytes} for row in self.provider.records.values()]}


@unittest.skipIf(os.name == 'nt', 'Windows stress containment NOT VERIFIED')
class StressContainmentTests(unittest.TestCase):
    def test_reused_pid_never_authorizes_kill_of_unrelated_process(self):
        observer = Observer()
        ended = [False]
        owner = StressContainment(100, observer=observer, root_exited=lambda: ended[0])
        owner.sample(force=True)
        observer.provider.records[101] = ProcessInfo(101, 1, 'unrelated-new-birth', 10)
        with patch('corridorkey.stress_containment.os.kill') as kill, patch('corridorkey.stress_containment.os.killpg') as group:
            owner.terminate_descendants()
            kill.assert_not_called()
            group.assert_not_called()
        ended[0] = True
        with patch('corridorkey.stress_containment.os.killpg', side_effect=ProcessLookupError):
            owner.verify()

    def test_partial_native_discovery_cannot_claim_clean_containment(self):
        observer = Observer()
        observer.status = 'unavailable'
        owner = StressContainment(100, observer=observer)
        owner.sample(force=True)
        observer.provider.records.clear()
        with patch('corridorkey.stress_containment.os.killpg', side_effect=ProcessLookupError), self.assertRaisesRegex(RuntimeError, 'NOT VERIFIED'):
            owner.verify()

    def test_malformed_native_identity_retains_pin_without_signal_authority(self):
        for invalid in (ProcessInfo(101, 100, '', 10), ProcessInfo(101, -1, 'owned', 10),
                        ProcessInfo(101, 100, 'owned', -1)):
            with self.subTest(invalid=invalid):
                observer = Observer()
                owner = StressContainment(100, observer=observer)
                owner.sample(force=True)
                observer.provider.records[101] = invalid
                owner.sample(force=True)
                self.assertIn(101, owner.pins)
                self.assertEqual(owner.pins[101]['birth_identity'], 'owned')
                self.assertIsNotNone(owner.failure)
                with patch('corridorkey.stress_containment.os.kill') as kill, patch('corridorkey.stress_containment.os.killpg') as group:
                    owner.terminate_descendants()
                    kill.assert_not_called()
                    group.assert_not_called()

    def test_more_than_256_sequential_children_retire_only_confirmed_identities(self):
        observer = Observer()
        owner = StressContainment(100, observer=observer)
        for index in range(300):
            pid = 101 + index
            observer.provider.records = {100: ProcessInfo(100, 1, 'root', 10),
                                         pid: ProcessInfo(pid, 100, f'child-{index}', 10)}
            observer.depths = {100: 0, pid: 1}
            owner.sample(force=True)
            self.assertIsNone(owner.failure)
            self.assertEqual(len(owner.pins), 2)
        self.assertEqual(owner.retired_count, 299)
        last_pid = 400
        original_read = observer.provider.read

        def denied(pid):
            if pid == last_pid:
                raise PermissionError('Injected native query denial')
            return original_read(pid)

        with patch.object(observer.provider, 'read', side_effect=denied):
            owner.sample(force=True)
        self.assertIn(last_pid, owner.pins)
        self.assertIn('unverified', str(owner.failure))

    def test_sample_reaped_cli_never_authorizes_root_group_signal(self):
        task = object.__new__(BenchmarkTask)
        process = Mock()
        process.pid = 100
        process.poll.return_value = None
        task.process = process
        task.stress_mode = True
        task.containment = Mock()
        self.assertIsNone(process.poll())  # The task loop admitted this iteration.

        def reap_during_sample():
            process.poll.return_value = 0

        task.containment.sample.side_effect = reap_during_sample
        task.containment.sample()
        with patch('corridorkey.benchmark_task.os.killpg') as root_group:
            task._signal(force=True)
            root_group.assert_not_called()
        task.containment.terminate_descendants.assert_called_once()
        process.send_signal.assert_not_called()

    def test_actual_separate_session_worker_and_native_like_descendant_are_force_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'stress.json'
            pid_file, heartbeat = root / 'pids', root / 'heartbeat'
            manifest.write_text(json.dumps({'kind': 'production_stress', 'fixture_mode': 'escaped',
                                            'pid_file': str(pid_file), 'heartbeat': str(heartbeat)}))
            fixture = Path(__file__).with_name('stress_desktop_fixture.py')
            task = BenchmarkTask(manifest, root, grace_seconds=.1, command_prefix=[sys.executable, str(fixture)])
            try:
                deadline = time.monotonic() + 10
                tracked = []
                while time.monotonic() < deadline:
                    snapshot = task.snapshot()
                    tracked = snapshot.get('stress_containment', {}).get('tracked_processes', [])
                    if 'stress_escaped_ready' in snapshot['log_tail'] and len(tracked) >= 3:
                        break
                    if task.done.is_set():
                        self.fail(str(snapshot))
                    time.sleep(.01)
                self.assertTrue(pid_file.exists())
                self.assertGreaterEqual(len(tracked), 3)
                task.cancel()
                self.assertTrue(task.done.wait(10), task.snapshot())
                snapshot = task.snapshot()
                self.assertTrue(snapshot['forced'])
                self.assertIsNone(snapshot['cleanup_error'], snapshot)
                self.assertFalse(task.thread.is_alive())
                assert task.process is not None and task.reader is not None
                self.assertIsNotNone(task.process.poll())
                self.assertFalse(task.reader.is_alive())
                observed = heartbeat.read_text()
                time.sleep(.1)
                self.assertEqual(heartbeat.read_text(), observed)
                self.assertIsNotNone(task.containment)
                assert task.containment is not None
                for pid in map(int, pid_file.read_text().split(',')):
                    self.assertIsNone(task.containment._read(pid))
            finally:
                task.cancel()
                task.thread.join(10)
                if task.thread.is_alive():
                    task._signal(force=True)
                    task.thread.join(5)
                self.assertFalse(task.thread.is_alive())
