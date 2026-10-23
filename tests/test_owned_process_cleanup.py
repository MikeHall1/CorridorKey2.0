"""Reaped numeric process IDs never authorize signals to an unknown group."""
import logging
import os
import queue
import subprocess
import sys
import threading
import unittest
from typing import Any
from unittest.mock import Mock, patch

from corridorkey.benchmark_task import BenchmarkTask
from corridorkey.worker_client import WorkerCleanupError, WorkerClient, WorkerLost


@unittest.skipUnless(os.name == 'posix', 'POSIX process-group ownership contract')
class OwnedCleanupTests(unittest.TestCase):
    def test_thread_start_failure_reaps_child_and_joins_started_owner(self):
        processes = []
        owners = []
        real_spawn = subprocess.Popen
        real_start = threading.Thread.start

        def spawn(*args, **kwargs):
            process = real_spawn(*args, **kwargs)
            processes.append(process)
            return process

        def start(owner):
            if owner.name == 'worker-events':
                raise RuntimeError('injected reader thread start failure')
            owners.append(owner)
            real_start(owner)

        try:
            with patch('corridorkey.worker_client.subprocess.Popen', side_effect=spawn), \
                    patch('corridorkey.worker_client.threading.Thread.start', new=start), \
                    self.assertRaisesRegex(RuntimeError, 'injected reader thread start failure'):
                WorkerClient(command=[sys.executable, '-c', 'import time; time.sleep(30)'],
                             shutdown_timeout=.05)
            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].poll())
            self.assertTrue(owners)
            self.assertTrue(all(not owner.is_alive() for owner in owners))
            self.assertTrue(all(pipe is not None and pipe.closed for pipe in
                                (processes[0].stdin, processes[0].stdout, processes[0].stderr)))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(3)
            for owner in owners:
                owner.join(3)

    def test_constructor_preserves_uncertain_cleanup_but_clean_failure_is_original(self):
        original = WorkerLost('invalid initial handshake')

        def uncertain(client):
            client.cleanup_error = 'Owned reader remains live'
            return False

        def interrupted(client):
            raise OSError('owned close failed')

        for close, expected in ((uncertain, 'Owned reader remains live'),
                                (interrupted, 'owned close failed'),
                                (lambda client: False, None)):
            with self.subTest(expected=expected), \
                    patch('corridorkey.worker_client.subprocess.Popen'), \
                    patch('corridorkey.worker_client.threading.Thread'), \
                    patch.object(WorkerClient, 'next_event', side_effect=original), \
                    patch.object(WorkerClient, 'close', autospec=True, side_effect=close):
                if expected is None:
                    with self.assertRaises(WorkerLost) as clean_raised:
                        WorkerClient()
                    self.assertIs(clean_raised.exception, original)
                else:
                    with self.assertRaises(WorkerCleanupError) as cleanup_raised:
                        WorkerClient()
                    self.assertIn(expected, cleanup_raised.exception.cleanup_error)
                    self.assertIs(cleanup_raised.exception.__cause__, original)

    def client(self, *, live=False):
        client: Any = WorkerClient.__new__(WorkerClient)
        client.closed = client.expected_close = False
        client.failure = client.cleanup_error = client.containment = None
        client.clean_stop = True
        client.kill_lock = threading.Lock()
        client.shutdown_timeout = .01
        client.outgoing = queue.Queue(8)
        client.process = Mock(pid=424242, returncode=None if live else 0)
        client.process.poll.side_effect = lambda: client.process.returncode

        def wait(timeout=None):
            if client.process.returncode is None:
                client.process.returncode = 0
            return client.process.returncode

        client.process.wait.side_effect = wait
        client.process.stdin = client.process.stdout = client.process.stderr = None
        for name in ('writer', 'reader', 'log_reader'):
            owner = Mock()
            owner.is_alive.return_value = False
            setattr(client, name, owner)
        return client

    def test_reaped_worker_unknown_group_never_receives_kill_or_restart(self):
        client = self.client()
        # A late native stopped event cannot erase independently latched
        # descendant cleanup uncertainty while the reader is being joined.
        client.reader.join.side_effect = lambda timeout: setattr(client, 'clean_stop', True)
        with patch('corridorkey.worker_client.os.killpg') as signal_group:
            self.assertFalse(client.close())
        self.assertEqual([call.args for call in signal_group.call_args_list], [(424242, 0)])
        self.assertIn('NOT VERIFIED', client.cleanup_error)
        with self.assertRaisesRegex(RuntimeError, 'cleanup is unverified'):
            client.restart()

    def test_close_wait_cannot_reap_during_failure_owner_root_signal(self):
        client = self.client(live=True)
        client.send = Mock()
        client._capture_descendants = lambda: None
        waiting, release_wait, reader_attempt = threading.Event(), threading.Event(), threading.Event()
        errors, signals = [], []

        def wait(timeout=None):
            if threading.current_thread().name == 'close-owner':
                waiting.set()
                if not release_wait.wait(2):
                    raise RuntimeError('test did not release owned wait')
                client.process.returncode = 0
            return client.process.returncode

        def signal_group(pid, sig):
            signals.append((int(sig), client.process.returncode))
            if sig == 0:
                raise ProcessLookupError

        def run(close):
            try:
                if close:
                    client.close()
                else:
                    reader_attempt.set()
                    client._terminate_owned()
            except BaseException as exc:
                logging.getLogger(__name__).exception('Owned cleanup interleaving failed')
                errors.append(exc)

        client.process.wait.side_effect = wait
        closer = threading.Thread(target=lambda: run(True), name='close-owner')
        reader = threading.Thread(target=lambda: run(False), name='reader-failure')
        with patch('corridorkey.worker_client.os.killpg', side_effect=signal_group):
            closer.start()
            try:
                self.assertTrue(waiting.wait(2))
                acquired = client.kill_lock.acquire(blocking=False)
                if acquired:
                    client.kill_lock.release()
                self.assertFalse(acquired, 'Class-owned wait must hold root-signal authority until reaped')
                reader.start()
                self.assertTrue(reader_attempt.wait(2))
            finally:
                release_wait.set()
                closer.join(3)
                if reader.ident is not None:
                    reader.join(3)
        self.assertFalse(closer.is_alive() or reader.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(signals)
        self.assertTrue(all(sig == 0 and returncode == 0 for sig, returncode in signals))

    def test_cleanly_reaped_crash_is_distinct_from_unverified_cleanup(self):
        client = self.client()
        client.failure = 'Observed worker crash; explicit reconnect required'
        client.process.returncode = 78
        with patch('corridorkey.worker_client.os.killpg', side_effect=ProcessLookupError):
            self.assertFalse(client.close())
        self.assertIsNone(client.cleanup_error)

    def test_capture_precedes_cooperative_wait_and_pinned_cleanup_follows(self):
        client = self.client(live=True)
        trace = []
        observer = Mock()
        observer.sample.side_effect = lambda **kwargs: trace.append('capture')
        observer.terminate_descendants.side_effect = lambda: trace.append('terminate-pinned')
        observer.verify.side_effect = lambda: trace.append('verify')
        original_wait = client.process.wait.side_effect

        def wait(timeout=None):
            trace.append('wait')
            return original_wait(timeout)

        client.process.wait.side_effect = wait
        client.send = Mock()
        with patch('corridorkey.stress_containment.StressContainment', return_value=observer):
            self.assertTrue(client.close())
        self.assertEqual(trace, ['capture', 'wait', 'terminate-pinned', 'verify'])
        client.send.assert_called_once_with('shutdown', {'drain': False})

    def test_live_buffered_reader_is_not_cross_thread_closed(self):
        client = self.client()
        pipe = client.process.stdout = Mock()
        client.reader.is_alive.return_value = True
        with patch('corridorkey.worker_client.os.killpg', side_effect=ProcessLookupError):
            self.assertFalse(client.close())
        self.assertIn('I/O owner remains live', client.cleanup_error)
        pipe.close.assert_not_called()

    def test_ordinary_benchmark_reaped_root_never_authorizes_group_signal(self):
        task: Any = BenchmarkTask.__new__(BenchmarkTask)
        task.process = Mock(pid=424242)
        task.process.poll.return_value = 0
        task.process.stdout = Mock()
        task.reader = Mock()
        task.reader.is_alive.return_value = False
        task.containment = None
        task.stress_mode = False
        with patch('corridorkey.benchmark_task.os.killpg') as signal_group:
            task._signal(force=False)
            task._signal(force=True)
            signal_group.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'without living leader'):
                task._cleanup()
        self.assertEqual([call.args for call in signal_group.call_args_list], [(424242, 0)])
        task.process.stdout.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
