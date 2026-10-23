import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from corridorkey.benchmark_report import export_report, read_report
from corridorkey.benchmark_task import BenchmarkTask

FIXTURE = Path(__file__).with_name('benchmark_fixture.py')


class BenchmarkTaskTests(unittest.TestCase):
    def task(self, root, mode):
        manifest = root / 'suite.json'
        manifest.write_text(json.dumps({'mode': mode}))
        task = BenchmarkTask(manifest, root, grace_seconds=.1, command_prefix=[sys.executable, str(FIXTURE)])
        self.addCleanup(task.thread.join, 10)
        self.addCleanup(task.cancel)
        return task

    def test_actual_process_report_export_and_bounded_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = self.task(root, 'logs')
            self.assertTrue(task.done.wait(10))
            snapshot = task.snapshot()
            self.assertEqual(snapshot['status'], 'complete')
            self.assertLessEqual(len(snapshot['log_tail']), 64 * 4096)
            self.assertIsNone(snapshot['cleanup_error'])
            report, raw = read_report(task.report_path)
            self.assertTrue(report['test_fixture'])
            target = root / 'export.json'
            export_report(task.report_path, target)
            self.assertEqual(target.read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                export_report(task.report_path, target)

    def test_cancel_ignoring_child_forces_and_reaps_without_fake_report(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(Path(folder), 'wait')
            deadline = time.monotonic() + 10
            while 'fixture_ready' not in task.snapshot()['log_tail'] and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIn('fixture_ready', task.snapshot()['log_tail'])
            task.cancel()
            self.assertTrue(task.done.wait(10))
            self.assertTrue(task.snapshot()['forced'])
            self.assertFalse(task.snapshot()['report_available'])
            self.assertIsNotNone(task.snapshot()['returncode'])
            self.assertIsNotNone(task.process)
            assert task.process is not None
            self.assertIsNotNone(task.process.poll())

    def test_hard_exit_is_terminal_and_not_restarted(self):
        with tempfile.TemporaryDirectory() as folder:
            task = self.task(Path(folder), 'crash')
            self.assertTrue(task.done.wait(10))
            snapshot = task.snapshot()
            self.assertEqual(snapshot['status'], 'failed')
            self.assertEqual(snapshot['returncode'], 78)
            self.assertFalse(snapshot['report_available'])

    def test_invalid_report_does_not_become_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            for raw in ['{"schema_version":1,"kind":"cpu_correctness","runs":[],"summary":{"success":true,"success":false}}',
                        '{"schema_version":1,"kind":"cpu_correctness","runs":[],"summary":{"success":true,"median_seconds":NaN}}',
                        '{"schema_version":1,"kind":"cpu_correctness","runs":[],"summary":{"success":true,"median_seconds":1e309}}']:
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    read_report(path)

    def test_deep_report_has_explicit_domain_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            path.write_text('{"deep":' + '[' * 2000 + '0' + ']' * 2000 + '}')
            with self.assertRaisesRegex(ValueError, 'nesting'):
                read_report(path)

    def test_report_depth_scan_preserves_quoted_brackets_and_escapes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            report = {'schema_version': 1, 'kind': 'cpu_correctness', 'summary': {'success': False},
                      'runs': [], 'warnings': ['[' * 1000 + '\\"' + '}' * 1000]}
            path.write_text(json.dumps(report))
            self.assertEqual(read_report(path)[0], report)

    @unittest.skipUnless(sys.platform != 'win32', 'POSIX process-group test; Windows containment NOT VERIFIED')
    def test_force_cancellation_stops_ignoring_descendant_heartbeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker, heartbeat = root / 'pid', root / 'heartbeat'
            suite = root / 'suite.json'
            suite.write_text(json.dumps({'mode': 'descendant', 'marker': str(marker), 'heartbeat': str(heartbeat)}))
            task = BenchmarkTask(suite, root, grace_seconds=.1, command_prefix=[sys.executable, str(FIXTURE)])
            try:
                deadline = time.monotonic() + 10
                while 'fixture_ready' not in task.snapshot()['log_tail'] and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIn('fixture_ready', task.snapshot()['log_tail'])
                task.cancel()
                self.assertTrue(task.done.wait(10))
                self.assertTrue(task.snapshot()['forced'])
                self.assertIsNotNone(task.process)
                assert task.process is not None
                self.assertIsNotNone(task.process.poll())
                observed = heartbeat.read_text()
                time.sleep(.15)
                self.assertEqual(heartbeat.read_text(), observed)
            finally:
                task.cancel()
                task.thread.join(10)

    def test_nonfinite_deadlines_reject_before_process_creation(self):
        from unittest.mock import patch
        with patch('corridorkey.benchmark_task.subprocess.Popen') as spawn:
            for value in (float('nan'), float('inf'), 0, -1, True):
                with self.assertRaises(ValueError):
                    BenchmarkTask(Path('suite'), Path('.'), grace_seconds=value)
            spawn.assert_not_called()

    @unittest.skipUnless(sys.platform != 'win32', 'POSIX inspection fault injection')
    def test_inspection_denial_still_joins_logs_and_closes_owned_pipe(self):
        import os
        from unittest.mock import patch
        real_signal = os.killpg

        def deny_inspection(pid, operation):
            if operation == 0:
                raise PermissionError('injected inspection denial')
            return real_signal(pid, operation)

        with tempfile.TemporaryDirectory() as folder, \
             patch('corridorkey.benchmark_task.os.killpg', side_effect=deny_inspection):
            task = self.task(Path(folder), 'logs')
            self.assertTrue(task.done.wait(10))
            self.assertEqual(task.snapshot()['status'], 'failed')
            self.assertIn('injected inspection denial', task.snapshot()['cleanup_error'])
            assert task.process is not None and task.process.stdout is not None and task.reader is not None
            self.assertTrue(task.process.stdout.closed)
            self.assertFalse(task.reader.is_alive())

    def test_real_paired_cli_dispatch_and_manifest_read_stay_off_calling_thread(self):
        import threading
        from unittest.mock import patch

        from corridorkey.local_json import read_object
        caller = threading.get_ident()
        readers = []

        def observed_read(*args, **kwargs):
            readers.append(threading.get_ident())
            return read_object(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            suite = root / 'invalid-pair.json'
            suite.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_pair', 'repetitions': 2}))
            original = suite.read_bytes()
            with patch('corridorkey.benchmark_task.read_object', side_effect=observed_read):
                task = BenchmarkTask(suite, root)
                try:
                    self.assertTrue(task.done.wait(10))
                finally:
                    task.cancel()
                    task.thread.join(10)
            self.assertTrue(readers)
            self.assertTrue(all(reader != caller for reader in readers))
            self.assertIn('corridorkey.benchmark_pair', task.command)
            state = task.snapshot()
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['returncode'], 1)
            self.assertTrue(state['report_available'])
            report, _ = read_report(task.report_path)
            self.assertEqual(report['kind'], 'inference_pair')
            self.assertEqual(report['pairs'], [])
            self.assertIsNone(report['summary']['speedup'])
            self.assertEqual(suite.read_bytes(), original)
            self.assertIsNone(state['cleanup_error'])
