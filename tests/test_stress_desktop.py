"""Strict stress-report inspection and actual owned desktop runner lifecycle."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.benchmark_report import export_report, read_report, report_view
from corridorkey.benchmark_task import BenchmarkTask
from tests.stress_desktop_fixture import report_for

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
FIXTURE = Path(__file__).with_name('stress_desktop_fixture.py')


class StressReportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'production_stress'}))
        self.path = self.root / 'report.json'

    def test_unverified_is_explicit_and_export_preserves_exact_bytes(self):
        report = report_for(self.manifest, self.path, 'unavailable')
        report['cases'][0]['id'] = 'clip.with.dots'
        self.path.write_text(json.dumps(report))
        loaded, raw = read_report(self.path)
        self.assertFalse(loaded['summary']['success'])
        self.assertIn('NOT VERIFIED', report_view(loaded))
        self.assertIn('no production stress', report_view(loaded))
        destination = self.root / 'copy.json'
        export_report(self.path, destination)
        self.assertEqual(destination.read_bytes(), raw)
        with self.assertRaises(FileExistsError):
            export_report(self.path, destination)

    def test_inconsistent_counts_scenarios_cleanup_verdict_and_nonfinite_refused(self):
        valid = report_for(self.manifest, self.path, 'complete')
        invalid = []
        row = copy.deepcopy(valid)
        row['summary']['measured_cases'] = 8
        invalid.append(row)
        row = copy.deepcopy(valid)
        row['cases'][0]['scenario'] = 'unknown'
        invalid.append(row)
        row = copy.deepcopy(valid)
        row['cases'][0]['id'] = row['cases'][1]['id']
        invalid.append(row)
        row = copy.deepcopy(valid)
        row['cases'][0]['cleanup']['verified'] = False
        invalid.append(row)
        row = copy.deepcopy(valid)
        row['acceptance_verdict'] = 'professional quality passed'
        invalid.append(row)
        row = copy.deepcopy(valid)
        row['cases'][0]['duration_seconds'] = float('nan')
        invalid.append(row)
        row = copy.deepcopy(valid)
        row['cases'][0]['status'] = {'measured': True}
        invalid.append(row)
        cancelled = report_for(self.manifest, self.path, 'cancelled')
        for case in cancelled['cases']:
            case['id'] = '$missing:' + case['scenario']
        self.path.write_text(json.dumps(cancelled))
        self.assertEqual(read_report(self.path)[0]['status'], 'cancelled')
        for report in invalid:
            with self.subTest(report=report):
                self.path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    read_report(self.path)

    def test_runner_selection_unavailable_and_abnormal_exit_remain_distinct(self):
        commands = []

        def command(module, *args):
            commands.append(module)
            return [sys.executable, str(FIXTURE), *args]

        for mode, expected in [('complete', 'complete'), ('unavailable', 'unavailable'), ('failed', 'failed'), ('abnormal', 'failed')]:
            with self.subTest(mode=mode):
                self.manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'production_stress', 'fixture_mode': mode}))
                with patch('corridorkey.benchmark_task.module_command', side_effect=command):
                    task = BenchmarkTask(self.manifest, self.root)
                    try:
                        self.assertTrue(task.done.wait(10))
                    finally:
                        task.cancel()
                        task.thread.join(10)
                self.assertEqual(task.snapshot()['status'], expected, task.snapshot())
                self.assertTrue(task.snapshot()['report_available'])
                self.assertIsNone(task.snapshot()['cleanup_error'])
                self.assertFalse(task.thread.is_alive())
                assert task.process is not None and task.reader is not None
                self.assertIsNotNone(task.process.poll())
                self.assertFalse(task.reader.is_alive())
        self.assertEqual(commands.count('corridorkey.stress_suite'), 4)


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class StressDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['stress-desktop', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.manifest = self.root / 'stress.json'
        self.window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(Path(__file__).with_name('worker_fixture.py'))])))
        self.window.benchmark.factory = lambda manifest, parent: BenchmarkTask(manifest, parent, grace_seconds=.2,
                                                   command_prefix=[sys.executable, str(FIXTURE)])
        self.window.benchmark.manifest.setText(str(self.manifest))
        self.window.benchmark.output.setText(str(self.root))
        self.window.show()
        self.window.benchmark.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        task = self.window.benchmark.task
        if task is not None:
            task.cancel()
            task.thread.join(10)
            self.assertFalse(task.thread.is_alive())
            if task.reader is not None:
                self.assertFalse(task.reader.is_alive())
        self.window.bridge.thread.join(10)
        self.assertFalse(self.window.bridge.thread.is_alive())
        self.window.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.window)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail(self.window.benchmark.status.text())

    def test_developer_panel_shows_unavailable_without_running_render_or_changing_manifest(self):
        self.manifest.write_text(json.dumps({'kind': 'production_stress', 'fixture_mode': 'unavailable'}))
        original = self.manifest.read_bytes()
        self.window.benchmark.request_run()
        self.until(lambda: self.window.benchmark.task is not None and self.window.benchmark.task.done.is_set())
        self.window.poll()
        task = self.window.benchmark.task
        assert task is not None
        self.assertEqual(task.snapshot()['status'], 'unavailable')
        self.assertIn('NOT VERIFIED', self.window.benchmark.report_view.toPlainText())
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(self.window.jobs, {})

    def test_close_cancels_owned_stress_runner_and_waits_for_real_terminal_report(self):
        self.manifest.write_text(json.dumps({'kind': 'production_stress', 'fixture_mode': 'wait'}))
        original = self.manifest.read_bytes()
        self.window.benchmark.request_run()
        self.until(lambda: self.window.benchmark.task is not None and 'stress_fixture_ready' in self.window.benchmark.task.snapshot()['log_tail'])
        task = self.window.benchmark.task
        assert task is not None
        self.assertFalse(self.window.close())
        self.until(lambda: task.done.is_set() and not self.window.isVisible())
        self.assertEqual(task.snapshot()['status'], 'cancelled')
        self.assertTrue(task.snapshot()['report_available'])
        assert task.process is not None and task.reader is not None
        self.assertIsNotNone(task.process.poll())
        self.assertFalse(task.reader.is_alive())
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(self.window.jobs, {})
