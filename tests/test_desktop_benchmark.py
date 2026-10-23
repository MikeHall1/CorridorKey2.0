import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
FIXTURE = Path(__file__).with_name('benchmark_fixture.py')
WORKER = Path(__file__).with_name('worker_fixture.py')


@unittest.skipUnless(QT_AVAILABLE, 'Optional requirements-desktop.txt is not installed')
class DesktopBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['benchmark-ui-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(WORKER)])))
        self.window.show()
        self.window.benchmark.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        task = self.window.benchmark.task
        if task is not None:
            task.thread.join(10)
        inspection = self.window.benchmark.inspection
        if inspection is not None:
            inspection.join(5)
        self.window.bridge.thread.join(10)
        self.window.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.window)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Benchmark UI condition not reached')

    def fixture(self, mode):
        from corridorkey.benchmark_task import BenchmarkTask
        suite = self.root / 'suite.json'
        suite.write_text(json.dumps({'mode': mode}))
        self.window.benchmark.factory = lambda manifest, parent: BenchmarkTask(manifest, parent, grace_seconds=.1,
                                                command_prefix=[sys.executable, str(FIXTURE)])
        self.window.benchmark.manifest.setText(str(suite))
        self.window.benchmark.output.setText(str(self.root))
        return suite

    def test_actual_task_waits_for_worker_shutdown_and_loads_baseline(self):
        self.fixture('logs')
        self.window.benchmark.request_run()
        self.until(lambda: self.window.benchmark.task is not None)
        self.assertTrue(self.window.bridge.done.is_set())
        task = self.window.benchmark.task
        assert task is not None
        self.until(task.done.is_set)
        self.assertEqual(task.snapshot()['status'], 'complete')
        self.until(lambda: 'test_fixture' in self.window.benchmark.report_view.toPlainText())
        self.assertIn('test_fixture', self.window.benchmark.report_view.toPlainText())
        self.assertIn('unmeasured', self.window.benchmark.report_view.toPlainText())
        self.window.benchmark.inspect(task.report_path, 'baseline')
        self.until(lambda: 'test_fixture' in self.window.benchmark.baseline.toPlainText())
        self.assertEqual(self.window.benchmark.report.text(), str(task.report_path))

    def test_active_render_rejects_benchmark_then_benchmark_blocks_new_render(self):
        self.fixture('wait')
        source = self.root / 'source.json'
        source.write_text(json.dumps({'gate': str(self.root / 'gate')}))
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root))
        self.window.update_settings({'output_format': 'exr'})
        self.window.enqueue()
        request_id = next(iter(self.window.jobs))
        self.until(lambda: self.window.jobs[request_id]['status'] == 'progress')
        self.window.benchmark.request_run()
        self.assertIn('Finish or cancel', self.window.benchmark.status.text())
        self.assertIsNone(self.window.benchmark.task)
        self.window.table.selectRow(0)
        self.window.cancel_selected()
        self.until(lambda: self.window.jobs[request_id]['status'] == 'cancelled')
        self.window.benchmark.request_run()
        self.until(lambda: self.window.benchmark.task is not None)
        task = self.window.benchmark.task
        assert task is not None
        self.until(lambda: 'fixture_ready' in task.snapshot()['log_tail'])
        self.window.enqueue()
        self.assertEqual(len(self.window.jobs), 1)
        self.window.new_worker()
        self.assertTrue(self.window.bridge.done.is_set())
        from PySide6.QtWidgets import QPushButton
        next(b for b in self.window.benchmark.findChildren(QPushButton) if b.text() == 'Cancel').click()
        self.until(task.done.is_set)
        self.assertTrue(task.snapshot()['forced'])
        self.assertFalse(task.snapshot()['report_available'])

    def test_real_runner_report_exposes_preflight_failure_without_fake_metrics(self):
        source = self.root / 'source.mp4'
        source.touch()
        batch = self.root / 'batch.json'
        batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch',
                        'jobs': [{'input': str(source), 'output_dir': str(self.root / 'unused'), 'profile_stages': True}]}))
        suite = self.root / 'suite.json'
        suite.write_text(json.dumps({'schema_version': 1, 'kind': 'inference', 'model_state': 'warm_session',
                                    'repetitions': 3, 'batch_manifest': str(batch)}))
        self.window.benchmark.manifest.setText(str(suite))
        self.window.benchmark.output.setText(str(self.root))
        self.window.benchmark.request_run()
        self.until(lambda: self.window.benchmark.task is not None)
        task = self.window.benchmark.task
        assert task is not None
        self.until(task.done.is_set)
        self.assertEqual(task.snapshot()['status'], 'failed')
        self.assertTrue(task.snapshot()['report_available'])
        report = json.loads(task.report_path.read_text())
        self.assertFalse(report['summary']['success'])
        self.assertIn('profile_stages', report['error']['message'])
        self.assertIsNone(report['inference']['quality'])
        self.assertFalse((self.root / 'unused').exists())

    def test_window_close_cancels_and_waits_for_actual_benchmark_cleanup(self):
        self.fixture('wait')
        self.window.benchmark.request_run()
        self.until(lambda: self.window.benchmark.task is not None)
        task = self.window.benchmark.task
        assert task is not None
        self.until(lambda: 'fixture_ready' in task.snapshot()['log_tail'])
        self.window.close()
        self.assertTrue(self.window.closing)
        self.until(task.done.is_set)
        self.until(lambda: not self.window.isVisible())
        self.assertTrue(task.cancelled.is_set())

    def test_close_waits_for_report_export_and_preserves_actual_bytes(self):
        import threading
        from unittest.mock import patch

        from corridorkey.benchmark_report import export_report
        source, target = self.root / 'report.json', self.root / 'export.json'
        source.write_text(json.dumps({'schema_version': 1, 'kind': 'cpu_correctness',
                                     'summary': {'success': False}, 'runs': []}))
        self.window.benchmark.report.setText(str(source))
        entered, release = threading.Event(), threading.Event()

        def slow_export(path, destination):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('test export gate')
            export_report(path, destination)

        with patch('corridorkey.desktop_benchmark.export_report', side_effect=slow_export), \
             patch('corridorkey.desktop_benchmark.QFileDialog.getSaveFileName', return_value=(str(target), 'JSON')):
            try:
                self.window.benchmark.choose_export()
                self.assertTrue(entered.wait(5))
                self.window.close()
                self.assertTrue(self.window.isVisible())
            finally:
                release.set()
            self.until(lambda: not self.window.isVisible())
        self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_selected_pair_manifest_runs_real_cli_and_displays_terminal_failure(self):
        suite = self.root / 'invalid-pair.json'
        suite.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_pair', 'repetitions': 2}))
        self.window.benchmark.manifest.setText(str(suite))
        self.window.benchmark.output.setText(str(self.root))
        self.window.benchmark.request_run()
        self.until(lambda: self.window.benchmark.task is not None)
        task = self.window.benchmark.task
        assert task is not None
        self.until(task.done.is_set)
        self.assertIn('corridorkey.benchmark_pair', task.command)
        self.assertEqual(task.snapshot()['status'], 'failed')
        self.assertTrue(task.snapshot()['report_available'])
        self.until(lambda: 'inference_pair' in self.window.benchmark.report_view.toPlainText())
        self.assertIn('inference_pair', self.window.benchmark.report_view.toPlainText())
        self.assertIn('"speedup": null', self.window.benchmark.report_view.toPlainText())
        self.assertTrue(self.window.bridge.done.is_set())
        self.window.benchmark.inspect(task.report_path)
        self.until(lambda: 'Actual local report loaded' in self.window.benchmark.status.text())
        self.assertEqual(self.window.benchmark.report.text(), str(task.report_path))

    def test_background_export_reports_malformed_or_unexpected_failure_terminally(self):
        from contextlib import nullcontext
        from unittest.mock import patch

        from tests.test_benchmark_report import PairedReportTests
        for defect in ('malformed', 'unexpected', 'system_exit'):
            with self.subTest(defect=defect):
                source = self.root / f'{defect}-report.json'
                target = self.root / f'{defect}-export.json'
                data = PairedReportTests().report(True)
                data['pairs'][0]['samples']['candidate']['runs'] = [42]
                source.write_text(json.dumps(data))
                original = source.read_bytes()
                self.window.benchmark.report.setText(str(source))
                expected = 'ValueError' if defect == 'malformed' else 'RuntimeError' if defect == 'unexpected' else 'SystemExit'
                fault = RuntimeError('injected export fault') if defect == 'unexpected' else SystemExit('injected worker exit')
                injected = (nullcontext() if defect == 'malformed' else
                            patch('corridorkey.desktop_benchmark.export_report', side_effect=fault))
                with injected, patch('corridorkey.desktop_benchmark.QFileDialog.getSaveFileName',
                                     return_value=(str(target), 'JSON')):
                    self.window.benchmark.choose_export()
                    self.until(lambda expected=expected: expected in self.window.benchmark.status.text())
                worker = self.window.benchmark.inspection
                self.assertIsNotNone(worker)
                assert worker is not None
                worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertFalse(self.window.benchmark.io_busy)
                self.assertFalse(target.exists())
                self.assertEqual(source.read_bytes(), original)
