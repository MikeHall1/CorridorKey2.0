"""Qt readiness workflow fixtures, never evidence of real model inference."""
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional requirements-desktop.txt is not installed')
class DesktopReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['readiness-ui-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_window import DesktopWindow
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.window = DesktopWindow()
        self.window.show()
        self.panel = self.window.readiness
        self.panel.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        if self.panel.io_thread is not None:
            self.panel.io_thread.join(5)
        self.window.bridge.thread.join(5)
        self.window.loader.thread.join(5)
        self.window.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.window)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Readiness UI condition not reached')

    def select_source(self):
        source = self.root / 'local.mp4'
        source.write_bytes(b'local fixture source, not encoded media')
        self.window.source.setText(str(source))
        self.window.output.setText(str(self.root))
        return source

    def test_opening_does_not_run_checks_and_missing_form_allows_source_free_checks(self):
        report = {'stages': [{'id': 'inference', 'status': 'NOT VERIFIED', 'message': 'Explicit render required'}]}
        with patch('corridorkey.desktop_readiness.check_readiness', return_value=report) as check:
            self.assertFalse(check.called)
            self.panel.inspect()
            self.until(lambda: not self.panel.io_busy)
            check.assert_called_once()
            self.assertIsNone(check.call_args.args[0])
            self.assertEqual(self.panel.report, report)
            self.assertIn('NOT VERIFIED', self.panel.details.toPlainText())
            self.assertFalse(self.window.jobs)

    def test_prepare_is_immutable_and_run_is_explicit_single_use(self):
        from corridorkey.worker_protocol import job_to_wire
        source = self.select_source()
        original = self.window.job()
        self.panel.start.setValue(3)
        self.panel.count.setValue(4)
        self.panel.prepare()
        self.until(lambda: not self.panel.io_busy)
        prepared = self.panel.prepared
        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(self.window.job(), original)
        self.assertFalse(self.window.jobs)
        self.assertIn(prepared.output_dir, self.panel.details.toPlainText())
        self.window.source.setText(str(self.root / 'changed.mp4'))
        with patch.object(self.window.bridge, 'send', return_value='readiness-job') as send:
            self.panel.run()
            send.assert_called_once_with('render', {'job': job_to_wire(prepared)})
            self.assertEqual(self.window.jobs['readiness-job']['job'], prepared)
            self.panel.run()
            self.assertEqual(send.call_count, 1)
            self.panel.cancel()
            self.assertEqual(send.call_args.args[0], 'cancel')
            self.assertEqual(send.call_args.args[1]['target'], 'readiness-job')
        self.assertIsNone(self.panel.prepared)
        self.assertEqual(source.read_bytes(), b'local fixture source, not encoded media')

    def test_repeated_renders_keep_only_current_protection_without_mutating_scan(self):
        source = self.select_source()
        original = self.window.job()
        self.panel.report = {'schema_version': 1, 'protected_paths': [str(source)], 'stages': []}
        for index in range(100):
            self.panel.prepared = original
            self.panel.prepared_selection = None
            self.panel.mark_submitted(f'local-fixture-{index}')
            self.panel.observe({'request_id': f'local-fixture-{index}', 'kind': 'rejected',
                                'payload': {'error': 'fixture only; not submitted to worker'}})
        self.assertEqual(self.panel.report['protected_paths'], [str(source)])
        self.assertLessEqual(len(self.panel.render_protected), 3)
        assert self.panel.render_observation is not None
        self.assertEqual(self.panel.render_observation['request_id'], 'local-fixture-99')
        self.assertEqual(source.read_bytes(), b'local fixture source, not encoded media')

    def test_submission_respects_ownership_guards_and_history_capacity(self):
        from corridorkey.session import JobRequest
        prepared = JobRequest.create('.', str(self.root / 'fresh'))
        self.panel.prepared = prepared
        with patch.object(self.window.bridge, 'send') as send:
            for mode in ('closing', 'queue', 'benchmark', 'full'):
                self.window.closing = mode == 'closing'
                self.window.pending_queue_root = self.root if mode == 'queue' else None
                self.window.benchmark.waiting = mode == 'benchmark'
                self.window.jobs = {str(i): {} for i in range(128)} if mode == 'full' else {}
                self.window.submit_readiness(prepared)
                self.assertFalse(send.called, mode)
            self.window.jobs = {}
            self.window.closing = False
            self.window.pending_queue_root = None
            self.window.benchmark.waiting = False

    def test_checks_cancel_without_accepting_late_success_and_window_waits(self):
        entered, release = threading.Event(), threading.Event()

        def delayed_check(job, selection, *, cancellation):
            entered.set()
            release.wait(5)
            return {'stages': [], 'fixture': True}

        with patch('corridorkey.desktop_readiness.check_readiness', side_effect=delayed_check):
            try:
                self.panel.inspect()
                self.assertTrue(entered.wait(2))
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(self.panel.cancellation.cancelled)
            finally:
                release.set()
            self.until(lambda: not self.window.isVisible())
        self.assertIsNone(self.panel.report)

    def test_export_is_background_owned_and_does_not_overwrite(self):
        self.panel.report = {'stages': [], 'actual_inference': 'NOT VERIFIED'}
        target = self.root / 'report.json'
        with patch('corridorkey.desktop_readiness.QFileDialog.getSaveFileName', return_value=(str(target), 'JSON')):
            self.panel.choose_export()
            self.until(lambda: not self.panel.io_busy)
            self.assertEqual(json.loads(target.read_text()), self.panel.report)
            before = target.read_bytes()
            self.panel.choose_export()
            self.until(lambda: not self.panel.io_busy)
            self.assertEqual(target.read_bytes(), before)
            self.assertIn('Error', self.panel.status.text())

    def test_range_edit_invalidates_snapshot_and_terminal_releases_submission(self):
        from corridorkey.session import JobRequest
        self.panel.prepared = JobRequest.create('.', str(self.root / 'fresh'))
        self.panel.count.setValue(2)
        self.assertIsNone(self.panel.prepared)
        self.panel.prepared = JobRequest.create('.', str(self.root / 'fresh'))
        self.panel.mark_submitted('fixture')
        self.panel.observe({'request_id': 'other', 'kind': 'result', 'payload': {}})
        self.assertEqual(self.panel.submitted_id, 'fixture')
        self.panel.observe({'request_id': 'fixture', 'kind': 'failed', 'payload': {'error': 'fixture failure'}})
        self.assertIsNone(self.panel.submitted_id)
        self.assertIn('fixture failure', self.panel.status.text())

    def test_checks_replace_display_and_invalidate_prepared_job(self):
        from corridorkey.session import JobRequest
        self.panel.prepared = JobRequest.create('.', str(self.root / 'fresh'))
        with patch('corridorkey.desktop_readiness.check_readiness', return_value={'stages': []}):
            self.panel.inspect()
            self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.prepared)
        self.assertFalse(self.panel.run_button.isEnabled())

    def test_changed_assets_require_new_snapshot(self):
        from corridorkey.asset_inventory import AssetSelection
        from corridorkey.session import JobRequest
        self.panel.prepared = JobRequest.create('.', str(self.root / 'fresh'))
        self.window.asset_selection = AssetSelection(self.root)
        with patch.object(self.window.bridge, 'send') as send:
            self.panel.run()
            self.assertFalse(send.called)
        self.assertIn('assets changed', self.panel.status.text())

    def test_late_preparation_cannot_reactivate_changed_range(self):
        from corridorkey.readiness import prepare_short_render
        entered, release = threading.Event(), threading.Event()
        self.select_source()

        def delayed_prepare(job, start, count):
            entered.set()
            release.wait(5)
            return prepare_short_render(job, start, count)

        with patch('corridorkey.desktop_readiness.prepare_short_render', side_effect=delayed_prepare):
            try:
                self.panel.prepare()
                self.assertTrue(entered.wait(2))
                self.panel.start.setValue(7)
            finally:
                release.set()
            self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.prepared)
        self.assertIn('Range changed', self.panel.status.text())
