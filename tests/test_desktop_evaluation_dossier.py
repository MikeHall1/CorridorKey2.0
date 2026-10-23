"""Owned dossier tasks with injected local operations; never run models or network."""
import importlib.util
import json
import os
import threading
import time
import unittest

from tests.qt_cleanup import dispose_widget

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def dossier():
    metric = {'units': 'alpha/pixel', 'status': 'partial', 'baseline_median': .2,
              'candidate_median': .3, 'absolute_delta': .1, 'relative_delta': .5,
              'screen': 'regression', 'coverage': {'expected': 3, 'assessed': 2, 'unassessed': 1},
              'repetitions': [{'pair_index': 2, 'status': 'unassessed', 'reason': 'Reference missing'}]}
    return {'coverage': {'expected_cases': 2, 'assessed_cases': 0, 'partial_cases': 1, 'unassessed_cases': 1},
            'cases': [{'id': 'clip1', 'split': 'synthetic', 'status': 'partial', 'metrics': {'alpha_mae': metric}}],
            'categories': [{'category': 'hair', 'split': 'synthetic', 'metrics': {'alpha_mae': {
                **metric, 'worst_case_median': {'case_id': 'clip1', 'relative_delta': .5},
                'worst_candidate_error': {'case_id': 'clip1', 'pair_index': 0, 'value': .3, 'baseline': .2, 'units': 'alpha/pixel'}}}}],
            'review_media': [{'case_id': 'clip1', 'side': 'baseline', 'bytes': 10}],
            'review_status': 'pending', 'acceptance_verdict': None, 'token': 'fixture'}


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class EvaluationDossierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['dossier-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_evaluation_dossier import EvaluationDossierPanel
        self.calls = []

        def runner(request, *, check, progress):
            self.calls.append(request)
            check()
            if request.operation == 'build':
                return {'status': 'complete', 'report': dossier(), 'error': None}
            return {'status': 'complete', 'destination': '/captured/dossier', 'usable': True,
                    'error': None, 'warning': None, 'partial_path': None}

        self.panel = EvaluationDossierPanel(runner=runner)
        self.panel.manifest_path.setText('/local/manifest.json')
        self.panel.parent_path.setText('/local/exports')

    def tearDown(self):
        if self.panel.task is not None:
            self.panel.task.cancel()
            self.panel.task.thread.join(3)
            self.assertFalse(self.panel.task.thread.is_alive())
        self.panel.poll()
        dispose_widget(self.panel)

    def finish(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.panel.poll()
            if not self.panel.io_busy:
                return
            time.sleep(.005)
        self.fail('Dossier task retained ownership unexpectedly')

    def build(self):
        self.panel.build()
        self.finish()
        self.assertIsInstance(self.panel.report, bytes)

    def test_explicit_build_human_partial_worstcase_and_metadata_export(self):
        self.assertEqual(self.calls, [])
        self.assertFalse(self.panel.include_media.isChecked())
        self.assertEqual(self.panel.max_media_bytes.text(), '0')
        self.build()
        self.assertIn('1 partial', self.panel.coverage.text())
        self.assertIn('10 encoded bytes', self.panel.media_summary.text())
        values = []
        for row in range(self.panel.table.rowCount()):
            for col in range(7):
                item = self.panel.table.item(row, col)
                assert item is not None
                values.append(item.text())
        cells = '\n'.join(values)
        self.assertIn('worst case clip1', cells)
        self.assertIn('largest error clip1: 0.3 alpha/pixel', cells)
        self.assertIn('Reference missing', cells)
        self.assertIn('Metric regression', cells)
        self.assertNotIn('fixture', cells)
        self.assertEqual(self.calls[0].manifest, '/local/manifest.json')
        report = self.panel.report
        self.panel.parent_path.setText('/other/parent')
        self.assertEqual(self.panel.report, report)
        self.panel.export()
        self.finish()
        self.assertFalse(self.calls[1].include_media)
        self.assertEqual(self.calls[1].max_media_bytes, 0)
        self.assertEqual(self.calls[1].parent, '/other/parent')
        self.assertIn('Metadata only', self.panel.status.text())
        self.assertIn('human review are pending', self.panel.status.text())
        self.assertEqual(self.panel.result_path.text(), '/captured/dossier')
        self.panel.manifest_path.setText('/new/manifest.json')
        self.assertIsNone(self.panel.report)
        self.assertEqual(self.panel.result_path.text(), '')

    def test_explicit_media_quota_exact_bounds_and_captured_options(self):
        self.build()
        self.panel.include_media.setChecked(True)
        for text in ('0', '-1', '1.5', '9', '9223372036854775808'):
            self.panel.max_media_bytes.setText(text)
            self.panel.export()
            self.assertEqual(len(self.calls), 1)
        self.panel.max_media_bytes.setText('9223372036854775807')
        self.panel.export()
        self.finish()
        self.assertTrue(self.calls[1].include_media)
        self.assertEqual(self.calls[1].max_media_bytes, 2**63 - 1)
        self.assertEqual(json.loads(self.calls[1].report)['review_status'], 'pending')
        self.assertIn('human review is pending', self.panel.status.text())
        self.panel.max_media_bytes.setText('100')
        self.assertIsNotNone(self.panel.report)
        self.assertEqual(self.panel.result_path.text(), '')

    def test_manifest_edited_and_restored_during_build_never_selects_stale_report(self):
        entered, release = threading.Event(), threading.Event()

        def runner(request, *, check, progress):
            entered.set()
            release.wait(3)
            return {'status': 'complete', 'report': dossier(), 'error': None}

        self.panel.runner = runner
        self.panel.build()
        try:
            self.assertTrue(entered.wait(2))
            self.panel.manifest_path.setText('/changed.json')
            self.panel.manifest_path.setText('/local/manifest.json')
        finally:
            release.set()
        self.finish()
        self.assertIsNone(self.panel.report)
        self.assertFalse(self.panel.export_button.isEnabled())
        self.assertIn('Manifest changed', self.panel.status.text())

    def test_pending_result_and_parent_busy_exclude_new_work(self):
        self.panel.build()
        assert self.panel.task is not None
        self.panel.task.thread.join(3)
        first = self.panel.task
        self.assertTrue(self.panel.io_busy)
        self.panel.build()
        self.assertIs(self.panel.task, first)
        self.finish()
        from PySide6.QtWidgets import QWidget
        class Parent(QWidget):
            dossier_busy = True

        parent = Parent()
        self.panel.setParent(parent)
        self.panel.build()
        self.assertIs(self.panel.task, first)
        self.panel.setParent(None)
        dispose_widget(parent)

    def test_close_cancels_and_waits_real_owner_even_after_cancel(self):
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()

        def runner(request, *, check, progress):
            from corridorkey.control import CancelledError
            entered.set()
            try:
                while not release.wait(.005):
                    check()
            except CancelledError:
                cancelled.set()
                release.wait(3)
                raise
            return {'status': 'complete', 'report': dossier()}

        self.panel.runner = runner
        self.panel.show()
        self.panel.build()
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(self.panel.close())
            self.assertTrue(cancelled.wait(2))
            self.assertTrue(self.panel.io_busy)
            assert self.panel.task is not None
            self.assertFalse(self.panel.task.done.is_set())
            self.assertTrue(self.panel.isVisible())
        finally:
            release.set()
        self.finish()
        self.assertFalse(self.panel.isVisible())
        self.assertIn('cancelled', self.panel.status.text())
        self.assertFalse(self.panel.close_when_done)
        self.panel.show()
        self.panel.poll()
        self.assertTrue(self.panel.isVisible())
        self.assertTrue(self.panel.build_button.isEnabled())
        self.assertTrue(self.panel.close())
        self.panel.show()
        self.panel.poll()
        self.assertTrue(self.panel.build_button.isEnabled())

    def test_baseexception_is_terminal_and_partial_export_is_visible(self):
        def broken(request, *, check, progress):
            raise SystemExit('fixture unexpected exit')

        self.panel.runner = broken
        self.panel.build()
        self.finish()
        self.assertIn('SystemExit', self.panel.status.text())
        self.assertIsNone(self.panel.report)
        self.panel.report = json.dumps(dossier()).encode()
        self.panel.runner = lambda *args, **kwargs: {'status': 'failed', 'usable': False,
            'destination': '/partial/dossier', 'partial_path': '/partial/stage', 'error': 'Disk full', 'warning': None}
        self.panel.export()
        self.finish()
        self.assertIn('Disk full', self.panel.status.text())
        self.assertIn('/partial/stage', self.panel.status.text())
        self.assertIn('not complete', self.panel.status.text())
        self.assertEqual(self.panel.result_path.text(), '/partial/dossier')
        self.assertNotIn('Review packet prepared', self.panel.status.text())

    def test_late_cancel_keeps_authoritative_usable_export_and_exact_destination(self):
        entered, release = threading.Event(), threading.Event()
        self.build()

        def committed(request, *, check, progress):
            entered.set()
            release.wait(3)
            return {'status': 'complete', 'destination': '/committed/exact', 'usable': True,
                    'warning': 'Cleanup alias remains', 'partial_path': '/private/alias', 'error': None}

        self.panel.runner = committed
        self.panel.export()
        try:
            self.assertTrue(entered.wait(2))
            self.panel.parent_path.setText('/edited/after/authorization')
            self.panel.cancel()
        finally:
            release.set()
        self.finish()
        self.assertEqual(self.panel.result_path.text(), '/committed/exact')
        self.assertIn('Published dossier is usable', self.panel.status.text())
        self.assertIn('Cleanup alias remains', self.panel.status.text())
        self.assertIn('/private/alias', self.panel.status.text())
