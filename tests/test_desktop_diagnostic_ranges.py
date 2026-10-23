"""Owned dialog orchestration mocks plus real parser/preparation/PNG contracts."""
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT, 'Optional Qt not installed')
class DiagnosticDialogCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['diagnostic-ranges-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_diagnostic_ranges import DiagnosticRangesPanel
        from corridorkey.session import JobRequest
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.job = JobRequest.create(self.root / 'source', self.root / 'old-output')
        self.template = {'schema_version': 1, 'job': asdict(self.job), 'source': {
            'path': self.job.input_path, 'start_frame': 10, 'num_frames': 8, 'native_hw': [6, 8],
            'native_fps': 24., 'scene_linear': False, 'source_colour_profile': 'legacy', 'alpha_mode': 'ignore'},
            'events': [{'frame': 12, 'relative_frame': 2, 'reasons': ['neural_fallback'], 'observed': True}],
            'coverage': {'expected_frames': 8, 'quick_observed_frames': 1, 'shot_observed_frames': 0, 'summary_only_events': 0},
            'warnings': ['MOCKED orchestration fixture; no source inspection.']}
        self.rows = [{'id': 'range-1', 'start_frame': 12, 'end_frame': 14, 'reasons': ['neural_fallback'], 'observed_frames': 1}]
        self.calls = []
        self.releases = []
        self.panel = DiagnosticRangesPanel(runner=self.run_fake)
        self.prepared, self.previews, self.preparing = [], [], []
        self.panel.prepared_job.connect(lambda job, context: self.prepared.append((job, context)))
        self.panel.preview_requested.connect(lambda options, context: self.previews.append((options, context)))
        self.panel.prepare_started.connect(self.preparing.append)

    def tearDown(self):
        from tests.qt_cleanup import dispose_widget
        for release in self.releases:
            release.set()
        self.panel.close()
        if self.panel.task is not None:
            self.panel.task.thread.join(5)
            self.assertFalse(self.panel.task.thread.is_alive())
        self.panel.poll()
        self.app.processEvents()
        dispose_widget(self.panel)
        self.folder.cleanup()

    def run_fake(self, request, *, check, progress):
        self.calls.append((request, threading.get_ident()))
        check()
        if request.operation == 'inspect':
            return {'status': 'complete', 'snapshot': json.loads(json.dumps(self.template)), 'candidates': list(self.rows)}
        if request.operation == 'preview':
            return {'status': 'complete', 'options': {'path': self.job.input_path, 'index': request.start,
                    'linear': False, 'alpha_mode': 'ignore', 'source_colour_profile': 'legacy', 'expected_hw': [6, 8]}}
        return self.fake_prepared(request)

    def fake_prepared(self, request):
        from corridorkey.session import JobRequest
        root = Path(request.parent) / ('prepared-' + uuid.uuid4().hex)
        root.mkdir()
        job = JobRequest(self.job.input_path, str(root / 'outputs'), self.job.config_json, self.job.writer_json)
        return {'status': 'complete', 'job': asdict(job), 'reserved_root': str(root)}

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.panel.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail(self.panel.status.text())

    def inspect(self):
        self.panel.inspect_selection(self.job, {}, 'A', str(self.root), protected_paths=['/protected'])
        self.until(lambda: self.panel.snapshot is not None and not self.panel.io_busy)
        self.panel.table.selectRow(0)



class MockedDiagnosticDialogTests(DiagnosticDialogCase):
    def test_pending_terminal_owns_busy_and_no_automatic_preview_or_prepare(self):
        self.panel.inspect_selection(self.job, {}, 'A', str(self.root))
        assert self.panel.task is not None
        self.panel.task.thread.join(5)
        self.assertTrue(self.panel.task.done.is_set())
        self.assertTrue(self.panel.io_busy)
        self.assertFalse(self.panel.inspect())
        self.panel.poll()
        self.assertFalse(self.panel.io_busy)
        self.assertEqual(self.prepared, [])
        self.assertEqual(self.previews, [])
        self.assertEqual(len(self.calls), 1)
        self.assertNotEqual(self.calls[0][1], threading.get_ident())

    def test_paging_selected_absolute_range_and_captured_preview_kwargs(self):
        self.template['source']['num_frames'] = 600
        self.template['events'] = [{'frame': 10 + i, 'relative_frame': i, 'reasons': ['heuristic'], 'observed': i != 500} for i in range(501)]
        self.inspect()
        self.panel.view.setCurrentIndex(1)
        self.assertEqual(self.panel.table.rowCount(), 500)
        self.panel.page.setValue(2)
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.panel.table.selectRow(0)
        self.assertEqual((self.panel.range_in.text(), self.panel.range_out.text()), ('510', '511'))
        observation = self.panel.table.item(0, 3)
        assert observation is not None
        self.assertIn('Summary only', observation.text())
        self.panel.preview()
        self.until(lambda: len(self.previews) == 1)
        options, context = self.previews[0]
        self.assertEqual((options['index'], options['expected_hw'], context), (510, (6, 8), 'A'))
        self.assertEqual(self.prepared, [])

    def test_prepare_captures_explicit_policies_and_protected_paths_without_submission(self):
        self.inspect()
        self.assertTrue(self.panel.full_quality.isChecked())
        self.panel.initial_hint_policy.setCurrentIndex(1)
        self.panel.correction_policy.setCurrentIndex(1)
        self.panel.prepare()
        self.until(lambda: len(self.prepared) == 1)
        request = self.calls[-1][0]
        self.assertTrue(request.full_quality)
        self.assertEqual((request.initial_hint_policy, request.correction_policy, request.protected_paths), ('drop', 'drop', ('/protected',)))
        self.assertEqual((request.start, request.end), (12, 14))
        self.assertEqual(self.preparing, ['A'])
        self.assertEqual(self.previews, [])
        self.assertTrue(Path(self.prepared[0][0].output_dir).parent.is_dir())
        self.assertFalse(Path(self.prepared[0][0].output_dir).exists())

    def test_prepare_started_invalidation_prevents_old_request_start(self):
        self.inspect()
        previous = self.panel.task
        self.panel.prepare_started.connect(lambda _context: self.panel.invalidate_context())
        self.panel.prepare()
        self.assertIs(self.panel.task, previous)
        self.assertFalse(self.panel.io_busy)
        self.assertIsNone(self.panel.context_id)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.prepared, [])
        self.assertFalse(self.panel.prepare_button.isEnabled())

    def test_A_B_A_context_changes_do_not_reauthorize_old_terminal(self):
        started, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def gated(request, **kwargs):
            started.set()
            release.wait(5)
            return self.run_fake(request, check=lambda: None, progress=kwargs['progress'])

        self.panel.runner = gated
        self.panel.inspect_selection(self.job, {}, 'A', str(self.root))
        self.assertTrue(started.wait(2))
        self.assertFalse(self.panel.inspect_selection(self.job, {}, 'B', str(self.root)))
        self.assertFalse(self.panel.inspect_selection(self.job, {}, 'A', str(self.root)))
        release.set()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.snapshot)
        self.assertFalse(self.panel.prepare_button.isEnabled())
        self.assertEqual(len(self.calls), 1)
        self.panel.runner = self.run_fake
        self.panel.inspect()
        self.until(lambda: self.panel.snapshot is not None)
        self.assertEqual(len(self.calls), 2)

    def test_edited_preparation_preserves_committed_path_but_emits_no_stale_job(self):
        self.inspect()
        started, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def gated(request, **kwargs):
            result = self.fake_prepared(request)
            started.set()
            release.wait(5)
            return result

        self.panel.runner = gated
        self.panel.prepare()
        self.assertTrue(started.wait(2))
        self.panel.range_in.setText('13')
        self.panel.range_in.setText('12')
        release.set()
        self.until(lambda: not self.panel.io_busy)
        self.assertEqual(self.prepared, [])
        self.assertIn('Reserved folder remains:', self.panel.status.text())
        self.assertIn('Reserved range folder:', self.panel.result_path.text())

    def test_close_waits_for_actual_worker_and_pending_terminal_after_cancellation(self):
        started, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def gated(request, *, check, progress):
            started.set()
            release.wait(5)
            check()
            return self.run_fake(request, check=check, progress=progress)

        self.panel.runner = gated
        self.panel.inspect_selection(self.job, {}, 'A', str(self.root))
        self.assertTrue(started.wait(2))
        self.assertFalse(self.panel.close())
        self.assertTrue(self.panel.isVisible())
        self.assertTrue(self.panel.io_busy)
        release.set()
        assert self.panel.task is not None
        self.panel.task.thread.join(5)
        self.assertTrue(self.panel.io_busy)
        self.panel.poll()
        self.assertFalse(self.panel.isVisible())
        self.assertFalse(self.panel.io_busy)
        self.assertEqual(self.prepared, [])

    def test_baseexception_terminal_and_failed_preview_revalidation_do_not_emit(self):
        from corridorkey.desktop_diagnostic_ranges import (
            DiagnosticRangeTask,
            RangeRequest,
        )
        for result in ('baseexception', 'invalid-result'):
            def bad_runner(*args, outcome=result, **kwargs):
                if outcome == 'baseexception':
                    raise SystemExit('controlled test worker failure')
                return []
            task = DiagnosticRangeTask(RangeRequest('inspect'), bad_runner)
            task.thread.join(5)
            self.assertTrue(task.done.is_set())
            self.assertTrue(task.busy)
            self.assertEqual(json.loads(task.results.get_nowait())['status'], 'failed')
            self.assertFalse(task.busy)
        self.inspect()

        def changed(*args, **kwargs):
            raise ValueError('Diagnostic trace changed')
        self.panel.runner = changed
        self.panel.preview()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.snapshot)
        self.assertEqual(self.previews, [])
        self.assertFalse(self.panel.prepare_button.isEnabled())


@unittest.skipUnless(QT, 'Optional Qt not installed')
class ActualDiagnosticDialogTests(DiagnosticDialogCase):
    def test_actual_core_prepare_and_emitted_kwargs_decode_real_PNG(self):
        import cv2
        import numpy as np

        from corridorkey.desktop_diagnostic_ranges import run_operation
        from corridorkey.desktop_loader import PreviewLoader
        from corridorkey.desktop_preview import PreviewFrame
        from tests.test_diagnostic_ranges import make_diagnostic_fixture
        fixture = self.root / 'actual'
        fixture.mkdir()
        job, record, parent = make_diagnostic_fixture(fixture)
        for i, path in enumerate(sorted(Path(job.input_path).glob('*.png'))):
            self.assertTrue(cv2.imwrite(str(path), np.full((64, 64, 3), 40 + i * 4, np.uint8)))
        original = {p: p.read_bytes() for p in fixture.rglob('*') if p.is_file()}
        self.panel.runner = run_operation
        self.panel.inspect_selection(job, record, 'actual', str(parent))
        self.until(lambda: self.panel.snapshot is not None and not self.panel.io_busy)
        self.assertIn('3 Quick Key / 5 shot', self.panel.coverage.text())
        self.assertIn('Historical source identity NOT VERIFIED', self.panel.coverage.text())
        self.panel.table.selectRow(0)
        self.assertEqual((self.panel.range_in.text(), self.panel.range_out.text()), ('3', '4'))
        self.panel.preview()
        self.until(lambda: len(self.previews) == 1)
        options, context = self.previews[0]
        self.assertEqual((options['expected_hw'], context), ((64, 64), 'actual'))
        loader = PreviewLoader()
        try:
            serial = loader.submit(**options)
            found_serial, frame, error = loader.results.get(timeout=5)
            self.assertEqual(found_serial, serial)
            self.assertIsNone(error)
            assert isinstance(frame, PreviewFrame)
            self.assertEqual(int(frame.pixels[0, 0, 0]), 52)
        finally:
            loader.stopping.set()
            loader.thread.join(5)
            self.assertFalse(loader.thread.is_alive())
        self.panel.prepare()
        self.until(lambda: len(self.prepared) == 1)
        prepared, context = self.prepared[0]
        config, _ = prepared.configurations()
        self.assertEqual((config.start_frame, config.num_frames, config.quick_key, context), (3, 1, False, 'actual'))
        self.assertTrue(Path(prepared.output_dir).parent.is_dir())
        self.assertFalse(Path(prepared.output_dir).exists())
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
