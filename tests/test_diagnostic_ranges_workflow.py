"""Actual desktop/CPU worker range preparation; no trained quality claims."""
import importlib.util
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
STACK = all(importlib.util.find_spec(name) for name in ('PySide6', 'torch', 'numpy', 'cv2', 'safetensors'))
FIXTURE = Path(__file__).with_name('stage2_worker_fixture.py')


@unittest.skipUnless(STACK, 'Requires desktop and explicit CPU runtime fixture')
class DiagnosticRangesWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['diagnostic-ranges-test', '-platform', 'offscreen'])

    def setUp(self):
        import cv2
        import numpy as np

        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        for index in range(9):
            self.assertTrue(cv2.imwrite(str(self.source / f'{index:04}.png'),
                np.full((64, 64, 3), 60 if index < 5 else 210, np.uint8)))
        self.originals = self.tree_bytes(self.source)
        bridge = DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(FIXTURE),
            '--fixture-root', str(self.root / 'fixture-assets')]))
        self.window = DesktopWindow(bridge)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.until(lambda: not self.window.isVisible())
        self.window.bridge.thread.join(10)
        self.window.loader.thread.join(10)
        task = self.window.diagnostic_ranges.task
        if task is not None:
            task.thread.join(10)
            self.assertFalse(task.thread.is_alive())
        self.assertFalse(self.window.bridge.thread.is_alive())
        self.assertFalse(self.window.loader.thread.is_alive())
        self.assertEqual(self.tree_bytes(self.source), self.originals)
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.window)
        self.folder.cleanup()

    @staticmethod
    def tree_bytes(path):
        return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob('*') if p.is_file()}

    def until(self, predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail('Desktop range workflow did not reach its terminal condition: ' + self.window.diagnostic_ranges.status.text())

    def render_parent(self):
        self.window.set_source(str(self.source))
        self.window.output.setText(str(self.root / 'exports'))
        self.window.update_settings({'output_format': 'exr', 'exr_dtype': 'float', 'outputs': ['alpha', 'fg'],
            'start_frame': 2, 'num_frames': 7, 'hann_chunk': 8, 'hann_stride': 4,
            'shape_quantum': 64, 'stage2_batch': 2, 'despill_strength': 0,
            'quick_key': True, 'quick_interval': 4, 'quick_context_radius': 1,
            'quick_fallback': 'neural', 'quick_auto_cuts': True, 'temp_dir': str(self.root / 'temp')})
        self.window.enqueue()
        request = list(self.window.jobs)[-1]
        self.until(lambda: self.window.jobs[request]['status'] in {'result', 'failed', 'lost', 'cancelled'})
        self.assertEqual(self.window.jobs[request]['status'], 'result', self.window.jobs[request])
        self.window.table.selectRow(0)
        return request, self.window.jobs[request]

    def inspect_parent(self):
        self.window.open_diagnostic_ranges()
        panel = self.window.diagnostic_ranges
        self.until(lambda: not panel.io_busy)
        self.assertIsNotNone(panel.snapshot, panel.status.text())
        return panel

    def select_cut(self, panel):
        panel.view.setCurrentIndex(1)
        row = next(index for index, event in enumerate(panel.events)
                   if event['frame'] == 5 and 'heuristic_cut' in event['reasons'])
        panel.page.setValue(row // 500 + 1)
        panel.table.selectRow(row % 500)
        self.assertEqual(panel.range_in.text(), '5')
        self.assertEqual(panel.range_out.text(), '6')

    def test_actual_diagnostic_source_offset_prepare_run_and_aligned_comparison(self):
        _, parent = self.render_parent()
        original_job = parent['job']
        parent_files = self.tree_bytes(Path(original_job.output_dir))
        panel = self.inspect_parent()
        self.select_cut(panel)
        panel.preview()
        self.until(lambda: not panel.io_busy and self.window.pixmap is not None)
        self.assertEqual(self.window.frame.value(), 5)
        self.assertIn('source frame 5', self.window.preview_description)
        assert self.window.pixmap is not None
        self.assertEqual(self.window.pixmap.toImage().pixelColor(0, 0).red(), 210)
        self.assertTrue(panel.full_quality.isChecked())
        panel.prepare()
        self.until(lambda: not panel.io_busy)
        cfg, _ = self.window.job().configurations()
        self.assertEqual((cfg.start_frame, cfg.num_frames), (5, 1))
        self.assertFalse(cfg.quick_key)
        self.assertEqual(len(self.window.jobs), 1, 'Preparation must not submit work')
        self.assertTrue(original_job.configurations()[0].quick_key)
        self.assertEqual(self.tree_bytes(Path(original_job.output_dir)), parent_files)
        reserved = Path(self.window.output.text())
        self.assertTrue(reserved.is_dir())
        self.assertEqual(list(reserved.iterdir()), [])
        self.window.enqueue()
        rerender_id = list(self.window.jobs)[-1]
        self.until(lambda: self.window.jobs[rerender_id]['status'] in {'result', 'failed', 'lost'})
        rerender = self.window.jobs[rerender_id]
        self.assertEqual(rerender['status'], 'result', rerender)
        record = rerender['event']['payload']['record']
        self.assertEqual(record['diagnostics']['source']['start_frame'], 5)
        self.assertEqual(record['diagnostics']['frames_submitted'], 1)
        self.assertNotEqual(rerender['job'].output_dir, original_job.output_dir)
        self.window.table.selectRow(0)
        self.window.view.setCurrentText('Alpha')
        self.window.synchronize_reference()
        self.window.table.selectRow(1)
        self.window.frame.setValue(5)
        self.until(lambda: self.window.reference is not None and self.window.pixmap is not None)
        self.assertIn('source frame 5', self.window.preview_description)
        self.assertIn('export frame 3', self.window.reference_description)
        self.assertIn('decoded frame 0', self.window.preview_description)
        self.assertEqual(self.tree_bytes(Path(original_job.output_dir)), parent_files)

    def test_future_editor_changes_during_preparation_never_apply_stale_settings(self):
        from corridorkey.desktop_diagnostic_ranges import run_operation
        self.render_parent()
        panel = self.inspect_parent()
        self.select_cut(panel)
        entered, release = threading.Event(), threading.Event()

        def gated(request, *, check, progress):
            if request.operation == 'prepare':
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('Test preparation gate was not released')
            return run_operation(request, check=check, progress=progress)

        panel.runner = gated
        try:
            panel.prepare()
            self.until(entered.is_set)
            self.window.update_settings({'seed': 41})
            changed = self.window.config.toPlainText()
            release.set()
            self.until(lambda: not panel.io_busy)
            self.assertEqual(self.window.config.toPlainText(), changed)
            self.assertEqual(len(self.window.jobs), 1)
            self.assertIn('not loaded', panel.status.text())
            self.assertIn('Reserved range folder:', panel.result_path.text())
        finally:
            release.set()

    def test_selection_round_trip_and_protected_asset_change_revoke_snapshot(self):
        from corridorkey.asset_inventory import AssetSelection
        self.render_parent()
        panel = self.inspect_parent()
        token = self.window._diagnostic_context
        self.window.table.clearSelection()
        self.window.table.selectRow(0)
        self.assertNotEqual(self.window._diagnostic_context, token)
        self.assertIsNone(panel.snapshot)
        self.window.open_diagnostic_ranges()
        self.until(lambda: not panel.io_busy)
        self.assertIsNotNone(panel.snapshot)
        asset_root = self.root / 'new-assets'
        asset_root.mkdir()
        self.window.asset_selection = AssetSelection(asset_root)
        self.window.refresh_diagnostic_context()
        self.assertIsNone(panel.snapshot)
        self.assertFalse(panel.prepare_button.isEnabled())

    def test_parent_close_waits_for_actual_diagnostic_thread_terminal(self):
        from corridorkey.desktop_diagnostic_ranges import run_operation
        self.render_parent()
        panel = self.window.diagnostic_ranges
        entered, release = threading.Event(), threading.Event()

        def gated(request, *, check, progress):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test inspection gate was not released')
            return run_operation(request, check=check, progress=progress)

        panel.runner = gated
        try:
            self.window.open_diagnostic_ranges()
            self.until(entered.is_set)
            self.window.close()
            self.app.processEvents()
            self.assertTrue(self.window.isVisible())
            self.assertTrue(panel.io_busy)
            release.set()
            self.until(lambda: not self.window.isVisible())
            assert panel.task is not None
            self.assertTrue(panel.task.done.is_set())
            self.assertFalse(panel.task.thread.is_alive())
        finally:
            release.set()
