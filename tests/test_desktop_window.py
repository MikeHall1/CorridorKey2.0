"""Actual offscreen widgets and child transport; no neural-quality assertions."""
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
FIXTURE = Path(__file__).with_name('worker_fixture.py')


@unittest.skipUnless(QT_AVAILABLE, 'Optional requirements-desktop.txt is not installed')
class DesktopWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['desktop-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        bridge = DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(FIXTURE)]))
        self.window = DesktopWindow(bridge)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.bridge.thread.join(10)
        if self.window.ofx_cache.task is not None:
            self.window.ofx_cache.task.thread.join(10)
        if self.window.cache_panel.task is not None:
            self.window.cache_panel.task.thread.join(10)
        if self.window.cache_panel.backup_panel.task is not None:
            self.window.cache_panel.backup_panel.task.thread.join(10)
        if self.window.queue_panel.transfer_panel.task is not None:
            self.window.queue_panel.transfer_panel.task.thread.join(10)
        if self.window.assets.audit_panel.task is not None:
            self.window.assets.audit_panel.task.thread.join(10)
        if self.window.assets.download_panel.task is not None:
            self.window.assets.download_panel.task.thread.join(10)
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
            time.sleep(0.01)
        self.fail('Desktop condition not reached')

    def test_source_preview_and_project_reopen_do_not_enqueue(self):
        import cv2
        import numpy as np

        from corridorkey.desktop_state import save_project
        source = self.root / 'frames'
        source.mkdir()
        cv2.imwrite(str(source / 'frame1.png'), np.full((20, 30, 3), 100, np.uint16))
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root / 'outputs'))
        self.window.request_preview()
        self.until(lambda: self.window.pixmap is not None)
        self.assertIn('30×20', self.window.preview_note.text())
        self.assertFalse(self.window.jobs)

        self.window.draft_quality()
        self.window.limit.setValue(7)
        self.window.update_settings({'foundation_cache_mb': 13})
        path = self.root / 'project.json'
        save_project(path, self.window.job(), 2)
        self.window.full_quality()
        self.window.open_project(path)
        config, _ = self.window.job().configurations()
        self.assertTrue(config.quick_key)
        self.assertEqual(config.foundation_cache_mb, 13)
        self.assertEqual(config.num_frames, 7)
        self.assertEqual(self.window.frame.value(), 2)
        self.assertFalse(self.window.jobs)

    def test_cache_inventory_close_waits_for_actual_owned_terminal(self):
        from corridorkey.cache_maintenance import inventory
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore
        source = self.root / 'source.bin'
        source.write_bytes(b'original source')
        workspace = self.root / 'cache'
        key = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'desktop'}))
        with BundleStore(workspace, 10000) as store:
            store.publish(key, {'alpha.exr': source}, {})
        original = {str(p): p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
        entered, release = threading.Event(), threading.Event()

        def gated(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture gate')
            return inventory(*args, **kwargs)

        self.window.recovery.workspace.setText(str(workspace))
        self.window.open_cache_management()
        panel = self.window.cache_panel
        self.assertEqual(panel.root_path.text(), str(workspace))
        self.assertFalse(self.window.jobs)
        with patch('corridorkey.cache_maintenance.inventory', side_effect=gated):
            panel.start('inventory')
            task = panel.task
            assert task is not None
            try:
                self.assertTrue(entered.wait(5))
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(panel.io_busy)
                self.assertTrue(task.cancellation.cancelled)
            finally:
                release.set()
            self.until(lambda: task.done.is_set() and not self.window.isVisible())
            task.thread.join(5)
            self.assertFalse(task.thread.is_alive())
        self.assertEqual(source.read_bytes(), b'original source')
        self.assertEqual(original, {str(p): p.read_bytes() for p in workspace.rglob('*') if p.is_file()})
        self.assertFalse(self.window.jobs)

    def test_backup_child_close_retains_main_window_until_terminal(self):
        from corridorkey.cache_backup import backup_workspace
        entered, release = threading.Event(), threading.Event()

        def gated(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture backup gate')
            return backup_workspace(*args, **kwargs)

        self.window.open_cache_management()
        maintenance = self.window.cache_panel
        maintenance.open_backup()
        panel = maintenance.backup_panel
        panel.source_path.setText(str(self.root))
        panel.destination_parent.setText(str(self.root))
        with patch('corridorkey.cache_backup.backup_workspace', side_effect=gated):
            panel.start()
            task = panel.task
            assert task is not None
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(maintenance.io_busy)
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(task.cancellation.cancelled)
            finally:
                release.set()
            self.until(lambda: task.done.is_set() and not self.window.isVisible())
            task.thread.join(5)
            self.assertFalse(task.thread.is_alive())
        self.assertFalse(self.window.jobs)
        self.assertFalse(list(self.root.glob('recovery-backup-*')))

    def test_queue_transfer_child_close_retains_main_window_until_terminal(self):
        from corridorkey.queue_transfer import export_history
        entered, release = threading.Event(), threading.Event()

        def gated(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture queue transfer gate')
            return export_history(*args, **kwargs)

        panel = self.window.queue_panel.transfer_panel
        panel.show()
        panel.source_path.setText(str(self.root))
        panel.destination_parent.setText(str(self.root))
        with patch('corridorkey.queue_transfer.export_history', side_effect=gated):
            panel.start()
            task = panel.task
            assert task is not None
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(self.window.queue_panel.io_busy)
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(task.cancellation.cancelled)
            finally:
                release.set()
            self.until(lambda: task.done.is_set() and not self.window.isVisible())
            task.thread.join(5)
            self.assertFalse(task.thread.is_alive())
        self.assertFalse(self.window.jobs)
        self.assertFalse(list(self.root.glob('queue-export-*')))

    def test_asset_audit_child_close_retains_main_window_until_terminal(self):
        from corridorkey.asset_audit import audit_assets
        entered, release = threading.Event(), threading.Event()

        def gated(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture asset audit gate')
            return audit_assets(*args, **kwargs)

        self.window.assets.root.setText(str(self.root))
        self.window.assets.open_audit()
        panel = self.window.assets.audit_panel
        with patch('corridorkey.asset_audit.audit_assets', side_effect=gated):
            panel.start()
            task = panel.task
            assert task is not None
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(self.window.assets.io_busy)
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(task.cancellation.cancelled)
            finally:
                release.set()
            self.until(lambda: task.done.is_set() and not self.window.isVisible())
            task.thread.join(5)
            self.assertFalse(task.thread.is_alive())
        self.assertFalse(self.window.jobs)
        self.assertFalse(list(self.root.rglob('*')))

    def test_asset_download_plan_close_retains_main_until_owned_terminal(self):
        entered, release = threading.Event(), threading.Event()

        def gated(*, check=None, progress=None, transport=None):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture download plan gate')
            if check is not None:
                check()
            return {'status': 'failed', 'plan': None, 'error': 'fixture terminal', 'warning': None}

        self.window.assets.root.setText(str(self.root))
        self.window.assets.open_download()
        panel = self.window.assets.download_panel
        self.until(lambda: not self.window.assets.io_busy)
        with patch('corridorkey.asset_download.resolve_plan', side_effect=gated):
            panel.resolve_plan()
            task = panel.task
            assert task is not None
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(self.window.assets.io_busy)
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(task.cancellation.cancelled)
            finally:
                release.set()
            self.until(lambda: task.done.is_set() and not self.window.isVisible())
            task.thread.join(5)
            self.assertFalse(task.thread.is_alive())
        self.assertFalse(self.window.jobs)
        self.assertFalse(list(self.root.rglob('*')))

    def test_actual_queue_export_preview_cancel_and_snapshot(self):
        source = self.root / 'source.json'
        source.write_text('{}')
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root))
        self.window.update_settings({'output_format': 'exr'})
        self.window.enqueue()
        first = next(iter(self.window.jobs))
        self.window.update_settings({'despill_strength': 0.1})
        self.until(lambda: self.window.jobs[first]['status'] == 'result')
        first_job = self.window.jobs[first]['job']
        self.assertEqual(first_job.configurations()[0].despill_strength, 0.5)
        self.window.table.selectRow(0)
        self.window.view.setCurrentText('Alpha')
        self.window.request_preview()
        self.until(lambda: self.window.pixmap is not None)
        self.assertIn('Coverage', self.window.preview_note.text())
        self.assertTrue(Path(first_job.output_dir).is_dir())
        source.write_text(json.dumps({'gate': str(self.root / 'gate')}))
        self.window.enqueue()
        second = list(self.window.jobs)[1]
        self.until(lambda: self.window.jobs[second]['status'] == 'progress')
        self.window.table.selectRow(1)
        self.window.cancel_selected()
        self.until(lambda: self.window.jobs[second]['status'] == 'cancelled')
        self.assertIn('cancelled', self.window.statusBar().currentMessage())
        self.window.clear_finished()
        self.assertEqual(self.window.table.rowCount(), 0)

    def test_actual_window_grab_and_explicit_export_errors(self):
        self.window.view.setCurrentText('Alpha')
        self.window.request_preview()
        self.assertIn('completed render', self.window.details.toPlainText())
        self.assertTrue(self.window.grab().save(str(self.root / 'window.png')))

    def test_ab_snapshot_retains_pixels_and_editor_state_is_honest(self):
        import cv2
        import numpy as np
        source = self.root / 'frames'
        source.mkdir()
        image = source / 'frame1.png'
        cv2.imwrite(str(image), np.zeros((20, 30, 3), np.uint8))
        self.window.set_source(str(source))
        self.window.request_preview()
        self.until(lambda: self.window.pixmap is not None)
        self.window.pin_reference()
        assert self.window.reference is not None
        reference = self.window.reference.toImage().pixelColor(0, 0).red()
        cv2.imwrite(str(image), np.full((20, 30, 3), 255, np.uint8))
        self.window.request_preview()
        self.assertIsNone(self.window.pixmap)
        self.until(lambda: self.window.pixmap is not None)
        assert self.window.pixmap is not None
        self.assertEqual(self.window.pixmap.toImage().pixelColor(0, 0).red(), 255)
        assert self.window.reference is not None
        self.assertEqual(self.window.reference.toImage().pixelColor(0, 0).red(), reference)
        settings = self.window._settings()
        settings.update(num_frames=9, start_mode='middle')
        self.window.config.setPlainText(json.dumps(settings))
        self.assertEqual(self.window.limit.value(), 9)
        self.assertEqual(self.window.start.currentText(), 'middle')
        pending_commands = self.window.bridge.commands.qsize()
        self.window.config.setPlainText('{')
        self.assertIn('incomplete', self.window.quality_note.text())
        self.assertIn('validate', self.window.quality_note.text())
        self.window.draft_quality()
        self.assertIn('incomplete', self.window.quality_note.text())
        self.assertEqual(self.window.config.toPlainText(), '{')
        self.assertEqual(self.window.bridge.commands.qsize(), pending_commands)
        self.assertEqual(self.window.reference.toImage().pixelColor(0, 0).red(), reference)
        self.assertIsNone(self.window.pixmap)

    def test_crash_records_are_consumed_before_explicit_new_worker(self):
        source = self.root / 'source.json'
        source.write_text('{"exit":true}')
        self.window.update_settings({'output_format': 'exr'})
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root))
        self.window.enqueue()
        request_id = next(iter(self.window.jobs))
        self.assertTrue(self.window.bridge.done.wait(10))
        self.window.new_worker()
        self.assertEqual(self.window.jobs[request_id]['status'], 'lost')
        self.assertTrue(self.window.jobs[request_id]['event']['payload']['outputs_may_be_partial'])
        self.assertFalse(self.window.bridge.commands.qsize())

    def test_synchronized_exports_follow_actual_source_offsets_and_reject_stale_pairs(self):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest
        source = self.root / 'clip.avi'
        writer = cv2.VideoWriter(str(source), int.from_bytes(b'MJPG', 'little'), 24., (32, 24))
        self.assertTrue(writer.isOpened())
        try:
            for i in range(6):
                writer.write(np.full((24, 32, 3), i * 30, np.uint8))
        finally:
            writer.release()
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root))
        for name, start, values in [('a', 1, [.1, .3, .5]), ('b', 2, [.7, .8, .9])]:
            output = self.root / name
            output.mkdir()
            for i, value in enumerate(values):
                cv2.imwrite(str(output / f'{i}.exr'), np.full((24, 32), value, np.float32))
            job = JobRequest.create(str(source), str(output), InferenceConfig(output_format='exr'))
            record = {'status': 'complete', 'input': job.input_path, 'diagnostics': {
                'source': {'start_frame': start, 'num_frames': 3, 'native_hw': [24, 32], 'native_fps': 24.},
                'frames_submitted': 3, 'outputs': {'alpha': str(output)}}}
            self.window.jobs[name] = {'job': job, 'status': 'result', 'event': {'payload': {'record': record}}}
        self.window.refresh_table()
        self.window.table.selectRow(0)
        self.window.view.setCurrentText('Alpha')
        self.window.synchronize_reference()
        self.until(lambda: self.window.reference is not None)
        self.window.table.selectRow(1)
        self.window.frame.setValue(2)
        self.until(lambda: self.window.reference is not None)
        assert self.window.reference is not None and self.window.pixmap is not None
        self.assertEqual(self.window.reference.toImage().pixelColor(0, 0).red(), 76)
        self.assertEqual(self.window.pixmap.toImage().pixelColor(0, 0).red(), 178)
        self.assertIn('source frame 2', self.window.reference_note.text())
        self.assertIn('export frame 1', self.window.reference_note.text())
        self.window.view.setCurrentText('Source')
        self.until(lambda: self.window.pixmap is not None)
        assert self.window.pixmap is not None
        self.assertAlmostEqual(self.window.pixmap.toImage().pixelColor(0, 0).red(), 60, delta=3)
        self.window.view.setCurrentText('Alpha')
        self.window.frame.setValue(5)
        self.assertEqual(self.window.preview_serial, -1)
        self.assertIsNone(self.window.pixmap)
        self.assertIn('common source frame', self.window.preview_note.text())
        self.window.loader.thread.join(.05)
        self.window.poll()
        self.assertIsNone(self.window.pixmap)
        self.assertEqual(self.window.bridge.commands.qsize(), 0)
        self.window.clear_reference()
        self.assertIsNone(self.window.moving_reference)

    def test_synchronized_a_refuses_old_records_without_selection_metadata(self):
        source = self.root / 'source.json'
        source.write_text('{}')
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root))
        self.window.update_settings({'output_format': 'exr'})
        self.window.enqueue()
        request_id = next(iter(self.window.jobs))
        self.until(lambda: self.window.jobs[request_id]['status'] == 'result')
        self.window.table.selectRow(0)
        self.window.view.setCurrentText('Alpha')
        self.window.synchronize_reference()
        self.assertIsNone(self.window.moving_reference)
        self.assertIn('no recorded source range', self.window.details.toPlainText())

    def test_source_in_out_render_project_and_marking_preserve_queued_settings(self):
        import cv2
        import numpy as np

        from corridorkey.desktop_state import save_project
        source = self.root / 'frames'
        source.mkdir()
        (source / '.worker-control.json').write_text('{}')
        for index in range(5):
            cv2.imwrite(str(source / f'{index}.png'), np.full((12, 16, 3), index * 40, np.uint8))
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root))
        self.window.update_settings({'output_format': 'exr'})
        self.window.range_in.setText('1')
        self.window.range_out.setText('3')
        self.window.apply_source_range()
        config, _ = self.window.job().configurations()
        self.assertEqual((config.start_frame, config.num_frames), (1, 2))
        self.window.enqueue()
        request_id = next(iter(self.window.jobs))
        self.until(lambda: self.window.jobs[request_id]['status'] == 'result')
        record = self.window.jobs[request_id]['event']['payload']['record']
        self.assertEqual(record['diagnostics']['source']['start_frame'], 1)
        self.assertEqual(record['diagnostics']['frames_submitted'], 2)
        project = self.root / 'range-project.json'
        save_project(project, self.window.job(), 0)
        self.window.range_out.setText('1')
        original = self.window.config.toPlainText()
        self.window.apply_source_range()
        self.assertEqual(self.window.config.toPlainText(), original)
        self.window.table.selectRow(0)
        self.window.view.setCurrentText('Alpha')
        self.window.frame.setValue(1)
        self.window.mark_range(True)
        self.assertEqual(self.window.range_out.text(), '3')
        self.window.range_in.setText('2')
        self.window.range_out.clear()
        self.window.apply_source_range()
        self.assertEqual(self.window.job().configurations()[0].start_frame, 2)
        self.assertEqual(self.window.jobs[request_id]['job'].configurations()[0].start_frame, 1)
        self.window.synchronize_reference()
        self.until(lambda: self.window.reference is not None)
        self.window.frame.setValue(2)
        self.until(lambda: self.window.reference is not None)
        self.window.mark_range(False)
        self.assertEqual(self.window.range_in.text(), '2')
        self.window.open_project(project)
        self.assertIsNone(self.window.moving_reference)
        self.assertEqual((self.window.range_in.text(), self.window.range_out.text()), ('1', '3'))
        self.window.start.setCurrentText('middle')
        self.assertIsNone(self.window._settings()['start_frame'])

    def completed_cache_job(self):
        import cv2
        import numpy as np
        source = self.root / 'cache-source'
        source.mkdir()
        (source / '.worker-control.json').write_text('{"export_fg":true}')
        for frame in range(4):
            cv2.imwrite(str(source / f'{frame}.png'), np.full((6, 8, 3), 100, np.uint8))
        self.window.set_source(str(source))
        self.window.output.setText(str(self.root))
        self.window.update_settings({'output_format': 'exr', 'start_frame': 1, 'num_frames': 2})
        self.window.enqueue()
        identifier = next(iter(self.window.jobs))
        self.until(lambda: self.window.jobs[identifier]['status'] == 'result')
        self.window.table.selectRow(0)
        self.window.open_ofx_cache()
        return identifier

    def test_prepare_ofx_from_actual_worker_record_keeps_selection_and_submits_no_render(self):
        from corridorkey.ofx_cache import PREFIX
        identifier = self.completed_cache_job()
        original = self.window.jobs[identifier]['job'].config_json
        panel = self.window.ofx_cache
        self.assertTrue(panel.isVisible())
        self.assertIn('[1, 3)', panel.selection.text())
        self.assertIn('Test-only', panel.selection.text())
        self.assertFalse(panel.cache_path.text())
        self.window.update_settings({'start_frame': 0, 'num_frames': 1})
        panel.parent_path.setText(str(self.root))
        panel.prepare()
        self.until(lambda: bool(panel.cache_path.text()) and not panel.io_busy)
        cache = Path(panel.cache_path.text())
        header = PREFIX.unpack(cache.read_bytes()[:PREFIX.size])
        self.assertEqual(header[1:5], (8, 6, 2, 1))
        self.assertEqual(len(self.window.jobs), 1)
        self.assertEqual(self.window.jobs[identifier]['job'].config_json, original)
        self.assertTrue(panel.copy_button.isEnabled())

    def test_parent_close_cancels_and_waits_for_owned_cache_preparation(self):
        import threading
        from unittest.mock import patch

        from corridorkey.ofx_cache import seal
        self.completed_cache_job()
        panel = self.window.ofx_cache
        entered, release = threading.Event(), threading.Event()

        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture gate')
            return seal(*args, **kwargs)

        with patch('corridorkey.ofx_cache.seal', side_effect=delayed):
            try:
                panel.parent_path.setText(str(self.root))
                panel.prepare()
                self.assertTrue(entered.wait(5))
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(panel.io_busy)
                assert panel.task is not None
                self.assertTrue(panel.task.cancellation.cancelled)
                self.window.open_ofx_cache()
                self.assertIn('closing', self.window.details.toPlainText())
            finally:
                release.set()
            self.until(lambda: not self.window.isVisible())
        self.assertFalse(list(self.root.rglob('*.ckcache')))
        self.assertEqual(len(self.window.jobs), 1)

    def test_invalid_cache_selection_cannot_prepare_an_older_render(self):
        identifier = self.completed_cache_job()
        record = self.window.jobs[identifier]['event']['payload']['record']
        panel = self.window.ofx_cache
        with self.assertRaises(ValueError):
            panel.load_record({**record, 'status': 'failed'})
        self.assertIsNone(panel.record)
        self.assertFalse(panel.prepare_button.isEnabled())
        panel.load_record(record)
        self.assertTrue(panel.prepare_button.isEnabled())
        self.window.table.clearSelection()
        self.assertIsNone(self.window.selected())
        self.window.open_ofx_cache()
        self.assertIsNone(panel.record)
        self.assertFalse(panel.prepare_button.isEnabled())
        self.assertFalse(self.window.ofx_cache.io_busy)
