import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
FIXTURE = Path(__file__).with_name('worker_fixture.py')


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt binding unavailable')
class DesktopCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['correction-ui-test', '-platform', 'offscreen'])

    def setUp(self):
        import cv2
        import numpy as np

        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(4):
            cv2.imwrite(str(self.source / f'frame{frame}.png'), np.full((6, 8, 3), 1000, np.uint16))
        (self.source / '.worker-control.json').write_text(json.dumps({'gate': str(self.root / 'gate')}))
        self.window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(FIXTURE)])))
        self.window.set_source(str(self.source))
        self.window.output.setText(str(self.root))
        self.window.update_settings({'output_format': 'exr', 'outputs': ['alpha']})
        self.window.show()
        with patch('corridorkey.desktop_window.QFileDialog.getExistingDirectory', return_value=str(self.root)):
            self.window.open_editor()
        editor = self.window.editor
        assert editor is not None
        self.editor = editor
        self.until(lambda: self.editor.displayed_frame == 0 and self.editor.pending is None)

    def tearDown(self):
        self.editor.canvas.clear_draft()
        self.editor.close()
        self.editor.service.thread.join(10)
        self.window.close()
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
            if self.window.editor is not None:
                self.window.editor.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Correction UI condition not reached: ' + self.editor.status.text())

    def paint_and_commit(self, kind):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        self.editor.kind.setCurrentText(kind)
        self.editor.radius.setValue(1)
        self.paint_point = self.editor.canvas.point(QPointF(self.editor.canvas.rect().center()))
        assert self.paint_point is not None
        QTest.mouseClick(self.editor.canvas, Qt.MouseButton.LeftButton, pos=self.editor.canvas.rect().center())
        self.assertTrue(self.editor.canvas.dirty)
        count = len(self.editor.entries)
        self.editor.commit()
        self.until(lambda: len(self.editor.entries) == count + 1 and self.editor.pending is None)

    def save(self):
        self.editor.save()
        self.until(lambda: self.editor.pending is None and self.window.job().configurations()[0].corrections_path is not None)
        path = self.window.job().configurations()[0].corrections_path
        assert path is not None
        return Path(path)

    def test_refinement_region_uses_existing_brush_save_and_history_ownership(self):
        from corridorkey.corrections import load_corrections

        old = self.window.job()
        label = 'Refinement region — expansion/feather only'
        self.assertGreaterEqual(self.editor.kind.findData('region'), 0)
        self.assertIn('Inactive', self.editor.region_note.text())
        self.paint_and_commit(label)
        assert self.editor.snapshot is not None
        entry = self.editor.snapshot['document']['entries'][0]
        self.assertEqual((entry['kind'], entry['purpose'], entry['target']), ('region', 'refinement', 'edge_finishing'))
        self.assertFalse(self.editor.kind.isEnabled())
        self.assertFalse(self.editor.direction.isEnabled())
        self.assertEqual(self.editor.direction.currentText(), 'both')
        saved = self.save()
        self.assertEqual(load_corrections(saved).to_dict()['schema_version'], 5)
        self.assertIsNone(old.configurations()[0].corrections_path)
        self.editor.request('undo')
        self.until(lambda: self.editor.pending is None and self.editor.snapshot is not None
                   and not self.editor.snapshot['document']['entries'])
        self.editor.request('redo')
        self.until(lambda: self.editor.pending is None and self.editor.snapshot is not None
                   and len(self.editor.snapshot['document']['entries']) == 1)
        assert self.editor.snapshot is not None
        self.assertEqual(self.editor.snapshot['document']['entries'][0]['id'], entry['id'])
        self.editor.new_mask()
        self.assertEqual(self.editor.parameters()['kind'], 'region')
        self.assertFalse(self.editor.direction.isEnabled())

    def test_real_queued_jobs_keep_old_manifest_and_native_corrected_outputs(self):
        import cv2
        self.window.enqueue()
        first = next(iter(self.window.jobs))
        self.until(lambda: self.window.jobs[first]['status'] == 'progress')
        self.paint_and_commit('keep')
        first_save = self.save()
        first_bytes = first_save.read_bytes()
        self.window.enqueue()
        second = list(self.window.jobs)[1]
        self.editor.new_mask()
        self.paint_and_commit('remove')
        self.editor.save()
        self.until(lambda: self.editor.pending is None and str(self.window.job().configurations()[0].corrections_path) != str(first_save))
        self.window.enqueue()
        third = list(self.window.jobs)[2]
        self.assertIsNone(self.window.jobs[first]['job'].configurations()[0].corrections_path)
        self.assertEqual(str(self.window.jobs[second]['job'].configurations()[0].corrections_path), str(first_save))
        self.assertEqual(first_save.read_bytes(), first_bytes)
        (self.root / 'gate').touch()
        self.until(lambda: all(self.window.jobs[key]['status'] == 'result' for key in [first, second, third]))
        values = []
        for key in [first, second, third]:
            output = Path(self.window.jobs[key]['event']['payload']['record']['diagnostics']['outputs']['alpha'])
            image = cv2.imread(str(min(output.glob('*.exr'))), -1)
            assert self.paint_point is not None
            x, y = self.paint_point
            values.append(image[y, x])
        self.assertEqual(values, [.5, 1., 0.])

    def test_draft_close_guard_history_and_recovery_snapshots(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        QTest.mouseClick(self.editor.canvas, Qt.MouseButton.LeftButton, pos=self.editor.canvas.rect().center())
        self.editor.close()
        self.assertTrue(self.editor.isVisible())
        self.assertFalse(self.editor.service.stopping.is_set())
        self.editor.clear_draft()
        self.paint_and_commit('keep')
        snapshot = self.editor.snapshot
        assert snapshot is not None
        self.assertTrue(Path(snapshot['checkpoint_path']).is_file())
        self.editor.request('undo')
        self.until(lambda: self.editor.pending is None and not self.editor.entries)
        self.editor.request('redo')
        self.until(lambda: self.editor.pending is None and len(self.editor.entries) == 1)
        snapshot = self.editor.snapshot
        assert snapshot is not None
        self.assertEqual(snapshot['invalidated'], [(0, 4)])

    def test_timeline_changes_prevent_wrong_attachment_and_reset_controls_work(self):
        self.editor.request('reset', frame=2)
        self.until(lambda: self.editor.pending is None and len(self.editor.entries) == 1)
        self.editor.request('shot', frame=2)
        self.until(lambda: self.editor.pending is None and self.editor.snapshot is not None and self.editor.snapshot['document']['shots'] == [[0, 2], [2, 4]])
        self.window.update_settings({'num_frames': 2})
        self.editor.save()
        self.until(lambda: self.editor.pending is None)
        self.assertIsNone(self.window.job().configurations()[0].corrections_path)
        self.assertIn('timeline differs', self.window.details.toPlainText())

    def test_loading_outside_selected_entry_range_preserves_viewer_frame(self):
        self.paint_and_commit('keep')
        self.editor.frame.setValue(3)
        self.editor.load_frame()
        self.until(lambda: self.editor.pending is None and self.editor.displayed_frame == 3)
        self.assertEqual(self.editor.frame.value(), 3)
        self.assertIsNone(self.editor.canvas.saved_overlay)

    def test_editing_range_preserves_unedited_imported_strength_precision(self):
        import cv2
        import numpy as np
        mask = self.root / 'soft.png'
        cv2.imwrite(str(mask), np.full((6, 8), 30000, np.uint16))
        self.editor.request('import', path=mask, kind='matte', frame=0, strength=.123456789,
                            direction='both', span=[0, 1])
        self.until(lambda: self.editor.pending is None and len(self.editor.entries) == 1)
        self.assertEqual(self.editor.strength.value(), .123)
        self.editor.end.setValue(2)
        self.editor.edit()
        self.until(lambda: self.editor.pending is None and self.editor.entries[0]['range'] == [0, 2])
        self.assertEqual(self.editor.entries[0]['strength'], .123456789)

    def test_actor_ready_does_not_admit_commands_before_initial_ui_snapshot(self):
        from corridorkey.desktop_corrections import CorrectionDialog
        other = CorrectionDialog(self.window.job(), self.root)
        try:
            self.assertTrue(other.service.ready.wait(10))
            self.assertIsNone(other.snapshot)
            self.assertFalse(other.request('reset', frame=0))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                other.poll()
                self.app.processEvents()
                if other.snapshot is not None and other.pending is None:
                    break
                time.sleep(.01)
            self.assertIsNotNone(other.snapshot)
            self.assertTrue(other.request('reset', frame=0))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                other.poll()
                if other.pending is None:
                    break
                time.sleep(.01)
            self.assertEqual(len(other.entries), 1)
        finally:
            other.canvas.clear_draft()
            other.close()
            other.service.thread.join(10)
            other.poll()
            from tests.qt_cleanup import dispose_widget
            dispose_widget(other)

    def test_reopening_retires_stopped_native_editor_and_keeps_dirty_draft_guard(self):
        from PySide6.QtCore import QCoreApplication, QEvent

        from corridorkey.desktop_corrections import CorrectionDialog
        for _ in range(3):
            previous = self.editor
            destroyed = []
            previous.destroyed.connect(lambda *_, events=destroyed: events.append(True))
            previous.close()
            previous.service.thread.join(10)
            previous.poll()
            with patch('corridorkey.desktop_window.QFileDialog.getExistingDirectory', return_value=str(self.root)):
                self.window.open_editor()
            assert self.window.editor is not None
            self.editor = self.window.editor
            self.until(lambda: self.editor.displayed_frame == 0 and self.editor.pending is None)
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.assertTrue(destroyed)
            self.assertEqual(len(self.window.findChildren(CorrectionDialog)), 1)
        self.editor.close()
        self.editor.service.thread.join(10)
        self.editor.canvas.dirty = True
        current = self.editor
        self.window.open_editor()
        self.assertIs(self.window.editor, current)
        self.assertIn('Close the current', self.window.details.toPlainText())
        self.editor.canvas.clear_draft()

    def test_parent_close_preserves_dirty_draft_before_any_owner_shutdown(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        for stopped in (False, True):
            if stopped:
                self.editor.service.stop()
                self.editor.service.thread.join(10)
            QTest.mouseClick(self.editor.canvas, Qt.MouseButton.LeftButton, pos=self.editor.canvas.rect().center())
            self.assertTrue(self.editor.canvas.dirty)
            self.assertFalse(self.window.close())
            self.assertTrue(self.window.isVisible())
            self.assertFalse(self.window.closing)
            self.assertFalse(self.window.bridge.stopping.is_set())
            self.assertFalse(self.window.loader.stopping.is_set())
            self.assertTrue(self.editor.canvas.dirty)
            if not stopped:
                self.assertFalse(self.editor.service.stopping.is_set())
            self.editor.clear_draft()
        self.window.close()
        self.window.bridge.thread.join(10)
        self.until(lambda: not self.window.isVisible())
        self.assertFalse(self.window.loader.thread.is_alive())
        self.assertFalse(self.window.isVisible())

    def test_async_editor_close_blocks_strokes_and_late_results_cannot_enable_them(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        with patch.object(self.editor.service, 'stop'):
            self.assertFalse(self.editor.close())
        self.assertTrue(self.editor.closing)
        self.assertFalse(self.editor.canvas.painting)
        QTest.mouseClick(self.editor.canvas, Qt.MouseButton.LeftButton, pos=self.editor.canvas.rect().center())
        self.assertFalse(self.editor.canvas.dirty)
        self.editor.service.events.put({'kind': 'error', 'serial': 999, 'error': 'fixture late result'})
        self.editor.poll()
        self.assertFalse(self.editor.canvas.painting)
        self.assertFalse(self.editor.request('reset', frame=0))
        self.editor.service.stop()
        self.editor.service.thread.join(10)
        self.editor.poll()
