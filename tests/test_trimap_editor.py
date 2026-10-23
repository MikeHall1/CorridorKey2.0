"""Explicit trimap editor persistence/class semantics; no learned quality claim."""
import importlib.util
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


class _TrimapFixture(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.session import JobRequest
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(3):
            self.assertTrue(cv2.imwrite(str(self.source / f'{frame:04}.png'), np.full((6, 9, 3), 100, np.uint8)))
        self.job = JobRequest.create(self.source, self.root / 'render', InferenceConfig(output_format='exr', model_scale=.5))
        self.editor = CorrectionEditor(self.job, self.root)
        self.trimap = self.root / 'trimap.png'
        self.classes = np.tile(np.array([0, 128, 255], np.uint8), (6, 3))
        self.assertTrue(cv2.imwrite(str(self.trimap), self.classes))


class TrimapEditorTests(_TrimapFixture):
    def test_exact_asset_schema_history_reopen_and_immutable_attachment(self):
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.corrections import load_corrections
        from corridorkey.session import JobRequest
        self.assertEqual(self.editor.document.to_dict()['schema_version'], 1)
        raw = self.trimap.read_bytes()
        self.editor.import_trimap(self.trimap, frame=1, strength=1, direction='both', span=[1, 2])
        data = self.editor.document.to_dict()
        self.assertEqual(data['schema_version'], 2)
        entry = data['entries'][0]
        self.assertEqual(entry['kind'], 'trimap')
        self.assertEqual(self.editor.document.asset_path(entry).read_bytes(), raw)
        identity = self.editor.document.identity
        self.editor.undo()
        self.assertEqual(self.editor.document.to_dict()['schema_version'], 1)
        self.editor.redo()
        self.assertEqual(self.editor.document.identity, identity)
        saved = self.editor.save()
        saved_bytes = saved.read_bytes()
        cfg, writer = self.job.configurations()
        cfg.corrections_path = str(saved)
        attached = JobRequest.create(self.job.input_path, self.job.output_dir, cfg, writer)
        reopened = CorrectionEditor(attached, self.root, existing=saved)
        self.assertEqual(reopened.document.identity, identity)
        self.assertEqual(load_corrections(saved).identity, identity)
        self.editor.edit(entry['id'], {'strength': .25})
        self.assertEqual(saved.read_bytes(), saved_bytes)
        self.assertEqual(attached.configurations()[0].corrections_path, str(saved.resolve()))
        self.assertIsNone(self.job.configurations()[0].corrections_path)
        self.assertEqual(self.editor.invalidated, [(0, 3)])

    def test_classes_cannot_be_coverage_or_binary_brush_reinterpretation(self):
        import numpy as np
        self.editor.import_trimap(self.trimap, frame=0, strength=1, direction='both', span=[0, 1])
        entry = self.editor.document.to_dict()['entries'][0]
        before = self.editor.document.identity
        with self.assertRaisesRegex(ValueError, 'reinterpreted'):
            self.editor.edit(entry['id'], {'kind': 'matte'})
        with self.assertRaisesRegex(ValueError, 'explicit trimap'):
            self.editor.import_mask(self.trimap, kind='trimap', frame=0, strength=1, direction='both', span=[0, 1])
        with self.assertRaises(ValueError):
            self.editor.commit_brush(np.zeros((6, 9), np.uint8), kind='trimap', frame=0, strength=1, direction='both', span=[0, 1])
        self.assertEqual(self.editor.document.identity, before)
        self.editor.import_mask(self.trimap, kind='matte', frame=0, strength=1, direction='both', span=[0, 1])
        matte = self.editor.document.to_dict()['entries'][-1]
        with self.assertRaisesRegex(ValueError, 'reinterpreted'):
            self.editor.edit(matte['id'], {'kind': 'trimap'})

    def test_invalid_classes_and_dimensions_preserve_history_and_encoded_source(self):
        import cv2
        import numpy as np
        before = self.editor.document.identity
        for pixels in (np.full((6, 9), 127, np.uint8), np.zeros((5, 9), np.uint8),
                       np.zeros((6, 9, 4), np.uint8)):
            self.assertTrue(cv2.imwrite(str(self.trimap), pixels))
            raw = self.trimap.read_bytes()
            with self.assertRaises(ValueError):
                self.editor.import_trimap(self.trimap, frame=0, strength=1, direction='both', span=[0, 1])
            self.assertEqual(self.editor.document.identity, before)
            self.assertEqual(self.trimap.read_bytes(), raw)

    def test_trimap_admission_precedes_unbounded_hash_and_rejects_lexical_alias(self):
        from corridorkey.trimap_refinement import MAX_ENCODED_BYTES
        alias = self.root / 'alias.png'
        alias.symlink_to(self.trimap)
        oversized = self.root / 'oversized.png'
        with oversized.open('wb') as stream:
            stream.seek(MAX_ENCODED_BYTES)
            stream.write(b'\0')
        before = self.editor.document.identity
        assets = set((self.editor.workspace / 'assets').iterdir())
        with patch('corridorkey.correction_editor.file_sha256', side_effect=AssertionError('trimap input must use bounded admission')):
            for path in (alias, oversized):
                with self.subTest(path=path.name), self.assertRaises((ValueError, OSError)):
                    self.editor.import_trimap(path, frame=0, strength=1, direction='both', span=[0, 1])
                self.assertEqual(self.editor.document.identity, before)
                self.assertEqual(set((self.editor.workspace / 'assets').iterdir()), assets)
            self.editor.import_trimap(self.trimap, frame=0, strength=1, direction='both', span=[0, 1])
        self.assertEqual(self.editor.document.to_dict()['entries'][-1]['kind'], 'trimap')

    def test_native_class_preview_does_not_convert_unknown_to_coverage(self):
        import cv2
        import numpy as np
        for suffix, dtype, values in (('.png', np.uint16, [0, 32768, 65535]),
                                      ('.exr', np.float32, [0, .5, 1])):
            path = self.root / ('precision' + suffix)
            self.assertTrue(cv2.imwrite(str(path), np.tile(np.array(values, dtype=dtype), (6, 3))))
            self.editor.import_trimap(path, frame=1, strength=.5, direction='both', span=[1, 2])
            entry = self.editor.document.to_dict()['entries'][-1]
            display, overlay = self.editor.view(1, entry['id'])
            self.assertEqual((display.width, display.height), (9, 6))
            assert overlay is not None
            np.testing.assert_array_equal(overlay, self.classes)
            self.assertIsNone(self.editor.view(2, entry['id'])[1])


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class TrimapDialogTests(_TrimapFixture):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['trimap-editor', '-platform', 'offscreen'])

    def until(self, dialog, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            dialog.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail(dialog.status.text())

    def test_canvas_distinguishes_background_foreground_and_unknown(self):
        import numpy as np

        from corridorkey.correction_canvas import CorrectionCanvas
        from tests.qt_cleanup import dispose_widget
        canvas = CorrectionCanvas()
        try:
            canvas.resize(360, 240)
            canvas.set_frame(np.full((6, 9, 3), 100, np.uint8), 9, 6, self.classes, overlay_kind='trimap')
            canvas.show()
            self.app.processEvents()
            image = canvas.grab().toImage()
            background, unknown, foreground = [image.pixelColor(x, 20) for x in (20, 60, 100)]
            self.assertGreater(background.red(), background.green())
            self.assertGreater(foreground.green(), foreground.red())
            self.assertEqual((unknown.red(), unknown.green(), unknown.blue()), (100, 100, 100))
        finally:
            dispose_widget(canvas)

    def test_close_cancels_and_waits_actual_import_owner(self):
        from corridorkey.desktop_corrections import CorrectionDialog
        from corridorkey.trimap_refinement import load_trimap_asset
        from tests.qt_cleanup import dispose_widget
        entered, release = threading.Event(), threading.Event()
        def gated(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test import gate')
            return load_trimap_asset(*args, **kwargs)
        dialog = CorrectionDialog(self.job, self.root)
        try:
            dialog.show()
            self.until(dialog, lambda: dialog.snapshot is not None and dialog.pending is None)
            with patch('corridorkey.trimap_refinement.load_trimap_asset', side_effect=gated):
                dialog.request('import_trimap', path=self.trimap, frame=0, strength=1, direction='both', span=[0, 1])
                self.assertTrue(entered.wait(5))
                dialog.close()
                self.assertFalse(dialog.service.done.is_set())
                self.assertTrue(dialog.service.thread.is_alive())
                release.set()
                dialog.service.thread.join(5)
            self.assertTrue(dialog.service.done.is_set())
            self.assertFalse(dialog.service.thread.is_alive())
        finally:
            release.set()
            dialog.close()
            dialog.service.thread.join(5)
            dialog.poll()
            dispose_widget(dialog)

    def test_explicit_import_class_overlay_no_paint_save_and_owned_close(self):
        from corridorkey.desktop_corrections import CorrectionDialog
        from tests.qt_cleanup import dispose_widget
        dialog = CorrectionDialog(self.job, self.root)
        captured = []
        dialog.saved.connect(lambda job, path: captured.append((job, path)))
        try:
            dialog.show()
            self.until(dialog, lambda: dialog.snapshot is not None and dialog.pending is None)
            with patch('corridorkey.desktop_corrections.QFileDialog.getOpenFileName', return_value=(str(self.trimap), '')):
                dialog.import_trimap()
            self.until(dialog, lambda: dialog.canvas.overlay_kind == 'trimap' and dialog.pending is None)
            self.assertFalse(dialog.canvas.painting)
            self.assertFalse(dialog.kind.isEnabled())
            selected = dialog.selected()
            assert selected is not None
            self.assertEqual(selected['kind'], 'trimap')
            dialog.commit()
            self.assertIn('cannot be binary painted', dialog.status.text())
            self.assertIsNone(dialog.pending)
            dialog.save()
            self.until(dialog, lambda: bool(captured) and dialog.pending is None)
            self.assertEqual(captured[0][0], self.job)
            self.assertTrue(Path(captured[0][1]).is_file())
            dialog.new_mask()
            self.assertTrue(dialog.canvas.painting)
            self.assertTrue(dialog.kind.isEnabled())
        finally:
            dialog.close()
            dialog.service.thread.join(5)
            self.assertTrue(dialog.service.done.is_set())
            self.assertFalse(dialog.service.thread.is_alive())
            dialog.poll()
            dispose_widget(dialog)
