import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.correction_rasters import CorrectionRasters
from corridorkey.corrections import load_corrections
from corridorkey.session import JobRequest


class CorrectionEditorTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(4):
            cv2.imwrite(str(self.source / f'frame{frame}.png'), np.full((6, 8, 3), 1000, np.uint16))
        self.job = JobRequest.create(self.source, self.root / 'render', InferenceConfig(output_format='exr'))
        self.editor = CorrectionEditor(self.job, self.root)

    def test_native_brush_history_save_and_queued_snapshot_are_immutable(self):
        mask = np.zeros((6, 8), np.uint8)
        mask[1:4, 2:5] = 255
        self.editor.commit_brush(mask, kind='keep', frame=1, strength=.5, direction='both', span=[1, 2])
        first = self.editor.save()
        original = first.read_bytes()
        document = load_corrections(first, source=self.source)
        rasters = CorrectionRasters(document, self.editor.meta)
        result = rasters.apply(np.zeros((6, 8), np.float32), 1)
        self.assertEqual(result[2, 3], .5)
        self.assertEqual(result[0, 0], 0)
        rasters.close()
        self.assertEqual(self.editor.invalidated, [(0, 4)])
        self.editor.reset(2)
        second = self.editor.save()
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), original)
        self.editor.undo()
        self.assertEqual(self.editor.document.identity, document.identity)
        self.editor.redo()
        self.assertEqual(self.editor.document.resets, (2,))
        self.editor.undo()
        self.editor.reset(3)
        self.assertGreater(self.editor.document.to_dict()['revision'], 2)
        with self.assertRaises(ValueError):
            self.editor.redo()

    def test_import_preserves_soft_float_bytes_and_reopen_geometry(self):
        path = self.root / 'matte.exr'
        cv2.imwrite(str(path), np.full((6, 8), .33333334, np.float32))
        raw = path.read_bytes()
        self.editor.import_mask(path, kind='matte', frame=1, strength=1, direction='forward', span=[1, 3])
        entry = self.editor.document.to_dict()['entries'][0]
        self.assertEqual(self.editor.document.asset_path(entry).read_bytes(), raw)
        saved = self.editor.save()
        restored = CorrectionEditor(self.job, self.root, existing=saved)
        self.assertEqual(restored.document.identity, self.editor.document.identity)
        view, overlay = restored.view(1, entry['id'])
        self.assertEqual((view.width, view.height), (8, 6))
        self.assertIsNotNone(overlay)
        self.editor.shot(3)
        self.assertEqual(self.editor.document.to_dict()['shots'], [[0, 3], [3, 4]])
        with self.assertRaises(ValueError):
            self.editor.shot(2)

    def test_source_protection_and_invalid_mask_do_not_change_history(self):
        with self.assertRaises(ValueError):
            CorrectionEditor(self.job, self.source)
        bad = self.root / 'bad.png'
        cv2.imwrite(str(bad), np.zeros((5, 8), np.uint8))
        before = self.editor.document.identity
        with self.assertRaises(ValueError):
            self.editor.import_mask(bad, kind='remove', frame=0, strength=1, direction='both', span=[0, 1])
        self.assertEqual(self.editor.document.identity, before)
        cv2.imwrite(str(self.source / 'frame0.png'), np.zeros((6, 8, 3), np.uint16))
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.editor.save()

    def test_inactive_range_has_no_overlay_and_history_is_bounded(self):
        from corridorkey.corrections import CorrectionDocument, CorrectionHistory
        mask = np.full((6, 8), 255, np.uint8)
        self.editor.commit_brush(mask, kind='keep', frame=1, strength=1, direction='both', span=[1, 2])
        entry = self.editor.document.to_dict()['entries'][0]
        self.assertIsNone(self.editor.view(3, entry['id'])[1])
        history = CorrectionHistory(self.editor.document, max_snapshots=3)
        for revision in range(2, 6):
            data = self.editor.document.to_dict()
            data['revision'] = revision
            history.apply(CorrectionDocument.parse(data))
        history.undo()
        history.undo()
        with self.assertRaises(ValueError):
            history.undo()

    def test_failed_checkpoint_leaves_selected_history_unchanged(self):
        from unittest.mock import patch
        before = self.editor.document.identity
        with patch('corridorkey.correction_editor.save_corrections', side_effect=OSError('disk full')), self.assertRaises(OSError):
            self.editor.reset(1)
        self.assertEqual(self.editor.document.identity, before)
        self.editor.reset(1)
        changed = self.editor.document.identity
        with patch('corridorkey.correction_editor.save_corrections', side_effect=OSError('disk full')), self.assertRaises(OSError):
            self.editor.undo()
        self.assertEqual(self.editor.document.identity, changed)

    def test_editor_canvas_admission_precedes_native_mask_allocation(self):
        from unittest.mock import patch
        metadata = {**self.editor.meta, 'native_hw': (100000, 100000)}
        existing = set(self.root.glob('corrections-*'))
        with patch('corridorkey.video_io.probe_frame_dir_meta', return_value=metadata), \
             self.assertRaisesRegex(ValueError, '64Mi-pixel'):
            CorrectionEditor(self.job, self.root)
        self.assertEqual(set(self.root.glob('corrections-*')), existing)

    def test_strict_correction_read_and_failed_oversized_save_preserve_files(self):
        from corridorkey.corrections import (
            MAX_CORRECTION_BYTES,
            CorrectionDocument,
            save_corrections,
        )
        path = self.editor.save()
        raw = path.read_text()
        path.write_text(raw.replace('"revision": 0', '"revision": 0, "revision": 9'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_corrections(path)
        path.write_text('original destination')
        data = self.editor.document.to_dict()
        data['entries'] = [{'id': 'x' * MAX_CORRECTION_BYTES, 'revision': 1, 'frame': 0, 'kind': 'reset'}]
        with self.assertRaisesRegex(ValueError, '4MiB'):
            save_corrections(path, CorrectionDocument.parse(data))
        self.assertEqual(path.read_text(), 'original destination')
