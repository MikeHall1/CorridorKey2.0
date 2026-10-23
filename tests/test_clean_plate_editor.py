"""Native plate/trimap previews and atomic origin publication; no neural model."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.correction_rasters import CorrectionRasters
from corridorkey.corrections import canonical, load_corrections
from corridorkey.session import JobRequest


class CleanPlateEditorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        reference = np.zeros((24, 32, 3), np.uint8)
        reference[:] = [20, 150, 20]
        reference[:, :, 1] += np.arange(32, dtype=np.uint8)[None, :]
        source = reference.copy()
        source[8:16, 10:22] = [20, 20, 220]
        for frame in range(4):
            self.assertTrue(cv2.imwrite(str(self.source / f'{frame}.png'), source))
        self.reference = self.root / 'reference.png'
        self.assertTrue(cv2.imwrite(str(self.reference), reference))
        samples = np.full((24, 32), 255, np.uint8)
        samples[6:18, 8:24] = 0
        self.samples = self.root / 'samples.png'
        self.assertTrue(cv2.imwrite(str(self.samples), samples))
        self.originals = {path: path.read_bytes() for path in [*self.source.iterdir(), self.reference, self.samples]}
        self.job = JobRequest.create(self.source, self.root / 'output', InferenceConfig(output_format='exr', start_frame=1))
        self.editor = CorrectionEditor(self.job, self.root)

    def prepare(self, **changes):
        parameters = {'frame': 0, 'reference_path': self.reference, 'reference_profile': 'srgb', 'samples_path': self.samples}
        self.editor.prepare_plate(**(parameters | changes))

    def test_explicit_inspection_accept_origin_save_reopen_edit_disable_and_source_preservation(self):
        before = self.editor.document.identity
        self.prepare()
        self.assertEqual(self.editor.document.identity, before)
        summary = self.editor.plate_summary()
        assert summary is not None
        self.assertTrue(summary['usable'])
        self.assertEqual(summary['source_frame'], 1)
        summary['request']['settings']['alignment'] = 'outside mutation'
        fresh = self.editor.plate_summary()
        assert fresh is not None
        self.assertEqual(fresh['request']['settings']['alignment'], 'declared')
        with self.assertRaisesRegex(ValueError, 'Inspect'):
            self.editor.accept_plate()
        self.editor.plate_view('screen_field')
        with self.assertRaisesRegex(ValueError, 'Inspect'):
            self.editor.accept_plate()
        _, overlay = self.editor.plate_view('guidance')
        assert overlay is not None
        self.assertEqual(overlay[11, 15], 255)
        self.editor.accept_plate()
        data = self.editor.document.to_dict()
        self.assertEqual(data['schema_version'], 4)
        self.assertEqual(len(data['entries']), 1)
        self.assertEqual(len(data['guidance']), 1)
        self.assertNotIn(str(self.samples), canonical(data['guidance'][0]['binding']))
        self.assertEqual(self.editor.revision, 1)
        self.assertEqual(data['guidance'][0]['observation']['source_samples'][0]['source_frame'], 1)
        path = self.editor.save()
        self.assertEqual(load_corrections(path, source=self.source).identity, self.editor.document.identity)
        reopened = CorrectionEditor(self.job, self.root, existing=path)
        self.assertEqual(reopened.document.identity, self.editor.document.identity)
        entry = data['entries'][0]
        self.editor.edit(entry['id'], {'strength': .5, 'range': [0, 2]})
        self.assertTrue(self.editor.document.to_dict()['guidance'])
        with self.assertRaisesRegex(ValueError, 'guidance|binding|generation'):
            self.editor.edit(entry['id'], {'frame': 1})
        self.editor.detach_guidance(entry['id'])
        self.assertFalse(self.editor.document.to_dict()['guidance'])
        self.assertNotIn('guidance_id', self.editor.document.to_dict()['entries'][0])
        self.editor.undo()
        self.editor.remove(entry['id'])
        self.assertEqual(self.editor.document.to_dict()['guidance'], [])
        rasters = CorrectionRasters(self.editor.document, self.editor.meta)
        try:
            alpha = np.full((24, 32), .4, np.float32)
            np.testing.assert_array_equal(rasters.apply(alpha, 0), alpha)
        finally:
            rasters.close()
        for original, content in self.originals.items():
            self.assertEqual(original.read_bytes(), content)

    def test_reference_mutation_prevents_acceptance_and_saved_reuse(self):
        self.prepare()
        self.editor.plate_view()
        before = self.editor.document.identity
        cv2.imwrite(str(self.reference), np.zeros((24, 32, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, 'hash|resource|changed'):
            self.editor.accept_plate()
        self.assertEqual(self.editor.document.identity, before)
        self.reference.write_bytes(self.originals[self.reference])
        self.editor.accept_plate()
        path = self.editor.save()
        cv2.imwrite(str(self.samples), np.zeros((24, 32), np.uint8))
        with self.assertRaisesRegex(ValueError, 'hash|resource|changed'):
            load_corrections(path, source=self.source)
        self.editor.remove(self.editor.document.to_dict()['entries'][0]['id'])
        load_corrections(self.editor.save(), source=self.source)  # Disabled origin has no stale reference dependency.

    def test_unusable_plate_sampling_and_thresholds_do_not_silently_obtain_success(self):
        with self.assertRaisesRegex(ValueError, 'samples|assumption'):
            self.prepare(samples_path=None)
        bad = self.root / 'bad.png'
        cv2.imwrite(str(bad), np.full((24, 32, 3), 255, np.uint8))
        self.prepare(reference_path=bad)
        summary = self.editor.plate_summary()
        assert summary is not None
        self.assertFalse(summary['usable'])
        self.editor.plate_view()
        with self.assertRaisesRegex(ValueError, 'usable'):
            self.editor.accept_plate()
        self.assertEqual(self.editor.document.to_dict()['entries'], [])

    def test_failed_checkpoint_preserves_preview_history_and_retry(self):
        self.prepare()
        self.editor.plate_view()
        before = self.editor.document.identity
        before_assets = set((self.editor.workspace / 'assets').iterdir())
        for _ in range(3):
            with patch('corridorkey.correction_editor.save_corrections', side_effect=OSError('fixture checkpoint full')), \
                    self.assertRaisesRegex(OSError, 'checkpoint full'):
                self.editor.accept_plate()
            self.assertEqual(set((self.editor.workspace / 'assets').iterdir()), before_assets)
        self.assertEqual(self.editor.document.identity, before)
        self.assertIsNotNone(self.editor.plate_preview)
        self.assertFalse(list(self.editor.workspace.glob('plate-preview-*/guidance-*')))
        self.editor.accept_plate()
        self.assertEqual(self.editor.revision, 1)

    def test_staging_cleanup_error_never_masks_failed_or_committed_outcome(self):
        self.prepare()
        self.editor.plate_view()
        unlink = Path.unlink

        def blocked(path, *args, **kwargs):
            if path.name.startswith('guidance-'):
                raise OSError('fixture private staging locked')
            return unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', blocked), \
                patch.object(self.editor, '_import_raster', side_effect=ValueError('fixture import error')), \
                self.assertRaisesRegex(ValueError, 'import error'):
            self.editor.accept_plate()
        self.assertEqual(self.editor.revision, 0)
        self.assertTrue(self.editor.snapshot()['tracking_cleanup_pending'])
        with self.assertRaisesRegex(OSError, 'staging cleanup'):
            self.editor.accept_plate()
        self.editor.discard_plate()
        self.prepare()
        self.editor.plate_view()
        with patch.object(Path, 'unlink', blocked):
            self.editor.accept_plate()
        self.assertEqual(self.editor.revision, 1)
        self.assertTrue(self.editor.document.to_dict()['guidance'])

    def test_reference_transfer_cannot_mutate_retained_thumbnail(self):
        self.prepare()
        transfer = self.editor.plate_reference()
        assert transfer is not None
        original = transfer.pixels.copy()
        transfer.pixels[:] = 0
        displayed, _ = self.editor.plate_view('reference')
        assert displayed is not None
        np.testing.assert_array_equal(displayed.pixels, original)
        displayed.pixels[:] = 0
        fresh = self.editor.plate_reference()
        assert fresh is not None
        np.testing.assert_array_equal(fresh.pixels, original)

    def test_reference_bounds_refuse_before_native_decode(self):
        from corridorkey.video_io import _read_frame_dir_views
        before = self.editor.document.identity
        with self.reference.open('ab') as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
        with patch('corridorkey.video_io._read_frame_dir_views', wraps=_read_frame_dir_views) as decode:
            with self.assertRaisesRegex(ValueError, 'bound|byte|large'):
                self.prepare()
            self.assertEqual(decode.call_count, 1)  # Source only; the oversized reference never enters OpenCV.
        self.assertEqual(self.editor.document.identity, before)
        self.assertIsNone(self.editor.plate_preview)
        self.reference.write_bytes(self.originals[self.reference])
        cv2.imwrite(str(self.reference), np.zeros((25, 32, 3), np.uint8))
        with patch('corridorkey.video_io._read_frame_dir_views', wraps=_read_frame_dir_views) as decode:
            with self.assertRaisesRegex(ValueError, 'geometry'):
                self.prepare()
            self.assertEqual(decode.call_count, 1)

    def test_uncommitted_asset_cleanup_failure_is_owned_bounded_and_retryable(self):
        self.prepare()
        self.editor.plate_view()
        unlink = Path.unlink

        def blocked(path, *args, **kwargs):
            if path.parent == self.editor.workspace / 'assets':
                raise OSError('fixture copied asset locked')
            return unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', blocked), \
                patch('corridorkey.correction_editor.save_corrections', side_effect=OSError('fixture checkpoint full')):
            for _ in range(8):
                with self.assertRaisesRegex(OSError, 'checkpoint full'):
                    self.editor.accept_plate()
            self.assertEqual(len(self.editor.tracking_cleanup_pending), 8)
            with self.assertRaisesRegex(OSError, 'Private asset cleanup'):
                self.editor.accept_plate()
            self.assertEqual(len(list((self.editor.workspace / 'assets').iterdir())), 8)
        self.editor.accept_plate()
        self.assertFalse(self.editor.tracking_cleanup_pending)
        self.assertEqual(len(list((self.editor.workspace / 'assets').iterdir())), 1)
        self.assertEqual(self.editor.revision, 1)


if __name__ == '__main__':
    unittest.main()
