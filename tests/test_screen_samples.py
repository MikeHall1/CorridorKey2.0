"""Synthetic spatial screen samples are explicit heuristics, not neural quality."""
import hashlib
import unittest

import numpy as np

from corridorkey.clean_plate import (
    INPUT_CONTRACT,
    CleanPlateProvenance,
    CleanPlateSettings,
    PlateMapping,
    assist_clean_plate,
)


def provenance():
    digest = hashlib.sha256(b'local fixture').hexdigest()
    return CleanPlateProvenance(digest, digest, digest, digest, 7, 0,
                                PlateMapping('single', 7, 1, 0, 1), INPUT_CONTRACT)


class ScreenSampleCoreTests(unittest.TestCase):
    def field(self, source, points, **kwargs):
        from corridorkey.clean_plate import screen_sample_reference
        return screen_sample_reference(source, points, **kwargs)

    def test_native_spatial_samples_preserve_signed_hdr_and_source(self):
        source = np.empty((24, 32, 3), np.float32)
        source[:, :16] = [-.1, 1.2, .03]
        source[:, 16:] = [.02, 2.4, .08]
        original = source.tobytes()
        field, samples, support, record = self.field(source, [[4, 12], [27, 12]], sample_radius=3, influence_radius=8)
        self.assertEqual(field.dtype, np.float32)
        np.testing.assert_allclose(field[12, 4], source[12, 4], atol=.001)
        self.assertLess(float(field[12, 4, 0]), 0)
        self.assertGreater(float(field[12, 27, 1]), 2.)
        self.assertFalse(support[0, 15])
        self.assertTrue(support[12, 4])
        self.assertEqual(int(samples.sum()), 98)
        self.assertEqual(record['points'], [[4, 12], [27, 12]])
        self.assertEqual(record['sample_colours_linear_bt709'][0], [float(x) for x in source[12, 4]])
        self.assertEqual(source.tobytes(), original)
        reversed_field, _, _, _ = self.field(source, [[27, 12], [4, 12]], sample_radius=3, influence_radius=8)
        np.testing.assert_allclose(reversed_field, field, rtol=1e-6, atol=1e-6)

    def test_bounded_tiled_generation_cancellation_and_invalid_inputs(self):
        source = np.full((24, 32, 3), [.03, .6, .05], np.float32)
        for points, extra in [([], {}), ([[1, 1]] * 2, {}), ([[True, 1]], {}), ([[32, 1]], {}),
                              ([[1.5, 1]], {}), ([[1, 1]], {'sample_radius': 0}),
                              ([[1, 1]], {'influence_radius': float('nan')}),
                              ([[1, 1]], {'max_pixels': 100}),
                              ([[x % 32, x // 32] for x in range(33)], {})]:
            with self.subTest(points=points, extra=extra), self.assertRaises((ValueError, TypeError)):
                self.field(source, points, **extra)
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            if checks == 3:
                raise RuntimeError('cancel fixture')
        with self.assertRaisesRegex(RuntimeError, 'cancel fixture'):
            self.field(source, [[4, 12], [27, 12]], check=cancel)
        self.assertEqual(checks, 3)
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.field(np.full_like(source, np.inf), [[2, 2]])

    def test_guidance_uses_existing_seed_contract_and_unknown_outside_reach(self):
        from corridorkey.clean_plate import restrict_screen_candidate
        source = np.full((24, 32, 3), [.04, .6, .05], np.float32)
        source[8:16, 13:20] = [.8, .03, .02]
        field, samples, support, record = self.field(source, [[5, 12], [26, 12]], sample_radius=3, influence_radius=9)
        candidate = assist_clean_plate(source, field, samples, settings=CleanPlateSettings('declared', screen_radius=0),
                                       provenance=provenance())
        restricted = restrict_screen_candidate(candidate, support, record)
        self.assertTrue(restricted.usable)
        self.assertEqual(int(restricted.labels()[12, 5]), 0)
        self.assertTrue(np.all(restricted.labels()[~support] == 128))
        self.assertTrue(np.all(restricted.validity()[~support] == 0))
        self.assertNotEqual(candidate.identity, restricted.identity)
        self.assertIn('synthetic', restricted.report()['diagnostics']['screen_field'])
        self.assertIn('not an empty plate', ' '.join(restricted.report()['diagnostics']['warnings']))
        self.assertEqual(restricted.report()['provenance']['screen_sampling'], record)
        self.assertEqual(candidate.trimap_bytes, candidate.labels().tobytes())
        self.assertFalse(restrict_screen_candidate(candidate, np.zeros_like(support), record).usable)
        with self.assertRaises(TypeError):
            restrict_screen_candidate(type(candidate), support, record)
        with self.assertRaises(ValueError):
            restrict_screen_candidate(candidate, support.astype(np.float32), record)


class ScreenSampleEditorTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        import cv2

        from corridorkey.config import InferenceConfig
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.session import JobRequest
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        image = np.full((24, 32, 3), [20, 150, 50], np.uint8)
        image[8:16, 13:20] = [20, 20, 220]
        for frame in range(3):
            self.assertTrue(cv2.imwrite(str(self.source / f'{frame}.png'), image))
        self.before = {path: path.read_bytes() for path in self.source.iterdir()}
        self.job = JobRequest.create(self.source, self.root / 'out', InferenceConfig(output_format='exr', start_frame=1))
        self.editor = CorrectionEditor(self.job, self.root)

    def prepare(self, **changes):
        self.editor.prepare_plate(**({'frame': 0, 'screen_points': [[5, 12], [26, 12]],
                                      'sample_radius': 3, 'influence_radius': 9} | changes))

    def test_inspect_accept_effect_provenance_save_undo_and_reopen(self):
        from pathlib import Path

        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.correction_rasters import CorrectionRasters
        from corridorkey.corrections import load_corrections
        original = self.editor.document.identity
        self.prepare(guidance_strength=.5)
        summary = self.editor.plate_summary()
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertTrue(summary['usable'], summary)
        self.assertEqual(summary['source_frame'], 1)
        self.assertEqual(summary['request']['screen_sampling']['points'], [[5, 12], [26, 12]])
        self.assertEqual(self.editor.document.identity, original)
        with self.assertRaisesRegex(ValueError, 'Inspect'):
            self.editor.accept_plate()
        self.editor.plate_view('screen_field')
        with self.assertRaisesRegex(ValueError, 'Inspect'):
            self.editor.accept_plate()
        self.editor.plate_view('guidance')
        preview = self.editor.plate_preview
        assert preview is not None
        preview_dir = preview['directory']
        self.editor.accept_plate()
        data = self.editor.document.to_dict()
        self.assertEqual(data['entries'][0]['strength'], .5)
        origin = data['guidance'][0]
        self.assertEqual(origin['kind'], 'clean_plate')
        self.assertEqual(origin['parameters']['settings']['screen_sampling']['kernel_version'], 'screen-colour-idw-v1')
        retained = Path(origin['resources'][0]['asset']['path'])
        self.assertEqual(retained.parent, self.editor.workspace / 'assets')
        self.assertTrue(retained.is_file())
        from corridorkey.video_io import _read_frame_dir_views
        _, decoded_reference = _read_frame_dir_views(str(retained), exr_linear=True, alpha_mode='ignore',
                                                      source_colour_profile='linear_bt709_d65')
        self.assertEqual(hashlib.sha256(decoded_reference.astype('<f4').tobytes()).hexdigest(),
                         summary['report']['provenance']['input_hashes']['reference_rgb_f32le'])
        self.assertFalse(preview_dir.exists())
        self.assertIn('synthetic', origin['observation']['diagnostics']['screen_field'])
        rasters = CorrectionRasters(self.editor.document, self.editor.meta)
        try:
            alpha = np.full((24, 32), .6, np.float32)
            result = rasters.apply(alpha, 0, guidance=np.full((24, 32, 3), .1, np.float32))
            self.assertAlmostEqual(float(result[12, 5]), .3, places=6)
        finally:
            rasters.close()
        saved = self.editor.save()
        self.assertEqual(load_corrections(saved, source=self.source).identity, self.editor.document.identity)
        reopened = CorrectionEditor(self.job, self.root, existing=saved)
        self.assertEqual(reopened.document.identity, self.editor.document.identity)
        self.editor.undo()
        self.assertEqual(self.editor.document.identity, original)
        self.assertTrue(retained.is_file())
        self.editor.redo()
        self.assertEqual(load_corrections(self.editor.save()).identity, reopened.document.identity)
        self.assertEqual({p: p.read_bytes() for p in self.before}, self.before)

    def test_schema5_region_and_synthetic_guidance_save_and_relocate_without_loss(self):
        from corridorkey.corrections import load_corrections, save_corrections
        mask = np.zeros((24, 32), np.uint8)
        mask[:, :12] = 255
        self.editor.commit_brush(mask, kind='region', frame=0, strength=.75, direction='both', span=[0, 2])
        region = self.editor.document.to_dict()['entries'][0]
        self.assertEqual(self.editor.document.to_dict()['schema_version'], 5)
        self.prepare()
        self.editor.plate_view('guidance')
        self.editor.accept_plate()
        data = self.editor.document.to_dict()
        self.assertEqual(data['schema_version'], 5)
        self.assertEqual(data['entries'][0], region)
        self.assertEqual(len(data['guidance']), 1)
        self.assertEqual(len(self.editor.document.asset_paths()), 3)
        relocated = self.root / 'relocated'
        relocated.mkdir()
        path = relocated / 'corrections.json'
        save_corrections(path, self.editor.document)
        restored = load_corrections(path, source=self.source)
        self.assertEqual(restored.identity, self.editor.document.identity)
        self.assertEqual(restored.to_dict()['entries'][0]['purpose'], 'refinement')
        self.editor.undo()
        self.assertEqual(self.editor.document.to_dict()['entries'], [region])
        self.editor.redo()
        self.assertEqual(self.editor.document.identity, restored.identity)

    def test_invalid_mode_strength_and_insufficient_samples_never_publish(self):
        for changes in ({'reference_path': self.source}, {'whole_frame_samples': True}, {'alignment': 'estimate_translation'},
                        {'compensate_exposure': True}, {'guidance_strength': True}, {'guidance_strength': -1},
                        {'guidance_strength': float('inf')}, {'screen_points': []}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                self.prepare(**changes)
        self.prepare(screen_points=[[1, 1]], sample_radius=1)
        summary = self.editor.plate_summary()
        assert summary is not None
        self.assertFalse(summary['usable'])
        self.editor.plate_view('guidance')
        with self.assertRaisesRegex(ValueError, 'usable'):
            self.editor.accept_plate()
        self.assertEqual(self.editor.document.to_dict()['entries'], [])

    def test_source_or_generated_reference_drift_refuses_acceptance(self):
        self.prepare()
        self.editor.plate_view('guidance')
        preview = self.editor.plate_preview
        assert preview is not None
        reference = preview['resources'][0]['asset']['path']
        from pathlib import Path
        Path(reference).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.editor.accept_plate()
        self.assertEqual(self.editor.document.to_dict()['entries'], [])
        self.editor.discard_plate()
        self.prepare()
        self.editor.plate_view('guidance')
        next(iter(self.before)).write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.editor.accept_plate()
        self.assertEqual(self.editor.document.to_dict()['entries'], [])

    def test_failed_import_cleans_generated_copy_and_preserves_preview_for_retry(self):
        from unittest.mock import patch
        self.prepare()
        self.editor.plate_view('guidance')
        assets = self.editor.workspace / 'assets'
        before = set(assets.iterdir())
        for _ in range(3):
            with patch.object(self.editor, '_import_raster', side_effect=OSError('publication fixture')), \
                    self.assertRaisesRegex(OSError, 'publication fixture'):
                self.editor.accept_plate()
            self.assertEqual(set(assets.iterdir()), before)
            self.assertIsNotNone(self.editor.plate_preview)
        self.editor.accept_plate()
        self.assertEqual(self.editor.revision, 1)

    def test_checkpoint_failure_and_locked_cleanup_are_bounded_and_retryable(self):
        from pathlib import Path
        from unittest.mock import patch
        self.prepare()
        self.editor.plate_view('guidance')
        before = set((self.editor.workspace / 'assets').iterdir())
        with patch('corridorkey.correction_editor.save_corrections', side_effect=OSError('checkpoint fixture')), \
                self.assertRaisesRegex(OSError, 'checkpoint fixture'):
            self.editor.accept_plate()
        self.assertEqual(set((self.editor.workspace / 'assets').iterdir()), before)
        unlink = Path.unlink
        def locked(path, *args, **kwargs):
            if path.parent == self.editor.workspace / 'assets' and path.suffix == '.exr':
                raise OSError('generated resource locked')
            return unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', locked), \
                patch.object(self.editor, '_import_raster', side_effect=OSError('import fixture')):
            for _ in range(6):
                with self.assertRaisesRegex(OSError, 'import fixture'):
                    self.editor.accept_plate()
            self.assertEqual(len(self.editor.tracking_cleanup_pending), 6)
            with self.assertRaisesRegex(OSError, 'cleanup is blocked'):
                self.editor.accept_plate()
            self.assertEqual(len(self.editor.tracking_cleanup_pending), 6)
        self.editor.accept_plate()
        self.assertFalse(self.editor.tracking_cleanup_pending)
        self.assertEqual(len(list((self.editor.workspace / 'assets').iterdir())), 2)

    def test_zero_strength_is_persisted_and_has_no_alpha_effect(self):
        from corridorkey.correction_rasters import CorrectionRasters
        self.prepare(guidance_strength=0.)
        self.editor.plate_view('guidance')
        self.editor.accept_plate()
        self.assertEqual(self.editor.document.to_dict()['entries'][0]['strength'], 0.)
        rasters = CorrectionRasters(self.editor.document, self.editor.meta)
        try:
            alpha = np.full((24, 32), .6, np.float32)
            np.testing.assert_array_equal(rasters.apply(alpha, 0), alpha)
        finally:
            rasters.close()


if __name__ == '__main__':
    unittest.main()
