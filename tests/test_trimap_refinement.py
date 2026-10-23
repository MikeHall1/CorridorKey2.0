"""Native class/filter/runtime CPU mechanics; no learned-quality acceptance."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'cv2', 'torch'))


@unittest.skipUnless(STACK, 'requires optional CPU numeric stack')
class TrimapRefinementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def asset(self, pixels, name='trimap.png'):
        import cv2
        path = self.root / name
        self.assertTrue(cv2.imwrite(str(path), pixels))
        return path

    def document(self, path, *, shape, frames=1, source_hash='0' * 64, entries=None):
        from corridorkey.corrections import CorrectionDocument
        entry = {'id': 'trimap', 'revision': 1, 'frame': 0, 'kind': 'trimap',
                 'asset': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                 'strength': 1., 'direction': 'both', 'range': [0, frames]}
        data = {'schema_version': 2, 'revision': 1,
                'source': {'sha256': source_hash, 'start_frame': 0, 'frame_count': frames,
                           'fps': 24., 'width': shape[1], 'height': shape[0]},
                'shots': [[0, frames]], 'entries': entries or [entry]}
        return CorrectionDocument.parse(data, self.root)

    def test_exact_classes_all_precisions_and_original_encoded_bytes(self):
        import numpy as np

        from corridorkey.trimap_refinement import load_trimap_asset
        for dtype, codes, extension in ((np.uint8, [0, 128, 255], 'png'),
                                        (np.uint16, [0, 32768, 65535], 'png'),
                                        (np.float32, [0, .5, 1], 'exr')):
            with self.subTest(dtype=dtype):
                path = self.asset(np.array([codes, codes], dtype=dtype), f'{dtype.__name__}.{extension}')
                raw = path.read_bytes()
                loaded = load_trimap_asset(path, expected_hw=(2, 3), expected_sha256=hashlib.sha256(raw).hexdigest())
                np.testing.assert_array_equal(loaded.labels, [[0, 128, 255]] * 2)
                self.assertFalse(loaded.labels.flags.writeable)
                self.assertEqual(loaded.encoded_bytes, raw)
                self.assertEqual(path.read_bytes(), raw)

    def test_strict_classes_shape_header_and_alias_admission_before_decoder(self):
        import cv2
        import numpy as np

        from corridorkey.trimap_refinement import load_trimap_asset
        invalid = self.asset(np.array([[0, 127, 255]], np.uint8))
        with self.assertRaisesRegex(ValueError, 'exact'):
            load_trimap_asset(invalid, expected_hw=(1, 3))
        colour = self.asset(np.zeros((2, 3, 4), np.uint8), 'rgba.png')
        with patch.object(cv2, 'imdecode', side_effect=AssertionError('decoder called')):
            for path, shape in ((invalid, (500, 500)), (colour, (2, 3))):
                with self.assertRaises(ValueError):
                    load_trimap_asset(path, expected_hw=shape)
        alias = self.root / 'alias.png'
        alias.symlink_to(invalid)
        with self.assertRaises((OSError, ValueError)):
            load_trimap_asset(alias, expected_hw=(1, 3))
        with self.assertRaisesRegex(ValueError, 'hash'):
            load_trimap_asset(invalid, expected_hw=(1, 3), expected_sha256='0' * 64)

    def test_filter_unknown_endpoints_known_constraints_and_strength_analytic(self):
        import math

        import numpy as np

        from corridorkey.trimap_refinement import refine_trimap
        alpha = np.zeros((3, 3), np.float32)
        labels = np.full((3, 3), 255, np.uint8)
        labels[1, 1] = 128
        guide = np.zeros((3, 3, 3), np.float32)
        result = refine_trimap(alpha, labels, guide)
        neighbour_weight = 4 * math.exp(-2) + 4 * math.exp(-4)
        self.assertAlmostEqual(float(result[1, 1]), neighbour_weight / (1 + neighbour_weight), places=6)
        self.assertTrue((result[labels == 255] == 1).all())
        half = refine_trimap(alpha, labels, guide, strength=.5)
        np.testing.assert_allclose(half, result * .5, atol=1e-7)
        np.testing.assert_array_equal(refine_trimap(alpha, labels, guide, strength=0), alpha)
        labels.fill(0)
        np.testing.assert_array_equal(refine_trimap(np.ones_like(alpha), labels, guide), alpha)
        labels.fill(128)
        np.testing.assert_allclose(refine_trimap(np.full_like(alpha, .4), labels, guide), np.full_like(alpha, .4), atol=1e-7)

    def test_order_and_same_path_soft_mask_cache_neural_and_temporal_skip(self):
        import numpy as np

        from corridorkey.correction_rasters import CorrectionRasters
        path = self.asset(np.full((3, 3), 128, np.uint8))
        doc = self.document(path, shape=(3, 3))
        data = doc.to_dict()
        matte = dict(data['entries'][0], id='matte', kind='matte')
        data['entries'].append(matte)
        from corridorkey.corrections import CorrectionDocument
        raster = CorrectionRasters(CorrectionDocument.parse(data, self.root),
                                   {'native_hw': (3, 3), 'num_frames': 1, 'native_fps': 24.})
        self.addCleanup(raster.close)
        self.assertEqual(len(raster._cache), 2)
        self.assertEqual({key[2] for key in raster._cache}, {'matte', 'trimap'})
        result = raster.apply(np.zeros((3, 3), np.float32), 0, guidance=np.zeros((3, 3, 3), np.float32))
        np.testing.assert_array_equal(result, np.full((3, 3), 128 / 255, np.float32))
        only = CorrectionRasters(doc, {'native_hw': (3, 3), 'num_frames': 1, 'native_fps': 24.})
        self.addCleanup(only.close)
        self.assertFalse(only.has_condition(0))
        self.assertFalse(only.has_effect(0, temporal=True))
        self.assertEqual(only.condition(0), (None, None))
        np.testing.assert_array_equal(only.apply(result, 0, temporal=True), result)
        with self.assertRaisesRegex(ValueError, 'guidance'):
            only.apply(result, 0)
        # Later authored constraints may override trimap known foreground.
        white = self.asset(np.full((3, 3), 255, np.uint8), 'white.png')
        ordered = self.document(white, shape=(3, 3)).to_dict()
        ordered['entries'].append(dict(ordered['entries'][0], id='remove', kind='remove'))
        ordered_raster = CorrectionRasters(CorrectionDocument.parse(ordered, self.root),
                                          {'native_hw': (3, 3), 'num_frames': 1, 'native_fps': 24.})
        self.addCleanup(ordered_raster.close)
        np.testing.assert_array_equal(ordered_raster.apply(result, 0, guidance=np.zeros((3, 3, 3), np.float32)), 0)

    def test_cancel_during_bounded_read_and_tiled_filter_preserves_input(self):
        import numpy as np

        from corridorkey.trimap_refinement import load_trimap_asset, refine_trimap
        path = self.asset(np.full((130, 130), 128, np.uint8))
        class Stop(Exception):
            pass
        def stop():
            raise Stop()
        with self.assertRaises(Stop):
            load_trimap_asset(path, expected_hw=(130, 130), check=stop)
        alpha = np.full((130, 130), .4, np.float32)
        before = alpha.copy()
        calls = 0
        def later():
            nonlocal calls
            calls += 1
            if calls == 5:
                raise Stop()
        with self.assertRaises(Stop):
            refine_trimap(alpha, np.full_like(alpha, 128), np.zeros((130, 130, 3), np.float32), check=later)
        np.testing.assert_array_equal(alpha, before)

    def test_actual_native_spatial_low_vram_quick_fallback_hdr_final_once(self):
        import cv2
        import numpy as np

        from corridorkey import draft
        from corridorkey.corrections import save_corrections
        from corridorkey.identities import source_identity
        from corridorkey.motion import Propagation
        from tests.test_spatial_draft import SpatialDraftTests
        helper = SpatialDraftTests()
        helper.root = self.root
        source = helper.source(exr=True)
        labels = np.full((65, 97), 128, np.uint8)
        labels[:, :20] = 0
        labels[:, -20:] = 255
        path = self.asset(labels)
        document = self.document(path, shape=labels.shape, frames=5, source_hash=source_identity(source)['sha256'])
        data = document.to_dict()
        data['entries'][0]['strength'] = .5
        from corridorkey.corrections import CorrectionDocument
        correction = self.root / 'correction.json'
        save_corrections(correction, CorrectionDocument.parse(data, self.root))
        source_before = {p: p.read_bytes() for p in source.iterdir()}
        original = draft.offline_frames
        def uncertain(*args, **kwargs):
            alpha = np.full((65, 97), .4, np.float32)
            return Propagation(alpha, np.zeros_like(alpha), np.ones_like(alpha, dtype=bool),
                               {'flow_evaluations': 4, 'motion_seconds': 0., 'uncertain_fraction': 1., 'motion_fraction': 0.})
        def forced(*args, **kwargs):
            return original(*args, interpolate=uncertain, **kwargs)
        for scale, low, quick, refine in ((1., False, False, False), (1., True, False, True),
                                         (.5, False, False, True), (.5, True, True, True)):
            with self.subTest(scale=scale, low=low, quick=quick):
                with patch.object(draft, 'offline_frames', forced):
                    model, diagnostics, _ = helper.run_job(source, scale=scale, low=low, quick=quick,
                                                          refine=refine, correction=correction)
                assert diagnostics.corrections is not None
                self.assertEqual(diagnostics.corrections['output_frames_refined'], 5)
                for output in Path(diagnostics.outputs['alpha']).glob('*.exr'):
                    alpha = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
                    np.testing.assert_allclose(alpha[:, :19], .2, atol=2e-6)
                    np.testing.assert_allclose(alpha[:, -19:], .7, atol=2e-6)
                    np.testing.assert_allclose(alpha[:, 25:70], .4, atol=2e-6)
                self.assertTrue(all(float(hint.abs().max()) == 0 for hint in model.stage1.seen_hints[:1]))
                if quick:
                    assert diagnostics.quick_key is not None
                    self.assertGreater(len(diagnostics.quick_key['extra_neural_frames']), 0)
        self.assertEqual({p: p.read_bytes() for p in source.iterdir()}, source_before)
        self.assertEqual(json.loads(correction.read_text())['schema_version'], 2)
