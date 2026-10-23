"""Decoded video geometry admission before any intentional draft resampling."""
import importlib.util
import unittest
from unittest.mock import patch

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2'))


@unittest.skipUnless(STACK, 'requires optional CPU numeric stack')
class VideoGeometryTests(unittest.TestCase):
    def test_changed_native_geometry_rejects_before_full_or_draft_resize(self):
        import numpy as np

        from corridorkey import video_io

        for draft in (False, True):
            meta = {'native_hw': (8, 8), 'model_native_hw': (4, 4) if draft else (8, 8),
                    'shape_hw': (128, 128), 'spatial_draft': {'enabled': True} if draft else None}
            for shape in ((12, 8, 3), (8, 12, 3), (8, 8, 4), (8, 8)):
                for need_model in (False, True):
                    with self.subTest(draft=draft, shape=shape, need_model=need_model), \
                            patch.object(video_io, 'resize_hwc_lanczos') as resize:
                        with self.assertRaisesRegex(ValueError, 'geometry'):
                            video_io.preprocess_video_frame_views(np.zeros(shape, dtype=np.uint8),
                                                                  meta, need_model=need_model)
                        resize.assert_not_called()

    def test_valid_native_geometry_keeps_existing_full_and_explicit_draft_paths(self):
        import numpy as np
        import torch

        from corridorkey import video_io

        source = np.full((8, 8, 3), 128, dtype=np.uint8)
        for draft in (False, True):
            meta = {'native_hw': (8, 8), 'model_native_hw': (4, 4) if draft else (8, 8),
                    'shape_hw': (128, 128), 'spatial_draft': {'enabled': True} if draft else None}
            for need_model in (False, True):
                with self.subTest(draft=draft, need_model=need_model):
                    model, foundation, review = video_io.preprocess_video_frame_views(
                        source, meta, need_model=need_model)
                    self.assertEqual(tuple(foundation.shape), (3, 128, 128))
                    self.assertEqual(tuple(review.shape), (3, 128, 128))
                    self.assertEqual(foundation.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(foundation).all())
                    if need_model:
                        assert model is not None
                        self.assertEqual(tuple(model.shape), (3, 128, 128))
                    else:
                        self.assertIsNone(model)
