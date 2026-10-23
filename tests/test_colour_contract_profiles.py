"""Declared primary conversion and real EXR decode, without model inference."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.colour_contract import describe_source_colour

STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'cv2'))


class ColourDescriptionTests(unittest.TestCase):
    def test_description_imports_without_numeric_or_model_libraries(self):
        script = '''import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('numpy', 'torch', 'cv2', 'safetensors'):
   raise AssertionError('unexpected numeric/model import '+fullname)
sys.meta_path.insert(0, Block())
from corridorkey.colour_contract import describe_source_colour
print(json.dumps(describe_source_colour('linear_bt2020_d65', scene_linear=True),allow_nan=False))
'''
        result = subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        self.assertEqual(data['working_basis'], 'engine_assumption_training_primaries_unverified')
        self.assertEqual(data['alpha'], 'coverage_not_transformed')
        self.assertTrue(data['conversion_applied'])

    def test_invalid_profiles_or_display_transfer_cannot_be_relabelled(self):
        for profile in ('acescg', 'pq', '', None, True):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                describe_source_colour(profile, scene_linear=True)
        for profile in ('linear_bt709_d65', 'linear_bt2020_d65'):
            with self.assertRaises(ValueError):
                describe_source_colour(profile, scene_linear=False)
        for linear in (False, True):
            data = describe_source_colour('legacy', scene_linear=linear)
            self.assertFalse(data['conversion_applied'])
            self.assertEqual(data['source_primaries'], 'assumed_bt709_d65')
            json.dumps(data, allow_nan=False)


@unittest.skipUnless(STACK, 'requires optional CPU numeric/EXR stack')
class ColourProfileDecodeTests(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def reference_matrix():
        """Independently derive from published xy, white-normalized RGB->XYZ."""
        import numpy as np

        def xyz(primaries):
            xy = np.asarray(primaries, dtype=np.float64)
            unscaled = np.stack((xy[:, 0] / xy[:, 1], np.ones(3),
                                 (1 - xy[:, 0] - xy[:, 1]) / xy[:, 1]))
            white = np.array([.3127 / .3290, 1, (1 - .3127 - .3290) / .3290])
            return unscaled * np.linalg.solve(unscaled, white)
        return np.linalg.solve(xyz(((.640, .330), (.300, .600), (.150, .060))),
                               xyz(((.708, .292), (.170, .797), (.131, .046))))

    def exr(self, rgb, *, alpha=None):
        import cv2
        import numpy as np
        data = rgb[..., ::-1]
        if alpha is not None:
            data = np.concatenate((data, alpha), axis=-1)
        path = self.root / 'frame_000000.exr'
        self.assertTrue(cv2.imwrite(str(path), data, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]))
        return path

    def meta(self, profile='legacy', *, alpha_mode='ignore'):
        from corridorkey.video_io import probe_frame_dir_meta
        return probe_frame_dir_meta(str(self.root), num_frames=-1, shape_quantum=1, fps=24,
                                    exr_linear=True, model_scale=1, alpha_mode=alpha_mode,
                                    source_colour_profile=profile)

    def test_analytical_primaries_neutral_and_signed_hdr_no_clipping(self):
        import numpy as np

        from corridorkey.colour_contract import to_working_rgb
        rgb = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-.25, 2, 4]]], np.float32)
        before = rgb.tobytes()
        actual = to_working_rgb(rgb, 'linear_bt2020_d65')
        expected = rgb.astype(np.float64) @ self.reference_matrix().T
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=5e-7)
        np.testing.assert_allclose(actual[0, 3], [1, 1, 1], atol=2e-7)
        self.assertLess(float(actual.min()), 0)
        self.assertGreater(float(actual.max()), 1)
        self.assertEqual(rgb.tobytes(), before)
        self.assertFalse(np.shares_memory(rgb, actual))
        self.assertEqual(actual.dtype, np.float32)

    def test_identity_profiles_preserve_same_array_and_signed_zero_bytes(self):
        import numpy as np

        from corridorkey.colour_contract import to_working_rgb
        rgb = np.array([[[-0., 0., -2], [4, .25, -0.]]], np.float32)
        before = rgb.tobytes()
        for profile in ('legacy', 'linear_bt709_d65'):
            self.assertIs(to_working_rgb(rgb, profile), rgb)
            self.assertEqual(rgb.tobytes(), before)
        self.exr(rgb)
        from corridorkey.video_io import _read_frame_dir_views
        with patch('corridorkey.colour_contract.to_working_rgb', side_effect=AssertionError('legacy must bypass matrix')):
            legacy_display, legacy_linear = _read_frame_dir_views(str(self.root / 'frame_000000.exr'), exr_linear=True)
        explicit_display, explicit_linear = _read_frame_dir_views(str(self.root / 'frame_000000.exr'), exr_linear=True,
                                                                source_colour_profile='linear_bt709_d65')
        self.assertEqual(legacy_linear.tobytes(), explicit_linear.tobytes())
        self.assertEqual(legacy_display.tobytes(), explicit_display.tobytes())

    def test_real_premultiplied_exr_unpremultiplies_before_primary_conversion_and_hints_unchanged(self):
        import numpy as np

        from corridorkey.video_io import _read_frame_dir_views, read_hint_image
        straight = np.array([[[-.25, 2, .125], [.2, .6, .1], [4, -.1, .5]]], np.float32)
        alpha = np.array([[[.25], [0], [1]]], np.float32)
        path = self.exr(straight * alpha, alpha=alpha)
        before = path.read_bytes()
        _display, actual = _read_frame_dir_views(str(path), exr_linear=True, alpha_mode='premultiplied',
                                                 source_colour_profile='linear_bt2020_d65')
        expected = straight.astype(np.float64) @ self.reference_matrix().T
        expected[0, 1] = 0
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=5e-7)
        np.testing.assert_array_equal(read_hint_image(str(path)), alpha[..., 0])
        self.assertEqual(path.read_bytes(), before)

    def test_every_sequence_reader_and_current_rgb_uses_same_working_basis(self):
        import numpy as np
        import torch

        from corridorkey import color_utils as cu
        from corridorkey.video_io import (
            DraftSourceReader,
            SequentialFrameDirWindowReader,
            iter_frame_dir_chunks,
            preprocess_frame_dir_path,
        )
        raw = np.array([[[.2, .6, .1], [-.1, 2, .25]], [[.5, .1, .3], [1, 1, 1]]], np.float32)
        self.exr(raw)
        meta = self.meta('linear_bt2020_d65')
        self.assertEqual(meta['source_colour_profile'], 'linear_bt2020_d65')
        self.assertEqual(meta['source_colour_contract'], describe_source_colour('linear_bt2020_d65', scene_linear=True))
        reader = DraftSourceReader(str(self.root), meta, is_dir=True)
        try:
            frame = reader.read(0)
        finally:
            reader.close()
        expected = (raw.astype(np.float64) @ self.reference_matrix().T).astype(np.float32)
        np.testing.assert_allclose(frame.rgb, expected, rtol=3e-7, atol=5e-7)
        np.testing.assert_allclose(frame.proxy, cu.linear_to_vfm_sdr(expected), rtol=2e-6, atol=2e-6)
        model, foundation, review = preprocess_frame_dir_path(meta['frame_paths'][0], meta, need_model=True)
        window = SequentialFrameDirWindowReader(str(self.root), meta)
        try:
            window_model, window_foundation, window_review = window.window(0, 1)
        finally:
            window.close()
        chunks = list(iter_frame_dir_chunks(str(self.root), meta, chunk_size=1, need_model=True))
        assert model is not None
        assert window_review is not None
        assert chunks[0][1] is not None
        self.assertTrue(torch.equal(window_model[0], model))
        self.assertTrue(torch.equal(chunks[0][1][0], model))
        self.assertTrue(torch.equal(window_foundation[0], foundation))
        self.assertTrue(torch.equal(window_review[0], review))
        iid_only = list(iter_frame_dir_chunks(str(self.root), meta, chunk_size=1, need_model=False))
        self.assertIsNone(iid_only[0][1])
        self.assertTrue(torch.equal(iid_only[0][2][0], foundation))
        expected_proxy = torch.from_numpy(frame.proxy.transpose(2, 0, 1)).to(torch.bfloat16)
        self.assertTrue(torch.equal(foundation, expected_proxy))

    def test_invalid_layout_nonfinite_overflow_and_witness_refuse(self):
        import numpy as np

        from corridorkey.colour_contract import to_working_rgb
        from corridorkey.video_io import _read_frame_dir_views, probe_frame_dir_meta
        for value in (np.zeros((1, 2, 4), np.float32), np.zeros((1, 2, 3), np.uint16),
                      np.zeros((0, 2, 3), np.float32), np.full((1, 2, 3), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                to_working_rgb(value, 'linear_bt2020_d65')
        with self.assertRaisesRegex(ValueError, 'overflow'):
            to_working_rgb(np.array([[[np.finfo(np.float32).max, -np.finfo(np.float32).max, 0]]], np.float32),
                           'linear_bt2020_d65')
        for profile in ('linear_bt709_d65', 'linear_bt2020_d65'):
            with self.assertRaisesRegex(ValueError, 'witness'):
                _read_frame_dir_views('does-not-exist', exr_linear=True, witness=object(), source_colour_profile=profile)
            with self.assertRaisesRegex(ValueError, 'scene-linear'):
                probe_frame_dir_meta('does-not-exist', num_frames=-1, shape_quantum=1, fps=24,
                                     exr_linear=False, model_scale=1, source_colour_profile=profile)


if __name__ == '__main__':
    unittest.main()
