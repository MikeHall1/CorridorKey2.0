import importlib.util
import unittest
from typing import TYPE_CHECKING, TypedDict

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation_colour import colour_metrics

class Contract(TypedDict):
    colour_space: str
    candidate_association: str
    reference_association: str


CONTRACT: Contract = {'colour_space': 'linear_srgb_bt709', 'candidate_association': 'straight',
            'reference_association': 'straight'}


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class ColourMetricTests(unittest.TestCase):
    def test_known_foreground_and_linear_composite_error(self):
        report = colour_metrics([[[1., 0., 0.]]], [[0.5]], [[[0., 1., 0.]]], [[1.]],
                                foreground_mask=[[True]], background=[0., 0., 0.], **CONTRACT)
        self.assertAlmostEqual(report['foreground_mae']['value'], 2 / 3)
        self.assertEqual(report['foreground_mae']['population'], 3)
        self.assertEqual(report['composite_mae']['value'], 0.5)
        self.assertEqual(report['alpha']['mae']['value'], 0.5)

    def test_association_equivalence_retains_hdr_and_negative_values(self):
        rgb = np.array([[[-0.5, 2., 8.], [4., -1., 0.3]]])
        alpha = np.array([[0.25, 0.5]])
        report = colour_metrics(rgb * alpha[..., None], alpha, rgb, alpha,
                                colour_space='linear_srgb_bt709', candidate_association='premultiplied',
                                reference_association='straight', foreground_mask=np.ones((1, 2), dtype=bool),
                                background=[-1., 3., 0.5])
        self.assertEqual(report['foreground_mae']['value'], 0)
        self.assertEqual(report['composite_mae']['value'], 0)
        self.assertEqual(report['candidate_rgb_range'], [-0.5, 2.])

    def test_missing_reference_mask_or_background_is_unavailable(self):
        report = colour_metrics([[[1., 1., 1.]]], [[1.]], **CONTRACT)
        self.assertIsNone(report['foreground_mae']['value'])
        self.assertIsNone(report['composite_mae']['value'])
        report = colour_metrics([[[1., 1., 1.]]], [[1.]], [[[0., 0., 0.]]], [[1.]], **CONTRACT)
        self.assertEqual(report['foreground_reason'], 'valid-reference foreground mask missing')
        self.assertIsNone(report['composite_mae']['value'])

    def test_reference_mask_and_zero_alpha_policy_are_counted(self):
        report = colour_metrics(np.zeros((1, 2, 3)), [[0., 0.]], np.ones((1, 2, 3)), [[1., 0.]],
                                colour_space='linear_srgb_bt709', candidate_association='premultiplied',
                                reference_association='straight', foreground_mask=[[True, True]], background=[0., 0., 0.])
        self.assertEqual(report['foreground_mae']['value'], 1)
        self.assertEqual(report['foreground_pixels'], 1)
        self.assertEqual(report['reference_excluded_foreground_pixels'], 1)
        self.assertEqual(report['candidate_undefined_foreground_pixels'], 1)
        self.assertEqual(report['composite_mae']['value'], 0.5)

    def test_rejects_invalid_domains_geometry_association_and_overflow(self):
        invalid_space: Contract = {**CONTRACT, 'colour_space': 'srgb'}
        invalid_association: Contract = {**CONTRACT, 'candidate_association': 'unknown'}
        for contract in (invalid_space, invalid_association):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                colour_metrics([[[0., 0., 0.]]], [[1.]], **contract)
        premultiplied: Contract = {**CONTRACT, 'candidate_association': 'premultiplied'}
        with self.assertRaisesRegex(ValueError, 'zero at zero alpha'):
            colour_metrics([[[1., 0., 0.]]], [[0.]], **premultiplied)
        with self.assertRaisesRegex(ValueError, 'finite'):
            colour_metrics([[[float('nan'), 0., 0.]]], [[1.]], **CONTRACT)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            colour_metrics(np.zeros((2, 2, 3)), [[1.]], **CONTRACT)
        with self.assertRaisesRegex(ValueError, 'arithmetic'):
            colour_metrics([[[1e308, 1e308, 1e308]]], [[1.]], [[[-1e308, 0., 0.]]], [[1.]],
                           foreground_mask=[[True]], **CONTRACT)

    def test_canonical_single_frame_batch_masks_match_unbatched_results(self):
        rgb = np.ones((1, 1, 1, 3))
        report = colour_metrics(rgb, np.ones((1, 1, 1)), np.zeros_like(rgb), np.ones((1, 1, 1)),
                                foreground_mask=[[True]], composite_mask=[[True]], background=[0., 0., 0.], **CONTRACT)
        self.assertEqual(report['foreground_mae']['value'], 1)
        self.assertEqual(report['composite_mae']['value'], 1)
        self.assertEqual(report['alpha']['mae']['value'], 0)

    def test_masked_out_extreme_differences_do_not_participate_in_arithmetic(self):
        candidate = np.array([[[0., 0., 0.], [1e308, 1e308, 1e308]]])
        target = -candidate
        report = colour_metrics(candidate, [[1., 1.]], target, [[1., 1.]], foreground_mask=[[True, False]],
                                composite_mask=[[True, False]], background=[0., 0., 0.], **CONTRACT)
        self.assertEqual(report['foreground_mae']['value'], 0)
        self.assertEqual(report['composite_mae']['value'], 0)
        with self.assertRaisesRegex(ValueError, 'epsilon'):
            colour_metrics(np.zeros((1, 1, 3)), [[1.]], unpremultiply_epsilon=10**1000, **CONTRACT)
