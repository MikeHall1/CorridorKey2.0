import importlib.util
import unittest
from typing import TYPE_CHECKING

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation_temporal import FLOW_CONVENTION, temporal_pair


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class TemporalTests(unittest.TestCase):
    def pair(self, before, now, ref_before=None, ref_now=None, **kwargs):
        return temporal_pair(before, now, ref_before, ref_now, flow_convention=FLOW_CONVENTION, **kwargs)

    def test_translating_alpha_uses_current_to_previous_pixel_flow(self):
        before, now = np.array([[0., 1., 0.]]), np.array([[0., 0., 1.]])
        backward = np.zeros((1, 3, 2))
        backward[..., 0] = -1
        forward = -backward
        report = self.pair(before, now, before, now, backward_flow=backward, forward_flow=forward,
                           visibility=np.ones((1, 3), dtype=bool), consistency_pixels=0)
        self.assertEqual(report['valid_pixels'], 2)
        self.assertEqual(report['out_of_bounds_pixels'], 1)
        self.assertEqual(report['temporal_residual_mae']['value'], 0)
        self.assertEqual(report['candidate_change_mae']['value'], 0)
        self.assertEqual(report['spatial_current']['mae']['value'], 0)

    def test_frozen_blank_matte_requires_spatial_accuracy_even_with_zero_temporal_residual(self):
        before, now = np.array([[0., 1., 0.]]), np.array([[0., 0., 1.]])
        flow = np.zeros((1, 3, 2))
        flow[..., 0] = -1
        report = self.pair(np.zeros_like(before), np.zeros_like(now), before, now, backward_flow=flow,
                           visibility=np.ones((1, 3), dtype=bool))
        self.assertEqual(report['temporal_residual_mae']['value'], 0)
        self.assertAlmostEqual(report['spatial_current']['mae']['value'], 1 / 3)

    def test_fractional_flow_uses_exact_bilinear_weights(self):
        report = self.pair([[0., 1.]], [[0.5, 1.]], [[0., 1.]], [[0.5, 1.]],
                           backward_flow=np.array([[[0.5, 0.], [0., 0.]]]), visibility=[[True, True]])
        self.assertEqual(report['candidate_change_mae']['value'], 0)
        self.assertEqual(report['temporal_residual_mae']['value'], 0)

    def test_visibility_previous_validity_and_consistency_exclude_correspondences(self):
        flow = np.zeros((1, 3, 2))
        forward = flow.copy()
        forward[0, 2, 0] = 2
        report = self.pair([[0., 0., 0.]], [[1., 1., 1.]], [[0., 0., 0.]], [[0., 0., 0.]],
                           backward_flow=flow, forward_flow=forward, visibility=[[False, True, True]],
                           previous_mask=[[True, False, True]], consistency_pixels=0.5)
        self.assertEqual(report['valid_pixels'], 0)
        self.assertIsNone(report['temporal_residual_mae']['value'])
        self.assertEqual(report['visibility_excluded_pixels'], 1)
        self.assertEqual(report['consistency_excluded_pixels'], 1)

    def test_cut_and_missing_prerequisites_are_unavailable_not_zero(self):
        for kwargs, reason in (({'cut': True}, 'shot cut'), ({}, 'backward flow missing'),
                               ({'backward_flow': np.zeros((1, 1, 2))}, 'visibility mask missing')):
            report = self.pair([[0.]], [[1.]], [[0.]], [[0.]], **kwargs)
            self.assertIsNone(report['temporal_residual_mae']['value'])
            self.assertIn(reason, report['reason'])
            self.assertEqual(report['spatial_current']['mae']['value'], 1)
        report = self.pair([[0.]], [[1.]], backward_flow=np.zeros((1, 1, 2)), visibility=[[True]])
        self.assertEqual(report['candidate_change_mae']['value'], 1)
        self.assertIsNone(report['temporal_residual_mae']['value'])

    def test_invalid_flow_masks_reference_pairs_and_conventions_fail(self):
        for kwargs in ({'backward_flow': np.zeros((2, 2, 2))}, {'visibility': [[1]], 'backward_flow': np.zeros((1, 1, 2))},
                       {'backward_flow': np.full((1, 1, 2), np.nan)}, {'forward_flow': np.zeros((1, 1, 2))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.pair([[0.]], [[0.]], [[0.]], [[0.]], **kwargs)
        with self.assertRaisesRegex(ValueError, 'both reference'):
            self.pair([[0.]], [[0.]], [[0.]], None)
        with self.assertRaisesRegex(ValueError, 'convention'):
            temporal_pair([[0.]], [[0.]], flow_convention='previous_to_current')

    def test_consistency_diagnostics_distinguish_provided_and_performed(self):
        flow = np.zeros((1, 1, 2))
        report = self.pair([[0.]], [[0.]], backward_flow=flow, forward_flow=flow)
        self.assertTrue(report['forward_flow_provided'])
        self.assertFalse(report['consistency_check'])
        report = self.pair([[0.]], [[0.]], [[0.]], [[0.]], backward_flow=flow,
                           forward_flow=flow, visibility=[[True]])
        self.assertTrue(report['consistency_check'])
