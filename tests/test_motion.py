import importlib.util
import unittest

HAS_STACK = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional NumPy/OpenCV CPU stack")
class MotionTests(unittest.TestCase):
    def test_causal_warp_has_two_flow_evaluations_and_native_displacement(self):
        from corridorkey.motion import propagate_alpha
        calls = []
        def flow(a, b):
            out = self.zero(a, b)
            out[..., 0] = -2 if not calls else 2
            calls.append(a.shape)
            return out
        alpha = self.alpha * 0
        alpha[:, 20:24] = 1
        result = propagate_alpha(self.rgb, self.rgb, alpha,
                                 settings=self.settings(max_side=32), estimator=flow)
        expected = alpha * 0
        expected[:, 24:28] = 1
        self.np.testing.assert_array_equal(result.alpha, expected)
        self.assertEqual(result.diagnostics["flow_evaluations"], 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(result.uncertain[:, :4].all())
    def setUp(self):
        import numpy as np
        from corridorkey.motion import interpolate_alpha, MotionSettings
        self.np = np
        self.interpolate = interpolate_alpha
        self.settings = MotionSettings
        self.rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
        self.alpha = np.full((64, 64), 0.3, dtype=np.float32)

    def zero(self, a, b):
        return self.np.zeros((*a.shape, 2), dtype=self.np.float32)

    def test_static_fractional_matte_blends_with_reliability(self):
        result = self.interpolate(self.rgb, self.rgb, self.rgb, self.alpha,
                                  self.alpha + 0.1, 0.25, estimator=self.zero)
        self.np.testing.assert_allclose(result.alpha, 0.325, atol=1e-6)
        self.np.testing.assert_array_equal(result.reliability, 1)
        self.assertFalse(result.uncertain.any())
        self.assertEqual(result.diagnostics["flow_evaluations"], 4)
        self.assertFalse(result.diagnostics["reliability_calibrated"])

    def test_reduced_flow_displacements_are_scaled_to_native_pixels(self):
        calls = []
        def flow(a, b):
            out = self.zero(a, b)
            out[..., 0] = (-2, 2, 2, -2)[len(calls)]
            calls.append(a.shape)
            return out
        left, right = self.alpha * 0, self.alpha * 0
        left[:, 20:24], right[:, 28:32] = 1, 1
        result = self.interpolate(self.rgb, self.rgb, self.rgb, left, right, 0.5,
                                  settings=self.settings(max_side=32), estimator=flow)
        expected = left * 0
        expected[:, 24:28] = 1
        self.np.testing.assert_array_equal(result.alpha, expected)
        self.assertEqual(calls, [(32, 32)] * 4)

    def test_actual_farneback_aligns_a_textured_translation(self):
        import cv2
        np = self.np
        texture = np.random.default_rng(8).uniform(0, 1, (64, 64, 3)).astype(np.float32)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
        left = np.zeros((64, 64), np.float32)
        left[16:48, 16:36] = 1
        target = np.roll(left, 2, axis=1)
        right = np.roll(left, 4, axis=1)
        result = self.interpolate(texture, np.roll(texture, 4, axis=1),
                                  np.roll(texture, 2, axis=1), left, right, 0.5)
        region = np.s_[8:-8, 8:-8]
        error = np.abs(result.alpha[region] - target[region]).mean()
        unaligned = np.abs(((left + right) / 2)[region] - target[region]).mean()
        self.assertLess(error, 0.03)
        self.assertLess(error, unaligned)

    def test_occlusion_consistency_and_disagreement_are_uncertain(self):
        result = self.interpolate(self.rgb, self.rgb, self.rgb, self.alpha * 0,
                                  self.alpha * 0 + 1, 0.5, estimator=self.zero)
        self.assertTrue(result.uncertain.all())
        def inconsistent(a, b):
            return self.zero(a, b) + 20
        result = self.interpolate(self.rgb, self.rgb, self.rgb, self.alpha,
                                  self.alpha, 0.5, estimator=inconsistent)
        self.assertTrue(result.uncertain.all())
        self.assertEqual(float(result.reliability[-1, -1]), 0)

    def test_photometric_mismatch_reduces_one_sides_weight(self):
        result = self.interpolate(self.rgb, self.rgb * 0, self.rgb, self.alpha,
                                  self.alpha * 0, 0.5, estimator=self.zero)
        self.assertGreater(float(result.alpha.mean()), 0.28)
        self.assertFalse(result.uncertain.any())

    def test_approximation_is_explicit_and_keeps_diagnostic_mask(self):
        from corridorkey.motion import conservative_approximation
        result = self.interpolate(self.rgb, self.rgb, self.rgb, self.alpha * 0,
                                  self.alpha * 0 + 1, 0.5, estimator=self.zero)
        approximation = conservative_approximation(result, self.alpha * 0 + 1)
        self.np.testing.assert_array_equal(approximation, 1)
        self.assertTrue(result.uncertain.all())

    def test_invalid_inputs_fields_and_precancelled_work_rejected(self):
        from corridorkey.control import JobControl, CancellationToken, CancelledError
        for alpha in (self.alpha + 1, self.alpha * float("nan"), self.alpha[:2]):
            with self.assertRaises(ValueError):
                self.interpolate(self.rgb, self.rgb, self.rgb, alpha, self.alpha, 0.5)
        with self.assertRaisesRegex(ValueError, "displacement"):
            self.interpolate(self.rgb, self.rgb, self.rgb, self.alpha, self.alpha, 0.5,
                             estimator=lambda a, b: self.zero(a, b) * float("nan"))
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(CancelledError):
            self.interpolate(self.rgb, self.rgb, self.rgb, self.alpha, self.alpha, 0.5,
                             control=JobControl(token))
        token = CancellationToken()
        calls = []
        def cancel_in_estimator(a, b):
            calls.append(True)
            token.cancel()
            return self.zero(a, b)
        with self.assertRaises(CancelledError):
            self.interpolate(self.rgb, self.rgb, self.rgb, self.alpha, self.alpha, 0.5,
                             control=JobControl(token), estimator=cancel_in_estimator)
        self.assertEqual(calls, [True])
        for settings in (self.settings(max_side=True), self.settings(min_reliability=2),
                         self.settings(photometric_scale=0)):
            with self.assertRaises(ValueError):
                settings.validate()
