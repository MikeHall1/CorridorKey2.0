import importlib.util
import unittest

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional CPU numeric stack")
class RefinementTests(unittest.TestCase):
    def test_off_and_constant_alpha_are_preserved(self):
        import numpy as np
        from corridorkey.refinement import refine_alpha
        alpha = np.random.default_rng(8).random((8, 8)).astype(np.float32)
        guidance = np.zeros((8, 8, 3), np.float32)
        np.testing.assert_array_equal(refine_alpha(alpha, guidance), alpha)
        np.testing.assert_allclose(refine_alpha(alpha * 0 + 0.4, guidance, radius=2), 0.4, atol=1e-6)

    def test_filter_has_real_soft_pixel_effect_preserves_hard_endpoints_and_respects_rgb_edge(self):
        import numpy as np
        from corridorkey.refinement import refine_alpha
        alpha = np.full((5, 5), 0.2, np.float32)
        alpha[2, 2], alpha[0, 0], alpha[-1, -1] = 0.8, 0, 1
        rgb = np.zeros((5, 5, 3), np.float32)
        smooth = refine_alpha(alpha, rgb, radius=1)
        self.assertLess(float(smooth[2, 2]), 0.8)
        self.assertEqual((float(smooth[0, 0]), float(smooth[-1, -1])), (0, 1))
        rgb[2, 2] = 1
        protected = refine_alpha(alpha, rgb, radius=1)
        self.assertGreater(float(protected[2, 2]), float(smooth[2, 2]))
        partial = refine_alpha(alpha, rgb * 0, radius=1, strength=0.5)
        np.testing.assert_allclose(partial, (alpha + smooth) / 2, atol=1e-6)

    def test_filter_validates_inputs_and_checks_cancellation(self):
        import numpy as np
        from corridorkey.refinement import refine_alpha
        from corridorkey.control import CancellationToken, CancelledError
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(CancelledError):
            refine_alpha(np.ones((2, 2)), np.zeros((2, 2, 3)), radius=1, check=token.check)
        with self.assertRaises(ValueError):
            refine_alpha(np.full((2, 2), float("nan")), np.zeros((2, 2, 3)), radius=1)

    def test_default_despill_matches_original_equations(self):
        import torch
        from corridorkey.runtime import _despill_average_srgb_torch
        fg = torch.rand(2, 3, 4, 4)
        r, g, b = fg[:, 0:1], fg[:, 1:2], fg[:, 2:3]
        spill = (g - (r + b) * 0.5).clamp(min=0)
        full = torch.cat((r + spill * 0.5, g - spill, b + spill * 0.5), dim=1)
        for strength in (0, 0.5, 1):
            torch.testing.assert_close(_despill_average_srgb_torch(fg, strength),
                                       fg * (1 - strength) + full * strength, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _despill_average_srgb_torch(torch.full((1, 3, 1, 1), 3e38), 0.5)

    def test_blue_screen_edge_weight_and_linear_luminance_preservation(self):
        import torch
        from corridorkey.runtime import _despill_average_srgb_torch
        fg = torch.tensor([0.1, 0.2, 2.0]).reshape(1, 3, 1, 1).expand(1, 3, 1, 3)
        alpha = torch.tensor([0, 0.5, 1]).reshape(1, 1, 1, 3)
        result = _despill_average_srgb_torch(fg, 1, screen="blue", preserve_luma=True,
                                            edges_only=True, alpha=alpha)
        torch.testing.assert_close(result[..., 0], fg[..., 0])
        torch.testing.assert_close(result[..., 2], fg[..., 2])
        self.assertLess(float(result[0, 2, 0, 1]), float(fg[0, 2, 0, 1]))
        weights = torch.tensor([0.2126, 0.7152, 0.0722]).reshape(1, 3, 1, 1)
        torch.testing.assert_close((result * weights).sum(1), (fg * weights).sum(1), atol=1e-6, rtol=1e-6)
