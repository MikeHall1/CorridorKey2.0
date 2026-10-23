import importlib.util
import unittest

HAS_STACK = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional NumPy/OpenCV CPU stack")
class PastDraftTests(unittest.TestCase):
    def render(self, approach, *, fallback="stop", adaptive=False, fail_once=False, fast=False):
        import numpy as np
        from corridorkey.quick_key import plan_anchors
        from corridorkey.draft import SourceFrame, past_frames
        from corridorkey.motion import Propagation
        stats, calls = {}, []
        current = -1
        def source(frame):
            nonlocal current
            current = frame
            return SourceFrame(np.full((8, 8, 3), frame + 0.25, np.float32),
                               np.full((8, 8, 3), 0.5, np.float32))
        def alpha(frame):
            self.assertLessEqual(frame, current, "past assembly requested a future neural matte")
            return np.zeros((8, 8), np.float32) if fail_once else np.full((8, 8), frame / 10, np.float32)
        def propagate(anchor, target, matte, **kwargs):
            uncertain = np.full((8, 8), fail_once and not matte.any())
            return Propagation(matte.copy(), np.ones((8, 8), np.float32), uncertain,
                               {"flow_evaluations": 2, "motion_seconds": 0, "motion_fraction": 0.2 if fast else 0})
        def neural(frame):
            self.assertEqual(frame, current)
            calls.append(frame)
            return np.ones((8, 8), np.float32)
        frames = list(past_frames(plan_anchors(8, 3, cuts=[4]), source, alpha, approach=approach,
                                  fallback=fallback, neural_fallback=neural, stats=stats,
                                  adaptive=adaptive, propagate=propagate))
        return frames, stats, calls

    def test_hold_baseline_keeps_current_rgb_and_uses_no_motion_or_future_mattes(self):
        import numpy as np
        frames, stats, calls = self.render("hold")
        self.assertEqual(stats["flow_evaluations"], 0)
        self.assertEqual(calls, [])
        for frame in frames:
            np.testing.assert_array_equal(frame.rgb, frame.index + 0.25)
            anchor = (0, 0, 0, 3, 4, 4, 4, 7)[frame.index]
            np.testing.assert_array_equal(frame.alpha, np.float32(anchor / 10))
            self.assertEqual(frame.diagnostics["look_ahead_frames"], 0)

    def test_causal_reads_only_past_and_logs_anchor_age(self):
        frames, stats, _ = self.render("causal")
        self.assertEqual(stats["flow_evaluations"], 8)
        self.assertEqual(stats["peak_retained_anchor_sources"], 1)
        self.assertEqual(frames[2].diagnostics["age_since_anchor_frames"], 2)
        self.assertEqual(frames[4].diagnostics["age_since_anchor_frames"], 0)

    def test_causal_uncertainty_requests_a_new_neural_anchor_and_resets_at_cuts(self):
        from corridorkey.draft import UncertainMotionError
        with self.assertRaises(UncertainMotionError):
            self.render("causal", fail_once=True)
        _, stats, calls = self.render("causal", fail_once=True, fallback="neural")
        self.assertEqual(calls, [1, 5])
        self.assertEqual(stats["promoted_anchors"], calls)

    def test_motion_diagnostic_can_request_adaptive_anchors_without_uncertainty(self):
        _, stats, calls = self.render("causal", fallback="neural", adaptive=True, fast=True)
        self.assertEqual(calls, [1, 2, 5, 6])
        self.assertTrue(all(item["reason"] == "motion_threshold" for item in stats["promotion_reasons"]))
