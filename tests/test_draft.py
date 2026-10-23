import importlib.util
import unittest

HAS_STACK = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional NumPy/OpenCV CPU stack")
class DraftTests(unittest.TestCase):
    def render(self, *, interval=3, fallback="stop", disagree=False, cuts=(4,), neural=None, adaptive=False):
        import numpy as np
        from functools import partial
        from corridorkey.draft import SourceFrame, offline_frames
        from corridorkey.motion import interpolate_alpha
        from corridorkey.quick_key import plan_anchors
        calls, anchors, stats = [], [], {}
        def source(t):
            calls.append(t)
            return SourceFrame(np.full((8, 8, 3), t + 0.25, np.float32),
                               np.full((8, 8, 3), 0.5, np.float32))
        def anchor(t):
            anchors.append(t)
            return source(t)
        def alpha(t):
            return np.full((8, 8), t % 2 if disagree else 0.5, np.float32)
        def zero(a, b):
            return np.zeros((*a.shape, 2), np.float32)
        frames = list(offline_frames(plan_anchors(8, interval, cuts=cuts), source, anchor, alpha,
                                    fallback=fallback, neural_fallback=neural, stats=stats,
                                    interpolate=partial(interpolate_alpha, estimator=zero), adaptive=adaptive))
        return frames, stats, calls, anchors

    def test_preserves_current_source_hdr_rgb_native_count_and_shot_boundaries(self):
        import numpy as np
        frames, stats, _, anchors = self.render()
        self.assertEqual([f.index for f in frames], list(range(8)))
        for frame in frames:
            np.testing.assert_array_equal(frame.rgb, frame.index + 0.25)
            np.testing.assert_array_equal(frame.alpha, 0.5)
            self.assertTrue(frame.diagnostics["source_rgb_draft"])
            left, right = frame.diagnostics["anchors"]
            self.assertEqual(left < 4, right < 4)
        self.assertEqual(anchors, [0, 3, 4, 7])
        self.assertEqual(stats["frames_emitted"], 8)
        self.assertEqual(stats["flow_evaluations"], 16)
        self.assertEqual(stats["peak_retained_anchor_sources"], 2)
        self.assertEqual(stats["max_look_ahead_frames"], 2)

    def test_interval_one_uses_no_motion_or_lookahead(self):
        frames, stats, _, anchors = self.render(interval=1)
        self.assertEqual(len(frames), 8)
        self.assertEqual(anchors, [])
        self.assertEqual(stats["flow_evaluations"], 0)
        self.assertEqual(stats["max_look_ahead_frames"], 0)

    def test_uncertain_stop_and_explicit_approximation_or_neural_fallback(self):
        import numpy as np
        from corridorkey.draft import UncertainMotionError
        with self.assertRaises(UncertainMotionError):
            self.render(disagree=True)
        _, approximate, _, _ = self.render(disagree=True, fallback="approximate")
        calls = []
        def neural(t):
            calls.append(t)
            return np.full((8, 8), 0.8, np.float32)
        frames, stats, _, _ = self.render(disagree=True, fallback="neural", neural=neural)
        self.assertEqual(calls, [1, 2, 5, 6])
        self.assertEqual(stats["extra_neural_frames"], calls)
        self.assertEqual(approximate["fallback_frames"], calls)
        for t in calls:
            np.testing.assert_array_equal(frames[t].alpha, np.float32(0.8))
        self.assertEqual(approximate["extra_neural_frames"], [])

    def test_missing_neural_callback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "callback"):
            self.render(fallback="neural")

    def test_adaptive_neural_promotions_are_reused_without_crossing_cuts(self):
        import numpy as np
        calls = []
        def neural(frame):
            calls.append(frame)
            return np.ones((8, 8), np.float32)
        _, stats, _, _ = self.render(disagree=True, fallback="neural", neural=neural, adaptive=True)
        self.assertEqual(calls, [1, 5])
        self.assertEqual(stats["promoted_anchors"], calls)
        self.assertEqual(stats["anchors"], [0, 1, 3, 4, 5, 7])
        self.assertEqual(stats["peak_retained_anchor_sources"], 2)
        with self.assertRaisesRegex(ValueError, "adaptive"):
            self.render(adaptive=True)
