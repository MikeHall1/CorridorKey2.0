import unittest
from corridorkey.quick_key import plan_anchors


class AnchorTests(unittest.TestCase):
    def test_causal_plan_does_not_use_future_cut_or_end_to_force_past_anchors(self):
        plan = plan_anchors(9, 4, cuts=[5], include_final=False)
        self.assertEqual(plan.anchors, (0, 4, 5))
        self.assertEqual(tuple(plan.context(4, 4, past_only=True)), (0, 1, 2, 3, 4))
        limited = plan_anchors(15, 10, max_gap=3, include_final=False)
        self.assertEqual(limited.anchors, (0, 3, 6, 9, 10, 13))
        self.assertNotIn(14, limited.anchors)
    def test_maximum_gap_adds_anchors_without_restarting_regular_schedule(self):
        plan = plan_anchors(15, 10, max_gap=3, corrections=[4], cuts=[12])
        self.assertEqual(plan.anchors, (0, 3, 4, 7, 10, 11, 12, 14))
        self.assertIn("max_gap", dict(plan.reasons)[7])
        self.assertIn("regular", dict(plan.reasons)[10])
        for frame in range(15):
            left, right = plan.bracket(frame)
            self.assertLessEqual(right - left, 3)
        for value in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                plan_anchors(15, 10, max_gap=value)
    def test_interval_is_native_distance_and_final_anchor_is_included(self):
        self.assertEqual(plan_anchors(18, 5).anchors, (0, 5, 10, 15, 17))
        self.assertEqual(plan_anchors(1, 10).anchors, (0,))
        self.assertEqual(plan_anchors(3, 10).anchors, (0, 2))

    def test_cuts_resets_corrections_and_duplicate_points(self):
        plan = plan_anchors(20, 5, cuts=[8, 8], resets=[14], corrections=[3, 11, 14])
        self.assertEqual(plan.segments, ((0, 8), (8, 14), (14, 20)))
        self.assertEqual(plan.anchors, (0, 3, 5, 7, 8, 11, 13, 14, 19))
        self.assertEqual(plan.bracket(6), (5, 7))
        self.assertEqual(plan.bracket(8), (8, 8))
        self.assertEqual(plan.bracket(12), (11, 13))
        self.assertIn("reset", dict(plan.reasons)[14])
        self.assertIn("correction", dict(plan.reasons)[14])
        self.assertEqual(tuple(plan.context(13, 4)), tuple(range(9, 14)))
        self.assertEqual(tuple(plan.context(14, 4)), tuple(range(14, 19)))

    def test_all_matrix_intervals_cover_native_frames_without_crossing_cuts(self):
        for count in (1, 2, 7, 18, 31):
            for interval in (1, 2, 3, 5, 8, 10):
                with self.subTest(count=count, interval=interval):
                    plan = plan_anchors(count, interval, cuts=[count // 2])
                    self.assertEqual(plan.full_rate, interval == 1)
                    if interval == 1:
                        self.assertEqual(plan.anchors, tuple(range(count)))
                    for frame in range(count):
                        left, right = plan.bracket(frame)
                        start, stop = plan.segment(frame)
                        self.assertTrue(start <= left <= frame <= right < stop)
                        self.assertLessEqual(right - left, interval)

    def test_invalid_indices_and_context_rejected(self):
        for count, interval in ((0, 1), (1, 0), (True, 2), (2, 1.5)):
            with self.assertRaises(ValueError):
                plan_anchors(count, interval)
        for values in ([3], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                plan_anchors(3, 2, corrections=values)
        plan = plan_anchors(5, 3)
        for frame in (-1, 5, True):
            with self.assertRaises(ValueError):
                plan.bracket(frame)
        with self.assertRaises(ValueError):
            plan.context(1, 2)
        with self.assertRaises(ValueError):
            plan.context(0, -1)
