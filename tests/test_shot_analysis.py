import importlib.util
import unittest

HAS_STACK = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional NumPy/OpenCV CPU stack")
class ShotAnalysisTests(unittest.TestCase):
    def analyze(self, values, **kwargs):
        import numpy as np
        from corridorkey.draft import SourceFrame
        from corridorkey.shot_analysis import analyze_shots
        frames = [np.full((16, 32, 3), value, np.float32) for value in values]
        return analyze_shots(len(frames), lambda t: SourceFrame(frames[t], frames[t]), **kwargs)

    def test_static_and_gentle_exposure_ramp_do_not_create_cuts(self):
        self.assertEqual(self.analyze([0.4] * 6)["detected_cuts"], [])
        self.assertEqual(self.analyze([0.2 + t * 0.01 for t in range(12)])["detected_cuts"], [])

    def test_hard_and_adjacent_cuts_have_exact_native_indices(self):
        result = self.analyze([0.1, 0.1, 0.8, 0.8, 0.2, 0.9])
        self.assertEqual(result["detected_cuts"], [2, 4, 5])
        self.assertEqual(result["frames_analyzed"], 6)
        self.assertFalse(result["semantic_cut_guarantee"])

    def test_flash_is_explicitly_a_boundary_candidate(self):
        self.assertEqual(self.analyze([0.1, 0.9, 0.1])["detected_cuts"], [1, 2])

    def test_stream_emits_ordered_diagnostics_and_bounds_proxy_dimensions(self):
        from corridorkey.shot_analysis import ShotSettings
        records = []
        result = self.analyze([0.1, 0.1, 0.8], settings=ShotSettings(max_side=16), emit=records.append)
        self.assertEqual([record["frame"] for record in records], [0, 1, 2])
        self.assertEqual(result["proxy_shape_hw"], [8, 16])
        self.assertFalse(records[0]["cut"])

    def test_invalid_settings_and_cooperative_cancellation(self):
        from corridorkey.shot_analysis import ShotSettings
        from corridorkey.control import JobControl, CancellationToken, CancelledError
        for settings in (ShotSettings(max_side=True), ShotSettings(cut_threshold=0),
                         ShotSettings(cut_threshold=float("nan"))):
            with self.assertRaises(ValueError):
                settings.validate()
        token = CancellationToken()
        def emit(record):
            token.cancel("cancel analysis")
        with self.assertRaisesRegex(CancelledError, "cancel analysis"):
            self.analyze([0.1, 0.8], control=JobControl(token), emit=emit)
