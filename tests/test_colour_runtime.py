import importlib.util
from pathlib import Path
import tempfile
import unittest

import test_quick_runtime
import test_correction_runtime

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional CPU numeric stack")
class ColourRuntimeTests(unittest.TestCase):
    run_job = test_quick_runtime.QuickRuntimeTests.run_job
    project = test_correction_runtime.CorrectionRuntimeTests.project

    def test_refinement_evaluates_each_output_frame_once_in_both_memory_paths(self):
        import cv2
        import numpy as np
        for low_vram in (False, True):
            with self.subTest(low_vram=low_vram), tempfile.TemporaryDirectory() as folder:
                _, d = self.run_job(folder, low_vram=low_vram, alpha_pattern=True,
                                     settings_overrides={"alpha_refine_radius": 1})
                self.assertEqual(d.alpha_refinement["frames_evaluated"], 9)
                self.assertEqual(d.frames_submitted, 9)
                frame = cv2.imread(str(sorted(Path(d.outputs["alpha"]).glob("*.exr"))[0]), -1)
                self.assertLess(float(frame[32, 32]), 0.8)
                np.testing.assert_array_equal([frame[0, 0], frame[-1, -1]], [0, 1])

    def test_authored_constraints_are_applied_after_spatial_refinement(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            options = {"alpha_refine_radius": 1}
            _, baseline = self.run_job(a, enabled=False, alpha_pattern=True, settings_overrides=options)
            _, corrected = self.run_job(b, enabled=False, alpha_pattern=True, settings_overrides=options,
                                        corrections=self.project)
            first = sorted(Path(baseline.outputs["alpha"]).glob("*.exr"))
            second = sorted(Path(corrected.outputs["alpha"]).glob("*.exr"))
            keep = cv2.imread(str(second[1]), -1)
            remove = cv2.imread(str(second[2]), -1)
            matte = cv2.imread(str(second[3]), -1)
            np.testing.assert_array_equal(keep[:, :32], 1)
            np.testing.assert_array_equal(remove[:, 32:], 0)
            np.testing.assert_allclose(matte, cv2.imread(str(first[3]), -1) * 0.5 + 0.4, atol=1e-6)

    def test_uint16_linear_association_roundtrip_is_reported_and_preserved_in_draft_export(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            _, d = self.run_job(folder, source_rgba=True, settings_overrides={
                "frame_dir_linear": True, "frame_dir_alpha": "premultiplied", "cutout_alpha_mode": "premultiplied"})
            self.assertEqual(d.source["decoded_source_dtypes_observed"], ["uint16"])
            self.assertEqual(d.source["source_alpha_mode"], "premultiplied")
            self.assertEqual(d.output_contracts["cutout"]["alpha_association"], "premultiplied")
            self.assertIsNone(d.output_contracts["alpha"]["rgb_representation"])
            self.assertIsNone(d.output_contracts["alpha"]["rgb_primaries"])
            self.assertIn("premultiplied", d.outputs["cutout"])
            file = sorted(Path(d.outputs["cutout"]).glob("*.exr"))[0]
            rgba = cv2.imread(str(file), -1)
            expected_bgr = np.array([10000, 20000, 30000], np.float32) / 32768 * 0.4
            np.testing.assert_allclose(rgba[..., :3], np.broadcast_to(expected_bgr, (64, 64, 3)), atol=1e-7)
            np.testing.assert_allclose(rgba[..., 3], 0.4, atol=1e-7)

    def test_model_nan_and_decode_overflow_fail_without_completed_output(self):
        from corridorkey.diagnostics import RunDiagnostics
        for value in (float("nan"), 1000.0):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                d = RunDiagnostics()
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    self.run_job(folder, enabled=False, bad_output=value, diagnostics=d)
                self.assertEqual(d.frames_submitted, 0)
                self.assertEqual(list(Path(folder, "temp").iterdir()), [])
