import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_STACK = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional NumPy/OpenCV CPU stack")
class CorrectionRasterTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from test_corrections import document_data
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = document_data()
        self.meta = {"native_hw": (2, 2), "start_frame": 0, "num_frames": 10, "native_fps": 24}

    def entry(self, kind="matte", value=255, strength=1, frame=3, name="mask.png", **kwargs):
        import cv2
        from test_corrections import raster_entry
        from corridorkey.identities import file_sha256
        value = self.np.full((2, 2), value, self.np.uint8) if self.np.isscalar(value) else value
        path = self.root / name
        self.assertTrue(cv2.imwrite(str(path), value))
        entry = raster_entry(frame, kind, name)
        entry.update(strength=strength, **kwargs)
        entry["asset"].update(path=name, sha256=file_sha256(path))
        return entry

    def rasters(self, entries):
        from corridorkey.corrections import CorrectionDocument
        from corridorkey.correction_rasters import CorrectionRasters
        self.data["entries"] = entries
        return CorrectionRasters(CorrectionDocument.parse(self.data, self.root), self.meta)

    def test_soft_strength_has_defined_matte_keep_remove_effects(self):
        for kind, expected in (("matte", 0.4), ("keep", 0.4), ("remove", 0.15)):
            with self.subTest(kind=kind):
                rasters = self.rasters([self.entry(kind, strength=0.25)])
                result = rasters.apply(self.np.full((2, 2), 0.2, self.np.float32), 3)
                self.np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_order_and_forward_static_range_are_explicit(self):
        keep = self.entry("keep", name="keep.png")
        remove = self.entry("remove", name="remove.png")
        alpha = self.np.full((2, 2), 0.3, self.np.float32)
        self.np.testing.assert_array_equal(self.rasters([keep, remove]).apply(alpha, 3), 0)
        self.np.testing.assert_array_equal(self.rasters([remove, keep]).apply(alpha, 3), 1)
        forward = self.entry("keep", direction="forward", range=[3, 5])
        rasters = self.rasters([forward])
        self.np.testing.assert_array_equal(rasters.apply(alpha, 4), 1)
        self.np.testing.assert_array_equal(rasters.apply(alpha, 2), alpha)
        self.np.testing.assert_array_equal(rasters.apply(alpha, 4, temporal=True), alpha)

    def test_partial_keep_does_not_fabricate_unknown_hint_pixels(self):
        rasters = self.rasters([self.entry("keep")])
        self.assertEqual(rasters.condition(3), (None, None))
        base = self.np.full((2, 2), 0.3, self.np.float32)
        hint, quality = rasters.condition(3, base)
        assert hint is not None
        self.np.testing.assert_array_equal(hint, 1)
        self.assertIsNone(quality)
        rasters = self.rasters([self.entry("matte", strength=0.5)])
        hint, quality = rasters.condition(3)
        assert hint is not None
        self.np.testing.assert_array_equal(hint, 1)
        self.assertEqual(quality, 0.5)

    def test_rgba_uses_coverage_and_uint16_keeps_precision(self):
        rgba = self.np.full((2, 2, 4), 255, self.np.uint8)
        rgba[..., 3] = 0
        rasters = self.rasters([self.entry(value=rgba)])
        self.np.testing.assert_array_equal(rasters.apply(self.np.ones((2, 2)), 3), 0)
        mask = self.np.full((2, 2), 32768, self.np.uint16)
        result = self.rasters([self.entry(value=mask)]).apply(self.np.zeros((2, 2)), 3)
        self.np.testing.assert_allclose(result, 32768 / 65535, atol=1e-7)

    def test_bad_dimensions_rgb_masks_nan_and_changed_assets_are_rejected(self):
        for mask in (self.np.zeros((1, 1), self.np.uint8), self.np.zeros((2, 2, 3), self.np.uint8),
                     self.np.full((2, 2), float("nan"), self.np.float32)):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                self.rasters([self.entry(value=mask, name="mask.exr" if mask.dtype.kind == "f" else "mask.png")])
        entries = [self.entry(name=f"mask{i}.png") for i in range(3)]
        rasters = self.rasters(entries)
        self.assertEqual(len(rasters._cache), 2)
        (self.root / "mask0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            rasters.apply(self.np.ones((2, 2)), 3)

    def test_mismatched_timeline_and_disabled_strength(self):
        self.meta["native_fps"] = 25
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.rasters([])
        self.meta["native_fps"] = 24
        rasters = self.rasters([self.entry(strength=0)])
        self.assertFalse(rasters.has_effect(3))
        self.assertEqual(rasters.condition(3), (None, None))
