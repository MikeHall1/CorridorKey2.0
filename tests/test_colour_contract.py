import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional CPU numeric stack")
class ColourContractTests(unittest.TestCase):
    def test_linear_premultiplied_input_uses_coverage_and_zero_policy(self):
        import cv2
        import numpy as np
        from corridorkey.video_io import read_image_rgb, probe_frame_dir_meta, DraftSourceReader
        with tempfile.TemporaryDirectory() as folder:
            straight = np.full((2, 3, 3), [-0.25, 2.0, 0.125], np.float32)
            alpha = np.array([[0, 0.25, 1], [0.5, 5e-7, 1]], np.float32)[..., None]
            bgra = np.concatenate((straight * alpha, alpha), axis=-1)
            bgra[0, 0, :3] = [10, 20, 30]
            path = Path(folder, "frame.exr")
            cv2.imwrite(str(path), bgra)
            decoded = read_image_rgb(path, alpha_mode="premultiplied")
            expected = straight[..., ::-1].copy()
            expected[alpha[..., 0] <= 1e-6] = 0
            np.testing.assert_array_equal(decoded, expected)
            self.assertGreater(float(read_image_rgb(path)[0, 0, 0]), 1)
            meta = probe_frame_dir_meta(folder, num_frames=-1, shape_quantum=64, fps=24,
                                        exr_linear=True, model_scale=1, alpha_mode="premultiplied")
            reader = DraftSourceReader(folder, meta, is_dir=True)
            np.testing.assert_array_equal(reader.read(0).rgb, expected)
            reader.close()

    def test_premultiplied_exr_composites_match_straight_over_multiple_backgrounds(self):
        import numpy as np
        import torch
        from corridorkey import color_utils as cu, runtime
        meta = {"model_native_hw": (2, 3), "output_hw": (2, 3), "pad": (0, 0, 0, 0), "despill_strength": 0}
        alpha = torch.tensor([[0, 0.5, 1], [1, 0.25, 0]], dtype=torch.float32)
        rgb = torch.tensor([-0.25, 2, 0.125]).reshape(3, 1, 1).expand(3, 2, 3)
        logits = torch.cat((alpha[None], cu.linear_to_asinh(rgb)), dim=0)[None]
        straight = runtime._render_predictions_exr_torch(meta, {"cutout"}, None, logits)["cutout"][0]
        premul = runtime._render_predictions_exr_torch(dict(meta, cutout_alpha_mode="premultiplied"),
                                                     {"cutout"}, None, logits)["cutout"][0]
        np.testing.assert_array_equal(straight[..., 3], premul[..., 3])
        for bg in (0, 1, 0.18, np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                   np.random.default_rng(8).random((2, 3, 3)).astype(np.float32)):
            a = straight[..., 3:4]
            np.testing.assert_allclose(straight[..., :3] * a + bg * (1 - a),
                                       premul[..., :3] + bg * (1 - a), atol=1e-6)
        np.testing.assert_array_equal(premul[..., :3][alpha.numpy() == 0], 0)

    def test_association_controls_reject_ambiguous_or_unsupported_paths(self):
        from corridorkey.config import InferenceConfig
        for values in ({"frame_dir_alpha": "premultiplied"}, {"frame_dir_alpha": "auto"},
                       {"cutout_alpha_mode": "premultiplied"}, {"cutout_alpha_mode": "auto"}):
            with self.assertRaises(ValueError):
                config = InferenceConfig()
                for name, value in values.items():
                    setattr(config, name, value)
                config.validate()
    def test_uint16_neighbouring_levels_survive_decode_and_linear_draft_path(self):
        import cv2
        import numpy as np
        from corridorkey.video_io import read_image_rgb, probe_frame_dir_meta, DraftSourceReader
        with tempfile.TemporaryDirectory() as folder:
            source = np.array([[30000, 30001]], np.uint16)
            path = Path(folder, "frame.png")
            cv2.imwrite(str(path), source)
            rgb = read_image_rgb(path)
            np.testing.assert_array_equal(rgb[0, :, 0], source.astype(np.float32)[0] / 65535)
            self.assertNotEqual(float(rgb[0, 0, 0]), float(rgb[0, 1, 0]))
            meta = probe_frame_dir_meta(folder, num_frames=-1, shape_quantum=128, fps=24,
                                        exr_linear=True, model_scale=1)
            reader = DraftSourceReader(folder, meta, is_dir=True)
            np.testing.assert_array_equal(reader.read(0).rgb, rgb)
            self.assertEqual(meta["decoded_source_dtypes_observed"], ["uint16"])
            reader.close()

    def test_mixed_precision_is_reported_per_decoded_frame_without_quantization(self):
        import cv2
        import numpy as np
        from corridorkey.video_io import probe_frame_dir_meta, preprocess_frame_dir_path
        with tempfile.TemporaryDirectory() as folder:
            cv2.imwrite(str(Path(folder, "frame0.png")), np.full((2, 2, 3), 128, np.uint8))
            cv2.imwrite(str(Path(folder, "frame1.png")), np.full((2, 2, 3), 30001, np.uint16))
            meta = probe_frame_dir_meta(folder, num_frames=-1, shape_quantum=64, fps=24,
                                        exr_linear=True, model_scale=1)
            preprocess_frame_dir_path(meta["frame_paths"][1], meta, need_model=True)
            self.assertEqual(meta["decoded_source_dtypes_observed"], ["uint8", "uint16"])

    def test_nonfinite_rgb_hints_and_exr_exports_fail_explicitly(self):
        import cv2
        import numpy as np
        from corridorkey.video_io import read_image_rgb, read_hint_image, ExrSequenceWriter
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "bad.exr")
            cv2.imwrite(str(path), np.full((2, 2), float("nan"), np.float32))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_image_rgb(path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_hint_image(path)
            writer = ExrSequenceWriter(str(Path(folder, "output")), 24, (2, 2), channels=1)
            try:
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    writer.write_many(np.full((1, 2, 2, 1), float("inf"), np.float32))
            finally:
                writer.release()

    def test_sequence_geometry_change_does_not_trigger_implicit_resize(self):
        import cv2
        import numpy as np
        from corridorkey.video_io import probe_frame_dir_meta, preprocess_frame_dir_path
        with tempfile.TemporaryDirectory() as folder:
            for index, size in enumerate((2, 3)):
                cv2.imwrite(str(Path(folder, f"frame{index}.png")), np.zeros((size, size, 3), np.uint8))
            meta = probe_frame_dir_meta(folder, num_frames=-1, shape_quantum=64, fps=24,
                                        exr_linear=False, model_scale=1)
            with self.assertRaisesRegex(ValueError, "geometry changed"):
                preprocess_frame_dir_path(meta["frame_paths"][1], meta, need_model=True)
