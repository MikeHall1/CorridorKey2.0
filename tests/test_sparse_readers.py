import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional CPU numeric stack")
class SparseReaderTests(unittest.TestCase):
    def test_sparse_video_windows_decode_gaps_without_retaining_or_preprocessing_them(self):
        import torch
        from corridorkey import video_io
        capture = MagicMock()
        capture.get.return_value = 24
        capture.read.return_value = (True, object())
        tensor = torch.ones(3, 2, 2)
        with (patch.object(video_io.cv2, "VideoCapture", return_value=capture),
              patch.object(video_io, "preprocess_video_frame_views", return_value=(tensor,) * 3) as preprocess):
            reader = video_io.SequentialVideoWindowReader("source", {"start_frame": 0, "num_frames": 100})
            reader.window(50, 51)
            self.assertEqual(preprocess.call_count, 1)
            self.assertEqual(capture.read.call_count, 51)
            self.assertEqual(list(reader.cache), [50])
            reader.close()

    def test_sparse_directory_windows_and_overlap_read_only_needed_frames(self):
        import torch
        from corridorkey import video_io
        tensor = torch.ones(3, 2, 2)
        with patch.object(video_io, "preprocess_frame_dir_path", return_value=(tensor,) * 3) as preprocess:
            reader = video_io.SequentialFrameDirWindowReader("source", {"frame_paths": list(range(100))})
            reader.window(50, 54)
            reader.window(52, 56)
            self.assertEqual(preprocess.call_count, 6)
            self.assertEqual(list(reader.cache), [52, 53, 54, 55])

    def test_float_exr_source_draft_keeps_signed_hdr_rgb_without_model_quantization(self):
        import cv2
        import numpy as np
        from corridorkey.video_io import DraftSourceReader
        from corridorkey.draft import DraftFrame
        from corridorkey.runtime import _render_source_draft
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "source.exr")
            bgr = np.array([[[-0.25, 2, 0.12345678]]], np.float32)
            cv2.imwrite(str(path), bgr)
            reader = DraftSourceReader(folder, {"frame_paths": [str(path)], "num_frames": 1,
                                                "output_hw": (1, 1), "exr_linear": True}, is_dir=True)
            source = reader.read(0)
            rgb = bgr[..., ::-1].copy()
            np.testing.assert_array_equal(source.rgb, rgb)
            frame = DraftFrame(0, source.rgb, np.full((1, 1), 0.3, np.float32), {})
            outputs = _render_source_draft(frame, {"despill_strength": 0}, {"fg", "cutout"}, None, use_exr=True)
            np.testing.assert_array_equal(outputs["fg"][0], bgr)
            np.testing.assert_array_equal(outputs["cutout"][0, ..., :3], bgr)
            reader.close()

    def test_preview_linear_cutout_despills_before_clipping_hdr_source(self):
        import numpy as np
        import torch
        from corridorkey.draft import DraftFrame
        from corridorkey.runtime import _render_source_draft
        frame = DraftFrame(0, np.array([[[0, 4, 0]]], np.float32),
                           np.full((1, 1), 0.3, np.float32), {})
        outputs = _render_source_draft(frame, {"despill_strength": 1, "cutout_linear": True},
                                       {"fg", "cutout", "alpha", "checker"},
                                       torch.zeros(3, 1, 1), use_exr=False)
        np.testing.assert_array_equal(outputs["cutout"][0, 0, 0], [65535, 0, 65535, 19660])
        np.testing.assert_array_equal(outputs["fg"][0, 0, 0], [0, 65535, 0])
        self.assertTrue(all(batch.dtype == np.uint16 for batch in outputs.values()))
