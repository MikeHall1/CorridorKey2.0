"""Optional reader review stacks preserve public pixels and shared runtime output."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


class ReaderReviewTests(unittest.TestCase):
    def test_default_independent_review_and_optout_overlap_sparse_adjusted_spatial(self):
        import cv2
        import numpy as np
        import torch

        from corridorkey import video_io

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'sequence'
            source.mkdir()
            video = root / 'source.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*'MJPG'), 24, (96, 64))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(82)
            try:
                for frame in range(8):
                    image = rng.integers(0, 256, (64, 96, 3), np.uint8)
                    self.assertTrue(cv2.imwrite(str(source / f'{frame:04}.png'), image))
                    writer.write(image)
            finally:
                writer.release()
            for kind in ('sequence', 'video'):
                for scale in (1., .5):
                    with self.subTest(kind=kind, scale=scale):
                        if kind == 'sequence':
                            cls = video_io.SequentialFrameDirWindowReader
                            path = source
                            meta = video_io.probe_frame_dir_meta(str(path), num_frames=8, shape_quantum=64,
                                                                 fps=24, exr_linear=False, model_scale=scale)
                            meta.update(linear_brightness=1.3, linear_contrast=.8, linear_contrast_pivot=.18)
                        else:
                            cls = video_io.SequentialVideoWindowReader
                            path = video
                            meta = video_io.probe_video_meta(str(path), num_frames=8, shape_quantum=64,
                                                             native_resolution=True, start_mode='begin', seed=0,
                                                             model_scale=scale, linear_brightness=1.3,
                                                             linear_contrast=.8, linear_contrast_pivot=.18)
                        default = cls(str(path), meta)
                        opted = cls(str(path), meta, include_review=False)
                        try:
                            for first, last in ((0, 3), (1, 4), (6, 8)):
                                with patch.object(torch, 'stack', wraps=torch.stack) as stack:
                                    expected = default.window(first, last)
                                    self.assertEqual(stack.call_count, 3)
                                with patch.object(torch, 'stack', wraps=torch.stack) as stack:
                                    actual = opted.window(first, last)
                                    self.assertEqual(stack.call_count, 2)
                                self.assertEqual(len(actual), 3)
                                self.assertIsNone(actual[2])
                                for a, b in zip(expected[:2], actual[:2]):
                                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                                assert expected[2] is not None
                                self.assertEqual(default.next_t, opted.next_t)
                                self.assertEqual(set(default.cache), set(opted.cache))
                                # Public default stacks must not alias each other or cached frames.
                                original_foundation = expected[1].clone()
                                original_cached = default.cache[first][2].clone()
                                expected[2].zero_()
                                torch.testing.assert_close(expected[1], original_foundation, rtol=0, atol=0)
                                torch.testing.assert_close(default.cache[first][2], original_cached, rtol=0, atol=0)
                        finally:
                            default.close()
                            opted.close()
                        if kind == 'video':
                            assert isinstance(default, video_io.SequentialVideoWindowReader)
                            assert isinstance(opted, video_io.SequentialVideoWindowReader)
                            self.assertFalse(default.cap.isOpened())
                            self.assertFalse(opted.cap.isOpened())

    def test_actual_cpu_runtime_default_vs_omitted_review_bytes(self):
        import cv2
        import numpy as np

        from corridorkey import runtime, video_io
        from tests.test_spatial_draft import SpatialDraftTests

        with tempfile.TemporaryDirectory() as temporary:
            helper = SpatialDraftTests()
            helper.root = Path(temporary)
            source = helper.source(exr=True)
            before = {p: p.read_bytes() for p in source.iterdir()}
            video = helper.root / 'source.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*'MJPG'), 24, (96, 64))
            self.assertTrue(writer.isOpened())
            try:
                for frame in range(5):
                    writer.write(np.full((64, 96, 3), 20 + frame * 30, np.uint8))
            finally:
                writer.release()
            before[video] = video.read_bytes()
            def with_review(source, meta, *, include_review=True):
                self.assertFalse(include_review)
                concrete = (video_io.SequentialFrameDirWindowReader if Path(source).is_dir()
                            else video_io.SequentialVideoWindowReader)
                return concrete(source, meta, include_review=True)
            cases = ((source, 1., False, False), (source, .5, True, False), (source, .5, True, True),
                     (video, 1., False, False), (video, .5, True, True))
            for selected, scale, low, quick in cases:
                with self.subTest(source=selected.name, scale=scale, low=low, quick=quick):
                    options = {'alpha_expand_radius': 1, 'alpha_feather_radius': 2}
                    with (patch.object(runtime, 'SequentialFrameDirWindowReader', with_review),
                          patch.object(runtime, 'SequentialVideoWindowReader', with_review)):
                        _, reference, _ = helper.run_job(selected, scale=scale, low=low, quick=quick,
                                                        outputs={'alpha', 'fg', 'cutout', 'checker'}, overrides=options)
                    _, result, _ = helper.run_job(selected, scale=scale, low=low, quick=quick,
                                                 outputs={'alpha', 'fg', 'cutout', 'checker'}, overrides=options)
                    for role in result.outputs:
                        reference_files = sorted(Path(reference.outputs[role]).glob('*.exr'))
                        result_files = sorted(Path(result.outputs[role]).glob('*.exr'))
                        self.assertEqual(len(reference_files), 5)
                        self.assertEqual(len(result_files), 5)
                        self.assertEqual([p.read_bytes() for p in reference_files], [p.read_bytes() for p in result_files])
            self.assertEqual({p: p.read_bytes() for p in before}, before)
