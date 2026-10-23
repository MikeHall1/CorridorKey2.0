"""Bounded native sequence indexing; no inference or Qt required."""
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

from corridorkey.preview_sequence import prepare_sequence, scan_sequence
from corridorkey.scheduling import frame_paths
from corridorkey.session import JobRequest


class PreviewSequenceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def image(self, name, value=.25, *, alpha=None):
        import cv2
        import numpy as np
        pixels = np.full((2, 3, 4 if alpha is not None else 3), value, np.float32)
        if alpha is not None:
            pixels[..., 3] = alpha
        if Path(name).suffix == '.png':
            pixels = np.rint(pixels * 255).astype(np.uint8)
        self.assertTrue(cv2.imwrite(str(self.root / name), pixels))

    def test_natural_order_matches_runtime_numeric_text_case_ties(self):
        for name in ('f10.png', 'f2.png', 'f02.png', 'A2b.png', 'a02B.png', '2f.png', 'fA.png'):
            self.image(name)
        index = scan_sequence(self.root)
        self.assertEqual(list(index.paths), frame_paths(self.root.resolve()))
        self.assertEqual(len(index.paths), 7)
        self.assertTrue(index.identity)
        self.assertEqual(scan_sequence(self.root).identity, index.identity)

    def test_exact_indexed_png_read_does_not_rescan_and_detects_changed_files(self):
        self.image('0.png', .2)
        self.image('1.png', .6)
        index = scan_sequence(self.root)
        original = {path: path.read_bytes() for path in self.root.iterdir()}
        with patch('corridorkey.preview_sequence.os.scandir', side_effect=AssertionError('unexpected rescan')):
            pixels = index.read_rgb(1)
            self.assertAlmostEqual(float(pixels[0, 0, 0]), .6, places=6)
        self.assertEqual({path: path.read_bytes() for path in self.root.iterdir()}, original)
        self.image('2.png', .8)
        self.assertEqual(len(index.paths), 2)  # new membership needs explicit re-prepare
        (self.root / '1.png').write_bytes(b'changed indexed file')
        with self.assertRaisesRegex(ValueError, 'changed'):
            index.read_rgb(1)
        with self.assertRaises(ValueError):
            index.path_at(True)

    def test_all_directory_entries_frames_and_path_bytes_are_admitted_before_sort(self):
        self.image('0.png')
        self.image('1.png')
        (self.root / 'unrelated.txt').write_text('local')
        limits_cases: tuple[tuple[dict[str, Any], str], ...] = (({'max_entries': 2}, 'entry limit'), ({'max_frames': 1}, 'frame limit'),
                                                            ({'max_path_bytes': 1}, 'path byte limit'))
        for limits, message in limits_cases:
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, message):
                scan_sequence(self.root, **limits)
        invalid_limits: tuple[Any, ...] = (True, 0, -1, 1.5)
        for value in invalid_limits:
            with self.assertRaises(ValueError):
                scan_sequence(self.root, max_entries=value)

    def test_cancelled_scan_and_invalid_video_never_decode(self):
        from corridorkey.control import CancellationToken, CancelledError
        self.image('0.png')
        token = CancellationToken()
        token.cancel('fixture cancelled')
        with self.assertRaises(CancelledError):
            scan_sequence(self.root, check=token.check)
        with self.assertRaisesRegex(ValueError, 'video/VFR'):
            scan_sequence(self.root / 'video.mp4')

    def test_prepare_native_exr_offsets_declared_timing_and_alpha_policy(self):
        from corridorkey.config import InferenceConfig
        self.image('0.exr', .1, alpha=.5)
        self.image('1.exr', .2, alpha=.5)
        self.image('2.exr', .3, alpha=.5)
        config = InferenceConfig(start_frame=1, num_frames=2, frame_dir_fps=12.,
                                 frame_dir_linear=True, frame_dir_alpha='premultiplied', source_colour_profile='linear_bt709_d65')
        job = JobRequest.create(self.root, self.root / 'unused-output', config)
        original = {path: path.read_bytes() for path in self.root.iterdir()}
        prepared = prepare_sequence(job)
        self.assertEqual((prepared.view.start, prepared.view.end, prepared.view.fps), (1, 3, 12.))
        self.assertEqual(prepared.view.options('Source', 2)['index'], 2)
        self.assertEqual(prepared.view.source_alpha, 'premultiplied')
        self.assertEqual(prepared.view.source_colour_profile, 'linear_bt709_d65')
        self.assertAlmostEqual(float(prepared.index.read_rgb(1, alpha_mode='premultiplied')[0, 0, 0]), .4, places=6)
        self.assertEqual({path: path.read_bytes() for path in self.root.iterdir()}, original)
        self.assertFalse(Path(job.output_dir).exists())

    def test_source_range_refuses_missing_start_and_uses_available_end(self):
        from corridorkey.config import InferenceConfig
        self.image('0.png')
        self.image('1.png')
        job = JobRequest.create(self.root, self.root / 'unused', InferenceConfig(start_frame=1, num_frames=20))
        self.assertEqual(prepare_sequence(job).view.count, 1)
        job = JobRequest.create(self.root, self.root / 'unused', InferenceConfig(start_frame=2, num_frames=1))
        with self.assertRaisesRegex(ValueError, 'source In'):
            prepare_sequence(job)

    def test_prepared_reader_rejects_dataclass_class_and_source_mismatch(self):
        from corridorkey.desktop_preview import read_frame
        from corridorkey.preview_sequence import SequenceIndex
        self.image('0.png')
        index = scan_sequence(self.root)
        wrong_instance: Any = SequenceIndex  # deliberate invalid runtime input
        with self.assertRaisesRegex(TypeError, 'instance'):
            read_frame(str(self.root), 0, sequence=wrong_instance)
        with self.assertRaisesRegex(ValueError, 'different source'):
            read_frame(str(self.root / 'different'), 0, sequence=index)

    def test_cooperative_cancel_is_checked_during_directory_walk(self):
        from corridorkey.control import CancellationToken, CancelledError
        for index in range(5):
            self.image(f'{index}.png')
        token = CancellationToken()
        checks = [0]

        def check():
            checks[0] += 1
            if checks[0] == 3:
                token.cancel('cancelled within directory walk')
            token.check()

        with self.assertRaises(CancelledError):
            scan_sequence(self.root, check=check)
        self.assertEqual(checks[0], 3)
