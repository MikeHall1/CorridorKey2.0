import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.desktop_comparison import common_range, completed_view
from corridorkey.desktop_preview import preview_pair
from corridorkey.session import JobRequest


class ComparisonTests(unittest.TestCase):
    def view(self, root, start=10, count=3):
        job = JobRequest.create(str(root / 'source.mp4'), str(root / 'out'), InferenceConfig(output_format='exr'))
        record: dict[str, Any] = {'status': 'complete', 'input': job.input_path, 'diagnostics': {
            'source': {'start_frame': start, 'num_frames': count, 'native_hw': [2, 3], 'native_fps': 24.},
            'frames_submitted': count, 'outputs': {'alpha': str(root / 'alpha')}}}
        return completed_view(job, record), job, record

    def test_actual_offset_intersection_and_half_open_boundaries(self):
        a, _, _ = self.view(Path('/tmp'), 10, 5)
        b, _, _ = self.view(Path('/tmp'), 12, 5)
        self.assertEqual(common_range(a, b), (12, 15))
        self.assertEqual(a.options('Alpha', 12)['index'], 2)
        self.assertEqual(b.options('Alpha', 12)['index'], 0)
        self.assertEqual(a.options('Source', 12)['index'], 12)
        for index in (9, 15, True):
            with self.assertRaises(ValueError):
                a.options('Alpha', index)

    def test_unknown_or_incomplete_metadata_never_guesses_offset(self):
        _, job, record = self.view(Path('/tmp'))
        for source in (None, {}, {'start_frame': True}, {'start_frame': 0, 'num_frames': 3}):
            record['diagnostics']['source'] = source
            with self.assertRaises(ValueError):
                completed_view(job, record)
        _, job, record = self.view(Path('/tmp'))
        record['diagnostics']['frames_submitted'] = 2
        with self.assertRaises(ValueError):
            completed_view(job, record)

    def test_mismatched_sources_geometry_timing_and_nonoverlap_reject(self):
        a, _, _ = self.view(Path('/tmp'))
        for b in (replace(a, source='/other'), replace(a, width=9), replace(a, fps=30.), replace(a, start=13)):
            with self.assertRaises(ValueError):
                common_range(a, b)
        with self.assertRaises(ValueError):
            a.options('Foreground', 10)

    def test_paired_real_decode_uses_distinct_relative_offsets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, _, _ = self.view(root, 10)
            b = replace(a, start=11, outputs=(('alpha', str(root / 'beta')),))
            for name, values in [('alpha', [.1, .3, .5]), ('beta', [.7, .8, .9])]:
                (root / name).mkdir()
                for i, value in enumerate(values):
                    cv2.imwrite(str(root / name / f'{i}.exr'), np.full((2, 3), value, np.float32))
            pair = preview_pair(a.options('Alpha', 11), b.options('Alpha', 11))
            self.assertEqual(int(pair.a.pixels[0, 0, 0]), 76)
            self.assertEqual(int(pair.b.pixels[0, 0, 0]), 178)
            (root / 'beta' / '2.exr').unlink()
            with self.assertRaisesRegex(ValueError, 'frame count'):
                preview_pair(a.options('Alpha', 11), b.options('Alpha', 11))
