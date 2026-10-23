import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from corridorkey.desktop_preview import preview


class DesktopPreviewTests(unittest.TestCase):
    def test_native_uint16_order_index_and_display_only_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            cv2.imwrite(str(path / 'frame10.png'), np.full((3, 5, 3), 65535, np.uint16))
            cv2.imwrite(str(path / 'frame2.png'), np.full((3, 5, 3), 16384, np.uint16))
            first = preview(root, 0, linear=True, max_side=4)
            self.assertEqual(first.count, 2)
            self.assertEqual((first.width, first.height), (5, 3))
            self.assertEqual(first.pixels.shape, (2, 4, 3))
            self.assertAlmostEqual(int(first.pixels[0, 0, 0]), 137, delta=1)
            self.assertEqual(int(preview(root, 1).pixels[0, 0, 0]), 255)
            self.assertEqual(int(cv2.imread(str(path / 'frame2.png'), -1)[0, 0, 0]), 16384)
            with self.assertRaises(ValueError):
                preview(root, 2)

    def test_coverage_and_linear_composites_use_actual_exports(self):
        with tempfile.TemporaryDirectory() as root:
            fg, alpha = Path(root) / 'fg.exr', Path(root) / 'alpha.exr'
            cv2.imwrite(str(fg), np.full((2, 3, 3), 0.25, np.float32))
            cv2.imwrite(str(alpha), np.full((2, 3), 0.5, np.float32))
            self.assertEqual(int(preview(str(alpha), 0, coverage=True).pixels[0, 0, 0]), 128)
            result = preview(str(fg), 0, linear=True, alpha_path=str(alpha), background=(0, 0, 0))
            self.assertAlmostEqual(int(result.pixels[0, 0, 0]), 99, delta=1)
            cv2.imwrite(str(alpha), np.full((2, 3), 2., np.float32))
            with self.assertRaises(ValueError):
                preview(str(fg), 0, alpha_path=str(alpha))

    def test_decode_failure_and_mismatched_exports_reject(self):
        with tempfile.TemporaryDirectory() as root:
            fg, alpha = Path(root) / 'fg.png', Path(root) / 'alpha.png'
            cv2.imwrite(str(fg), np.zeros((2, 3, 3), np.uint8))
            cv2.imwrite(str(alpha), np.zeros((3, 2), np.uint8))
            with self.assertRaises(ValueError):
                preview(str(fg), 0, alpha_path=str(alpha))
            with self.assertRaises(ValueError):
                preview(root, 9)
