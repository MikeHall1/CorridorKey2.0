"""Bounded real image admission before native allocation and shared-byte math."""
import hashlib
import os
import struct
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.control import CancellationToken, CancelledError
from corridorkey.native_image_admission import MAX_ENCODED_BYTES, admit_image
from corridorkey.video_io import _read_frame_dir_views, read_image_rgb


class NativeImageAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.image = np.full((24, 32, 3), (40, 130, 210), np.uint8)
        self.path = self.root / 'image.png'
        self.assertTrue(cv2.imwrite(str(self.path), self.image))

    def tearDown(self):
        self.temp.cleanup()

    def admit(self, path=None, **kwargs):
        return admit_image(path or self.path, expected_hw=(24, 32), max_pixels=1024, **kwargs)

    def test_padded_oversized_file_is_refused_before_any_decoder(self):
        with self.path.open('ab') as stream:
            stream.truncate(MAX_ENCODED_BYTES + 1)
        with patch('cv2.imdecode', side_effect=AssertionError('native decoder must not run')), patch('cv2.imread', side_effect=AssertionError('native decoder must not run')), self.assertRaisesRegex(ValueError, 'encoded byte bound'):
            self.admit()

    def test_geometry_and_unsupported_headers_refuse_without_decode(self):
        with patch('cv2.imdecode', side_effect=AssertionError('must not decode')):
            with self.assertRaisesRegex(ValueError, 'geometry'):
                admit_image(self.path, expected_hw=(32, 24), max_pixels=1024)
            with self.assertRaisesRegex(ValueError, 'geometry'):
                admit_image(self.path, expected_hw=None, max_pixels=500)
            self.path.write_bytes(b'GIF89a' + bytes(100))
            with self.assertRaisesRegex(ValueError, 'Only admitted'):
                self.admit()

    def test_symlink_hardlink_directory_alias_and_nonregular_refuse(self):
        alias = self.root / 'alias.png'
        alias.symlink_to(self.path)
        with self.assertRaises((ValueError, OSError)):
            self.admit(alias)
        alias.unlink()
        os.link(self.path, alias)
        with self.assertRaisesRegex(ValueError, 'single-link'):
            self.admit()
        alias.unlink()
        directory_alias = self.root / 'alias-directory'
        directory_alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'alias'):
            self.admit(directory_alias / self.path.name)
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.admit(self.root)

    def test_pinned_read_fences_growth_truncation_and_replacement(self):
        original = self.path.read_bytes()
        for mutation in ('grow', 'truncate', 'replace'):
            self.path.write_bytes(original)
            calls = [0]

            def check(calls=calls, mutation=mutation):
                calls[0] += 1
                if calls[0] == 2:
                    if mutation == 'grow':
                        with self.path.open('ab') as stream:
                            stream.write(b'extra')
                    elif mutation == 'truncate':
                        self.path.write_bytes(original[:20])
                    else:
                        replacement = self.root / 'replacement.png'
                        replacement.write_bytes(original)
                        replacement.replace(self.path)

            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'changed|replaced|truncated'):
                self.admit(check=check)

    def test_cancelled_pinned_read_closes_fd_and_retains_original(self):
        token = CancellationToken()
        before = self.path.read_bytes()
        calls = [0]

        def check():
            calls[0] += 1
            if calls[0] == 2:
                token.cancel('fixture cancel')
            token.check()

        with patch('corridorkey.native_image_admission.os.close', wraps=os.close) as closed:
            with self.assertRaises(CancelledError):
                self.admit(check=check)
            self.assertEqual(closed.call_count, 1)
        self.assertEqual(before, self.path.read_bytes())

    def test_png_truncation_crc_palette_and_trailing_data_refuse(self):
        original = self.path.read_bytes()
        corrupt = bytearray(original)
        corrupt[20] ^= 1
        for raw in (original[:20], original[:-3], bytes(corrupt), original + b'padding'):
            self.path.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.admit()

    def test_jpeg_baseline_progressive_gray_and_second_frame_refusal(self):
        for channels in (1, 3):
            for progressive in (0, 1):
                path = self.root / f'jpeg{channels}-{progressive}.jpg'
                image = self.image[..., 0] if channels == 1 else self.image
                cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_PROGRESSIVE, progressive])
                result = self.admit(path)
                self.assertEqual((result.channels, result.bit_depth, result.format), (channels, 8, 'jpeg'))
                np.testing.assert_array_equal(read_image_rgb(path), read_image_rgb(path, encoded_bytes=result.encoded_bytes))
        raw = path.read_bytes()
        path.write_bytes(raw[:-2])
        with self.assertRaisesRegex(ValueError, 'truncated'):
            self.admit(path)
        # Second SOF inside the scan must not bypass predecode geometry admission.
        sof = b'\xff\xc0' + struct.pack('>HBHHB', 11, 8, 65535, 65535, 1) + b'\x01\x11\x00'
        path.write_bytes(raw[:-2] + sof + b'\xff\xd9')
        with self.assertRaisesRegex(ValueError, 'Only one'):
            self.admit(path)

    def test_actual_png_u8_u16_gray_rgba_and_float_half_exr_share_exact_math(self):
        rng = np.random.default_rng(12)
        for extension, dtype, channels, half in (
            ('png', np.uint8, 3, False), ('png', np.uint16, 3, False),
            ('png', np.uint16, 1, False), ('png', np.uint8, 4, False),
            ('exr', np.float32, 3, False), ('exr', np.float32, 4, False),
            ('exr', np.float32, 1, True), ('exr', np.float32, 3, True),
        ):
            with self.subTest(extension=extension, dtype=dtype, channels=channels, half=half):
                shape = (24, 32) if channels == 1 else (24, 32, channels)
                if dtype == np.float32:
                    image = rng.uniform(-.1, 2, shape).astype(np.float32)
                    if channels == 4:
                        image[..., 3] = .5
                else:
                    image = (rng.integers(0, 255, shape, dtype=np.uint8) if dtype == np.uint8 else
                             rng.integers(0, 65535, shape, dtype=np.uint16))
                    if channels == 4:
                        image[..., 3] = 128
                path = self.root / f'pixels.{extension}'
                options = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF] if half else []
                self.assertTrue(cv2.imwrite(str(path), image, options))
                admitted = self.admit(path)
                self.assertEqual(admitted.channels, channels)
                for alpha in (('ignore', 'premultiplied') if channels == 4 else ('ignore',)):
                    np.testing.assert_array_equal(read_image_rgb(path, alpha_mode=alpha), read_image_rgb(path, alpha_mode=alpha, encoded_bytes=admitted.encoded_bytes))
                for linear in (False, True):
                    old = _read_frame_dir_views(str(path), exr_linear=linear)
                    captured = _read_frame_dir_views(str(path), exr_linear=linear, encoded_bytes=admitted.encoded_bytes)
                    for left, right in zip(old, captured):
                        np.testing.assert_array_equal(left, right)
                self.assertEqual(admitted.sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_exr_unsupported_flags_windows_channels_and_truncation(self):
        path = self.root / 'float.exr'
        cv2.imwrite(str(path), self.image.astype(np.float32) / 255)
        original = path.read_bytes()
        for flag in (0x200, 0x800, 0x1000):
            path.write_bytes(original[:4] + struct.pack('<I', 2 | flag) + original[8:])
            with self.assertRaisesRegex(ValueError, 'Tiled/deep/multipart'):
                self.admit(path)
        for raw in (original[:10], original[:-3]):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.admit(path)
        window_start = original.index(b'dataWindow\0box2i\0') + len(b'dataWindow\0box2i\0') + 4
        malformed = bytearray(original)
        struct.pack_into('<i', malformed, window_start, 1)
        path.write_bytes(malformed)
        with self.assertRaisesRegex(ValueError, 'windows'):
            self.admit(path)
        channel_start = original.index(b'channels\0chlist\0') + len(b'channels\0chlist\0') + 4
        malformed = bytearray(original)
        struct.pack_into('<i', malformed, channel_start + 2, 0)  # UINT channels unsupported.
        path.write_bytes(malformed)
        with self.assertRaisesRegex(ValueError, 'HALF/FLOAT'):
            self.admit(path)

    def test_immutable_capture_uses_original_bytes_after_path_replaced(self):
        admitted = self.admit()
        expected = read_image_rgb(self.path, encoded_bytes=admitted.encoded_bytes)
        cv2.imwrite(str(self.path), np.zeros_like(self.image))
        with patch('cv2.imread', side_effect=AssertionError('must decode captured bytes')):
            actual = read_image_rgb(self.path, encoded_bytes=admitted.encoded_bytes)
        np.testing.assert_array_equal(expected, actual)
        invalid: Any = admitted
        with self.assertRaises(FrozenInstanceError):
            invalid.sha256 = 'changed'
        self.assertNotIn(repr(admitted.encoded_bytes), repr(admitted))

    def test_header_phase_replacement_is_fenced_before_return(self):
        original = self.path.read_bytes()
        calls = [0]

        def check():
            calls[0] += 1
            if calls[0] == 4:
                replacement = self.root / 'during-header.png'
                replacement.write_bytes(original)
                replacement.replace(self.path)

        with self.assertRaisesRegex(ValueError, 'changed while admitting'):
            self.admit(check=check)
