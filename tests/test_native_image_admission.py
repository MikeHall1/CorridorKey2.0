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

    def test_explicit_uhd_ceilings_do_not_change_existing_defaults(self):
        from inspect import signature

        from corridorkey.native_image_admission import (
            HARD_MAX_ENCODED_BYTES,
            HARD_MAX_PIXELS,
            MAX_PIXELS,
        )
        self.assertEqual((MAX_PIXELS, MAX_ENCODED_BYTES), (4 * 1024**2, 64 * 1024**2))
        self.assertEqual(signature(admit_image).parameters['max_bytes'].default, MAX_ENCODED_BYTES)
        self.assertEqual((HARD_MAX_PIXELS, HARD_MAX_ENCODED_BYTES), (16 * 1024**2, 256 * 1024**2))
        self.assertEqual(admit_image(self.path, expected_hw=(24, 32), max_pixels=HARD_MAX_PIXELS,
                                   max_bytes=HARD_MAX_ENCODED_BYTES).shape, (24, 32))
        for kwargs in ({'max_pixels': HARD_MAX_PIXELS + 1}, {'max_bytes': HARD_MAX_ENCODED_BYTES + 1},
                       {'max_pixels': True}, {'max_bytes': True}, {'max_pixels': 16.}, {'max_bytes': float('nan')}):
            with (self.subTest(kwargs=kwargs), patch('cv2.imdecode', side_effect=AssertionError('must not decode')),
                  self.assertRaisesRegex(ValueError, 'hard cap')):
                admit_image(self.path, expected_hw=None, **{'max_pixels': MAX_PIXELS, **kwargs})

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


class WindowsPathClockReaderTests(unittest.TestCase):
    """Real descriptors retain ChangeTime while controlled path stats expose birthtime."""

    def setUp(self):
        import stat
        from types import SimpleNamespace

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / 'asset.bin'
        self.path.write_bytes(b'bounded local reader fixture')
        self.payload = self.path.read_bytes()
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.stat = os.stat
        self.lstat = os.lstat
        self.birthtimes = {}

        def birthtime(info):
            if not stat.S_ISREG(info.st_mode):
                return info
            fields = {key: getattr(info, key) for key in dir(info) if key.startswith('st_')}
            fields['st_ctime_ns'] = self.birthtimes.setdefault(
                (info.st_dev, info.st_ino), info.st_ctime_ns - 1_000_000)
            return SimpleNamespace(**fields)

        for scoped_patch in (
                patch('os.stat', side_effect=lambda *args, **kwargs: birthtime(self.stat(*args, **kwargs))),
                patch('os.lstat', side_effect=lambda *args, **kwargs: birthtime(self.lstat(*args, **kwargs))),
                patch('corridorkey.identities.sys.platform', 'win32')):
            scoped_patch.start()
            self.addCleanup(scoped_patch.stop)

    def test_native_image_preserves_path_identity_for_header_fence(self):
        path = self.root / 'native.png'
        self.assertTrue(cv2.imwrite(str(path), np.zeros((2, 3, 3), np.uint8)))
        image = admit_image(path, expected_hw=(2, 3), max_pixels=6)
        self.assertEqual(image.encoded_bytes, path.read_bytes())

    def test_trimap_reader_accepts_distinct_path_clock(self):
        from corridorkey.trimap_refinement import _read
        self.assertEqual(_read(self.path, lambda: None), (self.payload, self.digest))

    def test_tracked_correction_reader_accepts_distinct_path_clock(self):
        from corridorkey.corrections import load_tracked_asset
        path = self.root / 'mask.exr'
        pixels = np.full((2, 3), .5, np.float32)
        self.assertTrue(cv2.imwrite(str(path), pixels))
        result = load_tracked_asset(path, expected_hw=(2, 3),
                                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(np.array_equal(result, pixels))

    def test_guidance_resource_reader_accepts_distinct_path_clock(self):
        from corridorkey.guidance_provenance import verify_guidance_resources
        data = {'guidance': [{'resources': [{'asset': {'path': self.path.name, 'sha256': self.digest}}]}]}
        verify_guidance_resources(data, self.root)

    def test_trace_reader_accepts_distinct_path_clock(self):
        from corridorkey.diagnostic_ranges import _read_lines
        self.path.write_bytes(b'{"frame":0}\n')
        rows = list(_read_lines(self.path, 1, None))
        self.assertEqual(rows[0], {'frame': 0})
        self.assertEqual(rows[-1]['_receipt']['bytes'], self.path.stat().st_size)

    def test_writer_receipt_accepts_distinct_path_clock(self):
        from types import SimpleNamespace

        from corridorkey.durable_writer_prefix import DurableWriterPrefix
        owner: Any = SimpleNamespace(store=SimpleNamespace(max_bytes=4096),
                                     stats={'receipt_bytes_hashed': 0}, _check=lambda: None)
        self.assertEqual(DurableWriterPrefix._receipt(owner, self.path),
                         {'bytes': len(self.payload), 'sha256': self.digest})

    def test_review_media_copy_and_seal_accept_distinct_path_clock(self):
        from corridorkey.evaluation_dossier_export import _copy_media, _sealed
        destination = self.root / 'copied.bin'
        expected = {'bytes': len(self.payload), 'sha256': self.digest}
        identity = _copy_media(self.path, destination, expected, None)
        self.assertEqual(identity[-1], self.path.stat().st_ctime_ns)
        self.assertNotEqual(identity[-1], self.stat(self.path).st_ctime_ns)
        marker = b'complete'
        (self.root / 'COMPLETE.json').write_bytes(marker)
        self.assertTrue(_sealed(self.root, marker, [{'path': destination.name, **expected}]))

    def test_stress_reader_serializes_path_clock(self):
        from corridorkey.stress_suite import _file_digest
        result = _file_digest(self.path, lambda: None)
        self.assertEqual(result['sha256'], self.digest)
        self.assertEqual(result['identity'][-1], self.path.stat().st_ctime_ns)

    def test_repository_reader_accepts_distinct_path_clock(self):
        from corridorkey.benchmark import _repository_file
        self.assertEqual(_repository_file(self.path, None, 4096), (self.digest, len(self.payload)))

    def test_decode_fixture_reader_accepts_distinct_path_clock(self):
        from corridorkey.benchmark_decode_reuse import _CASES, _identity
        self.path.unlink()
        for case in _CASES:
            directory = self.root / case
            directory.mkdir()
            extension = 'png' if case == 'uint16_png' else 'exr'
            (directory / f'0000.{extension}').write_bytes(self.payload)
        rows = _identity(self.root, None, 1)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row['sha256'] == self.digest for row in rows))

    def test_sam2_asset_hashes_complete_before_mocked_model_builder(self):
        import contextlib
        from types import SimpleNamespace
        from unittest.mock import Mock

        from corridorkey.control import JobControl
        from corridorkey.subject_guidance import (
            ExclusiveResourceLease,
            LocalSAM2Provider,
        )
        config = self.root / 'config.yaml'
        config.write_bytes(b'config fixture')
        lease = ExclusiveResourceLease('cpu', 'clock-fixture', lambda: True)
        provider = LocalSAM2Provider(config, self.path, device='cpu', lease=lease)
        builder = Mock(side_effect=RuntimeError('asset hashes reached mocked builder'))
        modules = {
            'torch': SimpleNamespace(),
            'hydra': SimpleNamespace(initialize_config_dir=lambda **kwargs: contextlib.nullcontext()),
            'hydra.core.global_hydra': SimpleNamespace(GlobalHydra=SimpleNamespace(
                instance=lambda: SimpleNamespace(is_initialized=lambda: False))),
            'sam2.build_sam': SimpleNamespace(build_sam2=builder),
            'sam2.sam2_image_predictor': SimpleNamespace(SAM2ImagePredictor=object)}
        with patch('corridorkey.subject_guidance.importlib.import_module', side_effect=modules.__getitem__), \
                patch.object(provider, '_precision_policy'), \
                self.assertRaisesRegex(RuntimeError, 'asset hashes reached mocked builder'):
            provider._load(JobControl())
        builder.assert_called_once()

    def test_descriptor_clock_mutation_is_not_hidden_by_stable_path_birthtime(self):
        from types import SimpleNamespace

        from corridorkey.trimap_refinement import _read
        original_fstat = os.fstat
        changed = False

        def touch():
            nonlocal changed
            changed = True

        def descriptor_stat(fd):
            info = original_fstat(fd)
            values = {name: getattr(info, name) for name in dir(info) if name.startswith('st_')}
            # Model a real ChangeTime event; setting the original mtime may be a Windows no-op.
            values['st_ctime_ns'] += 1_000_000_000 if changed else 0
            return SimpleNamespace(**values)

        with patch('os.fstat', side_effect=descriptor_stat), self.assertRaisesRegex(ValueError, 'changed'):
            _read(self.path, touch)
