"""Actual bounded native image inspection; not render or GPU-quality evidence."""
import hashlib
import importlib.util
import os
import struct
import tempfile
import threading
import unittest
import zlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.desktop_preview import preview, preview_pair
from corridorkey.preview_sequence import scan_sequence

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class NativePreviewTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.path = self.root / 'pattern.png'
        self.pattern = np.zeros((3, 1302, 3), np.uint8)
        self.pattern[:, 1::2] = 255
        self.assertTrue(cv2.imwrite(str(self.path), self.pattern))

    def test_native_retains_single_pixel_pattern_and_legacy_thumbnail_default(self):
        original = self.path.read_bytes()
        thumbnail = preview(str(self.path), 0)
        native = preview(str(self.path), 0, native_pixels=True)
        self.assertEqual(thumbnail.pixels.shape[1], 1280)
        self.assertEqual(native.pixels.shape, self.pattern.shape)
        np.testing.assert_array_equal(native.pixels, self.pattern)
        self.assertTrue(native.native_pixels)
        self.assertFalse(thumbnail.native_pixels)
        self.assertEqual(self.path.read_bytes(), original)

    def test_admission_precedes_decoder_and_uses_captured_bytes_with_post_fence(self):
        from corridorkey import desktop_preview, video_io
        original = video_io.read_image_rgb
        def decoded(path, **kwargs):
            self.assertIsInstance(kwargs['encoded_bytes'], bytes)
            self.assertEqual(kwargs['expected_channels'], 3)
            result = original(path, **kwargs)
            self.path.write_bytes(self.path.read_bytes() + b'changed')
            return result
        with patch.object(video_io, 'read_image_rgb', side_effect=decoded), self.assertRaises(ValueError):
            preview(str(self.path), 0, native_pixels=True)
        with patch('corridorkey.native_image_admission.admit_image', side_effect=ValueError('bounded')) as admission, \
             patch.object(video_io, 'read_image_rgb') as decoder, self.assertRaisesRegex(ValueError, 'bounded'):
            desktop_preview.preview(str(self.path), 0, native_pixels=True)
        decoder.assert_not_called()
        self.assertEqual(admission.call_args.kwargs['max_pixels'], 16 * 1024**2)
        self.assertEqual(admission.call_args.kwargs['max_bytes'], 256 * 1024**2)

    def test_video_invalid_type_mixed_pair_and_stale_sequence_refuse(self):
        invalid_modes: tuple[Any, ...] = (1, 'true', None)
        for mode in invalid_modes:
            with self.subTest(mode=mode), self.assertRaises((TypeError, ValueError)):
                preview(str(self.path), 0, native_pixels=mode)
        with patch('corridorkey.desktop_preview.cv2.VideoCapture') as video, self.assertRaises(ValueError):
            preview(str(self.root / 'movie.mov'), 0, native_pixels=True)
        video.assert_not_called()
        with patch('corridorkey.desktop_preview.read_frame') as decoder, self.assertRaises(ValueError):
            preview_pair({'path': str(self.path), 'index': 0, 'native_pixels': True},
                         {'path': str(self.path), 'index': 0})
        decoder.assert_not_called()
        index = scan_sequence(self.root)
        self.path.write_bytes(self.path.read_bytes() + b'changed')
        with patch('corridorkey.video_io.read_image_rgb') as decoder, self.assertRaises(ValueError):
            preview(str(self.root), 0, sequence=index, native_pixels=True)
        decoder.assert_not_called()

    def test_native_composite_keeps_full_checker_phase_and_signed_hdr_display_only(self):
        fg, alpha = self.root / 'fg.exr', self.root / 'alpha.exr'
        self.assertTrue(cv2.imwrite(str(fg), np.full((3, 1302, 3), -2., np.float32)))
        self.assertTrue(cv2.imwrite(str(alpha), np.zeros((3, 1302), np.float32)))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fg, alpha)}
        result = preview(str(fg), 0, linear=True, alpha_path=str(alpha), native_pixels=True)
        self.assertEqual(result.pixels.shape, self.pattern.shape)
        self.assertEqual(result.pixels[0, 0, 0], result.pixels[0, 64, 0])
        self.assertNotEqual(result.pixels[0, 0, 0], result.pixels[0, 32, 0])
        plain = preview(str(fg), 0, linear=True, native_pixels=True)
        self.assertTrue(np.all(plain.pixels == 0))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        for background in ((0., 0., 0.), (1., 1., 1.), (.2, .3, .4)):
            solid = preview(str(fg), 0, linear=True, alpha_path=str(alpha), background=background, native_pixels=True)
            np.testing.assert_array_equal(solid.pixels[0, 0], np.rint(np.array(background) * 255).astype(np.uint8))
        from corridorkey.preview_background import BackgroundRecipe, prepare_background
        photo = prepare_background(BackgroundRecipe.capture(self.path, profile='assumed_srgb', expected_hw=(3, 1302)))
        composite = preview(str(fg), 0, linear=True, alpha_path=str(alpha), background_image=photo, native_pixels=True)
        np.testing.assert_array_equal(composite.pixels, self.pattern)


    def test_declared_pixel_and_encoded_limits_refuse_before_decoder(self):
        raw = bytearray(self.path.read_bytes())
        struct.pack_into('>II', raw, 16, 4097, 4097)
        struct.pack_into('>I', raw, 29, zlib.crc32(raw[12:29]) & 0xffffffff)
        self.path.write_bytes(raw)
        with patch('corridorkey.video_io.read_image_rgb') as decoder, self.assertRaisesRegex(ValueError, 'geometry'):
            preview(str(self.path), 0, native_pixels=True)
        decoder.assert_not_called()
        with self.path.open('ab') as stream:
            stream.truncate(256 * 1024**2 + 1)
        with patch('corridorkey.video_io.read_image_rgb') as decoder, self.assertRaisesRegex(ValueError, 'byte bound'):
            preview(str(self.path), 0, native_pixels=True)
        decoder.assert_not_called()

    def test_actual_uhd_png_and_exr_preserve_native_geometry_and_file_hashes(self):
        for extension, dtype, fill in (('png', np.uint8, 80), ('exr', np.float32, -2.)):
            with self.subTest(format=extension):
                path = self.root / f'uhd.{extension}'
                image = np.full((2160, 3840, 3), fill, dtype)
                self.assertTrue(cv2.imwrite(str(path), image))
                del image
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                result = preview(str(path), 0, linear=extension == 'exr', native_pixels=True)
                self.assertEqual(result.pixels.shape, (2160, 3840, 3))
                self.assertEqual((result.width, result.height), (3840, 2160))
                self.assertEqual(int(result.pixels[0, 0, 0]), 0 if extension == 'exr' else 80)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class NativePreviewUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_desktop_playback import DesktopPlaybackTests
        DesktopPlaybackTests.setUpClass()
        cls.app = DesktopPlaybackTests.app

    def setUp(self):
        from tests.test_desktop_playback import DesktopPlaybackTests
        self.fixture = DesktopPlaybackTests()
        original = tempfile.TemporaryDirectory
        def folder(*args, **kwargs):
            kwargs.setdefault('dir', '/private/tmp')
            return original(*args, **kwargs)
        with patch('tempfile.TemporaryDirectory', side_effect=folder):
            self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window = self.fixture.window

    def test_mode_change_fences_static_a_stale_results_and_refuses_autoplay(self):
        from corridorkey import video_io
        from corridorkey.desktop_preview import PreviewFrame
        window = self.window
        settings, jobs = window.config.toPlainText(), {k: v['job'] for k, v in window.jobs.items()}
        window.request_preview()
        self.fixture.until(lambda: window.pixmap is not None)
        window.pin_reference()
        self.assertIsNotNone(window.reference)
        ui_thread = threading.get_ident()
        threads = []
        original = video_io.read_image_rgb
        def decode(*args, **kwargs):
            threads.append(threading.get_ident())
            return original(*args, **kwargs)
        with patch.object(video_io, 'read_image_rgb', side_effect=decode):
            window.native_pixels.setChecked(True)
            self.assertIsNone(window.reference)
            self.fixture.until(lambda: window.pixmap is not None)
        self.assertTrue(threads)
        self.assertTrue(all(thread != ui_thread for thread in threads))
        self.assertIn('native image pixels', window.zoom_label.text())
        self.assertIn('1:1', window.zoom_label.text())
        window.playback.play()
        self.assertIn('still-only', window.playback.status.text())
        self.assertTrue(window.playback.engine is None or not window.playback.engine.playing)
        assert window.pixmap is not None
        old = window.pixmap.cacheKey()
        window.preview_serial += 100
        window.loader.results.put_nowait((window.preview_serial, PreviewFrame(np.zeros((2, 3, 3), np.uint8), 1, 3, 2, None, 'old'), None))
        window.poll()
        self.assertEqual(window.pixmap.cacheKey(), old)
        self.assertIn('Stale inspection mode', window.preview_note.text())
        self.assertEqual(window.config.toPlainText(), settings)
        self.assertEqual({k: v['job'] for k, v in window.jobs.items()}, jobs)

    def test_synchronized_pair_native_mode_is_both_sides_and_mode_change_redecodes(self):
        window = self.window
        window.table.selectRow(0)
        window.synchronize_reference()
        window.table.selectRow(1)
        window.frame.setValue(2)
        with patch.object(window.loader, 'submit', wraps=window.loader.submit) as submit:
            window.native_pixels.setChecked(True)
            self.fixture.until(lambda: window.pixmap is not None and window.reference is not None)
        pair = submit.call_args.kwargs['pair']
        self.assertTrue(pair['a']['native_pixels'])
        self.assertTrue(pair['b']['native_pixels'])
        self.assertEqual(window.image.pixmap().size(), window.reference_image.pixmap().size())
        window.native_pixels.setChecked(False)
        self.fixture.until(lambda: window.pixmap is not None and window.reference is not None)
        self.assertIn('decoded preview', window.zoom_label.text())

    def test_native_canvas_uses_nearest_pixels_at_magnification(self):
        from PySide6.QtGui import QImage, QPixmap
        image = np.zeros((4, 16, 3), np.uint8)
        image[:, 1::2] = 255
        qimage = QImage(image.data, 16, 4, image.strides[0], QImage.Format.Format_RGB888).copy()
        canvas = self.window.image
        canvas.setPixmap(QPixmap.fromImage(qimage))
        self.window.viewport_state.native_pixels = True
        self.window.viewport_state.set_transform(2.)
        self.app.processEvents()
        painted = canvas.grab().toImage()
        rectangle = canvas.image_rect()
        y = round(rectangle.center().y())
        values = [painted.pixelColor(x, y).red() for x in range(round(rectangle.left()) + 1, round(rectangle.right()) - 1)]
        self.assertEqual(set(values), {0, 255}, 'Native magnification must not interpolate intermediate grey pixels')

    def test_native_source_single_pixel_frame_is_full_size_and_latest_serial_wins(self):
        from corridorkey.desktop_preview import PreviewFrame
        window = self.window
        path = self.fixture.root / 'wide.png'
        pixels = np.zeros((3, 1302, 3), np.uint8)
        pixels[:, 1::2] = 255
        self.assertTrue(cv2.imwrite(str(path), pixels))
        window.source.setText(str(path))
        window.view.setCurrentText('Source')
        window.native_pixels.setChecked(True)
        self.fixture.until(lambda: window.pixmap is not None)
        assert window.pixmap is not None
        self.assertEqual(window.pixmap.width(), 1302)
        before = window.pixmap.cacheKey()
        window.loader.results.put_nowait((window.preview_serial - 1,
            PreviewFrame(np.zeros((1, 1, 3), np.uint8), 1, 1, 1, None, 'obsolete', True), None))
        window.poll()
        self.assertEqual(window.pixmap.cacheKey(), before)
