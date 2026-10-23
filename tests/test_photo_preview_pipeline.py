"""Actual PNG/native EXR photographic preview plumbing; no inference claims."""
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

from corridorkey.color_utils import linear_to_srgb, srgb_to_linear
from corridorkey.desktop_comparison import RenderView
from corridorkey.desktop_loader import PreviewLoader
from corridorkey.desktop_preview import PreviewFrame, preview, preview_pair
from corridorkey.playback import Playback
from corridorkey.preview_background import (
    BackgroundRecipe,
    PreparedBackground,
    prepare_background,
)


class PhotoPreviewPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.photo = self.root / 'photo.png'
        self.assertTrue(cv2.imwrite(str(self.photo), np.full((2, 3, 3), [0, 255, 0], np.uint8)))
        self.recipe = BackgroundRecipe.capture(self.photo, profile='assumed_srgb', expected_hw=(2, 3))
        self.background = prepare_background(self.recipe)
        self.fg, self.alpha = self.root / 'fg', self.root / 'alpha'
        self.fg.mkdir()
        self.alpha.mkdir()
        for index, coverage in enumerate((0., .5, 1.)):
            self.assertTrue(cv2.imwrite(str(self.fg / f'{index}.exr'), np.full((2, 3, 3), [0., 0., 1.], np.float32)))
            self.assertTrue(cv2.imwrite(str(self.alpha / f'{index}.exr'), np.full((2, 3), coverage, np.float32)))
        self.view = RenderView(str(self.root / 'source'), 4, 3, 2, 3, 24.,
                               (('fg', str(self.fg)), ('alpha', str(self.alpha))), True, False, 'ignore')
        self.originals = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}

    def tearDown(self):
        for path, original in self.originals.items():
            self.assertEqual(path.read_bytes(), original)
        self.temp.cleanup()

    def test_linear_photo_composite_native_then_display_resize_and_source_offset(self):
        for frame, rgb in ((4, [0, 255, 0]), (5, [188, 188, 0]), (6, [255, 0, 0])):
            options = self.view.options('Photo composite', frame, background_image=self.background)
            self.assertEqual(options['index'], frame - 4)
            result = preview(**options, max_side=2)
            self.assertEqual((result.width, result.height), (3, 2))
            self.assertEqual(result.pixels.shape, (1, 2, 3))
            np.testing.assert_allclose(result.pixels[0, 0], rgb, atol=1)
            self.assertIn('display only', result.note)

    def test_display_encoded_foreground_shared_linear_math_and_immutable_pair(self):
        foreground = self.root / 'foreground.png'
        alpha = self.root / 'coverage.png'
        self.assertTrue(cv2.imwrite(str(foreground), np.full((2, 3, 3), [0, 0, 128], np.uint8)))
        self.assertTrue(cv2.imwrite(str(alpha), np.full((2, 3), 128, np.uint8)))
        options = {'path': str(foreground), 'index': 0, 'alpha_path': str(alpha), 'background_image': self.background}
        pair = preview_pair(options, dict(options))
        expected = linear_to_srgb(srgb_to_linear(np.array([128 / 255, 0, 0], np.float32)) * (128 / 255)
                                  + np.array([0, 1, 0], np.float32) * (1 - 128 / 255))
        np.testing.assert_allclose(pair.a.pixels[0, 0], np.rint(expected * 255), atol=1)
        np.testing.assert_array_equal(pair.a.pixels, pair.b.pixels)
        second = prepare_background(self.recipe)
        with self.assertRaisesRegex(ValueError, 'same immutable'):
            preview_pair(options, dict(options, background_image=second))

    def test_missing_photo_geometry_and_invalid_composite_modes_refuse(self):
        with self.assertRaisesRegex(ValueError, 'Prepare'):
            self.view.options('Photo composite', 4)
        with self.assertRaises(ValueError):
            preview(str(self.fg), 0, background_image=self.background)
        with self.assertRaises(ValueError):
            preview(str(self.fg), 0, alpha_path=str(self.alpha), background=(0., 0., 0.), background_image=self.background)
        wrong = self.root / 'wrong.exr'
        self.assertTrue(cv2.imwrite(str(wrong), np.zeros((3, 3, 3), np.float32)))
        with self.assertRaises(ValueError):
            preview(str(wrong), 0, linear=True, alpha_path=str(self.alpha), background_image=self.background)
        wrong_photo = prepare_background(BackgroundRecipe.capture(wrong, profile='linear_bt709_d65', expected_hw=(3, 3)))
        with self.assertRaises(ValueError):
            preview(str(self.fg), 0, linear=True, alpha_path=str(self.alpha), background_image=wrong_photo)

    def test_playback_pair_carries_one_snapshot_and_stale_completion_is_fenced(self):
        now = [0.]
        engine = Playback(self.view, source_identity='source', timebase='sequence_declared', timing_evidence='24fps declared',
                          mode='Photo composite', reference=self.view, background_image=self.background, clock=lambda: now[0])
        engine.play()
        request = engine.tick()
        assert request is not None
        self.assertIs(request.options['pair']['a']['background_image'], self.background)
        self.assertIs(request.options['pair']['b']['background_image'], self.background)
        pair = preview_pair(**request.options['pair'])
        np.testing.assert_array_equal(pair.a.pixels, pair.b.pixels)
        engine.pause()
        self.assertFalse(engine.complete(request))
        engine.play()
        next_request = engine.tick()
        assert next_request is not None
        self.assertTrue(engine.complete(next_request))

    def test_loader_owns_prepare_and_cancel_fences_late_asset(self):
        loader = PreviewLoader()
        started, release = threading.Event(), threading.Event()
        native = prepare_background

        def gated(recipe, *, check):
            started.set()
            release.wait(3)
            return native(recipe, check=check)

        try:
            with patch('corridorkey.desktop_loader.prepare_background', side_effect=gated):
                serial = loader.submit(prepare_background=self.recipe)
                self.assertTrue(started.wait(3))
                loader.cancel(serial)
                release.set()
                received, result, error = loader.results.get(timeout=3)
                self.assertEqual(received, serial)
                self.assertIsNone(result)
                self.assertIsNotNone(error)
            serial = loader.submit(prepare_background=self.recipe)
            received, result, error = loader.results.get(timeout=3)
            self.assertEqual(received, serial)
            self.assertIsNone(error)
            assert isinstance(result, PreparedBackground)
            self.assertEqual(result.identity, self.background.identity)
            self.assertEqual(loader.requests.maxsize, 1)
            self.assertEqual(loader.results.maxsize, 1)
        finally:
            release.set()
            loader.stopping.set()
            loader.thread.join(3)
            self.assertFalse(loader.thread.is_alive())

    def test_loader_superseded_prepare_never_returns_old_asset_and_manual_preview_survives(self):
        loader = PreviewLoader()
        started, release = threading.Event(), threading.Event()

        def gated(recipe, *, check):
            started.set()
            release.wait(3)
            return prepare_background(recipe, check=check)

        try:
            with patch('corridorkey.desktop_loader.prepare_background', side_effect=gated):
                old = loader.submit(prepare_background=self.recipe)
                self.assertTrue(started.wait(3))
                latest = loader.submit(**self.view.options('Photo composite', 5, background_image=self.background))
                release.set()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        serial, result, error = loader.results.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if serial == old:
                        self.assertIsNone(result)
                        continue
                    self.assertEqual(serial, latest)
                    self.assertIsNone(error)
                    assert isinstance(result, PreviewFrame)
                    np.testing.assert_allclose(result.pixels[0, 0], [188, 188, 0], atol=1)
                    break
                else:
                    self.fail('Latest manual thumbnail not delivered')
        finally:
            release.set()
            loader.stopping.set()
            loader.thread.join(3)
            self.assertFalse(loader.thread.is_alive())
