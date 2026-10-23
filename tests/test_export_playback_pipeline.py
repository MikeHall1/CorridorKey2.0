"""Frozen export playback membership through the existing decoder owner."""
import importlib.util
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

from corridorkey.desktop_comparison import RenderView
from corridorkey.desktop_loader import PreviewLoader
from corridorkey.desktop_preview import preview, preview_pair
from corridorkey.playback import Playback
from corridorkey.preview_background import BackgroundRecipe, prepare_background
from corridorkey.preview_exports import (
    PreparedExportIndexes,
    capture_export_request,
    prepare_export_indexes,
)


class ExportPlaybackPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(5):
            self.assertTrue(cv2.imwrite(str(self.source / f'{frame}.png'), np.full((2, 3, 3), frame * 40, np.uint8)))
        self.a = self.make_view('a', 1, (.1, .2, .3))
        self.b = self.make_view('b', 2, (.8, .9))
        self.originals = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}

    def tearDown(self):
        for path, raw in self.originals.items():
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), raw)

    def make_view(self, name, start, values):
        foreground, alpha = self.root / name / 'fg', self.root / name / 'alpha'
        foreground.mkdir(parents=True)
        alpha.mkdir()
        for index, value in enumerate(values):
            self.assertTrue(cv2.imwrite(str(foreground / f'{index}.exr'), np.full((2, 3, 3), value, np.float32)))
            self.assertTrue(cv2.imwrite(str(alpha / f'{index}.exr'), np.full((2, 3), .5, np.float32)))
        return RenderView(str(self.source), start, len(values), 2, 3, 24.,
                          (('fg', str(foreground)), ('alpha', str(alpha))), True, False, 'ignore')

    def test_prepared_ab_has_independent_fg_alpha_indexes_and_no_perframe_scan(self):
        bundle = prepare_export_indexes(capture_export_request(self.b, mode='Checker composite', reference=self.a))
        options = bundle.options_for(2)
        self.assertEqual(options['pair']['a']['index'], 1)
        self.assertEqual(options['pair']['b']['index'], 0)
        self.assertIsNot(options['pair']['a']['sequence'], options['pair']['a']['alpha_sequence'])
        with patch('corridorkey.desktop_preview.frame_paths', side_effect=AssertionError('no frame sorting during indexed playback')), patch('os.scandir', side_effect=AssertionError('no directory scanning during indexed playback')):
            first = preview_pair(**options['pair'])
            second = preview_pair(**bundle.options_for(3)['pair'])
        self.assertLess(int(first.a.pixels[0, 0, 0]), int(first.b.pixels[0, 0, 0]))
        self.assertLess(int(first.b.pixels[0, 0, 0]), int(second.b.pixels[0, 0, 0]))
        self.assertEqual(first.a.count, 3)
        self.assertEqual(first.b.count, 2)

    def test_added_membership_ignored_until_explicit_reprepare_and_missing_leaf_refuses(self):
        request = capture_export_request(self.b, mode='Alpha')
        bundle = prepare_export_indexes(request)
        added = Path(dict(self.b.outputs)['alpha']) / 'extra.exr'
        self.assertTrue(cv2.imwrite(str(added), np.ones((2, 3), np.float32)))
        self.assertEqual(preview(**bundle.options_for(2)).count, 2)
        with self.assertRaises(ValueError):
            prepare_export_indexes(request)
        added.unlink()
        original = Path(dict(self.b.outputs)['alpha']) / '0.exr'
        raw = original.read_bytes()
        original.unlink()
        with self.assertRaises((ValueError, OSError)):
            preview(**bundle.options_for(2))
        original.write_bytes(raw)
        refreshed = prepare_export_indexes(request)
        self.assertEqual(preview(**refreshed.options_for(2)).count, 2)

    def test_source_global_and_photo_output_local_coordinates_share_existing_colour_path(self):
        photo = self.root / 'photo.png'
        self.assertTrue(cv2.imwrite(str(photo), np.full((2, 3, 3), [0, 255, 0], np.uint8)))
        background = prepare_background(BackgroundRecipe.capture(photo, profile='assumed_srgb', expected_hw=(2, 3)))
        request = capture_export_request(self.b, mode='Photo composite', reference=self.a, reference_mode='Source')
        bundle = prepare_export_indexes(request)
        options = bundle.options_for(2, background_image=background)
        self.assertEqual(options['pair']['a']['index'], 2)
        self.assertEqual(options['pair']['b']['index'], 0)
        self.assertIs(options['pair']['b']['background_image'], background)
        with patch('corridorkey.desktop_preview.frame_paths', side_effect=AssertionError('all required streams indexed')):
            result = preview_pair(**options['pair'])
        np.testing.assert_allclose(result.a.pixels[0, 0], [80, 80, 80], atol=1)
        self.assertGreater(int(result.b.pixels[0, 0, 1]), int(result.b.pixels[0, 0, 0]))

    def test_kernel_attachment_preserves_cursor_clock_and_fences_old_decode(self):
        now = [0.]
        engine = Playback(self.b, source_identity='fixture', timebase='sequence_declared', timing_evidence='explicit24',
                          mode='Alpha', clock=lambda: now[0])
        engine.seek(3)
        generation = engine.generation
        bundle = prepare_export_indexes(capture_export_request(self.b, mode='Alpha'))
        engine.bind_export_indexes(bundle)
        self.assertEqual(engine.frame, 3)
        self.assertEqual(engine.generation, generation)
        engine.play()
        request = engine.tick()
        assert request is not None
        self.assertIs(request.options['sequence'], bundle.options_for(3)['sequence'])
        self.assertEqual(preview(**request.options).count, 2)
        with self.assertRaisesRegex(ValueError, 'Pause and fence'):
            engine.bind_export_indexes(bundle)
        engine.pause()
        self.assertFalse(engine.complete(request))
        with self.assertRaises(ValueError):
            engine.bind_export_indexes(prepare_export_indexes(capture_export_request(self.b, mode='Foreground')))

    @unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional desktop dependency unavailable')
    def test_controller_later_photo_terminal_retires_preparation_without_consuming_photo(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.desktop_playback import PlaybackControls
        from corridorkey.session import JobRequest

        photo = self.root / 'background.png'
        self.assertTrue(cv2.imwrite(str(photo), np.full((2, 3, 3), 128, np.uint8)))
        background = prepare_background(BackgroundRecipe.capture(photo, profile='assumed_srgb', expected_hw=(2, 3)))
        request = capture_export_request(self.b, mode='Alpha')
        job = JobRequest.create(self.source, self.root / 'unused', InferenceConfig())
        for captured in (job, request):
            messages = []
            controls: Any = SimpleNamespace(preparation_serial=10, preparation_context=(0, captured),
                                            navigation_generation=0, workspace=SimpleNamespace(closing=False),
                                            status=SimpleNamespace(setText=messages.append))
            self.assertFalse(PlaybackControls.accept_preparation(controls, 11, background, None))
            self.assertIsNone(controls.preparation_serial)
            self.assertIsNone(controls.preparation_context)
            self.assertTrue(messages)
        bundle = prepare_export_indexes(request)
        controls = SimpleNamespace(preparation_serial=None)
        self.assertTrue(PlaybackControls.accept_preparation(controls, 9, bundle, None))

    def test_loader_preparation_cancel_and_later_metadata_terminal_retire_one_owner(self):
        loader = PreviewLoader()
        started, release = threading.Event(), threading.Event()
        request = capture_export_request(self.b, mode='Alpha')
        native = prepare_export_indexes

        def gated(request, *, check):
            started.set()
            release.wait(3)
            return native(request, check=check)

        try:
            with patch('corridorkey.desktop_loader.prepare_export_indexes', side_effect=gated):
                serial = loader.submit(prepare_exports=request)
                self.assertTrue(started.wait(3))
                loader.cancel(serial)
                release.set()
                received, frame, error = loader.results.get(timeout=3)
                self.assertEqual(received, serial)
                self.assertIsNone(frame)
                self.assertIsNotNone(error)
            latest = loader.submit(prepare_exports=request)
            received, frame, error = loader.results.get(timeout=3)
            self.assertEqual(received, latest)
            assert isinstance(frame, PreparedExportIndexes)
            self.assertEqual(frame.request.identity, request.identity)
            self.assertIsNone(error)
            self.assertEqual(loader.requests.maxsize, 1)
            self.assertEqual(loader.results.maxsize, 1)
        finally:
            release.set()
            loader.stopping.set()
            loader.thread.join(3)
            self.assertFalse(loader.thread.is_alive())
