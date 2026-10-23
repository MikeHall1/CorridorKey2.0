"""Actual range decode and CPU runtime exports; neural stages are substitutes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.session import JobRequest
from corridorkey.video_io import (
    DraftSourceReader,
    SequentialFrameDirWindowReader,
    SequentialHintFrameDirWindowReader,
    SequentialHintVideoWindowReader,
    SequentialVideoWindowReader,
    iter_video_frame_chunks,
    probe_frame_dir_meta,
    probe_video_meta,
)
from tests import test_quick_runtime


class SourceRangeTests(unittest.TestCase):
    run_job = test_quick_runtime.QuickRuntimeTests.run_job

    def test_config_rejects_ambiguous_or_invalid_explicit_starts(self):
        for name, value in [('start_frame', v) for v in (-1, True, 1.5, '2')]:
            with self.assertRaises(ValueError):
                config = InferenceConfig()
                setattr(config, name, value)
                config.validate()
        with self.assertRaisesRegex(ValueError, 'requires start_mode'):
            InferenceConfig(start_frame=2, start_mode='middle').validate()
        InferenceConfig(start_frame=0).validate()

    def test_actual_sequence_source_hint_and_editor_share_selected_offset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, hints = root / 'source', root / 'hints'
            source.mkdir()
            hints.mkdir()
            for i in range(5):
                cv2.imwrite(str(source / f'{i}.png'), np.full((8, 8, 3), i * 40, np.uint8))
                cv2.imwrite(str(hints / f'{i}.png'), np.full((8, 8), i * 40, np.uint8))
            meta = probe_frame_dir_meta(str(source), num_frames=2, start_frame=2, shape_quantum=64,
                                       fps=24., exr_linear=False, model_scale=1.)
            self.assertEqual((meta['start_frame'], meta['num_frames']), (2, 2))
            self.assertEqual([Path(p).stem for p in meta['frame_paths']], ['2', '3'])
            reader = SequentialFrameDirWindowReader(str(source), meta)
            hint_reader = SequentialHintFrameDirWindowReader(str(hints), meta)
            try:
                self.assertEqual([Path(p).stem for p in hint_reader.paths], ['2', '3'])
                hint = hint_reader.window(0, 2)
                self.assertEqual(float(hint[0].mean()), .314453125)  #80/255 in established bfloat16 hint storage.
            finally:
                reader.close()
                hint_reader.close()
            job = JobRequest.create(source, root / 'output', InferenceConfig(start_frame=2, num_frames=2, shape_quantum=64))
            editor = CorrectionEditor(job, root)
            self.assertEqual((editor.binding['start_frame'], editor.binding['frame_count']), (2, 2))
            for start in (5, -1, True):
                with self.assertRaises(ValueError):
                    probe_frame_dir_meta(str(source), num_frames=-1, start_frame=start, shape_quantum=64,
                                         fps=24., exr_linear=False, model_scale=1.)

    def test_real_video_explicit_start_to_end_and_legacy_middle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'clip.avi')
            writer = cv2.VideoWriter(path, int.from_bytes(b'MJPG', 'little'), 24., (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(8):
                    writer.write(np.full((24, 32, 3), i * 20, np.uint8))
            finally:
                writer.release()
            options = {'num_frames': -1, 'shape_quantum': 64, 'native_resolution': True, 'start_mode': 'begin',
                       'seed': 0, 'model_scale': 1., 'linear_brightness': 1., 'linear_contrast': 1., 'linear_contrast_pivot': .18}
            meta = probe_video_meta(path, **options, start_frame=3)
            self.assertEqual((meta['start_frame'], meta['num_frames']), (3, 5))
            reader = DraftSourceReader(path, meta, is_dir=False)
            try:
                self.assertAlmostEqual(float(reader.read(0).proxy.mean()), 60 / 255, delta=.015)
            finally:
                reader.close()
            options.update(num_frames=3, start_mode='middle')
            legacy = probe_video_meta(path, **options)
            self.assertEqual((legacy['start_frame'], legacy['num_frames']), (2, 3))
            with self.assertRaises(ValueError):
                probe_video_meta(path, **options, start_frame=3)

    def test_seek_refusal_releases_capture_and_never_decodes_wrong_range(self):
        meta = {'start_frame': 2, 'num_frames': 2, 'native_fps': 24.}
        constructors = [lambda: SequentialVideoWindowReader('clip', meta),
                        lambda: SequentialHintVideoWindowReader('hint', meta),
                        lambda: DraftSourceReader('clip', meta, is_dir=False),
                        lambda: next(iter_video_frame_chunks('clip', meta, chunk_size=1, need_model=False))]
        for constructor in constructors:
            capture = MagicMock()
            capture.isOpened.return_value = True
            capture.get.side_effect = lambda key: 24. if key == cv2.CAP_PROP_FPS else 8
            capture.set.return_value = False
            with patch('corridorkey.video_io.cv2.VideoCapture', return_value=capture), self.assertRaisesRegex(RuntimeError, 'refused seek'):
                constructor()
            capture.release.assert_called_once()
            capture.read.assert_not_called()

    def test_actual_trimmed_exports_in_both_memory_paths(self):
        for low_vram in (False, True):
            with self.subTest(low_vram=low_vram), tempfile.TemporaryDirectory() as folder:
                _, diagnostics = self.run_job(folder, low_vram=low_vram, settings_overrides={
                    'start_frame': 3, 'num_frames': 4})
                assert diagnostics.source is not None
                self.assertEqual(diagnostics.source['start_frame'], 3)
                self.assertEqual(diagnostics.frames_submitted, 4)
                outputs = sorted(Path(diagnostics.outputs['cutout']).glob('*.exr'))
                self.assertEqual(len(outputs), 4)
                from corridorkey.color_utils import srgb_to_linear
                foreground = sorted(Path(diagnostics.outputs['fg']).glob('*.exr'))
                for index, path in enumerate(foreground):
                    expected = srgb_to_linear(np.array((80 + (index + 3) * 4) / 255, np.float32))
                    np.testing.assert_allclose(cv2.imread(str(path), -1), expected, atol=1e-7)
