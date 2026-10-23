"""Input preservation at the actual session and renderer boundaries."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.session import InferenceSession, JobRequest

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'cv2', 'numpy'))
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


@unittest.skipUnless(STACK, 'requires actual CPU export adapters')
class OutputSafetyRuntimeTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'plate'
        self.source.mkdir()
        self.assertTrue(cv2.imwrite(str(self.source / '0000.png'), np.full((64, 64, 3), 128, np.uint8)))
        self.output = self.root / 'render'
        self.cfg = InferenceConfig(output_format='exr', outputs={'alpha'}, exr_dtype='float',
                                   hann_chunk=8, hann_stride=4, shape_quantum=64,
                                   temp_dir=self.root / 'scratch')

    def collision_hint(self):
        import cv2
        import numpy as np

        hint = self.output / 'plate_alpha_exr' / 'alpha_000000.exr'
        hint.parent.mkdir(parents=True)
        self.assertTrue(cv2.imwrite(str(hint), np.full((64, 64), .9, np.float32)))
        return hint

    def test_initial_hint_collision_refused_before_session_model_or_runtime_mutation(self):
        from corridorkey.runtime import run_inference

        hint = self.collision_hint()
        original = hint.read_bytes()
        source_bytes = (self.source / '0000.png').read_bytes()
        loader, runner = Mock(), Mock()
        job = JobRequest.create(self.source, self.output, self.cfg, initial_hint_path=hint)
        with (InferenceSession(loader=loader, runner=runner) as session,
              self.assertRaisesRegex(ValueError, 'protect|overlap|collision')):
            session.run(job)
        loader.assert_not_called()
        runner.assert_not_called()
        # No model is needed: direct runtime must reject before dereferencing it.
        with self.assertRaisesRegex(ValueError, 'protect|overlap|collision'):
            run_inference(None, str(self.source), str(self.output), self.cfg, WriterConfig(),
                          initial_hint_path=str(hint), hint_video_path=None)
        self.assertFalse(self.cfg.temp_dir.exists())
        self.assertEqual(hint.read_bytes(), original)
        self.assertEqual((self.source / '0000.png').read_bytes(), source_bytes)

    def test_correction_asset_and_settings_document_collisions_precede_model(self):
        import cv2
        import numpy as np

        from corridorkey.corrections import CorrectionDocument, save_corrections
        from corridorkey.identities import file_sha256, source_identity

        mask = self.output / 'plate_alpha_exr' / 'alpha_000000.exr'
        mask.parent.mkdir(parents=True)
        self.assertTrue(cv2.imwrite(str(mask), np.full((64, 64), .9, np.float32)))
        data = {'schema_version': 1, 'revision': 0,
                'source': {'sha256': source_identity(self.source)['sha256'], 'start_frame': 0,
                           'frame_count': 1, 'fps': 24, 'width': 64, 'height': 64},
                'shots': [[0, 1]], 'entries': [{'id': 'keep', 'revision': 0, 'kind': 'keep', 'frame': 0,
                'strength': 1, 'direction': 'both', 'range': [0, 1],
                'asset': {'path': str(mask), 'sha256': file_sha256(mask)}}]}
        document = self.root / 'corrections.json'
        save_corrections(document, CorrectionDocument.parse(data, self.root))
        for location in (document, self.output / 'plate_settings.txt'):
            with self.subTest(document=location.name):
                if location != document:
                    data['entries'] = []
                    save_corrections(location, CorrectionDocument.parse(data, self.root))
                self.cfg.corrections_path = location
                before = {path: path.read_bytes() for path in (location, mask)}
                loader = Mock()
                with (InferenceSession(loader=loader, runner=Mock()) as session,
                      self.assertRaisesRegex(ValueError, 'protect|overlap|collision')):
                    session.run(JobRequest.create(self.source, self.output, self.cfg))
                loader.assert_not_called()
                self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertFalse(self.cfg.temp_dir.exists())

    def test_unrelated_existing_export_can_be_rendered_again_with_exact_pixels(self):
        import cv2
        import numpy as np
        from test_spatial_draft import cpu_spatial_context, make_spatial_cpu_model

        from corridorkey.runtime import run_inference

        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_bytes(b'ordinary unrelated existing output')
        source_bytes = (self.source / '0000.png').read_bytes()
        destination = self.output / 'plate_alpha_exr' / 'alpha_000000.exr'
        results = []
        with cpu_spatial_context():
            for _ in range(2):
                run_inference(make_spatial_cpu_model(), str(self.source), str(self.output), self.cfg,
                              WriterConfig(), initial_hint_path=None, hint_video_path=None)
                results.append(cv2.imread(str(destination), -1))
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_allclose(results[1], .4, atol=1e-6)
        self.assertEqual(sentinel.read_bytes(), b'ordinary unrelated existing output')
        self.assertEqual((self.source / '0000.png').read_bytes(), source_bytes)


class OutputAdmissionBoundaryTests(unittest.TestCase):
    def test_gui_request_validation_is_lightweight_but_session_runs_bulk_guard(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'plate.mov'
            source.write_bytes(b'path-only request fixture')
            job = JobRequest.create(source, Path(folder) / 'render', InferenceConfig(output_format='exr'))
            cfg, _ = job.configurations()
            loader = Mock()
            with patch('corridorkey.output_paths.validate_output_paths', side_effect=ValueError('bulk guard invoked')) as guard:
                job.validate_source(cfg)
                guard.assert_not_called()
                with (InferenceSession(loader=loader, runner=Mock()) as session,
                      self.assertRaisesRegex(ValueError, 'bulk guard invoked')):
                    session.run(job)
                guard.assert_called_once()
                loader.assert_not_called()
