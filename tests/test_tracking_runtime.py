"""Shared CPU runtime consumes saved tracked constraints; toy model only."""
import hashlib
import tempfile
import unittest

import cv2
import numpy as np
import test_quick_runtime

from corridorkey.config import InferenceConfig
from corridorkey.correction_tracking_context import tracking_decode
from corridorkey.corrections import (
    CorrectionDocument,
    canonical,
    save_corrections,
    track_binding,
)
from corridorkey.identities import file_sha256, source_identity


class TrackingRuntimeTests(unittest.TestCase):
    run_job = test_quick_runtime.QuickRuntimeTests.run_job

    def project(self, source, root):
        mask = np.zeros((64, 64), np.float32)
        mask[10:20, 8:18] = 1.
        authored = root / 'authored.png'
        self.assertTrue(cv2.imwrite(str(authored), np.rint(mask * 65535).astype(np.uint16)))
        data = {'schema_version': 1, 'revision': 1,
                'source': {'sha256': source_identity(source)['sha256'], 'start_frame': 0, 'frame_count': 9,
                           'fps': 24., 'width': 64, 'height': 64}, 'shots': [[0, 9]],
                'entries': [{'id': 'artist', 'revision': 1, 'frame': 0, 'kind': 'keep', 'strength': .5,
                             'direction': 'forward', 'range': [0, 1], 'asset': {'path': str(authored), 'sha256': file_sha256(authored)}}]}
        settings = {'kernel_version': 'correction-tracking-v1', 'direction': 'forward',
                    'unreliable_policy': 'request_anchor', 'max_uncertain_fraction': .25,
                    'motion': {'max_side': 640, 'consistency_pixels': 1.5, 'photometric_scale': .15,
                               'min_reliability': .2, 'max_disagreement': .25},
                    'max_frames': 300, 'max_pixels': 4 * 1024**2, 'max_anchor_bytes': 64 * 1024**2,
                    'max_anchors': 32, 'estimator_identity': 'fixture-persisted-translation-not-neural-inference'}
        binding = track_binding(data, ['artist'], cuts=[], decode=tracking_decode(InferenceConfig(), source), tracking=settings)
        rows, observations = [], []
        for frame in range(4):
            derived = np.roll(mask, frame * 3, axis=1)
            path = root / f'derived-{frame}.exr'
            self.assertTrue(cv2.imwrite(str(path), derived))
            rows.append({'frame': frame, 'asset': {'path': str(path), 'sha256': file_sha256(path)}})
            observations.append({'frame': frame, 'source_frame': frame, 'status': 'authored_anchor' if frame == 0 else 'preview',
                                 'candidate_identity': 'b' * 64, 'source_proxy_sha256': {str(frame): 'c' * 64},
                                 'derived_mask_sha256': hashlib.sha256(derived.astype('<f4').tobytes()).hexdigest(),
                                 'diagnostics': {'fixture': True, 'reliability_calibrated': False}})
        data.update(schema_version=3, revision=2, tracks=[{'id': 'tracked', 'owner_id': 'artist', 'anchor_ids': ['artist'],
                    'range': [0, 4], 'direction': 'forward', 'binding': binding,
                    'binding_sha256': hashlib.sha256(canonical(binding).encode()).hexdigest(),
                    'frames': rows, 'observations': observations}])
        path = root / 'tracked.json'
        save_corrections(path, CorrectionDocument.parse(data, root))
        return path

    def test_full_and_quick_both_memory_paths_export_tracked_mask_once(self):
        from pathlib import Path
        for quick in (False, True):
            for low_vram in (False, True):
                with self.subTest(quick=quick, low_vram=low_vram), tempfile.TemporaryDirectory() as folder:
                    _, diagnostics = self.run_job(folder, enabled=quick, low_vram=low_vram, corrections=self.project)
                    assert diagnostics.corrections is not None
                    files = sorted(Path(diagnostics.outputs['alpha']).glob('*.exr'))
                    alpha = cv2.imread(str(files[3]), cv2.IMREAD_UNCHANGED)
                    np.testing.assert_allclose(alpha[10:20, 17:27], .7, atol=1e-6)
                    np.testing.assert_allclose(alpha[10:20, 8:17], .4, atol=1e-6)
                    np.testing.assert_allclose(cv2.imread(str(files[4]), cv2.IMREAD_UNCHANGED), .4, atol=1e-6)
                    self.assertEqual(diagnostics.corrections['hint_frame_evaluations_conditioned'], 0)
                    self.assertEqual(diagnostics.corrections['output_frames_refined'], 4)
                    self.assertEqual(diagnostics.corrections['directional_range_mode'], 'tracked_final_constraints_and_authored_static_ranges')


if __name__ == '__main__':
    unittest.main()
