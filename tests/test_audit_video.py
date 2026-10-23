"""Video timing and same-frame native-guidance regression proofs."""
import unittest
from unittest.mock import Mock, patch


class VideoAuditTests(unittest.TestCase):
    def test_missing_or_invalid_fps_is_not_invented(self):
        from corridorkey import video_io
        for fps in (0.0, -1.0, float('nan'), float('inf')):
            capture = Mock()
            capture.isOpened.return_value = True
            values = {video_io.cv2.CAP_PROP_FPS: fps,
                      video_io.cv2.CAP_PROP_FRAME_WIDTH: 64,
                      video_io.cv2.CAP_PROP_FRAME_HEIGHT: 64,
                      video_io.cv2.CAP_PROP_FRAME_COUNT: 10}
            capture.get.side_effect = values.__getitem__
            with self.subTest(fps=fps), patch.object(video_io.cv2, 'VideoCapture', return_value=capture):
                with self.assertRaisesRegex(ValueError, 'metadata|FPS'):
                    video_io.probe_video_meta('fixture.mov', num_frames=0, shape_quantum=64,
                                              native_resolution=True, start_mode='begin', seed=0,
                                              model_scale=1, linear_brightness=1,
                                              linear_contrast=1, linear_contrast_pivot=.5)
                capture.release.assert_called_once()

    def test_combined_spatial_refine_and_trimap_reads_native_frame_once(self):
        import numpy as np
        from test_spatial_draft import SpatialDraftTests
        from test_trimap_refinement import TrimapRefinementTests

        from corridorkey.corrections import save_corrections
        from corridorkey.identities import source_identity

        case = SpatialDraftTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        source = case.source()
        trimap = TrimapRefinementTests()
        trimap.root = case.root
        asset = trimap.asset(np.full((65, 97), 128, np.uint8))
        document = trimap.document(asset, shape=(65, 97), frames=5,
                                   source_hash=source_identity(source)['sha256'])
        path = case.root / 'trimap.json'
        save_corrections(path, document)
        _, diagnostics, _ = case.run_job(source, refine=True, correction=path, outputs={'alpha'})
        assert diagnostics is not None
        assert diagnostics.execution is not None
        stats = diagnostics.execution['native_trimap_guidance']
        assert isinstance(stats, dict)
        self.assertEqual(stats['frames_requested'], 5)
        self.assertEqual(stats['reader_restarts'], 0)
