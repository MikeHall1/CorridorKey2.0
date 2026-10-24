"""Generated native UHD/DCI source analysis; no neural execution or media download."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


class ShotProfileUhdTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()

    def test_actual_uhd_rgb_and_rgba_png_native_source_alignment(self):
        import cv2
        import numpy as np

        from corridorkey import color_utils as cu
        from corridorkey.config import InferenceConfig
        from corridorkey.identities import file_sha256
        from corridorkey.native_image_admission import MAX_PIXELS, admit_image
        from corridorkey.session import JobRequest
        from corridorkey.shot_analysis import analyze_shots
        from corridorkey.shot_profiles import analyze_job, apply_profiles
        shape = (2160, 3840)
        for channels in (3, 4):
            with self.subTest(channels=channels):
                source = self.root / f'png{channels}'
                source.mkdir()
                for source_index in range(3):
                    image = np.empty((*shape, channels), np.uint8)
                    image[..., :3] = (20 + source_index * 2, 80 + source_index * 2, 160 + source_index * 2)
                    if channels == 4:
                        image[..., 3] = 0 if source_index == 0 else 128
                    self.assertTrue(cv2.imwrite(str(source / f'{source_index}.png'), image))
                    del image
                before = {p.name: file_sha256(p) for p in source.iterdir()}
                with self.assertRaisesRegex(ValueError, 'pixel bound'):
                    admit_image(source / '1.png', expected_hw=None, max_pixels=MAX_PIXELS)
                job = JobRequest.create(source, self.root / 'output', InferenceConfig(start_frame=1,
                    num_frames=2, frame_dir_fps=30., temp_dir=self.root / 'must-not-create'))
                observed = []
                def inspect(count, read_source, *, observed=observed, **kwargs):
                    self.assertEqual(count, 2)
                    def read(frame):
                        result = read_source(frame)
                        self.assertEqual(result.rgb.shape, (*shape, 3))
                        self.assertEqual(result.proxy.shape, (*shape, 3))
                        self.assertEqual(result.rgb.dtype, np.float32)
                        expected = np.array([162 + frame * 2, 82 + frame * 2, 22 + frame * 2], np.float32) / 255
                        np.testing.assert_array_equal(result.proxy[0, 0], expected)
                        # Scalar-vs-native-array SIMD power differs by a few float32 ulps; proxy/legacy remain exact.
                        np.testing.assert_allclose(result.rgb[0, 0], cu.srgb_to_linear(expected), rtol=0, atol=1e-7)
                        observed.append(frame)
                        return result
                    return analyze_shots(count, read, **kwargs)
                decoder = cv2.imdecode
                with (patch('corridorkey.shot_analysis.analyze_shots', side_effect=inspect),
                      patch('cv2.imread', side_effect=AssertionError('must decode admitted bytes, not reopen source')),
                      patch('cv2.imdecode', wraps=decoder) as decoded):
                    plan = analyze_job(job)
                    self.assertEqual(decoded.call_count, 2)
                self.assertEqual(observed, [0, 1])
                self.assertEqual(plan.to_dict()['source_range'], [1, 3])
                self.assertEqual(apply_profiles(job, plan).configurations()[0].frame_dir_fps, 30.)
                self.assertEqual(before, {p.name: file_sha256(p) for p in source.iterdir()})
                self.assertFalse((self.root / 'must-not-create').exists())
                self.assertFalse((self.root / 'output').exists())

    def test_actual_dci4k_linear_hdr_rgb_and_premultiplied_rgba_exr(self):
        import cv2
        import numpy as np

        from corridorkey import color_utils as cu
        from corridorkey.config import InferenceConfig
        from corridorkey.identities import file_sha256
        from corridorkey.native_image_admission import admit_image
        from corridorkey.session import JobRequest
        from corridorkey.shot_analysis import analyze_shots
        from corridorkey.shot_profiles import (
            ANALYSIS_MAX_PIXELS,
            analyze_job,
            apply_profiles,
        )
        shape = (2160, 4096)
        for channels in (3, 4):
            with self.subTest(channels=channels):
                source = self.root / f'exr{channels}'
                source.mkdir()
                cv2.imwrite(str(source / '0.exr'), np.zeros((2, 3, channels), np.float32))
                path = source / '1.exr'
                image = np.empty((*shape, channels), np.float32)
                image[..., :3] = (2., .5, -.25)
                if channels == 4:
                    image[..., :3] *= .5
                    image[..., 3] = .5
                    image[0, 0, 3] = 0.
                self.assertTrue(cv2.imwrite(str(path), image,
                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
                     cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_NO]))
                del image
                # Actual uncompressed FLOAT RGB/RGBA exceeds the unchanged old byte default.
                self.assertGreater(path.stat().st_size, 64 * 1024**2)
                with self.assertRaisesRegex(ValueError, 'encoded byte bound'):
                    admit_image(path, expected_hw=shape, max_pixels=ANALYSIS_MAX_PIXELS)
                before = file_sha256(path)
                job = JobRequest.create(source, self.root / 'output', InferenceConfig(start_frame=1,
                    num_frames=1, frame_dir_fps=23.976, frame_dir_linear=True,
                    frame_dir_alpha='premultiplied' if channels == 4 else 'ignore',
                    source_colour_profile='linear_bt709_d65', temp_dir=self.root / 'must-not-create'))
                seen = []
                def inspect(count, read_source, *, channels=channels, seen=seen, **kwargs):
                    self.assertEqual(count, 1)
                    def read(frame):
                        result = read_source(frame)
                        self.assertEqual(result.rgb.shape, (*shape, 3))
                        self.assertEqual(result.proxy.shape, (*shape, 3))
                        self.assertEqual(result.rgb.dtype, np.float32)
                        expected = np.array([-.25, .5, 2.], np.float32)
                        np.testing.assert_array_equal(result.rgb[0, 1], expected)
                        np.testing.assert_allclose(result.proxy[0, 1],
                            np.clip(cu.linear_to_srgb(np.maximum(expected, 0)), 0, 1), rtol=0, atol=1e-7)
                        if channels == 4:
                            np.testing.assert_array_equal(result.rgb[0, 0], np.zeros(3, np.float32))
                        seen.append(frame)
                        return result
                    return analyze_shots(count, read, **kwargs)
                decoder = cv2.imdecode
                with (patch('corridorkey.shot_analysis.analyze_shots', side_effect=inspect),
                      patch('cv2.imread', side_effect=AssertionError('must decode admitted bytes, not reopen source')),
                      patch('cv2.imdecode', wraps=decoder) as decoded):
                    plan = analyze_job(job)
                    self.assertEqual(decoded.call_count, 1)
                self.assertEqual(seen, [0])
                self.assertEqual(plan.to_dict()['source_range'], [1, 2])
                self.assertEqual(apply_profiles(job, plan).configurations()[0].frame_dir_fps, 23.976)
                self.assertEqual(file_sha256(path), before)
                self.assertFalse((self.root / 'must-not-create').exists())
                self.assertFalse((self.root / 'output').exists())

    def small_job(self):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest
        source = self.root / 'small'
        source.mkdir()
        cv2.imwrite(str(source / '0.png'), np.full((8, 12, 3), (20, 80, 160), np.uint8))
        return JobRequest.create(source, self.root / 'output', InferenceConfig(temp_dir=self.root / 'scratch'))

    def test_header_and_encoded_over_budget_reject_before_native_decode(self):
        import struct
        import zlib

        from corridorkey.shot_profiles import ANALYSIS_MAX_ENCODED_BYTES, analyze_job
        job = self.small_job()
        path = Path(job.input_path) / '0.png'
        raw = bytearray(path.read_bytes())
        struct.pack_into('>II', raw, 16, 5000, 5000)  # Tiny file, over-budget declared pixels; no giant allocation.
        struct.pack_into('>I', raw, 29, zlib.crc32(raw[12:29]))
        path.write_bytes(raw)
        with (patch('cv2.imread', side_effect=AssertionError('native decode ran')),
              patch('cv2.imdecode', side_effect=AssertionError('native decode ran')),
              self.assertRaisesRegex(ValueError, 'pixel bound')):
            analyze_job(job)
        with path.open('wb') as stream:
            stream.truncate(ANALYSIS_MAX_ENCODED_BYTES + 1)  # Sparse metadata bound fixture only.
        with (patch('cv2.imdecode', side_effect=AssertionError('native decode ran')),
              patch('corridorkey.shot_profiles.read_file', side_effect=AssertionError('must reject before hash read')),
              self.assertRaisesRegex(ValueError, '256 MiB')):
            analyze_job(job)
        self.assertFalse((self.root / 'scratch').exists())

    def test_replacement_before_frame_admission_rejects_bound_sha_before_decode(self):
        import cv2
        import numpy as np

        from corridorkey.native_image_admission import admit_image
        from corridorkey.shot_profiles import analyze_job
        job = self.small_job()
        path = Path(job.input_path) / '0.png'
        calls = []
        def changed(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                cv2.imwrite(str(path), np.full((8, 12, 3), 220, np.uint8))
            return admit_image(*args, **kwargs)
        with (patch('corridorkey.native_image_admission.admit_image', side_effect=changed),
              patch('cv2.imdecode', side_effect=AssertionError('changed bytes must not reach decoder')),
              self.assertRaisesRegex(ValueError, 'bound source fingerprint')):
            analyze_job(job)
        self.assertEqual(len(calls), 2)

    def test_replacement_after_admission_decodes_only_pinned_bytes_then_refuses_publication(self):
        import hashlib

        import cv2
        import numpy as np

        from corridorkey import color_utils as cu
        from corridorkey.native_image_admission import admit_image
        from corridorkey.shot_profiles import analyze_job
        from corridorkey.video_io import _read_frame_dir_views
        job = self.small_job()
        path = Path(job.input_path) / '0.png'
        original = path.read_bytes()
        calls, decoded = [], []
        def changed(*args, **kwargs):
            result = admit_image(*args, **kwargs)
            calls.append(True)
            if len(calls) == 2:
                cv2.imwrite(str(path), np.full((8, 12, 3), 220, np.uint8))
            return result
        def inspect(*args, **kwargs):
            self.assertEqual(hashlib.sha256(kwargs['encoded_bytes']).hexdigest(), hashlib.sha256(original).hexdigest())
            proxy, linear = _read_frame_dir_views(*args, **kwargs)
            np.testing.assert_allclose(linear[0, 0], cu.srgb_to_linear(np.array([160, 80, 20], np.float32) / 255), rtol=0, atol=1e-7)
            decoded.append(True)
            return proxy, linear
        with (patch('corridorkey.native_image_admission.admit_image', side_effect=changed),
              patch('corridorkey.video_io._read_frame_dir_views', side_effect=inspect),
              patch('cv2.imread', side_effect=AssertionError('unadmitted path was reopened')),
              self.assertRaisesRegex(ValueError, 'source changed')):
            analyze_job(job)
        self.assertEqual(decoded, [True])
        self.assertNotEqual(path.read_bytes(), original)  # Only this deliberate fixture mutation touched the source.
        self.assertFalse((self.root / 'output').exists())

    def test_actual_decoded_geometry_must_match_admitted_native_header(self):
        import numpy as np

        from corridorkey.shot_profiles import analyze_job
        job = self.small_job()
        fake = np.zeros((2, 3, 3), np.float32)
        with (patch('corridorkey.video_io._read_frame_dir_views', return_value=(fake, fake)),
              self.assertRaisesRegex(ValueError, 'decoded geometry')):
            analyze_job(job)

    def test_small_source_records_are_canonical_equal_to_original_streaming_reader(self):
        import json

        from corridorkey.shot_analysis import analyze_shots
        from corridorkey.shot_profiles import analyze_job
        from corridorkey.video_io import DraftSourceReader, probe_frame_dir_meta
        job = self.small_job()
        config, _ = job.configurations()
        meta = probe_frame_dir_meta(job.input_path, num_frames=config.num_frames, start_frame=config.start_frame,
            shape_quantum=config.shape_quantum, fps=config.frame_dir_fps, exr_linear=config.frame_dir_linear,
            model_scale=config.model_scale, alpha_mode=config.frame_dir_alpha, source_colour_profile=config.source_colour_profile)
        legacy = DraftSourceReader(job.input_path, meta, is_dir=True)
        old_records, new_records = [], []
        try:
            analyze_shots(meta['num_frames'], legacy.read, emit=old_records.append)
        finally:
            legacy.close()
        def inspect(count, read_source, **kwargs):
            old_emit = kwargs['emit']
            def emit(row):
                new_records.append(dict(row))
                old_emit(row)
            kwargs['emit'] = emit
            return analyze_shots(count, read_source, **kwargs)
        with patch('corridorkey.shot_analysis.analyze_shots', side_effect=inspect):
            analyze_job(job)
        self.assertEqual(json.dumps(old_records, sort_keys=True), json.dumps(new_records, sort_keys=True))

    def test_cancel_after_native_decode_refuses_result_and_leaves_source_unchanged(self):
        from corridorkey.control import CancellationToken, CancelledError
        from corridorkey.shot_profiles import analyze_job
        from corridorkey.video_io import _read_frame_dir_views
        job = self.small_job()
        path = Path(job.input_path) / '0.png'
        before = path.read_bytes()
        token = CancellationToken()
        def cancel(*args, **kwargs):
            result = _read_frame_dir_views(*args, **kwargs)
            token.cancel('cancel after codec returns')
            return result
        with (patch('corridorkey.video_io._read_frame_dir_views', side_effect=cancel),
              self.assertRaisesRegex(CancelledError, 'after codec')):
            analyze_job(job, check=token.check)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.root / 'output').exists())
        self.assertFalse((self.root / 'scratch').exists())

    def test_membership_change_between_scan_and_content_pin_refuses_before_decode(self):
        import cv2
        import numpy as np

        from corridorkey.shot_profiles import _source_pin, analyze_job
        job = self.small_job()
        def changed(*args, **kwargs):
            cv2.imwrite(str(Path(job.input_path) / '1.png'), np.zeros((8, 12, 3), np.uint8))
            return _source_pin(*args, **kwargs)
        with (patch('corridorkey.shot_profiles._source_pin', side_effect=changed),
              patch('cv2.imdecode', side_effect=AssertionError('stale membership must not decode')),
              self.assertRaisesRegex(ValueError, 'membership changed')):
            analyze_job(job)
