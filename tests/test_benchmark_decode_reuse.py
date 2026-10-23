"""Actual tiny sequence decoding/preprocessing, never neural/GPU performance."""
import copy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.benchmark_decode_reuse import (
    _ReadHook,
    run_decode_reuse,
    validate_manifest,
    validate_report,
)
from corridorkey.control import CancelledError


class DecodeReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def run_research(self, **kwargs):
        return run_decode_reuse(artifact_parent=self.root, frame_count=8, shape_hw=(16, 16), chunk=4,
                                stride=2, cache_bytes=16 * 16 * 3 * 4 * 4, **kwargs)

    def test_actual_fixture_matrix_exact_and_useful_vs_thrashing_schedules(self):
        import cv2
        import numpy as np

        progress = []
        report = self.run_research(progress=lambda *args: progress.append(args))
        self.assertEqual(report['status'], 'complete', report['error'])
        validate_report(report)
        self.assertTrue(report['summary']['success'])
        self.assertEqual(len(report['results']), 9)
        self.assertEqual(progress[-1], ('decode_reuse', 9, 9))
        self.assertEqual(report['inference'], {'status': 'not_measured', 'fps': None, 'quality': None})
        for row in report['results']:
            self.assertEqual([p['order'] for p in row['pairs']], [['reference', 'candidate'], ['candidate', 'reference'], ['reference', 'candidate']])
            self.assertTrue(row['exact_equal'])
            for pair in row['pairs']:
                self.assertTrue(pair['validation']['exact_equal'])
                self.assertGreater(pair['validation']['reference']['hash_seconds'], 0)
                self.assertEqual(pair['reference']['retained_bytes'], 0)
                self.assertLessEqual(pair['candidate']['peak_retained_bytes'], report['settings']['cache_bytes'])
                expected_reference = {'overlap': 8, 'sequential_prepasses': 24, 'quick_current_anchor': 11}[row['schedule']]
                self.assertEqual(pair['reference']['decode_calls'], expected_reference)
                if row['schedule'] != 'quick_current_anchor':
                    self.assertEqual(pair['candidate']['decode_calls'], expected_reference)
                    self.assertFalse(row['summary']['useful_decode_reuse'])
                    self.assertIn('model', pair['validation']['reference']['digests'])
                else:
                    self.assertLess(pair['candidate']['decode_calls'], expected_reference)
                    self.assertTrue(row['summary']['useful_decode_reuse'])
                    self.assertIn('rgb', pair['validation']['reference']['digests'])
                self.assertIn('native_rgb', pair['validation']['reference']['digests'])
                self.assertIn('proxy', pair['validation']['reference']['digests'])
        root = Path(report['artifact_dir'])
        png = cv2.imread(str(root / 'uint16_png/0000.png'), cv2.IMREAD_UNCHANGED)
        hdr = cv2.imread(str(root / 'signed_hdr_exr/0000.exr'), cv2.IMREAD_UNCHANGED)
        alpha = cv2.imread(str(root / 'premultiplied_bt2020_exr/0000.exr'), cv2.IMREAD_UNCHANGED)[..., 3]
        self.assertEqual(png.dtype, np.uint16)
        self.assertEqual(hdr.dtype, np.float32)
        self.assertTrue(np.signbit(hdr[0, 0, 0]))
        self.assertLess(float(hdr.min()), 0)
        self.assertGreater(float(hdr.max()), 1)
        self.assertEqual(float(alpha.min()), 0)
        self.assertEqual(float(alpha.max()), 1)
        self.assertTrue(report['fixture_identity']['preserved'])
        self.assertTrue(report['cleanup']['hook_restored'])

    def test_zero_budget_no_cached_copy_or_hits(self):
        report = run_decode_reuse(artifact_parent=self.root, frame_count=8, shape_hw=(16, 16), chunk=4,
                                  stride=2, cache_bytes=0)
        self.assertEqual(report['status'], 'complete', report['error'])
        validate_report(report)
        for row in report['results']:
            for pair in row['pairs']:
                self.assertEqual(pair['candidate']['decode_calls'], pair['reference']['decode_calls'])
                self.assertEqual(pair['candidate']['copy_bytes'], 0)
                self.assertEqual(pair['candidate']['cache_hits'], 0)

    def test_cache_isolation_owned_copies_dtype_info_and_cleanup(self):
        import cv2
        import numpy as np

        from corridorkey import video_io
        path = self.root / 'source.png'
        cv2.imwrite(str(path), np.full((16, 16, 3), 32768, dtype=np.uint16))
        original = video_io.read_image_rgb
        with _ReadHook(video_io, 4096, True, None, paths=(path,)) as hook:
            first = video_io.read_image_rgb(path)
            expected = first.copy()
            first[:] = 99
            info = {}
            second = video_io.read_image_rgb(path, info=info)
            np.testing.assert_array_equal(second, expected)
            second[:] = 42
            np.testing.assert_array_equal(video_io.read_image_rgb(path), expected)
            self.assertEqual(info['decoded_dtype'], 'uint16')
            self.assertEqual(hook.stats['decode_calls'], 1)
            self.assertEqual(hook.stats['cache_hits'], 2)
            with self.assertRaisesRegex(RuntimeError, 'owned'), _ReadHook(video_io, 4096, True, None, paths=(path,)):
                pass
        self.assertIs(video_io.read_image_rgb, original)
        self.assertEqual(hook.stats['retained_bytes'], 0)
        with _ReadHook(video_io, 1, True, None, paths=(path,)) as hook:
            video_io.read_image_rgb(path)
            video_io.read_image_rgb(path)
            self.assertEqual(hook.stats['oversize'], 2)
            self.assertEqual(hook.stats['cache_hits'], 0)

    def test_native_association_and_channel_contracts_never_alias(self):
        import cv2
        import numpy as np

        from corridorkey import video_io
        path = self.root / 'rgba.exr'
        image = np.full((16, 16, 4), 2.0, dtype=np.float32)
        image[..., 3] = .5
        image[0, 0, 3] = 0
        self.assertTrue(cv2.imwrite(str(path), image, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]))
        original = video_io.read_image_rgb
        with _ReadHook(video_io, 8192, True, None, paths=(path,)) as hook:
            ignored = video_io.read_image_rgb(path, alpha_mode='ignore', expected_channels=4)
            unpremult = video_io.read_image_rgb(path, alpha_mode='premultiplied', expected_channels=4)
            self.assertEqual(float(ignored[1, 1, 0]), 2)
            self.assertEqual(float(unpremult[1, 1, 0]), 4)
            self.assertEqual(float(unpremult[0, 0, 0]), 0)
            self.assertEqual(hook.stats['decode_calls'], 2)
            with self.assertRaisesRegex(ValueError, 'channel layout'):
                video_io.read_image_rgb(path, alpha_mode='ignore', expected_channels=3)
        self.assertIs(video_io.read_image_rgb, original)
        np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), image)

    def test_cancel_error_restore_hooks_and_report_partial_artifacts(self):
        from corridorkey import video_io
        original = (video_io.read_image_rgb, video_io.preprocess_frame_dir_path, video_io._read_frame_dir_views)
        def cancel_progress(*unused):
            raise CancelledError('controlled cancel')
        report = self.run_research(progress=cancel_progress)
        self.assertEqual(report['status'], 'cancelled')
        self.assertTrue(Path(report['artifact_dir']).is_dir())
        self.assertTrue(report['cleanup']['hook_restored'])
        validate_report(report)
        with patch('corridorkey.benchmark_decode_reuse._schedule', side_effect=RuntimeError('injected read failure')), self.assertLogs('corridorkey.benchmark_decode_reuse', level='ERROR'):
            failed = self.run_research()
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('injected read failure', failed['error'])
        self.assertTrue(failed['cleanup']['hook_restored'])
        self.assertEqual(original, (video_io.read_image_rgb, video_io.preprocess_frame_dir_path, video_io._read_frame_dir_views))
        validate_report(failed)

    def test_foreign_paths_and_other_thread_bypass_hook_unchanged(self):
        import cv2
        import numpy as np

        from corridorkey import video_io
        owned, foreign = self.root / 'owned.png', self.root / 'foreign.png'
        image = np.full((16, 16, 3), 1024, dtype=np.uint16)
        cv2.imwrite(str(owned), image)
        cv2.imwrite(str(foreign), image + 1)
        expected_owned = video_io.read_image_rgb(owned)
        expected_foreign = video_io.read_image_rgb(foreign)
        results = []
        def work():
            results.append(video_io.read_image_rgb(owned))
            results.append(video_io._read_frame_dir_views(str(owned), exr_linear=False)[0])
        with _ReadHook(video_io, 4096, True, None, paths=(owned,)) as hook:
            video_io.read_image_rgb(owned)
            before = dict(hook.stats)
            np.testing.assert_array_equal(video_io.read_image_rgb(foreign), expected_foreign)
            thread = threading.Thread(target=work)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(hook.stats, before)
        self.assertEqual(len(results), 2)
        np.testing.assert_array_equal(results[0], expected_owned)
        np.testing.assert_array_equal(results[1], expected_owned)

    def test_active_arm_cancellation_restores_hooks(self):
        from corridorkey import benchmark_decode_reuse as research
        from corridorkey import video_io
        original = video_io.read_image_rgb
        def cancel():
            if research._GUARD.locked():
                raise CancelledError('inside active arm')
        report = self.run_research(check=cancel)
        self.assertEqual(report['status'], 'cancelled')
        self.assertTrue(report['cleanup']['hook_restored'])
        self.assertIs(video_io.read_image_rgb, original)
        validate_report(report)

    def test_manifest_bounds_are_pure_and_before_creation(self):
        base = {'schema_version': 1, 'kind': 'decode_reuse', 'artifact_parent': str(self.root)}
        for fields in ({'cache_bytes': True}, {'repetitions': 2}, {'frame_count': 129}, {'shape_hw': [512, 512], 'frame_count': 128},
                       {'stride': 9}, {'unknown': 1}, {'shape_hw': [16, False]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_manifest({**base, **fields})
        with self.assertRaises(ValueError):
            run_decode_reuse(artifact_parent=self.root, frame_count=128, shape_hw=(512, 512))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_report_refuses_fabricated_measurements(self):
        report = self.run_research()
        for change in ('fps', 'parity', 'negative_time', 'missing_row', 'budget', 'empty_validation', 'empty_workload', 'lying_median', 'integer_truth'):
            modified = copy.deepcopy(report)
            if change == 'fps':
                modified['inference']['fps'] = 100
            elif change == 'parity':
                modified['results'][0]['pairs'][0]['validation']['candidate']['digests']['native_rgb'] = '0' * 64
            elif change == 'negative_time':
                modified['results'][0]['pairs'][0]['candidate']['elapsed_seconds'] = -1
            elif change == 'missing_row':
                modified['results'].pop()
            elif change == 'empty_validation':
                for side in ('reference', 'candidate'):
                    modified['results'][0]['pairs'][0]['validation'][side].update(digests={}, observations={})
            elif change == 'empty_workload':
                for side in ('reference', 'candidate'):
                    for key in ('decode_calls', 'read_requests', 'preprocess_calls', 'cache_hits', 'cache_misses'):
                        modified['results'][0]['pairs'][0][side][key] = 0
            elif change == 'lying_median':
                modified['results'][0]['summary']['median_candidate_seconds'] = 10000
            elif change == 'integer_truth':
                modified['fixture_identity']['preserved'] = 1
            else:
                modified['results'][0]['pairs'][0]['candidate']['peak_retained_bytes'] = modified['settings']['cache_bytes'] + 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_report(modified)


if __name__ == '__main__':
    unittest.main()
