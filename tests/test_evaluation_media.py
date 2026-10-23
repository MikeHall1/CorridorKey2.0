import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_CV = importlib.util.find_spec('cv2') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation_media import (
        evaluate_sequence_manifest,
        load_sequence_manifest,
        read_alpha_frame,
    )


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class MediaEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'evaluation.json'
        self.data = {'schema_version': 1, 'kind': 'alpha_sequence_evaluation', 'fps': 24,
                     'reference_provenance': {'author': 'test generator', 'origin': 'unit fixture',
                                              'permitted_use': 'local synthetic test', 'split': 'synthetic'},
                     'frames': []}
        for frame in (0, 2):
            entry: dict[str, int | str] = {'frame': frame}
            for role, values in (('candidate', [[0., 0.5]]), ('reference', [[0., 1.]]),
                                 ('baseline', [[1., 0.]])):
                name = f'{role}-{frame}.npy'
                np.save(self.root / name, np.array(values))
                entry[role] = name
            self.data['frames'].append(entry)
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def test_manifest_streams_frames_and_records_provenance_and_identities(self):
        report = evaluate_sequence_manifest(self.path)
        self.assertEqual(report['frame_ids'], [0, 2])
        self.assertEqual(report['candidate']['frames'], 2)
        self.assertEqual(report['candidate']['mae']['value'], 0.25)
        self.assertEqual(report['candidate_minus_baseline']['mae'], -0.75)
        self.assertEqual(len(report['assets']), 6)
        self.assertEqual(report['reference_provenance']['split'], 'synthetic')
        np.save(self.root / 'candidate-2.npy', np.ones((1, 2)))
        after = evaluate_sequence_manifest(self.path)
        self.assertNotEqual(report['assets'][-3]['sha256'], after['assets'][-3]['sha256'])
        self.assertNotEqual(report['candidate']['mae'], after['candidate']['mae'])

    def test_manifest_requires_alignment_known_fields_and_reference_provenance(self):
        bad = []
        for update in ({'fps': 0}, {'fps': True}, {'fps': 10**1000}, {'schema_version': True}, {'boundary_radius': -1}, {'unknown': 1},
                       {'reference_provenance': {}}, {'channels': {'candidate': 'rgb'}}, {'frames': []}):
            bad.append({**self.data, **update})
        broken = json.loads(json.dumps(self.data))
        broken['frames'][1].pop('baseline')
        bad.append(broken)
        broken = json.loads(json.dumps(self.data))
        broken['frames'][1]['frame'] = 0
        bad.append(broken)
        for value in bad:
            self.path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                load_sequence_manifest(self.path)
        self.path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_sequence_manifest(self.path)

    def test_missing_reference_is_unavailable_and_corrupt_assets_fail(self):
        for entry in self.data['frames']:
            entry.pop('reference')
        self.data.pop('reference_provenance')
        self.save()
        report = evaluate_sequence_manifest(self.path)
        self.assertIsNone(report['candidate']['mae']['value'])
        self.assertIsNone(report['candidate_minus_baseline']['mae'])
        (self.root / 'candidate-0.npy').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            evaluate_sequence_manifest(self.path)

    def test_masks_are_binary_and_layout_is_explicit(self):
        path = self.root / 'rgba.npy'
        value = np.zeros((2, 2, 4))
        value[..., 3] = 0.25
        np.save(path, value)
        with self.assertRaisesRegex(ValueError, 'layout'):
            read_alpha_frame(path)
        decoded, _ = read_alpha_frame(path, channel='alpha')
        np.testing.assert_array_equal(decoded, np.full((2, 2), 0.25))
        with self.assertRaisesRegex(ValueError, 'binary'):
            read_alpha_frame(path, channel='alpha', mask=True)
        np.save(path, np.zeros((1, 1, 3)))
        read_alpha_frame(path, channel='replicated_gray')
        np.save(path, np.array([[[0., 0.2, 0.]]]))
        with self.assertRaisesRegex(ValueError, 'channels differ'):
            read_alpha_frame(path, channel='replicated_gray')

    def test_changed_read_and_cancellation_propagate(self):
        with patch('corridorkey.evaluation_media.file_sha256', side_effect=['before', 'after']), \
                self.assertRaisesRegex(ValueError, 'changed'):
            read_alpha_frame(self.root / 'candidate-0.npy')
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            evaluate_sequence_manifest(self.path, check=lambda: (_ for _ in ()).throw(RuntimeError('cancel')))

    @unittest.skipUnless(HAS_CV, 'requires OpenCV')
    def test_actual_uint16_png_and_float_exr_decode_preserve_alpha_levels(self):
        import os

        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
        import cv2

        for suffix, value in (('.png', np.array([[0, 1, 32768, 65535]], dtype=np.uint16)),
                              ('.exr', np.array([[0., 0.001, 0.3, 1.]], dtype=np.float32))):
            path = self.root / f'alpha{suffix}'
            self.assertTrue(cv2.imwrite(str(path), value))
            decoded, identity = read_alpha_frame(path)
            expected = value.astype(float) / 65535 if suffix == '.png' else value
            np.testing.assert_array_equal(decoded, expected)
            self.assertEqual(identity['shape'], [1, 4])

    def test_real_cli_protects_inputs_and_preserves_report_on_failure(self):
        output = self.root / 'report.json'
        command = [sys.executable, '-m', 'corridorkey.evaluation_media', str(self.path), '--output']
        result = subprocess.run([*command, str(output)], check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = output.read_bytes()
        result = subprocess.run([*command, str(self.path)], check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        (self.root / 'candidate-2.npy').write_bytes(b'corrupt')
        result = subprocess.run([*command, str(output)], check=False, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), before)

    @unittest.skipUnless(HAS_CV, 'requires OpenCV')
    def test_native_decoder_failure_is_a_contextual_data_error_without_fallback(self):
        import cv2

        path = self.root / 'corrupt.png'
        path.write_bytes(b'local invalid raster fixture')
        with patch('cv2.imread', side_effect=cv2.error('decoder fixture failure')), \
                self.assertRaisesRegex(ValueError, 'alpha decoder failed') as caught:
            read_alpha_frame(path)
        self.assertIsInstance(caught.exception.__cause__, cv2.error)
