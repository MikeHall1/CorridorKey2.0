import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_CV = importlib.util.find_spec('cv2') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation_colour_media import (
        evaluate_colour_manifest,
        load_colour_manifest,
        read_rgb_frame,
    )


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class ColourMediaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'colour.json'
        entry: dict[str, int | str] = {'frame': 0}
        for role, value in [('candidate_rgb', np.full((1, 2, 3), 0.25)),
                            ('candidate_alpha', np.ones((1, 2))), ('reference_rgb', np.zeros((1, 2, 3))),
                            ('reference_alpha', np.ones((1, 2))), ('baseline_rgb', np.ones((1, 2, 3))),
                            ('baseline_alpha', np.ones((1, 2))), ('foreground_mask', np.ones((1, 2), dtype=bool))]:
            np.save(self.root / f'{role}.npy', value)
            entry[role] = f'{role}.npy'
        self.data = {'schema_version': 1, 'kind': 'colour_sequence_evaluation', 'colour_space': 'linear_srgb_bt709',
                     'associations': {'candidate': 'straight', 'reference': 'straight', 'baseline': 'straight'},
                     'fps': 24, 'background': [0., 0., 0.], 'reference_provenance': {
                         'author': 'fixture', 'origin': 'synthetic', 'permitted_use': 'local test', 'split': 'synthetic'},
                     'frames': [entry]}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def test_manifest_evaluates_explicit_linear_frames_and_identities(self):
        report = evaluate_colour_manifest(self.path)
        self.assertEqual(report['candidate']['foreground_mae']['value'], 0.25)
        self.assertEqual(report['candidate']['composite_mae']['value'], 0.25)
        self.assertEqual(report['candidate_minus_baseline']['foreground_mae'], -0.75)
        self.assertEqual(len(report['assets']), 7)
        self.assertEqual(report['reference_provenance']['split'], 'synthetic')
        np.save(self.root / 'candidate_rgb.npy', np.full((1, 2, 3), 0.5))
        after = evaluate_colour_manifest(self.path)
        self.assertNotEqual(report['assets'], after['assets'])

    def test_report_retains_composite_evaluation_contract(self):
        first = evaluate_colour_manifest(self.path)['evaluation_contract']
        self.assertEqual(first['constant_background'], [0., 0., 0.])
        self.assertEqual(first['rgb_layouts']['candidate_rgb'], 'rgb')
        self.assertEqual(first['alpha_channels']['candidate_alpha'], 'gray')
        self.assertEqual(first['unpremultiply_epsilon'], 1e-6)
        self.data['background'] = [1., 0., 0.]
        self.save()
        second = evaluate_colour_manifest(self.path)['evaluation_contract']
        self.assertNotEqual(first, second)

    def test_schema_rejects_ambiguous_contracts_pairs_and_provenance(self):
        for change in ({'colour_space': 'srgb'}, {'fps': True}, {'fps': 10**1000},
                       {'associations': {'candidate': 'straight'}}, {'reference_provenance': {}},
                       {'unpremultiply_epsilon': -1}, {'unknown': 1}):
            self.path.write_text(json.dumps({**self.data, **change}))
            with self.subTest(change=change), self.assertRaises((TypeError, ValueError)):
                load_colour_manifest(self.path)
        self.data['frames'][0].pop('reference_alpha')
        self.save()
        with self.assertRaisesRegex(ValueError, 'pairs'):
            load_colour_manifest(self.path)

    def test_missing_reference_yields_unavailable_and_masks_are_binary(self):
        for role in ('reference_alpha', 'reference_rgb'):
            self.data['frames'][0].pop(role)
        self.data.pop('reference_provenance')
        self.save()
        self.assertEqual(evaluate_colour_manifest(self.path)['status'], 'unavailable')
        np.save(self.root / 'foreground_mask.npy', np.full((1, 2), 0.25))
        with self.assertRaisesRegex(ValueError, 'binary'):
            evaluate_colour_manifest(self.path)

    @unittest.skipUnless(HAS_CV, 'requires OpenCV')
    def test_actual_raster_rgb_order_uint16_levels_and_signed_hdr_exr(self):
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
        import cv2

        for suffix, value in [('.png', np.array([[[1, 32768, 65535]]], dtype=np.uint16)),
                              ('.exr', np.array([[[-2., 0.5, 4.]]], dtype=np.float32))]:
            path = self.root / f'rgb{suffix}'
            self.assertTrue(cv2.imwrite(str(path), value))
            decoded, _ = read_rgb_frame(path)
            expected = value[..., ::-1] / 65535 if suffix == '.png' else value[..., ::-1]
            np.testing.assert_array_equal(decoded, expected)
        rgba = self.root / 'rgba.npy'
        np.save(rgba, np.ones((1, 1, 4)))
        with self.assertRaisesRegex(ValueError, 'layout'):
            read_rgb_frame(rgba)
        self.assertEqual(read_rgb_frame(rgba, layout='rgba')[0].shape, (1, 1, 3))

    def test_real_cli_protects_assets_and_existing_report_on_failure(self):
        output = self.root / 'report.json'
        command = [sys.executable, '-m', 'corridorkey.evaluation_colour_media', str(self.path), '--output']
        process = subprocess.run([*command, str(output)], check=False, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        before = output.read_bytes()
        process = subprocess.run([*command, str(self.path)], check=False, capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        (self.root / 'candidate_rgb.npy').write_bytes(b'corrupt')
        process = subprocess.run([*command, str(output)], check=False, capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(output.read_bytes(), before)
