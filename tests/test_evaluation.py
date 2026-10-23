import importlib.util
import unittest
from typing import TYPE_CHECKING

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation import alpha_metrics


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class AlphaMetricTests(unittest.TestCase):
    def test_analytic_error_and_gradient_populations(self):
        predicted = np.array([[0, 0.5], [1, 0.5]])
        reference = np.array([[0, 0], [1, 1]])
        result = alpha_metrics(predicted, reference, boundary_radius=0)
        self.assertEqual(result['mae']['value'], 0.25)
        self.assertEqual(result['normalized_sad'], result['mae'])
        self.assertEqual(result['sad']['value'], 1)
        self.assertEqual(result['mae']['population'], 4)
        self.assertEqual(result['boundary_mae']['population'], 4)
        self.assertEqual(result['forward_gradient_mae']['population'], 4)
        self.assertEqual(result['forward_gradient_mae']['value'], 0.5)

    def test_identity_and_missing_reference_are_different(self):
        value = np.array([[0, 0.3, 1]])
        self.assertEqual(alpha_metrics(value, value)['mae']['value'], 0)
        missing = alpha_metrics(value)
        self.assertEqual(missing['status'], 'unavailable')
        for name in ('mae', 'sad', 'normalized_sad', 'boundary_mae', 'forward_gradient_mae'):
            self.assertIsNone(missing[name]['value'])

    def test_masks_weight_pixels_and_exclude_invalid_gradient_pairs(self):
        predicted = np.zeros((2, 1, 2))
        target = np.ones_like(predicted)
        mask = np.array([[[True, True]], [[True, False]]])
        result = alpha_metrics(predicted, target, mask=mask)
        self.assertEqual(result['mae']['value'], 1)
        self.assertEqual(result['sad']['value'], 3)
        self.assertEqual(result['mae']['population'], 3)
        self.assertEqual(result['forward_gradient_mae']['population'], 1)
        self.assertEqual(result['forward_gradient_mae']['value'], 0)
        self.assertIsNone(result['boundary_mae']['value'])
        empty = alpha_metrics(predicted, target, mask=np.zeros_like(mask))
        self.assertEqual(empty['status'], 'unavailable')
        self.assertIsNone(empty['mae']['value'])

    def test_boundary_dilation_is_spatial_only_and_clips_at_image_edges(self):
        target = np.array([[[0, 0, 0.5, 0, 0]], [[0, 0, 0, 0, 0]]])
        result = alpha_metrics(target, target, boundary_radius=1)
        self.assertEqual(result['boundary_mae']['population'], 5)
        self.assertEqual(alpha_metrics([[0]], [[0]])['forward_gradient_mae']['population'], 0)

    def test_rejects_invalid_range_shape_mask_and_radius(self):
        for value in ([[float('nan')]], [[float('inf')]], [[-0.1]], [[255]], [], ['bad']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                alpha_metrics(value, [[0]])
        for mask in ([[1]], np.ones((2, 2), dtype=bool)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                alpha_metrics([[0]], [[0]], mask=mask)
        for radius in (True, -1, 65):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                alpha_metrics([[0]], [[0]], boundary_radius=radius)
        with self.assertRaisesRegex(ValueError, 'shapes must match'):
            alpha_metrics([[0]], [[0, 0]])

    def test_comparison_uses_same_mask_and_missing_inputs_stay_unavailable(self):
        from corridorkey.evaluation import compare_alpha

        result = compare_alpha([[0, 0.5]], [[0, 1]], baseline=[[1, 0]])
        self.assertEqual(result['candidate_minus_baseline']['mae'], -0.75)
        self.assertEqual(result['candidate']['mae']['population'],
                         result['baseline']['mae']['population'])
        self.assertEqual(compare_alpha([[0]], baseline=[[1]])['comparison_status'], 'unavailable')
        self.assertIsNone(compare_alpha([[0]], [[1]])['candidate_minus_baseline']['mae'])
        with self.assertRaisesRegex(ValueError, 'shapes must match'):
            compare_alpha([[0]], baseline=[[0, 0]])

    def test_real_cli_records_local_identities_and_protects_inputs(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        from corridorkey.identities import file_sha256

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate, reference, baseline = [root / f'{name}.npy'
                                              for name in ('candidate', 'reference', 'baseline')]
            for path, value in ((candidate, [[0, 0.5]]), (reference, [[0, 1]]), (baseline, [[1, 0]])):
                np.save(path, np.array(value))
            report = root / 'report.json'
            command = [sys.executable, '-m', 'corridorkey.evaluation', '--candidate', str(candidate),
                       '--reference', str(reference), '--baseline', str(baseline)]
            process = subprocess.run([*command, '--output', str(report)], check=False,
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result['candidate_minus_baseline']['mae'], -0.75)
            before = file_sha256(candidate)
            self.assertEqual(result['assets']['candidate']['sha256'], before)
            process = subprocess.run([*command, '--output', str(candidate)], check=False,
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertEqual(file_sha256(candidate), before)
            self.assertEqual(json.loads(report.read_text()), result)

    def test_file_evaluator_disallows_pickled_objects_and_missing_paths(self):
        import tempfile
        from pathlib import Path

        from corridorkey.evaluation import evaluate_files

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'objects.npy')
            np.save(path, np.array([object()], dtype=object))
            with self.assertRaises(ValueError):
                evaluate_files(path)
            with path.open('wb') as stream:
                np.savez(stream, alpha=np.zeros((1, 1)))
            with self.assertRaises(ValueError):
                evaluate_files(path)
            with self.assertRaises(FileNotFoundError):
                evaluate_files(Path(folder, 'absent.npy'))
