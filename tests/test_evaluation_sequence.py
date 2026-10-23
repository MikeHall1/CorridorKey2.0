import importlib.util
import unittest
from typing import TYPE_CHECKING

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation import alpha_metrics
    from corridorkey.evaluation_sequence import AlphaAccumulator, compare_stream


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class StreamTests(unittest.TestCase):
    def test_stream_matches_batch_with_unequal_mask_populations(self):
        candidate = np.array([[[0., 0.]], [[0.5, 1.]]])
        target = np.array([[[1., 1.]], [[0., 0.]]])
        mask = np.array([[[True, True]], [[True, False]]])
        report = compare_stream(iter(candidate), iter(target), masks=iter(mask))['candidate']
        batch = alpha_metrics(candidate, target, mask=mask)
        for name in ('mae', 'normalized_sad', 'sad', 'boundary_mae', 'forward_gradient_mae'):
            self.assertEqual(report[name]['value'], batch[name]['value'])
            self.assertEqual(report[name]['population'], batch[name]['population'])
        self.assertAlmostEqual(report['mae']['value'], 2.5 / 3)
        self.assertEqual(report['frames'], 2)

    def test_missing_reference_and_empty_masks_stay_unavailable(self):
        report = compare_stream([[[0.]]], baseline_frames=[[[1.]]])
        self.assertIsNone(report['candidate_minus_baseline']['mae'])
        self.assertEqual(report['candidate']['reason'], 'reference missing')
        report = compare_stream([[[0.]]], [[[1.]]], masks=[np.zeros((1, 1), dtype=bool)])
        self.assertEqual(report['candidate']['reason'], 'evaluation mask is empty')

    def test_lengths_geometry_and_empty_sequences_are_rejected(self):
        for candidate, reference in (([], []), ([[[0]]], []), ([], [[[0]]]),
                                     ([[[0]], [[0, 0]]], [[[0]], [[0, 0]]])):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare_stream(candidate, reference)
        with self.assertRaisesRegex(ValueError, 'geometry differs'):
            compare_stream([[[0]]], baseline_frames=[[[0, 0]]])
        accumulator = AlphaAccumulator()
        accumulator.add([[0]])
        with self.assertRaisesRegex(ValueError, 'availability'):
            accumulator.add([[0]], [[0]])

    def test_single_pass_and_cooperative_cancellation(self):
        class Cancelled(Exception):
            pass

        pulled = []

        def frames():
            for i in range(100):
                pulled.append(i)
                yield [[0.]]

        def check():
            if len(pulled) == 3:
                raise Cancelled()

        with self.assertRaises(Cancelled):
            compare_stream(frames(), check=check)
        self.assertEqual(pulled, [0, 1, 2])

    def test_accumulator_does_not_retain_frame_arrays(self):
        import weakref

        accumulator = AlphaAccumulator()
        references = []
        def add_frame():
            frame = np.ones((4, 4))
            references.append(weakref.ref(frame))
            accumulator.add(frame, np.zeros((4, 4)))
        for _ in range(20):
            add_frame()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertEqual(accumulator.snapshot()['mae']['value'], 1)
