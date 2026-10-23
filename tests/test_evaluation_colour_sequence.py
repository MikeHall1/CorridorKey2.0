import importlib.util
import unittest
from typing import TYPE_CHECKING

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation_colour_sequence import (
        ColourAccumulator,
        compare_colour_stream,
    )


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class ColourStreamTests(unittest.TestCase):
    def evaluate(self, frames):
        return compare_colour_stream(frames, colour_space='linear_srgb_bt709', candidate_association='straight',
                                     reference_association='straight', baseline_association='straight')

    def frame(self, value, mask):
        return {'candidate_rgb': np.full((1, 2, 3), value), 'candidate_alpha': np.ones((1, 2)),
                'reference_rgb': np.zeros((1, 2, 3)), 'reference_alpha': np.ones((1, 2)),
                'foreground_mask': np.array([mask], dtype=bool), 'background': [0., 0., 0.]}

    def test_accumulates_channels_and_alpha_without_averaging_frame_means(self):
        first = self.frame(1., [True, True])
        second = self.frame(0.5, [True, False])
        result = self.evaluate(iter([first, second]))['candidate']
        self.assertAlmostEqual(result['foreground_mae']['value'], 2.5 / 3)
        self.assertEqual(result['foreground_mae']['population'], 9)
        self.assertEqual(result['composite_mae']['value'], 0.75)
        self.assertEqual(result['alpha']['mae']['value'], 0)
        self.assertEqual(result['frames'], 2)

    def test_missing_roles_empty_sequences_and_geometry_changes_fail(self):
        first = self.frame(1., [True, True])
        second = dict(first)
        second.pop('background')
        with self.assertRaisesRegex(ValueError, 'roles'):
            self.evaluate([first, second])
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.evaluate([])
        second = self.frame(1., [True, True])
        second['candidate_rgb'] = np.ones((2, 2, 3))
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.evaluate([first, second])

    def test_baseline_uses_identical_masks_and_background(self):
        frame = self.frame(0.25, [True, False])
        frame.update(baseline_rgb=np.ones((1, 2, 3)), baseline_alpha=np.ones((1, 2)))
        result = self.evaluate([frame])
        self.assertEqual(result['candidate_minus_baseline']['foreground_mae'], -0.75)
        self.assertEqual(result['candidate_minus_baseline']['composite_mae'], -0.75)
        self.assertEqual(result['status'], 'measured')

    def test_partial_availability_and_array_release(self):
        import weakref

        accumulator = ColourAccumulator(colour_space='linear_srgb_bt709', candidate_association='straight',
                                        reference_association='straight')
        refs = []

        def add():
            frame = np.ones((1, 1, 3))
            refs.append(weakref.ref(frame))
            accumulator.add(frame, [[1.]], np.zeros((1, 1, 3)), [[1.]], background=[0., 0., 0.])

        for _ in range(5):
            add()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(accumulator.snapshot()['status'], 'partial')
        self.assertIsNone(accumulator.snapshot()['foreground_mae']['value'])

    def test_hdr_accumulation_overflow_rejects_frame_without_corrupting_prior_state(self):
        accumulator = ColourAccumulator(colour_space='linear_srgb_bt709', candidate_association='straight',
                                        reference_association='straight')
        for _ in range(5):
            accumulator.add(np.full((1, 1, 3), 1e307), [[1.]], np.zeros((1, 1, 3)), [[1.]],
                            foreground_mask=[[True]])
        with self.assertRaisesRegex(ValueError, 'accumulation overflow'):
            accumulator.add(np.full((1, 1, 3), 1e307), [[1.]], np.zeros((1, 1, 3)), [[1.]],
                            foreground_mask=[[True]])
        snapshot = accumulator.snapshot()
        self.assertEqual(snapshot['frames'], 5)
        self.assertEqual(snapshot['foreground_mae']['population'], 15)
        self.assertTrue(np.isfinite(snapshot['foreground_mae']['value']))
