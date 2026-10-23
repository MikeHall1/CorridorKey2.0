import importlib.util
import unittest
from typing import TYPE_CHECKING

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation_temporal_sequence import (
        TemporalAccumulator,
        compare_temporal_stream,
    )


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class TemporalStreamTests(unittest.TestCase):
    def test_aggregates_pair_populations_and_spatial_frames_once(self):
        frames = [
            {'frame': 0, 'candidate': [[0., 0.]], 'reference': [[0., 0.]]},
            {'frame': 1, 'candidate': [[1., 1.]], 'reference': [[0., 0.]],
             'backward_flow': np.zeros((1, 2, 2)), 'visibility': [[True, True]]},
            {'frame': 2, 'candidate': [[0.5, 0.5]], 'reference': [[0., 0.]],
             'backward_flow': np.zeros((1, 2, 2)), 'visibility': [[True, False]]}]
        report = compare_temporal_stream(iter(frames))['candidate']
        self.assertEqual(report['spatial']['frames'], 3)
        self.assertEqual(report['spatial']['mae']['population'], 6)
        self.assertEqual(report['spatial']['mae']['value'], 0.5)
        self.assertAlmostEqual(report['temporal_residual_mae']['value'], 2.5 / 3)
        self.assertEqual(report['temporal_residual_mae']['population'], 3)

    def test_owned_previous_arrays_survive_generator_buffer_reuse(self):
        reused = np.zeros((1, 1))

        def frames():
            yield {'frame': 0, 'candidate': reused, 'reference': [[0.]]}
            reused[:] = 1
            yield {'frame': 1, 'candidate': reused, 'reference': [[0.]],
                   'backward_flow': np.zeros((1, 1, 2)), 'visibility': [[True]]}

        report = compare_temporal_stream(frames())['candidate']
        self.assertEqual(report['temporal_residual_mae']['value'], 1)

    def test_cuts_and_missing_pairs_do_not_score_zero(self):
        report = compare_temporal_stream([{'frame': 0, 'candidate': [[0.]], 'reference': [[0.]]},
                                           {'frame': 1, 'candidate': [[1.]], 'reference': [[0.]], 'cut': True}])['candidate']
        self.assertEqual(report['cut_pairs'], 1)
        self.assertIsNone(report['temporal_residual_mae']['value'])
        self.assertEqual(report['status'], 'partial')
        with self.assertRaisesRegex(ValueError, 'empty'):
            compare_temporal_stream([])

    def test_gap_flow_requires_explicit_preceding_selected_frame(self):
        accumulator = TemporalAccumulator()
        accumulator.add(0, [[0.]], [[0.]])
        with self.assertRaisesRegex(ValueError, 'nonconsecutive'):
            accumulator.add(3, [[1.]], [[0.]], backward_flow=np.zeros((1, 1, 2)), visibility=[[True]])
        self.assertEqual(accumulator.snapshot()['frames'], 1)
        accumulator.add(3, [[1.]], [[0.]], backward_flow=np.zeros((1, 1, 2)), visibility=[[True]], flow_previous_frame=0)
        self.assertEqual(accumulator.snapshot()['temporal_residual_mae']['value'], 1)

    def test_baseline_deltas_and_cooperative_cancellation(self):
        frames = [{'frame': 0, 'candidate': [[0.]], 'baseline': [[1.]], 'reference': [[0.]]},
                  {'frame': 1, 'candidate': [[0.]], 'baseline': [[0.]], 'reference': [[0.]],
                   'backward_flow': np.zeros((1, 1, 2)), 'visibility': [[True]]}]
        report = compare_temporal_stream(frames)
        self.assertEqual(report['candidate_minus_baseline']['spatial_mae'], -0.5)
        self.assertEqual(report['candidate_minus_baseline']['temporal_residual_mae'], -1)
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            compare_temporal_stream(frames, check=lambda: (_ for _ in ()).throw(RuntimeError('cancel')))

    def test_singleton_settings_cut_and_link_types_are_validated_upfront(self):
        with self.assertRaisesRegex(ValueError, 'convention'):
            TemporalAccumulator(flow_convention='wrong')
        for threshold in (-1, float('inf'), 10**1000):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                TemporalAccumulator(consistency_pixels=threshold)
        import json

        accumulator = TemporalAccumulator()
        invalid = json.loads('{"cut": "wrong"}')
        with self.assertRaisesRegex(TypeError, 'cut'):
            accumulator.add(0, [[0.]], **invalid)
        with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
            accumulator.add(0, [[0.]], flow_previous_frame=False)

    def test_snapshots_cannot_mutate_internal_geometry(self):
        accumulator = TemporalAccumulator()
        accumulator.add(0, [[0.]], [[0.]])
        result = accumulator.snapshot()
        result['shape_hw'][0] = 99
        result['spatial']['shape_hw'][1] = 99
        accumulator.add(1, [[0.]], [[0.]])
        self.assertEqual(accumulator.snapshot()['shape_hw'], [1, 1])
        self.assertEqual(accumulator.snapshot()['frames'], 2)

    def test_candidate_and_baseline_geometry_must_match_even_without_references(self):
        with self.assertRaisesRegex(ValueError, 'baseline temporal geometry'):
            compare_temporal_stream([{'frame': 0, 'candidate': [[0.]], 'baseline': [[0., 0.]]}])
