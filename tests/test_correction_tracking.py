"""Motion-guided corrections: deterministic fixtures, not model/quality evidence."""
import json
import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from corridorkey.control import CancellationToken, CancelledError, JobControl
from corridorkey.correction_tracking import (
    TrackingAnchor,
    TrackingFrame,
    TrackingRequest,
    track_corrections,
)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.source = 'a' * 64
        self.request = TrackingRequest(self.source, 0, 5, source_start=100, direction='forward')
        self.mask = np.zeros((16, 20), np.float32)
        self.mask[5:10, 5:10] = 1
        self.anchor = TrackingAnchor.create('anchor', 1, 0, 'keep', .7, self.mask)
        self.reads = []

    def read(self, frame):
        self.reads.append(frame)
        # Encode displacement in uniform brightness for the injected estimator.
        proxy = np.full((*self.mask.shape, 3), (frame - 100) / 255, np.float32)
        return TrackingFrame(self.source, frame, proxy)

    @staticmethod
    def flow(previous, following):
        result = np.zeros((*previous.shape, 2), np.float32)
        result[..., 0] = float(following[0, 0]) - float(previous[0, 0])
        return result

    def run_track(self, request=None, anchors=None, **kwargs):
        return list(track_corrections(request or self.request, anchors or (self.anchor,), self.read,
                    estimator=kwargs.pop('estimator', self.flow), estimator_identity='fixture-translation-v1', **kwargs))

    def test_actual_cpu_farneback_translates_textured_reference(self):
        # Actual optical flow only; no pretrained inference/production quality claim.
        base = np.random.default_rng(32).random((64, 80, 3), dtype=np.float32)
        mask = np.zeros((64, 80), np.float32)
        mask[24:40, 25:40] = 1
        anchor = TrackingAnchor.create('artist', 0, 0, 'keep', 1, mask)
        request = TrackingRequest(self.source, 0, 4, direction='forward')
        def reader(frame):
            return TrackingFrame(self.source, frame, np.roll(base, frame, axis=1))
        results = list(track_corrections(request, (anchor,), reader))
        predicted = results[-1].mask()
        assert predicted is not None
        expected = np.roll(mask, 3, axis=1)
        intersection = np.sum((predicted > .5) & (expected > .5))
        union = np.sum((predicted > .5) | (expected > .5))
        self.assertGreater(float(intersection / union), .95)
        self.assertEqual(results[-1].status, 'preview')
        self.assertTrue(results[-1].requires_acceptance)

    def test_forward_motion_and_immutable_artist_mask(self):
        before = self.mask.copy()
        results = self.run_track()
        self.assertEqual(results[0].status, 'authored_anchor')
        result = results[-1]
        self.assertEqual(result.source_frame, 104)
        mask = result.mask()
        assert mask is not None
        np.testing.assert_array_equal(mask[5:10, 9:14], 1)
        np.testing.assert_array_equal(mask[5:10, :9], 0)
        np.testing.assert_array_equal(self.mask, before)
        self.assertFalse(mask.flags.writeable)
        self.mask[:] = 0
        self.assertGreater(float(self.anchor.mask().sum()), 0)
        self.assertTrue(all(r.requires_acceptance for r in results))

    def test_backward_motion_and_absolute_reader_indices(self):
        anchor = replace(self.anchor, frame=4)
        results = self.run_track(replace(self.request, direction='backward'), (anchor,))
        self.assertEqual([r.local_frame for r in results], [4, 3, 2, 1, 0])
        mask = results[-1].mask()
        assert mask is not None
        np.testing.assert_array_equal(mask[5:10, 1:6], 1)
        self.assertTrue(all(100 <= index < 105 for index in self.reads))

    def test_bidirectional_anchors_agree_on_translated_subject(self):
        right_mask = np.roll(self.mask, 4, axis=1)
        right = TrackingAnchor.create('second', 1, 4, 'keep', .7, right_mask)
        results = self.run_track(replace(self.request, direction='both'), (self.anchor, right))
        middle = results[2].mask()
        assert middle is not None
        np.testing.assert_array_equal(middle[5:10, 7:12], 1)
        self.assertFalse(json.loads(results[2].diagnostics_json)['reliability_calibrated'])

    def test_cut_reset_segments_require_their_own_anchor(self):
        for name in ('cuts', 'resets'):
            request = replace(self.request, **{name: (2,)})
            results = self.run_track(request)
            self.assertEqual([r.status for r in results[2:]], ['anchor_required'] * 3)
            self.assertNotIn(102, self.reads)
            self.reads.clear()
            allowed = self.run_track(replace(request, allow_cross_boundaries=True))
            self.assertEqual(allowed[-1].status, 'preview')
            self.reads.clear()

    @staticmethod
    def bad_flow(previous, following):
        return np.full((*previous.shape, 2), 1000, np.float32)

    def test_unreliable_stop_or_request_until_additional_anchor(self):
        extra = replace(self.anchor, identifier='new-anchor', frame=3)
        for policy, status in (('stop', 'stopped'), ('request_anchor', 'anchor_required')):
            results = self.run_track(replace(self.request, unreliable_policy=policy),
                                     (self.anchor, extra), estimator=self.bad_flow)
            self.assertEqual(results[1].status, status)
            self.assertIsNone(results[1].mask())
            self.assertEqual(results[2].status, status)
            self.assertEqual(results[3].status, 'authored_anchor')
            self.assertEqual(results[4].status, status)

    def test_explicit_approximation_is_labeled_and_separate(self):
        result = self.run_track(replace(self.request, unreliable_policy='retain_approximation'),
                                estimator=self.bad_flow)[1]
        self.assertEqual(result.status, 'approximation')
        self.assertTrue(json.loads(result.diagnostics_json)['approximation_retained'])
        mask = result.mask()
        assert mask is not None
        np.testing.assert_array_equal(mask, self.anchor.mask())

    def test_identities_change_with_anchors_revision_settings_and_boundaries(self):
        original = self.run_track()[1].identity
        self.assertEqual(self.run_track()[1].identity, original)
        for request, anchors in (
            (self.request, (replace(self.anchor, revision=2),)),
            (replace(self.request, max_uncertain_fraction=.5), (self.anchor,)),
            (replace(self.request, cuts=(4,)), (self.anchor,)),
            (self.request, (self.anchor, replace(self.anchor, identifier='extra', frame=4))),
        ):
            self.assertNotEqual(self.run_track(request, anchors)[1].identity, original)

    def test_streaming_reads_only_current_dependencies_and_records_full_contract(self):
        stream = track_corrections(self.request, (self.anchor,), self.read,
                    estimator=self.flow, estimator_identity='fixture-v1')
        first = next(stream)
        self.assertEqual(self.reads, [100])
        provenance = json.loads(first.provenance_json)
        self.assertEqual(provenance['track']['request']['source_start'], 100)
        self.assertEqual(provenance['track']['anchors'][0]['id'], 'anchor')
        with self.assertRaises(ValueError):
            self.anchor.mask().setflags(write=True)
        stream.close()

    def test_cancel_during_solver_releases_without_preview_publication(self):
        token = CancellationToken()
        def solver(previous, following):
            token.cancel('cancel during flow')
            return self.flow(previous, following)
        stream = track_corrections(self.request, (self.anchor,), self.read,
                    estimator=solver, estimator_identity='fixture-v1', control=JobControl(token))
        next(stream)
        with self.assertRaises(CancelledError):
            next(stream)

    def test_cancelled_stream_stops_before_another_source_read(self):
        token = CancellationToken()
        stream = track_corrections(self.request, (self.anchor,), self.read,
                    estimator=self.flow, estimator_identity='fixture-v1', control=JobControl(token))
        next(stream)
        count = len(self.reads)
        token.cancel('user cancelled')
        with self.assertRaises(CancelledError):
            next(stream)
        self.assertEqual(len(self.reads), count)

    def test_geometry_anchor_frame_and_byte_budgets_precede_reader(self):
        for request in (replace(self.request, max_frames=2), replace(self.request, max_pixels=1),
                        replace(self.request, max_anchor_bytes=1)):
            with self.assertRaises(ValueError):
                self.run_track(request)
        self.assertEqual(self.reads, [])

    def test_rejects_mixed_operations_duplicate_anchors_and_classes(self):
        for anchor in (replace(self.anchor, identifier='other', frame=4, kind='remove'),
                       replace(self.anchor, identifier='other', frame=4, strength=.2),
                       replace(self.anchor, identifier='other', frame=0)):
            with self.assertRaises(ValueError):
                self.run_track(anchors=(self.anchor, anchor))
        invalid_cases: list[Any] = [TrackingRequest, replace(self.request, motion=Any)]
        for invalid in invalid_cases:
            with self.assertRaises(TypeError):
                self.run_track(invalid)
        invalid_boundaries: Any = [2]
        with self.assertRaises(ValueError):
            self.run_track(replace(self.request, cuts=invalid_boundaries))
        with self.assertRaises(ValueError):
            TrackingAnchor.create('trimap', 0, 0, 'trimap', 1, self.mask)

    def test_reader_wrong_source_or_frame_rejects(self):
        for reader in (lambda frame: replace(self.read(frame), source_identity='b' * 64),
                       lambda frame: replace(self.read(frame), source_frame=frame + 1)):
            with self.assertRaises(ValueError):
                list(track_corrections(self.request, (self.anchor,), reader))

    def test_injected_estimator_requires_explicit_identity(self):
        with self.assertRaises(ValueError):
            list(track_corrections(self.request, (self.anchor,), self.read, estimator=self.flow))


if __name__ == '__main__':
    unittest.main()
