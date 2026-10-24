"""Opt-in actual two-anchor disagreement; display diagnostics never change tracking."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from corridorkey.control import CancellationToken, CancelledError, JobControl
from corridorkey.correction_tracking import (
    TrackingAnchor,
    TrackingFrame,
    TrackingRequest,
    track_corrections,
)
from corridorkey.motion import interpolate_alpha
from corridorkey.tracking_overlays import (
    OverlayReceipt,
    build_tracking_overlays,
    read_bundle,
    render_layer,
    write_bundle,
)


class TrackingDisagreementTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((8, 16, 3), .4, np.float32)
        self.left = np.zeros((8, 16), np.float32)
        self.right = self.left.copy()
        self.right[:, 5:11] = .75
        self.calls = []

    def flow(self, previous, following):
        self.calls.append(previous.shape)
        return np.zeros((*previous.shape, 2), np.float32)

    def candidates(self, capture=False, *, same=False, cuts=(), resets=(), direction='both', anchors=2,
                   uncertain_limit: float = 1., policy='request_anchor'):
        request = TrackingRequest('a' * 64, 0, 5, source_start=10, source_decode_identity='b' * 64,
                                  max_uncertain_fraction=uncertain_limit, cuts=cuts, resets=resets, direction=direction,
                                  unreliable_policy=policy)
        authored = (TrackingAnchor.create('left', 0, 0, 'matte', .5, self.left),
                    TrackingAnchor.create('right', 0, 4, 'matte', .5, self.left if same else self.right))[:anchors]
        return list(track_corrections(request, authored,
            lambda absolute: TrackingFrame('a' * 64, absolute, self.rgb),
            estimator=self.flow, estimator_identity='zero-flow-disagreement-fixture', capture_diagnostics=capture))

    def bundle(self, candidate):
        bundle = build_tracking_overlays(candidate, document_identity='c' * 64, decode_identity='b' * 64)
        assert bundle is not None
        return bundle

    def test_motion_capture_retains_actual_immutable_difference_without_pixel_change(self):
        ordinary = interpolate_alpha(self.rgb, self.rgb, self.rgb, self.left, self.right, .5, estimator=self.flow)
        measured = interpolate_alpha(self.rgb, self.rgb, self.rgb, self.left, self.right, .5,
                                     estimator=self.flow, capture_disagreement=True)
        self.assertIsNone(ordinary.disagreement_bytes)
        self.assertIsInstance(measured.disagreement_bytes, bytes)
        assert measured.disagreement_bytes is not None
        np.testing.assert_array_equal(np.frombuffer(measured.disagreement_bytes, '<f4').reshape(8, 16), self.right)
        for name in ('alpha', 'reliability', 'uncertain'):
            np.testing.assert_array_equal(getattr(ordinary, name), getattr(measured, name))
        self.assertEqual(len(self.calls), 8)

    def test_candidates_capture_parity_measured_zero_and_explicit_missing(self):
        ordinary = self.candidates()
        flow_calls = len(self.calls)
        measured = self.candidates(True)
        self.assertEqual(len(self.calls), flow_calls * 2)
        for a, b in zip(ordinary, measured, strict=True):
            for name in ('mask_bytes', 'reliability_bytes', 'uncertain_bytes', 'provenance_json'):
                self.assertEqual(getattr(a, name), getattr(b, name))
            before, after = json.loads(a.diagnostics_json), json.loads(b.diagnostics_json)
            before.pop('motion_seconds', None)
            after.pop('motion_seconds', None)
            self.assertEqual(before, after)
            self.assertIsNone(a.disagreement_bytes)
        for index in (0, 4):
            self.assertIsNone(measured[index].disagreement_bytes)
        self.assertEqual(measured[2].source_frame, 12)
        payload = measured[2].disagreement_bytes
        assert payload is not None
        np.testing.assert_array_equal(np.frombuffer(payload, '<f4').reshape(8, 16), self.right)
        zero = self.candidates(True, same=True)[2]
        assert zero.disagreement_bytes is not None
        self.assertFalse(np.frombuffer(zero.disagreement_bytes, '<f4').any())
        self.assertTrue(all(row.disagreement_bytes is None for row in self.candidates(True, anchors=1)))
        self.assertTrue(all(row.disagreement_bytes is None for row in self.candidates(True, direction='forward')))
        self.assertTrue(all(row.disagreement_bytes is None for row in self.candidates(True, cuts=(2,))))
        self.assertTrue(all(row.disagreement_bytes is None for row in self.candidates(True, resets=(2,))))

    def test_old_two_and_optional_three_map_receipts_are_strictly_bound(self):
        for measured in (False, True):
            bundle = self.bundle(self.candidates(measured)[2])
            self.assertEqual([layer.kind for layer in bundle.maps],
                             ['reliability', 'uncertainty'] + (['disagreement'] if measured else []))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                receipt = write_bundle(bundle, root, remaining_budget=64 * 1024 * 1024)
                self.assertEqual(read_bundle(receipt, root, expected_binding=bundle.binding), bundle)
                if measured:
                    display = render_layer(bundle, 'disagreement')
                    self.assertIn('warped', display.legend['display'].lower())
                    self.assertFalse(display.legend['calibrated'])
                    tampered = receipt.to_dict()
                    tampered['maps'][2]['name'] = '../outside.u8'
                    with self.assertRaises(ValueError):
                        OverlayReceipt(json.dumps(tampered).encode())
                else:
                    with self.assertRaisesRegex(ValueError, 'unavailable'):
                        render_layer(bundle, 'disagreement')
        candidate = self.candidates(True)[2]
        for payload in (b'', np.full((8, 16), np.nan, '<f4').tobytes(), np.full((8, 16), 2., '<f4').tobytes()):
            with self.assertRaises(ValueError):
                self.bundle(replace(candidate, disagreement_bytes=payload))

    def test_measured_failed_target_is_distinct_from_later_uncomputed_stop(self):
        values = self.candidates(True, uncertain_limit=.1, policy='stop')
        self.assertEqual(values[1].status, 'stopped')
        self.assertIsNotNone(values[1].disagreement_bytes)
        self.assertEqual(values[2].status, 'stopped')
        self.assertIsNone(values[2].disagreement_bytes)
        self.assertIsNone(build_tracking_overlays(values[2], document_identity='c' * 64, decode_identity='b' * 64))

    def test_third_map_quota_cancel_proxy_bounds_and_binding_tamper(self):
        candidate = self.candidates(True)[2]
        bundle = build_tracking_overlays(candidate, document_identity='c' * 64, decode_identity='b' * 64, max_side=8)
        assert bundle is not None
        self.assertEqual((bundle.maps[2].width, bundle.maps[2].height), (8, 4))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = write_bundle(bundle, root, remaining_budget=64 * 1024 * 1024)
            changed = receipt.to_dict()
            changed['maps'][2]['bytes'] += 1
            with self.assertRaises(ValueError):
                OverlayReceipt(json.dumps(changed).encode())
            with self.assertRaisesRegex(ValueError, 'binding'):
                read_bundle(receipt, root, expected_binding=replace(bundle.binding, source_frame=13))
            leaf = root / receipt.to_dict()['maps'][2]['name']
            leaf.write_bytes(bytes(len(bundle.maps[2].payload)))
            with self.assertRaises(ValueError):
                read_bundle(receipt, root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ValueError, 'budget'):
                write_bundle(bundle, root, remaining_budget=receipt.bytes_charged - 1)
            self.assertEqual(list(root.iterdir()), [])
            def cancel_after_two():
                if len(list(root.iterdir())) >= 2:
                    raise CancelledError('cancel optional third publication')
            with self.assertRaises(CancelledError):
                write_bundle(bundle, root, remaining_budget=64 * 1024 * 1024, check=cancel_after_two)
            self.assertEqual(list(root.iterdir()), [])
        with self.assertRaises(ValueError):
            self.bundle(replace(candidate, width=4 * 1024 * 1024))
        with self.assertRaises(ValueError):
            build_tracking_overlays(candidate, document_identity='c' * 64, decode_identity='b' * 64, max_side=257)
        token = CancellationToken()
        token.cancel('before capture')
        before = len(self.calls)
        with self.assertRaises(CancelledError):
            interpolate_alpha(self.rgb, self.rgb, self.rgb, self.left, self.right, .5,
                              control=JobControl(token), estimator=self.flow, capture_disagreement=True)
        self.assertEqual(len(self.calls), before)


if __name__ == '__main__':
    unittest.main()
