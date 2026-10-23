"""Persisted derived coverage, ordered once-only composition and admission."""
import copy
import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.corrections import (
    CorrectionDocument,
    CorrectionHistory,
    canonical,
    invalidated_ranges,
    load_corrections,
    load_tracked_asset,
    save_corrections,
    track_binding,
)

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'cv2'))


@unittest.skipUnless(STACK, 'requires optional CPU image adapters')
class TrackedCorrectionTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.shape = (2, 4)
        self.left = np.zeros(self.shape, np.float32)
        self.left[:, 0] = 1
        self.right = np.zeros(self.shape, np.float32)
        self.right[:, 2] = 1
        self.decode = {'contract': 'corridorkey-native-rgb-v1', 'source_kind': 'frames',
                       'frame_dir_linear': False, 'frame_dir_alpha': 'ignore', 'source_colour_profile': 'legacy',
                       'linear_brightness': 1., 'linear_contrast': 1., 'linear_contrast_pivot': .18}
        self.tracking = {'kernel_version': 'correction-tracking-v1', 'direction': 'both',
                         'unreliable_policy': 'request_anchor', 'max_uncertain_fraction': .25,
                         'motion': {'max_side': 640, 'consistency_pixels': 1.5, 'photometric_scale': .15,
                                    'min_reliability': .2, 'max_disagreement': .25},
                         'max_frames': 300, 'max_pixels': 4 * 1024 * 1024,
                         'max_anchor_bytes': 64 * 1024 * 1024, 'max_anchors': 32,
                         'estimator_identity': 'fixture-persisted-flow-not-actual-inference'}
        self.data = {'schema_version': 1, 'revision': 0,
                     'source': {'sha256': 'a' * 64, 'start_frame': 5, 'frame_count': 5,
                                'fps': 24., 'width': 4, 'height': 2},
                     'shots': [[0, 5]], 'entries': [self.entry('anchor', self.left, frame=1)]}
        self.meta = {'native_hw': self.shape, 'start_frame': 5, 'num_frames': 5, 'native_fps': 24.}

    def asset(self, name, mask, *, half=False):
        import cv2
        path = self.root / name
        flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT] if path.suffix == '.exr' else []
        self.assertTrue(cv2.imwrite(str(path), mask, flags))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def entry(self, identifier, mask, *, frame=1, kind='keep', strength=.5):
        import numpy as np
        return {'id': identifier, 'revision': 0, 'frame': frame, 'kind': kind,
                'strength': strength, 'direction': 'both', 'range': [0, 5],
                'asset': self.asset(identifier + '.png', np.rint(mask * 65535).astype(np.uint16))}

    def tracked(self, *, data=None, anchor_ids=None, span=(1, 4), cuts=None, direction='both'):
        data = copy.deepcopy(self.data if data is None else data)
        anchor_ids = ['anchor'] if anchor_ids is None else anchor_ids
        settings = {**self.tracking, 'direction': direction}
        binding = track_binding(data, anchor_ids, cuts=cuts or [], decode=self.decode, tracking=settings)
        track = {'id': 'tracked', 'owner_id': anchor_ids[0], 'anchor_ids': anchor_ids,
                 'range': list(span), 'direction': direction, 'binding': binding,
                 'binding_sha256': hashlib.sha256(canonical(binding).encode()).hexdigest(),
                 'frames': [{'frame': frame, 'asset': self.asset(f'track-{frame}.exr', self.left if frame == 1 else self.right)}
                            for frame in range(*span)],
                 'observations': [{'frame': frame, 'source_frame': data['source']['start_frame'] + frame,
                                   'status': 'authored_anchor' if frame in {entry['frame'] for entry in binding['anchors']} else 'preview',
                                   'candidate_identity': 'b' * 64, 'source_proxy_sha256': {str(frame): 'c' * 64},
                                   'derived_mask_sha256': hashlib.sha256((self.left if frame == 1 else self.right).astype('<f4').tobytes()).hexdigest(),
                                   'diagnostics': {'reliability_calibrated': False, 'uncertain_fraction': .1}}
                                  for frame in range(*span)]}
        return {**data, 'schema_version': 3, 'revision': data['revision'] + 1, 'tracks': [track]}

    def rasters(self, data):
        from corridorkey.correction_rasters import CorrectionRasters
        return CorrectionRasters(CorrectionDocument.parse(data, self.root), self.meta)

    def test_final_replacement_moves_mask_once_at_original_slot_with_later_override(self):
        import numpy as np
        self.data['entries'].append(self.entry('later', np.ones(self.shape, np.float32), kind='remove'))
        data = self.tracked()
        before = {path: path.read_bytes() for path in self.root.iterdir()}
        rasters = self.rasters(data)
        try:
            actual = rasters.apply(np.full(self.shape, .2, np.float32), 2)
            expected = np.full(self.shape, .1, np.float32)
            expected[:, 2] = .3
            np.testing.assert_allclose(actual, expected)
            self.assertLessEqual(len(rasters._cache), 2)
            # Outside accepted span the original static range remains intact.
            outside = rasters.apply(np.zeros(self.shape, np.float32), 0)
            np.testing.assert_array_equal(outside, self.left * .25)
        finally:
            rasters.close()
        self.assertEqual({path: path.read_bytes() for path in self.root.iterdir()}, before)

    def test_two_anchors_replace_only_once_and_do_not_create_temporal_hints(self):
        import numpy as np
        self.data['entries'].append(self.entry('second', self.right, frame=3))
        data = self.tracked(anchor_ids=['anchor', 'second'])
        original = self.rasters(self.data)
        tracked = self.rasters(data)
        try:
            alpha = np.zeros(self.shape, np.float32)
            np.testing.assert_array_equal(tracked.apply(alpha, 2), self.right * .5)
            np.testing.assert_array_equal(tracked.apply(alpha, 2, temporal=True), original.apply(alpha, 2, temporal=True))
            self.assertFalse(tracked.has_condition(2))
            self.assertEqual(tracked.document.frames, original.document.frames)
            before, quality = original.condition(1, alpha)
            after, tracked_quality = tracked.condition(1, alpha)
            assert before is not None and after is not None
            np.testing.assert_array_equal(before, after)
            self.assertEqual(quality, tracked_quality)
        finally:
            original.close()
            tracked.close()

    def test_owner_outside_original_static_range_still_replaces_on_derived_frames(self):
        import numpy as np
        self.data['entries'][0]['range'] = [1, 2]
        rasters = self.rasters(self.tracked())
        try:
            self.assertTrue(rasters.has_effect(2))
            np.testing.assert_array_equal(rasters.apply(np.zeros(self.shape, np.float32), 2), self.right * .5)
        finally:
            rasters.close()

    def test_relocation_identity_detached_inventory_and_save_reopen(self):
        document = CorrectionDocument.parse(self.tracked(), self.root)
        inventories = document.asset_bindings()
        self.assertIsInstance(inventories, tuple)
        self.assertEqual(len(inventories), 4)
        inventories[0]['path'] = 'mutated detached result'
        self.assertNotEqual(document.asset_bindings()[0]['path'], inventories[0]['path'])
        self.assertEqual(len(document.asset_paths()), 4)
        destination = self.root / 'moved'
        destination.mkdir()
        path = destination / 'corrections.json'
        save_corrections(path, document)
        reopened = load_corrections(path)
        self.assertEqual(reopened.identity, document.identity)
        self.assertEqual(reopened.asset_paths(), document.asset_paths())
        self.assertEqual(reopened.to_dict()['tracks'][0]['binding'], document.to_dict()['tracks'][0]['binding'])

    def test_track_accept_remove_undo_redo_invalidate_full_shot(self):
        before = CorrectionDocument.parse(self.data, self.root)
        after = CorrectionDocument.parse(self.tracked(), self.root)
        self.assertEqual(invalidated_ranges(before, after), [(0, 5)])
        history = CorrectionHistory(before)
        self.assertEqual(history.apply(after), [(0, 5)])
        self.assertEqual(history.undo(), [(0, 5)])
        self.assertEqual(history.current.identity, before.identity)
        self.assertEqual(history.redo(), [(0, 5)])
        self.assertEqual(history.current.identity, after.identity)

    def test_anchor_source_order_and_boundary_changes_refuse_stale_derived_results(self):
        base = self.tracked()
        for field in ('anchor_revision', 'anchor_strength', 'anchor_hash', 'source', 'order', 'reset'):
            data = copy.deepcopy(base)
            if field == 'anchor_revision':
                data['entries'][0]['revision'] += 1
            elif field == 'anchor_strength':
                data['entries'][0]['strength'] = .75
            elif field == 'anchor_hash':
                data['entries'][0]['asset']['sha256'] = 'b' * 64
            elif field == 'source':
                data['source']['sha256'] = 'c' * 64
            else:
                data['entries'].insert(0, {'id': 'reset', 'revision': 1, 'frame': 0 if field == 'order' else 2, 'kind': 'reset'})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'stale'):
                CorrectionDocument.parse(data, self.root)
        # Accepting revision does not enter its own input binding.
        base['revision'] += 1
        CorrectionDocument.parse(base, self.root)

    def test_live_decode_or_cut_context_change_is_explicit_error(self):
        document = CorrectionDocument.parse(self.tracked(), self.root)
        document.validate_track_context(cuts=[], decode=self.decode)
        for cuts, decode in (([0], self.decode), ([], {**self.decode, 'frame_dir_linear': True})):
            with self.assertRaisesRegex(ValueError, 'context changed'):
                document.validate_track_context(cuts=cuts, decode=decode)

    def test_cross_cut_reset_shot_and_bad_direction_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'cross'):
            CorrectionDocument.parse(self.tracked(cuts=[2]), self.root)
        data = copy.deepcopy(self.data)
        data['entries'][0]['range'] = [0, 2]
        data['shots'] = [[0, 2], [2, 5]]
        with self.assertRaisesRegex(ValueError, 'cross'):
            CorrectionDocument.parse(self.tracked(data=data), self.root)
        data = copy.deepcopy(self.data)
        data['entries'].append({'id': 'reset', 'revision': 1, 'frame': 2, 'kind': 'reset'})
        with self.assertRaisesRegex(ValueError, 'cross'):
            CorrectionDocument.parse(self.tracked(data=data), self.root)
        with self.assertRaisesRegex(ValueError, 'direction'):
            CorrectionDocument.parse(self.tracked(direction='backward'), self.root)

    def test_interleaved_or_different_operation_anchors_and_duplicate_frames_refuse(self):
        for case in ('interleaved', 'kind', 'strength', 'sameframe'):
            data = copy.deepcopy(self.data)
            second = self.entry('second-' + case, self.right, frame=3)
            if case == 'interleaved':
                data['entries'].append(self.entry('middle', self.left, kind='remove'))
            elif case == 'kind':
                second['kind'] = 'remove'
            elif case == 'strength':
                second['strength'] = .25
            else:
                second['frame'] = 1
            data['entries'].append(second)
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.tracked(data=data, anchor_ids=['anchor', second['id']])

    def test_overlap_missing_frame_changed_hash_and_unknown_schema_refuse(self):
        base = self.tracked()
        data = copy.deepcopy(base)
        data['tracks'].append({**data['tracks'][0], 'id': 'other'})
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            CorrectionDocument.parse(data, self.root)
        data = copy.deepcopy(base)
        data['tracks'][0]['frames'].pop()
        with self.assertRaisesRegex(ValueError, 'every accepted frame'):
            CorrectionDocument.parse(data, self.root)
        data = copy.deepcopy(base)
        data['tracks'][0]['binding_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'stale'):
            CorrectionDocument.parse(data, self.root)
        for version in (1, 2, 4, True):
            data = copy.deepcopy(base)
            data['schema_version'] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                CorrectionDocument.parse(data, self.root)

    def test_budgets_and_all_derived_extents_are_admitted_before_decode(self):
        data = self.tracked()
        document = CorrectionDocument.parse(data, self.root)
        with patch('corridorkey.corrections.MAX_TRACK_BYTES', 1), \
                patch('corridorkey.corrections.load_tracked_asset') as decode, self.assertRaisesRegex(ValueError, '512 MiB'):
            document.verify_assets()
        decode.assert_not_called()
        with patch('corridorkey.corrections.MAX_TRACK_FRAMES', 2), self.assertRaisesRegex(ValueError, 'budget|count'):
            CorrectionDocument.parse(data, self.root)
        path = Path(data['tracks'][0]['frames'][0]['asset']['path'])
        with path.open('r+b') as stream:
            stream.truncate(1024**2 + self.shape[0] * self.shape[1] * 8 + 1)
        with patch('cv2.imdecode') as decode, self.assertRaisesRegex(ValueError, 'bounded'):
            load_tracked_asset(path, expected_hw=self.shape, expected_sha256='a' * 64)
        decode.assert_not_called()

    def test_encoded_geometry_and_float32_admitted_before_native_decode(self):
        import numpy as np
        for name, mask, half in (('wrong.exr', np.ones((4, 8), np.float32), False),
                                 ('half.exr', self.left, True), ('png.png', (self.left * 255).astype(np.uint8), False)):
            asset = self.asset(name, mask, half=half)
            with self.subTest(name=name), patch('cv2.imdecode') as decode, self.assertRaises(ValueError):
                load_tracked_asset(asset['path'], expected_hw=self.shape, expected_sha256=asset['sha256'])
            decode.assert_not_called()

    def test_derived_aliases_corruption_noncoverage_and_cancellation_refuse(self):
        import numpy as np

        from corridorkey.control import CancellationToken, CancelledError
        asset = self.asset('valid.exr', self.right)
        original = Path(asset['path'])
        alias = self.root / 'alias.exr'
        alias.symlink_to(original)
        with self.assertRaises(ValueError):
            load_tracked_asset(alias, expected_hw=self.shape, expected_sha256=asset['sha256'])
        hard = self.root / 'hard.exr'
        os.link(original, hard)
        with self.assertRaises(ValueError):
            load_tracked_asset(original, expected_hw=self.shape, expected_sha256=asset['sha256'])
        hard.unlink()
        with self.assertRaisesRegex(ValueError, 'hash'):
            load_tracked_asset(original, expected_hw=self.shape, expected_sha256='b' * 64)
        invalid = self.asset('noncoverage.exr', np.full(self.shape, 1.1, np.float32))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            load_tracked_asset(invalid['path'], expected_hw=self.shape, expected_sha256=invalid['sha256'])
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(CancelledError):
            load_tracked_asset(original, expected_hw=self.shape, expected_sha256=asset['sha256'], check=token.check)

    def test_observations_require_bound_indices_primitive_diagnostics_and_explicit_approximation(self):
        base = self.tracked()
        for field, value in (('frame', True), ('source_frame', 99), ('status', 'anchor_required'),
                             ('candidate_identity', 'bad'), ('derived_mask_sha256', 'bad'),
                             ('source_proxy_sha256', {'3': 'c' * 64}), ('diagnostics', {'bad': float('nan')}),
                             ('diagnostics', {'bad': {1, 'mixed'}})):
            data = copy.deepcopy(base)
            data['tracks'][0]['observations'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                CorrectionDocument.parse(data, self.root)
        data = copy.deepcopy(base)
        data['tracks'][0]['observations'][1]['status'] = 'approximation'
        with self.assertRaisesRegex(ValueError, 'approximation policy'):
            CorrectionDocument.parse(data, self.root)
        track = data['tracks'][0]
        track['binding']['tracking']['unreliable_policy'] = 'retain_approximation'
        track['binding_sha256'] = hashlib.sha256(canonical(track['binding']).encode()).hexdigest()
        CorrectionDocument.parse(data, self.root)
        data['tracks'][0]['observations'][1]['diagnostics'] = {'oversized': ['x' * 4096] * 17}
        with self.assertRaisesRegex(ValueError, '64 KiB'):
            CorrectionDocument.parse(data, self.root)

    def test_decoded_mask_sha_binds_accepted_candidate_even_when_encoded_hash_matches(self):
        data = self.tracked()
        data['tracks'][0]['observations'][1]['derived_mask_sha256'] = 'e' * 64
        document = CorrectionDocument.parse(data, self.root)
        with self.assertRaisesRegex(ValueError, 'candidate provenance'):
            document.verify_assets()
        with self.assertRaisesRegex(ValueError, 'candidate provenance'):
            self.rasters(data)


if __name__ == '__main__':
    unittest.main()
