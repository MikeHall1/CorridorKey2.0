"""Non-neural region purpose partition; CPU fixtures, not quality evidence."""
import unittest
from pathlib import Path

import numpy as np

from tests import test_refinement_regions as fixtures


class RegionPurposeTests(unittest.TestCase):
    root: Path
    shape: tuple[int, int]
    meta: dict

    setUp = fixtures.RefinementRegionTests.setUp
    document = fixtures.RefinementRegionTests.document

    def test_purpose_partition_and_no_alpha_hint_effect(self):
        base = self.document([np.full(self.shape, 255, np.uint8)]).to_dict()
        from corridorkey.correction_rasters import CorrectionRasters
        from corridorkey.corrections import CorrectionDocument
        for purpose, target, method in [('composite', 'despill', 'composite_region'),
                                        ('diagnostic', 'tracking_overlay', 'diagnostic_region')]:
            with self.subTest(purpose=purpose):
                data = {**base, 'entries': [{**base['entries'][0], 'purpose': purpose, 'target': target, 'strength': .5}]}
                doc = CorrectionDocument.parse(data, self.root)
                rasters = CorrectionRasters(doc, self.meta)
                self.addCleanup(rasters.close)
                self.assertIsNone(rasters.refinement_region(0))
                np.testing.assert_array_equal(getattr(rasters, method)(0), np.full(self.shape, .5, np.float32))
                self.assertIsNone(getattr(rasters, method)(2))
                alpha = np.full(self.shape, .25, np.float32)
                np.testing.assert_array_equal(rasters.apply(alpha, 0), alpha)
                self.assertEqual(rasters.condition(0), (None, None))
                self.assertEqual(doc.frames, ())

    def test_mixed_purposes_zero_disabled_fractional_and_bad_pairs(self):
        import copy

        from corridorkey.correction_rasters import CorrectionRasters
        from corridorkey.corrections import CorrectionDocument
        base = self.document([np.zeros(self.shape,np.uint8),np.full(self.shape,128,np.uint8),np.full(self.shape,255,np.uint8)]).to_dict()
        base['entries'][0].update(purpose='composite',target='despill')
        base['entries'][1].update(purpose='diagnostic',target='tracking_overlay',strength=.5)
        base['entries'][2].update(purpose='diagnostic',target='tracking_overlay',strength=.1)
        rasters = CorrectionRasters(CorrectionDocument.parse(base,self.root),self.meta)
        self.addCleanup(rasters.close)
        self.assertIsNone(rasters.refinement_region(0))
        composite = rasters.composite_region(0)
        diagnostic = rasters.diagnostic_region(0)
        assert composite is not None and diagnostic is not None
        np.testing.assert_array_equal(composite,np.zeros(self.shape,np.float32))
        np.testing.assert_array_equal(diagnostic,np.full(self.shape,np.float32(128)/np.float32(255)*np.float32(.5),np.float32))
        base['entries'][0]['strength'] = 0
        disabled = CorrectionRasters(CorrectionDocument.parse(base,self.root),self.meta)
        self.addCleanup(disabled.close)
        self.assertIsNone(disabled.composite_region(0))
        for purpose,target in [('neural','roi'),('neural',None),('diagnostic','despill'),('composite','tracking_overlay')]:
            bad = copy.deepcopy(base)
            bad['entries'][0].update(purpose=purpose,target=target)
            with self.assertRaises(ValueError):
                CorrectionDocument.parse(bad,self.root)


class TrackedPurposeTests(unittest.TestCase):
    def test_all_purposes_preserve_tracking_overlap_offsets_and_bindings(self):
        import copy
        import hashlib

        from corridorkey.correction_rasters import CorrectionRasters
        from corridorkey.corrections import CorrectionDocument, canonical, track_binding
        from tests.test_tracked_refinement_regions import RegionTrackingCoreTests
        fixture = RegionTrackingCoreTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for purpose, target in [('composite','despill'),('diagnostic','tracking_overlay')]:
            for direction in ('forward','backward','both'):
                data, candidates = fixture.build(direction,both_anchors=direction=='both',overlap=True)
                for entry in data['entries']:
                    entry.update(purpose=purpose,target=target)
                track = data['tracks'][0]
                old = track['binding']
                track['binding'] = track_binding(data,track['anchor_ids'],cuts=old['cuts'],decode=old['decode'],tracking=old['tracking'])
                track['binding_sha256'] = hashlib.sha256(canonical(track['binding']).encode()).hexdigest()
                document = CorrectionDocument.parse(data,fixture.root)
                rasters = CorrectionRasters(document,fixture.meta)
                self.addCleanup(rasters.close)
                for candidate in candidates:
                    mask = candidate.mask()
                    assert mask is not None
                    weights = rasters.region(candidate.local_frame,purpose,target)
                    assert weights is not None
                    np.testing.assert_array_equal(weights,np.maximum(mask*np.float32(.4),np.float32(.2)*np.float32(.6)))
                    self.assertIsNone(rasters.refinement_region(candidate.local_frame))
                static = rasters.region(0,purpose,target)
                self.assertIsNotNone(static)
                stale = copy.deepcopy(data)
                stale['entries'][0]['purpose'] = 'refinement'
                stale['entries'][0]['target'] = 'edge_finishing'
                with self.assertRaises(ValueError):
                    CorrectionDocument.parse(stale,fixture.root)

    def test_each_purpose_editor_tracking_history_reopen_portable_and_corruption(self):
        from tests.test_tracked_refinement_regions import RegionEditorTests
        for purpose in ('composite','diagnostic'):
            fixture = RegionEditorTests()
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            editor = fixture.editor
            editor.undo()
            editor.import_mask(fixture.root/'soft.exr',kind='region',purpose=purpose,frame=1,strength=.4,direction='both',span=[0,4])
            fixture.entry = editor.document.to_dict()['entries'][0]
            original = editor.document.identity
            fixture.accept()
            accepted = editor.document.identity
            anchor = editor.document.to_dict()['tracks'][0]['binding']['anchors'][0]
            self.assertEqual(anchor['purpose'],purpose)
            display, overlay = editor.view(2,identifier=fixture.entry['id'])
            self.assertIn(purpose,display.note)
            self.assertIsNotNone(overlay)
            editor.undo()
            self.assertEqual(editor.document.identity,original)
            editor.redo()
            self.assertEqual(editor.document.identity,accepted)
            path = editor.save()
            from corridorkey.correction_editor import CorrectionEditor
            reopened = CorrectionEditor(fixture.job,fixture.root,existing=path)
            self.assertEqual(reopened.document.identity,accepted)
            # Reuse the same actual portable collection and corruption assertions.
            editor.undo()
            fixture.test_accepted_track_portable_collection_and_tampered_overlay_refusal()
