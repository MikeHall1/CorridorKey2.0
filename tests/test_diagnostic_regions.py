"""Captured-map opacity only; no new measurements or neural execution."""
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.control import CancelledError
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.tracking_overlays import OverlayMap, render_layer
from tests import test_tracking_overlay_editor as fixtures
from tests.test_tracking_overlays import build


class DiagnosticRendererTests(unittest.TestCase):
    def test_area_proxy_alpha_only_full_zero_fraction_none_and_cancel(self):
        original = build(max_side=8)
        bundle = replace(original, maps=tuple(OverlayMap(m.kind, m.width, m.height,
                         bytes([128]) * (m.width * m.height)) for m in original.maps))
        before = bundle.identity
        plain = render_layer(bundle, 'uncertainty', opacity=.55)
        pixels = np.frombuffer(plain.pixels, np.uint8).reshape(4, 8, 4)
        for native in (np.zeros((8, 16), np.float32), np.ones((8, 16), np.float32),
                       np.tile(np.array([0., 1.], np.float32), (8, 8))):
            layer = render_layer(bundle, 'uncertainty', opacity=.55, region=native)
            rgba = np.frombuffer(layer.pixels, np.uint8).reshape(4, 8, 4)
            coverage = cv2.resize(native, (8, 4), interpolation=cv2.INTER_AREA)
            np.testing.assert_array_equal(rgba[..., :3], pixels[..., :3])
            np.testing.assert_array_equal(rgba[..., 3], np.rint(pixels[..., 3] * coverage).astype(np.uint8))
            self.assertIn('statistics unchanged', layer.legend['scope'])
        self.assertEqual(render_layer(bundle, 'uncertainty', opacity=.55, region=None), plain)
        self.assertEqual(render_layer(bundle, 'uncertainty', opacity=.55, region=np.ones((8,16),np.float32)).pixels,plain.pixels)
        self.assertEqual(bundle.identity, before)
        with self.assertRaises(CancelledError):
            render_layer(bundle, 'uncertainty', region=np.ones((8,16),np.float32),
                         check=lambda: (_ for _ in ()).throw(CancelledError('cancel')))
        for bad in (np.zeros((4,8),np.float32), np.full((8,16),np.nan,np.float32), np.ones((8,16),np.uint8)):
            with self.assertRaises(ValueError):
                render_layer(bundle, 'uncertainty', region=bad)


class DiagnosticRegionEditorTests(unittest.TestCase):
    editor: CorrectionEditor

    setUp = fixtures.TrackingOverlayEditorTests.setUp
    tearDown = fixtures.TrackingOverlayEditorTests.tearDown
    make_editor = fixtures.TrackingOverlayEditorTests.make_editor
    prepare = fixtures.TrackingOverlayEditorTests.prepare
    private = fixtures.TrackingOverlayEditorTests.private
    def add_region(self, value=255, strength=1):
        self.editor.commit_brush(np.full((24,32),value,np.uint8),kind='region',purpose='diagnostic',
                                 frame=0,strength=strength,direction='both',span=[0,3])
        return self.editor.document.to_dict()['entries'][-1]

    def test_real_capture_zero_full_and_stale_document(self):
        self.add_region(value=0)
        self.prepare()
        source, _ = self.editor.view(1)
        displayed, _ = fixtures.tracking_overlay_view(self.editor, 1, 'uncertainty', opacity=1)
        np.testing.assert_array_equal(displayed.pixels, source.pixels)
        self.assertIn('clips display opacity only', displayed.note)
        before = self.private()['frames'].copy()
        full = self.add_region()
        with self.assertRaisesRegex(ValueError,'current diagnostic'):
            fixtures.tracking_overlay_view(self.editor, 1, 'uncertainty')
        self.prepare()
        preview = self.private()
        observations = json.dumps(preview['frames'], sort_keys=True)
        full_view, _ = fixtures.tracking_overlay_view(self.editor, 1, 'reliability')
        with patch('corridorkey.correction_rasters.CorrectionRasters.diagnostic_region',return_value=None):
            unscoped, _ = fixtures.tracking_overlay_view(self.editor, 1, 'reliability')
        np.testing.assert_array_equal(full_view.pixels,unscoped.pixels)
        self.assertEqual(json.dumps(preview['frames'],sort_keys=True),observations)
        self.assertEqual(before[0]['frame'],preview['frames'][0]['frame'])
        self.assertEqual(full['purpose'],'diagnostic')

    def test_late_region_asset_mutation_cannot_publish(self):
        entry = self.add_region()
        self.prepare()
        original = self.editor.view
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.editor.document.base_dir / entry['asset']['path']
            path.write_bytes(path.read_bytes() + b'x')
            return result
        with patch.object(self.editor,'view',side_effect=mutate), self.assertRaisesRegex(ValueError,'asset changed|Asset changed'):
            fixtures.tracking_overlay_view(self.editor,1,'reliability')

    def test_late_captured_map_or_candidate_mask_mutation_cannot_publish(self):
        self.add_region()
        for asset in ('map', 'candidate'):
            self.prepare()
            preview = self.private()
            if asset == 'map':
                receipt = preview['diagnostic_receipts'][1]
                path = preview['diagnostic_directory'] / receipt.to_dict()['maps'][0]['name']
            else:
                path = Path(next(row['asset']['path'] for row in preview['track']['frames'] if row['frame'] == 1))
            original = self.editor.view
            def mutate(*args, _original=original, _path=path, **kwargs):
                result = _original(*args, **kwargs)
                _path.write_bytes(_path.read_bytes() + b'x')
                return result
            with patch.object(self.editor,'view',side_effect=mutate), self.assertRaisesRegex(ValueError,'asset changed'):
                fixtures.tracking_overlay_view(self.editor,1,'reliability')
