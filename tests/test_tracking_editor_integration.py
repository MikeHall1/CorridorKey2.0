"""Actual local PNG/EXR tracking publication, never pretrained inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.control import CancelledError
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.correction_rasters import CorrectionRasters
from corridorkey.corrections import load_corrections
from corridorkey.session import JobRequest


class TrackingEditorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for index in range(4):
            cv2.imwrite(str(self.source / f'frame{index}.png'), np.full((24, 32, 3), 100, np.uint8))
        self.originals = {p.name: p.read_bytes() for p in self.source.iterdir()}
        self.job = JobRequest.create(self.source, self.root / 'render', InferenceConfig(output_format='exr'))
        self.editor = CorrectionEditor(self.job, self.root)
        mask = np.zeros((24, 32), np.uint8)
        mask[6:18, 8:24] = 255
        self.editor.commit_brush(mask, kind='keep', frame=0, strength=.5, direction='both', span=[0, 4])
        self.entry = self.editor.document.to_dict()['entries'][0]
        self.authored = self.editor.document.asset_path(self.entry).read_bytes()

    def prepare(self, direction='forward', span=None):
        self.editor.prepare_tracking(anchor_ids=[self.entry['id']], span=span or [0, 4], direction=direction)

    def test_inspect_accept_once_save_reopen_undo_and_authored_edit_invalidation(self):
        before = self.editor.document.identity
        self.prepare()
        self.assertEqual(self.editor.document.identity, before)
        snapshot = self.editor.snapshot()
        snapshot['tracking_preview']['frames'][0]['status'] = 'mutated outside owner'
        summary = self.editor.tracking_summary()
        assert summary is not None
        self.assertEqual(summary['frames'][0]['status'], 'authored_anchor')
        with self.assertRaisesRegex(ValueError, 'inspected'):
            self.editor.accept_tracking()
        display, overlay = self.editor.tracking_view(2)
        self.assertEqual((display.width, display.height), (32, 24))
        self.assertEqual(overlay[10, 12], 255)
        self.editor.accept_tracking()
        tracked = self.editor.document.identity
        rasters = CorrectionRasters(self.editor.document, self.editor.meta)
        self.addCleanup(rasters.close)
        alpha = rasters.apply(np.zeros((24, 32), np.float32), 2)
        self.assertEqual(alpha[10, 12], .5)  # .75 would mean duplicate keep application.
        saved = self.editor.save()
        reopened = CorrectionEditor(self.job, self.root, existing=saved)
        self.assertEqual(reopened.document.identity, tracked)
        self.editor.undo()
        self.assertEqual(self.editor.document.identity, before)
        self.editor.redo()
        self.assertEqual(self.editor.document.identity, tracked)
        self.editor.edit(self.entry['id'], {'strength': .75})
        self.assertEqual(self.editor.document.to_dict()['tracks'], [])
        self.assertEqual(self.editor.invalidated, [(0, 4)])
        self.assertEqual(self.editor.document.asset_path(self.entry).read_bytes(), self.authored)
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, self.originals)

    def test_source_change_after_inspection_prevents_acceptance(self):
        self.prepare()
        self.editor.tracking_view(1)
        before = self.editor.document.identity
        cv2.imwrite(str(self.source / 'frame2.png'), np.zeros((24, 32, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.editor.accept_tracking()
        self.assertEqual(self.editor.document.identity, before)

    def test_cancelled_replacement_preserves_previous_preview_and_cleans_staging(self):
        self.prepare()
        previous = self.editor.tracking_preview
        directories = set(self.editor.workspace.iterdir())
        with patch('corridorkey.correction_tracking_editor.track_corrections', side_effect=CancelledError('fixture cancel')), \
                self.assertRaises(CancelledError):
            self.prepare()
        self.assertIs(self.editor.tracking_preview, previous)
        self.assertEqual(set(self.editor.workspace.iterdir()), directories)
        self.editor.discard_tracking()
        self.assertIsNone(self.editor.tracking_preview)

    def test_publication_failure_preserves_history_and_preview_for_retry(self):
        self.prepare()
        self.editor.tracking_view(1)
        before = self.editor.document.identity
        with patch('corridorkey.correction_editor.save_corrections', side_effect=OSError('fixture full disk')), \
                self.assertRaises(OSError):
            self.editor.accept_tracking()
        self.assertEqual(self.editor.document.identity, before)
        self.assertIsNotNone(self.editor.tracking_preview)
        self.editor.accept_tracking()
        self.assertTrue(self.editor.document.to_dict()['tracks'])

    def test_shot_and_reset_boundaries_refuse_without_publication(self):
        self.editor.edit(self.entry['id'], {'range': [0, 1]})
        self.editor.shot(2)
        before = self.editor.document.identity
        with self.assertRaisesRegex(ValueError, 'boundary'):
            self.prepare()
        self.assertEqual(self.editor.document.identity, before)
        self.assertIsNone(self.editor.tracking_preview)

    def test_derived_asset_identity_and_inventory_survive_relocation(self):
        from corridorkey.benchmark import correction_identity
        from corridorkey.corrections import save_corrections
        self.prepare()
        self.editor.tracking_view(0)
        self.editor.accept_tracking()
        destination = self.root / 'moved' / 'document.json'
        save_corrections(destination, self.editor.document)
        loaded = load_corrections(destination, source=self.source)
        self.assertEqual(loaded.identity, self.editor.document.identity)
        self.assertEqual(len(loaded.asset_paths()), 5)
        self.assertEqual(len(correction_identity(destination, self.source)['assets']), 5)
        with self.assertRaisesRegex(ValueError, 'context'):
            loaded.validate_track_context(cuts=[2], decode=loaded.to_dict()['tracks'][0]['binding']['decode'])

    def test_cleanup_failure_preserves_published_preview_and_committed_history(self):
        self.prepare()
        previous = self.editor.tracking_preview
        with patch('corridorkey.correction_tracking_editor.shutil.rmtree', side_effect=OSError('fixture locked')):
            self.prepare()
            current = self.editor.tracking_preview
            assert current is not None
            self.assertIsNot(current, previous)
            self.assertTrue(current['directory'].is_dir())
            self.assertTrue(self.editor.tracking_cleanup_pending)
            self.assertTrue(self.editor.snapshot()['tracking_cleanup_warnings'])
            revision = self.editor.revision
            self.editor.edit(self.entry['id'], {'strength': .6})
            self.assertEqual(self.editor.revision, revision + 1)
            self.assertIsNone(self.editor.tracking_preview)
        self.prepare()  # Retries and removes old private directories before work.
        self.assertFalse(self.editor.tracking_cleanup_pending)

    def test_anchor_budget_refuses_before_source_scan_or_decode(self):
        from corridorkey.correction_tracking import TrackingRequest

        def bounded(*args, **kwargs):
            return TrackingRequest(*args, **kwargs, max_anchor_bytes=1)

        with patch('corridorkey.correction_tracking_editor.TrackingRequest', side_effect=bounded), \
                patch('corridorkey.preview_sequence.scan_sequence') as scan, self.assertRaisesRegex(ValueError, '64 MiB'):
            self.prepare()
        scan.assert_not_called()
        self.assertIsNone(self.editor.tracking_preview)

    def test_actual_flow_preview_publishes_moving_mask_and_reopens(self):
        source = self.root / 'moving'
        source.mkdir()
        pixels = np.random.default_rng(32).integers(0, 256, (64, 80, 3), dtype=np.uint8)
        for frame in range(4):
            cv2.imwrite(str(source / f'{frame}.png'), np.roll(pixels, frame, axis=1))
        editor = CorrectionEditor(JobRequest.create(source, self.root / 'moving-render', InferenceConfig(output_format='exr')), self.root)
        mask = np.zeros((64, 80), np.uint8)
        mask[24:40, 25:40] = 255
        editor.commit_brush(mask, kind='keep', frame=0, strength=1., direction='forward', span=[0, 1])
        entry = editor.document.to_dict()['entries'][0]
        authored = editor.document.asset_path(entry).read_bytes()
        editor.prepare_tracking(anchor_ids=[entry['id']], span=[0, 4], direction='forward')
        editor.tracking_view(3)
        editor.accept_tracking()
        loaded = load_corrections(editor.save(), source=source)
        rasters = CorrectionRasters(loaded, editor.meta)
        try:
            actual = rasters.apply(np.zeros((64, 80), np.float32), 3) > .5
        finally:
            rasters.close()
        expected = np.roll(mask, 3, axis=1) > 0
        iou = np.sum(actual & expected) / np.sum(actual | expected)
        self.assertGreater(float(iou), .95)
        self.assertEqual(editor.document.asset_path(entry).read_bytes(), authored)

    def test_output_and_recovery_guards_protect_derived_assets_without_decode(self):
        from corridorkey.corrections import CorrectionDocument, save_corrections
        from corridorkey.output_paths import output_layout, validate_output_paths
        from corridorkey.recovery import _separate_workspace
        self.prepare()
        self.editor.tracking_view(0)
        self.editor.accept_tracking()
        config, _ = self.job.configurations()
        output = self.root / 'guard-render'
        layout = output_layout(self.source, config)['outputs']['alpha']
        target = output / layout['path'] / (layout['frame_name'] + '_000000.exr')
        target.parent.mkdir(parents=True)
        data = self.editor.document.to_dict()
        asset = data['tracks'][0]['frames'][0]['asset']
        target.write_bytes(Path(asset['path']).read_bytes())
        asset['path'] = str(target)
        path = self.root / 'guard-corrections.json'
        save_corrections(path, CorrectionDocument.parse(data, self.editor.document.base_dir))
        config.corrections_path = str(path)
        job = JobRequest.create(self.source, self.root / 'unrelated-output', config)
        before = target.read_bytes()
        with patch('cv2.imdecode', side_effect=AssertionError('guard must not decode')), self.assertRaisesRegex(ValueError, 'protected'):
            validate_output_paths(self.source, output, config)
        with self.assertRaisesRegex(ValueError, 'dependencies'):
            _separate_workspace(target.parent, [job], {})
        self.assertEqual(target.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
