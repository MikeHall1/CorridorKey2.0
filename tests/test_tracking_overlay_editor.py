"""Explicit diagnostic receipts/views from real sequence tracking, no neural work."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.control import CancelledError
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.correction_tracking_editor import (
    accept_tracking,
    discard_tracking,
    prepare_tracking,
    tracking_overlay_view,
    tracking_summary,
    tracking_view,
)
from corridorkey.session import JobRequest


class TrackingOverlayEditorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        for index in range(5):
            self.assertTrue(cv2.imwrite(str(self.source / f'{index:03d}.png'), np.full((24, 32, 3), 100, np.uint8)))
        self.originals = {p.name: p.read_bytes() for p in self.source.iterdir()}
        self.job = JobRequest.create(self.source, self.root / 'render',
                                     InferenceConfig(output_format='exr', start_frame=2, num_frames=3))
        self.editor, self.entry = self.make_editor(self.job)
        self.authored = self.editor.document.asset_path(self.entry).read_bytes()

    def make_editor(self, job):
        editor = CorrectionEditor(job, self.root)
        mask = np.zeros((24, 32), np.uint8)
        mask[6:18, 8:24] = 255
        editor.commit_brush(mask, kind='keep', frame=0, strength=.5, direction='both', span=[0, 3])
        return editor, editor.document.to_dict()['entries'][0]

    def private(self, editor=None):
        preview = (editor if editor is not None else self.editor).tracking_preview
        assert preview is not None
        return preview

    def summary(self, editor=None):
        summary = tracking_summary(editor if editor is not None else self.editor)
        assert summary is not None
        return summary

    def tearDown(self):
        self.assertEqual(self.originals, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual(self.authored, self.editor.document.asset_path(self.entry).read_bytes())
        self.temp.cleanup()

    def prepare(self, *, capture=True):
        prepare_tracking(self.editor, anchor_ids=[self.entry['id']], span=[0, 3], direction='forward',
                         capture_diagnostics=capture)

    def test_default_capture_disabled_adds_no_maps_or_flow_work(self):
        from corridorkey.correction_tracking import propagate_alpha
        with patch('corridorkey.tracking_overlays.build_tracking_overlays', side_effect=AssertionError('default must not capture')), patch('corridorkey.correction_tracking.propagate_alpha', wraps=propagate_alpha) as flow:
            prepare_tracking(self.editor, anchor_ids=[self.entry['id']], span=[0, 3], direction='forward')
            default_calls = flow.call_count
        private = self.editor.tracking_preview
        assert private is not None
        self.assertFalse((private['directory'] / 'diagnostics').exists())
        self.assertFalse(self.summary()['diagnostics']['captured'])
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            tracking_overlay_view(self.editor, 1, 'reliability')
        with patch('corridorkey.correction_tracking.propagate_alpha', wraps=propagate_alpha) as flow:
            self.prepare()
            self.assertEqual(flow.call_count, default_calls)
        self.assertEqual(default_calls, 2)

    def test_actual_maps_offset_bound_readonly_composite_and_opacity_no_acceptance(self):
        before = self.editor.document.identity
        self.prepare()
        summary = tracking_summary(self.editor)
        assert summary is not None
        diagnostic = summary['diagnostics']
        self.assertEqual(diagnostic['available_frames'], [1, 2])
        self.assertEqual(diagnostic['frame_events'][1], {'frame': 1, 'source_frame': 3, 'status': 'preview', 'available': True, 'reason': ''})
        self.assertEqual(diagnostic['budget_bytes'], 64 * 1024**2)
        self.assertGreater(diagnostic['bytes_charged'], 0)
        self.assertLess(len(json.dumps(diagnostic)), 64 * 1024)
        self.assertIsInstance(diagnostic['legend'], str)
        source, _ = self.editor.view(1)
        original_pixels = source.pixels.copy()
        with patch.object(self.editor, 'view', return_value=(source, None)):
            zero, overlay = tracking_overlay_view(self.editor, 1, 'reliability', opacity=0)
            full, _ = tracking_overlay_view(self.editor, 1, 'reliability', opacity=1)
        self.assertIsNone(overlay)
        np.testing.assert_array_equal(zero.pixels, original_pixels)
        np.testing.assert_array_equal(source.pixels, original_pixels)
        self.assertFalse(np.shares_memory(full.pixels, source.pixels))
        self.assertFalse(full.pixels.flags.writeable)
        self.assertIn('local frame 1 / source frame 3', full.note)
        self.assertIn('heuristic', full.note)
        self.assertEqual((full.width, full.height), (32, 24))
        self.assertFalse(self.private()['inspected'])
        self.assertEqual(self.editor.document.identity, before)
        with self.assertRaisesRegex(ValueError, 'inspected'):
            accept_tracking(self.editor)
        # Snapshots are detached metadata, not maps or receipt payloads.
        diagnostic['frame_events'][0]['status'] = 'changed externally'
        self.assertEqual(self.summary()['diagnostics']['frame_events'][0]['status'], 'authored_anchor')

    def test_anchor_missing_maps_and_invalid_arguments_refuse_without_backfill(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            tracking_overlay_view(self.editor, 0, 'uncertainty')
        for frame, kind, opacity in ((True, 'reliability', .5), (1, 'unknown', .5), (1, 'reliability', True),
                                     (1, 'reliability', -.1), (1, 'reliability', float('nan'))):
            with self.subTest(frame=frame, kind=kind, opacity=opacity), self.assertRaises(ValueError):
                tracking_overlay_view(self.editor, frame, kind, opacity)

    def test_stopped_motion_frame_maps_are_inspectable_without_a_mask(self):
        target = self.source / '003.png'
        original = target.read_bytes()
        try:
            self.assertTrue(cv2.imwrite(str(target), np.full((24, 32, 3), 255, np.uint8)))
            editor, entry = self.make_editor(self.job)
            prepare_tracking(editor, anchor_ids=[entry['id']], span=[0, 3], direction='forward',
                             unreliable_policy='stop', capture_diagnostics=True)
            summary = self.summary(editor)
            self.assertFalse(summary['complete'])
            events = summary['diagnostics']['frame_events']
            self.assertEqual(events[1]['status'], 'stopped')
            self.assertTrue(events[1]['available'])
            self.assertFalse(events[2]['available'])
            display, overlay = tracking_overlay_view(editor, 1, 'uncertainty')
            self.assertIsNone(overlay)
            self.assertEqual((display.width, display.height), (32, 24))
            with self.assertRaisesRegex(ValueError, 'no retained mask'):
                tracking_view(editor, 1)
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                tracking_overlay_view(editor, 2, 'uncertainty')
            zero, _ = tracking_overlay_view(editor, 1, 'uncertainty', opacity=0)
            full, _ = tracking_overlay_view(editor, 1, 'uncertainty', opacity=1)
            self.assertTrue(np.any(zero.pixels != full.pixels))
            self.assertFalse(self.private(editor)['inspected'])
        finally:
            target.write_bytes(original)

    def test_stale_document_decode_source_and_tampered_receipt_refuse(self):
        self.prepare()
        preview = self.editor.tracking_preview
        assert preview is not None
        original_job = self.editor.job
        config, writer = original_job.configurations()
        self.editor.job = JobRequest.create(self.source, self.root / 'render', replace(config, frame_dir_linear=True), writer)
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            tracking_overlay_view(self.editor, 1, 'reliability')
        self.editor.job = original_job
        target = self.source / '003.png'
        original = target.read_bytes()
        try:
            target.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                tracking_overlay_view(self.editor, 1, 'reliability')
        finally:
            target.write_bytes(original)
        # Reprepare after source restoration so stat snapshots are fresh.
        self.prepare()
        native_mask = Path(self.private()['track']['frames'][1]['asset']['path'])
        native_mask.write_bytes(b'tampered coverage')
        with self.assertRaisesRegex(ValueError, 'mask changed'):
            tracking_overlay_view(self.editor, 1, 'reliability')
        self.prepare()
        preview = self.private()
        receipt = preview['diagnostic_receipts'][1].to_dict()
        leaf = preview['diagnostic_directory'] / receipt['maps'][0]['name']
        leaf.write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            tracking_overlay_view(self.editor, 1, 'reliability')
        self.prepare()
        self.editor.edit(self.entry['id'], {'strength': .75})
        with self.assertRaisesRegex(ValueError, 'No current'):
            tracking_overlay_view(self.editor, 1, 'reliability')

    def test_separate_budget_and_cancel_failure_leave_previous_preview_intact(self):
        self.prepare(capture=False)
        old = self.editor.tracking_preview
        assert old is not None
        with patch('corridorkey.correction_tracking_editor.MAX_DIAGNOSTIC_BYTES', 1), self.assertRaises(ValueError):
            self.prepare()
        self.assertIs(self.editor.tracking_preview, old)
        self.assertEqual(list(self.editor.workspace.glob('tracking-preview-*')), [old['directory']])
        from corridorkey.tracking_overlays import write_bundle
        cancelled = False
        original_check = self.editor.check

        def capture_then_cancel(*args, **kwargs):
            nonlocal cancelled
            result = write_bundle(*args, **kwargs)
            cancelled = True
            return result

        def check():
            if cancelled:
                raise CancelledError('cancel after map receipt')

        self.editor.check = check
        try:
            with patch('corridorkey.tracking_overlays.write_bundle', side_effect=capture_then_cancel), self.assertRaises(CancelledError):
                self.prepare()
        finally:
            self.editor.check = original_check
        self.assertIs(self.editor.tracking_preview, old)
        self.assertEqual(list(self.editor.workspace.glob('tracking-preview-*')), [old['directory']])

    def test_accept_removes_only_maps_saved_tracks_have_no_backfill_and_cleanup_failure_visible(self):
        self.prepare()
        private = self.editor.tracking_preview
        assert private is not None
        directory, diagnostics = private['directory'], private['diagnostic_directory']
        tracking_view(self.editor, 1)
        accept_tracking(self.editor)
        self.assertTrue(directory.is_dir())
        self.assertFalse(diagnostics.exists())
        self.assertTrue(list(directory.glob('*.exr')))
        saved = self.editor.save()
        reopened = CorrectionEditor(self.job, self.root, existing=saved)
        with self.assertRaisesRegex(ValueError, 'Saved tracks'):
            tracking_overlay_view(reopened, 1, 'reliability')
        self.editor.undo()  # A second track may not overlap the same accepted anchor.
        self.prepare()
        private = self.private()
        tracking_view(self.editor, 1)
        import shutil
        rmtree = shutil.rmtree

        def fail_maps(path):
            if path == private['diagnostic_directory']:
                raise OSError('map cleanup blocked')
            return rmtree(path)

        with patch('corridorkey.correction_tracking_editor.shutil.rmtree', side_effect=fail_maps):
            accept_tracking(self.editor)
        self.assertIsNone(self.editor.tracking_preview)
        self.assertIn(private['diagnostic_directory'], self.editor.tracking_cleanup_pending)
        self.assertIn('map cleanup blocked', self.editor.tracking_cleanup_warnings[-1])
        self.assertTrue(self.editor.document.to_dict()['tracks'])

    def test_discard_removes_private_maps_and_real_linear_exr_can_be_inspected(self):
        self.prepare()
        directory = self.private()['directory']
        discard_tracking(self.editor)
        self.assertFalse(directory.exists())
        exr_source = self.root / 'linear-source'
        exr_source.mkdir()
        for index in range(3):
            self.assertTrue(cv2.imwrite(str(exr_source / f'{index}.exr'), np.full((24, 32, 3), .4, np.float32)))
        job = JobRequest.create(exr_source, self.root / 'exr-output', InferenceConfig(output_format='exr', num_frames=3,
                                frame_dir_linear=True, source_colour_profile='linear_bt709_d65'))
        editor, entry = self.make_editor(job)
        prepare_tracking(editor, anchor_ids=[entry['id']], span=[0, 3], direction='forward', capture_diagnostics=True)
        display, overlay = tracking_overlay_view(editor, 1, 'reliability')
        self.assertIsNone(overlay)
        self.assertEqual(display.pixels.dtype, np.uint8)
        self.assertIn('source frame 1', display.note)


if __name__ == '__main__':
    unittest.main()
