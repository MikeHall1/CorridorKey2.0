"""Selected capture uses the real CPU runtime with context-sensitive neural substitutes."""
import json
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.control import CancellationToken, CancelledError
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.desktop_comparison import completed_view
from corridorkey.preview_execution import prepare_preview, selected_preview_view
from corridorkey.session import InferenceSession, JobRequest
from corridorkey.worker_protocol import job_to_wire, parse_request
from tests import test_mixed_finishing_recovery as fixtures
from tests import test_stage2_incremental_recovery as temporal
from tests.test_durable_stage2 import cpu_runtime_context


class ConditionedPreviewTests(unittest.TestCase):
    root: Path
    source: Path
    hints: Path
    initial: Path

    def setUp(self: Any):
        fixtures.MixedFinishingPreproofTests.setUp(self)
        fixtures.MixedFinishingPreproofTests.scene(self)
        self.assertTrue(cv2.imwrite(str(self.initial), np.full((64, 64), 56000, np.uint16)))
        for frame in range(8, 14):
            image = cv2.imread(str(self.source / '0.png'))
            self.assertTrue(cv2.imwrite(str(self.source / f'{frame}.png'), np.roll(image, frame, axis=0)))
            self.assertTrue(cv2.imwrite(str(self.hints / f'{frame}.png'), np.full((64, 64), frame * 2000, np.uint16)))
        self.config = replace(self.config, num_frames=12, hann_chunk=8, hann_stride=4,
                              stage2_batch=3)
        editor = CorrectionEditor(self.job(self.config, 'editor'), self.root)
        mask = np.zeros((64, 64), np.uint8)
        mask[9:27, 12:39] = 255
        editor.commit_brush(mask, kind='matte', frame=2, strength=.35, direction='both', span=[0, 12])
        editor.reset(9)
        self.config = replace(self.config, corrections_path=editor.save())

    def job(self, config, name):
        return JobRequest.create(self.source, self.root / name, config,
                                 initial_hint_path=self.initial, hint_video_path=self.hints)

    def execute(self, config, name, selected=None, *, initial=True, full=True, zero_rvm=False, job_request=None):
        trace = {key: [] for key in ('iid', 'rvm', 'stage1', 'stage2')}
        model = temporal.Stage2IncrementalRecoveryTests.model(self, trace, [])
        original = model.stage2.forward

        def grouped(rgb, baton):
            result = original(rgb, baton)
            # Whole-group dependence detects selecting before Stage2, even with correct batons.
            result[:, :, 0] += baton.float().mean() * .03 + rgb.float().mean() * .02
            return result

        model.stage2.forward = grouped
        if zero_rvm:
            forward = model.stage1.forward
            def no_recurrence(*args, **kwargs):
                kwargs['rvm_feat'] = kwargs['rvm_feat'] * 0
                return forward(*args, **kwargs)
            model.stage1.forward = no_recurrence
        job = job_request or self.job(config, name)
        job = replace(job, initial_hint_path=job.initial_hint_path if initial else None,
                      hint_video_path=job.hint_video_path if full else None)
        with InferenceSession(loader=lambda: model, disposer=lambda _: None) as session:
            record = session.run(job, preview_frame=selected).to_dict()
        return record, trace

    def test_selected_exact_full_context_pixels_and_groups_normal_and_low(self):
        originals = {path: path.read_bytes() for path in self.source.iterdir()}
        with cpu_runtime_context():
            for low in (False, True):
                config = replace(self.config, low_vram=low)
                full, full_trace = self.execute(config, f'full-{low}')
                target = 5 if low else 6
                selected, selected_trace = self.execute(config, f'selected-{low}', target)
                self.assertEqual(full_trace, selected_trace)
                self.assertTrue(all(full_trace[key] for key in full_trace))
                position = 0
                for call in full_trace['stage2']:
                    count = call['inputs']['args'][0]['shape'][1]
                    if position <= target - 1 < position + count:
                        self.assertGreater(target - 1, position)
                        self.assertLess(target - 1, position + count - 1)
                    position += count
                self.assertEqual(selected['status'], 'preview_complete')
                self.assertIsNone(selected['end_to_end_fps'])
                self.assertEqual(selected['diagnostics']['frames_submitted'], 1)
                outcome = selected['diagnostics']['preview_selection']
                self.assertEqual((outcome['source_frame'], outcome['relative_frame'], outcome['context_frames_processed']), (target, target - 1, 12))
                self.assertTrue(outcome['full_context_processed'])
                with self.assertRaises(ValueError):
                    completed_view(self.job(config, f'selected-{low}'), selected)
                for role, directory in full['diagnostics']['outputs'].items():
                    expected = sorted(Path(directory).glob('*.exr'))[target - 1]
                    captured = list(Path(selected['diagnostics']['outputs'][role]).glob('*.exr'))
                    self.assertEqual(len(captured), 1)
                    pixels = cv2.imread(str(captured[0]), -1)
                    self.assertEqual(pixels.dtype, np.float32)
                    np.testing.assert_array_equal(pixels, cv2.imread(str(expected), -1))
                    if role == 'fg':
                        self.assertLess(float(pixels.min()), 0)
                        self.assertGreater(float(pixels.max()), 1)
        self.assertEqual(originals, {path: path.read_bytes() for path in self.source.iterdir()})

    def test_fixture_is_sensitive_to_real_conditioning_and_context_changes(self):
        def pixels(record, source_frame=6):
            index = source_frame - 1
            return cv2.imread(str(sorted(Path(record['diagnostics']['outputs']['alpha']).glob('*.exr'))[index]), -1)
        with cpu_runtime_context():
            baseline, _ = self.execute(self.config, 'sensitivity-full')
            reference = pixels(baseline)
            for name, config, options in (
                ('initial', self.config, {'initial': False}),
                ('full-hint', self.config, {'full': False}),
                ('carry', replace(self.config, carry_hint=False), {}),
                ('rvm', self.config, {'zero_rvm': True}),
                ('correction', replace(self.config, corrections_path=None), {}),
                ('hann', replace(self.config, hann_chunk=3, hann_stride=2), {}),
                ('stage2-group', replace(self.config, stage2_batch=1), {}),
            ):
                changed, _ = self.execute(config, 'sensitivity-' + name, **options)
                self.assertTrue(np.any(reference != pixels(changed)), name + ' must affect target pixels')
            editor = CorrectionEditor(self.job(self.config, 'reset-edit'), self.root, existing=self.config.corrections_path)
            reset_id = next(entry['id'] for entry in editor.document.to_dict()['entries'] if entry['kind'] == 'reset')
            editor.remove(reset_id)
            no_reset, _ = self.execute(replace(self.config, corrections_path=editor.save()), 'no-reset')
            self.assertTrue(np.any(pixels(baseline, 11) != pixels(no_reset, 11)), 'reset must affect post-boundary target')
            restarted, _ = self.execute(replace(self.config, corrections_path=None, start_frame=6, num_frames=1), 'restarted')
            restarted_pixels = cv2.imread(str(next(Path(restarted['diagnostics']['outputs']['alpha']).glob('*.exr'))), -1)
            self.assertTrue(np.any(reference != restarted_pixels), 'one-frame restart must not masquerade as full context')

    def test_prepared_conditioned_recipe_real_capture_and_dependency_drift(self):
        scratch = self.root / 'preview-scratch'
        scratch.mkdir()
        job = self.job(self.config, 'normal-export-never-written')
        with self.assertRaisesRegex(ValueError, 'separate'):
            prepare_preview(job, scratch_parent=self.hints, declared_fps=self.config.frame_dir_fps)
        recipe = prepare_preview(job, scratch_parent=scratch, declared_fps=self.config.frame_dir_fps, draft_scale=.5)
        prepared = recipe.job_for(6, 'final')
        try:
            self.assertEqual(prepared.job.initial_hint_path, job.initial_hint_path)
            self.assertEqual(prepared.job.hint_video_path, job.hint_video_path)
            effective, _ = prepared.job.configurations()
            self.assertEqual(effective, replace(job.configurations()[0], temp_dir=prepared.workspace.root / 'scratch'))
            with cpu_runtime_context():
                record, _ = self.execute(effective, 'unused', 6, job_request=prepared.job)
            prepared.check_source()
            view = selected_preview_view(prepared, record)
            self.assertEqual((view.start, view.count, view.options('Alpha', 6)['index']), (6, 1, 0))
            self.assertFalse(Path(job.output_dir).exists())
            changed = dict(record, status='complete')
            with self.assertRaises(ValueError):
                selected_preview_view(prepared, changed)
            record['diagnostics']['preview_selection']['full_context_processed'] = 1
            with self.assertRaises(ValueError):
                selected_preview_view(prepared, record)
        finally:
            prepared.workspace.cleanup()
        from corridorkey.corrections import load_corrections
        document = load_corrections(self.config.corrections_path)
        for path in (self.source / '12.png', self.hints / '12.png', self.initial,
                     Path(self.config.corrections_path), document.asset_paths()[0]):
            recipe = prepare_preview(job, scratch_parent=scratch, declared_fps=self.config.frame_dir_fps)
            before = path.read_bytes()
            try:
                path.write_bytes(before + b'drift')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    recipe.source.check(6)
            finally:
                path.write_bytes(before)
        self.assertEqual(list(scratch.iterdir()), [])

    def test_selected_controls_fail_closed_and_cancel_does_not_fake_success(self):
        job = self.job(self.config, 'cancelled-selected')
        for selected in (-1, True, 1.5):
            with (self.subTest(selected=selected),
                  InferenceSession(loader=lambda: self.fail('must not load'), disposer=lambda _: None) as session,
                  self.assertRaises(ValueError)):
                session.run(job, preview_frame=cast(Any, selected))
        with (InferenceSession(loader=lambda: self.fail('must not load'), runner=lambda *args: None) as session,
              self.assertRaisesRegex(ValueError, 'custom runners')):
            session.run(job, preview_frame=6)
        for action, selected in (('render', 6), ('preview', True), ('preview', None), ('preview', -1)):
            with self.assertRaises(ValueError):
                parse_request({'schema_version': 1, 'request_id': 'invalid', 'action': action,
                    'payload': {'job': job_to_wire(job), 'selected_frame': selected}})
        token = CancellationToken()
        disposed = []
        trace = {key: [] for key in ('iid', 'rvm', 'stage1', 'stage2')}
        model = temporal.Stage2IncrementalRecoveryTests.model(self, trace, [])
        def progress(event):
            if event.stage == 'preview_context_frames' and event.completed >= 7:
                token.cancel('superseded after capture; not successful partial render')
        with cpu_runtime_context(), InferenceSession(loader=lambda: model, disposer=disposed.append) as session:
            with self.assertRaises(CancelledError):
                session.run(job, preview_frame=6, cancellation=token, progress=progress)
            assert session.last_result is not None
            self.assertEqual(session.last_result.to_dict()['status'], 'cancelled')
            self.assertEqual(disposed, [model])

    def test_non_neural_worker_fixture_does_not_leak_selection_into_recovery(self):
        from corridorkey.worker_protocol import Request
        from tests.worker_fixture import FixtureEngine
        engine = FixtureEngine()
        engine.selected_frame = 6
        def recover(request, cancellation, progress):
            self.assertIsNone(engine.selected_frame)
            return {'status': 'fixture-only'}
        engine.recover = recover
        result = engine.run(Request('recover', 'recover', '{}'), CancellationToken(), lambda _: None)
        self.assertEqual(result['status'], 'fixture-only')

    def test_selected_output_telemetry_counts_only_captured_frame(self):
        job = replace(self.job(self.config, 'profiled-selected'), profile_stages=True)
        with cpu_runtime_context():
            record, _ = self.execute(self.config, 'unused', 6, job_request=job)
        telemetry = record['diagnostics']['output_telemetry']
        self.assertEqual(telemetry['expected_frames'], 1)
        self.assertEqual(telemetry['registered_frames'], 1)
        self.assertIsNone(record['end_to_end_fps'])

    def test_manifest_swap_after_parsing_cannot_rebind_dependency_graph(self):
        from corridorkey import preview_execution
        manifest = Path(self.config.corrections_path)
        replacement = json.loads(manifest.read_bytes())
        original_mask = (manifest.parent / replacement['entries'][0]['asset']['path']).resolve()
        new_mask = original_mask.with_name('different-valid-mask.png')
        new_mask.write_bytes(original_mask.read_bytes())
        replacement['entries'][0]['asset']['path'] = str(new_mask)
        scratch = self.root / 'swap-scratch'
        scratch.mkdir()
        read_file = preview_execution.read_file
        swapped = []

        def swap_manifest(pin, **kwargs):
            result = read_file(pin, **kwargs)
            if pin.path == str(manifest) and kwargs.get('capture') and not swapped:
                manifest.write_text(json.dumps(replacement))
                swapped.append(True)
            return result

        with (patch.object(preview_execution, 'read_file', side_effect=swap_manifest),
              self.assertRaisesRegex(ValueError, 'changed')):
            prepare_preview(self.job(self.config, 'never-export'), scratch_parent=scratch,
                            declared_fps=self.config.frame_dir_fps)
        self.assertEqual(swapped, [True])
        self.assertEqual(list(scratch.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
