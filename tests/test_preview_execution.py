"""Real local image preparation and owned-preview scratch contracts, no models."""
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.control import CancellationToken, CancelledError
from corridorkey.preview_execution import (
    PreviewPreparationError,
    PreviewWorkspace,
    create_preview_workspace,
    prepare_preview,
)
from corridorkey.progressive_preview import PreviewArtifact, ProgressivePreview
from corridorkey.session import JobRequest


class PreviewExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        self.scratch = self.root / 'scratch'
        self.scratch.mkdir()
        for name, value in (('frame10.png', 210), ('frame2.png', 100), ('frame1.png', 20)):
            self.assertTrue(cv2.imwrite(str(self.source / name), np.full((24, 32, 3), value, np.uint8)))
        self.job = JobRequest.create(self.source, self.root / 'exports',
                                     InferenceConfig(start_frame=1, num_frames=2, frame_dir_fps=25,
                                                     despill_strength=.27, alpha_refine_radius=2,
                                                     stage2_batch=3, outputs={'alpha', 'fg'},
                                                     temp_dir=self.root / 'original-temp'), WriterConfig())
        self.before = {p.name: p.read_bytes() for p in self.source.iterdir()}

    def tearDown(self):
        self.assertEqual(self.before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.temp.cleanup()

    def recipe(self, **kwargs):
        return prepare_preview(self.job, scratch_parent=self.scratch, declared_fps=25, **kwargs)

    def test_capture_is_readonly_frozen_order_range_and_declared_model_refs(self):
        with patch('cv2.imread', side_effect=AssertionError('decode')), patch('cv2.imdecode', side_effect=AssertionError('decode')):
            recipe = self.recipe(draft_scale=.5, model_references={'asset_root': str(self.root / 'assets'), 'generation': 'abc'})
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertEqual([Path(p).name for p in recipe.source.index.paths], ['frame1.png', 'frame2.png', 'frame10.png'])
        self.assertEqual((recipe.source.source_start, recipe.source.source_end), (1, 3))
        self.assertEqual(json.loads(recipe.source.requested_settings_json), asdict(self.job))
        self.assertEqual(recipe.context.source_identity, recipe.source.source_identity)
        self.assertEqual([stage.name for stage in recipe.stages], ['draft', 'final'])
        self.assertFalse(json.loads(recipe.source.provenance_json)['restarted_temporal_context'])
        self.assertIn('full-timeline inference', recipe.summary()['warning'])
        self.assertIn('NOT VERIFIED', json.loads(recipe.source.provenance_json)['model_compatibility'])

    def test_actual_png_full_context_draft_final_preserves_other_settings_and_original(self):
        recipe = self.recipe(draft_scale=.5)
        with patch('cv2.imread', side_effect=AssertionError('decode')), patch('cv2.imdecode', side_effect=AssertionError('decode')):
            draft = recipe.job_for(2, recipe.stages[0])
            final = recipe.job_for(2, 'final')
            again = recipe.job_for(2, 'final')
        self.assertNotEqual(final.workspace.root, again.workspace.root)
        self.assertEqual(final.identity, again.identity)
        self.assertNotEqual(final.identity, draft.identity)
        self.assertEqual(final.source_sha256, hashlib.sha256(self.before['frame10.png']).hexdigest())
        original, writer = self.job.configurations()
        for prepared, scale in ((draft, .5), (final, 1.0), (again, 1.0)):
            config, actual_writer = prepared.job.configurations()
            self.assertEqual(actual_writer, writer)
            expected = replace(original, model_scale=scale, output_format='exr',
                               temp_dir=prepared.workspace.root / 'scratch')
            self.assertEqual(config, expected)
            self.assertEqual(Path(prepared.job.output_dir), prepared.workspace.root / 'exports')
            self.assertEqual(json.loads(prepared.requested_settings_json), asdict(self.job))
            self.assertEqual(json.loads(prepared.provenance_json)['execution_range'], [1, 3])
            prepared.workspace.cleanup()
            self.assertTrue(prepared.workspace.closed)
            self.assertFalse(prepared.workspace.root.exists())
        self.assertEqual(self.job.configurations()[0], original)
        self.assertFalse(Path(self.job.output_dir).exists())

    def test_actual_float_exr_header_preserves_native_scale_colour_and_precision(self):
        exr_source = self.root / 'exr-source'
        exr_source.mkdir()
        path = exr_source / 'native.exr'
        self.assertTrue(cv2.imwrite(str(path), np.full((24, 32, 3), 1.7, np.float32)))
        config = replace(self.job.configurations()[0], start_frame=0, model_scale=.75, frame_dir_linear=True,
                         source_colour_profile='linear_bt709_d65', exr_dtype='float')
        job = JobRequest.create(exr_source, self.root / 'exr-export', config)
        recipe = prepare_preview(job, scratch_parent=self.scratch, declared_fps=25, draft_scale=.25)
        work = recipe.job_for(0, 'final')
        effective, _ = work.job.configurations()
        self.assertEqual((effective.exr_dtype, effective.frame_dir_linear, effective.model_scale,
                          effective.source_colour_profile), ('float', True, .75, 'linear_bt709_d65'))
        self.assertEqual(work.source_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        work.workspace.cleanup()

    def test_explicit_fps_context_refusals_and_invalid_draft_do_not_create_work(self):
        for scale in (0, -1, 1, True, float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.recipe(draft_scale=scale)
        with self.assertRaisesRegex(ValueError, 'FPS'):
            prepare_preview(self.job, scratch_parent=self.scratch, declared_fps=24)
        config, _ = self.job.configurations()
        for changed in (replace(config, quick_key=True),):
            with self.assertRaisesRegex(ValueError, 'context'):
                prepare_preview(JobRequest.create(self.source, self.root / 'out', changed), scratch_parent=self.scratch, declared_fps=25)
        for hint_field in ('initial_hint_path', 'hint_video_path'):
            with self.assertRaises(FileNotFoundError):
                prepare_preview(replace(self.job, **{hint_field: str(self.root / 'hint.png')}), scratch_parent=self.scratch, declared_fps=25)
        video = self.root / 'clip.mov'
        video.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'video/VFR'):
            prepare_preview(replace(self.job, input_path=str(video)), scratch_parent=self.scratch, declared_fps=25)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_strict_original_duplicate_json_and_bad_limits_refuse(self):
        duplicate = replace(self.job, config_json=self.job.config_json[:-1] + ',"model_scale":0.25}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prepare_preview(duplicate, scratch_parent=self.scratch, declared_fps=25)
        for kwargs in ({'max_frames': 2}, {'max_entries': 2}, {'max_pixels': True}, {'max_encoded_bytes': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.recipe(**kwargs)
        for parent in (self.source, Path(self.job.output_dir)):
            if not parent.exists():
                parent.mkdir()
            with self.assertRaisesRegex(ValueError, 'separate'):
                prepare_preview(self.job, scratch_parent=parent, declared_fps=25)

    def test_frame_pixel_encoded_and_estimated_output_bounds_before_workspace(self):
        with self.assertRaisesRegex(ValueError, 'encoded/read byte bounds'):
            self.recipe(max_encoded_bytes=8)
        for kwargs, message in (({'max_pixels': 100}, 'geometry'), ({'max_cleanup_bytes': 1000}, 'EXR')):
            recipe = self.recipe(**kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                recipe.job_for(1, 'final')
            self.assertEqual(list(self.scratch.iterdir()), [])
        recipe = self.recipe()
        for frame in (0, 3, True, -1):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                recipe.job_for(frame, 'final')
        with self.assertRaises(ValueError):
            recipe.job_for(1, 'draft')

    def test_directory_and_indexed_source_changes_fence_without_rescan(self):
        recipe = self.recipe()
        changed = self.source / 'frame2.png'
        original = changed.read_bytes()
        try:
            changed.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                recipe.job_for(1, 'final')
        finally:
            changed.write_bytes(original)
        recipe = self.recipe()
        extra = self.source / 'frame0.png'
        try:
            extra.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'membership'):
                recipe.source.check(1)
        finally:
            extra.unlink()
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_cancel_capture_native_admission_and_workspace_admission_leave_no_work(self):
        token = CancellationToken()
        token.cancel('test')
        with self.assertRaises(CancelledError):
            self.recipe(check=token.check)
        recipe = self.recipe()
        with self.assertRaises(CancelledError):
            recipe.job_for(1, 'final', check=token.check)
        with patch('corridorkey.preview_execution.shutil.disk_usage', return_value=type('Disk', (), {'free': 0})()), self.assertRaisesRegex(ValueError, 'free-space'):
            recipe.job_for(1, 'final')
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_workspace_exact_count_bytes_cleanup_and_idempotent_release(self):
        work = create_preview_workspace(self.scratch, max_cleanup_files=4, max_cleanup_bytes=100)
        (work.root / 'nested').mkdir()
        (work.root / 'nested' / 'data').write_bytes(b'abc')
        self.assertEqual(work.check_budget(), {'entries': 3, 'bytes': 35})
        work.cleanup()
        work.cleanup()
        self.assertTrue(work.closed)
        self.assertIsNone(work.cleanup_error)

    def test_cleanup_refuses_aliases_hardlinks_count_bytes_or_replaced_root(self):
        for kind in ('symlink', 'hardlink', 'count', 'bytes', 'replace'):
            with self.subTest(kind=kind):
                work = create_preview_workspace(self.scratch, max_cleanup_files=3, max_cleanup_bytes=100)
                sentinel = self.root / f'outside-{kind}'
                sentinel.write_bytes(b'preserve')
                if kind == 'symlink':
                    (work.root / 'bad').symlink_to(sentinel)
                elif kind == 'hardlink':
                    os.link(sentinel, work.root / 'bad')
                elif kind == 'count':
                    for index in range(3):
                        (work.root / str(index)).write_bytes(b'x')
                elif kind == 'bytes':
                    (work.root / 'big').write_bytes(bytes(100))
                else:
                    work.root.rename(work.root.with_name(work.root.name + '-retained'))
                    work.root.mkdir()
                    (work.root / 'foreign').write_bytes(b'foreign')
                with self.assertRaises(ValueError):
                    work.cleanup()
                self.assertFalse(work.closed)
                self.assertIsNotNone(work.cleanup_error)
                self.assertTrue(work.root.exists())
                self.assertEqual(sentinel.read_bytes(), b'preserve')

    def test_model_nested_file_creation_preference_and_content_drift_are_fenced(self):
        assets = self.root / 'assets'
        nested = assets / 'vfms/cradio/C-RADIOv4-SO400M/nested'
        nested.mkdir(parents=True)
        weights = assets / 'weights'
        weights.mkdir()
        checkpoint = weights / 'corridorkey_v2.safetensors'
        checkpoint.write_bytes(b'fixture weights, no model load')
        native_model = nested / 'model.safetensors'
        native_model.write_bytes(b'fixture nested checkpoint')
        refs = {'asset_root': str(assets)}
        recipe = self.recipe(model_references=refs)
        work = recipe.job_for(1, 'final')
        native_model.write_bytes(b'changed nested checkpoint')
        with self.assertRaisesRegex(ValueError, 'model metadata'):
            recipe.source.check(1)
        work.workspace.cleanup()
        recipe = self.recipe(model_references=refs)
        (nested / 'added-config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'model metadata'):
            recipe.source.check(1)
        recipe = self.recipe(model_references=refs)
        (weights / 'foundation_stats.pt').write_bytes(b'new preferred legacy weights')
        with self.assertRaisesRegex(ValueError, 'model paths'):
            recipe.source.check(1)

    def test_model_pin_bound_and_prepared_frame_content_hash_without_full_rescan(self):
        assets = self.root / 'bounded-assets'
        tree = assets / 'vfms/cradio/C-RADIOv4-SO400M'
        tree.mkdir(parents=True)
        for index in range(2048):
            (tree / str(index)).touch()
        with self.assertRaisesRegex(ValueError, 'pin count'):
            self.recipe(model_references={'asset_root': str(assets)})
        recipe = self.recipe()
        work = recipe.job_for(1, 'final')
        selected = self.source / 'frame2.png'
        before = selected.stat()
        original = selected.read_bytes()
        try:
            # Simulate same-length valid replacement with restored mtime, beyond
            # the sequence index stat contract; per-job content fence still refuses.
            encoded = cv2.imencode('.png', np.full((24, 32, 3), 101, np.uint8))[1].tobytes()
            selected.write_bytes(encoded)
            os.utime(selected, ns=(before.st_atime_ns, before.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'changed'):
                work.check_source()
        finally:
            selected.write_bytes(original)
            work.workspace.cleanup()

    def test_preparation_double_fault_retains_primary_and_typed_owned_workspace(self):
        recipe = self.recipe()
        with patch.object(JobRequest, 'create', side_effect=ValueError('primary validation failure')), patch.object(PreviewWorkspace, 'cleanup', side_effect=OSError('cleanup refused')), self.assertRaises(PreviewPreparationError) as caught:
            recipe.job_for(1, 'final')
        failure = caught.exception
        self.assertIsInstance(failure.primary_error, ValueError)
        self.assertIsInstance(failure.cleanup_failure, OSError)
        self.assertIs(failure.__cause__, failure.primary_error)
        self.assertIn(str(failure.workspace.root), str(failure))
        self.assertTrue(failure.workspace.root.is_dir())
        self.assertTrue((failure.workspace.root / '.preview-owner').exists())
        self.assertIn('primary validation failure', str(failure))
        failure.workspace.cleanup()
        self.assertTrue(failure.workspace.closed)

    def test_restored_mtime_same_size_source_change_invalidates_scheduler_reuse(self):
        recipe = self.recipe()
        scheduler = ProgressivePreview(recipe.context, recipe.stages, enabled=True)
        request = scheduler.next_request()
        assert request is not None
        self.assertTrue(scheduler.complete(request, PreviewArtifact(request.identity, b'old pixels')))
        self.assertEqual(scheduler.snapshot().status, 'final')
        selected = self.source / 'frame2.png'
        info, original = selected.stat(), selected.read_bytes()
        replacement = cv2.imencode('.png', np.full((24, 32, 3), 101, np.uint8))[1].tobytes()
        self.assertEqual(len(replacement), len(original))
        try:
            selected.write_bytes(replacement)
            os.utime(selected, ns=(info.st_atime_ns, info.st_mtime_ns))
            self.assertNotEqual(selected.stat().st_ctime_ns, info.st_ctime_ns)
            with self.assertRaisesRegex(ValueError, 'changed'):
                recipe.source.check(1)
            updated = self.recipe()
            self.assertNotEqual(updated.source.source_identity, recipe.source.source_identity)
            self.assertNotEqual(updated.context.identity, recipe.context.identity)
            scheduler.bind(updated.context, updated.stages)
            self.assertNotEqual(scheduler.snapshot().status, 'final')
            fresh = scheduler.next_request()
            assert fresh is not None
            self.assertNotEqual(fresh.cache_identity, request.cache_identity)
        finally:
            selected.write_bytes(original)

    def test_final_rmdir_failure_retains_retired_proof_for_safe_empty_root_retry(self):
        work = create_preview_workspace(self.scratch)
        (work.root / 'data').write_bytes(b'owned')
        original_rmdir = Path.rmdir

        def fail_final(path):
            if path == work.root:
                raise OSError('final rmdir failure')
            return original_rmdir(path)

        with patch.object(Path, 'rmdir', fail_final), self.assertRaisesRegex(OSError, 'final rmdir'):
            work.cleanup()
        self.assertFalse(work.closed)
        self.assertEqual(list(work.root.iterdir()), [])
        self.assertIn('final rmdir', work.cleanup_error or '')
        work.cleanup()
        self.assertTrue(work.closed)
        self.assertFalse(work.root.exists())
        self.assertIsNone(work.cleanup_error)

    def test_retired_workspace_never_adopts_new_contents_or_replaced_inode(self):
        for kind in ('new contents', 'replacement'):
            with self.subTest(kind=kind):
                work = create_preview_workspace(self.scratch)
                with patch.object(Path, 'rmdir', side_effect=OSError('retain empty root')), self.assertRaises(OSError):
                    work.cleanup()
                if kind == 'replacement':
                    work.root.rename(work.root.with_name(work.root.name + '-original'))
                    work.root.mkdir()
                foreign = work.root / 'foreign'
                foreign.write_bytes(b'preserve')
                with self.assertRaisesRegex(ValueError, 'new contents|replaced'):
                    work.cleanup()
                self.assertEqual(foreign.read_bytes(), b'preserve')
                self.assertFalse(work.closed)

    def test_marker_initialization_and_final_cleanup_double_fault_can_retry_empty_root(self):
        original_open = Path.open

        def fail_marker(path, *args, **kwargs):
            if path.name == '.preview-owner' and args and args[0] == 'xb':
                raise OSError('marker initialization failure')
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'open', fail_marker), patch.object(Path, 'rmdir', side_effect=OSError('final cleanup failure')), self.assertRaises(PreviewPreparationError) as caught:
            create_preview_workspace(self.scratch)
        failure = caught.exception
        self.assertIn('marker initialization', str(failure.primary_error))
        self.assertTrue(failure.workspace.root.is_dir())
        self.assertEqual(list(failure.workspace.root.iterdir()), [])
        failure.workspace.cleanup()
        self.assertTrue(failure.workspace.closed)

    def test_marker_minimum_byte_budget_refuses_before_creating_directory(self):
        for budget in (1, 31):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'mandatory 32-byte'):
                create_preview_workspace(self.scratch, max_cleanup_bytes=budget)
            self.assertEqual(list(self.scratch.iterdir()), [])
        exact = create_preview_workspace(self.scratch, max_cleanup_bytes=32)
        self.assertEqual(exact.check_budget(), {'entries': 1, 'bytes': 32})
        exact.cleanup()
        self.assertTrue(exact.closed)

    def test_cleanup_cancel_and_unlink_failure_retain_discoverable_owner(self):
        work = create_preview_workspace(self.scratch)
        (work.root / 'data').write_bytes(b'keep')
        token = CancellationToken()
        token.cancel('keep workspace')
        with self.assertRaises(CancelledError):
            work.cleanup(token.check)
        self.assertTrue((work.root / '.preview-owner').exists())
        with patch.object(Path, 'unlink', side_effect=OSError('unlink failed')), self.assertRaises(OSError):
            work.cleanup()
        self.assertIn('unlink failed', work.cleanup_error or '')
        self.assertTrue((work.root / '.preview-owner').exists())
        work.cleanup()
        self.assertTrue(work.closed)


if __name__ == '__main__':
    unittest.main()
