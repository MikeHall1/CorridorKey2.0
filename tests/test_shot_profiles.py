"""Bounded source-bound shot recommendations never change temporal recipes."""
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest
from corridorkey.shot_profiles import ProfilePlan, base_recipe_key


class ShotProfileCodecTests(unittest.TestCase):
    def setUp(self):
        self.job = JobRequest.create('/tmp/source', '/tmp/output',
            InferenceConfig(start_frame=10, num_frames=6, stage2_batch=4))
        self.data = {'schema_version': 1, 'kind': 'shot_processing_profiles', 'enabled': True,
            'recipe_sha256': base_recipe_key(self.job), 'source_sha256': 'a' * 64,
            'source_range': [10, 16], 'read_budget': 1024,
            'rows': [{'range': [10, 13], 'enabled': True, 'stage2_batch': 2,
                      'alpha_expand_radius': 0, 'alpha_feather_radius': 1,
                      'signals': {'mean_rgb_change': .01, 'mean_edge_detail': .1}},
                     {'range': [13, 16], 'enabled': False, 'stage2_batch': 1,
                      'alpha_expand_radius': 1, 'alpha_feather_radius': 0,
                      'signals': {'mean_rgb_change': .5, 'mean_edge_detail': .2}}],
            'analysis': {'calibrated': False, 'resource_fit': 'unavailable'}}

    def test_absolute_ranges_gaps_disable_and_batch_bound(self):
        plan = ProfilePlan.parse(self.data)
        config, _ = self.job.configurations()
        self.assertEqual(plan.settings(10, config), (2, 0, 1))
        self.assertEqual(plan.settings(13, config), (4, 0, 0))
        self.assertEqual(plan.settings(9, config), (4, 0, 0))
        disabled = ProfilePlan.parse({**self.data, 'enabled': False})
        self.assertEqual(disabled.settings(10, config), (4, 0, 0))
        with self.assertRaises(ValueError):
            plan.validate_recipe(replace(self.job, config_json=self.job.config_json.replace('"stage2_batch": 4', '"stage2_batch": 1')))

    def test_strict_schema_ranges_types_and_bounds(self):
        for mutation in ('overlap', 'foreign', 'bool', 'extra', 'missing', 'too_many'):
            data = copy.deepcopy(self.data)
            if mutation == 'overlap': data['rows'][1]['range'][0] = 12
            if mutation == 'foreign': data['rows'][1]['range'][1] = 17
            if mutation == 'bool': data['rows'][0]['stage2_batch'] = True
            if mutation == 'extra': data['rows'][0]['model_scale'] = .5
            if mutation == 'missing': del data['source_sha256']
            if mutation == 'too_many': data['rows'] *= 300
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ProfilePlan.parse(data)

    def test_binding_excludes_only_payload_and_destination(self):
        config, writer = self.job.configurations()
        with_profiles = JobRequest.create(self.job.input_path, '/tmp/new-output',
            replace(config, shot_profiles=self.data), writer)
        self.assertEqual(base_recipe_key(with_profiles), base_recipe_key(self.job))
        for changes in ({'start_frame': 11}, {'alpha_feather_radius': 1}, {'stage2_batch': 3},
                        {'quick_key': True}, {'temp_dir': '/tmp/another'}):
            changed = JobRequest.create(self.job.input_path, self.job.output_dir, replace(config, **changes), writer)
            with self.assertRaises(ValueError):
                ProfilePlan.parse(self.data).validate_recipe(changed)

    def test_editor_strict_json_rejects_duplicate_nonfinite_and_deep_text(self):
        import json
        raw = json.dumps(self.data)
        self.assertEqual(ProfilePlan.from_bytes(raw.encode()), ProfilePlan.parse(self.data))
        duplicate = raw.replace('"enabled": true', '"enabled": false, "enabled": true', 1)
        for invalid in (duplicate, raw.replace('0.01', 'NaN'), '[' * 30 + '0' + ']' * 30):
            with self.subTest(text=invalid[:50]), self.assertRaises((ValueError, TypeError)):
                ProfilePlan.from_bytes(invalid.encode())
        with self.assertRaises(ValueError):
            ProfilePlan.from_bytes(b' ' * (1024 * 1024 + 1))

    def test_gap_and_direct_constructor_validation(self):
        data = copy.deepcopy(self.data)
        data['rows'][1]['range'][0] = 14
        data['rows'][1]['enabled'] = True
        config, _ = self.job.configurations()
        plan = ProfilePlan.parse(data)
        self.assertEqual(plan.settings(13, config), (4, 0, 0))
        for raw in ('{}', plan.payload + ' ', ' ' * (1024 * 1024 + 1)):
            with self.assertRaises(ValueError):
                ProfilePlan(raw)


class ShotProfileAnalysisTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.source = self.root / 'frames'
        self.source.mkdir()
        for frame, value in enumerate((30, 30, 30, 220, 220, 220)):
            cv2.imwrite(str(self.source / f'{frame}.png'), np.full((16, 32, 3), value, np.uint8))
        self.job = JobRequest.create(self.source, self.root / 'outputs',
            InferenceConfig(start_frame=1, num_frames=4, stage2_batch=3, output_format='exr', outputs={'alpha'}))

    def test_source_analysis_absolute_shots_proposals_and_save_reopen(self):
        from corridorkey.shot_profiles import (
            analyze_job,
            apply_profiles,
            load_profiles,
            save_profiles,
        )
        from corridorkey.worker_protocol import job_from_wire, job_to_wire
        before = self.job.config_json
        plan = analyze_job(self.job)
        data = plan.to_dict()
        self.assertEqual(data['source_range'], [1, 5])
        self.assertEqual([row['range'] for row in data['rows']], [[1, 3], [3, 5]])
        self.assertFalse(data['analysis']['calibrated'])
        self.assertEqual(data['analysis']['resource_fit'], 'unavailable')
        updated = apply_profiles(self.job, plan)
        self.assertEqual(self.job.config_json, before)
        self.assertEqual(updated.initial_hint_path, self.job.initial_hint_path)
        self.assertEqual(updated.configurations()[0].start_frame, 1)
        self.assertEqual(updated.configurations()[0].num_frames, 4)
        frozen = job_from_wire(job_to_wire(updated))
        data['rows'][0]['alpha_feather_radius'] = 8
        frozen_profiles = frozen.configurations()[0].shot_profiles
        assert frozen_profiles is not None
        self.assertEqual(frozen_profiles['rows'][0]['alpha_feather_radius'], 1)
        destination = self.root / 'profiles.json'
        save_profiles(destination, plan, updated)
        original = destination.read_bytes()
        base, loaded = load_profiles(destination)
        self.assertEqual(loaded, plan)
        self.assertIsNone(base.configurations()[0].shot_profiles)
        self.assertEqual(job_to_wire(apply_profiles(base, loaded)), job_to_wire(updated))
        with self.assertRaises(FileExistsError):
            save_profiles(destination, plan, self.job)
        self.assertEqual(destination.read_bytes(), original)
        for target in (self.source / 'new.json', self.root / 'outputs/new.json'):
            with self.assertRaises(ValueError):
                save_profiles(target, plan, self.job)

    def test_source_drift_membership_budget_cancel_and_disabled(self):
        from corridorkey.control import CancellationToken, CancelledError
        from corridorkey.shot_profiles import analyze_job, apply_profiles
        with self.assertRaises(ValueError):
            analyze_job(self.job, read_budget=1)
        token = CancellationToken()
        token.cancel('stop')
        with self.assertRaises(CancelledError):
            analyze_job(self.job, check=token.check)
        plan = analyze_job(self.job)
        self.cv2.imwrite(str(self.source / '0.png'), self.np.full((16, 32, 3), 31, self.np.uint8))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            apply_profiles(self.job, plan)
        disabled = ProfilePlan.parse({**plan.to_dict(), 'enabled': False})
        disabled_profiles = apply_profiles(self.job, disabled).configurations()[0].shot_profiles
        assert disabled_profiles is not None
        self.assertFalse(disabled_profiles['enabled'])

    def test_effective_batch_caps_do_not_split_equal_adjacent_rows(self):
        from corridorkey.shot_profiles import analyze_job
        data = analyze_job(self.job).to_dict()
        config, _ = self.job.configurations()
        items = [(i, None) for i in range(4)]
        plan = ProfilePlan.parse(data)
        self.assertEqual([len(group) for group in plan.batches(items, config, 1)], [3, 1])
        data['rows'][1]['stage2_batch'] = 1
        plan = ProfilePlan.parse(data)
        groups = list(plan.batches(items, config, 1))
        self.assertEqual([len(group) for group in groups], [2, 1, 1])
        self.assertEqual([item[0] for group in groups for item in group], list(range(4)))

    def test_existing_project_batch_and_queue_codecs_preserve_profiles(self):
        import json

        from corridorkey.batch import jobs_from_data
        from corridorkey.desktop_state import load_project, save_project
        from corridorkey.queue_store import QueueStore
        from corridorkey.shot_profiles import analyze_job, apply_profiles
        from corridorkey.worker_protocol import job_from_wire, job_to_wire
        updated = apply_profiles(self.job, analyze_job(self.job))
        project = self.root / 'project.json'
        save_project(project, updated, 2)
        self.assertEqual(job_to_wire(load_project(project)[0]), job_to_wire(updated))
        self.assertEqual(load_project(project)[1], 2)
        settings = json.loads(updated.config_json)
        batch = {'schema_version': 1, 'kind': 'inference_batch', 'jobs': [{
            'input': updated.input_path, 'output_dir': updated.output_dir, 'settings': settings}]}
        jobs, _ = jobs_from_data(batch, self.root / 'batch.json')
        self.assertEqual(job_to_wire(jobs[0]), job_to_wire(updated))
        queue = QueueStore.create(self.root)
        store = QueueStore(queue)
        try:
            store.start_session()
            payload = store.prepare('profile', 'render', {'job': job_to_wire(updated)})
            settings['shot_profiles']['rows'][0]['alpha_feather_radius'] = 9
            self.assertNotEqual(payload['job']['config_json'], json.dumps(settings))
        finally:
            store.close()
        record = QueueStore.inspect(queue, request_id='profile')
        self.assertEqual(job_to_wire(job_from_wire(record['request']['payload']['job'])), job_to_wire(updated))

    def test_enabled_profiles_rejected_at_derived_request_boundaries(self):
        from corridorkey.asset_inventory import AssetSelection
        from corridorkey.calibration import CalibrationProfile, inventory_calibration
        from corridorkey.portable_project import inventory_project
        from corridorkey.preview_execution import prepare_preview
        from corridorkey.readiness import prepare_short_render
        from corridorkey.shot_profiles import analyze_job, apply_profiles
        job = apply_profiles(self.job, analyze_job(self.job))
        operations = [lambda: prepare_short_render(job, 1, 2),
            lambda: prepare_preview(job, scratch_parent=self.root, declared_fps=24.),
            lambda: inventory_project(job, 0),
            lambda: inventory_calibration(job, sample_start=1, frame_count=2,
                profiles=(CalibrationProfile.capture('native', {}),), asset_selection=AssetSelection(self.root),
                scratch_parent=self.root, report_parent=self.root)]
        for operation in operations:
            with self.assertRaisesRegex(ValueError, 'disable profiles explicitly'):
                operation()
        profiles = job.configurations()[0].shot_profiles
        assert profiles is not None
        disabled = replace(job.configurations()[0], shot_profiles={**profiles, 'enabled': False})
        self.assertEqual(prepare_short_render(JobRequest.create(job.input_path, job.output_dir, disabled), 1, 2).configurations()[0].num_frames, 2)

    def test_diagnostic_range_rejects_enabled_profiles_before_reserving(self):
        from corridorkey.diagnostic_ranges import inspect_diagnostics, prepare_range
        from corridorkey.diagnostics import json_value
        from corridorkey.shot_profiles import ProfilePlan, base_recipe_key
        from tests.test_diagnostic_ranges import make_diagnostic_fixture
        folder = self.root / 'diagnostics'
        folder.mkdir()
        job, record, parent = make_diagnostic_fixture(folder)
        data = {
            'schema_version': 1, 'kind': 'shot_processing_profiles', 'enabled': True,
            'recipe_sha256': base_recipe_key(job), 'source_sha256': 'a' * 64, 'source_range': [2, 7],
            'read_budget': 1024, 'rows': [{'range': [2, 7], 'enabled': True, 'stage2_batch': 1,
                'alpha_expand_radius': 0, 'alpha_feather_radius': 0,
                'signals': {'mean_rgb_change': 0., 'mean_edge_detail': 0.}}],
            'analysis': {'calibrated': False, 'resource_fit': 'unavailable'}}
        cfg, writer = job.configurations()
        updated = JobRequest.create(job.input_path, job.output_dir, replace(cfg, shot_profiles=ProfilePlan.parse(data).to_dict()), writer)
        record['requested_settings'] = json_value(updated.configurations()[0])
        snapshot = inspect_diagnostics(updated, record)
        before = set(parent.iterdir())
        with self.assertRaisesRegex(ValueError, 'disable profiles explicitly'):
            prepare_range(snapshot, 2, 5, parent)
        self.assertEqual(set(parent.iterdir()), before)

    def test_detail_cut_fragmentation_and_source_change_during_analysis(self):
        from unittest.mock import patch

        from corridorkey.shot_profiles import analyze_job
        np, cv2 = self.np, self.cv2
        for frame in range(6):
            texture = ((np.indices((16, 32))[1] + frame) % 3 * 15 + 90).astype(np.uint8)
            cv2.imwrite(str(self.source / f'{frame}.png'), np.repeat(texture[..., None], 3, axis=2))
        data = analyze_job(self.job).to_dict()
        self.assertGreater(data['rows'][0]['signals']['mean_edge_detail'], .02)
        self.assertEqual(data['rows'][0]['alpha_feather_radius'], 0)
        for frame, value in enumerate((30, 30, 30, 220, 220, 220)):
            cv2.imwrite(str(self.source / f'{frame}.png'), np.full((16, 32, 3), value, np.uint8))
        with patch('corridorkey.shot_profiles.MAX_ROWS', 1), self.assertRaisesRegex(ValueError, '256 rows'):
            analyze_job(self.job)
        changed = False
        def progress(event):
            nonlocal changed
            if event.stage == 'shot_analysis' and event.completed == 2 and not changed:
                cv2.imwrite(str(self.source / '4.png'), np.full((16, 32, 3), 100, np.uint8))
                changed = True
        with self.assertRaisesRegex(ValueError, 'source changed'):
            analyze_job(self.job, progress=progress)

    def test_ofx_rebinding_rejects_enabled_profiles(self):
        from corridorkey.ofx_capture import capture_job_config, verify_capture
        from corridorkey.shot_profiles import analyze_job, apply_profiles
        from tests.test_ofx_capture import capture_fixture
        directory = self.root / 'host-fixture'
        directory.mkdir()
        manifest, *_ = capture_fixture(directory)
        plan = verify_capture(manifest, read_budget=1024**2)
        job = apply_profiles(self.job, analyze_job(self.job))
        with self.assertRaisesRegex(ValueError, 'disable profiles explicitly'):
            capture_job_config(plan, job.configurations()[0])

    def test_generated_video_selected_range_uses_existing_decoder_without_models(self):
        from corridorkey.shot_profiles import analyze_job, apply_profiles
        video = self.root / 'clip.avi'
        writer = self.cv2.VideoWriter(str(video), self.cv2.VideoWriter_fourcc(*'MJPG'), 24., (32, 16))
        self.assertTrue(writer.isOpened(), 'Local MJPG fixture encoder unavailable')
        try:
            for value in (30, 30, 30, 220, 220, 220):
                writer.write(self.np.full((16, 32, 3), value, self.np.uint8))
        finally:
            writer.release()
        job = JobRequest.create(video, self.root / 'video-output', InferenceConfig(start_frame=1,
            num_frames=4, output_format='exr', outputs={'alpha'}))
        before = video.read_bytes()
        plan = analyze_job(job)
        self.assertEqual(plan.to_dict()['source_range'], [1, 5])
        self.assertEqual([row['range'] for row in plan.to_dict()['rows']], [[1, 3], [3, 5]])
        self.assertEqual(apply_profiles(job, plan).configurations()[0].start_frame, 1)
        self.assertEqual(video.read_bytes(), before)
        changed = plan.to_dict()
        changed['source_range'] = [0, 4]
        changed['rows'] = [{**changed['rows'][0], 'range': [0, 4]}]
        with self.assertRaisesRegex(ValueError, 'source changed'):
            apply_profiles(job, ProfilePlan.parse(changed))
