"""Connected CPU neural substitutes; no pretrained/GPU quality or speed claims."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


class ShotProfileRuntimeTests(unittest.TestCase):
    def test_one_timeline_profiles_match_per_frame_clean_results_all_paths(self):
        import cv2
        import numpy as np
        import torch

        from corridorkey import draft
        from corridorkey.config import InferenceConfig
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.identities import file_sha256
        from corridorkey.motion import Propagation
        from corridorkey.session import JobRequest
        from corridorkey.shot_profiles import ProfilePlan, analyze_job
        from tests import test_spatial_draft as fixture

        with tempfile.TemporaryDirectory() as folder:
            helper = fixture.SpatialDraftTests()
            helper.root = Path(folder).resolve()
            source = helper.source(exr=True)
            for frame in (5, 6):
                (source / f'{frame:04}.exr').write_bytes((source / '0004.exr').read_bytes())
            source_before = {p.name: file_sha256(p) for p in source.iterdir()}
            authored_job = JobRequest.create(source, helper.root / 'unused', InferenceConfig(start_frame=1,
                num_frames=6, frame_dir_linear=True, source_colour_profile='linear_bt2020_d65', output_format='exr'))
            editor = CorrectionEditor(authored_job, helper.root)
            region = np.zeros((65, 97), np.uint8)
            region[:, 18:78] = 255
            editor.commit_brush(region, kind='region', frame=0, strength=1., direction='both', span=[0, 3])
            editor.commit_brush(region, kind='region', frame=3, strength=1., direction='both', span=[3, 6])
            keep = np.zeros_like(region)
            keep[20:30, 25:35] = 255
            editor.commit_brush(keep, kind='keep', frame=2, strength=.5, direction='both', span=[2, 3])
            editor.reset(3)
            correction = editor.save()
            correction_before = correction.read_bytes()
            first = helper.root / 'first.png'
            full = helper.root / 'hints'
            full.mkdir()
            cv2.imwrite(str(first), np.full((65, 97), 96, np.uint8))
            for frame in range(7):
                cv2.imwrite(str(full / f'{frame:04}.png'), np.full((65, 97), 70 + frame * 8, np.uint8))
            original_factory = fixture.make_spatial_cpu_model
            batch_sizes = []
            def patterned_model(**kwargs):
                model = original_factory(**kwargs)
                forward = model.stage2.forward
                def patterned(rgb, baton):
                    batch_sizes.append(rgb.shape[1])
                    result = forward(rgb, baton)
                    values = torch.arange(result.shape[-1]).remainder(13).float() / 12
                    result[:, :, 0] = values.view(1, 1, 1, -1)
                    return result
                model.stage2.forward = patterned
                return model
            old_offline = draft.offline_frames
            def uncertain(*args, **kwargs):
                alpha = np.tile(np.arange(97, dtype=np.float32) % 13 / 12, (65, 1))
                return Propagation(alpha, np.zeros_like(alpha), np.ones_like(alpha, dtype=bool),
                    {'flow_evaluations': 4, 'motion_seconds': 0., 'uncertain_fraction': 1., 'motion_fraction': 0.})
            def forced(*args, **kwargs):
                return old_offline(*args, interpolate=uncertain, **kwargs)
            cases = ((1., False, False), (1., True, False), (.5, False, False),
                     (1., False, True), (.5, True, True))
            with patch.object(fixture, 'make_spatial_cpu_model', patterned_model), patch.object(draft, 'offline_frames', forced):
                for scale, low, quick in cases:
                    with self.subTest(scale=scale, low=low, quick=quick):
                        overrides = {'start_frame': 1, 'num_frames': 6, 'source_colour_profile': 'linear_bt2020_d65',
                            'cutout_alpha_mode': 'premultiplied', 'quick_cut_frames': [3] if quick else [],
                            'quick_reset_frames': [3] if quick else []}
                        common = {'scale': scale, 'low': low, 'quick': quick, 'correction': correction,
                                  'outputs': {'alpha', 'fg', 'cutout'}, 'first_hint': str(first), 'full_hint': str(full)}
                        base_model, baseline, _ = helper.run_job(source, overrides=overrides, **common)
                        assert baseline.effective is not None
                        settings = json.loads(json.dumps(baseline.effective))
                        settings['outputs'] = set(settings['outputs'])
                        settings['temp_dir'] = Path(settings['temp_dir'])
                        base = JobRequest.create(source, helper.root / 'ignored', InferenceConfig(**settings),
                            initial_hint_path=first, hint_video_path=full)
                        data = analyze_job(base).to_dict()
                        data['rows'] = [{'range': [2, 6], 'enabled': True, 'stage2_batch': 1,
                            'alpha_expand_radius': 1, 'alpha_feather_radius': 2,
                            'signals': {'mean_rgb_change': 0., 'mean_edge_detail': 0.}}]
                        plan = ProfilePlan.parse(data)
                        batch_sizes.clear()
                        model, profiled, _ = helper.run_job(source, overrides={**overrides, 'shot_profiles': plan.to_dict()}, **common)
                        self.assertTrue(batch_sizes)
                        self.assertEqual(set(batch_sizes), {1})
                        _, finished, _ = helper.run_job(source, overrides={**overrides,
                            'alpha_expand_radius': 1, 'alpha_feather_radius': 2}, **common)
                        self.assertEqual(len(model.stage1.seen_hints), len(base_model.stage1.seen_hints))
                        for left, right in zip(model.stage1.seen_hints, base_model.stage1.seen_hints):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                        self.assertEqual(model.stage1.shapes, base_model.stage1.shapes)
                        assert profiled.alpha_refinement is not None
                        self.assertTrue(profiled.alpha_refinement['edge_finishing']['active'])
                        self.assertEqual(profiled.alpha_refinement['edge_finishing']['frames_evaluated'], 4)
                        for name, directory in profiled.outputs.items():
                            for index, path in enumerate(sorted(Path(directory).glob('*.exr'))):
                                expected_record = finished if 1 <= index < 5 else baseline
                                expected = sorted(Path(expected_record.outputs[name]).glob('*.exr'))[index]
                                np.testing.assert_array_equal(cv2.imread(str(path), -1), cv2.imread(str(expected), -1))
                        if quick:
                            assert profiled.quick_key is not None and baseline.quick_key is not None
                            self.assertEqual(profiled.quick_key['effective_cuts'], baseline.quick_key['effective_cuts'])
                            self.assertTrue(profiled.quick_key['extra_neural_frames'])
                        else:
                            foreground = cv2.imread(str(next(Path(profiled.outputs['fg']).glob('*.exr'))), -1)
                            self.assertLess(float(foreground.min()), 0.)
                            self.assertGreater(float(foreground.max()), 1.)
                        disabled = {**plan.to_dict(), 'enabled': False}
                        _, off, _ = helper.run_job(source, overrides={**overrides, 'shot_profiles': disabled}, **common)
                        for name, directory in off.outputs.items():
                            for left, right in zip(sorted(Path(directory).glob('*.exr')),
                                                   sorted(Path(baseline.outputs[name]).glob('*.exr'))):
                                self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertEqual(correction.read_bytes(), correction_before)
            self.assertEqual(source_before, {p.name: file_sha256(p) for p in source.iterdir()})

    def test_enabled_profiles_keep_one_persistent_model_and_cli_applies_explicitly(self):
        from types import SimpleNamespace

        import cv2
        import numpy as np

        import infer
        from corridorkey.config import InferenceConfig
        from corridorkey.session import InferenceSession, JobRequest
        from corridorkey.shot_profiles import analyze_job, apply_profiles, save_profiles
        from tests.test_durable_stage2 import cpu_runtime_context, make_cpu_model
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source = root / 'source'
            source.mkdir()
            for index in range(3):
                cv2.imwrite(str(source / f'{index}.png'), np.full((64, 64, 3), 90, np.uint8))
            config = InferenceConfig(stage2_batch=2, hann_chunk=4, hann_stride=2, shape_quantum=64,
                output_format='exr', outputs={'alpha'}, temp_dir=root / 'temp')
            job = JobRequest.create(source, root / 'one', config)
            active = apply_profiles(job, analyze_job(job))
            loads = []
            def load():
                loads.append(True)
                return make_cpu_model()
            with cpu_runtime_context(), InferenceSession(loader=load, disposer=lambda _: None) as session:
                session.run(active)
                session.run(replace(active, output_dir=str(root / 'two')))
            self.assertEqual(len(loads), 1)
            fake = SimpleNamespace(weights=None, package_root=None, asset_root=None, last_result=None, run=lambda request: request)
            args = ['--input', str(source), '--output_dir', str(root / 'cli'), '--output_format', 'exr',
                    '--outputs', 'alpha', '--hann_chunk', '4', '--hann_stride', '2']
            base = infer.main(args, session=fake)
            assert isinstance(base, JobRequest)
            plan = analyze_job(base)
            path = root / 'profiles.json'
            save_profiles(path, plan, base)
            result = infer.main([*args, '--shot_profiles', str(path)], session=fake)
            assert isinstance(result, JobRequest)
            self.assertEqual(result.configurations()[0].shot_profiles, plan.to_dict())

    def test_requested_binding_precedes_auto_policy_and_effective_cap_is_checked(self):
        from dataclasses import replace

        import cv2
        import numpy as np

        from corridorkey import runtime
        from corridorkey.session import JobRequest
        from corridorkey.shot_profiles import analyze_job
        from tests.test_spatial_draft import SpatialDraftTests
        with tempfile.TemporaryDirectory() as folder:
            helper = SpatialDraftTests()
            helper.root = Path(folder).resolve()
            source = helper.root / 'source'
            source.mkdir()
            for frame in range(3):
                cv2.imwrite(str(source / f'{frame}.png'), np.full((64, 64, 3), 90, np.uint8))
            captured = []
            def auto(config, shape, frames):
                captured.append(replace(config))
                config.hann_chunk, config.hann_stride, config.low_vram = 2, 1, True
                return {'scope': 'explicit CPU policy fixture, not measured VRAM'}
            with patch.object(runtime, 'apply_auto_vram', side_effect=auto):
                helper.run_job(source, scale=1., overrides={'auto_vram': True})
                requested = captured[0]
                job = JobRequest.create(source, helper.root / 'new', requested)
                plan = analyze_job(job)
                _, result, _ = helper.run_job(source, scale=1., overrides={'auto_vram': True, 'shot_profiles': plan.to_dict()})
                assert isinstance(result.effective, dict)
                self.assertTrue(result.effective['low_vram'])
            def smaller(config, shape, frames):
                config.stage2_batch = 1
                return {'scope': 'hypothetical effective-cap regression fixture'}
            with patch.object(runtime, 'apply_auto_vram', side_effect=smaller), self.assertRaisesRegex(ValueError, 'effective global'):
                helper.run_job(source, scale=1., overrides={'auto_vram': True, 'shot_profiles': plan.to_dict()})
            # An unrelated manual setting edit is rejected before output creation/model access.
            bad = replace(requested, shot_profiles=plan.to_dict(), alpha_feather_radius=2)
            output = helper.root / 'must-not-create'
            from corridorkey.config import WriterConfig
            with self.assertRaisesRegex(ValueError, 'base recipe changed'):
                runtime.run_inference(object(), str(source), str(output), bad, WriterConfig(),
                                      initial_hint_path=None, hint_video_path=None)
            self.assertFalse(output.exists())
