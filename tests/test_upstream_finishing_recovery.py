"""Actual A/edit-B/clean-B upstream replay, no pretrained GPU inference."""
import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2', 'safetensors'))


@unittest.skipUnless(STACK, 'requires local CPU numeric/export fixture stack')
class UpstreamFinishingRecoveryTests(unittest.TestCase):
    def test_finishing_edit_fresh_attempt_matches_clean_all_runtime_paths(self):
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
        import cv2
        import numpy as np
        import torch

        from corridorkey import draft
        from corridorkey.config import InferenceConfig
        from corridorkey.identities import file_sha256
        from corridorkey.motion import Propagation
        from corridorkey.recovery import run_recoverable
        from corridorkey.session import InferenceSession, JobRequest
        from tests.test_durable_stage2 import cpu_runtime_context, make_cpu_model

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            for frame in range(9):
                cv2.imwrite(str(source / f'{frame:03}.png'), np.full((64, 64, 3), 80 + frame * 4, np.uint8))
            source_before = {p.name: file_sha256(p) for p in source.iterdir()}
            implementation = root / 'implementation'
            (implementation / 'corridorkey').mkdir(parents=True)
            (implementation / 'corridorkey/runtime.py').write_text('# explicit fixture implementation pin')
            assets = {}
            for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint'):
                assets[name] = root / name
                assets[name].write_bytes(b'local synthetic asset, not pretrained weights')
            assets['cradio_dir'] = root / 'cradio'
            assets['cradio_dir'].mkdir()
            (assets['cradio_dir'] / 'config.json').write_text('{}')
            env = {'backend': 'cpu-test-fixture', 'precision': 'fixture-float32',
                   'dependencies': {'fixture': '1'}, 'runtime': {'synthetic': True}}
            models = []

            def factory(**kwargs):
                def load():
                    model = make_cpu_model()
                    # Reuse the established deterministic chronological RVM fixture.
                    from tests import test_rvm_prepass
                    helper = test_rvm_prepass.RvmPrepassTests()
                    helper.setUp()
                    self.addCleanup(helper.doCleanups)
                    delattr(type(model.foundations), 'rvm')
                    model.foundations.rvm = helper.model.foundations.rvm
                    model.foundations.rvm_mean = helper.model.foundations.rvm_mean
                    model.foundations.rvm_std = helper.model.foundations.rvm_std
                    model.fixture_rvm_calls = helper.calls
                    model.fixture_iid_calls = []
                    model.fixture_stage1_calls = []
                    iid_forward, stage1_forward = model.foundations.forward, model.stage1.forward
                    def iid(**kwargs):
                        model.fixture_iid_calls.append(len(kwargs['rgb_moge']))
                        return iid_forward(**kwargs)
                    def stage1(*args, **kwargs):
                        model.fixture_stage1_calls.append(args[0].shape[1])
                        return stage1_forward(*args, **kwargs)
                    model.foundations.forward = iid
                    model.stage1.forward = stage1
                    original = model.stage2.forward
                    def patterned(rgb, baton):
                        prediction = original(rgb, baton)
                        values = torch.arange(64).remainder(13).float() / 12
                        prediction[:, :, 0] = values.view(1, 1, 1, 64)
                        from corridorkey.color_utils import linear_to_asinh
                        prediction[:, :, 1:4] = linear_to_asinh(torch.tensor([-.2, 2.4, .35]).view(1, 1, 3, 1, 1))
                        return prediction
                    model.stage2.forward = patterned
                    models.append(model)
                    return model
                return InferenceSession(loader=load, disposer=lambda _: None, **kwargs)

            original_offline = draft.offline_frames
            def uncertain(*args, **kwargs):
                alpha = np.tile(np.arange(64, dtype=np.float32) % 13 / 12, (64, 1))
                return Propagation(alpha, np.zeros_like(alpha), np.ones_like(alpha, dtype=bool),
                                   {'flow_evaluations': 4, 'motion_seconds': 0.,
                                    'uncertain_fraction': 1., 'motion_fraction': 0.})
            def forced(*args, **kwargs):
                return original_offline(*args, interpolate=uncertain, **kwargs)

            def output_hashes(record):
                return {f'{name}/{path.name}': file_sha256(path)
                        for name, directory in record['outputs'].items()
                        for path in Path(directory).glob('*.exr')}

            # Each independently enabled cache, then all three in each runtime path.
            cases = [(False, False, 1., flags, stage2) for flags in
                     ((True, False, False), (False, True, False), (False, False, True),
                      (True, True, False), (True, False, True), (False, True, True), (True, True, True))
                     for stage2 in (False, True)]
            cases += [(True, False, 1., (True, True, True), True),
                      (False, False, .5, (True, True, True), True),
                      (False, True, 1., (True, True, True), True),
                      (True, True, .5, (True, True, True), True)]
            from tests.test_upstream_finishing_identity import DESPILL_EDITS
            coverage = {'alpha_expand_radius': 1, 'alpha_feather_radius': 2}
            cases = [(*case, True, {**coverage, **DESPILL_EDITS[-1]}) for case in cases]
            cases += [(False, False, 1., (True, True, True), True, True, edits) for edits in DESPILL_EDITS]
            # Durable stores alone retain full-job identity; opt-ins remain independent.
            cases += [(False, False, 1., (True, True, True), stage2, False, DESPILL_EDITS[-1])
                      for stage2 in (False, True)]
            with cpu_runtime_context(), patch.object(draft, 'offline_frames', forced):
                for index, (low, quick, scale, flags, stage2_reuse, upstream_reuse, edits) in enumerate(cases):
                    with self.subTest(low=low, quick=quick, scale=scale, flags=flags, stage2_reuse=stage2_reuse, upstream_reuse=upstream_reuse, edits=edits):
                        cfg = InferenceConfig(hann_chunk=8, hann_stride=4, low_vram=low,
                            shape_quantum=64, quick_key=quick, quick_interval=4, quick_context_radius=1,
                            quick_fallback='neural', model_scale=scale, outputs={'alpha', 'fg', 'cutout', 'checker'},
                            cutout_alpha_mode='premultiplied', output_format='exr', exr_dtype='float',
                            temp_dir=root / 'temp', stage2_batch=2, despill_strength=.65, despill_space='linear')
                        job = JobRequest.create(source, root / f'output{index}', cfg)
                        workspace = root / f'workspace{index}'
                        def run(request, budget=8_000_000, *, workspace=workspace, flags=flags, stage2_reuse=stage2_reuse, upstream_reuse=upstream_reuse):
                            return run_recoverable([request], workspace=workspace, max_cache_bytes=budget,
                                assets=assets, environment=env, implementation_root=implementation,
                                session_factory=factory, durable_stage2=True,
                                reuse_stage2_finishing_edits=stage2_reuse, durable_writer_prefix=True,
                                durable_foundations=flags[0], durable_rvm=flags[1], durable_hann=flags[2],
                                reuse_upstream_finishing_edits=upstream_reuse)
                        first = run(job)
                        self.assertEqual(first['status'], 'complete', first.get('error'))
                        first_job = first['jobs'][0]
                        preserved = output_hashes(first_job)
                        self.assertGreater(len(models[-1].stage2.calls), 0)
                        for calls in (models[-1].fixture_iid_calls, models[-1].fixture_stage1_calls,
                                      models[-1].fixture_rvm_calls):
                            self.assertGreater(len(calls), 0, 'A must really exercise every neural fixture stage')
                        edited_cfg = replace(cfg, **edits)
                        edited = JobRequest.create(source, job.output_dir, edited_cfg)
                        second = run(edited)
                        self.assertEqual(second['status'], 'complete', second.get('error'))
                        second_job = second['jobs'][0]
                        self.assertEqual(second_job['mode'], 'render')
                        self.assertTrue(second_job['new_session_run'])
                        self.assertNotEqual(first['journal'], second['journal'])
                        self.assertNotEqual(first_job['key'], second_job['key'])
                        self.assertNotEqual(first_job['output_dir'], second_job['output_dir'])
                        self.assertEqual(bool(models[-1].stage2.calls), not stage2_reuse)
                        for enabled, calls in zip(flags, (models[-1].fixture_iid_calls,
                                models[-1].fixture_rvm_calls, models[-1].fixture_stage1_calls)):
                            self.assertEqual(bool(calls), not (enabled and upstream_reuse))
                        stats = second_job['origin_record']['diagnostics']
                        self.assertEqual(stats['durable_stage2']['batch_hits'] > 0, stage2_reuse)
                        for enabled, name, hit in zip(flags,
                                ('durable_foundations', 'durable_rvm', 'durable_hann'),
                                ('hits', 'chunk_hits', 'window_hits')):
                            if enabled:
                                if upstream_reuse:
                                    self.assertGreater(stats[name][hit], 0)
                                else:
                                    miss = {'durable_foundations': 'misses', 'durable_rvm': 'chunk_misses',
                                            'durable_hann': 'window_misses'}[name]
                                    self.assertGreater(stats[name][miss], 0)
                                self.assertEqual(stats[name]['parent_kind'], 'upstream_finishing_context' if upstream_reuse else 'whole_job_exr_replay')
                        self.assertEqual(stats['durable_writer_prefix']['restored_prefix_frames'], 0)
                        if quick:
                            self.assertGreater(len(stats['quick_key']['extra_neural_frames']), 0)
                        # A distinct requested destination changes no output identity; zero budget forces recomputation.
                        clean_request = replace(edited, output_dir=str(root / f'clean{index}'))
                        clean = run_recoverable([clean_request], workspace=root / f'clean-cache{index}',
                            max_cache_bytes=0, assets=assets, environment=env,
                            implementation_root=implementation, session_factory=factory)
                        self.assertEqual(clean['status'], 'complete', clean.get('error'))
                        self.assertGreater(len(models[-1].stage2.calls), 0)
                        clean_job = clean['jobs'][0]
                        self.assertEqual(output_hashes(second_job), output_hashes(clean_job))
                        if set(edits) != {'despill_space'}:
                            self.assertNotEqual(output_hashes(second_job), preserved)
                        if set(edits) <= {'despill_strength', 'despill_screen', 'despill_space',
                                          'despill_preserve_luma', 'despill_edges_only'}:
                            current = output_hashes(second_job)
                            for name, value in preserved.items():
                                if name.startswith(('alpha/', 'fg/')):
                                    self.assertEqual(current[name], value, name)
                        if not quick:
                            raw = cv2.imread(str(next(Path(second_job['outputs']['fg']).glob('*.exr'))), -1)
                            self.assertEqual(raw.dtype, np.float32)
                            self.assertLess(raw.min(), 0)
                            self.assertGreater(raw.max(), 1)
                        for cutout, alpha_path in zip(sorted(Path(second_job['outputs']['cutout']).glob('*.exr')),
                                                      sorted(Path(second_job['outputs']['alpha']).glob('*.exr'))):
                            np.testing.assert_array_equal(cv2.imread(str(cutout), -1)[..., 3], cv2.imread(str(alpha_path), -1))
                        self.assertEqual(output_hashes(first_job), preserved)
                        for name, directory in second_job['outputs'].items():
                            for left, right in zip(sorted(Path(directory).glob('*.exr')),
                                    sorted(Path(clean_job['outputs'][name]).glob('*.exr'))):
                                np.testing.assert_array_equal(cv2.imread(str(left), -1), cv2.imread(str(right), -1))
            self.assertEqual(source_before, {p.name: file_sha256(p) for p in source.iterdir()})


    def test_live_full_job_drift_under_finishing_context_cannot_seal_output(self):
        from corridorkey.recovery import run_recoverable
        from tests.test_recovery_runner import RecoveryRunnerTests
        helper = RecoveryRunnerTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.change_source = True
        from corridorkey.recovery_store import CacheInvalid
        with self.assertRaisesRegex(CacheInvalid, 'dependencies changed during render'):
            run_recoverable(helper.jobs[:1], workspace=helper.workspace,
                max_cache_bytes=1_000_000, assets=helper.assets, environment=helper.environment,
                implementation_root=helper.root / 'implementation', session_factory=helper.factory,
                durable_foundations=True, durable_rvm=True, durable_hann=True,
                reuse_upstream_finishing_edits=True)
        report = json.loads(next((helper.workspace / 'reports').glob('*.json')).read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['jobs'], [])
        self.assertIn('dependencies changed', report['error']['message'])
        journal = json.loads(Path(report['journal']).read_text())['payload']
        self.assertEqual(journal['jobs'][0][-1]['status'], 'failed')
        self.assertEqual(list((helper.workspace / 'entries').iterdir()), [])
