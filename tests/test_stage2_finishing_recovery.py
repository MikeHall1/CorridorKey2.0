"""Actual recoverable runtime attempts reuse raw predictions, never edited output."""
import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2', 'safetensors'))


@unittest.skipUnless(STACK, 'requires local CPU numeric/export fixture stack')
class Stage2FinishingRecoveryTests(unittest.TestCase):
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

            from tests.test_upstream_finishing_identity import DESPILL_EDITS
            coverage = {'alpha_expand_radius': 1, 'alpha_feather_radius': 2}
            cases = [(low, quick, scale, coverage) for low, quick, scale in
                     ((False, False, 1.), (True, False, 1.), (False, False, .5),
                      (False, True, 1.), (True, True, .5))]
            cases += [(False, False, 1., edits) for edits in DESPILL_EDITS]
            cases += [(True, True, .5, {**coverage, **DESPILL_EDITS[-1]})]
            with cpu_runtime_context(), patch.object(draft, 'offline_frames', forced):
                for index, (low, quick, scale, edits) in enumerate(cases):
                    with self.subTest(low=low, quick=quick, scale=scale, edits=edits):
                        cfg = InferenceConfig(hann_chunk=8, hann_stride=4, low_vram=low,
                            shape_quantum=64, quick_key=quick, quick_interval=4, quick_context_radius=1,
                            quick_fallback='neural', model_scale=scale, outputs={'alpha', 'fg', 'cutout', 'checker'},
                            cutout_alpha_mode='premultiplied', output_format='exr', exr_dtype='float',
                            temp_dir=root / 'temp', stage2_batch=2, despill_strength=.65, despill_space='linear')
                        job = JobRequest.create(source, root / f'output{index}', cfg)
                        workspace = root / f'workspace{index}'
                        def run(request, budget=8_000_000, *, workspace=workspace):
                            return run_recoverable([request], workspace=workspace, max_cache_bytes=budget,
                                assets=assets, environment=env, implementation_root=implementation,
                                session_factory=factory, durable_stage2=True,
                                reuse_stage2_finishing_edits=True, durable_writer_prefix=True)
                        first = run(job)
                        self.assertEqual(first['status'], 'complete', first.get('error'))
                        first_job = first['jobs'][0]
                        preserved = output_hashes(first_job)
                        self.assertGreater(len(models[-1].stage2.calls), 0)
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
                        self.assertEqual(models[-1].stage2.calls, [])
                        stats = second_job['origin_record']['diagnostics']
                        self.assertGreater(stats['durable_stage2']['batch_hits'], 0)
                        self.assertEqual(stats['durable_stage2']['parent_kind'], 'stage2_finishing_context')
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

    def test_new_context_corruption_budget_binding_and_actual_call_contract(self):
        import torch

        from corridorkey.durable_stage2 import DurableStage2Batches
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore, CacheInvalid
        from corridorkey.stage2_identity import stage2_finishing_context
        parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
            'assets': {}, 'environment': {'runtime': {'synthetic': True}},
            'settings': {'alpha_expand_radius': 0, 'alpha_feather_radius': 0, 'despill_strength': .5,
                         'despill_screen': 'green', 'despill_space': 'legacy',
                         'despill_preserve_luma': False, 'despill_edges_only': False},
            'source': {}, 'hints': {'first': None, 'full': None}, 'corrections': None,
            'writer': {}, 'profile_stages': False, 'implementation': {}, 'scope': 'fixture'}))
        context = stage2_finishing_context(parent)
        rgb = torch.zeros(1, 2, 3, 64, 64)
        baton = torch.zeros(1, 2, 1, 4, 4)
        prediction = torch.full((1, 2, 4, 64, 64), .4)
        with tempfile.TemporaryDirectory() as folder:
            def adapter(store, validate=lambda: None):
                result = DurableStage2Batches(store, context, validate_context=validate)
                result.bind(None)
                return result
            def identity(owner, **changes):
                return owner.identity(**{'frames': (0, 1), 'policy': 'full_rate',
                                          'rgb': rgb, 'baton': baton, **changes})
            with BundleStore(Path(folder) / 'store', 2_000_000) as store:
                owner = adapter(store)
                key = identity(owner)
                assert key is not None
                self.assertTrue(owner.put(key, prediction))
                for changes in ({'frames': (1, 0)}, {'rgb': rgb.half()},
                                {'baton': baton.bfloat16()}, {'policy': 'quick_fallback'},
                                {'rgb': torch.ones_like(rgb)}, {'baton': torch.ones_like(baton)},
                                {'rgb': torch.zeros(1, 2, 3, 64, 128),
                                 'baton': torch.zeros(1, 2, 1, 4, 8)}):
                    changed = identity(owner, **changes)
                    assert changed is not None
                    self.assertIsNone(owner.get(changed))
                entry = store._entry(key)
                payload = next(entry.glob('payload-*'))
                payload.write_bytes(b'corrupt new-context tensor')
                self.assertIsNone(owner.get(key))
                self.assertEqual(owner.corrupt, 1)
                self.assertTrue(owner.put(key, prediction))
                self.assertIsNotNone(owner.get(key))
                from corridorkey.control import CancelledError
                def cancelled():
                    raise CancelledError('stop cache operation')
                owner.check = cancelled
                with self.assertRaises(CancelledError):
                    owner.get(key)
                owner.check = None
                self.assertIsNotNone(adapter(store).get(key))
                def drift():
                    raise CacheInvalid('full current job dependencies changed')
                with self.assertRaisesRegex(CacheInvalid, 'full current job'):
                    adapter(store, validate=drift)
            with BundleStore(Path(folder) / 'zero', 0) as store:
                owner = adapter(store)
                with patch('corridorkey.durable_stage2.tensor_digest',
                           side_effect=AssertionError('zero quota must not hash tensors')):
                    self.assertIsNone(identity(owner))
                self.assertEqual(owner.budget_rejected, 1)

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
                durable_stage2=True, reuse_stage2_finishing_edits=True)
        report = json.loads(next((helper.workspace / 'reports').glob('*.json')).read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['jobs'], [])
        self.assertIn('dependencies changed', report['error']['message'])
        journal = json.loads(Path(report['journal']).read_text())['payload']
        self.assertEqual(journal['jobs'][0][-1]['status'], 'failed')
        self.assertEqual(list((helper.workspace / 'entries').iterdir()), [])

    def test_option_requires_explicit_stage2_and_boolean_before_io(self):
        from corridorkey.recovery import run_recoverable
        invalid_values: list[Any] = [True, 1, 'true', None]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_recoverable([object()], workspace='/unused', max_cache_bytes=0,
                    assets={}, environment={}, reuse_stage2_finishing_edits=value)
