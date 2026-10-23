"""Different explicit scales cannot share even bit-identical raw predictions."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'safetensors'))


@unittest.skipUnless(STACK, 'Requires the local CPU tensor/recovery stack')
class SpatialRecoveryTests(unittest.TestCase):
    def test_same_prediction_inputs_do_not_replay_across_scale_or_resampler_changes(self):
        import torch

        from corridorkey.config import InferenceConfig
        from corridorkey.durable_stage2 import DurableStage2Batches
        from corridorkey.recovery_identity import job_identity
        from corridorkey.recovery_store import BundleStore
        from corridorkey.session import JobRequest

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.bin'
            source.write_bytes(b'Local identity fixture; no neural inference or decoded footage.')
            assets = {}
            for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint'):
                path = root / name
                path.write_bytes(b'Explicit synthetic asset identity, not model weights.')
                assets[name] = path
            assets['cradio_dir'] = root / 'cradio'
            assets['cradio_dir'].mkdir()
            (assets['cradio_dir'] / 'config.json').write_text('{}')
            implementation = root / 'implementation'
            code = implementation / 'corridorkey'
            code.mkdir(parents=True)
            (code / 'runtime.py').write_text('# identity-only shared engine fixture\n')
            resampler = code / 'spatial_draft.py'
            resampler.write_text('# spatial-draft-v1 resampling policy\n')

            def parent(scale):
                job = JobRequest.create(source, root / 'output',
                    InferenceConfig(model_scale=scale, output_format='exr', outputs={'alpha'}))
                return job_identity(job, assets=assets, implementation_root=implementation,
                    environment={'backend': 'explicit-cpu-fixture', 'precision': 'float32',
                                 'dependencies': {'fixture': '1'}, 'runtime': {'synthetic': True}})

            native, spatial = parent(1.0), parent(0.5)
            self.assertNotEqual(native.key, spatial.key)
            # Deliberately keep geometry and tensor bits equal: padding or flat
            # imagery can make those insufficient to identify a draft policy.
            rgb = torch.zeros(1, 2, 3, 64, 64)
            baton = torch.zeros(1, 2, 1, 4, 4)
            prediction = torch.full((1, 2, 4, 64, 64), .4)
            with BundleStore(root / 'cache', 2_000_000) as store:
                def adapter(identity):
                    result = DurableStage2Batches(store, identity, validate_context=lambda: None)
                    result.bind(None)
                    return result

                def key(owner):
                    result = owner.identity(frames=(0, 1), policy='full_rate', rgb=rgb, baton=baton)
                    assert result is not None
                    return result

                baseline, draft = adapter(native), adapter(spatial)
                self.assertTrue(baseline.put(key(baseline), prediction))
                self.assertIsNotNone(baseline.get(key(baseline)))
                self.assertIsNone(draft.get(key(draft)))
                self.assertTrue(draft.put(key(draft), prediction))
                resampler.write_text('# spatial-draft-v2 changed resampling policy\n')
                changed = adapter(parent(0.5))
                self.assertNotEqual(changed.parent.key, spatial.key)
                self.assertIsNone(changed.get(key(changed)))
