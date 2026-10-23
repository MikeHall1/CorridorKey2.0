import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'torch'))
if HAS_STACK or TYPE_CHECKING:
    import numpy as np
    import torch

    from corridorkey.durable_foundations import DurableFeatureFrames
    from corridorkey.foundation_cache import FoundationWindowCache
    from corridorkey.recovery_identity import ReplayIdentity, canonical
    from corridorkey.recovery_store import BundleStore, CacheInvalid


@unittest.skipUnless(HAS_STACK, 'requires optional CPU tensor adapters')
class DurableFeatureTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
                                               'assets': {}, 'environment': {'runtime': {'synthetic': True}},
                                               'fixture': 'normalized IID protocol, no model inference'}))
        self.display = torch.arange(5, dtype=torch.float32).div(10).reshape(5, 1, 1, 1).expand(5, 3, 4, 4).clone()
        self.evaluated = []

    def adapter(self, store):
        adapter = DurableFeatureFrames(store, self.parent, validate_context=lambda: None)
        adapter.bind(None)
        return adapter

    def extract(self, display):
        self.evaluated.extend(round(float(value) * 10) for value in display[:, 0, 0, 0])
        return {'moge': display[:, 0:1, 0:1, 0:1].float().clone()}

    def window(self, cache, display, start=0):
        return cache.window(display, start, device='cpu', channels={'moge': 1, 'cradio': 0},
                            grid_hw=(1, 1), extract=self.extract)

    def test_partial_run_frames_survive_new_owner_and_only_missing_frames_evaluate(self):
        with BundleStore(self.root / 'store', 100000) as store:
            cache = FoundationWindowCache(0, persistent=self.adapter(store))
            first = self.window(cache, self.display[:3])
            torch.testing.assert_close(first['moge'][:, 0, 0, 0], self.display[:3, 0, 0, 0], rtol=0, atol=0)
            cache.close()
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            cache = FoundationWindowCache(1000, persistent=adapter)
            result = self.window(cache, self.display)
            torch.testing.assert_close(result['moge'][:, 0, 0, 0], self.display[:, 0, 0, 0], rtol=0, atol=0)
            self.assertEqual(adapter.hits, 3)
            self.assertEqual(cache.frames_evaluated, 2)
            self.assertEqual(self.evaluated, [0, 1, 2, 3, 4])
            self.window(cache, self.display)
            self.assertEqual(adapter.hits, 3)  # Recovered frames enter the bounded RAM cache.
            self.assertEqual(cache.frames_evaluated, 2)
            self.assertLessEqual(cache.cache.bytes_used, cache.cache.max_bytes)

    def test_bfloat16_proxy_bits_and_input_changes_have_distinct_keys(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            first = self.window(FoundationWindowCache(0, persistent=adapter), self.display[:1].bfloat16())
            second = self.window(FoundationWindowCache(0, persistent=adapter), self.display[:1].bfloat16())
            torch.testing.assert_close(first['moge'], second['moge'], rtol=0, atol=0)
            self.assertEqual(adapter.hits, 1)
            self.window(FoundationWindowCache(0, persistent=adapter), self.display[:1])
            self.assertEqual(self.evaluated, [0, 0])
            changed = self.display[:1].clone() + 0.1
            self.window(FoundationWindowCache(0, persistent=adapter), changed)
            self.assertEqual(self.evaluated, [0, 0, 1])

    def test_corrupt_and_wrong_dtype_entries_are_recomputed_without_stale_values(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            proxy = self.display[0]
            identity = adapter._identity(0, proxy, {'moge': 1}, (1, 1))
            wrong = self.root / 'half.npy'
            np.save(wrong, np.ones((1, 1, 1), dtype=np.float16))
            store.publish(identity, {'moge.npy': wrong}, {'fixture': 'wrong producer dtype'})
            self.assertIsNone(adapter.get(0, proxy, {'moge': 1}, (1, 1)))
            self.assertEqual(adapter.corrupt, 1)
            result = self.window(FoundationWindowCache(0, persistent=adapter), self.display[:1])
            self.assertEqual(float(result['moge'].item()), 0)
            (store.root / 'entries' / identity.key / 'payload-000000.bin').write_bytes(b'corrupt')
            self.window(FoundationWindowCache(0, persistent=adapter), self.display[:1])
            self.assertEqual(adapter.corrupt, 2)
            self.assertEqual(self.evaluated, [0, 0])

    def test_loaded_tensors_are_detached_from_closed_maps_and_sealed_storage(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            features = {'moge': torch.tensor([[[-0.5, 70000.]]], dtype=torch.float32)}
            adapter.put(0, self.display[0], {'moge': 1}, (1, 2), features, producer_frames=[0])
            loaded = adapter.get(0, self.display[0], {'moge': 1}, (1, 2))
            assert loaded is not None
            loaded['moge'].zero_()
            restored = adapter.get(0, self.display[0], {'moge': 1}, (1, 2))
            assert restored is not None
            torch.testing.assert_close(restored['moge'], features['moge'], rtol=0, atol=0)
            store.discard(adapter._identity(0, self.display[0], {'moge': 1}, (1, 2)))
            torch.testing.assert_close(restored['moge'], features['moge'], rtol=0, atol=0)

    def test_binding_budget_and_invalid_producer_are_explicit(self):
        with BundleStore(self.root / 'store', 0) as store:
            adapter = DurableFeatureFrames(store, self.parent, validate_context=lambda: None)
            with self.assertRaisesRegex(CacheInvalid, 'bound shared session'):
                adapter.get(0, self.display[0], {'moge': 1}, (1, 1))
            adapter.bind(None)
            result = self.window(FoundationWindowCache(0, persistent=adapter), self.display[:1])
            self.assertEqual(result['moge'].dtype, torch.float32)
            self.assertEqual(adapter.budget_rejected, 1)
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
            with self.assertRaisesRegex(ValueError, 'finite normalized float32'):
                adapter.put(0, self.display[0], {'moge': 1}, (1, 1),
                            {'moge': torch.full((1, 1, 1), float('nan'))}, producer_frames=[0])

    def test_low_vram_prepass_reuses_sealed_frames_and_closes_output_maps(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch

        from corridorkey import runtime
        from corridorkey.diagnostics import RunDiagnostics
        from corridorkey.resources import close_mapping

        class Foundations(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.evaluated = 0

            def forward(self, *, rgb_moge, rgb_cradio):
                self.evaluated += len(rgb_moge)
                return {'moge': rgb_moge[:, :1, :1, :1].float().clone()}

        model = SimpleNamespace(foundations=Foundations(), stage1=SimpleNamespace(moge_dim=1, cradio_dim=0))
        meta = {'num_frames': 5, 'shape_hw': (64, 64)}
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            outputs = []
            for index in range(2):
                diagnostics = RunDiagnostics()
                chunks = (item for item in [(0, None, self.display[:3]), (3, None, self.display[3:])])
                with (patch.object(runtime, 'iter_frame_dir_chunks', return_value=chunks),
                      patch.object(runtime, '_build_foundation_inputs', side_effect=lambda rgb: (rgb, rgb)),
                      patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()),
                      patch.object(torch.cuda, 'empty_cache')):
                    paths = runtime.precompute_foundations_to_memmap(
                        model, 'fixture', meta, self.root / f'prepass-{index}', chunk=3,
                        reader_kind='frame_dir', durable_foundations=adapter, diagnostics=diagnostics)
                array = np.memmap(paths['moge'], mode='r', dtype=np.float32, shape=paths['moge_shape'])
                outputs.append(np.array(array, copy=True))
                close_mapping(array)
                assert isinstance(diagnostics.foundation_cache, dict)
                self.assertEqual(diagnostics.foundation_cache['foundation_frames_evaluated'], 5 if index == 0 else 0)
                (self.root / f'prepass-{index}_moge.f32').unlink()
            np.testing.assert_array_equal(*outputs)
            self.assertEqual(model.foundations.evaluated, 5)
            self.assertEqual(adapter.hits, 5)

    def test_shared_session_binds_before_runner_and_rejects_mismatched_assets(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.session import InferenceSession, JobRequest

        source = self.root / 'source.bin'
        source.write_bytes(b'explicit CPU fixture')
        job = JobRequest.create(source, self.root / 'output', InferenceConfig(output_format='exr'))
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = DurableFeatureFrames(store, self.parent, validate_context=lambda: None)
            calls = []

            def run(model, job, config, writer, diagnostics):
                self.assertTrue(adapter.bound)
                self.assertIs(diagnostics.durable_foundations, adapter)
                calls.append('runner')

            with InferenceSession(loader=object, runner=run, disposer=lambda _: None) as session:
                result = session.run(job, durable_foundations=adapter)
                self.assertEqual(result.to_dict()['diagnostics']['durable_foundations']['hits'], 0)
                session._asset_paths = {'wrong': str(source)}
                with self.assertRaises(CacheInvalid):
                    session.run(job, durable_foundations=adapter)
            self.assertFalse(adapter.bound)
            self.assertEqual(calls, ['runner'])

    def test_mutation_between_lookup_and_load_is_rejected(self):
        from unittest.mock import patch

        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            self.window(FoundationWindowCache(0, persistent=adapter), self.display[:1])
            original = np.load

            def mutate(file, **kwargs):
                with Path(file).open('wb') as stream:
                    np.save(stream, np.full((1, 1, 1), 99., dtype=np.float32), allow_pickle=False)
                return original(file, **kwargs)

            with patch.object(np, 'load', side_effect=mutate):
                self.assertIsNone(adapter.get(0, self.display[0], {'moge': 1}, (1, 1)))
            self.assertEqual(adapter.corrupt, 1)
            self.assertEqual(list((store.root / 'entries').iterdir()), [])

    def test_cancelled_private_stage_never_seals_and_is_cleaned(self):
        from corridorkey.control import CancelledError

        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            calls = 0

            def check():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise CancelledError('cancel temporary tensor write')

            adapter.check = check
            with self.assertRaisesRegex(CancelledError, 'tensor write'):
                adapter.put(0, self.display[0], {'moge': 1}, (1, 1),
                            {'moge': torch.zeros(1, 1, 1)}, producer_frames=[0])
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
            self.assertEqual(list((store.root / 'entries').iterdir()), [])

    def test_actual_process_exit_recovers_private_tensor_stage(self):
        import subprocess
        import sys

        workspace = self.root / 'store'
        code = '''
import os
import sys
from corridorkey.recovery_store import BundleStore
with BundleStore(sys.argv[1],100000) as store:
    with store.private_stage() as folder:
        (folder/'payload-000000.bin').write_bytes(b'interrupted tensor temporary')
        os._exit(75)
'''
        result = subprocess.run([sys.executable, '-c', code, str(workspace)], check=False)
        self.assertEqual(result.returncode, 75)
        with BundleStore(workspace, 100000):
            self.assertEqual(list((workspace / 'staging').iterdir()), [])
            self.assertEqual(list((workspace / 'entries').iterdir()), [])

    def test_tensor_write_disk_failure_preserves_primary_and_no_partial_entry(self):
        from unittest.mock import patch

        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            with (patch.object(np, 'save', side_effect=OSError('fixture disk full')),
                  self.assertRaisesRegex(OSError, 'disk full')):
                adapter.put(0, self.display[0], {'moge': 1}, (1, 1),
                            {'moge': torch.zeros(1, 1, 1)}, producer_frames=[0])
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
            self.assertEqual(adapter.stored, 0)
