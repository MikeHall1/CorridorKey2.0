import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ('torch', 'numpy', 'safetensors'))
if HAS_STACK or TYPE_CHECKING:
    import torch
    from safetensors.torch import save

    from corridorkey.durable_rvm import DurableRvmChunks, tensor_digest
    from corridorkey.recovery_identity import ReplayIdentity, canonical
    from corridorkey.recovery_store import BundleStore, CacheInvalid


@unittest.skipUnless(HAS_STACK, 'requires optional CPU checkpoint tensor adapters')
class DurableRvmTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
                                               'assets': {}, 'environment': {'runtime': {'synthetic': True}}}))
        self.display = torch.full((2, 3, 64, 64), 0.5, dtype=torch.bfloat16)
        self.features = torch.arange(2, dtype=torch.float16).reshape(2, 1, 1, 1)
        self.state = torch.tensor([[[0.125], [-0.5]]], dtype=torch.bfloat16)

    def adapter(self, store):
        adapter = DurableRvmChunks(store, self.parent, validate_context=lambda: None)
        adapter.bind(None)
        return adapter

    def identity(self, adapter, state=None, *, start=0, shot_start=0):
        return adapter.identity(start, shot_start, self.display, state, channels=1, grid_hw=(1, 1))

    def test_exact_state_dtype_and_feature_bits_survive_reopen(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                workspace = self.root / str(dtype)
                state = self.state.to(dtype)
                with BundleStore(workspace, 100000) as store:
                    adapter = self.adapter(store)
                    identity = self.identity(adapter)
                    self.assertTrue(adapter.put(identity, self.features, state))
                with BundleStore(workspace, 100000) as store:
                    adapter = self.adapter(store)
                    restored = adapter.get(identity)
                    assert restored is not None
                    self.assertEqual(restored['state'].dtype, dtype)
                    self.assertEqual(tensor_digest(restored['state']), tensor_digest(state))
                    torch.testing.assert_close(restored['features'], self.features, rtol=0, atol=0)
                    restored['features'].fill_(99)
                    second = adapter.get(identity)
                    assert second is not None
                    torch.testing.assert_close(second['features'], self.features, rtol=0, atol=0)
                    self.assertEqual(adapter.frames_reused, 4)

    def test_incoming_state_precision_and_bits_select_different_chains(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter, self.state, start=2)
            adapter.put(identity, self.features, self.state)
            self.assertIsNotNone(adapter.get(identity))
            changed = self.identity(adapter, self.state + 1, start=2)
            precision = self.identity(adapter, self.state.float(), start=2)
            self.assertNotEqual(identity.key, changed.key)
            self.assertNotEqual(identity.key, precision.key)
            self.assertIsNone(adapter.get(changed))
            self.assertIsNone(adapter.get(precision))
            self.assertNotEqual(identity.key, self.identity(adapter, start=2, shot_start=2).key)

    def test_corrupt_or_incomplete_state_bundle_is_rejected_atomically(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            for tensors in ({'features': self.features},
                            {'features': self.features.float(), 'state': self.state},
                            {'features': self.features, 'state': self.state[:, :1]},
                            {'features': self.features, 'state': self.state.float().fill_(float('nan'))}):
                file = self.root / 'bad.safetensors'
                file.write_bytes(save(tensors))
                store.publish(identity, {'rvm.safetensors': file}, {'state_dtype': 'torch.bfloat16'})
                self.assertIsNone(adapter.get(identity))
                self.assertFalse(store._entry(identity).exists())
            self.assertEqual(adapter.corrupt, 4)
            adapter.put(identity, self.features, self.state)
            (store._entry(identity) / 'payload-000000.bin').write_bytes(b'broken')
            self.assertIsNone(adapter.get(identity))
            self.assertEqual(adapter.corrupt, 5)

    def test_bound_parent_budget_and_producer_validation_are_explicit(self):
        with BundleStore(self.root / 'store', 0) as store:
            adapter = DurableRvmChunks(store, self.parent, validate_context=lambda: None)
            with self.assertRaisesRegex(CacheInvalid, 'bound'):
                self.identity(adapter)
            adapter.bind(None)
            identity = self.identity(adapter)
            self.assertFalse(adapter.put(identity, self.features, self.state))
            self.assertEqual(adapter.budget_rejected, 1)
            with self.assertRaisesRegex(ValueError, 'normalized float16'):
                adapter.put(identity, self.features.float(), self.state)
            with self.assertRaisesRegex(ValueError, 'RVM state'):
                adapter.put(identity, self.features, torch.zeros(1, 1, 1))
            other = DurableRvmChunks(store, ReplayIdentity(canonical({**self.parent.to_dict(), 'changed': True})),
                                     validate_context=lambda: None)
            other.bind(None)
            with self.assertRaisesRegex(CacheInvalid, 'another typed job'):
                other.get(identity)

    def test_valid_tensors_with_invalid_provenance_are_recomputed(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            file = self.root / 'valid.safetensors'
            file.write_bytes(save({'features': self.features, 'state': self.state}))
            record = adapter._record(self.state)
            for bad in (None, [], {**record, 'stage': 'iid_foundation_frame'},
                        {**record, 'state_dtype': 'torch.float32'}, {**record, 'state_shape': [1, 1, 1]},
                        {**record, 'parent_identity': {}}, {**record, 'hann_output_resume': 0}):
                store.publish(identity, {'rvm.safetensors': file}, bad)
                self.assertIsNone(adapter.get(identity))
                self.assertFalse(store._entry(identity).exists())


    def test_capacity_cancel_and_disk_failure_never_leave_partial_state(self):
        from unittest.mock import patch

        from corridorkey.control import CancelledError

        with BundleStore(self.root / 'small', 256) as store:
            adapter = self.adapter(store)
            self.assertFalse(adapter.put(self.identity(adapter), self.features, self.state))
            self.assertEqual(adapter.budget_rejected, 1)
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            with (patch.object(Path, 'write_bytes', side_effect=OSError('fixture checkpoint disk full')),
                  self.assertRaisesRegex(OSError, 'checkpoint disk full')):
                adapter.put(identity, self.features, self.state)
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
            calls = 0

            def check():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise CancelledError('cancel after tensor serialization')

            adapter.check = check
            with self.assertRaisesRegex(CancelledError, 'serialization'):
                adapter.put(identity, self.features, self.state)
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
            self.assertEqual(list((store.root / 'entries').iterdir()), [])

    def test_actual_hard_process_exit_preserves_completed_state_bundle(self):
        import subprocess
        import sys

        workspace = self.root / 'store'
        code = '''
import os
import sys
import torch
from corridorkey.durable_rvm import DurableRvmChunks
from corridorkey.recovery_identity import ReplayIdentity
from corridorkey.recovery_store import BundleStore
with BundleStore(sys.argv[1],100000) as store:
    adapter=DurableRvmChunks(store,ReplayIdentity(sys.argv[2]),validate_context=lambda:None)
    adapter.bind(None)
    display=torch.full((2,3,64,64),0.5,dtype=torch.bfloat16)
    identity=adapter.identity(0,0,display,None,channels=1,grid_hw=(1,1))
    features=torch.arange(2,dtype=torch.float16).reshape(2,1,1,1)
    state=torch.tensor([[[0.125],[-0.5]]],dtype=torch.bfloat16)
    adapter.put(identity,features,state)
    os._exit(76)
'''
        result = subprocess.run([sys.executable, '-c', code, str(workspace), self.parent.payload], check=False)
        self.assertEqual(result.returncode, 76)
        with BundleStore(workspace, 100000) as store:
            adapter = self.adapter(store)
            restored = adapter.get(self.identity(adapter))
            assert restored is not None
            self.assertEqual(tensor_digest(restored['state']), tensor_digest(self.state))
            torch.testing.assert_close(restored['features'], self.features, rtol=0, atol=0)

    def test_loaded_model_mismatch_or_changed_context_prevents_runner_access(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.session import InferenceSession, JobRequest

        source = self.root / 'source.bin'
        source.write_bytes(b'CPU binding fixture')
        job = JobRequest.create(source, self.root / 'out', InferenceConfig(output_format='exr'))
        calls = []
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = DurableRvmChunks(store, self.parent, validate_context=lambda: None)
            with InferenceSession(loader=object, runner=lambda *args: calls.append('runner'),
                                  disposer=lambda _: calls.append('discard')) as session:
                session._asset_paths = {'wrong': str(source)}
                with self.assertRaisesRegex(CacheInvalid, 'producer differs'):
                    session.run(job, durable_rvm=adapter)
            self.assertEqual(calls, ['discard'])
            self.assertFalse(adapter.bound)

            def changed():
                raise CacheInvalid('changed job context')

            adapter = DurableRvmChunks(store, self.parent, validate_context=changed)
            with self.assertRaisesRegex(CacheInvalid, 'changed job context'):
                adapter.bind(None)
            self.assertFalse(adapter.bound)

    def test_mutation_after_lookup_is_rejected_before_tensor_deserialization(self):
        from unittest.mock import patch

        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            adapter.put(identity, self.features, self.state)
            original = Path.read_bytes

            def mutate(file):
                file.write_bytes(save({'features': self.features, 'state': self.state + 1}))
                return original(file)

            with patch.object(Path, 'read_bytes', mutate):
                self.assertIsNone(adapter.get(identity))
            self.assertEqual(adapter.corrupt, 1)
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
