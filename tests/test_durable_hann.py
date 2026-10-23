import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ('torch', 'numpy', 'safetensors'))
if HAS_STACK or TYPE_CHECKING:
    import torch
    from safetensors.torch import save

    from corridorkey.durable_hann import DurableHannWindows, replay_state_digest
    from corridorkey.durable_rvm import tensor_digest
    from corridorkey.recovery_identity import ReplayIdentity, canonical
    from corridorkey.recovery_store import BundleStore, CacheInvalid


@unittest.skipUnless(HAS_STACK, 'requires optional CPU event adapters')
class DurableHannTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
                                               'assets': {}, 'environment': {'runtime': {'synthetic': True}}}))
        self.inputs = {'rgb': torch.zeros(1, 2, 3, 64, 64), 'hint': torch.zeros(1, 2, 1, 64, 64),
                       'hint_quality': torch.zeros(1, 2), 'moge': torch.zeros(1, 2, 1, 1, 1),
                       'cradio': None, 'rvm': None}
        self.outputs = {'baton': torch.arange(32).reshape(1, 2, 1, 4, 4).bfloat16(),
                        'aux': torch.full((1, 2, 4, 32, 32), 0.125, dtype=torch.bfloat16)}
        self.state = {'carry': {}, 'accumulators': {}, 'weights': {}, 'pending_frames': {}, 'previous_segment': None}

    def adapter(self, store):
        adapter = DurableHannWindows(store, self.parent, validate_context=lambda: None)
        adapter.bind(None)
        return adapter

    def identity(self, adapter, *, carry=True):
        return adapter.identity(schedule={'windows': [(0, 2)]}, incoming_replay_sha256=replay_state_digest(**self.state),
                                index=0, start=0, stop=2, next_start=2, segment=(0, 2), carry=carry, d_16=1,
                                inputs=self.inputs)

    def test_exact_baton_aux_alpha_bits_and_dtypes_survive_reopen(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            workspace = self.root / str(dtype)
            outputs = {name: value.to(dtype) for name, value in self.outputs.items()}
            with BundleStore(workspace, 100000) as store:
                adapter = self.adapter(store)
                identity = self.identity(adapter)
                self.assertTrue(adapter.put(identity, outputs))
            with BundleStore(workspace, 100000) as store:
                restored = self.adapter(store).get(identity)
                assert restored is not None
                for name, value in outputs.items():
                    expected = value[:, :, :1] if name == 'aux' else value
                    self.assertEqual(tensor_digest(restored[name]), tensor_digest(expected))
                restored['baton'].zero_()
                again = self.adapter(store).get(identity)
                assert again is not None
                self.assertEqual(tensor_digest(again['baton']), tensor_digest(outputs['baton']))

    def test_all_live_replay_state_and_exact_model_inputs_invalidate(self):
        baseline = replay_state_digest(**self.state)
        changes = ({'carry': {0: torch.ones(1, 64, 64)}},
                   {'accumulators': {0: torch.ones(1, 4, 4)}, 'weights': {0: 0.5}},
                   {'pending_frames': {0: torch.ones(3, 64, 64)}}, {'previous_segment': (0, 2)})
        for change in changes:
            self.assertNotEqual(baseline, replay_state_digest(**{**self.state, **change}))
        with self.assertRaises(ValueError):
            replay_state_digest(**{**self.state, 'weights': {0: 1.0}})
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            original = self.identity(adapter)
            self.inputs['hint_quality'].fill_(-1)
            self.assertNotEqual(original.key, self.identity(adapter).key)
            self.state['previous_segment'] = (0, 2)
            self.assertNotEqual(original.key, self.identity(adapter).key)

    def test_incomplete_or_invalid_provenance_event_is_rejected(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            file = self.root / 'bad.safetensors'
            for tensors, record in (({'baton': self.outputs['baton']}, None),
                                    ({'baton': self.outputs['baton'], 'aux': self.outputs['aux']}, None),
                                    ({'baton': self.outputs['baton'], 'aux': self.outputs['aux'][:, :, :1]}, None)):
                file.write_bytes(save({name: value.contiguous() for name, value in tensors.items()}))
                store.publish(identity, {'hann.safetensors': file}, record)
                self.assertIsNone(adapter.get(identity))
            self.assertEqual(adapter.corrupt, 3)

    def test_binding_zero_budget_no_carry_and_recurrent_state_guard(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = DurableHannWindows(store, self.parent, validate_context=lambda: None)
            with self.assertRaisesRegex(CacheInvalid, 'bound'):
                self.identity(adapter)
            adapter.bind(None)
            identity = self.identity(adapter, carry=False)
            adapter.put(identity, self.outputs)
            result = adapter.get(identity)
            assert result is not None
            self.assertEqual(set(result), {'baton'})
            with self.assertRaisesRegex(ValueError, 'recurrent'):
                adapter.put(identity, {**self.outputs, 'recurrent_states': (torch.zeros(1),)})
        with BundleStore(self.root / 'zero', 0) as store:
            adapter = self.adapter(store)
            self.assertFalse(adapter.put(self.identity(adapter), self.outputs))
            self.assertEqual(adapter.budget_rejected, 1)

    def test_capacity_disk_full_and_cancellation_never_seal_partial_event(self):
        from unittest.mock import patch

        from corridorkey.control import CancelledError

        with BundleStore(self.root / 'small', 256) as store:
            adapter = self.adapter(store)
            self.assertFalse(adapter.put(self.identity(adapter, carry=False), self.outputs))
            self.assertEqual(adapter.budget_rejected, 1)
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            with (patch.object(Path, 'write_bytes', side_effect=OSError('fixture Hann disk full')),
                  self.assertRaisesRegex(OSError, 'Hann disk full')):
                adapter.put(identity, self.outputs)
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
            calls = 0

            def check():
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise CancelledError('cancel serialized Hann event')

            adapter.check = check
            with self.assertRaisesRegex(CancelledError, 'serialized Hann'):
                adapter.put(identity, self.outputs)
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
            self.assertEqual(list((store.root / 'staging').iterdir()), [])

    def test_signed_large_raw_values_and_post_lookup_mutation(self):
        from unittest.mock import patch

        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            outputs = {name: value.float().clone() for name, value in self.outputs.items()}
            outputs['baton'][0, 0, 0, 0, :2] = torch.tensor([-0.5, 70000.])
            outputs['aux'][0, 0, 0, 0, :2] = torch.tensor([-1., 2.])
            adapter.put(identity, outputs)
            restored = adapter.get(identity)
            assert restored is not None
            torch.testing.assert_close(restored['baton'], outputs['baton'], rtol=0, atol=0)
            torch.testing.assert_close(restored['aux'], outputs['aux'][:, :, :1], rtol=0, atol=0)
            original = Path.read_bytes

            def mutate(file):
                file.write_bytes(save({'baton': outputs['baton'] + 1, 'aux': outputs['aux'][:, :, :1].contiguous()}))
                return original(file)

            with patch.object(Path, 'read_bytes', mutate):
                self.assertIsNone(adapter.get(identity))
            self.assertEqual(adapter.corrupt, 1)
            self.assertEqual(list((store.root / 'entries').iterdir()), [])

    def test_actual_process_exit_preserves_event_and_releases_owner(self):
        import subprocess
        import sys

        code = '''
import os
import sys
import torch
from corridorkey.durable_hann import DurableHannWindows,replay_state_digest
from corridorkey.recovery_identity import ReplayIdentity
from corridorkey.recovery_store import BundleStore
with BundleStore(sys.argv[1],100000) as store:
    adapter=DurableHannWindows(store,ReplayIdentity(sys.argv[2]),validate_context=lambda:None)
    adapter.bind(None)
    inputs={'rgb':torch.zeros(1,2,3,64,64),'hint':torch.zeros(1,2,1,64,64),
            'hint_quality':torch.zeros(1,2),'moge':torch.zeros(1,2,1,1,1),'cradio':None,'rvm':None}
    incoming=replay_state_digest(carry={},accumulators={},weights={},pending_frames={},previous_segment=None)
    identity=adapter.identity(schedule={'windows':[(0,2)]},incoming_replay_sha256=incoming,
        index=0,start=0,stop=2,next_start=2,segment=(0,2),carry=True,d_16=1,inputs=inputs)
    adapter.put(identity,{'baton':torch.arange(32).reshape(1,2,1,4,4).bfloat16(),
                         'aux':torch.full((1,2,4,32,32),0.125,dtype=torch.bfloat16)})
    os._exit(77)
'''
        workspace = self.root / 'store'
        result = subprocess.run([sys.executable, '-c', code, str(workspace), self.parent.payload], check=False)
        self.assertEqual(result.returncode, 77)
        with BundleStore(workspace, 100000) as store:
            adapter = self.adapter(store)
            event = adapter.get(self.identity(adapter))
            assert event is not None
            self.assertEqual(tensor_digest(event['baton']), tensor_digest(self.outputs['baton']))
            self.assertEqual(list((store.root / 'staging').iterdir()), [])

    def test_shared_session_rejects_loaded_model_mismatch_before_runner(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.session import InferenceSession, JobRequest

        source = self.root / 'source.bin'
        source.write_bytes(b'CPU fixture')
        job = JobRequest.create(source, self.root / 'out', InferenceConfig(output_format='exr'))
        calls = []
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = DurableHannWindows(store, self.parent, validate_context=lambda: None)
            with InferenceSession(loader=object, runner=lambda *args: calls.append('runner'),
                                  disposer=lambda _: calls.append('discard')) as session:
                session._asset_paths = {'wrong': str(source)}
                with self.assertRaisesRegex(CacheInvalid, 'producer differs'):
                    session.run(job, durable_hann=adapter)
            self.assertEqual(calls, ['discard'])
            self.assertFalse(adapter.bound)
