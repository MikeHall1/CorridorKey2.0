"""Real prepass/cache plumbing with deterministic CPU tensor recurrence."""
import importlib.util
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ('torch', 'numpy', 'cv2', 'safetensors'))
if HAS_STACK or TYPE_CHECKING:
    import numpy as np
    import torch

    from corridorkey import runtime
    from corridorkey.control import CancellationToken, CancelledError, JobControl
    from corridorkey.diagnostics import RunDiagnostics
    from corridorkey.durable_rvm import DurableRvmChunks
    from corridorkey.recovery_identity import ReplayIdentity, canonical
    from corridorkey.recovery_store import BundleStore
    from corridorkey.recurrent_identity import RvmChunkIdentity


@unittest.skipUnless(HAS_STACK, 'requires optional CPU prepass adapters; no pretrained inference')
class RvmPrepassTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
                                               'assets': {}, 'environment': {'runtime': {'synthetic': True}}}))
        self.display = torch.arange(6).float().div(10).reshape(6, 1, 1, 1).expand(6, 3, 64, 64).bfloat16()
        calls = self.calls = []

        class RVM(torch.nn.Module):
            embed_dim = 1

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward(self, rgb, state=None):
                calls.append((rgb.shape[1], None if state is None else state.clone()))
                state = torch.zeros(1, 2, 1) if state is None else state
                features = []
                for frame in rgb[0]:
                    state = state + frame.mean()
                    features.append(state[:, 1:, :].reshape(1, 1, 1, 1))
                return torch.stack(features, dim=1), state

        self.model = SimpleNamespace(foundations=SimpleNamespace(rvm=RVM(), rvm_mean=torch.tensor([0.25]),
                                                                 rvm_std=torch.tensor([0.5])))
        self.sequence = 0

    def run_prepass(self, adapter=None, *, cancel_at=None, resets=(0, 3), chunk=2):
        def source(*args, **kwargs):
            for start in range(0, len(self.display), chunk):
                yield start, None, self.display[start:start + chunk]

        token = CancellationToken()

        def progress(event):
            if event.stage == 'rvm_prepass' and event.completed == cancel_at:
                token.cancel('interrupt after sealed RVM chunk')

        diagnostics = RunDiagnostics()
        path = self.root / f'prepass-{self.sequence}.f16'
        self.sequence += 1
        with (patch.object(runtime, 'iter_frame_dir_chunks', side_effect=source),
              patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()),
              patch.object(torch.cuda, 'empty_cache')):
            runtime.precompute_rvm_to_memmap(self.model, 'fixture', {'num_frames': 6, 'shape_hw': (64, 64)},
                                             path, chunk=chunk, reader_kind='frame_dir', reset_frames=resets,
                                             durable_rvm=adapter, diagnostics=diagnostics,
                                             control=JobControl(token, progress))
        result = np.fromfile(path, dtype=np.float16)
        path.unlink()
        return result, diagnostics

    def adapter(self, store):
        adapter = DurableRvmChunks(store, self.parent, validate_context=lambda: None)
        adapter.bind(None)
        return adapter

    def test_interrupted_prepass_restores_state_and_only_evaluates_remaining_chunks(self):
        reference, _ = self.run_prepass()
        self.calls.clear()
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            with self.assertRaisesRegex(CancelledError, 'sealed RVM chunk'):
                self.run_prepass(adapter, cancel_at=2)
            self.assertEqual(adapter.stored, 1)
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            resumed, diagnostics = self.run_prepass(adapter)
            np.testing.assert_array_equal(resumed, reference)
            self.assertEqual((adapter.hits, adapter.frames_reused), (1, 2))
            assert isinstance(diagnostics.execution, dict)
            self.assertEqual(diagnostics.execution['rvm_frame_evaluations'], 4)
            self.assertEqual(sum(count for count, _ in self.calls), 6)
            self.assertIsNotNone(self.calls[1][1])  # Missing continuation starts from restored state.
            self.assertIsNone(self.calls[2][1])  # Shot boundary remains an explicit reset.
            self.assertIsNotNone(self.calls[3][1])

    def test_missing_middle_chunk_recomputes_from_restored_state_then_reuses_suffix(self):
        with BundleStore(self.root / 'store', 100000) as store:
            adapter = self.adapter(store)
            reference, _ = self.run_prepass(adapter)
            entries = list((store.root / 'entries').iterdir())
            for entry in entries:
                import json
                data = json.loads((entry / 'complete.json').read_text())['identity']
                if data['start'] == 2:
                    store.discard(RvmChunkIdentity(canonical(data)))
            self.calls.clear()
            adapter = self.adapter(store)
            repeated, diagnostics = self.run_prepass(adapter)
            np.testing.assert_array_equal(repeated, reference)
            self.assertEqual(len(self.calls), 1)
            self.assertIsNotNone(self.calls[0][1])
            assert isinstance(diagnostics.execution, dict)
            self.assertEqual(diagnostics.execution['rvm_frame_evaluations'], 1)
            self.assertEqual(adapter.frames_reused, 5)

    def test_added_reset_invalidates_affected_chain_without_previous_shot_state(self):
        with BundleStore(self.root / 'store', 100000) as store:
            self.run_prepass(self.adapter(store))
            self.calls.clear()
            adapter = self.adapter(store)
            changed, _ = self.run_prepass(adapter, resets=(0, 2, 3))
            reference, _ = self.run_prepass(resets=(0, 2, 3))
            np.testing.assert_array_equal(changed, reference)
            self.assertEqual(adapter.frames_reused, 5)
            self.assertIsNone(self.calls[0][1])


    def test_native_rvm_code_with_tiny_random_weights_restores_exact_state(self):
        from corridorkey import rvm

        config = {'embed_dim': 16, 'enc_num_layers': 1, 'enc_num_heads': 4, 'enc_head_dim': 4,
                  'enc_mlp_hidden': 32, 'rnn_num_xa_blocks': 1, 'rnn_num_heads': 4,
                  'rnn_head_dim': 4, 'rnn_mlp_hidden': 32}
        with torch.random.fork_rng(devices=[]), patch.dict(rvm.RVM_CFG, {'cpu-fixture': config}):
            torch.manual_seed(1525)
            native = rvm.RVMVideoSiamMAE('cpu-fixture').eval()
        self.model = SimpleNamespace(foundations=SimpleNamespace(
            rvm=native, rvm_mean=torch.zeros(16), rvm_std=torch.ones(16)))
        reference, _ = self.run_prepass()
        with BundleStore(self.root / 'store', 1000000) as store:
            initial, _ = self.run_prepass(self.adapter(store))
            np.testing.assert_array_equal(initial, reference)
        evaluated = []

        def count(module, args):
            evaluated.append(args[0].shape[1])

        hook = native.register_forward_pre_hook(count)
        try:
            with BundleStore(self.root / 'store', 1000000) as store:
                adapter = self.adapter(store)
                repeated, diagnostics = self.run_prepass(adapter)
                np.testing.assert_array_equal(repeated, reference)
                self.assertEqual(evaluated, [])
                self.assertEqual(adapter.frames_reused, 6)
                assert isinstance(diagnostics.execution, dict)
                self.assertEqual(diagnostics.execution['rvm_frame_evaluations'], 0)
                for entry in (store.root / 'entries').iterdir():
                    import json
                    data = json.loads((entry / 'complete.json').read_text())['identity']
                    if data['start'] == 2:
                        store.discard(RvmChunkIdentity(canonical(data)))
                        break
                resumed, diagnostics = self.run_prepass(self.adapter(store))
                np.testing.assert_array_equal(resumed, reference)
                self.assertEqual(evaluated, [1])
        finally:
            hook.remove()
