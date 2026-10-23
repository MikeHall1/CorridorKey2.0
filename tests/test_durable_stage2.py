"""Exact Stage2 batches and real CPU runtime replay; neural substitutes only."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

HAS_STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2', 'safetensors'))
if HAS_STACK or TYPE_CHECKING:
    import torch
    from safetensors.torch import save

    from corridorkey.durable_rvm import tensor_digest
    from corridorkey.durable_stage2 import DurableStage2Batches
    from corridorkey.recovery_identity import ReplayIdentity, canonical
    from corridorkey.recovery_store import BundleStore, CacheInvalid
    from corridorkey.stage2_identity import Stage2BatchIdentity


def make_cpu_model(*, dtype=None, tuple_output=False) -> Any:
    """Public tiny test fixture for shared-runtime/worker integration, never production fallback."""
    class Foundations(torch.nn.Module):
        rvm = None
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, *, rgb_cradio, rgb_moge):
            return {'moge': torch.zeros(len(rgb_moge), 1, 1, 1)}

    class Stage1(torch.nn.Module):
        d_16 = moge_dim = 1
        cradio_dim = 0
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, rgb, hint, moge_feat, **kwargs):
            count = rgb.shape[1]
            return {'baton': torch.full((1, count, 1, 4, 4), 0.4),
                    'aux': torch.full((1, count, 4, 32, 32), 0.4)}

    class Stage2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = []
        def forward(self, rgb, baton):
            self.calls.append(rgb.shape[1])
            result = torch.zeros(1, rgb.shape[1], 4, 64, 64, dtype=dtype or torch.float32)
            result[:, :, 0] = 0.4
            return (result, torch.ones_like(result)) if tuple_output else result

    model = torch.nn.Module()
    model.foundations, model.stage1, model.stage2 = Foundations(), Stage1(), Stage2()
    return model


@contextmanager
def cpu_runtime_context():
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda module: module))
        stack.enter_context(patch.object(torch.cuda, 'synchronize'))
        stack.enter_context(patch.object(torch.cuda, 'empty_cache'))
        stack.enter_context(patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()))
        yield


@unittest.skipUnless(HAS_STACK, 'requires optional CPU numeric stack')
class DurableStage2Tests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
            'assets': {}, 'environment': {'runtime': {'synthetic': True}}}))
        self.rgb = torch.zeros(1, 2, 3, 64, 64)
        self.baton = torch.zeros(1, 2, 1, 4, 4)
        self.prediction = torch.arange(2 * 4 * 64 * 64).reshape(1, 2, 4, 64, 64).float() / 100000

    def adapter(self, store):
        adapter = DurableStage2Batches(store, self.parent, validate_context=lambda: None)
        adapter.bind(None)
        return adapter

    def identity(self, adapter, **overrides):
        identity = adapter.identity(**{'frames': (0, 4), 'policy': 'quick_anchor',
                                   'rgb': self.rgb, 'baton': self.baton, **overrides})
        assert identity is not None
        return identity

    def test_exact_dtypes_signed_zero_and_independent_loaded_values(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            prediction = self.prediction.to(dtype)
            prediction[0, 0, 0, 0, 0] = -0.0
            with BundleStore(self.root / str(dtype), 2_000_000) as store:
                adapter = self.adapter(store)
                identity = self.identity(adapter)
                self.assertTrue(adapter.put(identity, prediction))
                restored = adapter.get(identity)
                self.assertEqual(tensor_digest(restored), tensor_digest(prediction))
                restored.zero_()
                self.assertEqual(tensor_digest(adapter.get(identity)), tensor_digest(prediction))
                self.assertEqual(adapter.snapshot()['stage2_frame_instances_reused'], 4)

    def test_identity_binds_whole_actual_call_and_parent(self):
        with BundleStore(self.root / 'store', 2_000_000) as store:
            adapter = self.adapter(store)
            original = self.identity(adapter)
            for change in ({'frames': (4, 0)}, {'frames': (1, 4)}, {'policy': 'quick_fallback'},
                           {'rgb': self.rgb.bfloat16()}, {'baton': self.baton.half()},
                           {'autocast_dtype': 'torch.float16'}):
                self.assertNotEqual(original.key, self.identity(adapter, **change).key)
            self.rgb[0, 0, 0, 0, 0] = -0.0
            self.assertNotEqual(original.key, self.identity(adapter).key)
            with self.assertRaises(ValueError):
                self.identity(adapter, frames=(True, 4))
            with self.assertRaises(ValueError):
                self.identity(adapter, frames=(0,))
            with self.assertRaises(CacheInvalid):
                adapter.get(Stage2BatchIdentity)
            with self.assertRaises(ValueError):
                bad_payload: Any = original
                Stage2BatchIdentity(bad_payload)
            data = original.to_dict()
            data['parent_key'] = 'f' * 64
            with self.assertRaises(CacheInvalid):
                adapter.get(Stage2BatchIdentity(canonical(data)))
            with self.assertRaises(ValueError):
                Stage2BatchIdentity(original.payload.replace('"schema_version":1', '"schema_version":true'))

    def test_oversize_and_zero_budget_skip_before_hash_or_host_copy(self):
        for limit in (0, 100):
            with BundleStore(self.root / str(limit), limit) as store:
                adapter = self.adapter(store)
                with patch('corridorkey.durable_stage2.tensor_digest', side_effect=AssertionError('must not hash')):
                    self.assertIsNone(adapter.identity(frames=(0, 4), policy='quick_anchor', rgb=self.rgb, baton=self.baton))
                self.assertEqual(adapter.budget_rejected, 1)
        with BundleStore(self.root / 'size', 2_000_000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            with (patch('corridorkey.durable_stage2.MAX_TENSOR_BYTES', 10),
                  patch('corridorkey.durable_stage2.save', side_effect=AssertionError('must not serialize'))):
                self.assertFalse(adapter.put(identity, self.prediction))

    def test_corrupt_wrong_provenance_and_header_never_materialize(self):
        with BundleStore(self.root / 'store', 2_000_000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            path = self.root / 'bad'
            for blob, record in ((save({'wrong': self.prediction}), adapter._record(self.prediction)),
                                 (save({'prediction': self.prediction}), None)):
                path.write_bytes(blob)
                store.publish(identity, {'prediction.safetensors': path}, record)
                with patch('corridorkey.durable_stage2.load', side_effect=AssertionError('must validate before load')):
                    self.assertIsNone(adapter.get(identity))
                self.assertFalse(store._entry(identity).exists())
            adapter.put(identity, self.prediction)
            payload = next(store._entry(identity).glob('payload-*'))
            payload.write_bytes(b'corrupt')
            self.assertIsNone(adapter.get(identity))
            self.assertEqual(adapter.corrupt, 3)

    def test_bounded_metadata_growth_and_entry_count_before_materialization(self):
        from corridorkey.recovery_store import _read_record_bounded
        path = self.root / 'metadata.json'
        path.write_bytes(b'{}')
        original = os.fdopen
        requested = []
        class GrowingReader:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def fileno(self):
                return self.stream.fileno()
            def read(self, size):
                requested.append(size)
                with path.open('ab') as writer:
                    writer.write(b'x' * 1000)
                return self.stream.read(size)
        def opened(descriptor, *args, **kwargs):
            return GrowingReader(original(descriptor, *args, **kwargs))
        with patch('corridorkey.recovery_store.os.fdopen', opened), self.assertRaises(CacheInvalid):
            _read_record_bounded(path, 8)
        self.assertEqual(requested, [3])
        with BundleStore(self.root / 'entrybound', 2_000_000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            adapter.put(identity, self.prediction)
            with self.assertRaises(CacheInvalid):
                store.lookup(identity, max_metadata_bytes=4, max_entry_files=3)
            with self.assertRaises(CacheInvalid):
                store._owned(store._entry(identity), max_entry_files=2)

    def test_declared_payload_growth_stops_before_excess_write_or_hash(self):
        import hashlib

        from corridorkey.recovery_store import _copy_checked, _file_sha256_bounded
        source, target = self.root / 'growing', self.root / 'partial'
        source.write_bytes(b'abcEXTRA')
        with self.assertRaises(CacheInvalid):
            _copy_checked(source, target, hashlib.sha256(b'abc').hexdigest(), expected_size=3)
        self.assertLessEqual(target.stat().st_size, 3)
        source.write_bytes(b'abc')
        calls = 0
        def grow():
            nonlocal calls
            calls += 1
            if calls == 1:
                with source.open('ab') as writer:
                    writer.write(b'EXTRA')
        with self.assertRaises(CacheInvalid):
            _file_sha256_bounded(source, 3, check=grow)
        self.assertEqual(calls, 1)

    def test_disk_failure_cancellation_and_sealed_prefix_preservation(self):
        from corridorkey.control import CancelledError
        with BundleStore(self.root / 'store', 2_000_000) as store:
            adapter = self.adapter(store)
            identity = self.identity(adapter)
            with patch.object(Path, 'write_bytes', side_effect=OSError('disk full')), self.assertRaises(OSError):
                adapter.put(identity, self.prediction)
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
            adapter.check = lambda: (_ for _ in ()).throw(CancelledError('stop'))
            with self.assertRaises(CancelledError):
                adapter.put(identity, self.prediction)
            adapter.check = None
            self.assertTrue(adapter.put(identity, self.prediction))
            self.assertEqual(tensor_digest(adapter.get(identity)), tensor_digest(self.prediction))
            self.assertEqual(list((store.root / 'staging').iterdir()), [])

    def run_runtime(self, folder, adapter, *, low=False, quick=False, model=None, settings=None, corrections=False, diagnostics=None):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig, WriterConfig
        from corridorkey.diagnostics import RunDiagnostics
        from corridorkey.runtime import run_inference
        root = Path(folder)
        source = self.root / 'source'
        source.mkdir(exist_ok=True)
        for frame in range(9):
            path = source / f'frame{frame}.png'
            if not path.exists():
                cv2.imwrite(str(path), np.full((64, 64, 3), 80 + frame * 4, np.uint8))
        cfg = InferenceConfig(hann_chunk=8, hann_stride=4, low_vram=low, shape_quantum=64,
            quick_key=quick, quick_interval=4, quick_context_radius=1, quick_fallback='neural',
            outputs={'alpha', 'fg'}, output_format='exr', exr_dtype='float',
            temp_dir=root / 'temp', stage2_batch=2, despill_strength=0)
        for name, value in (settings or {}).items():
            setattr(cfg, name, value)
        if corrections:
            from corridorkey.corrections import CorrectionDocument, save_corrections
            from corridorkey.identities import file_sha256, source_identity
            mask = self.root / 'keep.png'
            document_path = self.root / 'corrections.json'
            if not document_path.exists():
                cv2.imwrite(str(mask), np.full((64, 64), 255, np.uint8))
                document = {'schema_version': 1, 'revision': 1,
                    'source': {'sha256': source_identity(source)['sha256'], 'start_frame': 0,
                        'frame_count': 9, 'fps': 24, 'width': 64, 'height': 64},
                    'shots': [[0, 9]], 'entries': [{'id': 'keep', 'revision': 1, 'kind': 'keep',
                        'frame': 1, 'asset': {'path': mask.name, 'sha256': file_sha256(mask)},
                        'strength': 0.5, 'direction': 'both', 'range': [1, 2]}]}
                save_corrections(document_path, CorrectionDocument.parse(document, self.root))
            cfg.corrections_path = document_path
        diagnostics = RunDiagnostics() if diagnostics is None else diagnostics
        diagnostics.durable_stage2 = adapter
        model = make_cpu_model() if model is None else model
        with cpu_runtime_context():
            run_inference(model, str(source), str(root / 'output'), cfg, WriterConfig(),
                initial_hint_path=None, hint_video_path=None, diagnostics=diagnostics)
        self.assertEqual(list(cfg.temp_dir.iterdir()), [])
        return model, diagnostics

    def test_real_full_low_and_quick_replay_raw_tuple_before_refinement(self):
        import cv2
        import numpy as np
        for low, quick in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(low=low, quick=quick), BundleStore(self.root / f'store{low}{quick}', 4_000_000) as store:
                adapter = self.adapter(store)
                _, first = self.run_runtime(self.root / f'first{low}{quick}', adapter, low=low, quick=quick,
                    model=make_cpu_model(tuple_output=True), settings={'alpha_refine_radius': 1})
                before = adapter.stored
                model, second = self.run_runtime(self.root / f'second{low}{quick}', adapter, low=low, quick=quick,
                    model=make_cpu_model(tuple_output=True), settings={'alpha_refine_radius': 1})
                self.assertEqual(model.stage2.calls, [])
                assert first.execution is not None and second.execution is not None
                self.assertEqual(second.execution['stage2_calls'], 0)
                self.assertEqual(second.execution['stage2_frame_evaluations'], 0)
                self.assertEqual(second.execution['completed_prediction_frames'], first.execution['completed_prediction_frames'])
                self.assertGreater(adapter.hits, 0)
                self.assertEqual(adapter.stored, before)
                assert first.alpha_refinement is not None and second.alpha_refinement is not None
                self.assertEqual(first.alpha_refinement['frames_evaluated'], second.alpha_refinement['frames_evaluated'])
                for name in first.outputs:
                    for left, right in zip(sorted(Path(first.outputs[name]).glob('*.exr')),
                                           sorted(Path(second.outputs[name]).glob('*.exr'))):
                        np.testing.assert_array_equal(cv2.imread(str(left), -1), cv2.imread(str(right), -1))

    def test_neural_fallback_batches_and_corrected_outputs_replay_once(self):
        import cv2
        import numpy as np

        from corridorkey.diagnostics import RunDiagnostics
        from tests.test_quick_runtime import QuickRuntimeTests
        for low in (False, True):
            with BundleStore(self.root / f'fallback{low}', 4_000_000) as store:
                adapter = self.adapter(store)
                records = []
                helper = QuickRuntimeTests()
                for name in ('first', 'replay'):
                    diagnostics = RunDiagnostics()
                    diagnostics.durable_stage2 = adapter
                    (self.root / f'{name}-fallback{low}').mkdir()
                    helper.run_job(self.root / f'{name}-fallback{low}', low_vram=low,
                        fallback='neural', flashing=True, diagnostics=diagnostics)
                    records.append(diagnostics)
                assert records[0].quick_key is not None and records[1].execution is not None
                self.assertEqual(records[0].quick_key['extra_neural_frames'], [1, 3, 5, 7])
                self.assertEqual(records[1].execution['stage2_calls'], 0)
                self.assertEqual(records[1].execution['completed_prediction_frames'], 7)
            with BundleStore(self.root / f'correction{low}', 4_000_000) as store:
                adapter = self.adapter(store)
                _, first = self.run_runtime(self.root / f'first-correction{low}', adapter,
                    low=low, corrections=True)
                _, replay = self.run_runtime(self.root / f'replay-correction{low}', adapter,
                    low=low, corrections=True)
                assert first.corrections is not None and replay.corrections is not None
                self.assertEqual(first.corrections['output_frames_refined'], 1)
                self.assertEqual(replay.corrections['output_frames_refined'], 1)
                for left, right in zip(sorted(Path(first.outputs['alpha']).glob('*.exr')),
                                       sorted(Path(replay.outputs['alpha']).glob('*.exr'))):
                    np.testing.assert_array_equal(cv2.imread(str(left), -1), cv2.imread(str(right), -1))
                frame = sorted(Path(replay.outputs['alpha']).glob('*.exr'))[1]
                np.testing.assert_allclose(cv2.imread(str(frame), -1), 0.7, atol=1e-6)

    def test_malformed_output_counts_forward_before_tuple_selection(self):
        from corridorkey.diagnostics import RunDiagnostics
        diagnostics = RunDiagnostics()
        model = make_cpu_model()
        with patch.object(model.stage2, 'forward', return_value=()) as forward, self.assertRaises(IndexError):
            self.run_runtime(self.root / 'malformed', None, model=model, diagnostics=diagnostics)
        assert diagnostics.execution is not None
        self.assertEqual(diagnostics.execution['stage2_calls'], 1)
        self.assertEqual(diagnostics.execution['stage2_frame_evaluations'], forward.call_args.args[0].shape[1])
        self.assertGreater(diagnostics.execution['stage2_frame_evaluations'], 0)
        self.assertEqual(diagnostics.execution['completed_prediction_frames'], 0)
        self.assertEqual(list((self.root / 'malformed' / 'temp').iterdir()), [])

    def test_default_off_has_no_durable_hash_and_failure_reuses_prefix(self):
        with patch('corridorkey.durable_stage2.tensor_digest', side_effect=AssertionError('default must not hash')):
            _, default = self.run_runtime(self.root / 'default', None)
        self.assertIsNone(default.durable_stage2)
        with BundleStore(self.root / 'prefix', 4_000_000) as store:
            adapter = self.adapter(store)
            original = adapter.put
            calls = 0
            def stop_after_seal(*args):
                nonlocal calls
                result = original(*args)
                calls += 1
                if calls == 2:
                    raise OSError('interrupted after sealed prefix')
                return result
            with patch.object(adapter, 'put', side_effect=stop_after_seal), self.assertRaises(OSError):
                self.run_runtime(self.root / 'failed', adapter)
            self.assertEqual(list((self.root / 'failed' / 'temp').iterdir()), [])
            resumed = self.adapter(store)
            _, record = self.run_runtime(self.root / 'resumed', resumed)
            self.assertEqual(resumed.hits, 2)
            assert record.execution is not None and default.execution is not None
            self.assertLess(record.execution['stage2_frame_evaluations'], default.execution['stage2_frame_evaluations'])
            self.assertEqual(record.frames_submitted, 9)
