"""Provider-interface fixtures only; no SAM2 model or weights executed."""
import contextlib
import importlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from corridorkey.control import CancellationToken, CancelledError, JobControl
from corridorkey.subject_guidance import (
    Click,
    ExclusiveResourceLease,
    GuidanceFrame,
    GuidanceRequest,
    GuidanceSession,
    GuidanceUnavailable,
    LocalSAM2Provider,
    ProviderPrediction,
    StaleGuidance,
)


class FixtureProvider:
    device = 'cpu'

    def __init__(self):
        self.frame = None
        self.embeddings = 0
        self.resets = 0
        self.previous = []
        self.active = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.embedding_bytes = 64
        self.output: ProviderPrediction | None = None

    def set_frame(self, frame, control):
        control.check()
        self.frame = frame
        self.embeddings += 1
        return self.embedding_bytes

    def predict(self, clicks, previous_logits, control):
        self.previous.append(previous_logits)
        self.active = True
        self.entered.set()
        if self.block and not self.release.wait(3):
            raise RuntimeError('fixture unblock deadline')
        self.active = False
        control.check()
        if self.output is not None:
            return self.output
        assert self.frame is not None
        mask = bytearray(self.frame.width * self.frame.height)
        for click in clicks:
            mask[click.y * self.frame.width + click.x] = 255 if click.positive else 0
        return ProviderPrediction(bytes(mask), 'binary_u8', struct.pack('<f', 1), '{"provider":"fixture-v1"}', fixture=True)

    def reset(self):
        if self.active:
            raise AssertionError('reset while native operation is active')
        self.resets += 1
        self.frame = None

    def close(self):
        self.reset()


class SubjectGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.provider = FixtureProvider()
        self.exclusive = True
        self.lease = ExclusiveResourceLease('cpu', 'owned-fixture', lambda: self.exclusive)
        self.session = GuidanceSession(self.provider, self.lease)
        self.frame = GuidanceFrame('a' * 64, 12, 'explicit-assumed-sRGB-u8-proxy-v1', 8, 6, bytes(8 * 6 * 3))
        self.session.bind(self.frame)
        self.clicks = (Click(3, 2, True),)
        self.addCleanup(self.session.close)

    def test_import_is_stdlib_only_and_dataclass_classes_are_refused(self):
        output = subprocess.check_output([sys.executable, '-S', '-c',
            'import sys; import corridorkey.subject_guidance; assert not any(k in sys.modules for k in ("numpy","torch","sam2","PySide6")); print("PASS")'],
            cwd=Path(__file__).resolve().parents[1], text=True)
        self.assertEqual(output.strip(), 'PASS')
        invalid_cases: list[tuple[Any, Any]] = [(self.session.bind, GuidanceFrame), (self.session.predict, GuidanceRequest)]
        for operation, value in invalid_cases:
            with self.assertRaises(TypeError):
                operation(value)

    def test_positive_negative_refinement_clear_and_embedding_reuse(self):
        first = self.session.predict(self.session.request(self.clicks))
        self.assertEqual(first.mask_bytes[19], 255)
        clicks = (*self.clicks, Click(3, 2, False), Click(5, 2, True))
        refined = self.session.predict(self.session.request(clicks))
        self.assertEqual(refined.mask_bytes[19], 0)
        self.assertEqual(refined.mask_bytes[21], 255)
        self.assertIsNotNone(self.provider.previous[-1])
        self.assertEqual(self.provider.embeddings, 1)
        self.assertTrue(refined.requires_acceptance)
        provenance = json.loads(refined.provenance_json)
        self.assertIn('no SAM2 inference', provenance['evidence'])
        self.assertEqual(provenance['source_frame'], 12)
        self.session.clear()
        self.assertIsNone(self.session.last_result)
        self.session.predict(self.session.request(self.clicks))
        self.assertIsNone(self.provider.previous[-1])
        self.assertEqual(self.provider.embeddings, 1)

    def test_changed_click_order_restarts_logits_and_new_request_fences_old(self):
        self.session.predict(self.session.request(self.clicks))
        old = self.session.request(self.clicks)
        fresh = self.session.request((Click(2, 2, True), *self.clicks))
        with self.assertRaises(StaleGuidance):
            self.session.predict(old)
        self.session.predict(fresh)
        self.assertIsNone(self.provider.previous[-1])

    def test_clear_during_native_prediction_defers_reset_and_rejects_late_result(self):
        self.provider.block = True
        errors = []
        request = self.session.request(self.clicks)
        def run():
            try:
                self.session.predict(request)
            except StaleGuidance as exc:
                errors.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(self.provider.entered.wait(3))
        resets = self.provider.resets
        self.session.clear()
        self.assertEqual(self.provider.resets, resets)
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            self.session.predict(self.session.request(self.clicks))
        with self.assertRaisesRegex(RuntimeError, 'retire'):
            self.session.close()
        self.provider.release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], StaleGuidance)
        self.assertIsNone(self.session.last_result)
        self.assertGreater(self.provider.resets, resets)

    def test_rebinding_changes_frame_source_decode_identity_and_resets_embedding(self):
        first = self.session.predict(self.session.request(self.clicks))
        for frame in (replace(self.frame, source_frame=13), replace(self.frame, source_sha256='b' * 64),
                      replace(self.frame, decode_identity='different-proxy-interpretation')):
            self.session.bind(frame)
            result = self.session.predict(self.session.request(self.clicks))
            self.assertNotEqual(first.frame_identity, result.frame_identity)
            self.assertNotEqual(first.identity, result.identity)
        self.assertEqual(self.provider.embeddings, 4)

    def test_cancellation_and_lost_lease_never_publish(self):
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(CancelledError):
            self.session.predict(self.session.request(self.clicks), control=JobControl(token))
        self.assertEqual(self.provider.embeddings, 0)
        self.exclusive = False
        with self.assertRaisesRegex(GuidanceUnavailable, 'exclusive'):
            self.session.predict(self.session.request(self.clicks))
        self.assertEqual(self.provider.embeddings, 0)
        self.assertIsNone(self.session.last_result)

    def test_reset_failure_preserves_original_error_and_blocks_reuse(self):
        self.session.predict(self.session.request(self.clicks))
        self.provider.output = ProviderPrediction(bytes([1]) * 48, 'binary_u8', None, '{}', True)
        with patch.object(self.provider, 'reset', side_effect=OSError('fixture native cleanup uncertain')):
            with self.assertRaisesRegex(ValueError, 'Binary coarse guidance') as raised:
                self.session.predict(self.session.request(self.clicks))
            self.assertIn('cleanup NOT VERIFIED', raised.exception.__notes__[0])
        self.assertIsNotNone(self.session.cleanup_error)
        with self.assertRaisesRegex(GuidanceUnavailable, 'cleanup is uncertain'):
            self.session.predict(self.session.request(self.clicks))

    def test_lost_lease_after_native_prediction_rejects_result(self):
        original = self.provider.predict
        def lost(*args):
            result = original(*args)
            self.exclusive = False
            return result
        with patch.object(self.provider, 'predict', side_effect=lost), self.assertRaises(GuidanceUnavailable):
            self.session.predict(self.session.request(self.clicks))
        self.assertIsNone(self.session.last_result)

    def test_bad_clicks_frames_and_payload_bounds_refuse(self):
        invalid: list[Any] = [(Click(True, 2, True),), (Click(8, 2, True),), (Click(2, 2, False),),
                              [Click(2, 2, True)], (Click(2, 2, True),) * 257]
        for clicks in invalid:
            with self.subTest(clicks=str(clicks)[:30]), self.assertRaises(ValueError):
                self.session.predict(self.session.request(clicks))
        with self.assertRaises(ValueError):
            self.session.bind(replace(self.frame, rgb_bytes=bytearray(self.frame.rgb_bytes)))
        with self.assertRaises(ValueError):
            GuidanceSession(self.provider, self.lease, max_pixels=0)
        self.provider.embedding_bytes = 999
        limited = GuidanceSession(self.provider, self.lease, max_embedding_bytes=32)
        limited.bind(self.frame)
        with self.assertRaisesRegex(ValueError, 'embedding'):
            limited.predict(limited.request(self.clicks))
        limited.close()

    def test_invalid_masks_logits_and_nonfinite_provenance_refuse(self):
        for output in (ProviderPrediction(bytes([1]) * 48, 'binary_u8', None, '{}', True),
                       ProviderPrediction(struct.pack('<f', float('nan')) * 48, 'coverage_f32', None, '{}', True),
                       ProviderPrediction(bytes(48), 'binary_u8', b'x', '{}', True),
                       ProviderPrediction(bytes(48), 'binary_u8', None, '{"x":NaN}', True)):
            self.provider.output = output
            with self.assertRaises(ValueError):
                self.session.predict(self.session.request(self.clicks))

    def test_optional_adapter_calls_official_local_api_with_explicit_precision(self):
        # Inject API fixtures, not a real SAM2 package/model execution.
        np = importlib.import_module('numpy')
        calls = []
        active_context = [False]
        tensor = SimpleNamespace(numel=lambda: 8, element_size=lambda: 4)
        model = SimpleNamespace(to=lambda **kwargs: calls.append(('dtype', kwargs)) or model)
        class Predictor:
            def __init__(self, value, **kwargs):
                calls.append(('predictor', kwargs))
                self._features = {'image_embed': tensor, 'high_res_feats': [tensor]}
            def set_image(self, image):
                calls.append(('image', image.shape))
            def predict(self, **kwargs):
                calls.append(('predict', kwargs))
                return np.zeros((1, 6, 8), dtype=np.bool_), np.ones(1), np.zeros((1, 256, 256), np.float32)
            def reset_predictor(self):
                calls.append(('reset', None))
        def build(*args, **kwargs):
            self.assertTrue(active_context[0])
            calls.append(('builder', (args, kwargs)))
            return model
        @contextlib.contextmanager
        def configuration(**kwargs):
            active_context[0] = True
            try:
                yield
            finally:
                active_context[0] = False
        def autocast(*args, **kwargs):
            calls.append(('autocast', kwargs))
            return contextlib.nullcontext()
        torch = SimpleNamespace(__version__='fixture-not-real-torch', float32='float32', bfloat16='bfloat16',
            inference_mode=contextlib.nullcontext, autocast=autocast, get_float32_matmul_precision=lambda: 'high',
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)))
        global_hydra = SimpleNamespace(instance=lambda: SimpleNamespace(is_initialized=lambda: False))
        modules = {'torch': torch, 'numpy': np, 'hydra': SimpleNamespace(initialize_config_dir=configuration),
                   'hydra.core.global_hydra': SimpleNamespace(GlobalHydra=global_hydra),
                   'sam2.build_sam': SimpleNamespace(build_sam2=build),
                   'sam2.sam2_image_predictor': SimpleNamespace(SAM2ImagePredictor=Predictor)}
        with tempfile.TemporaryDirectory() as folder:
            config, checkpoint = Path(folder) / 'tiny.yaml', Path(folder) / 'tiny.pt'
            config.write_text('model: fixture')
            checkpoint.write_bytes(b'fixture-not-weights')
            provider = LocalSAM2Provider(config, checkpoint, device='cpu', lease=self.lease)
            with patch('corridorkey.subject_guidance.importlib.metadata.version', return_value='fixture-source'), \
                    patch('corridorkey.subject_guidance.importlib.import_module', side_effect=lambda name: modules[name]):
                self.assertEqual(provider.set_frame(self.frame, JobControl()), 64)
                prediction = provider.predict(self.clicks, None, JobControl())
                self.assertEqual(len(prediction.mask_bytes), 48)
                self.assertEqual(len(prediction.logits_bytes or b''), 256 * 256 * 4)
                self.assertFalse(active_context[0])
                self.assertEqual(json.loads(prediction.provenance_json)['precision'], 'float32')
                self.assertEqual(json.loads(prediction.provenance_json)['matmul_precision'], 'high')
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)  # CPU policy never mutates irrelevant CUDA flags.
                builder_kwargs = next(value[1] for name, value in calls if name == 'builder')
                self.assertEqual(builder_kwargs['device'], 'cpu')
                self.assertFalse(builder_kwargs['apply_postprocessing'])
                self.assertTrue(all(value['enabled'] is False for name, value in calls if name == 'autocast'))
                provider.close()

    def test_cuda_float32_refuses_each_relaxed_policy_without_global_changes(self):
        for matmul_tf32, cudnn_tf32, precision in ((True, False, 'highest'), (False, True, 'highest'),
                                                  (False, False, 'high'), (False, False, 'medium')):
            with self.subTest(matmul_tf32=matmul_tf32, cudnn_tf32=cudnn_tf32, precision=precision), tempfile.TemporaryDirectory() as folder:
                config, checkpoint = Path(folder) / 'tiny.yaml', Path(folder) / 'tiny.pt'
                config.write_bytes(b'c')
                checkpoint.write_bytes(b'w')
                lease = ExclusiveResourceLease('cuda:0', 'fixture-not-real-GPU', lambda: True)
                provider = LocalSAM2Provider(config, checkpoint, device='cuda:0', lease=lease)
                torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
                    backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=matmul_tf32)),
                                             cudnn=SimpleNamespace(allow_tf32=cudnn_tf32)),
                    get_float32_matmul_precision=lambda precision=precision: precision)
                imports = []
                def imported(name, imports=imports, torch=torch):
                    imports.append(name)
                    if name == 'torch':
                        return torch
                    if name == 'hydra.core.global_hydra':
                        return SimpleNamespace(GlobalHydra=SimpleNamespace(instance=lambda: SimpleNamespace(is_initialized=lambda: False)))
                    return SimpleNamespace()
                with patch('corridorkey.subject_guidance.importlib.import_module', side_effect=imported), \
                        self.assertRaisesRegex(GuidanceUnavailable, 'TF32 or relaxed'):
                    provider.set_frame(self.frame, JobControl())
                self.assertNotIn('sam2.build_sam', imports)
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, matmul_tf32)
                self.assertEqual(torch.backends.cudnn.allow_tf32, cudnn_tf32)
                self.assertEqual(torch.get_float32_matmul_precision(), precision)

    def test_cpu_and_explicit_bfloat16_do_not_refuse_irrelevant_cuda_policy(self):
        torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                 cudnn=SimpleNamespace(allow_tf32=True)), get_float32_matmul_precision=lambda: 'high')
        with tempfile.TemporaryDirectory() as folder:
            cpu = LocalSAM2Provider(Path(folder) / 'config', Path(folder) / 'weights', device='cpu', lease=self.lease)
            cpu._precision_policy(torch)
            cuda = LocalSAM2Provider(Path(folder) / 'config', Path(folder) / 'weights', device='cuda', precision='bfloat16',
                                     lease=ExclusiveResourceLease('cuda', 'fixture', lambda: True))
            cuda._precision_policy(torch)
            strict = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                      cudnn=SimpleNamespace(allow_tf32=False)), get_float32_matmul_precision=lambda: 'highest')
            cuda.precision = 'float32'
            cuda._precision_policy(strict)
            cuda._torch = torch
            with self.assertRaisesRegex(GuidanceUnavailable, 'TF32 or relaxed'):
                cuda._contexts()  # Recheck policy for every embedding/prediction operation.

    def test_initial_asset_identity_survives_dependency_import_boundaries(self):
        for mode in ('config_growth', 'checkpoint_growth', 'new_inode', 'symlink', 'hardlink'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                config, checkpoint = Path(folder) / 'tiny.yaml', Path(folder) / 'tiny.pt'
                config.write_bytes(b'c')
                checkpoint.write_bytes(b'w')
                provider = LocalSAM2Provider(config, checkpoint, device='cpu', lease=self.lease)
                imported_names = []
                def imported(name, imported_names=imported_names, mode=mode, config=config, checkpoint=checkpoint):
                    imported_names.append(name)
                    if name == 'torch':
                        if mode == 'config_growth':
                            with config.open('r+b') as stream:
                                stream.truncate(4 * 1024**2 + 1)
                        elif mode == 'checkpoint_growth':
                            with checkpoint.open('r+b') as stream:
                                stream.truncate(2 * 1024**3 + 1)  # Sparse local fixture; no asset download/allocation.
                        elif mode == 'new_inode':
                            replacement = config.with_suffix('.new')
                            replacement.write_bytes(b'c')
                            os.replace(replacement, config)
                        elif mode == 'symlink':
                            config.unlink()
                            config.symlink_to(checkpoint)
                        else:
                            os.link(config, config.with_suffix('.alias'))
                    if name == 'hydra.core.global_hydra':
                        return SimpleNamespace(GlobalHydra=SimpleNamespace(instance=lambda: SimpleNamespace(is_initialized=lambda: False)))
                    return SimpleNamespace()
                with patch('corridorkey.subject_guidance.importlib.import_module', side_effect=imported), \
                        self.assertRaises((GuidanceUnavailable, ValueError)):
                    provider.set_frame(self.frame, JobControl())
                self.assertNotIn('sam2.build_sam', imported_names)
                self.assertIsNone(provider.predictor)

    def test_asset_pin_is_rechecked_before_and_after_model_loading(self):
        for stage in ('provider_import', 'model_load'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                config, checkpoint = Path(folder) / 'tiny.yaml', Path(folder) / 'tiny.pt'
                config.write_bytes(b'c')
                checkpoint.write_bytes(b'w')
                provider = LocalSAM2Provider(config, checkpoint, device='cpu', lease=self.lease)
                calls = []
                model: Any = SimpleNamespace()
                model.to = lambda model=model, **kwargs: model
                def builder(*args, config=config, calls=calls, model=model, **kwargs):
                    calls.append('model')
                    config.write_bytes(b'changed during model loading')
                    return model
                def imported(name, stage=stage, config=config, builder=builder):
                    if name == 'torch':
                        return SimpleNamespace(float32='float32')
                    if name == 'hydra':
                        return SimpleNamespace(initialize_config_dir=lambda **kwargs: contextlib.nullcontext())
                    if name == 'hydra.core.global_hydra':
                        return SimpleNamespace(GlobalHydra=SimpleNamespace(instance=lambda: SimpleNamespace(is_initialized=lambda: False)))
                    if name == 'sam2.build_sam':
                        if stage == 'provider_import':
                            config.write_bytes(b'changed during module import')
                        return SimpleNamespace(build_sam2=builder)
                    return SimpleNamespace(SAM2ImagePredictor=lambda model, **kwargs: SimpleNamespace())
                with patch('corridorkey.subject_guidance.importlib.import_module', side_effect=imported), \
                        self.assertRaisesRegex(ValueError, 'initially admitted identity'):
                    provider.set_frame(self.frame, JobControl())
                self.assertEqual(calls, [] if stage == 'provider_import' else ['model'])
                self.assertIsNone(provider.predictor)

    def test_optional_local_adapter_missing_assets_is_actionable_without_import_or_download(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LocalSAM2Provider(Path(folder) / 'tiny.yaml', Path(folder) / 'tiny.pt', device='cpu', lease=self.lease)
            with patch('corridorkey.subject_guidance.importlib.import_module', side_effect=AssertionError('must not import model')), \
                    self.assertRaisesRegex(GuidanceUnavailable, 'No download'):
                provider.set_frame(self.frame, JobControl())
            provider.close()

    def test_missing_dependencies_and_conflicting_hydra_leave_environment_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            config, checkpoint = Path(folder) / 'tiny.yaml', Path(folder) / 'tiny.pt'
            config.write_text('model: local fixture, not an executable model')
            checkpoint.write_bytes(b'fixture, not weights')
            provider = LocalSAM2Provider(config, checkpoint, device='cpu', lease=self.lease)
            with patch('corridorkey.subject_guidance.importlib.import_module', side_effect=ModuleNotFoundError('optional SAM2 missing')), \
                    self.assertRaisesRegex(GuidanceUnavailable, 'isolated environment'):
                provider.set_frame(self.frame, JobControl())
            fake: Any = type('Hydra', (), {'instance': lambda: type('Global', (), {'is_initialized': lambda self: True})()})
            actual_import = importlib.import_module
            def imported(name):
                if name == 'hydra.core.global_hydra':
                    return type('Module', (), {'GlobalHydra': fake})
                return actual_import(name)
            # No Hydra install is needed: injected module fixtures exercise only admission.
            with patch('corridorkey.subject_guidance.importlib.import_module', side_effect=lambda name: imported(name) if name == 'hydra.core.global_hydra' else object()), \
                    self.assertRaisesRegex(GuidanceUnavailable, 'context must remain untouched'):
                provider.set_frame(self.frame, JobControl())


if __name__ == '__main__':
    unittest.main()
