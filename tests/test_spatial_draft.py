"""Shape-aware CPU neural substitutes; no pretrained/GPU quality claims."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2'))


def make_spatial_cpu_model(*, use_rvm=False, use_cradio=False) -> Any:
    import torch

    from corridorkey import color_utils as cu

    class RVM(torch.nn.Module):
        embed_dim = 1
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, rgb, state=None):
            return torch.zeros(1, rgb.shape[1], 1, rgb.shape[-2] // 16, rgb.shape[-1] // 16), 1

    class Foundations(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.rvm = RVM() if use_rvm else None
            self.rvm_mean, self.rvm_std = torch.zeros(1), torch.ones(1)
        def forward(self, *, rgb_cradio, rgb_moge):
            shape = (len(rgb_moge), 1, rgb_moge.shape[-2] // 14, rgb_moge.shape[-1] // 14)
            result = {'moge': torch.zeros(shape)}
            if use_cradio:
                result['cradio'] = torch.zeros(shape)
            return result

    class Stage1(torch.nn.Module):
        d_16 = moge_dim = 1
        cradio_dim = int(use_cradio)
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.seen_hints, self.shapes = [], []
        def forward(self, rgb, hint, moge_feat, **kwargs):
            b, t, _, h, w = rgb.shape
            self.shapes.append((h, w))
            self.seen_hints.append(hint.clone())
            return {'baton': torch.full((b, t, 1, h // 16, w // 16), .4),
                    'aux': torch.full((b, t, 4, h // 2, w // 2), .4)}

    class Stage2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.shapes = []
        def forward(self, rgb, baton):
            b, t, _, h, w = rgb.shape
            if tuple(baton.shape) != (b, t, 1, h // 16, w // 16):
                raise AssertionError('Shape-aware fixture baton mismatch')
            self.shapes.append((h, w))
            alpha = torch.full((b, t, 1, h, w), .4)
            fg = torch.tensor([-1.0, 2.0, .5]).view(1, 1, 3, 1, 1).expand(b, t, 3, h, w)
            return torch.cat((alpha, cu.linear_to_asinh(fg)), dim=2)

    model = torch.nn.Module()
    model.foundations, model.stage1, model.stage2 = Foundations(), Stage1(), Stage2()
    return model


@contextmanager
def cpu_spatial_context():
    import torch
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda module: module))
        stack.enter_context(patch.object(torch.cuda, 'synchronize'))
        stack.enter_context(patch.object(torch.cuda, 'empty_cache'))
        stack.enter_context(patch.object(torch, 'autocast', side_effect=lambda *args, **kwargs: nullcontext()))
        yield


@unittest.skipUnless(STACK, 'requires optional CPU numeric stack')
class SpatialDraftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def source(self, *, exr=False, frames=5, name='input'):
        import cv2
        import numpy as np
        source = self.root / name
        source.mkdir()
        for frame in range(frames):
            image = np.full((65, 97, 3), 1000 + frame * 5000, dtype=np.uint16)
            image[10:30, 20:50, :] = 50000
            if exr:
                image = image.astype(np.float32) / 65535 * 4 - 1
                extension = 'exr'
            else:
                extension = 'png'
            self.assertTrue(cv2.imwrite(str(source / f'{frame:04}.{extension}'), image))
        return source

    def corrections(self, source, *, frames=5):
        import cv2
        import numpy as np

        from corridorkey.corrections import CorrectionDocument, save_corrections
        from corridorkey.identities import source_identity
        yy, xx = np.mgrid[:65, :97]
        mask = ((xx + yy) % 2 * 65535).astype(np.uint16)
        path = self.root / 'mask.png'
        cv2.imwrite(str(path), mask)
        from corridorkey.identities import file_sha256
        data = {'schema_version': 1, 'revision': 1,
                'source': {'sha256': source_identity(source)['sha256'], 'start_frame': 0,
                           'frame_count': frames, 'fps': 24.0, 'width': 97, 'height': 65},
                'shots': [[0, frames]],
                'entries': [{'id': 'matte', 'revision': 1, 'frame': 0, 'kind': 'matte',
                             'asset': {'path': str(path), 'sha256': file_sha256(path)},
                             'strength': .5, 'direction': 'both', 'range': [0, frames]}]}
        document = CorrectionDocument.parse(data, self.root)
        destination = self.root / 'corrections.json'
        save_corrections(destination, document)
        return destination, mask.astype(np.float32) / 65535

    def run_job(self, source, *, scale=.5, low=False, quick=False, refine=False, correction=None,
                outputs=None, use_rvm=False, use_cradio=False, video_writer=None, overrides=None, control=None, first_hint=None, full_hint=None):
        from corridorkey import runtime
        from corridorkey.config import InferenceConfig, WriterConfig
        from corridorkey.diagnostics import RunDiagnostics
        model = make_spatial_cpu_model(use_rvm=use_rvm, use_cradio=use_cradio)
        output = self.root / ('out-' + str(len(list(self.root.glob('out-*')))))
        cfg = InferenceConfig(model_scale=scale, shape_quantum=64, hann_chunk=4, hann_stride=2,
                              auto_vram=False, low_vram=low, stage2_batch=2, outputs=outputs or {'alpha', 'fg'},
                              output_format='exr', exr_dtype='float', frame_dir_linear=source.suffix != '.avi' and any(source.glob('*.exr')),
                              temp_dir=self.root / 'scratch', quick_key=quick, quick_interval=3, quick_context_radius=1,
                              quick_fallback='neural', alpha_refine_radius=1 if refine else 0,
                              corrections_path=correction, despill_strength=0)
        for name, value in (overrides or {}).items():
            setattr(cfg, name, value)
        diagnostics = RunDiagnostics()
        if control is not None:
            diagnostics.control = control
        with ExitStack() as stack:
            stack.enter_context(cpu_spatial_context())
            if video_writer is not None:
                stack.enter_context(patch.object(runtime, 'ThreadedVideoWriter', video_writer))
            runtime.run_inference(model, str(source), str(output), cfg, WriterConfig(),
                                  initial_hint_path=first_hint, hint_video_path=full_hint, diagnostics=diagnostics)
        return model, diagnostics, output

    def test_odd_uint16_hdr_fullrate_all_outputs_both_memory_paths(self):
        import cv2
        import numpy as np
        for exr in (False, True):
            source = self.source(exr=exr, name='hdr' if exr else 'png')
            before = {p: p.read_bytes() for p in source.iterdir()}
            for low in (False, True):
                with self.subTest(exr=exr, low=low):
                    model, diagnostics, _ = self.run_job(source, low=low, outputs={'alpha', 'fg', 'checker', 'cutout'},
                                                          use_rvm=True, use_cradio=True)
                    assert diagnostics.source is not None
                    self.assertEqual(set(model.stage2.shapes), {(64, 64)})
                    self.assertEqual(diagnostics.frames_submitted, 5)
                    self.assertEqual(diagnostics.source['model_native_hw'], [32, 48])
                    self.assertEqual(diagnostics.source['native_hw'], [65, 97])
                    self.assertTrue(diagnostics.source['spatial_draft']['model_grid_reduced'])
                    for role, directory in diagnostics.outputs.items():
                        paths = list(Path(directory).glob('*.exr'))
                        self.assertEqual(len(paths), 5)
                        image = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
                        self.assertEqual(image.shape[:2], (65, 97))
                        self.assertTrue(np.isfinite(image).all())
                        self.assertIn('_spatial_draft', paths[0].name)
                        self.assertTrue(diagnostics.output_contracts[role]['spatial_draft']['enabled'])
                        if role == 'alpha':
                            np.testing.assert_allclose(image, .4, atol=2e-6)
                        if role == 'fg':
                            np.testing.assert_allclose(image, np.broadcast_to([.5, 2, -1], image.shape), atol=2e-6)
                    self.assertEqual(list((self.root / 'scratch').iterdir()), [])
            self.assertEqual({p: p.read_bytes() for p in source.iterdir()}, before)

    def test_native_constraints_after_refinement_are_not_downsampled_or_doubled(self):
        import cv2
        import numpy as np
        source = self.source()
        correction, mask = self.corrections(source)
        expected = .4 * .5 + mask * .5
        for low in (False, True):
            _, diagnostics, _ = self.run_job(source, low=low, correction=correction, refine=True)
            for path in Path(diagnostics.outputs['alpha']).glob('*.exr'):
                np.testing.assert_allclose(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), expected, atol=2e-6)
            assert diagnostics.corrections is not None and diagnostics.execution is not None
            self.assertEqual(diagnostics.corrections['output_frames_refined'], 5)
            self.assertEqual(diagnostics.execution['spatial_guidance']['status'], 'closed')
            self.assertEqual(diagnostics.execution['spatial_guidance']['frames_requested'], 5)

    def test_quick_fallback_native_guidance_restarts_and_native_masks_once(self):
        import cv2
        import numpy as np

        from corridorkey import draft
        from corridorkey.motion import Propagation
        source = self.source()
        correction, mask = self.corrections(source)
        def uncertain(*args, **kwargs):
            alpha = np.full((65, 97), .4, dtype=np.float32)
            return Propagation(alpha, np.zeros_like(alpha), np.ones_like(alpha, dtype=bool),
                               {'flow_evaluations': 4, 'motion_seconds': 0.0, 'uncertain_fraction': 1.0, 'motion_fraction': 0.0})
        original = draft.offline_frames
        def forced_frames(*args, **kwargs):
            return original(*args, interpolate=uncertain, **kwargs)
        for low in (False, True):
            with patch('corridorkey.draft.offline_frames', forced_frames):
                _, diagnostics, _ = self.run_job(source, low=low, quick=True, correction=correction, refine=True)
            assert diagnostics.quick_key is not None and diagnostics.execution is not None
            self.assertGreater(len(diagnostics.quick_key['extra_neural_frames']), 0)
            self.assertGreater(diagnostics.execution['spatial_guidance']['reader_restarts'], 0)
            self.assertEqual(diagnostics.frames_submitted, 5)
            for path in Path(diagnostics.outputs['alpha']).glob('*.exr'):
                np.testing.assert_allclose(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), .2 + .5 * mask, atol=2e-6)

    def test_scaled_first_and_full_hints_preserve_selected_source_alignment(self):
        import cv2
        import numpy as np
        import torch

        source = self.source()
        hints = self.root / 'hints'
        hints.mkdir()
        codes = [6553, 13107, 19660, 26214, 32767]
        for frame, code in enumerate(codes):
            self.assertTrue(cv2.imwrite(str(hints / f'{frame:04}.png'), np.full((65, 97), code, dtype=np.uint16)))
        first = self.root / 'first.png'
        self.assertTrue(cv2.imwrite(str(first), np.full((65, 97), 49151, dtype=np.uint16)))
        protected = [*source.iterdir(), *hints.iterdir(), first]
        before = {path: path.read_bytes() for path in protected}
        for low in (False, True):
            model, diagnostics, _ = self.run_job(source, low=low, first_hint=str(first), full_hint=str(hints),
                                                 overrides={'start_frame': 1, 'num_frames': 3, 'carry_hint': False})
            assert diagnostics.source is not None
            self.assertEqual(diagnostics.source['start_frame'], 1)
            self.assertEqual(diagnostics.frames_submitted, 3)
            self.assertEqual(len(model.stage1.seen_hints), 1)
            native_model = model.stage1.seen_hints[0][0, :, 0, 16:48, 8:56]
            expected = torch.tensor(np.array([49151, codes[2], codes[3]], dtype=np.float32) / np.float32(65535)).to(torch.bfloat16)
            self.assertTrue(torch.equal(native_model, expected[:, None, None].expand(3, 32, 48)))
        self.assertEqual({path: path.read_bytes() for path in protected}, before)

    def test_premultiplied_hdr_rgba_decode_before_linear_resize_and_native_output_association(self):
        import cv2
        import numpy as np
        import torch

        from corridorkey import color_utils as cu
        from corridorkey import video_io

        source = self.root / 'rgba'
        source.mkdir()
        yy, xx = np.mgrid[:65, :97]
        alpha = np.where((xx + yy) % 2, .25, 1).astype(np.float32)
        alpha[0, 0] = 0
        straight = np.broadcast_to(np.array([.25, 2, -.5], dtype=np.float32), (65, 97, 3)).copy()
        image = np.concatenate((straight[..., ::-1] * alpha[..., None], alpha[..., None]), axis=-1)
        for frame in range(2):
            self.assertTrue(cv2.imwrite(str(source / f'{frame:04}.exr'), image,
                                       [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]))
        before = {path: path.read_bytes() for path in source.iterdir()}
        meta = video_io.probe_frame_dir_meta(str(source), num_frames=2, shape_quantum=64, fps=24,
                                            exr_linear=True, model_scale=.5, alpha_mode='premultiplied',
                                            source_colour_profile='linear_bt709_d65')
        _, decoded = video_io._read_frame_dir_views(meta['frame_paths'][0], exr_linear=True, alpha_mode='premultiplied',
                                                   source_colour_profile='linear_bt709_d65')
        expected_native = straight.copy()
        expected_native[0, 0] = 0
        np.testing.assert_array_equal(decoded, expected_native)
        original_views = video_io._spatial_model_views
        def captured_preprocess(path):
            seen = []
            def capture(linear, review_linear, metadata, *, need_model):
                seen.append(linear.copy())
                return original_views(linear, review_linear, metadata, need_model=need_model)
            with patch.object(video_io, '_spatial_model_views', capture):
                model_rgb, _, _ = video_io.preprocess_frame_dir_path(path, meta, need_model=True)
            assert isinstance(model_rgb, torch.Tensor)
            self.assertEqual(len(seen), 1)
            return model_rgb, seen[0]

        model_rgb, pre_linear = captured_preprocess(meta['frame_paths'][0])
        # This boundary is float32 linear RGB, before model encoding or bfloat16.
        # 64 float32 eps covers the finite eight-tap separable filter arithmetic;
        # the constant interior is distant from the explicitly zero-coverage pixel.
        resample_bound = float(64 * np.finfo(np.float32).eps * 3)
        np.testing.assert_allclose(pre_linear[8:24, 8:40],
                                   np.broadcast_to([.25, 2, -.5], (16, 32, 3)), rtol=0, atol=resample_bound)
        # Independently bound bfloat16 quantization in ENCODED space: half of
        # a bfloat16 binade ULP plus eight float32 eps for encoding arithmetic.
        # Convert that interval through the declared monotonic asinh inverse;
        # no fixed linear-error tolerance is selected from an observed error.
        linear64 = pre_linear.transpose(2, 0, 1).astype(np.float64)
        knee, ref = float(cu.HDR_ASINH_KNEE), float(cu.HDR_ASINH_REF)
        denominator = np.arcsinh(ref / knee)
        encoded64 = np.arcsinh(linear64 / knee) / denominator
        exponents = np.frexp(np.abs(encoded64))[1]
        half_ulp = np.maximum(np.ldexp(np.ones_like(encoded64), exponents - 9), 2.0 ** -134)
        half_ulp = np.where(encoded64 == 0, 2.0 ** -134, half_ulp)
        encoding_roundoff = 8 * np.finfo(np.float32).eps * np.maximum(1, np.abs(encoded64))
        encoded_bound = half_ulp + encoding_roundoff
        actual_encoded = model_rgb.float().numpy()[:, 16:48, 8:56]
        self.assertTrue(np.all(np.abs(actual_encoded - encoded64) <= encoded_bound))
        decoded_model = cu.asinh_to_linear(model_rgb.float())
        assert isinstance(decoded_model, torch.Tensor)
        actual_linear = decoded_model.numpy()[:, 16:48, 8:56]
        lower = np.sinh((encoded64 - encoded_bound) * denominator) * knee
        upper = np.sinh((encoded64 + encoded_bound) * denominator) * knee
        inverse_roundoff = 32 * np.finfo(np.float32).eps * np.maximum(1, np.abs(linear64))
        self.assertTrue(np.all(actual_linear >= lower - inverse_roundoff))
        self.assertTrue(np.all(actual_linear <= upper + inverse_roundoff))

        # A separate true ramp checks the pre-bfloat16 resampler. Derive the
        # normalized Lanczos4 first moment in double precision, independently
        # of OpenCV's float coefficient table; no bfloat16 error enters here.
        ramp = straight.copy()
        ramp[..., 0] = xx.astype(np.float32) / 96 * 3 - .5
        ramp_path = self.root / 'rgba-ramp.exr'
        ramp_image = np.concatenate((ramp[..., ::-1] * alpha[..., None], alpha[..., None]), axis=-1)
        self.assertTrue(cv2.imwrite(str(ramp_path), ramp_image, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]))
        _, ramp_linear = captured_preprocess(str(ramp_path))
        coordinates = (np.arange(48, dtype=np.float64) + .5) * (97 / 48) - .5
        samples = np.floor(coordinates)[:, None] + np.arange(-3, 5)
        distances = coordinates[:, None] - samples
        weights = np.sinc(distances) * np.sinc(distances / 4)
        weights /= weights.sum(axis=1, keepdims=True)
        expected_red = (weights * samples).sum(axis=1) / 96 * 3 - .5
        np.testing.assert_allclose(ramp_linear[8:24, 8:40, 0],
                                   np.broadcast_to(expected_red[8:40], (16, 32)), rtol=0, atol=resample_bound)
        np.testing.assert_allclose(ramp_linear[8:24, 8:40, 1], 2, rtol=0, atol=resample_bound)
        np.testing.assert_allclose(ramp_linear[8:24, 8:40, 2], -.5, rtol=0, atol=resample_bound)
        _, diagnostics, _ = self.run_job(source, outputs={'alpha', 'fg', 'cutout'},
                                        overrides={'frame_dir_alpha': 'premultiplied',
                                                   'source_colour_profile': 'linear_bt709_d65',
                                                   'cutout_alpha_mode': 'premultiplied', 'cutout_linear': True})
        path = next(Path(diagnostics.outputs['cutout']).glob('*.exr'))
        cutout = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        np.testing.assert_allclose(cutout[..., :3], np.broadcast_to([.2, .8, -.4], cutout[..., :3].shape), atol=2e-6)
        np.testing.assert_allclose(cutout[..., 3], .4, atol=2e-6)
        self.assertEqual({path: path.read_bytes() for path in source.iterdir()}, before)

    def test_exact_scaled_scratch_budget_counts_native_quick_anchors(self):
        from corridorkey import runtime

        source = self.source()
        before = {path: path.read_bytes() for path in source.iterdir()}
        _, baseline, _ = self.run_job(source, low=True, quick=True, use_rvm=True, use_cradio=True)
        assert baseline.execution is not None
        planned = baseline.execution['temp_maps']
        # Five reduced 64x64 model frames: RVM10 + MoGe20 + C-RADIO20 + baton160;
        # three Quick Key alpha anchors stay native65x97 float32, not32x48.
        required = 10 + 20 + 20 + 160 + 3 * 65 * 97 * 4
        self.assertEqual(planned['required_bytes'], required)
        _, admitted, _ = self.run_job(source, low=True, quick=True, use_rvm=True, use_cradio=True,
                                      overrides={'temp_map_budget_bytes': required})
        assert admitted.execution is not None
        self.assertEqual(admitted.execution['temp_maps']['created_bytes'], required)
        self.assertEqual(admitted.execution['temp_maps']['reserved_bytes'], 0)
        self.assertEqual(admitted.execution['temp_maps']['cleanup'], 'complete')
        for role in baseline.outputs:
            left = sorted(Path(baseline.outputs[role]).glob('*.exr'))
            right = sorted(Path(admitted.outputs[role]).glob('*.exr'))
            self.assertEqual([path.read_bytes() for path in left], [path.read_bytes() for path in right])
        with (patch.object(runtime, 'ExrSequenceWriter') as writer,
              patch.object(runtime.np, 'memmap') as mapping,
              self.assertRaisesRegex(ValueError, 'Managed inference scratch requires')):
            self.run_job(source, low=True, quick=True, use_rvm=True, use_cradio=True,
                         overrides={'temp_map_budget_bytes': required - 1})
        writer.assert_not_called()
        mapping.assert_not_called()
        self.assertEqual(list((self.root / 'scratch').iterdir()), [])
        self.assertEqual({path: path.read_bytes() for path in source.iterdir()}, before)

    def test_native_branch_identity_and_padding_no_saving(self):
        import numpy as np
        import torch

        from corridorkey import runtime
        from corridorkey.spatial_draft import describe_spatial_draft
        value = torch.tensor([[[[-0.0, 2.0], [-1, .5]]]])
        meta = {'model_native_hw': (2, 2), 'output_hw': (2, 2)}
        self.assertIs(runtime._resize_output_torch(value, meta), value)
        self.assertTrue(np.signbit(value.numpy()[0, 0, 0, 0]))
        self.assertFalse(describe_spatial_draft({'model_scale': 1}, shape_quantum=64))
        described = describe_spatial_draft({'model_scale': .9, 'native_hw': (65, 97), 'model_native_hw': (58, 87),
                                           'shape_hw': (128, 128), 'output_hw': (65, 97)}, shape_quantum=128)
        assert isinstance(described, dict)
        self.assertFalse(described['model_grid_reduced'])

    def test_out_of_range_coverage_same_policy_renderer_override_and_quick(self):
        import numpy as np
        import torch

        from corridorkey import runtime
        meta = {'model_native_hw': (1, 2), 'output_hw': (1, 4), 'pad': (0, 0, 0, 0),
                'spatial_draft': {'enabled': True}}
        logits = torch.zeros(1, 4, 1, 2)
        logits[0, 0, 0] = torch.tensor([-1.0, 2.0])
        native = runtime._resize_output_torch(runtime._crop_native_torch(logits[:, :1], meta), meta).clamp(0, 1)
        np.testing.assert_array_equal(native.numpy()[0, 0, 0], [0, 0, 1, 1])
        checker = torch.zeros(3, 1, 4)
        normal = runtime._render_predictions_exr_torch(meta, {'alpha'}, checker, logits)['alpha']
        overridden = runtime._render_predictions_exr_torch(meta, {'alpha'}, checker, logits, native_alpha=native)['alpha']
        np.testing.assert_array_equal(normal, overridden)
        np.testing.assert_array_equal(normal[0, 0, :, 0], [0, 0, 1, 1])
        video = runtime._render_predictions_torch(meta, {'alpha'}, checker, logits)['alpha']
        np.testing.assert_array_equal(video[0, 0, :, 0], [0, 0, 65535, 65535])

    def test_nonfinite_resampling_refuses_before_any_proxy_for_model_and_prepass(self):
        import numpy as np

        from corridorkey import video_io
        source = self.source()
        meta = video_io.probe_frame_dir_meta(str(source), num_frames=5, shape_quantum=64, fps=24,
                                            exr_linear=False, model_scale=.5)
        for need_model in (False, True):
            with (patch.object(video_io, 'resize_hwc_lanczos', return_value=np.full((32, 48, 3), np.inf, dtype=np.float32)),
                  patch.object(video_io.cu, 'linear_to_vfm_sdr') as proxy,
                  self.assertRaisesRegex(ValueError, 'non-finite')):
                video_io.preprocess_frame_dir_path(meta['frame_paths'][0], meta, need_model=need_model)
            proxy.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            video_io._spatial_model_views(np.zeros((32, 48, 3), dtype=np.float32),
                                          np.full((32, 48, 3), np.inf, dtype=np.float32), meta, need_model=False)

    def test_video_renderer_native_linear_rgb_and_alpha_shapes(self):
        import numpy as np
        import torch

        from corridorkey import color_utils as cu
        from corridorkey import runtime
        meta = {'model_native_hw': (3, 5), 'output_hw': (7, 11), 'pad': (0, 0, 0, 0),
                'spatial_draft': {'enabled': True}, 'cutout_linear': True, 'despill_space': 'linear',
                'despill_strength': .5, 'despill_edges_only': True}
        logits = torch.zeros(2, 4, 3, 5)
        logits[:, 0] = .4
        logits[:, 1:] = cu.linear_to_asinh(torch.full((2, 3, 3, 5), .25))
        results = runtime._render_predictions_torch(meta, {'alpha', 'fg', 'checker', 'cutout'}, torch.zeros(3, 7, 11), logits)
        for role, result in results.items():
            self.assertEqual(result.shape, (2, 7, 11, 4 if role == 'cutout' else 3))
            self.assertEqual(result.dtype, np.uint16)

    def test_video_input_uses_reduced_grid_and_native_export(self):
        import cv2
        import numpy as np
        source = self.root / 'source.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*'MJPG'), 24, (96, 64))
        self.assertTrue(writer.isOpened())
        for frame in range(5):
            writer.write(np.full((64, 96, 3), 30 + frame * 20, dtype=np.uint8))
        writer.release()
        before = source.read_bytes()
        model, diagnostics, _ = self.run_job(source)
        self.assertEqual(set(model.stage2.shapes), {(64, 64)})
        assert diagnostics.source is not None
        self.assertEqual(diagnostics.source['output_hw'], [64, 96])
        self.assertEqual(source.read_bytes(), before)

    def test_guidance_lifetime_and_cancel_close(self):
        from corridorkey.control import CancelledError
        from corridorkey.spatial_draft import NativeGuidance
        readers = []
        class Reader:
            next_frame = frames_decoded = 0
            closed = False
            def __init__(self):
                readers.append(self)
            def read(self, frame):
                self.frames_decoded += 1
                self.next_frame = frame + 1
                return SimpleNamespace(rgb=frame)
            def close(self):
                self.closed = True
        guidance = NativeGuidance(Reader, check=lambda: None)
        self.assertEqual([guidance.read(frame) for frame in (0, 4, 1)], [0, 4, 1])
        self.assertTrue(readers[0].closed)
        self.assertEqual(guidance.stats['reader_restarts'], 1)
        guidance.close()
        self.assertTrue(all(reader.closed for reader in readers))
        def cancel():
            raise CancelledError('stop')
        cancelled = NativeGuidance(Reader, check=cancel)
        with self.assertRaises(CancelledError):
            cancelled.read(0)
        cancelled.close()
        self.assertIsNone(cancelled.reader)


if __name__ == '__main__':
    unittest.main()
