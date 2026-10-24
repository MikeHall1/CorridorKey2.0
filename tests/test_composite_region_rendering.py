"""CPU colour/export and orchestration fixtures; not neural/GPU quality evidence."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock, call, patch

import cv2
import numpy as np
import torch

from corridorkey import color_utils as cu
from corridorkey import runtime
from corridorkey.draft import DraftFrame


class CompositeRenderingTests(unittest.TestCase):
    def fixture(self):
        rgb = torch.tensor([-.25, 3., .125]).reshape(1, 3, 1, 1).expand(2, 3, 2, 4).clone()
        alpha = torch.tensor([0., 1e-9, .5, 1., .2, .7, .9, .1]).reshape(1, 1, 2, 4).expand(2, 1, 2, 4)
        logits = torch.cat((alpha, cu.linear_to_asinh(rgb)), dim=1)
        meta = {'model_native_hw': (2, 4), 'output_hw': (2, 4), 'pad': (0, 0, 0, 0),
                'despill_strength': .7}
        checker = torch.full((3, 2, 4), .3)
        return rgb, alpha, logits, meta, checker

    def test_working_domain_blend_exact_endpoints_edge_luma_and_signed_zero(self):
        rgb, alpha, _, meta, _ = self.fixture()
        rgb[0, 0, 0, 0] = -0.
        region = torch.tensor([0., 1., .25, .75, 1., 0., .5, .3]).reshape(1, 1, 2, 4).expand(2, 1, 2, 4)
        for screen in ('green', 'blue'):
            for edge in (False, True):
                for luma in (False, True):
                    options = dict(meta, despill_screen=screen, despill_edges_only=edge, despill_preserve_luma=luma)
                    full = runtime._despill_configured(rgb, alpha, options)
                    result = runtime._despill_configured(rgb, alpha, options, composite_region=region)
                    expected = rgb * (1 - region) + full * region
                    zero = (region == 0).expand_as(rgb)
                    one = (region == 1).expand_as(rgb)
                    np.testing.assert_array_equal(result[zero].numpy().view(np.uint32), rgb[zero].numpy().view(np.uint32))
                    torch.testing.assert_close(result[one], full[one], rtol=0, atol=0)
                    fraction = ((region > 0) & (region < 1)).expand_as(rgb)
                    torch.testing.assert_close(result[fraction], expected[fraction], rtol=0, atol=0)

    def test_disabled_global_despill_is_bit_exact_with_fractional_region(self):
        rgb = torch.rand((2, 3, 7, 9), generator=torch.Generator().manual_seed(317))
        alpha = torch.full((2, 1, 7, 9), .5)
        region = torch.rand(alpha.shape, generator=torch.Generator().manual_seed(101))
        actual = runtime._despill_configured(rgb, alpha, {'despill_strength': 0}, composite_region=region)
        np.testing.assert_array_equal(actual.numpy().view(np.uint32), rgb.numpy().view(np.uint32))

    def test_all_render_paths_zero_full_and_alpha_fg_identity(self):
        rgb, alpha, logits, meta, checker = self.fixture()
        outputs = {'alpha', 'fg', 'cutout', 'checker'}
        for exr in (False, True):
            for linear in (False, True):
                for association in ('straight', 'premultiplied') if exr else ('straight',):
                    options = dict(meta, despill_space='linear' if linear else 'legacy', cutout_alpha_mode=association)
                    render = runtime._render_predictions_exr_torch if exr else runtime._render_predictions_torch
                    for draft in (False, True):
                        with self.subTest(exr=exr, linear=linear, association=association, draft=draft):
                            def run(settings, region=None, *, draft=draft, exr=exr, render=render):
                                if draft:
                                    frame = DraftFrame(0, rgb[0].permute(1, 2, 0).numpy(), alpha[0, 0].numpy(), {})
                                    return runtime._render_source_draft(frame, settings, outputs, checker, use_exr=exr,
                                                                        composite_region=None if region is None else region[:1])
                                return render(settings, outputs, checker, logits, composite_region=region)
                            full, off = run(options), run(dict(options, despill_strength=0))
                            zero = run(options, torch.zeros_like(alpha))
                            one = run(options, torch.ones_like(alpha))
                            for name in outputs:
                                np.testing.assert_array_equal(one[name], full[name])
                                np.testing.assert_array_equal(zero[name], off[name])
                                if name in {'alpha', 'fg'}:
                                    np.testing.assert_array_equal(zero[name], full[name])
                            if exr:
                                self.assertTrue(np.any(off['fg'] < 0))
                                self.assertTrue(np.any(off['fg'] > 1))

    def test_fractional_render_matches_reference_in_selected_domain_before_association(self):
        rgb, alpha, logits, meta, checker = self.fixture()
        weight = torch.tensor([0., .25, 1., .7, 0., 1., .5, .3]).reshape(1, 1, 2, 4).expand(2, 1, 2, 4)
        for exr in (False, True):
            for domain in ('legacy', 'linear', 'cutout_linear'):
                for draft in (False, True):
                    for edge, luma in ((False, False), (True, True)):
                        settings = dict(meta, despill_space='linear' if domain == 'linear' else 'legacy',
                                        cutout_linear=domain == 'cutout_linear', despill_edges_only=edge,
                                        despill_preserve_luma=luma, cutout_alpha_mode='premultiplied')
                        source = rgb if draft else cu.asinh_to_linear(logits[:, 1:4])
                        a, w = (alpha[:1], weight[:1]) if draft else (alpha, weight)
                        source = source[:1] if draft else source
                        display = cu.linear_to_srgb(source.clamp(min=0))
                        assert isinstance(display, torch.Tensor)
                        display = display.clamp(0, 1)
                        def blend(value, *, a=a, w=w, edge=edge, luma=luma):
                            candidate = runtime._despill_average_srgb_torch(value, .7, alpha=a,
                                edges_only=edge, preserve_luma=luma)
                            return torch.where(w == 0, value, torch.where(w == 1, candidate,
                                               value * (1 - w) + candidate * w))
                        working = source if exr else source.clamp(min=0)
                        if draft:
                            frame = DraftFrame(0, rgb[0].permute(1, 2, 0).numpy(), alpha[0, 0].numpy(), {})
                            actual = runtime._render_source_draft(frame, settings, {'cutout', 'checker'}, checker,
                                                                  use_exr=exr, composite_region=w)
                        else:
                            renderer = runtime._render_predictions_exr_torch if exr else runtime._render_predictions_torch
                            actual = renderer(settings, {'cutout', 'checker'}, checker, logits, composite_region=w)
                        cutout = blend(working if exr or domain != 'legacy' else display)
                        if exr:
                            expected_cutout = runtime._chw_to_bgra_f32_cpu(cutout * a, a)
                        else:
                            if domain == 'linear':
                                cutout = cu.linear_to_srgb(cutout.clamp(min=0))
                                assert isinstance(cutout, torch.Tensor)
                                cutout = cutout.clamp(0, 1)
                            expected_cutout = runtime._srgb_to_bgra_uint16_cpu(cutout, a)
                        foreground = blend(working) if exr or domain == 'linear' else cu.srgb_to_linear(blend(display))
                        comp = foreground * a + cu.srgb_to_linear(checker.unsqueeze(0)) * (1 - a)
                        if exr:
                            expected_checker = runtime._chw_to_bgr_f32_cpu(comp)
                        else:
                            checker_display = cu.linear_to_srgb(comp.clamp(min=0))
                            assert isinstance(checker_display, torch.Tensor)
                            expected_checker = runtime._srgb_to_bgr_uint16_cpu(checker_display.clamp(0, 1))
                        np.testing.assert_array_equal(actual['cutout'], expected_cutout)
                        np.testing.assert_array_equal(actual['checker'], expected_checker)

    def test_renderer_rejects_non_native_nonfinite_range_dtype_and_device(self):
        _, alpha, logits, meta, checker = self.fixture()
        bad = [torch.ones(2, 1, 1, 4), torch.ones_like(alpha, dtype=torch.float64),
               torch.full_like(alpha, float('nan')), torch.full_like(alpha, -0.1),
               torch.full_like(alpha, 1.1), torch.empty(alpha.shape, device='meta')]
        for renderer in (runtime._render_predictions_torch, runtime._render_predictions_exr_torch):
            for region in bad:
                with self.subTest(renderer=renderer.__name__, region=region.shape), self.assertRaises(ValueError):
                    renderer(meta, {'cutout', 'checker'}, checker, logits, composite_region=region)

    def test_mask_free_outputs_ignore_region_and_batch_without_active_does_not_allocate(self):
        _, alpha, logits, meta, checker = self.fixture()
        for renderer in (runtime._render_predictions_torch, runtime._render_predictions_exr_torch):
            with patch.object(runtime, '_native_composite_region', side_effect=AssertionError('unneeded mask')):
                renderer(meta, {'alpha', 'fg'}, checker, logits, composite_region=torch.full_like(alpha, float('nan')))
        with patch.object(runtime.np, 'empty', side_effect=AssertionError('unneeded allocation')):
            self.assertIsNone(runtime._composite_region_batch(None, [0, 1], meta, {'cutout'}, 'cpu'))
            rasters = Mock()
            rasters.composite_region.return_value = None
            self.assertIsNone(runtime._composite_region_batch(rasters, [0, 1], meta, {'cutout'}, 'cpu'))
            rasters.composite_region.assert_has_calls([call(0), call(1)])
            rasters.reset_mock()
            self.assertIsNone(runtime._composite_region_batch(rasters, [0, 1], meta, {'alpha', 'fg'}, 'cpu'))
            rasters.composite_region.assert_not_called()

    def test_batch_local_frame_mapping_mixed_none_zero_fraction_and_single_transfer(self):
        _, _, _, meta, _ = self.fixture()
        rasters = Mock()
        rasters.composite_region.side_effect = [None, np.zeros((2, 4), np.float32), np.full((2, 4), .3, np.float32)]
        original = torch.Tensor.to
        transfers = []
        def transfer(tensor, *args, **kwargs):
            transfers.append(tuple(tensor.shape))
            return original(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, 'to', transfer):
            result = runtime._composite_region_batch(rasters, [0, 2, 5], dict(meta, start_frame=100), {'cutout', 'checker'}, 'cpu')
        self.assertEqual(transfers, [(3, 1, 2, 4)])
        self.assertEqual(rasters.composite_region.call_args_list, [call(i) for i in (0, 2, 5)])
        assert result is not None
        np.testing.assert_array_equal(result[:, 0].numpy(), np.stack([np.ones((2, 4)), np.zeros((2, 4)), np.full((2, 4), np.float32(.3))]))


class CompositeRuntimeTests(unittest.TestCase):
    """Actual shared orchestration, decode and EXR writes; fixture models only."""
    def setUp(self):
        from tests.test_spatial_draft import SpatialDraftTests
        self.fixture = SpatialDraftTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.source = self.fixture.source(exr=True, frames=5)
        for index, path in enumerate(sorted(self.source.iterdir())):
            rgb = np.full((65, 97, 3), [0.1, 2 + index * .1, -.2], np.float32)
            self.assertTrue(cv2.imwrite(str(path), rgb[..., ::-1]))

    def correction(self, *, strength=1.):
        from corridorkey.corrections import CorrectionDocument, save_corrections
        from corridorkey.identities import source_identity
        mask = np.zeros((65, 97), np.float32)
        mask[:, 32:64] = .25
        mask[:, 64:] = 1
        asset = self.root / 'region.exr'
        self.assertTrue(cv2.imwrite(str(asset), mask))
        data = {'schema_version': 5, 'revision': 1,
            'source': {'sha256': source_identity(self.source)['sha256'], 'start_frame': 1,
                       'frame_count': 3, 'fps': 24., 'width': 97, 'height': 65},
            'shots': [[0, 3]], 'tracks': [], 'guidance': [],
            'entries': [{'id': 'despill', 'revision': 1, 'frame': 1, 'kind': 'region',
                         'purpose': 'composite', 'target': 'despill', 'direction': 'both',
                         'strength': strength, 'range': [1, 3],
                         'asset': {'path': asset.name, 'sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}}]}
        document = CorrectionDocument.parse(data, self.root)
        path = self.root / 'corrections.json'
        save_corrections(path, document)
        return path, mask, document

    @staticmethod
    def arrays(diagnostics, name):
        return [cv2.imread(str(path), -1) for path in sorted(Path(diagnostics.outputs[name]).glob('*.exr'))]

    def test_real_exports_all_memory_spatial_and_quick_paths_nonzero_offset(self):
        for low, scale, quick in ((False, 1., False), (True, 1., False),
                                  (False, .5, False), (True, .5, False),
                                  (False, 1., True), (True, 1., True), (False, .5, True)):
            for dtype in ('float', 'half'):
                with self.subTest(low=low, scale=scale, quick=quick, dtype=dtype):
                    path, mask, document = self.correction()
                    before = {p: p.read_bytes() for p in self.source.iterdir()}
                    options = {'start_frame': 1, 'num_frames': 3, 'despill_strength': .7,
                               'exr_dtype': dtype, 'cutout_alpha_mode': 'premultiplied'}
                    baseline_model, baseline, _ = self.fixture.run_job(
                        self.source, low=low, scale=scale, quick=quick, outputs={'alpha', 'fg', 'cutout', 'checker'}, overrides=options)
                    model, regional, _ = self.fixture.run_job(
                        self.source, low=low, scale=scale, quick=quick, correction=path,
                        outputs={'alpha', 'fg', 'cutout', 'checker'}, overrides=options)
                    self.assertEqual(baseline.execution, regional.execution)
                    self.assertEqual(baseline_model.stage1.shapes, model.stage1.shapes)
                    for left, right in zip(baseline_model.stage1.seen_hints, model.stage1.seen_hints, strict=True):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    assert regional.corrections is not None
                    self.assertEqual(regional.corrections['identity'], document.identity)
                    for name in ('alpha', 'fg'):
                        for left, right in zip(self.arrays(baseline, name), self.arrays(regional, name), strict=True):
                            np.testing.assert_array_equal(left, right)
                    # Frame zero has no active region, despite source start=1; it remains whole-frame.
                    for name in ('cutout', 'checker'):
                        left, right = self.arrays(baseline, name), self.arrays(regional, name)
                        np.testing.assert_array_equal(left[0], right[0])
                        for frame in (1, 2):
                            np.testing.assert_array_equal(left[frame][:, 64:], right[frame][:, 64:])
                            self.assertTrue(np.any(left[frame][:, :32, :3] != right[frame][:, :32, :3]))
                    self.assertEqual(before, {p: p.read_bytes() for p in self.source.iterdir()})
                    self.assertTrue(np.any(mask == .25))

    def test_disabled_default_and_raw_outputs_do_not_request_regions(self):
        from corridorkey.correction_rasters import CorrectionRasters
        path, _, _ = self.correction(strength=0)
        _, baseline, _ = self.fixture.run_job(self.source, scale=1, outputs={'alpha', 'fg', 'cutout'},
            overrides={'start_frame': 1, 'num_frames': 3, 'despill_strength': .7})
        _, disabled, _ = self.fixture.run_job(self.source, scale=1, correction=path, outputs={'alpha', 'fg', 'cutout'},
            overrides={'start_frame': 1, 'num_frames': 3, 'despill_strength': .7})
        for name in baseline.outputs:
            for left, right in zip(self.arrays(baseline, name), self.arrays(disabled, name), strict=True):
                np.testing.assert_array_equal(left, right)
        path, _, _ = self.correction()
        with patch.object(CorrectionRasters, 'composite_region', side_effect=AssertionError('unneeded mask work')):
            self.fixture.run_job(self.source, correction=path, quick=True, outputs={'alpha', 'fg'},
                overrides={'start_frame': 1, 'num_frames': 3, 'despill_strength': .7})

    def test_bad_mask_and_cancel_reject_before_first_submit(self):
        from corridorkey.control import CancelledError
        from corridorkey.correction_rasters import CorrectionRasters
        path, _, _ = self.correction()
        for value in (np.zeros((64, 97), np.float32), np.full((65, 97), np.nan, np.float32),
                      np.full((65, 97), 1.1, np.float32), np.zeros((65, 97), np.float64)):
            with patch.object(CorrectionRasters, 'composite_region', return_value=value), self.assertRaisesRegex(ValueError, 'Composite region'):
                self.fixture.run_job(self.source, correction=path, outputs={'cutout'},
                    overrides={'start_frame': 1, 'num_frames': 3, 'despill_strength': .7})
        with patch.object(CorrectionRasters, 'composite_region', side_effect=CancelledError('cancel mask')), self.assertRaises(CancelledError):
            self.fixture.run_job(self.source, correction=path, outputs={'checker'},
                overrides={'start_frame': 1, 'num_frames': 3, 'despill_strength': .7})
        self.assertEqual(list((self.root / 'scratch').iterdir()), [])

    def test_actual_tracked_regions_move_through_global_export_and_stale_binding_fails(self):
        from corridorkey.corrections import (
            CorrectionDocument,
            canonical,
            save_corrections,
            track_binding,
        )
        from corridorkey.identities import source_identity
        from tests.test_tracked_refinement_regions import RegionTrackingCoreTests
        fixture = RegionTrackingCoreTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        data, _ = fixture.build('both', both_anchors=True, overlap=True)
        source = fixture.root / 'frames'
        source.mkdir()
        for index in range(106):
            self.assertTrue(cv2.imwrite(str(source / f'{index:04}.png'), np.full((16, 20, 3), [30, 230, 10], np.uint8)))
        data['source']['sha256'] = source_identity(source)['sha256']
        for entry in data['entries']:
            entry.update(purpose='composite', target='despill')
        track = data['tracks'][0]
        old = track['binding']
        track['binding'] = track_binding(data, track['anchor_ids'], cuts=old['cuts'], decode=old['decode'], tracking=old['tracking'])
        track['binding_sha256'] = hashlib.sha256(canonical(track['binding']).encode()).hexdigest()
        document = CorrectionDocument.parse(data, fixture.root)
        correction = fixture.root / 'composite.json'
        save_corrections(correction, document)
        _, baseline, _ = self.fixture.run_job(source, scale=1, outputs={'fg', 'alpha', 'cutout'},
            overrides={'start_frame': 100, 'num_frames': 6, 'despill_strength': .7})
        _, regional, _ = self.fixture.run_job(source, scale=1, correction=correction, outputs={'fg', 'alpha', 'cutout'},
            overrides={'start_frame': 100, 'num_frames': 6, 'despill_strength': .7})
        from corridorkey.correction_rasters import CorrectionRasters
        with_rasters = CorrectionRasters(document, fixture.meta)
        self.addCleanup(with_rasters.close)
        fg = self.arrays(regional, 'fg')
        actual = self.arrays(regional, 'cutout')
        for frame in range(6):
            weight = with_rasters.composite_region(frame)
            assert weight is not None
            alpha = torch.from_numpy(actual[frame][..., 3]).reshape(1, 1, 16, 20)
            rgb = torch.from_numpy(fg[frame][..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0)
            ordinary = runtime._despill_configured(rgb, alpha, {'despill_strength': .7})
            expected = rgb * (1 - torch.from_numpy(weight)) + ordinary * torch.from_numpy(weight)
            np.testing.assert_array_equal(actual[frame][..., :3], runtime._chw_to_bgr_f32_cpu(expected)[0])
            for name in ('alpha', 'fg'):
                np.testing.assert_array_equal(self.arrays(baseline, name)[frame], self.arrays(regional, name)[frame])
        stale = copy.deepcopy(data)
        stale['entries'][0]['strength'] = .9
        with self.assertRaises(ValueError):
            CorrectionDocument.parse(stale, fixture.root)

    def test_global_video_writer_plumbing_reuses_region_for_both_outputs(self):
        class Writer:
            batches: ClassVar[dict] = {}
            def __init__(self, path, *args, **kwargs):
                self.path = str(path)
                self.batches[self.path] = []
            def write_many(self, batch):
                self.batches[self.path].append(batch.copy())
            def release(self):
                pass
        path, _, _ = self.correction()
        for quick in (False, True):
            for space in ('legacy', 'linear'):
                common = {'start_frame': 1, 'num_frames': 3, 'despill_strength': .7,
                          'output_format': 'video', 'despill_space': space}
                _, baseline, _ = self.fixture.run_job(self.source, quick=quick, outputs={'alpha', 'fg', 'cutout', 'checker'},
                    video_writer=Writer, overrides=common)
                _, regional, _ = self.fixture.run_job(self.source, quick=quick, correction=path,
                    outputs={'alpha', 'fg', 'cutout', 'checker'}, video_writer=Writer, overrides=common)
                for name in baseline.outputs:
                    left = np.concatenate(Writer.batches[str(baseline.outputs[name])])
                    right = np.concatenate(Writer.batches[str(regional.outputs[name])])
                    np.testing.assert_array_equal(left[0], right[0])
                    if name in ('alpha', 'fg'):
                        np.testing.assert_array_equal(left, right)
                    else:
                        np.testing.assert_array_equal(left[:, :, 64:], right[:, :, 64:])
                        self.assertTrue(np.any(left[1:, :, :32, :3] != right[1:, :, :32, :3]))

    def test_quick_neural_fallback_preserves_model_work_hints_and_applies_only_at_output(self):
        from corridorkey.corrections import CorrectionDocument, save_corrections
        from corridorkey.identities import source_identity
        from tests.test_quick_runtime import QuickRuntimeTests
        fixture = QuickRuntimeTests()
        left_model: Any
        model: Any
        def correction(source, root):
            asset = root / 'mask.exr'
            cv2.imwrite(str(asset), np.zeros((64, 64), np.float32))
            data = {'schema_version': 5, 'revision': 1,
                'source': {'sha256': source_identity(source)['sha256'], 'start_frame': 0,
                    'frame_count': 9, 'fps': 24., 'width': 64, 'height': 64},
                'shots': [[0, 9]], 'tracks': [], 'guidance': [],
                'entries': [{'id': 'region', 'revision': 1, 'frame': 0, 'kind': 'region', 'purpose': 'composite',
                    'target': 'despill', 'direction': 'both', 'strength': 1., 'range': [0, 9],
                    'asset': {'path': asset.name, 'sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}}]}
            path = root / 'corrections.json'
            save_corrections(path, CorrectionDocument.parse(data, root))
            return path
        original_write = cv2.imwrite
        def author_green(path, image, *args, **kwargs):
            if str(path).endswith('.png'):
                image = image.copy()
                image[..., 1] = np.minimum(image[..., 1].astype(np.uint16) + 15, 255).astype(np.uint8)
            return original_write(path, image, *args, **kwargs)
        for low in (False, True):
            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                with patch.object(cv2, 'imwrite', author_green):
                    left_model, baseline = fixture.run_job(first, low_vram=low, fallback='neural', flashing=True,
                        settings_overrides={'despill_strength': .7})
                    model, regional = fixture.run_job(second, low_vram=low, fallback='neural', flashing=True,
                        corrections=correction, settings_overrides={'despill_strength': .7})
                assert regional.quick_key is not None and baseline.quick_key is not None
                self.assertTrue(regional.quick_key['extra_neural_frames'])
                self.assertEqual(baseline.execution, regional.execution)
                self.assertEqual(baseline.quick_key['extra_neural_frames'], regional.quick_key['extra_neural_frames'])
                for left, right in zip(left_model.stage1.seen_hints, model.stage1.seen_hints, strict=True):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for name in ('alpha', 'fg'):
                    for left, right in zip(self.arrays(baseline, name), self.arrays(regional, name), strict=True):
                        np.testing.assert_array_equal(left, right)
                for fg, cutout in zip(self.arrays(regional, 'fg'), self.arrays(regional, 'cutout'), strict=True):
                    np.testing.assert_array_equal(fg, cutout[..., :3])
                self.assertTrue(any(np.any(left[..., :3] != right[..., :3]) for left, right in
                    zip(self.arrays(baseline, 'cutout'), self.arrays(regional, 'cutout'), strict=True)))


if __name__ == '__main__':
    unittest.main()
