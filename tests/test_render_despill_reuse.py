"""Same-domain render reuse and guards; CPU fixtures, not GPU performance proof."""
import gc
import unittest
import weakref
from unittest.mock import Mock, patch

import torch

from corridorkey import runtime
from corridorkey.draft import DraftFrame
from tests import test_composite_region_rendering as composite


class DespillReuseTests(unittest.TestCase):
    fixture = composite.CompositeRenderingTests.fixture

    def render(self, *, exr=False, draft=False, outputs=None, options=None, region=None):
        rgb, alpha, logits, meta, checker = self.fixture()
        meta.update(options or {})
        outputs = outputs if outputs is not None else {'cutout', 'checker'}
        if draft:
            frame = DraftFrame(0, rgb[0].permute(1, 2, 0).numpy(), alpha[0, 0].numpy(), {})
            return runtime._render_source_draft(frame, meta, outputs, checker, use_exr=exr,
                composite_region=None if region is None else region[:1])
        renderer = runtime._render_predictions_exr_torch if exr else runtime._render_predictions_torch
        return renderer(meta, outputs, checker, logits, composite_region=region)

    def test_neural_exr_both_reuses_one_candidate(self):
        with patch.object(runtime, '_despill_configured', wraps=runtime._despill_configured) as despill:
            self.render(exr=True)
        self.assertEqual(despill.call_count, 1)

    def test_video_computes_only_used_domains_once(self):
        cases = [({'cutout', 'checker'}, {}, 1), ({'cutout'}, {}, 1),
                 ({'checker'}, {}, 1), ({'cutout', 'checker'}, {'despill_space': 'linear'}, 1),
                 ({'cutout', 'checker'}, {'cutout_linear': True}, 2),
                 ({'cutout', 'checker'}, {'cutout_linear': True, 'despill_space': 'linear'}, 1),
                 ({'fg', 'alpha'}, {}, 0)]
        for draft in (False, True):
            for outputs, options, expected in cases:
                with self.subTest(draft=draft, outputs=outputs, options=options):
                    with patch.object(runtime, '_despill_configured', wraps=runtime._despill_configured) as despill:
                        self.render(draft=draft, outputs=outputs, options=options)
                    self.assertEqual(despill.call_count, expected)
                    if expected == 2:
                        # Mixed cutout-linear / legacy checker must not share different inputs.
                        self.assertFalse(torch.equal(despill.call_args_list[0].args[0], despill.call_args_list[1].args[0]))

    def test_combined_outputs_equal_individual_outputs_exactly_all_flags(self):
        _, alpha, _, _, _ = self.fixture()
        weight = torch.linspace(0, 1, 8).reshape(1, 1, 2, 4).expand_as(alpha)
        for exr in (False, True):
            for draft in (False, True):
                for settings in ({}, {'despill_space': 'linear', 'despill_preserve_luma': True},
                                 {'cutout_linear': True}, {'despill_screen': 'blue', 'despill_edges_only': True},
                                 {'despill_strength': 0}, {'cutout_alpha_mode': 'premultiplied'}):
                    for region in (None, torch.zeros_like(alpha), torch.ones_like(alpha), weight):
                        with self.subTest(exr=exr, draft=draft, settings=settings, region=region is not None):
                            both = self.render(exr=exr, draft=draft, options=settings, region=region)
                            for output in ('cutout', 'checker'):
                                single = self.render(exr=exr, draft=draft, outputs={output}, options=settings, region=region)
                                self.assertEqual(both[output].dtype, single[output].dtype)
                                self.assertEqual(both[output].tobytes(), single[output].tobytes())

    def test_strength_zero_skips_region_get_allocation_and_transfer(self):
        _, _, _, meta, _ = self.fixture()
        rasters = Mock()
        rasters.composite_region.side_effect = AssertionError('unneeded mask retrieval')
        with (patch.object(runtime.np, 'empty', side_effect=AssertionError('unneeded allocation')),
              patch.object(runtime.torch, 'from_numpy', side_effect=AssertionError('unneeded transfer'))):
            self.assertIsNone(runtime._composite_region_batch(rasters, [0, 1], dict(meta, despill_strength=0), {'cutout', 'checker'}, 'cpu'))
        rasters.composite_region.assert_not_called()

    def test_strength_zero_direct_renderer_still_validates_mask_and_metadata(self):
        _, alpha, _, _, _ = self.fixture()
        for exr in (False, True):
            for draft in (False, True):
                with self.subTest(exr=exr, draft=draft):
                    with self.assertRaisesRegex(ValueError, 'non-finite'):
                        self.render(exr=exr, draft=draft, options={'despill_strength': 0}, region=torch.full_like(alpha, float('nan')))
                    for outputs in ({'alpha'}, {'fg'}, set()):
                        with self.assertRaises((TypeError, ValueError)):
                            self.render(exr=exr, draft=draft, outputs=outputs, options={'despill_strength': 'invalid'})

    def test_candidates_remain_batch_local_and_inputs_are_not_mutated(self):
        rgb, alpha, logits, meta, checker = self.fixture()
        original = runtime._despill_configured
        refs, byte_counts = [], []
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            refs.append(weakref.ref(result))
            byte_counts.append(result.nelement() * result.element_size())
            return result
        snapshots = [x.numpy().tobytes() for x in (rgb, alpha, logits, checker)]
        with patch.object(runtime, '_despill_configured', capture):
            runtime._render_predictions_exr_torch(meta, {'checker'}, checker, logits)
            runtime._render_predictions_torch(meta, {'checker'}, checker, logits)
        gc.collect()
        self.assertEqual(len(refs), 2)
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(byte_counts, [2 * 3 * 2 * 4 * 4] * 2)
        self.assertEqual(snapshots, [x.numpy().tobytes() for x in (rgb, alpha, logits, checker)])

    def test_unused_linear_overflow_does_not_reject_valid_legacy_display(self):
        linear = torch.tensor([2e36, 1e36, 2e36]).reshape(1, 3, 1, 1)
        logits = torch.cat((torch.full((1, 1, 1, 1), .5), runtime.cu.linear_to_asinh(linear)), dim=1)
        meta = {'model_native_hw': (1, 1), 'output_hw': (1, 1), 'pad': (0, 0, 0, 0),
                'linear_brightness': .01, 'despill_strength': .7}
        decoded = runtime._linear_adjust_inverse_torch(runtime.cu.asinh_to_linear(logits[:, 1:]), meta)
        self.assertTrue(bool(torch.isfinite(decoded).all()))
        self.assertFalse(bool(torch.isfinite(decoded[:, 0] + decoded[:, 2]).all()))
        display = runtime._render_predictions_torch(meta, {'cutout'}, None, logits)['cutout']
        # Chosen display input is clipped white, so independent excess-green oracle is zero spill.
        self.assertEqual(display[0, 0, 0].tolist(), [65535, 65535, 65535, 32767])
        with self.assertRaisesRegex(ValueError, 'despill channel sum'):
            runtime._render_predictions_torch(dict(meta, despill_space='linear'), {'cutout'}, None, logits)
        with self.assertRaisesRegex(ValueError, 'despill channel sum'):
            runtime._render_predictions_torch(dict(meta, cutout_linear=True), {'cutout'}, None, logits)

    def test_strength_zero_corrupt_job_asset_still_rejected_before_writer(self):
        fixture = composite.CompositeRuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        path, _, document = fixture.correction()
        asset = document.asset_paths()[0]
        asset.write_bytes(asset.read_bytes() + b'corrupt')
        with patch.object(runtime, 'ExrSequenceWriter') as writer, self.assertRaises(ValueError):
            fixture.fixture.run_job(fixture.source, correction=path, outputs={'cutout'},
                overrides={'start_frame': 1, 'num_frames': 3, 'despill_strength': 0})
        writer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
