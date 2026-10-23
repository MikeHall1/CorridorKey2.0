"""Native coverage finishing oracles; CPU substitutes are not neural quality evidence."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


class EdgeFinishingTests(unittest.TestCase):
    def test_native_morphology_and_feather_match_independent_oracle(self):
        import cv2
        import numpy as np

        from corridorkey.refinement import finish_alpha

        alpha = np.random.default_rng(17).random((277, 291), dtype=np.float32)
        for expansion in (-2, 0, 2):
            for feather in (0, 3):
                with self.subTest(expansion=expansion, feather=feather):
                    expected = alpha.copy()
                    if expansion:
                        padded = np.pad(alpha, abs(expansion), mode='edge')
                        neighbours = [padded[y:y + 277, x:x + 291]
                                      for y in range(2 * abs(expansion) + 1)
                                      for x in range(2 * abs(expansion) + 1)]
                        expected = (np.maximum.reduce(neighbours) if expansion > 0
                                    else np.minimum.reduce(neighbours))
                    if feather:
                        expected = cv2.GaussianBlur(expected, (7, 7), 1.5,
                                                    borderType=cv2.BORDER_REPLICATE)
                    actual = finish_alpha(alpha, expansion=expansion, feather=feather)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertTrue(np.all((actual >= 0) & (actual <= 1)))

    def test_disabled_validation_and_cancellation(self):
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.control import CancellationToken, CancelledError
        from corridorkey.refinement import finish_alpha

        alpha = np.array([[0, .1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(finish_alpha(alpha), alpha)
        for field in ('alpha_expand_radius', 'alpha_feather_radius'):
            for value in (True, 1.5, float('nan'), 33, -33):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    cfg = InferenceConfig()
                    setattr(cfg, field, value)
                    cfg.validate()
        for alpha in (np.array([[float('nan')]]), np.array([[-.1]]), np.array([[1.1]]), np.zeros((0, 2))):
            with self.assertRaises(ValueError):
                finish_alpha(alpha, expansion=1)
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(CancelledError):
            finish_alpha(np.ones((3, 3)), feather=1, check=token.check)
        calls = []
        def stop():
            calls.append(1)
            if len(calls) == 4:
                raise CancelledError('tile cancellation')
        with self.assertRaises(CancelledError):
            finish_alpha(np.ones((600, 600)), feather=2, check=stop)
        self.assertEqual(len(calls), 4)

    def test_legacy_worker_codec_defaults_and_new_roundtrip(self):
        import json

        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_from_wire, job_to_wire

        wire = job_to_wire(JobRequest.create('.', '.', InferenceConfig(alpha_expand_radius=-2, alpha_feather_radius=3)))
        cfg, _ = job_from_wire(wire).configurations()
        self.assertEqual((cfg.alpha_expand_radius, cfg.alpha_feather_radius), (-2, 3))
        old = json.loads(wire['config_json'])
        del old['alpha_expand_radius'], old['alpha_feather_radius']
        wire['config_json'] = json.dumps(old)
        cfg, _ = job_from_wire(wire).configurations()
        self.assertEqual((cfg.alpha_expand_radius, cfg.alpha_feather_radius), (0, 0))

    def test_cli_and_desktop_roundtrip_without_submission(self):
        import os
        from types import SimpleNamespace

        from PySide6.QtWidgets import QApplication, QLineEdit

        import infer
        from corridorkey.desktop_settings import SettingsPanel
        from corridorkey.desktop_state import pretty_settings
        from corridorkey.session import JobRequest

        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)
        session = SimpleNamespace(weights=None, last_result=None)
        session.run = lambda job: job
        job = infer.main(['--input', '.', '--output_dir', '.', '--alpha_expand_radius', '-2',
                          '--alpha_feather_radius', '3'], session=session)
        assert isinstance(job, JobRequest)
        panel = SettingsPanel()
        try:
            panel.load(*pretty_settings(job))
            expand = panel.controls['engine', 'alpha_expand_radius']
            feather = panel.controls['engine', 'alpha_feather_radius']
            assert isinstance(expand, QLineEdit) and isinstance(feather, QLineEdit)
            self.assertEqual(expand.text(), '-2')
            feather.setText('4')
            config, writer = panel.values()
            cfg, _ = JobRequest(job.input_path, job.output_dir, config, writer).configurations()
            self.assertEqual((cfg.alpha_expand_radius, cfg.alpha_feather_radius), (-2, 4))
        finally:
            panel.close()

    def test_actual_export_pixels_all_paths_no_anchor_or_hint_leak(self):
        import tempfile

        import cv2
        import numpy as np
        import torch

        from corridorkey import draft
        from corridorkey.motion import Propagation
        from corridorkey.refinement import finish_alpha
        from tests import test_spatial_draft as fixture

        with tempfile.TemporaryDirectory() as temporary:
            helper = fixture.SpatialDraftTests()
            helper.root = Path(temporary)
            source = helper.source(exr=True)
            before = {p: p.read_bytes() for p in source.iterdir()}
            factory, original_frames = fixture.make_spatial_cpu_model, draft.offline_frames

            def patterned_model(**kwargs):
                model = factory(**kwargs)
                original = model.stage2.forward
                def forward(rgb, baton):
                    logits = original(rgb, baton)
                    width = logits.shape[-1]
                    values = torch.arange(width).remainder(13).float() / 12
                    logits[:, :, 0] = values.view(1, 1, 1, width)
                    return logits
                model.stage2.forward = forward
                return model

            def uncertain(*args, **kwargs):
                alpha = np.tile(np.arange(97, dtype=np.float32) % 13 / 12, (65, 1))
                return Propagation(alpha, np.zeros_like(alpha), np.ones_like(alpha, dtype=bool),
                                   {'flow_evaluations': 4, 'motion_seconds': 0., 'uncertain_fraction': 1., 'motion_fraction': 0.})

            def forced(*args, **kwargs):
                return original_frames(*args, interpolate=uncertain, **kwargs)

            cases = ((1., False, False, 'offline'), (1., True, False, 'offline'),
                     (.5, False, False, 'offline'), (.5, True, False, 'offline'),
                     (1., False, True, 'offline'), (.5, True, True, 'offline'),
                     (.5, False, True, 'hold'), (1., True, True, 'causal'))
            with patch.object(fixture, 'make_spatial_cpu_model', patterned_model), patch.object(draft, 'offline_frames', forced):
                for scale, low, quick, approach in cases:
                    with self.subTest(scale=scale, low=low, quick=quick, approach=approach):
                        common = {'quick_approach': approach, 'cutout_alpha_mode': 'premultiplied'}
                        expansion = 1 if low else -1
                        base_model, baseline, _ = helper.run_job(source, overrides=common, scale=scale, low=low,
                                                                quick=quick, refine=True, outputs={'alpha', 'fg', 'cutout', 'checker'})
                        model, result, _ = helper.run_job(source, overrides={**common, 'alpha_expand_radius': expansion,
                                                                           'alpha_feather_radius': 2}, scale=scale, low=low,
                                                         quick=quick, refine=True, outputs={'alpha', 'fg', 'cutout', 'checker'})
                        assert result.alpha_refinement is not None
                        record = result.alpha_refinement['edge_finishing']
                        self.assertEqual(record['frames_evaluated'], 5)
                        self.assertGreater(record['frames_changed'], 0)
                        for a, b in zip(base_model.stage1.seen_hints, model.stage1.seen_hints):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        for base_path, path in zip(sorted(Path(baseline.outputs['alpha']).glob('*.exr')),
                                                   sorted(Path(result.outputs['alpha']).glob('*.exr'))):
                            raw = cv2.imread(str(base_path), cv2.IMREAD_UNCHANGED)
                            actual = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                            assert raw is not None and actual is not None
                            expected = finish_alpha(raw, expansion=expansion, feather=2)
                            np.testing.assert_array_equal(actual, expected)
                        for a, b in zip(sorted(Path(baseline.outputs['fg']).glob('*.exr')),
                                        sorted(Path(result.outputs['fg']).glob('*.exr'))):
                            self.assertEqual(a.read_bytes(), b.read_bytes())
                        for cutout, fg, alpha in zip(sorted(Path(result.outputs['cutout']).glob('*.exr')),
                                                     sorted(Path(result.outputs['fg']).glob('*.exr')),
                                                     sorted(Path(result.outputs['alpha']).glob('*.exr'))):
                            rgba = cv2.imread(str(cutout), cv2.IMREAD_UNCHANGED)
                            rgb = cv2.imread(str(fg), cv2.IMREAD_UNCHANGED)
                            coverage = cv2.imread(str(alpha), cv2.IMREAD_UNCHANGED)
                            assert rgba is not None and rgb is not None and coverage is not None
                            if not quick:
                                self.assertLess(float(rgb.min()), 0)
                                self.assertGreater(float(rgb.max()), 1)
                            np.testing.assert_array_equal(rgba[..., 3], coverage)
                            np.testing.assert_allclose(rgba[..., :3], rgb * coverage[..., None], atol=1e-7)
                            self.assertTrue(np.isfinite(rgba).all())
                        if quick and approach == 'offline':
                            assert result.quick_key is not None
                            self.assertGreater(len(result.quick_key['extra_neural_frames']), 0)
            self.assertEqual({p: p.read_bytes() for p in source.iterdir()}, before)

    def test_authored_matte_wins_and_disabled_is_bitwise_old_path(self):
        import tempfile

        import cv2
        import numpy as np

        from corridorkey.refinement import finish_alpha
        from tests.test_spatial_draft import SpatialDraftTests

        with tempfile.TemporaryDirectory() as temporary:
            helper = SpatialDraftTests()
            helper.root = Path(temporary)
            source = helper.source(exr=True)
            correction, mask = helper.corrections(source)
            for scale, low, quick in ((1., False, False), (.5, True, False), (.5, True, True)):
                with self.subTest(scale=scale, low=low, quick=quick):
                    with patch('corridorkey.refinement.finish_alpha', side_effect=AssertionError('disabled called')):
                        _, old, _ = helper.run_job(source, scale=scale, low=low, quick=quick,
                                                  correction=correction, outputs={'alpha'})
                    _, result, _ = helper.run_job(source, scale=scale, low=low, quick=quick,
                                                 correction=correction, outputs={'alpha'},
                                                 overrides={'alpha_expand_radius': 2, 'alpha_feather_radius': 2})
                    path = min(Path(result.outputs['alpha']).glob('*.exr'))
                    actual = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    assert actual is not None
                    expected = finish_alpha(np.full((65, 97), .4, np.float32), expansion=2, feather=2)
                    np.testing.assert_allclose(actual, expected * .5 + mask * .5, atol=1e-7)
                    # Disabled jobs retain the exact original constrained pixels.
                    original = cv2.imread(str(min(Path(old.outputs['alpha']).glob('*.exr'))), cv2.IMREAD_UNCHANGED)
                    assert original is not None
                    np.testing.assert_allclose(original, .2 + mask * .5, atol=1e-7)

    def test_trimap_classes_override_finishing_at_boundaries(self):
        import tempfile

        import cv2
        import numpy as np

        from corridorkey.corrections import save_corrections
        from corridorkey.identities import source_identity
        from tests.test_spatial_draft import SpatialDraftTests
        from tests.test_trimap_refinement import TrimapRefinementTests

        with tempfile.TemporaryDirectory() as temporary:
            helper = SpatialDraftTests()
            helper.root = Path(temporary)
            trimaps = TrimapRefinementTests()
            trimaps.root = helper.root
            source = helper.source(exr=True)
            labels = np.full((65, 97), 128, np.uint8)
            labels[:, :20], labels[:, -20:] = 0, 255
            asset = trimaps.asset(labels)
            document = trimaps.document(asset, shape=labels.shape, frames=5,
                                        source_hash=source_identity(source)['sha256'])
            correction = helper.root / 'trimap.json'
            save_corrections(correction, document)
            for scale, low, quick in ((1., False, False), (.5, True, False), (.5, True, True)):
                with self.subTest(scale=scale, low=low, quick=quick):
                    _, result, _ = helper.run_job(source, scale=scale, low=low, quick=quick,
                                                 correction=correction, outputs={'alpha'},
                                                 overrides={'alpha_expand_radius': 2, 'alpha_feather_radius': 2})
                    for path in Path(result.outputs['alpha']).glob('*.exr'):
                        actual = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                        assert actual is not None
                        np.testing.assert_array_equal(actual[labels == 0], 0)
                        np.testing.assert_array_equal(actual[labels == 255], 1)
                    assert result.corrections is not None
                    self.assertEqual(result.corrections['output_frames_refined'], 5)

    def test_actual_runtime_cancellation_in_finishing_preserves_source(self):
        import tempfile

        from corridorkey import refinement
        from corridorkey.control import CancellationToken, CancelledError, JobControl
        from tests.test_spatial_draft import SpatialDraftTests

        with tempfile.TemporaryDirectory() as temporary:
            helper = SpatialDraftTests()
            helper.root = Path(temporary)
            source = helper.source(exr=True)
            before = {p: p.read_bytes() for p in source.iterdir()}
            original = refinement.finish_alpha
            for quick in (False, True):
                with self.subTest(quick=quick):
                    token = CancellationToken()
                    def cancel(alpha, token=token, **kwargs):
                        token.cancel('cancel during native finishing')
                        return original(alpha, **kwargs)
                    with patch.object(refinement, 'finish_alpha', cancel), self.assertRaisesRegex(CancelledError, 'native finishing'):
                        helper.run_job(source, quick=quick, control=JobControl(token),
                                       overrides={'alpha_feather_radius': 2})
            self.assertEqual({p: p.read_bytes() for p in source.iterdir()}, before)
            self.assertEqual(list(helper.root.glob('out-*/*/*.exr')), [])
            self.assertEqual(list((helper.root / 'scratch').glob('**/*.f32')), [])
