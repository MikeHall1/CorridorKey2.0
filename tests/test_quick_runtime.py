"""Real CPU decode/memmap/EXR orchestration with deterministic neural substitutes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional CPU numeric stack")
class QuickRuntimeTests(unittest.TestCase):
    def run_job(self, root, *, low_vram=False, enabled=True, interval=4,
                fallback="stop", flashing=False, cuts=(), diagnostics=None, context_radius=1,
                auto_cuts=False, scene_cut=False, use_rvm=False, adaptive=False, max_gap=None,
                approach="offline", context_dependent=False, future_change=False, corrections=None,
                settings_overrides=None, alpha_pattern=False, source_rgba=False, bad_output=None, use_cradio=False):
        import cv2
        import numpy as np
        import torch
        from corridorkey import runtime
        from corridorkey.config import InferenceConfig, WriterConfig
        from corridorkey.diagnostics import RunDiagnostics

        class RVM(torch.nn.Module):
            embed_dim = 1
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.calls = []
            def forward(self, rgb, state=None):
                self.calls.append((rgb.shape[1], state))
                return torch.zeros(1, rgb.shape[1], 1, 1, 1), (state or 0) + rgb.shape[1]

        class Foundations(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.rvm = RVM() if use_rvm else None
                self.rvm_mean, self.rvm_std = torch.zeros(1), torch.ones(1)
            def forward(self, *, rgb_cradio, rgb_moge):
                features = {"moge": torch.zeros(len(rgb_moge), 1, 1, 1)}
                if use_cradio:
                    features["cradio"] = torch.zeros(len(rgb_cradio), 1, 1, 1)
                return features

        class Stage1(torch.nn.Module):
            d_16 = moge_dim = 1
            cradio_dim = 1 if use_cradio else 0
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.seen = []
                self.seen_hints = []
            def forward(self, rgb, hint, moge_feat, **kwargs):
                self.seen.append((len(rgb[0]), kwargs["t_stride"], kwargs["hint_quality"].clone()))
                self.seen_hints.append(hint.clone())
                count = rgb.shape[1]
                value = float(rgb.float().mean().clamp(0, 1)) * 0.5 if context_dependent else 0.4
                return {"baton": torch.full((1, count, 1, 4, 4), value),
                        "aux": torch.full((1, count, 4, rgb.shape[-2] // 2, rgb.shape[-1] // 2), 0.4)}

        class Stage2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
            def forward(self, rgb, baton):
                result = torch.zeros(1, rgb.shape[1], 4, 64, 64)
                result[:, :, 0] = 0.4
                if context_dependent:
                    result[:, :, 0] = baton[0, :, 0].mean(dim=(-1, -2)).reshape(1, -1, 1, 1)
                if alpha_pattern:
                    result[:, :, 0] = 0.2
                    result[:, :, 0, 32, 32] = 0.8
                    result[:, :, 0, 0, 0] = 0
                    result[:, :, 0, -1, -1] = 1
                if bad_output is not None:
                    result[:, :, 1] = bad_output
                return result

        root = Path(root)
        source = root / "source"
        source.mkdir(exist_ok=True)
        for t in range(9):
            value = (240 if t % 2 else 10) if flashing else 80 + t * 4
            if scene_cut:
                value = 30 if t < 4 else 210
            if future_change and t > 4:
                value = 220
            image = np.full((64, 64, 3), value, np.uint8)
            if source_rgba:
                image = np.zeros((64, 64, 4), np.uint16)
                image[:] = [10000, 20000, 30000, 32768]
            cv2.imwrite(str(source / f"frame{t}.png"), image)
        model = torch.nn.Module()
        model.foundations, model.stage1, model.stage2 = Foundations(), Stage1(), Stage2()
        cfg = InferenceConfig(hann_chunk=8, hann_stride=4, low_vram=low_vram, shape_quantum=64,
                              quick_key=enabled, quick_interval=interval, quick_context_radius=context_radius,
                              quick_fallback=fallback, quick_cut_frames=list(cuts),
                              quick_auto_cuts=auto_cuts,
                              quick_adaptive=adaptive, quick_max_gap=max_gap,
                              quick_approach=approach,
                              outputs={"alpha", "fg", "cutout", "checker"},
                              output_format="exr", exr_dtype="float", temp_dir=root / "temp",
                              stage2_batch=2, despill_strength=0)
        if corrections is not None:
            cfg.corrections_path = corrections(source, root)
        for name, value in (settings_overrides or {}).items():
            setattr(cfg, name, value)
        diagnostics = diagnostics if diagnostics is not None else RunDiagnostics()
        with (patch.object(torch.nn.Module, "cuda", lambda module: module),
              patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "empty_cache"),
              patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext())):
            runtime.run_inference(model, str(source), str(root / "output"), cfg, WriterConfig(),
                                  initial_hint_path=None, hint_video_path=None, diagnostics=diagnostics)
        self.assertEqual(list((root / "temp").iterdir()), [])
        return model, diagnostics

    def test_temp_map_budget_exact_boundary_and_rejection_before_writers(self):
        import cv2
        import numpy as np

        from corridorkey import runtime
        from corridorkey.diagnostics import RunDiagnostics
        for low_vram, enabled, use_rvm, use_cradio in ((False, False, False, False),
                    (False, True, True, False), (True, False, True, False),
                    (True, True, True, False), (True, True, False, True)):
            with self.subTest(low_vram=low_vram, quick=enabled, rvm=use_rvm):
                required = (18 if use_rvm else 0) + (36 + 288 if low_vram else 0) + (49152 if enabled else 0) + (36 if use_cradio else 0)
                with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                    _, baseline = self.run_job(first, low_vram=low_vram, enabled=enabled, use_rvm=use_rvm, use_cradio=use_cradio)
                    _, limited = self.run_job(second, low_vram=low_vram, enabled=enabled, use_rvm=use_rvm, use_cradio=use_cradio,
                        settings_overrides={"temp_map_budget_bytes": required})
                    assert limited.execution is not None
                    report = limited.execution["temp_maps"]
                    self.assertEqual(report["required_bytes"], required)
                    self.assertEqual(report["created_bytes"], required)
                    self.assertEqual(report["peak_reserved_bytes"], required)
                    self.assertEqual(report["reserved_bytes"], 0)
                    self.assertEqual(report["cleanup"], "complete")
                    for name in baseline.outputs:
                        for left, right in zip(sorted(Path(baseline.outputs[name]).glob("*.exr")),
                                               sorted(Path(limited.outputs[name]).glob("*.exr"))):
                            np.testing.assert_array_equal(cv2.imread(str(left), -1), cv2.imread(str(right), -1))
                if required:
                    with tempfile.TemporaryDirectory() as folder:
                        diagnostics = RunDiagnostics()
                        with (patch.object(runtime, "ExrSequenceWriter") as writer,
                              patch.object(runtime.np, "memmap") as mapping,
                              self.assertRaisesRegex(ValueError, "Managed inference scratch requires")):
                            self.run_job(folder, low_vram=low_vram, enabled=enabled, use_rvm=use_rvm, use_cradio=use_cradio,
                                diagnostics=diagnostics, settings_overrides={"temp_map_budget_bytes": required - 1})
                        writer.assert_not_called()
                        mapping.assert_not_called()
                        assert diagnostics.execution is not None
                        self.assertEqual(diagnostics.execution["temp_maps"]["status"], "rejected")
                        self.assertEqual(list((Path(folder)/"temp").iterdir()), [])

    def test_temp_map_creation_failure_and_cancel_cleanup(self):
        from corridorkey import runtime
        from corridorkey.control import CancelledError, JobControl
        from corridorkey.diagnostics import RunDiagnostics
        with tempfile.TemporaryDirectory() as folder:
            diagnostics = RunDiagnostics()
            def failed_map(path, **kwargs):
                Path(path).write_bytes(b"partial")
                raise OSError("disk full")
            with patch.object(runtime.np, "memmap", side_effect=failed_map), self.assertRaisesRegex(OSError, "disk full"):
                self.run_job(folder, diagnostics=diagnostics)
            assert diagnostics.execution is not None
            report = diagnostics.execution["temp_maps"]
            self.assertEqual(report["created_bytes"], 0)
            self.assertEqual(report["cleanup"], "complete")
            self.assertEqual(list((Path(folder)/"temp").iterdir()), [])
        with tempfile.TemporaryDirectory() as folder:
            control = JobControl()
            diagnostics = RunDiagnostics()
            diagnostics.control = control
            original = runtime.np.memmap
            def cancel_after_map(*a, **kw):
                result = original(*a, **kw)
                control.cancellation.cancel()
                return result
            with patch.object(runtime.np, "memmap", side_effect=cancel_after_map), self.assertRaises(CancelledError):
                self.run_job(folder, diagnostics=diagnostics)
            assert diagnostics.execution is not None
            self.assertEqual(diagnostics.execution["temp_maps"]["cleanup"], "complete")
            self.assertEqual(list((Path(folder)/"temp").iterdir()), [])

    def test_sparse_both_memory_paths_keep_all_current_source_frames_and_count_neural_work(self):
        import cv2
        import numpy as np
        from corridorkey import color_utils as cu
        for low_vram in (False, True):
            with self.subTest(low_vram=low_vram), tempfile.TemporaryDirectory() as folder:
                model, d = self.run_job(folder, low_vram=low_vram)
                self.assertEqual(d.frames_submitted, 9)
                self.assertEqual(d.quick_key["anchors"], [0, 4, 8])
                self.assertEqual(d.execution["stage1_frame_evaluations"], 7)
                self.assertEqual(d.execution["stage2_frame_evaluations"], 3)
                self.assertEqual(d.quick_key["flow_evaluations"], 24)
                self.assertTrue(all(stride == 1 for _, stride, _ in model.stage1.seen))
                self.assertEqual(d.quick_key["foreground"], "source_rgb_draft")
                for t, file in enumerate(sorted(Path(d.outputs["fg"]).glob("*.exr"))):
                    value = cu.srgb_to_linear(np.array((80 + t * 4) / 255, dtype=np.float32))
                    np.testing.assert_allclose(cv2.imread(str(file), cv2.IMREAD_UNCHANGED), value, atol=1e-7)
                self.assertEqual(len(list(Path(d.outputs["fg"]).glob("*.exr"))), 9)

    def test_interval_one_is_the_original_full_rate_render_path(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _, baseline = self.run_job(first, enabled=False)
            _, full_rate = self.run_job(second, interval=1)
            self.assertFalse(full_rate.quick_key["active"])
            self.assertEqual(baseline.execution, full_rate.execution)
            for name in baseline.outputs:
                for left, right in zip(sorted(Path(baseline.outputs[name]).glob("*.exr")),
                                       sorted(Path(full_rate.outputs[name]).glob("*.exr"))):
                    np.testing.assert_array_equal(cv2.imread(str(left), -1), cv2.imread(str(right), -1))

    def test_additional_neural_fallback_is_real_context_work_and_is_counted(self):
        for low_vram in (False, True):
            with self.subTest(low_vram=low_vram), tempfile.TemporaryDirectory() as folder:
                _, d = self.run_job(folder, low_vram=low_vram, fallback="neural", flashing=True)
                self.assertEqual(d.quick_key["extra_neural_frames"], [1, 3, 5, 7])
                self.assertEqual(d.execution["stage2_frame_evaluations"], 7)
                self.assertEqual(d.execution["stage1_frame_evaluations"], 19)

    def test_stop_fallback_preserves_partial_status_and_cleans_temporary_maps(self):
        from corridorkey.draft import UncertainMotionError
        from corridorkey.diagnostics import RunDiagnostics
        with tempfile.TemporaryDirectory() as folder:
            d = RunDiagnostics()
            with self.assertRaises(UncertainMotionError):
                self.run_job(folder, flashing=True, diagnostics=d)
            self.assertEqual(d.frames_submitted, 1)
            self.assertEqual(d.quick_key["fallback_frames"], [1])
            self.assertEqual(list(Path(folder, "temp").iterdir()), [])

    def test_cut_contexts_reset_carried_hints_and_keep_boundary_anchors(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            model, d = self.run_job(folder, cuts=[4], context_radius=2)
            self.assertEqual(d.quick_key["anchors"], [0, 3, 4, 8])
            self.assertTrue(torch.all(model.stage1.seen[0][2] == -1))
            self.assertTrue(torch.any(model.stage1.seen[1][2] > 0))
            self.assertTrue(torch.all(model.stage1.seen[2][2] == -1))

    def test_cancelled_draft_render_closes_temporary_maps(self):
        from corridorkey.control import JobControl, CancellationToken, CancelledError
        from corridorkey.diagnostics import RunDiagnostics
        token = CancellationToken()
        def progress(event):
            if event.stage == "quick_key_output" and event.completed == 1:
                token.cancel("cancel draft")
        d = RunDiagnostics()
        d.control = JobControl(token, progress)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(CancelledError, "cancel draft"):
                self.run_job(folder, diagnostics=d)
            self.assertEqual(d.frames_submitted, 1)
            self.assertEqual(list(Path(folder, "temp").iterdir()), [])

    def test_explicit_approximation_exports_labelled_trace_without_hidden_neural_work(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            _, d = self.run_job(folder, flashing=True, fallback="approximate")
            records = [json.loads(line) for line in Path(d.quick_key["frame_diagnostics_path"]).read_text().splitlines()]
            self.assertEqual([r["frame"] for r in records], list(range(9)))
            self.assertEqual([r["frame"] for r in records if r["fallback"] == "approximate"], [1, 3, 5, 7])
            self.assertEqual(d.execution["stage2_frame_evaluations"], 3)
            self.assertEqual(d.quick_key["extra_neural_frames"], [])
            self.assertTrue(all("source_rgb_draft" in output for output in d.outputs.values()))

    def test_detected_cut_controls_contexts_and_actual_chronological_rvm_resets(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            model, d = self.run_job(folder, scene_cut=True, auto_cuts=True, use_rvm=True)
            self.assertEqual(d.quick_key["analysis"]["detected_cuts"], [4])
            self.assertEqual(d.quick_key["anchors"], [0, 3, 4, 8])
            self.assertEqual(model.foundations.rvm.calls, [(4, None), (4, None), (1, 4)])
            self.assertEqual(d.execution["rvm_frame_evaluations"], 9)
            self.assertEqual(d.quick_key["analysis"]["source_frames_decoded"], 9)
            trace = Path(d.quick_key["analysis"]["trace_path"])
            self.assertEqual([json.loads(line)["frame"] for line in trace.read_text().splitlines()], list(range(9)))

    def test_cancelled_analysis_preserves_partial_diagnostics_without_rendering(self):
        from corridorkey.control import JobControl, CancellationToken, CancelledError
        from corridorkey.diagnostics import RunDiagnostics
        token = CancellationToken()
        def progress(event):
            if event.stage == "shot_analysis" and event.completed == 2:
                token.cancel("cancel shots")
        d = RunDiagnostics()
        d.control = JobControl(token, progress)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(CancelledError, "cancel shots"):
                self.run_job(folder, auto_cuts=True, diagnostics=d)
            self.assertEqual(d.frames_submitted, 0)
            self.assertEqual(d.quick_key["analysis"]["status"], "interrupted")
            self.assertEqual(d.quick_key["analysis"]["frames_analyzed"], 2)

    def test_maximum_gap_and_adaptive_promotions_reach_the_shared_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            _, d = self.run_job(folder, max_gap=2)
            self.assertEqual(d.quick_key["anchors"], [0, 2, 4, 6, 8])
            self.assertEqual(d.execution["stage2_frame_evaluations"], 5)

        with tempfile.TemporaryDirectory() as folder:
            _, d = self.run_job(folder, adaptive=True, fallback="neural", flashing=True)
            self.assertEqual(d.quick_key["promoted_anchors"], [1, 5])
            self.assertEqual(d.quick_key["anchors"], [0, 1, 4, 5, 8])
            self.assertEqual(d.execution["stage2_frame_evaluations"], 5)

    def test_hold_and_causal_approaches_keep_native_outputs_without_future_anchor_reads(self):
        for approach in ("hold", "causal"):
            for low_vram in (False, True):
                with self.subTest(approach=approach, low_vram=low_vram), tempfile.TemporaryDirectory() as folder:
                    _, d = self.run_job(folder, approach=approach, low_vram=low_vram)
                    self.assertEqual(d.frames_submitted, 9)
                    self.assertEqual(d.quick_key["max_look_ahead_frames"], 0)
                    self.assertEqual(d.quick_key["anchor_source_frames_decoded"], 0)
                    self.assertEqual(d.quick_key["flow_evaluations"], 0 if approach == "hold" else 12)

    def test_causal_native_contexts_cannot_change_earlier_alpha_when_future_rgb_changes(self):
        import cv2
        import numpy as np
        for low_vram in (False, True):
            with self.subTest(low_vram=low_vram), tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                _, before = self.run_job(a, approach="causal", context_radius=4, context_dependent=True,
                                         fallback="approximate", low_vram=low_vram, auto_cuts=True, use_rvm=True)
                _, after = self.run_job(b, approach="causal", context_radius=4, context_dependent=True,
                                        fallback="approximate", future_change=True, low_vram=low_vram, auto_cuts=True, use_rvm=True)
                self.assertEqual(before.quick_key["analysis"]["detected_cuts"], [])
                self.assertEqual(after.quick_key["analysis"]["detected_cuts"], [5])
                self.assertEqual(after.quick_key["planned_anchors"], [0, 4, 5])
                first = sorted(Path(before.outputs["alpha"]).glob("*.exr"))
                second = sorted(Path(after.outputs["alpha"]).glob("*.exr"))
                for left, right in zip(first[:5], second[:5]):
                    np.testing.assert_array_equal(cv2.imread(str(left), -1), cv2.imread(str(right), -1))
                self.assertFalse(np.array_equal(cv2.imread(str(first[-1]), -1), cv2.imread(str(second[-1]), -1)))
