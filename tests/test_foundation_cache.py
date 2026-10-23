"""CPU feature/scheduler tests with deterministic model substitutes, no weights."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))
if HAS_STACK:
    import numpy as np
    import torch
    from corridorkey.foundation_cache import FoundationWindowCache
    from corridorkey import runtime
    from corridorkey.config import InferenceConfig, WriterConfig
    from corridorkey.diagnostics import RunDiagnostics
    from corridorkey.scheduling import window_starts


@unittest.skipUnless(HAS_STACK, "requires optional CPU PyTorch, NumPy and OpenCV")
class FeatureCacheTests(unittest.TestCase):
    def test_overlap_reuses_features_without_changing_frame_values(self):
        display = torch.arange(15, dtype=torch.float32).view(15, 1, 1, 1).expand(15, 3, 4, 4)
        evaluated = []

        def extract(source):
            values = source[:, 0, 0, 0]
            evaluated.extend(values.tolist())
            return {"moge": values[:, None, None, None].expand(-1, 1, 2, 2),
                    "cradio": (values + 1)[:, None, None, None].expand(-1, 2, 2, 2)}

        cache = FoundationWindowCache(8 * 3 * 2 * 2 * 4)
        for start in window_starts(15, 8, 4):
            result = cache.window(display[start:start + 8], start, device="cpu",
                                  channels={"moge": 1, "cradio": 2}, grid_hw=(2, 2), extract=extract)
            for name, offset in (("moge", 0), ("cradio", 1)):
                expected = (display[start:start + 8, 0, 0, 0] + offset)
                torch.testing.assert_close(result[name][:, 0, 0, 0], expected, rtol=0, atol=0)
            self.assertLessEqual(cache.cache.bytes_used, cache.cache.max_bytes)
        self.assertEqual(evaluated, list(range(15)))
        self.assertEqual(cache.snapshot()["hits"], 9)
        self.assertEqual(cache.snapshot()["foundation_frames_evaluated"], 15)
        cache.close()
        self.assertEqual(cache.snapshot()["resident_bytes"], 0)

    def test_owned_storage_does_not_retain_batch_and_cannot_be_mutated_through_result(self):
        features = torch.arange(12, dtype=torch.float32).view(3, 1, 2, 2)
        features[0, 0, 0, 0] = -0.25
        features[0, 0, 0, 1] = 70000
        cache = FoundationWindowCache(3 * 16)
        output = cache.window(features, 0, device="cpu", channels={"moge": 1}, grid_hw=(2, 2),
                              extract=lambda _: {"moge": features})
        output["moge"].zero_()
        features.zero_()
        payload = cache.cache.get(2)["moge"]
        self.assertEqual(payload.untyped_storage().nbytes(), 16)
        self.assertEqual(payload.flatten().tolist(), [8, 9, 10, 11])
        self.assertEqual(cache.cache.get(0)["moge"][0, 0, :].tolist(), [-0.25, 70000])

    def test_tiny_budget_recomputes_without_precision_change(self):
        cache = FoundationWindowCache(1)
        source = torch.tensor([[[[0.125]]], [[[2.0]]]], dtype=torch.float32)
        extract = Mock(side_effect=lambda data: {"moge": data * 0.5})
        for _ in range(2):
            actual = cache.window(source, 0, device="cpu", channels={"moge": 1}, grid_hw=(1, 1), extract=extract)
            torch.testing.assert_close(actual["moge"], source * 0.5, rtol=0, atol=0)
        self.assertEqual(cache.snapshot()["foundation_frames_evaluated"], 4)
        self.assertEqual(cache.snapshot()["oversized_frames"], 4)
        self.assertEqual(cache.snapshot()["peak_payload_bytes"], 0)

    def test_disabled_mode_preserves_extraction_batch_and_failed_calls_are_not_completed_frames(self):
        cache = FoundationWindowCache(0)
        source = torch.ones(3, 1, 1, 1)
        extract = Mock(return_value={"moge": source})
        cache.window(source, 0, device="cpu", channels={"moge": 1}, grid_hw=(1, 1), extract=extract)
        self.assertIs(extract.call_args.args[0], source)
        with self.assertRaisesRegex(OSError, "extract failed"):
            cache.window(source, 0, device="cpu", channels={"moge": 1}, grid_hw=(1, 1),
                         extract=Mock(side_effect=OSError("extract failed")))
        self.assertEqual(cache.snapshot()["foundation_frames_attempted"], 6)
        self.assertEqual(cache.snapshot()["foundation_frames_evaluated"], 3)

    def test_invalid_feature_contract_rejected_instead_of_caching_corrupt_output(self):
        for returned in ({"moge": torch.zeros(2, 1, 1, 1, dtype=torch.float16)},
                         {"moge": torch.zeros(1, 1, 1, 1)}, {"rvm": torch.zeros(2, 1, 1, 1)}):
            cache = FoundationWindowCache(8)
            with self.assertRaises(ValueError):
                cache.window(torch.ones(2, 1, 1, 1), 0, device="cpu", channels={"moge": 1},
                             grid_hw=(1, 1), extract=lambda _: returned)
            self.assertEqual(cache.snapshot()["entries"], 0)


@unittest.skipUnless(HAS_STACK, "requires optional CPU PyTorch, NumPy and OpenCV")
class MemorySelectionTests(unittest.TestCase):
    def test_auto_budget_capped_by_both_user_and_available_memory(self):
        for free, expected in ((16 * 2**30, 8 * 2**30), (5 * 2**30, 6 * 2**30)):
            config = InferenceConfig(auto_vram=True, auto_vram_budget_mb=8192)
            with (patch.object(torch.cuda, "mem_get_info", return_value=(free, 24 * 2**30)),
                  patch.object(torch.cuda, "memory_reserved", return_value=2**30),
                  patch.object(torch.cuda, "memory_allocated", return_value=2**30)):
                policy = runtime.apply_auto_vram(config, (1088, 1920), 80)
            self.assertEqual(policy["selection_budget_bytes"], expected)
            estimate = runtime.estimate_peak_vram_gb(1088 * 1920 / 1e6, min(config.hann_chunk, 80), config.low_vram)
            self.assertLessEqual(estimate * 1.1 + 0.3, expected / 2**30)

    def test_explicit_budget_rejects_unfittable_profile_or_already_loaded_model(self):
        for allocated in (0, 2 * 2**30):
            with (patch.object(torch.cuda, "mem_get_info", return_value=(8 * 2**30, 24 * 2**30)),
                  patch.object(torch.cuda, "memory_reserved", return_value=0),
                  patch.object(torch.cuda, "memory_allocated", return_value=allocated)):
                with self.assertRaises(MemoryError):
                    runtime.apply_auto_vram(InferenceConfig(auto_vram=True, auto_vram_budget_mb=1024), (1088, 1920), 80)

    def test_automatic_low_vram_choice_records_ram_cache_override_without_mutating_request(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "source.mp4")
            source.touch()
            config = InferenceConfig(auto_vram=True, foundation_cache_mb=64, output_format="exr",
                                     outputs={"alpha"}, temp_dir=Path(folder, "temp"))
            diagnostics = RunDiagnostics()
            model = SimpleNamespace(stage1=SimpleNamespace(moge_dim=1, cradio_dim=0, d_16=1),
                                    foundations=SimpleNamespace(rvm=None), eval=lambda: None, modules=lambda: [])
            writer = Mock(path=str(Path(folder, "output")))

            def select(cfg, *args):
                cfg.low_vram = True
                return {"selection_budget_bytes": 2**30}

            with (patch.object(runtime, "probe_video_meta", return_value={
                      "num_frames": 1, "shape_hw": (128, 128), "output_hw": (2, 2), "native_fps": 24}),
                  patch.object(runtime, "apply_auto_vram", side_effect=select),
                  patch.object(runtime, "ExrSequenceWriter", return_value=writer),
                  patch.object(runtime, "_phase1_setup", side_effect=OSError("stop after policy")),
                  patch.object(torch.cuda, "empty_cache")):
                with self.assertRaisesRegex(OSError, "stop after policy"):
                    runtime.run_inference(model, str(source), str(Path(folder, "out")), config, WriterConfig(),
                                          initial_hint_path=None, hint_video_path=None, diagnostics=diagnostics)
            self.assertEqual(config.foundation_cache_mb, 64)
            self.assertFalse(config.low_vram)
            assert isinstance(diagnostics.effective, dict)
            self.assertEqual(diagnostics.effective["foundation_cache_mb"], 0)
            self.assertEqual(diagnostics.foundation_cache["mode"], "low_vram_disk_prepass")
            assert isinstance(diagnostics.execution, dict)
            self.assertEqual(diagnostics.execution["temp_maps"]["required_bytes"], 144)
            self.assertEqual(diagnostics.execution["temp_maps"]["cleanup"], "complete")
            self.assertTrue(diagnostics.warnings)


@unittest.skipUnless(HAS_STACK, "requires optional CPU PyTorch, NumPy and OpenCV")
class StreamReuseTests(unittest.TestCase):
    def run_stream(self, budget, count=15, *, cradio=True, fail=False, offset=0, precomputed=False, cancel=False, persistent=None, durable_hann=None, disk_baton=False, recurrent=False):
        import numpy as np
        import torch

        diagnostics = RunDiagnostics()
        diagnostics.durable_foundations = persistent
        diagnostics.durable_hann = durable_hann
        from corridorkey.control import JobControl, CancellationToken, CancelledError
        token = CancellationToken()
        if cancel:
            def progress(event):
                if event.stage == "stage1_windows" and event.completed == 1:
                    token.cancel("cancel stage1")
            diagnostics.control = JobControl(token, progress)
        owner = self

        class Foundations(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rvm = SimpleNamespace(embed_dim=1)
                self.batches = []

            def forward(self, *, rgb_cradio, rgb_moge, **kwargs):
                owner.assertEqual(kwargs, {}, "RVM must stay in its chronological prepass")
                self.batches.append(rgb_moge[:, 0, 0, 0].float().tolist())
                values = rgb_moge.float().mean(dim=(1, 2, 3))
                result = {"moge": (values / 2)[:, None, None, None].expand(-1, 1, 2, 2)}
                if cradio:
                    result["cradio"] = (values + 0.125)[:, None, None, None].expand(-1, 1, 2, 2)
                return result

        class Stage1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.d_16 = self.moge_dim = 1
                self.recurrent_64 = torch.nn.ModuleList([torch.nn.Identity()] if recurrent else [])
                self.cradio_dim = 1 if cradio else 0
                self.seen = []

            def forward(self, rgb, hint, moge_feat, *, cradio_feat, rvm_feat, **kwargs):
                self.seen.append((rgb.clone(), rvm_feat.clone()))
                if fail and len(self.seen) == 2:
                    raise OSError("injected Stage 1 failure")
                value = moge_feat + rvm_feat + hint.float().mean(dim=(-2, -1), keepdim=True) * 0.125
                if cradio_feat is not None:
                    value = value + cradio_feat
                baton = value.repeat_interleave(4, -2).repeat_interleave(4, -1)
                aux = torch.zeros(1, rgb.shape[1], 4, 64, 64)
                aux[:, :, 0] = rgb[:, :, 0, 0, 0, None, None]
                return {"baton": baton, "aux": aux}

        class Reader:
            closed = False

            def __init__(self, source, meta):
                values = (torch.arange(count, dtype=torch.float32) + offset) / 32
                self.source = values[:, None, None, None].expand(-1, 3, 128, 128).to(torch.bfloat16)

            def window(self, first, last):
                rgb = self.source[first:last]
                return rgb, rgb, rgb

            def close(self):
                Reader.closed = True

        model = SimpleNamespace(stage1=Stage1(), foundations=Foundations())
        frames = []
        with tempfile.TemporaryDirectory() as folder:
            rvm_path = Path(folder, "rvm.f16")
            rvm = np.memmap(rvm_path, dtype=np.float16, mode="w+", shape=(count, 1, 2, 2))
            for t in range(count):
                rvm[t] = t / 8
            rvm.flush()
            del rvm
            foundation_paths = None
            if precomputed:
                foundation_paths = {}
                for name in (["moge", "cradio"] if cradio else ["moge"]):
                    path = str(Path(folder, name + ".f32"))
                    shape = (count, 1, 2, 2)
                    features = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
                    for t in range(count):
                        features[t] = (t + offset) / 64 if name == "moge" else (t + offset) / 32 + 0.125
                    features.flush()
                    del features
                    foundation_paths[name] = path
                    foundation_paths[name + "_shape"] = shape
            mappings = []
            original_memmap = np.memmap
            def tracked(*args, **kwargs):
                result = original_memmap(*args, **kwargs)
                mappings.append(result)
                return result
            with (patch.object(np, "memmap", side_effect=tracked) if persistent is None and durable_hann is None else nullcontext(),
                  patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext()),
                  patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "empty_cache")):
                arguments = dict(reader_cls=Reader, reader_kind="frame_dir", chunk=8, stride=4,
                                 baton_path=Path(folder, "unused"), initial_hint=torch.ones(1, 128, 128),
                                 initial_hint_quality=0.95, full_hint_video_path=None, full_hint_quality=0.95,
                                 rvm_path=rvm_path, foundation_paths=foundation_paths, carry_hint_from_aux=True,
                                 carry_hint_quality=0.95, on_baton_frames=None if disk_baton else frames.extend,
                                 foundation_cache_bytes=budget, diagnostics=diagnostics)
                if cancel:
                    with self.assertRaisesRegex(CancelledError, "cancel stage1"):
                        runtime.hann_stage1_stream(model, "fixture", {"num_frames": count, "shape_hw": (128, 128)}, **arguments)
                elif fail:
                    with self.assertRaisesRegex(OSError, "Stage 1 failure"):
                        runtime.hann_stage1_stream(model, "fixture", {"num_frames": count, "shape_hw": (128, 128)}, **arguments)
                else:
                    runtime.hann_stage1_stream(model, "fixture", {"num_frames": count, "shape_hw": (128, 128)}, **arguments)
            self.assertTrue(all(mapping._mmap.closed for mapping in mappings))
            if disk_baton and not cancel and not fail:
                shape = (count, 1, 8, 8)
                baton = np.fromfile(Path(folder, "unused"), dtype=np.float16).reshape(shape)
                frames = [(frame, baton[frame].copy(), None) for frame in range(count)]
        self.assertTrue(Reader.closed)
        return frames, model, diagnostics

    def test_cached_and_uncached_blending_match_on_short_and_final_windows(self):
        for count in (1, 7, 15):
            with self.subTest(count=count):
                reference, baseline_model, baseline = self.run_stream(0, count)
                cached, cached_model, diagnostics = self.run_stream(256, count)
                self.assertEqual([f[0] for f in cached], list(range(count)))
                for expected, actual in zip(reference, cached):
                    np.testing.assert_array_equal(actual[1], expected[1])
                self.assertEqual(diagnostics.foundation_cache["foundation_frames_evaluated"], count)
                self.assertLessEqual(diagnostics.foundation_cache["peak_payload_bytes"], 256)
                self.assertEqual(diagnostics.foundation_cache["resident_bytes"], 0)
                for expected, actual in zip(baseline_model.stage1.seen, cached_model.stage1.seen):
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                if count == 15:
                    self.assertEqual(baseline.foundation_cache["foundation_frames_evaluated"], 24)
                    self.assertEqual(diagnostics.foundation_cache["hits"], 9)

    def test_optional_cradio_and_oversized_payloads_keep_outputs_valid(self):
        reference, _, _ = self.run_stream(0, cradio=False)
        recomputed, _, diagnostics = self.run_stream(1, cradio=False)
        for expected, actual in zip(reference, recomputed):
            np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(diagnostics.foundation_cache["oversized_frames"], 9)
        self.assertEqual(diagnostics.foundation_cache["peak_payload_bytes"], 0)

    def test_failure_releases_cache_and_next_job_cannot_reuse_stale_frame_ids(self):
        _, _, failed = self.run_stream(256, fail=True)
        self.assertEqual(failed.foundation_cache["resident_bytes"], 0)
        reference, _, _ = self.run_stream(0, offset=16)
        cached, _, diagnostics = self.run_stream(256, offset=16)
        for expected, actual in zip(reference, cached):
            np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(diagnostics.foundation_cache["foundation_frames_evaluated"], 15)

    def test_precomputed_mode_without_cradio_matches_reference(self):
        reference, _, _ = self.run_stream(0, cradio=False)
        precomputed, model, _ = self.run_stream(0, cradio=False, precomputed=True)
        self.assertEqual(model.foundations.batches, [])
        for expected, actual in zip(reference, precomputed):
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_cancellation_closes_cached_and_precomputed_window_resources(self):
        for precomputed in (False, True):
            with self.subTest(precomputed=precomputed):
                self.run_stream(256, precomputed=precomputed, cancel=True)


    def test_durable_normal_stream_reuses_frames_with_identical_batons(self):
        import numpy as np

        from corridorkey.durable_foundations import DurableFeatureFrames
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore

        parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
                                          'assets': {}, 'environment': {'runtime': {'synthetic': True}},
                                          'fixture': 'normal runtime CPU substitutes'}))
        reference = []
        with tempfile.TemporaryDirectory() as folder:
            for index in range(2):
                with BundleStore(Path(folder) / 'store', 100000) as store:
                    adapter = DurableFeatureFrames(store, parent, validate_context=lambda: None)
                    adapter.bind(None)
                    frames, model, diagnostics = self.run_stream(0, count=7, persistent=adapter)
                    assert isinstance(diagnostics.foundation_cache, dict)
                    if index == 0:
                        reference = frames
                        self.assertEqual(diagnostics.foundation_cache['foundation_frames_evaluated'], 7)
                    else:
                        self.assertEqual(diagnostics.foundation_cache['foundation_frames_evaluated'], 0)
                        self.assertEqual(model.foundations.batches, [])
                        self.assertEqual(adapter.hits, 7)
                        for expected, actual in zip(reference, frames):
                            np.testing.assert_array_equal(actual[1], expected[1])


    def test_hann_replay_restores_carry_and_overlap_after_interruption_in_both_paths(self):
        import numpy as np

        from corridorkey.durable_hann import DurableHannWindows
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore

        parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
                                          'assets': {}, 'environment': {'runtime': {'synthetic': True}}}))
        for disk in (False, True):
            reference, _, _ = self.run_stream(0, precomputed=disk, disk_baton=disk)
            with self.subTest(disk=disk), tempfile.TemporaryDirectory() as folder:
                with BundleStore(Path(folder) / 'store', 3000000) as store:
                    adapter = DurableHannWindows(store, parent, validate_context=lambda: None)
                    adapter.bind(None)
                    self.run_stream(0, precomputed=disk, disk_baton=disk, durable_hann=adapter, cancel=True)
                    self.assertEqual(adapter.stored, 1)
                with BundleStore(Path(folder) / 'store', 3000000) as store:
                    adapter = DurableHannWindows(store, parent, validate_context=lambda: None)
                    adapter.bind(None)
                    resumed, model, _ = self.run_stream(0, precomputed=disk, disk_baton=disk, durable_hann=adapter)
                    self.assertEqual(adapter.hits, 1)
                    self.assertEqual(len(model.stage1.seen), 2)
                    for expected, actual in zip(reference, resumed):
                        np.testing.assert_array_equal(actual[1], expected[1])
                    again, model, _ = self.run_stream(0, precomputed=disk, disk_baton=disk, durable_hann=adapter)
                    self.assertEqual(len(model.stage1.seen), 0)
                    self.assertEqual(adapter.hits, 4)
                    for expected, actual in zip(reference, again):
                        np.testing.assert_array_equal(actual[1], expected[1])
                    import json
                    for entry in (store.root / 'entries').iterdir():
                        data = json.loads((entry / 'complete.json').read_text())['identity']
                        if data['index'] == 1:
                            (entry / 'payload-000000.bin').write_bytes(b'corrupt middle event')
                            break
                    repaired, model, _ = self.run_stream(0, precomputed=disk, disk_baton=disk, durable_hann=adapter)
                    self.assertEqual(len(model.stage1.seen), 1)
                    self.assertEqual(adapter.corrupt, 1)
                    for expected, actual in zip(reference, repaired):
                        np.testing.assert_array_equal(actual[1], expected[1])


    def test_hann_replay_rejects_active_stage1_recurrent_sites_before_lookup(self):
        from corridorkey.durable_hann import DurableHannWindows
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore

        parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
                                          'assets': {}, 'environment': {'runtime': {'synthetic': True}}}))
        with tempfile.TemporaryDirectory() as folder, BundleStore(Path(folder) / 'store', 3000000) as store:
            adapter = DurableHannWindows(store, parent, validate_context=lambda: None)
            adapter.bind(None)
            self.run_stream(0, count=7, durable_hann=adapter)
            with self.assertRaisesRegex(ValueError, 'disabled Stage1 recurrent sites'):
                self.run_stream(0, count=7, durable_hann=adapter, recurrent=True)
            self.assertEqual(adapter.hits, 0)
