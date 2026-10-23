"""CPU adapter tests; skip explicitly when the optional numeric stack is absent."""
import importlib.util
from pathlib import Path
import tempfile
import sys
import subprocess
import time
import threading
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))
if HAS_STACK:
    import cv2
    import numpy as np
    import torch
    from corridorkey import runtime
    from corridorkey.video_io import (
        ExrSequenceWriter, SequentialHintFrameDirWindowReader,
        WriterConfig, ThreadedVideoWriter, preprocess_hint_frame, read_image_rgb,
    )
    from corridorkey.config import InferenceConfig
    from corridorkey.diagnostics import RunDiagnostics


@unittest.skipUnless(HAS_STACK, "requires CPU PyTorch, NumPy and OpenCV (no weights required)")
class AdapterTests(unittest.TestCase):
    def test_low_vram_overrides_reset_for_next_normal_job(self):
        from corridorkey.blocks import SwiGLU
        model = torch.nn.Module()
        model.stage1 = torch.nn.Sequential(SwiGLU(dim=8))
        runtime.configure_inference_memory(model, InferenceConfig(low_vram=True, low_vram_mlp_chunk_tokens=2))
        self.assertEqual(model.stage1[0].inference_chunk_tokens, 2)
        self.assertEqual(model.stage1.low_vram_enc_frame_chunk, 1)
        runtime.configure_inference_memory(model, InferenceConfig())
        self.assertEqual(model.stage1[0].inference_chunk_tokens, 0)
        self.assertEqual(model.stage1.low_vram_enc_frame_chunk, 0)

    def test_runtime_frame_accounting_and_real_alpha_exr_output(self):
        # Replace neural work with a deterministic CPU test double. Keep real
        # batching, rendering, EXR writing, diagnostics and cleanup.
        class Stage2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def cuda(self):
                return self

            def forward(self, rgb, baton):
                result = torch.zeros(rgb.shape[0], rgb.shape[1], 4, *rgb.shape[-2:])
                result[:, :, 0] = 0.25
                return result

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "source.mp4")
            source.touch()
            meta = {"num_frames": 3, "shape_hw": (128, 128), "model_native_hw": (2, 2),
                    "output_hw": (2, 2), "native_fps": 24, "pad": (0, 0, 0, 0)}
            model = torch.nn.Module()
            model.stage1 = torch.nn.Identity()
            model.stage2 = Stage2()
            model.foundations = torch.nn.Module()
            model.foundations.rvm = None
            cfg = InferenceConfig(outputs={"alpha"}, output_format="exr", exr_dtype="float",
                                  stage2_batch=2, temp_dir=Path(folder, "cache"))
            diagnostics = RunDiagnostics()

            def stage1(*args, **kwargs):
                kwargs["on_baton_frames"]([
                    (t, np.zeros((1, 8, 8), dtype=np.float16), torch.zeros(3, 128, 128))
                    for t in range(3)])

            with (patch.object(runtime, "probe_video_meta", return_value=meta),
                  patch.object(runtime, "_phase1_setup"),
                  patch.object(runtime, "hann_stage1_stream", side_effect=stage1),
                  patch.object(runtime, "make_checker", side_effect=AssertionError("unneeded checker")),
                  patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext()),
                  patch.object(torch.cuda, "empty_cache")):
                runtime.run_inference(model, str(source), str(Path(folder, "output")), cfg, WriterConfig(),
                                      initial_hint_path=None, hint_video_path=None, diagnostics=diagnostics)
            self.assertEqual(diagnostics.frames_submitted, 3)
            self.assertIsNotNone(diagnostics.first_frame_ready_seconds)
            assert isinstance(diagnostics.effective, dict)
            self.assertEqual(diagnostics.effective["stage2_batch"], 2)
            outputs = list(Path(diagnostics.outputs["alpha"]).glob("*.exr"))
            self.assertEqual(len(outputs), 3)
            for path in outputs:
                alpha = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                np.testing.assert_array_equal(alpha, np.full((2, 2), 0.25, dtype=np.float32))
            self.assertEqual(list(Path(folder, "cache").iterdir()), [])

    def test_rgba_hint_uses_coverage_not_luminance(self):
        rgba = np.full((2, 3, 4), 255, dtype=np.uint8)
        rgba[:, :, 3] = 0
        hint = preprocess_hint_frame(rgba, {"shape_hw": (2, 3), "model_native_hw": (2, 3)})
        self.assertEqual(float(hint.sum()), 0)

    def test_partial_clip_hint_directory_matches_source_offset(self):
        with tempfile.TemporaryDirectory() as folder:
            for index in range(5):
                cv2.imwrite(str(Path(folder, f"frame{index}.PNG")),
                            np.full((2, 3), index * 50, dtype=np.uint8))
            reader = SequentialHintFrameDirWindowReader(folder, {
                "start_frame": 2, "num_frames": 2, "shape_hw": (2, 3),
                "model_native_hw": (2, 3)})
            values = reader.window(0, 2).float().mean(dim=(1, 2, 3)).numpy()
            np.testing.assert_allclose(values, [100 / 255, 150 / 255], atol=0.002)

    def test_hint_video_rejects_mismatched_or_missing_fps_and_releases_capture(self):
        from corridorkey.video_io import SequentialHintVideoWindowReader
        for fps in (25, 0, float("nan")):
            with self.subTest(fps=fps):
                capture = MagicMock()
                capture.get.side_effect = lambda prop: fps if prop == cv2.CAP_PROP_FPS else 100
                with patch.object(cv2, "VideoCapture", return_value=capture):
                    with self.assertRaisesRegex(ValueError, "FPS"):
                        SequentialHintVideoWindowReader("hint.mp4", {
                            "start_frame": 0, "num_frames": 3, "native_fps": 24})
                capture.release.assert_called_once()

    def test_float_exr_roundtrip_retains_signed_hdr_and_alpha(self):
        with tempfile.TemporaryDirectory() as folder:
            frames = np.array([[[[-0.25, 2.0, 0.1, 0.3]]]], dtype=np.float32)
            writer = ExrSequenceWriter(folder, 24, (1, 1), channels=4, dtype="float")
            writer.write_many(frames)
            writer.release()
            result = cv2.imread(str(Path(folder, "frame_000000.exr")), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(result, frames[0])

    def test_exr_failure_surfaces_on_release(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cv2, "imwrite", return_value=False):
            writer = ExrSequenceWriter(folder, 24, (2, 2), channels=1)
            writer.write_many(np.zeros((1, 2, 2, 1), dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "imwrite failed"):
                writer.release()

    def test_failed_encoder_process_is_reaped(self):
        # Python deliberately rejects FFmpeg arguments and exits. This exercises
        # a real child-process/broken-pipe failure without requiring FFmpeg.
        with tempfile.TemporaryDirectory() as folder:
            writer = ThreadedVideoWriter(str(Path(folder, "output.mp4")), 24, (128, 128),
                                         WriterConfig(ffmpeg_bin=sys.executable))
            writer.write_many(np.zeros((1, 128, 128, 3), dtype=np.uint16))
            with self.assertRaises((BrokenPipeError, RuntimeError)):
                writer.release()
            self.assertIsNotNone(writer.proc.poll())
            self.assertFalse(writer.worker.thread.is_alive())

    def test_encoder_that_never_reads_is_terminated_on_deadline(self):
        from corridorkey import video_io
        real_popen = subprocess.Popen

        def stalled_encoder(*args, **kwargs):
            return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

        with tempfile.TemporaryDirectory() as folder, patch.object(
                video_io.subprocess, "Popen", side_effect=stalled_encoder):
            started = time.monotonic()
            writer = ThreadedVideoWriter(str(Path(folder, "output.mp4")), 24, (512, 512),
                                         WriterConfig(ffmpeg_bin=sys.executable, timeout_seconds=0.15))
            writer.write_many(np.zeros((1, 512, 512, 3), dtype=np.uint16))
            with self.assertRaises(TimeoutError):
                writer.release()
            self.assertLess(time.monotonic() - started, 4)
            self.assertIsNotNone(writer.proc.poll())
            self.assertFalse(writer.worker.thread.is_alive())

    def test_cancelled_stalled_encoder_is_reaped(self):
        from corridorkey import video_io
        from corridorkey.control import CancellationToken, CancelledError
        real_popen = subprocess.Popen
        token = CancellationToken()
        def stalled_encoder(*args, **kwargs):
            return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        with tempfile.TemporaryDirectory() as folder, patch.object(
                video_io.subprocess, "Popen", side_effect=stalled_encoder):
            writer = ThreadedVideoWriter(str(Path(folder, "output.mp4")), 24, (512, 512),
                                         WriterConfig(ffmpeg_bin=sys.executable), check=token.check)
            writer.write_many(np.zeros((1, 512, 512, 3), dtype=np.uint16))
            timer = threading.Timer(0.1, token.cancel, args=("cancel encoder",))
            timer.start()
            try:
                with self.assertRaisesRegex(CancelledError, "cancel encoder"):
                    writer.release()
            finally:
                timer.join()
            self.assertIsNotNone(writer.proc.poll())
            self.assertFalse(writer.worker.thread.is_alive())

    def test_encoder_finalization_honors_deadline_and_cancellation(self):
        from corridorkey import video_io
        from corridorkey.control import CancellationToken, CancelledError
        real_popen = subprocess.Popen
        def stalled_finalize(*args, **kwargs):
            return real_popen([sys.executable, "-c",
                              "import sys,time; sys.stdin.buffer.read(); time.sleep(30)"], **kwargs)
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as folder, patch.object(
                    video_io.subprocess, "Popen", side_effect=stalled_finalize):
                token = CancellationToken()
                writer = ThreadedVideoWriter(str(Path(folder, "output.mp4")), 24, (2, 2),
                                             WriterConfig(ffmpeg_bin=sys.executable,
                                                          timeout_seconds=60 if cancel else 0.15),
                                             check=token.check)
                writer.write_many(np.zeros((1, 2, 2, 3), dtype=np.uint16))
                timer = threading.Timer(0.1, token.cancel)
                if cancel:
                    timer.start()
                started = time.monotonic()
                try:
                    with self.assertRaises(CancelledError if cancel else TimeoutError):
                        writer.release()
                finally:
                    if cancel:
                        timer.join()
                self.assertLess(time.monotonic() - started, 4)
                self.assertIsNotNone(writer.proc.poll())
                self.assertFalse(writer.worker.thread.is_alive())

    def test_generated_fixture_has_correct_rgb_alpha_and_frame_count(self):
        from corridorkey.fixtures import generate
        from corridorkey.scheduling import frame_paths
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder, "fixture")
            generate(output, frames=3, width=16, height=8)
            rgb_paths, alpha_paths = frame_paths(output / "rgb"), frame_paths(output / "alpha")
            self.assertEqual((len(rgb_paths), len(alpha_paths)), (3, 3))
            rgb = cv2.imread(rgb_paths[1], cv2.IMREAD_UNCHANGED)
            alpha = cv2.imread(alpha_paths[1], cv2.IMREAD_UNCHANGED)
            self.assertEqual(rgb.shape, (8, 16, 3))
            np.testing.assert_array_equal(rgb[:, :, 0], alpha)
            np.testing.assert_array_equal(rgb[:, :, 2], alpha)
            self.assertTrue(np.all(rgb[:, :, 1] == 255))
            with self.assertRaises(FileExistsError):
                generate(output)

    def test_high_bit_png_retains_native_levels_and_grayscale_remains_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder, "source.png"))
            cv2.imwrite(path, np.full((2, 2, 3), 10000, dtype=np.uint16))
            np.testing.assert_allclose(read_image_rgb(path), np.float32(10000 / 65535), atol=1e-8)
            cv2.imwrite(path, np.full((2, 2), 128, dtype=np.uint8))
            self.assertEqual(read_image_rgb(path).shape, (2, 2, 3))

    def test_runtime_failure_releases_all_writers_and_temp_directory(self):
        # Model work is intentionally mocked: this tests orchestration failure,
        # not neural inference, GPU behavior, or numerical quality.
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder, "source.mp4")
            source.touch()
            temp = Path(folder, "cache")
            first, second = MagicMock(), MagicMock()
            first.release.side_effect = OSError("encoder cleanup failed")
            model = MagicMock()
            model.foundations.rvm = None
            meta = {"num_frames": 1, "shape_hw": (128, 128), "output_hw": (2, 2), "native_fps": 24}
            with (patch.object(runtime, "probe_video_meta", return_value=meta),
                  patch.object(runtime, "ThreadedVideoWriter", side_effect=[first, second]),
                  patch.object(runtime, "_phase1_setup"),
                  patch.object(runtime, "hann_stage1_stream", side_effect=ValueError("inference failed")),
                  patch.object(torch.cuda, "empty_cache")):
                with self.assertRaisesRegex(ValueError, "inference failed") as caught:
                    runtime.run_inference(model, str(source), str(Path(folder, "output")),
                                          InferenceConfig(temp_dir=temp), WriterConfig(),
                                          initial_hint_path=None, hint_video_path=None)
            first.release.assert_called_once()
            second.release.assert_called_once()
            self.assertEqual(list(temp.iterdir()), [])
            self.assertIn("encoder cleanup failed", caught.exception.__notes__[0])

    def test_public_model_alias_is_stable(self):
        from corridorkey import CorridorKeyV2
        from corridorkey import CorridorKeyV2 as again
        self.assertIs(CorridorKeyV2, again)
        self.assertTrue(issubclass(CorridorKeyV2, runtime.CorridorKeyV3))
