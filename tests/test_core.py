import tempfile
import threading
import unittest
from pathlib import Path

from corridorkey.config import InferenceConfig
from corridorkey.lifecycle import WriterWorker, cleanup_all
from corridorkey.scheduling import frame_paths, window_starts


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_legacy_all_frame_limits(self):
        for count in (-5, -1, 0, 1):
            InferenceConfig(num_frames=count).validate()
        InferenceConfig(linear_brightness=-1, linear_contrast=-1).validate()

    def test_reject_invalid_settings_before_model_loading(self):
        for changes in (
            {"hann_stride": 81}, {"hann_chunk": 0}, {"hann_stride": -1},
            {"stage2_batch": 0}, {"model_scale": 1.5}, {"model_scale": float("nan")},
            {"linear_brightness": 0}, {"linear_contrast": 0},
            {"hint_quality": 1.1}, {"carry_hint_quality": float("nan")},
            {"frame_dir_fps": float("inf")}, {"outputs": set()},
            {"outputs": {"raw_fg"}}, {"shape_quantum": 63},
            {"output_format": "png"}, {"exr_dtype": "uint8"},
            {"start_mode": "end"}, {"seed": -1},
            {"low_vram": "false"}, {"auto_vram": 1}, {"frame_dir_fps": "24"},
            {"foundation_cache_mb": -1}, {"foundation_cache_mb": True},
            {"foundation_cache_mb": 1, "low_vram": True},
            {"auto_vram_budget_mb": 8192}, {"auto_vram": True, "auto_vram_budget_mb": 0},
            {"quick_key": "true"}, {"quick_interval": 0}, {"quick_interval": True},
            {"quick_context_radius": -1}, {"quick_flow_max_side": 8},
            {"quick_fallback": "silent"}, {"quick_min_reliability": float("nan")},
            {"quick_max_disagreement": 2}, {"quick_key": True, "quick_cut_frames": [True]},
            {"quick_auto_cuts": True}, {"quick_key": True, "quick_interval": 1, "quick_auto_cuts": True},
            {"quick_cut_threshold": 0}, {"quick_analysis_max_side": True},
            {"quick_adaptive": True}, {"quick_key": True, "quick_adaptive": True},
            {"quick_max_gap": True}, {"quick_max_gap": 3}, {"quick_motion_threshold": 0},
            {"quick_approach": "live"}, {"quick_approach": "causal"},
            {"frame_dir_alpha": "auto"}, {"frame_dir_alpha": "premultiplied"},
            {"cutout_alpha_mode": "premultiplied"}, {"alpha_refine_radius": True}, {"alpha_refine_radius": 5},
            {"alpha_refine_strength": float("nan")}, {"alpha_refine_sigma": 1e-300},
            {"despill_preserve_luma": True}, {"despill_space": "auto"}, {"despill_screen": "red"},
            {"quick_key": True, "quick_approach": "hold", "quick_adaptive": True, "quick_fallback": "neural"},
            {"quick_cut_frames": [1]}, {"quick_key": True, "quick_interval": 1, "quick_reset_frames": [1]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config = InferenceConfig()
                for name, value in changes.items():
                    setattr(config, name, value)
                config.validate()


class SchedulingTests(unittest.TestCase):
    def test_all_frames_covered_in_chronological_order(self):
        for count in (1, 2, 7, 80, 81, 163):
            for chunk, stride in ((1, 1), (8, 4), (24, 24), (80, 40)):
                starts = window_starts(count, chunk, stride)
                self.assertEqual(starts, sorted(set(starts)))
                covered = {t for start in starts for t in range(start, min(start + chunk, count))}
                self.assertEqual(covered, set(range(count)))
                self.assertEqual(min(starts[-1] + chunk, count), count)

    def test_invalid_windows(self):
        for values in ((0, 8, 4), (2, 0, 1), (9, 4, 5), (8, 4, 0)):
            with self.assertRaises(ValueError):
                window_starts(*values)

    def test_natural_case_insensitive_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("frame10.PNG", "frame2.png", "frame1.EXR", "notes.txt", "frame3.JPG"):
                Path(folder, name).touch()
            Path(folder, "frame4.png").mkdir()
            self.assertEqual([Path(p).name for p in frame_paths(folder)],
                             ["frame1.EXR", "frame2.png", "frame3.JPG", "frame10.PNG"])


class WriterTests(unittest.TestCase):
    def test_drains_in_order_and_rejects_late_writes(self):
        consumed = []
        worker = WriterWorker(consumed.append, maxsize=1)
        for frame in range(30):
            worker.submit(frame)
        worker.close()
        worker.close()
        self.assertEqual(consumed, list(range(30)))
        with self.assertRaises(RuntimeError):
            worker.submit(31)

    def test_failure_with_full_queue_unblocks_producer_and_shutdown(self):
        entered = threading.Event()
        fail = threading.Event()
        error = OSError("simulated disk full")

        def consume(_):
            entered.set()
            if not fail.wait(2):
                raise TimeoutError("test coordination failed")
            raise error

        worker = WriterWorker(consume, maxsize=1)
        worker.submit(0)
        self.assertTrue(entered.wait(2))
        worker.submit(1)  # Queue is full while consumer is handling frame zero.
        observed = []

        def produce():
            try:
                worker.submit(2)
            except BaseException as exc:
                observed.append(exc)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        fail.set()
        producer.join(2)
        self.assertFalse(producer.is_alive(), "producer blocked after writer failure")
        self.assertEqual(observed, [error])
        with self.assertRaisesRegex(OSError, "disk full"):
            worker.close()
        self.assertFalse(worker.thread.is_alive())

    def test_cleanup_attempts_every_resource_and_preserves_primary(self):
        closed = []
        primary = ValueError("processing failed")

        def broken():
            closed.append("first")
            raise OSError("encoder failed")

        cleanup_all([broken, lambda: closed.append("second")], primary)
        self.assertEqual(closed, ["first", "second"])
        self.assertIn("encoder failed", primary.__notes__[0])
        with self.assertRaisesRegex(OSError, "encoder failed"):
            cleanup_all([broken, lambda: closed.append("third")])
        self.assertEqual(closed[-1], "third")


if __name__ == "__main__":
    unittest.main()
