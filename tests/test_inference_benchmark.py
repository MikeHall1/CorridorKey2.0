"""Benchmark orchestration tests with explicit test doubles; no neural evidence."""
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from corridorkey import benchmark
from corridorkey.session import InferenceSession

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "benchmark CUDA branches tested with doubles require optional numeric imports")
class InferenceBenchmarkTests(unittest.TestCase):
    def setUp(self):
        from corridorkey import runtime
        import torch
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        (self.root / "source.mp4").touch()
        self.batch = self.root / "batch.json"
        self.batch.write_text(json.dumps({"schema_version": 1, "kind": "inference_batch", "jobs": [
            {"input": "source.mp4", "output_dir": "original-output", "settings": {"output_format": "exr"}}]}))
        self.manifest_path = self.root / "benchmark.json"
        weights = self.root / "weights"
        weights.write_bytes(b"test weights, not a checkpoint")
        foundation = self.root / "foundation"
        foundation.mkdir()
        self.paths = SimpleNamespace(weights=weights, foundation_stats=weights, rvm_ckpt=weights,
                                     moge_checkpoint=weights, cradio_dir=foundation, validate=Mock())
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        for target, values in (
            (benchmark, {"ROOT": self.root, "repository_identity": lambda: {"test_double": True},
                         "environment": lambda: {"hardware": {}}, "command": lambda *a: b"test driver"}),
            (runtime.RuntimePaths, {"from_package": Mock(return_value=self.paths)}),
            (torch.cuda, {"is_available": Mock(return_value=True)}),
        ):
            for name, value in values.items():
                self.patches.enter_context(patch.object(target, name, value))

    def manifest(self, state):
        values = {"schema_version": 1, "kind": "inference", "batch_manifest": "batch.json",
                  "model_state": state, "repetitions": 3}
        self.manifest_path.write_text(json.dumps(values))
        return values

    def test_warm_session_warms_up_once_and_reuses_model_for_three_runs(self):
        import corridorkey.session as sessions
        loader, runner = Mock(return_value=object()), Mock()
        factory = lambda **kwargs: InferenceSession(loader=loader, runner=runner, disposer=lambda m: None)
        with patch.object(sessions, "InferenceSession", factory):
            report = benchmark.run_inference_benchmark(self.manifest_path, self.manifest("warm_session"))
        self.assertTrue(report["summary"]["success"], report.get("error"))
        self.assertEqual(len(report["warmup"]), 1)
        self.assertEqual(len(report["runs"]), 3)
        self.assertTrue(all(r["jobs"][0]["model_reused"] for r in report["runs"]))
        loader.assert_called_once()
        self.assertEqual(runner.call_count, 4)
        self.assertFalse((self.root / "original-output").exists())

    def test_cold_mode_uses_three_distinct_child_invocations_and_output_paths(self):
        manifests = []

        def process(args, **kwargs):
            self.assertEqual(args[1:3], ["-m", "corridorkey.batch"])
            batch = json.loads(Path(args[3]).read_text())
            manifests.append(batch)
            self.assertEqual(batch["jobs"][0]["input"], str(self.root / "source.mp4"))
            Path(args[5]).write_text(json.dumps({"status": "complete", "jobs": [
                {"status": "complete", "diagnostics": {"execution": {"backend": "test_double"}}}]}))
            return subprocess.CompletedProcess(args, 0, "test child", "")

        with patch.object(benchmark.subprocess, "run", side_effect=process):
            report = benchmark.run_inference_benchmark(self.manifest_path, self.manifest("cold_process"))
        self.assertTrue(report["summary"]["success"], report.get("error"))
        self.assertEqual(len(manifests), 3)
        self.assertEqual(len({m["jobs"][0]["output_dir"] for m in manifests}), 3)

    def test_warm_failure_keeps_partial_records_and_no_success_metrics(self):
        import corridorkey.session as sessions
        factory = lambda **kwargs: InferenceSession(loader=lambda: object(),
                                                    runner=Mock(side_effect=OSError("failed export")),
                                                    disposer=lambda m: None)
        with patch.object(sessions, "InferenceSession", factory):
            report = benchmark.run_inference_benchmark(self.manifest_path, self.manifest("warm_session"))
        self.assertFalse(report["summary"]["success"])
        self.assertEqual(report["warmup"][0]["status"], "failed")
        self.assertEqual(report["runs"], [])
        self.assertIsNone(report["inference"]["model_only_fps"])
