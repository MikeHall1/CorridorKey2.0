from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.diagnostics import RunDiagnostics, write_json_atomic
from corridorkey.session import InferenceSession, JobRequest
from corridorkey.batch import load_jobs, run_batch
from corridorkey.control import CancellationToken, CancelledError


class JobFixture(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.source = self.root / "source.mp4"
        self.source.touch()
        self.config = InferenceConfig(output_format="exr")

    def job(self, name="out", config=None):
        return JobRequest.create(self.source, self.root / name, config or self.config)


class SessionTests(JobFixture):
    def test_colour_refinement_options_reach_shared_session_without_implicit_modes(self):
        import infer
        runner = Mock()
        with InferenceSession(loader=lambda: object(), runner=runner, disposer=lambda model: None) as session:
            infer.main(["--input", str(self.source), "--output_dir", str(self.root / "out"),
                        "--output_format", "exr", "--outputs", "alpha", "cutout",
                        "--cutout_alpha_mode", "premultiplied", "--alpha_refine_radius", "1",
                        "--despill_space", "linear", "--despill_screen", "blue",
                        "--despill_preserve_luma", "--despill_edges_only"], session=session)
        config = runner.call_args.args[2]
        self.assertEqual(config.frame_dir_alpha, "ignore")
        self.assertEqual(config.cutout_alpha_mode, "premultiplied")
        self.assertEqual(config.alpha_refine_radius, 1)
        self.assertEqual((config.despill_screen, config.despill_space), ("blue", "linear"))
        self.assertTrue(config.despill_preserve_luma)
        self.assertTrue(config.despill_edges_only)
    def test_invalid_correction_preflight_does_not_load_or_discard_healthy_model(self):
        from test_corrections import document_data, raster_entry
        from corridorkey.identities import file_sha256
        mask = self.root / "mask.png"
        mask.write_bytes(b"hash transport fixture, not a decoded image")
        data = document_data()
        data["entries"] = [raster_entry()]
        data["entries"][0]["asset"]["sha256"] = file_sha256(mask)
        project = self.root / "corrections.json"
        project.write_text(json.dumps(data))
        job = self.job(config=replace(self.config, corrections_path=project))
        loader, dispose = Mock(return_value=object()), Mock()
        with InferenceSession(loader=loader, runner=Mock(), disposer=dispose) as session:
            with self.assertRaisesRegex(ValueError, "source content"):
                session.run(job)
            loader.assert_not_called()
            session.run(self.job())
            with self.assertRaisesRegex(ValueError, "source content"):
                session.run(job)
            dispose.assert_not_called()
            self.assertIsNotNone(session.last_result.to_dict()["diagnostics"]["correction_preflight_seconds"])
            data["source"]["sha256"] = file_sha256(self.source)
            project.write_text(json.dumps(data))
            result = session.run(job)
            self.assertEqual(result.to_dict()["diagnostics"]["corrections"]["revision"], 0)
            mask.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "asset hash"):
                session.run(job)
            loader.assert_called_once()

    def test_pre_cancelled_request_does_not_load_model(self):
        token = CancellationToken()
        token.cancel()
        loader = Mock()
        with InferenceSession(loader=loader, runner=Mock(), disposer=lambda model: None) as session:
            with self.assertRaises(CancelledError):
                session.run(self.job(), cancellation=token)
            self.assertEqual(session.last_result.to_dict()["status"], "cancelled")
        loader.assert_not_called()

    def test_token_cancellation_after_load_discards_model_and_records_no_success_fps(self):
        token, dispose = CancellationToken(), Mock()
        events = []

        def progress(event):
            events.append(event.stage)
            if event.stage == "model_load" and event.completed == 1:
                token.cancel()

        with InferenceSession(loader=lambda: object(), runner=Mock(), disposer=dispose) as session:
            with self.assertRaises(CancelledError):
                session.run(self.job(), cancellation=token, progress=progress)
            record = session.last_result.to_dict()
        self.assertIn("model_load", events)
        self.assertEqual(record["status"], "cancelled")
        self.assertIsNone(record["end_to_end_fps"])
        dispose.assert_called_once()
    def test_two_jobs_load_once_and_snapshots_are_independent(self):
        model, loader, dispose = object(), Mock(), Mock()
        loader.return_value = model
        seen = []

        def runner(loaded, job, cfg, writer, diag):
            self.assertIs(loaded, model)
            seen.append(cfg.hann_chunk)
            cfg.hann_chunk = 1
            diag.effective = {"hann_chunk": 1}
            diag.frames_submitted = 3
            diag.frame_ready()

        job = self.job()
        self.config.hann_chunk = 999  # Request must retain the earlier settings.
        with InferenceSession(loader=loader, runner=runner, disposer=dispose) as session:
            first = session.run(job)
            second = session.run(job)
        loader.assert_called_once()
        dispose.assert_called_once_with(model)
        self.assertEqual(seen, [80, 80])
        self.assertFalse(first.to_dict()["model_reused"])
        self.assertIsNone(first.to_dict()["model_load_seconds"], "a test loader must not invent model timings")
        self.assertGreaterEqual(first.to_dict()["model_initialization_seconds"], 0)
        self.assertTrue(second.to_dict()["model_reused"])
        self.assertEqual(second.to_dict()["model_load_seconds"], 0)
        self.assertEqual(first.to_dict()["requested_settings"]["hann_chunk"], 80)
        copy = first.to_dict()
        copy["requested_settings"]["hann_chunk"] = 123
        self.assertEqual(first.to_dict()["requested_settings"]["hann_chunk"], 80)

    def test_inference_failure_discards_and_reloads_without_masking_error(self):
        loader, runner, dispose = Mock(side_effect=[object(), object()]), Mock(), Mock()
        error = OSError("disk full")
        runner.side_effect = [error, None]
        dispose.side_effect = [RuntimeError("cleanup failed"), None]
        with InferenceSession(loader=loader, runner=runner, disposer=dispose) as session:
            with self.assertRaisesRegex(OSError, "disk full"):
                session.run(self.job())
            failed = session.last_result.to_dict()
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["end_to_end_fps"])
            self.assertIn("cleanup failed", failed["error"]["notes"][0])
            session.run(self.job("next"))
        self.assertEqual(loader.call_count, 2)

    def test_bad_source_rejected_before_loading_and_does_not_discard_healthy_model(self):
        loader, runner, dispose = Mock(return_value=object()), Mock(), Mock()
        with InferenceSession(loader=loader, runner=runner, disposer=dispose) as session:
            absent = replace(self.job(), input_path=str(self.root / "absent"))
            with self.assertRaises(FileNotFoundError):
                session.run(absent)
            loader.assert_not_called()
            session.run(self.job())
            with self.assertRaises(FileNotFoundError):
                session.run(absent)
            dispose.assert_not_called()
            session.run(self.job())
            loader.assert_called_once()

    def test_cancelled_job_is_recorded_and_not_reused(self):
        dispose = Mock()
        with InferenceSession(loader=lambda: object(), runner=Mock(side_effect=KeyboardInterrupt),
                              disposer=dispose) as session:
            with self.assertRaises(KeyboardInterrupt):
                session.run(self.job())
            self.assertEqual(session.last_result.to_dict()["status"], "cancelled")
        dispose.assert_called_once()

    def test_reject_concurrent_requests_and_close_while_busy(self):
        entered, resume = threading.Event(), threading.Event()

        def runner(*args):
            entered.set()
            if not resume.wait(3):
                raise TimeoutError("test coordination failed")

        errors = []
        with InferenceSession(loader=lambda: object(), runner=runner, disposer=lambda model: None) as session:
            def run():
                try:
                    session.run(self.job())
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaisesRegex(RuntimeError, "busy"):
                    session.run(self.job())
                with self.assertRaisesRegex(RuntimeError, "busy"):
                    session.close()
            finally:
                resume.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.run(self.job())

    def test_cli_and_launcher_share_one_session(self):
        import infer
        from run_interactive import execute_inference
        loader, runner = Mock(return_value=object()), Mock()
        report = self.root / "job.json"
        args = ["--input", str(self.source), "--output_dir", str(self.root / "out"),
                "--output_format", "exr", "--diagnostics", str(report)]
        with InferenceSession(loader=loader, runner=runner, disposer=lambda model: None) as session:
            infer.main(args, session=session)
            self.assertEqual(execute_inference([sys.executable, "infer.py"] + args, session), 0)
        loader.assert_called_once()
        self.assertEqual(runner.call_count, 2)
        self.assertTrue(json.loads(report.read_text())["model_reused"])

    def test_cli_passes_explicit_cache_and_auto_budget_to_shared_core(self):
        import infer
        runner = Mock()
        with InferenceSession(loader=lambda: object(), runner=runner, disposer=lambda model: None) as session:
            infer.main(["--input", str(self.source), "--output_dir", str(self.root / "output"),
                        "--output_format", "exr", "--foundation_cache_mb", "64", "--auto_vram_budget_mb", "8192"],
                       session=session)
        config = runner.call_args.args[2]
        self.assertTrue(config.auto_vram)
        self.assertEqual(config.foundation_cache_mb, 64)
        self.assertEqual(config.auto_vram_budget_mb, 8192)

    def test_cli_quick_settings_are_snapshotted_by_shared_session(self):
        import infer
        runner = Mock()
        with InferenceSession(loader=lambda: object(), runner=runner, disposer=lambda model: None) as session:
            infer.main(["--input", str(self.source), "--output_dir", str(self.root / "output"),
                        "--output_format", "exr", "--quick_key", "--quick_interval", "3",
                        "--quick_context_radius", "1", "--quick_cut_frames", "8", "12",
                        "--quick_fallback", "neural", "--quick_auto_cuts", "--quick_adaptive",
                        "--quick_max_gap", "2", "--quick_approach", "causal"], session=session)
        config = runner.call_args.args[2]
        self.assertTrue(config.quick_key)
        self.assertEqual(config.quick_interval, 3)
        self.assertEqual(config.quick_cut_frames, [8, 12])
        self.assertEqual(config.quick_fallback, "neural")
        self.assertTrue(config.quick_auto_cuts)
        self.assertTrue(config.quick_adaptive)
        self.assertEqual(config.quick_max_gap, 2)
        self.assertEqual(config.quick_approach, "causal")

    def test_cli_records_failure_without_overwriting_it_with_report_error(self):
        import infer
        with InferenceSession(loader=lambda: object(), runner=Mock(side_effect=OSError("failed render")),
                              disposer=lambda model: None) as session:
            with patch.object(infer, "write_json_atomic", side_effect=OSError("report disk full")):
                with self.assertRaisesRegex(OSError, "failed render") as caught:
                    infer._run_job(session, self.job(), self.root / "job.json")
            self.assertIn("report disk full", caught.exception.__notes__[0])

    def test_bad_serialized_config_does_not_lock_session(self):
        with InferenceSession(loader=lambda: object(), runner=Mock(), disposer=lambda model: None) as session:
            with self.assertRaises(json.JSONDecodeError):
                session.run(replace(self.job(), config_json="broken"))
            session.run(self.job())


class DiagnosticTests(unittest.TestCase):
    def test_stage_sync_is_opt_in_and_cleanup_scope_can_skip_gpu(self):
        sync = Mock()
        clock = iter([0, 1, 2, 3, 4, 5])
        diag = RunDiagnostics(profile_stages=True, clock=lambda: next(clock))
        diag.synchronize = sync
        with diag.stage("compute"):
            pass
        with diag.stage("cleanup", gpu=False):
            pass
        self.assertEqual(sync.call_count, 2)
        self.assertEqual([s["seconds"] for s in diag.stages], [1, 1])
        quiet = RunDiagnostics()
        quiet.synchronize = sync
        with quiet.stage("ordinary_run"):
            pass
        self.assertEqual(quiet.stages, [])
        self.assertEqual(sync.call_count, 2)

    def test_atomic_report_failure_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "report.json")
            write_json_atomic(path, {"status": "complete"})
            with self.assertRaises(ValueError):
                write_json_atomic(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["report.json"])


class BatchTests(JobFixture):
    def test_manifest_paths_aliases_and_invalid_fields(self):
        path = self.root / "batch.json"
        values = {"schema_version": 1, "kind": "inference_batch", "jobs": [
            {"input": "source.mp4", "output_dir": "out", "settings": {
                "outputs": ["foreground"], "output_format": "exr", "temp_dir": "cache",
                "foundation_cache_mb": 64, "auto_vram": True, "auto_vram_budget_mb": 8192}}]}
        path.write_text(json.dumps(values))
        jobs, weights = load_jobs(path)
        self.assertIsNone(weights)
        self.assertEqual(jobs[0].input_path, str(self.source))
        self.assertEqual(jobs[0].configurations()[0].outputs, {"fg"})
        self.assertEqual(jobs[0].configurations()[0].temp_dir, self.root / "cache")
        self.assertEqual(jobs[0].configurations()[0].foundation_cache_mb, 64)
        self.assertEqual(jobs[0].configurations()[0].auto_vram_budget_mb, 8192)
        values["jobs"].append(values["jobs"][0])
        path.write_text(json.dumps(values))
        with self.assertRaisesRegex(ValueError, "distinct output_dir"):
            load_jobs(path)
        values["jobs"] = [{"input": "source.mp4", "output_dir": "out", "mistyped": True}]
        path.write_text(json.dumps(values))
        with self.assertRaisesRegex(ValueError, "unknown"):
            load_jobs(path)

    def test_batch_records_completed_and_failed_jobs_and_stops(self):
        loader = Mock(return_value=object())
        runner = Mock(side_effect=[None, OSError("encoder failed"), None])
        report = self.root / "report.json"
        factory = lambda **kwargs: InferenceSession(loader=loader, runner=runner, disposer=lambda m: None)
        with self.assertRaisesRegex(OSError, "encoder failed"):
            run_batch([self.job("one"), self.job("two"), self.job("three")],
                      report_path=report, session_factory=factory)
        data = json.loads(report.read_text())
        self.assertEqual(data["status"], "failed")
        self.assertEqual([r["status"] for r in data["jobs"]], ["complete", "failed"])
        self.assertEqual(runner.call_count, 2)
        loader.assert_called_once()
