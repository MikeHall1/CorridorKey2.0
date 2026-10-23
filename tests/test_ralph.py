import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("ralph_loop", Path(__file__).resolve().parents[1] / "scripts/ralph_loop.py")
ralph = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ralph)


class RalphTests(unittest.TestCase):
    def test_checkpoint_records_evidence_remaining_and_next_action(self):
        plan = {"tasks": [
            {"id": "first", "status": "done", "evidence": ["test log"]},
            {"id": "next", "status": "ready", "acceptance": "next checks"},
            {"id": "gpu", "status": "external", "acceptance": "real GPU"}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "checkpoint.json")
            result = ralph.save_checkpoint(plan, cycles=5, reason="cycle limit", path=path)
            self.assertEqual(result, json.loads(path.read_text()))
            self.assertEqual(result["next_action"]["id"], "next")
            self.assertEqual(result["completed"][0]["evidence"], ["test log"])
            self.assertEqual(result["remaining"][-1]["status"], "external")
            self.assertFalse(result["product_complete"])
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_invalid_execution_budgets_are_rejected(self):
        for arguments in (["--max-iterations", "0"], ["--iteration-timeout", "nan"],
                          ["--iteration-timeout", "inf"]):
            with self.subTest(arguments=arguments), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    ralph.main(["run", *arguments])
            self.assertEqual(caught.exception.code, 2)

    def test_continuation_has_no_implicit_five_cycle_limit(self):
        with patch.object(ralph, "run", return_value=0) as run:
            self.assertEqual(ralph.main(["run"]), 0)
            self.assertIsNone(run.call_args.args[0].max_iterations)
            self.assertEqual(ralph.main(["run", "--max-iterations", "6"]), 0)
            self.assertEqual(run.call_args.args[0].max_iterations, 6)

    def test_standalone_passes_continue_past_checkpoint_without_duplicate_suites(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ralph, "ROOT", Path(folder)):
            plan = {"tasks": [{"id": "cpu", "status": "ready", "acceptance": "actual checks"}]}
            calls = []

            def pass_then_stop(*args):
                calls.append(args)
                if len(calls) == 6:
                    Path(folder, ".ralph/STOP").touch()
                return 0

            with (patch.object(ralph, "load_plan", return_value=plan),
                  patch.object(ralph.shutil, "which", return_value="codex"),
                  patch.object(ralph, "agent_pass", side_effect=pass_then_stop),
                  patch.object(ralph, "fingerprint", side_effect=(str(i) for i in range(12))),
                  patch.object(ralph, "verify") as verify,
                  patch("sys.stdout", io.StringIO())):
                self.assertEqual(ralph.main(["run"]), 130)
            self.assertEqual(len(calls), 6)
            verify.assert_not_called()
            checkpoint = json.loads(Path(folder, ".ralph/checkpoint.json").read_text())
            self.assertEqual(checkpoint["cycles_completed"], 6)
            self.assertFalse(checkpoint["product_complete"])
            self.assertFalse(Path(folder, ".ralph/runner.lock").exists())

    def test_plan_rejects_completion_without_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "plan.json")
            path.write_text(json.dumps({"schema_version": 1, "tasks": [
                {"id": "test", "status": "done", "acceptance": "test", "evidence": []}]}))
            with self.assertRaisesRegex(ValueError, "evidence"):
                ralph.load_plan(path)

    def test_external_work_is_not_silently_actionable_or_complete(self):
        plan = {"tasks": [{"id": "gpu", "status": "external"}, {"id": "cpu", "status": "ready"}]}
        self.assertEqual(ralph.next_task(plan)["id"], "cpu")
        plan["tasks"][1]["status"] = "done"
        self.assertIsNone(ralph.next_task(plan))

    def test_agent_pass_stdin_and_output_without_real_model_call(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ralph, "ROOT", Path(folder)):
            with tempfile.TemporaryFile(mode="w+") as output:
                code = ralph.agent_pass([sys.executable, "-c", "import sys; print(sys.stdin.read())"],
                                        "test instruction", output, output, 5)
                output.seek(0)
                self.assertEqual(code, 0)
                self.assertIn("test instruction", output.read())

    def test_stop_file_terminates_managed_child(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ralph, "ROOT", Path(folder)):
            Path(folder, ".ralph").mkdir()
            Path(folder, ".ralph/STOP").touch()
            with tempfile.TemporaryFile(mode="w+") as output:
                code = ralph.agent_pass([sys.executable, "-c", "import time; time.sleep(30)"],
                                        "test", output, output, 5)
            self.assertEqual(code, 130)

    def test_agent_timeout_is_not_completion(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ralph, "ROOT", Path(folder)):
            with tempfile.TemporaryFile(mode="w+") as output:
                code = ralph.agent_pass([sys.executable, "-c", "import time; time.sleep(30)"],
                                        "test", output, output, 0.01)
            self.assertEqual(code, 124)
