import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import verify


class VerifyTests(unittest.TestCase):
    def test_cli_without_agent_plan_runs_all_existing_gates_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'benchmark-results/verification'
            completed = [subprocess.CompletedProcess([], code, f'gate {index}\n', 'details\n')
                         for index, code in enumerate((1, 0, 0, 0))]
            with patch.object(verify, 'ROOT', root), \
                    patch.object(verify.subprocess, 'run', side_effect=completed) as process, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify.main(['--python', 'selected-python', '--output-dir', str(output)]), 1)
            commands = [call.args[0] for call in process.call_args_list]
            self.assertEqual(commands, [
                ['selected-python', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                ['selected-python', '-m', 'compileall', '-q', 'corridorkey', 'scripts', 'tests',
                 'infer.py', 'run_interactive.py'],
                ['selected-python', 'scripts/code_health.py', '--output-dir', str(output / 'code-health')],
                ['git', 'diff', '--check']])
            self.assertTrue(all(call.kwargs['cwd'] == root for call in process.call_args_list))
            self.assertTrue(all(call.kwargs['timeout'] == 1800 for call in process.call_args_list))
            self.assertEqual((output / 'check-0.txt').read_text(), 'gate 0\ndetails\n')
            report = json.loads((output / 'verification.json').read_text())
            self.assertFalse(report['success'])
            self.assertEqual(len(report['checks']), 4)
            self.assertFalse((root / '.ralph').exists())

    def test_successful_gates_return_success(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(verify.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')):
            self.assertTrue(verify.verify('selected-python', Path(folder))['success'])

    def test_timed_out_gate_retains_partial_output_and_all_four_gate_results(self):
        for stdout, stderr, expected in ((b'partial stdout\n', b'partial stderr\n',
                                          'partial stdout\npartial stderr\n'),
                                         ('text output', 'text error', 'text outputtext error'),
                                         (None, None, '')):
            with self.subTest(stdout=stdout), tempfile.TemporaryDirectory() as folder:
                timeout = subprocess.TimeoutExpired(['unit tests'], 1800, output=stdout, stderr=stderr)
                passed = subprocess.CompletedProcess([], 0, 'later gate\n', '')
                with patch.object(verify.subprocess, 'run', side_effect=[timeout, passed, passed, passed]):
                    report = verify.verify('selected-python', Path(folder))
                self.assertFalse(report['success'])
                self.assertEqual(len(report['checks']), 4)
                self.assertEqual(report['checks'][0]['status'], 'TIMEOUT')
                self.assertIsNone(report['checks'][0]['returncode'])
                self.assertEqual(report['checks'][0]['timeout_seconds'], 1800)
                self.assertEqual(Path(folder, 'check-0.txt').read_text(), expected)
                self.assertEqual(Path(folder, 'check-3.txt').read_text(), 'later gate\n')
                self.assertEqual(json.loads(Path(folder, 'verification.json').read_text()), report)
