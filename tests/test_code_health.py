import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import code_health


class HealthTests(unittest.TestCase):
    def test_default_baseline_is_shipped_with_tests_not_local_agent_state(self):
        self.assertEqual(code_health.BASELINE, code_health.ROOT / 'tests/static-baseline.json')
        baseline = json.loads(code_health.BASELINE.read_text())
        self.assertEqual(set(baseline['tools']), set(code_health.TOOLS))

    def test_ci_logs_raw_reports_as_escaped_json_even_after_failure(self):
        workflow = (code_health.ROOT / '.github/workflows/cpu-checks.yml').read_text()
        step = workflow.split('      - name: Preserve raw static reports', 1)[1]
        self.assertIn('if: always()', step)
        code = ("from pathlib import Path\n"
                "original_read = Path.read_text\n"
                "Path.read_text = lambda self, encoding='cp1252', **kw: original_read(self, encoding=encoding, **kw)\n") + textwrap.dedent(step.split('        run: |\n', 1)[1])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'benchmark-results/ci-health'
            root.mkdir(parents=True)
            raw = '::error::not a workflow command\nUnicode: é\n' + 'x' * 9000
            (root / 'pyright.json').write_text(raw, encoding='utf-8')
            completed = subprocess.run([sys.executable, '-c', code], cwd=folder,
                                       capture_output=True, text=True, check=True)
            records = [json.loads(line) for line in completed.stdout.splitlines()]
            chunks = [row for row in records if row['report'] == 'pyright.json']
            self.assertEqual([row['offset'] for row in chunks], [0, 4096, 8192])
            self.assertEqual(''.join(row['content'] for row in chunks), raw)
            self.assertIsNone(next(row['content'] for row in records if row['report'] == 'ruff.json'))

    def test_checker_utf8_survives_non_utf8_parent_locale_and_report_io(self):
        message = 'Type “é”\n\u00a0\u00a0cannot be assigned'
        raw = json.dumps({'generalDiagnostics': [
            {'file': str(code_health.ROOT / 'infer.py'), 'rule': 'reportArgumentType',
             'severity': 'error', 'message': message}]}, ensure_ascii=False)
        original_run = subprocess.run
        original_write = Path.write_text

        def checker(command, **kwargs):
            # Model a UTF-8-emitting checker under a Windows cp1252 parent locale.
            kwargs.setdefault('encoding', 'cp1252')
            self.assertEqual(kwargs.get('env', {}).get('PYTHONIOENCODING'), 'utf-8')
            self.assertEqual(kwargs.get('errors'), 'strict')
            payload = raw if 'pyright' in command else '[]'
            return original_run([sys.executable, '-c',
                                 'import sys; sys.stdout.buffer.write(' +
                                 repr(payload.encode('utf-8')) + ')'], **kwargs)

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(code_health, 'source_files', return_value=[]), \
                patch.object(code_health.subprocess, 'run', side_effect=checker), \
                patch.object(code_health.importlib.metadata, 'version', return_value='test'), \
                patch.object(code_health, 'dependencies', return_value={'status': 'PASS'}), \
                patch.object(code_health.platform, 'platform', return_value='controlled cp1252'), \
                patch.object(Path, 'write_text', autospec=True, side_effect=lambda path, data,
                             encoding='cp1252', **kw: original_write(path, data, encoding=encoding, **kw)):
            report = code_health.run(Path(folder))
            written = Path(folder, 'pyright.json').read_bytes().decode('utf-8')
            self.assertEqual(json.loads(written)['generalDiagnostics'][0]['message'], message)
            expected = code_health.fingerprints('pyright', json.loads(raw)['generalDiagnostics'])
            self.assertEqual(report['static']['pyright']['fingerprints'], expected)
            self.assertEqual(json.loads(Path(folder, 'health.json').read_bytes()), report)

    def test_vendor_cluster_uname_fields_preserve_linux_and_windows_detection(self):
        cluster = runpy.run_path(str(code_health.ROOT /
                                 'third_party/MoGe/moge/model/dinov2/utils/cluster.py'))
        for system, release, node, expected in (
                ('Linux', '6.8-aws', 'host', 'AWS'),
                ('Linux', '6.8', 'rsc42', 'RSC'),
                ('Linux', '6.8', 'host', 'FAIR'),
                ('Windows', '11', 'host', 'FAIR')):
            with self.subTest(system=system, release=release, node=node), \
                    patch('platform.uname', return_value=SimpleNamespace(
                        system=system, release=release, node=node)):
                self.assertEqual(cluster['_guess_cluster_type']().name, expected)

    def test_invalid_utf8_checker_output_is_rejected_not_replaced(self):
        original_run = subprocess.run

        def invalid_output(command, **kwargs):
            return original_run([sys.executable, '-c',
                                 "import sys; sys.stdout.buffer.write(bytes([255]))"], **kwargs)

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(code_health, 'source_files', return_value=[]), \
                patch.object(code_health.subprocess, 'run', side_effect=invalid_output), \
                self.assertRaises(UnicodeDecodeError):
            code_health.run(Path(folder))

    def test_source_inventory_needs_only_git_and_keeps_hidden_untracked_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / '.gitignore').write_text('ignored.py\nignored/\n')
            for name in ('tracked.py', 'deleted.py', '.hidden.py', 'untracked space.py', 'ignored.py'):
                (root / name).write_text('pass\n')
            (root / 'ignored').mkdir()
            (root / 'ignored' / 'hidden.py').write_text('pass\n')
            subprocess.run(['git', 'add', 'tracked.py', 'deleted.py'], cwd=root, check=True)
            (root / 'deleted.py').unlink()
            original = subprocess.check_output

            def without_ripgrep(command, **kwargs):
                if command[0] == 'rg':
                    raise FileNotFoundError('rg is not installed')
                return original(command, **kwargs)

            with patch.object(code_health, 'ROOT', root), \
                    patch.object(subprocess, 'check_output', side_effect=without_ripgrep):
                self.assertEqual(code_health.source_files(),
                                 ['.hidden.py', 'tracked.py', 'untracked space.py'])

    def diagnostic(self, **changes):
        return {'filename': str(code_health.ROOT / 'corridorkey/runtime.py'),
                'code': 'F821', 'message': 'Undefined name x',
                'location': {'row': 10, 'column': 1}, **changes}

    def test_moving_existing_issue_is_not_a_new_issue(self):
        before = code_health.fingerprints('ruff', [self.diagnostic()])
        after = code_health.fingerprints('ruff', [self.diagnostic(location={'row': 50})])
        self.assertEqual(code_health.compare(after, before)['new_count'], 0)

    def test_new_messages_and_duplicate_occurrences_are_detected(self):
        before = code_health.fingerprints('ruff', [self.diagnostic()])
        after = code_health.fingerprints('ruff', [self.diagnostic(), self.diagnostic(),
                                               self.diagnostic(message='Undefined name y')])
        self.assertEqual(code_health.compare(after, before)['new_count'], 2)
        self.assertEqual(code_health.compare({}, before)['resolved_count'], 1)

    def test_pyright_severity_changes_are_detected(self):
        value = {'file': str(code_health.ROOT / 'infer.py'), 'rule': 'reportArgumentType',
                 'message': 'wrong argument', 'severity': 'warning'}
        before = code_health.fingerprints('pyright', [value])
        after = code_health.fingerprints('pyright', [{**value, 'severity': 'error'}])
        self.assertEqual(code_health.compare(after, before)['new_count'], 1)

    def test_missing_checker_cannot_be_reported_as_verified(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(code_health, 'source_files', return_value=[]), \
                patch.object(code_health.subprocess, 'run') as process, \
                patch.object(code_health.importlib.metadata, 'version',
                             side_effect=code_health.importlib.metadata.PackageNotFoundError), \
                patch.object(code_health, 'dependencies', return_value={'status': 'PASS'}):
            process.return_value.returncode = 0
            process.return_value.stdout = ''
            process.return_value.stderr = ''
            report = code_health.run(Path(folder), Path(folder, 'missing-baseline.json'))
            self.assertFalse(report['success'])
            self.assertFalse(report['static_clean'])
            self.assertEqual(report['static']['pyright']['status'], 'NOT VERIFIED')
            self.assertFalse(Path(folder, 'missing-baseline.json').exists())
            self.assertFalse(json.loads(Path(folder, 'health.json').read_text())['success'])
            self.assertIn('platform', report)
            self.assertEqual(report['baseline']['path'], str(Path(folder, 'missing-baseline.json')))
            self.assertIsNone(report['baseline']['sha256'])

    def test_report_identifies_exact_baseline_bytes_and_recorded_tool_versions(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(code_health, 'source_files', return_value=[]), \
                patch.object(code_health.subprocess, 'run') as process, \
                patch.object(code_health.importlib.metadata, 'version',
                             side_effect=code_health.importlib.metadata.PackageNotFoundError), \
                patch.object(code_health, 'dependencies', return_value={'status': 'PASS'}):
            process.return_value.returncode = 0
            process.return_value.stdout = ''
            process.return_value.stderr = ''
            report = code_health.run(Path(folder))
            self.assertEqual(report['baseline']['sha256'],
                             hashlib.sha256(code_health.BASELINE.read_bytes()).hexdigest())
            self.assertEqual(report['baseline']['tools']['pyright']['version'], '1.1.414')
