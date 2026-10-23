import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import code_health


class HealthTests(unittest.TestCase):
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
