import json
import sys
import tempfile
import unittest
from pathlib import Path

from corridorkey.desktop_smoke import smoke


class DesktopSmokeTests(unittest.TestCase):
    def test_actual_dispatcher_backend_and_worker_lifetime_submit_no_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            report = smoke([sys.executable, '-m', 'corridorkey.app_process'], Path(folder))
            data = json.loads(report.read_text())
            self.assertEqual(data['status'], 'passed', data)
            self.assertTrue(data['clean_shutdown'])
            self.assertEqual(data['worker_status']['payload']['pending'], [])
            self.assertIsNone(data['worker_status']['payload']['active'])
            self.assertIsNone(data['cleanup_error'])

    def test_failure_retains_report_without_replacing_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            old = parent / 'report.json'
            old.write_text('existing')
            report = smoke([str(parent / 'missing-executable')], parent)
            self.assertEqual(json.loads(report.read_text())['status'], 'failed')
            self.assertEqual(old.read_text(), 'existing')
