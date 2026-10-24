"""Explicit optional Stage 2 policy and truthful current-attempt report display."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from tests.qt_cleanup import dispose_widget

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class DesktopRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['recovery-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_recovery import RecoveryPanel
        self.panel = RecoveryPanel()
        self.received = []
        self.panel.run_requested.connect(self.received.append)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.panel.batch.setText(str(Path(self.folder.name) / 'batch.json'))
        self.panel.workspace.setText(str(Path(self.folder.name) / 'cache'))

    def tearDown(self):
        dispose_widget(self.panel)

    def test_default_independent_explicit_request_captures_stage2_policy(self):
        self.assertFalse(self.panel.stage2.isChecked())
        self.panel.stage2.setChecked(True)
        self.assertEqual(self.received, [])
        self.panel.request(False)
        self.assertEqual(len(self.received), 1)
        captured = self.received[0]
        self.assertTrue(captured['durable_stage2'])
        self.assertFalse(captured['resume'])
        for flag in ('durable_foundations', 'durable_rvm', 'durable_hann'):
            self.assertFalse(captured[flag])
        self.panel.stage2.setChecked(False)
        self.assertTrue(captured['durable_stage2'])
        self.panel.request(True)
        self.assertTrue(self.received[1]['resume'])
        self.assertFalse(self.received[1]['durable_stage2'])

    def test_finishing_reuse_is_explicit_and_requires_stage2(self):
        self.assertFalse(self.panel.finishing_edits.isChecked())
        self.panel.finishing_edits.setChecked(True)
        self.panel.request(False)
        self.assertEqual(self.received, [])
        self.assertIn('requires', self.panel.status.text())
        self.panel.stage2.setChecked(True)
        self.panel.request(False)
        payload = self.received[0]
        self.assertTrue(payload['reuse_stage2_finishing_edits'])
        self.panel.finishing_edits.setChecked(False)
        self.panel.load_payload(payload)
        self.assertTrue(self.panel.finishing_edits.isChecked())
        self.panel.load_payload({**payload, 'reuse_stage2_finishing_edits': False})
        self.assertFalse(self.panel.finishing_edits.isChecked())
        legacy = dict(payload)
        del legacy['reuse_stage2_finishing_edits']
        self.panel.load_payload(legacy)
        self.assertFalse(self.panel.finishing_edits.isChecked())
        self.assertEqual(len(self.received), 1)

    def test_loading_legacy_and_new_saved_settings_never_submits(self):
        payload = {'batch': self.panel.batch.text(), 'workspace': self.panel.workspace.text(),
                   'max_cache_bytes': 0, 'durable_stage2': True}
        self.panel.replay_summary.setText('Old report counters')
        self.panel.load_payload(payload)
        self.assertEqual(self.panel.replay_summary.text(), 'No Stage 2 recovery report inspected.')
        self.assertTrue(self.panel.stage2.isChecked())
        self.assertEqual(self.received, [])
        del payload['durable_stage2']
        self.panel.load_payload(payload)
        self.assertFalse(self.panel.stage2.isChecked())
        self.assertEqual(self.panel.budget.text(), '0')
        self.assertEqual(self.received, [])

    def test_report_current_replay_counts_exclude_historical_origin(self):
        origin = {'diagnostics': {'execution': {'stage2_calls': 2, 'stage2_frame_evaluations': 4,
                                               'completed_prediction_frames': 10},
                                  'durable_stage2': {'batch_hits': 3, 'stage2_frame_instances_reused': 6,
                                                    'stored_batches': 2, 'corrupt_entries': 1,
                                                    'budget_rejected_batches': 4, 'bytes_loaded': 240}}}
        report = {'jobs': [{'mode': 'render', 'new_session_run': True, 'origin_record': origin},
                           {'mode': 'verified_cache_restore', 'new_session_run': False, 'origin_record': origin}]}
        self.panel.results.put(({'report': report, 'scope': 'Saved report; no new validation.'}, None))
        self.panel.poll()
        text = self.panel.replay_summary.text()
        self.assertIn('Actual Stage 2 model calls: 2', text)
        self.assertIn('Frames evaluated by Stage 2: 4', text)
        self.assertIn('Predictions completed (computed + replayed): 10', text)
        self.assertIn('Verified batches reused: 3', text)
        self.assertIn('Prediction frame instances reused: 6', text)
        self.assertIn('Batches skipped by cache/staging limits: 4', text)
        self.assertIn('Completed output jobs verified/restored: 1', text)
        self.assertIn('origin counters are historical', text)
        self.assertEqual(self.received, [])

    def test_failed_attempt_and_legacy_missing_counters_never_imply_zero_compute(self):
        from corridorkey.desktop_recovery import stage2_summary
        report = {'jobs': [], 'failed_attempt': {'diagnostics': {
            'execution': {'stage2_calls': 1, 'stage2_frame_evaluations': 2},
            'durable_stage2': {'batch_hits': 0, 'stage2_frame_instances_reused': 0}}}}
        text = stage2_summary(report)
        self.assertIn('Actual Stage 2 model calls: 1', text)
        self.assertIn('Verified batches reused: 0', text)
        self.assertNotIn('Predictions completed', text)
        self.assertIn('No current-attempt Stage 2 counters recorded.', stage2_summary({'jobs': []}))
