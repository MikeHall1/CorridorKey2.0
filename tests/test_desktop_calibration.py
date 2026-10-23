"""Offscreen explicit calibration UI; local identities, no real pretrained model."""
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from corridorkey.asset_inventory import AssetSelection
from corridorkey.config import InferenceConfig
from corridorkey.desktop_calibration import CalibrationDialog, calibration_view
from corridorkey.session import JobRequest


class FixtureTask:
    """UI lifecycle double only; never eligible measured production evidence."""
    def __init__(self, plan):
        self.plan = plan
        self.done = threading.Event()
        self.cancelled = 0
        self.cleanup_error = None
        self.report_path: Path | None = None
        self.revision = 0
        self.status = 'running'
        self.summary = {'groups': [], 'recommendations': []}

    def cancel(self):
        self.cancelled += 1

    def snapshot(self):
        return {'status': self.status, 'revision': self.revision, 'progress': {'completed_jobs': 0, 'total_jobs': 9},
                'case_summaries': [], 'summary': self.summary, 'report_path': str(self.report_path) if self.report_path else None,
                'cleanup_error': self.cleanup_error, 'artifact_error': None, 'error': None,
                'retained_paths': {}}

    def finish(self, *, cleanup_error=None):
        self.cleanup_error = cleanup_error
        self.status = 'failed' if cleanup_error else 'complete'
        self.revision += 1
        self.done.set()


class DesktopCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        for index in range(4):
            self.assertTrue(cv2.imwrite(str(self.source / f'{index}.png'), np.full((24, 32, 3), 100, np.uint8)))
        self.originals = {path.name: path.read_bytes() for path in self.source.iterdir()}
        self.assets = self.root / 'assets'
        for relative in ('weights/corridorkey_v2.safetensors', 'weights/foundation_stats.safetensors',
                         'vfms/rvm/pretrain_rvm_small16_256_204031069.npz', 'vfms/moge/model.pt',
                         'vfms/cradio/C-RADIOv4-SO400M/config.json',
                         'vfms/cradio/C-RADIOv4-SO400M/model.safetensors'):
            path = self.assets / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'UI identity fixture; not loadable pretrained weights')
        self.scratch = self.root / 'scratch'
        self.reports = self.root / 'reports'
        self.scratch.mkdir()
        self.reports.mkdir()
        self.job = JobRequest.create(self.source, self.root / 'original-export',
                                     InferenceConfig(output_format='exr', num_frames=4))
        self.dialog = CalibrationDialog(self.job, frame=1, selection=AssetSelection(self.assets), factory=FixtureTask)
        self.dialog.timer.stop()
        self.dialog.frame_count.setValue(2)
        self.dialog.scratch_parent.setText(str(self.scratch))
        self.dialog.report_parent.setText(str(self.reports))

    def task(self) -> FixtureTask:
        task = self.dialog.task
        assert isinstance(task, FixtureTask)
        return task

    def inventory(self):
        inventory = self.dialog.inventory
        assert isinstance(inventory, dict)
        return inventory

    def tearDown(self):
        if self.dialog.task is not None:
            self.task().finish()
        self.dialog.shutdown()
        self.wait(lambda: not self.dialog.io_busy)
        self.dialog.deleteLater()
        self.app.processEvents()
        self.assertEqual(self.originals, {path.name: path.read_bytes() for path in self.source.iterdir()})
        self.assertFalse(Path(self.job.output_dir).exists())
        self.temp.cleanup()

    def wait(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.dialog.poll()
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.005)
        self.fail('Owned calibration operation did not retire in time.')

    def capture(self):
        self.dialog.capture_metadata()
        self.wait(lambda: not self.dialog.io_busy)
        self.assertIsNotNone(self.dialog.inventory, self.dialog.status.text())

    def verify(self):
        self.dialog.read_budget.setValue(64)
        self.capture()
        self.dialog.verify()
        self.wait(lambda: not self.dialog.io_busy)
        self.assertIsNotNone(self.dialog.prepared_plan, self.dialog.status.text())
        return self.dialog.prepared_plan

    def test_open_no_io_then_actual_metadata_zero_budget_verify_and_explicit_workload(self):
        from corridorkey.calibration import inventory_calibration, plan_calibration
        with patch('corridorkey.calibration.inventory_calibration', side_effect=AssertionError('opening must not inspect')), patch('corridorkey.calibration.plan_calibration', side_effect=AssertionError('opening must not hash')):
            unopened = CalibrationDialog(self.job, selection=AssetSelection(self.assets), factory=FixtureTask)
            unopened.timer.stop()
            self.assertEqual(unopened.read_budget.value(), 0)
            self.assertIsNone(unopened.plan)
            self.assertTrue(unopened.shutdown())
            unopened.deleteLater()
        self.assertTrue(callable(inventory_calibration))
        self.assertTrue(callable(plan_calibration))
        self.capture()
        self.assertIn('9 actual jobs / 18 frames', self.dialog.workload.text())
        self.assertGreater(self.inventory()['read_bytes_required'], 0)
        self.dialog.verify()
        self.wait(lambda: not self.dialog.io_busy)
        self.assertIsNone(self.dialog.plan)
        self.assertIn('read budget', self.dialog.status.text())
        self.assertFalse(self.dialog.start_button.isEnabled())
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertEqual(list(self.reports.iterdir()), [])

    def test_actual_verified_plan_signal_wait_begin_cancel_no_recursive_request_or_settings_change(self):
        plan = self.verify()
        before = self.job.config_json
        requests, cancels = [], []
        self.dialog.run_requested.connect(requests.append)
        self.dialog.cancel_requested.connect(lambda: (cancels.append(True), self.dialog.cancel()))
        self.dialog.request_start()
        self.assertEqual(requests, [plan])
        self.assertTrue(self.dialog.waiting)
        self.assertIsNone(self.dialog.task)
        self.assertFalse(self.dialog.verify_button.isEnabled())
        self.dialog.begin(plan)
        self.assertFalse(self.dialog.waiting)
        self.assertIs(self.task().plan, plan)
        self.dialog.cancel()  # Programmatic cancellation must not emit a recursive request.
        self.assertEqual(cancels, [])
        self.dialog.cancel_button.click()
        self.assertEqual(cancels, [True])
        self.assertGreaterEqual(self.task().cancelled, 1)
        self.assertTrue(self.dialog.busy)
        self.assertEqual(self.job.config_json, before)
        self.task().finish()
        self.dialog.poll()
        self.assertFalse(self.dialog.busy)
        self.assertIn('NOT VERIFIED', self.dialog.details.toPlainText())

    def test_profile_edits_invalidate_verified_identity_and_unsupported_source_in_are_precise(self):
        self.verify()
        self.dialog.profiles.setPlainText('[{"id":"draft","purpose":"preview","overrides":{"model_scale":0.5,"hann_chunk":4,"hann_stride":2,"stage2_batch":2,"low_vram":true,"foundation_cache_mb":0,"quick_key":false}}]')
        self.assertIsNone(self.dialog.plan)
        self.assertIsNone(self.dialog.capture)
        self.assertFalse(self.dialog.start_button.isEnabled())
        self.dialog.capture_metadata()
        self.wait(lambda: not self.dialog.io_busy)
        self.assertEqual(self.inventory()['profiles'][0]['purpose'], 'preview')
        self.assertIn('model_scale=0.5', self.dialog.details.toPlainText())
        self.dialog.sample_start.setText(str(2**40))
        self.dialog.capture_metadata()
        self.wait(lambda: not self.dialog.io_busy)
        self.assertIsNone(self.dialog.capture)
        self.assertNotEqual(self.dialog.sample_start.text(), '0')
        self.assertFalse(self.dialog.start_button.isEnabled())
        self.assertIn('frame', self.dialog.status.text().lower())

    def test_cancel_late_verification_never_enables_start_and_close_retains_thread(self):
        from corridorkey.calibration import plan_calibration
        self.dialog.read_budget.setValue(64)
        self.capture()
        committed, release = threading.Event(), threading.Event()

        def blocked(*args, **kwargs):
            result = plan_calibration(*args, **kwargs)
            committed.set()
            self.assertTrue(release.wait(5))
            return result

        with patch('corridorkey.calibration.plan_calibration', side_effect=blocked):
            self.dialog.verify()
            self.assertTrue(committed.wait(5))
            self.dialog.close()
            self.assertFalse(self.dialog.shutdown())
            self.assertIsNone(self.dialog.plan)
            self.assertTrue(self.dialog.io_busy)
            release.set()
            self.wait(lambda: not self.dialog.io_busy)
        self.assertIsNone(self.dialog.plan)
        self.assertFalse(self.dialog.start_button.isEnabled())
        self.assertTrue(self.dialog.shutdown())

    def test_worker_cleanup_uncertainty_retains_owner_while_artifact_error_is_separate(self):
        plan = self.verify()
        self.dialog.request_start()
        self.dialog.begin(plan)
        owner = self.task()
        owner.finish(cleanup_error='descendant retirement NOT VERIFIED')
        self.dialog.poll()
        self.assertIs(self.dialog.task, owner)
        self.assertFalse(self.dialog.shutdown())
        self.assertIn('owner retained', self.dialog.status.text())
        self.assertFalse(self.dialog.capture_button.isEnabled())
        owner.finish()
        self.assertTrue(self.dialog.shutdown())

    def test_report_expiry_recommendations_duration_and_exclusions_are_readonly(self):
        plan = self.verify()
        self.dialog.request_start()
        self.dialog.begin(plan)
        report = self.reports / 'actual-ui-fixture-report.json'
        report.write_text(json.dumps({'schema_version': 1, 'type': 'corridorkey_calibration_report',
            'status': 'complete', 'created_utc': (datetime.now(UTC) - timedelta(days=8)).isoformat(),
            'binding_identity': 'fixture-not-production-evidence',
            'plan': {'requested_job': {'config_json': self.job.config_json},
                     'identities': {'source': [None] * 4}, 'summary': {'frame_count': 2}},
            'summary': {'recommendations': [{'purpose': 'preview', 'profile': 'fixture', 'median_fps': 2.,
                                           'sample_seconds_range': [.8, 1.2], 'basis': 'UI fixture only; not actual GPU measurement',
                                           'estimated_render_duration': {'status': 'NOT VERIFIED', 'frames': 4,
                                               'min_seconds': None, 'central_seconds': None, 'max_seconds': None,
                                               'scope': 'Fixture has no eligible CUDA measurements; no estimate.'}}]},
            'trials': [{'profile': 'fixture', 'state': 'warm_session', 'eligible': False,
                        'error': 'CPU fixture excluded from inference recommendation'},
                       {'profile': 'not-started', 'state': 'cold_process', 'eligible': False, 'status': 'not_run',
                        'exclusion_reason': 'Global prerequisite failed; remaining calibration trial was not started'}]}))
        self.task().report_path = report
        self.task().finish()
        self.dialog.poll()
        before = self.job.config_json
        self.dialog.inspect_report()
        self.wait(lambda: not self.dialog.io_busy)
        text = self.dialog.details.toPlainText()
        self.assertIn('Expired', text)
        self.assertIn('NOT been reverified', text)
        self.assertIn('render duration status: NOT VERIFIED', text)
        self.assertIn('CPU fixture excluded', text)
        self.assertIn('Global prerequisite failed; remaining calibration trial was not started', text)
        self.assertIn('no estimate', text)
        self.assertEqual(self.job.config_json, before)
        self.assertFalse(hasattr(self.dialog, 'apply_requested'))

    def test_clean_idle_shutdown_hides_visible_dialog(self):
        self.dialog.show()
        self.app.processEvents()
        self.assertTrue(self.dialog.isVisible())
        self.assertTrue(self.dialog.shutdown())
        self.app.processEvents()
        self.assertFalse(self.dialog.isVisible())

    def test_compact_formatter_never_serializes_large_records_or_arrays(self):
        class NoRead(list):
            def __iter__(self):
                raise AssertionError('Full trial arrays must not be traversed on GUI thread')
        rendered = calibration_view({'status': 'complete', 'trials': NoRead([{}] * 100000),
                                     'requested_job': {'giant': 'x' * 1000000},
                                     'case_summaries': [{'profile': 'last', 'state': 'warm_session', 'error': 'x' * 100000}] * 72})
        self.assertLessEqual(len(rendered.encode()), 64 * 1024)
        self.assertIn('last', rendered)
        self.assertNotIn('giant', rendered)


if __name__ == '__main__':
    unittest.main()
