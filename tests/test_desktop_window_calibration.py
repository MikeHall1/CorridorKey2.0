"""Actual window/child retirement; controlled task fixtures are not measurements."""
import copy
import importlib.util
import threading
import time
import unittest
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import test_calibration as calibration_fixtures
import test_desktop_window as window_fixtures


class ControlledTask:
    """Retained Python control fixture; no inference or performance results."""
    def __init__(self, plan):
        self.plan = plan
        self.done = threading.Event()
        self.cleanup_error: str | None = None
        self.artifact_error = None
        self.report_path = None
        self.revision = 0
        self.cancel_count = 0

    def cancel(self):
        self.cancel_count += 1
        self.revision += 1
        self.done.set()

    def snapshot(self) -> dict[str, Any]:
        return {'revision': self.revision, 'status': 'fixture control only; no measurements',
                'cleanup_error': self.cleanup_error, 'artifact_error': None, 'summary': {},
                'report_path': None, 'retained_paths': {}, 'progress': {}}


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt binding unavailable')
class CalibrationWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        window_fixtures.DesktopWindowTests.setUpClass()

    def setUp(self):
        self.fixture = window_fixtures.DesktopWindowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window, self.root, self.app = self.fixture.window, self.fixture.root, self.fixture.app
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.tasks = []
        self.start_observations = []

    def until(self, predicate):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.01)
        panel = self.window.calibration
        self.fail('Calibration condition not reached: ' + (panel.status.text() if panel is not None else 'no panel'))

    def install_panel(self, *, default_selection=False, failing_factory=False):
        from corridorkey.calibration import plan_calibration
        from corridorkey.desktop_calibration import CalibrationDialog
        folder = self.root / 'calibration-data'
        folder.mkdir()
        job, options = calibration_fixtures.calibration_fixture(folder)
        plan = plan_calibration(job, **options)
        selection = options['asset_selection']
        self.window.populate_job(job, frame=1)
        if default_selection:
            # Simulate a source checkout with its normal default asset tree;
            # real tiny files are verified, never deserialized as model weights.
            for module in ('corridorkey.desktop_window', 'corridorkey.desktop_calibration'):
                self.patches.enter_context(patch(module + '.__file__', str(selection.root / 'corridorkey' / 'module.py')))
            self.window.asset_selection = None
        else:
            self.window.asset_selection = selection

        def factory(received):
            self.start_observations.append((received, self.window.bridge.done.is_set(),
                                            self.window.bridge.cleanup_error, self.window.progressive.busy))
            if failing_factory:
                raise ValueError('fixture task-start failure')
            task = ControlledTask(received)
            self.tasks.append(task)
            return task

        panel = CalibrationDialog(job, self.window, frame=1,
                                  selection=None if default_selection else selection, factory=factory)
        panel.plan = plan
        panel.run_requested.connect(self.window.start_calibration)
        panel.cancel_requested.connect(self.window.cancel_calibration)
        self.window.calibration = panel
        return panel, plan, job

    def start_resident(self):
        self.window.bridge.send('status', {})
        self.until(lambda: self.window.backend.text().startswith('Worker PID'))
        self.assertFalse(self.window.bridge.done.is_set())

    def test_opening_dialog_does_no_io_render_or_history_mutation(self):
        from PySide6.QtWidgets import QPushButton

        from corridorkey.worker_protocol import job_to_wire
        panel, _, _ = self.install_panel()
        panel.shutdown()
        panel.deleteLater()
        self.window.calibration = None
        original = job_to_wire(self.window.job())
        with patch('corridorkey.calibration.inventory_calibration', side_effect=AssertionError('No opening I/O')), \
                patch('corridorkey.calibration.plan_calibration', side_effect=AssertionError('No opening hashes')):
            button = next(button for button in self.window.findChildren(QPushButton) if button.text() == 'Calibrate performance…')
            button.click()
            self.app.processEvents()
            opened = self.window.calibration
            assert opened is not None
            self.assertTrue(opened.isVisible())
            self.assertFalse(opened.io_busy)
            self.assertIsNone(opened.task)
            self.assertIsNone(opened.prepared_plan)
        self.assertEqual(job_to_wire(self.window.job()), original)
        self.assertFalse(self.window.jobs)

    def test_default_assets_start_only_after_actual_resident_retirement(self):
        from corridorkey.worker_protocol import job_to_wire
        panel, plan, _ = self.install_panel(default_selection=True)
        self.start_resident()
        original = job_to_wire(self.window.job())
        self.window.start_calibration(plan)
        self.until(lambda: bool(self.tasks))
        self.assertEqual(self.start_observations, [(plan, True, None, False)])
        self.assertTrue(self.window.calibration_processing)
        bridge = self.window.bridge
        self.window.new_worker()
        self.assertIs(self.window.bridge, bridge)
        self.window.enqueue()
        self.assertFalse(self.window.jobs)
        self.assertEqual(job_to_wire(self.window.job()), original)
        self.window.stop_worker()
        self.assertEqual(self.tasks[0].cancel_count, 1)
        self.until(lambda: not self.window.calibration_processing)
        self.assertFalse(panel.waiting)

    def test_explicit_selection_change_refuses_before_stopping_resident(self):
        from corridorkey.asset_inventory import AssetSelection
        panel, plan, _ = self.install_panel()
        self.start_resident()
        self.window.asset_selection = AssetSelection(self.root / 'different-models')
        self.window.start_calibration(plan)
        self.assertIn('assets changed', panel.status.text())
        self.assertFalse(self.window.bridge.done.is_set())
        self.assertIsNone(self.window.pending_calibration)
        self.assertFalse(self.start_observations)
        self.assertFalse(self.window.jobs)

    def test_cancelled_pending_handoff_waits_for_progressive_io_without_starting_task(self):
        from corridorkey.preview_execution import prepare_preview
        panel, plan, _ = self.install_panel()
        self.start_resident()
        entered, release = threading.Event(), threading.Event()
        scratch = self.root / 'progressive-scratch'
        scratch.mkdir()
        self.window.progressive.scratch.setText(str(scratch))

        def blocked(*args, **kwargs):
            recipe = prepare_preview(*args, **kwargs)
            entered.set()
            if not release.wait(10):
                raise TimeoutError('fixture capture release')
            return recipe

        with patch('corridorkey.desktop_progressive.prepare_preview', new=blocked):
            self.window.progressive.enabled.setChecked(True)
            self.window.progressive.begin()
            try:
                self.assertTrue(entered.wait(3))
                self.window.start_calibration(plan)
                self.until(lambda: self.window.bridge.done.is_set())
                self.assertTrue(self.window.progressive.busy)
                self.assertFalse(self.start_observations)
                self.window.cancel_calibration()
                self.assertIsNone(self.window.pending_calibration)
            finally:
                release.set()
            self.until(lambda: not self.window.progressive.busy)
        self.assertFalse(self.start_observations)
        self.assertFalse(panel.waiting)
        self.assertFalse(list(scratch.iterdir()))

    def test_task_start_failure_is_actionable_and_never_retried(self):
        panel, plan, _ = self.install_panel(failing_factory=True)
        self.start_resident()
        self.window.start_calibration(plan)
        self.until(lambda: 'fixture task-start failure' in panel.status.text())
        for _ in range(3):
            self.window.poll()
        self.assertEqual(len(self.start_observations), 1)
        self.assertIsNone(panel.task)
        self.assertFalse(panel.waiting)
        self.assertFalse(self.window.calibration_processing)
        self.assertFalse(self.window.jobs)

    def test_unverified_cleanup_blocks_new_owner_and_parent_close(self):
        panel, plan, _ = self.install_panel()
        self.window.start_calibration(plan)
        self.until(lambda: bool(self.tasks))
        task = self.tasks[0]
        task.done.set()
        task.cleanup_error = 'fixture containment NOT VERIFIED'
        task.revision += 1
        original = copy.copy(self.window.jobs)
        self.assertTrue(self.window.calibration_processing)
        bridge = self.window.bridge
        self.window.new_worker()
        self.window.enqueue()
        self.assertIs(self.window.bridge, bridge)
        self.assertEqual(self.window.jobs, original)
        self.window.close()
        self.assertTrue(self.window.isVisible())
        self.assertFalse(panel.shutdown())
        # The control fixture has no native owner. Explicitly model successful
        # subsequent containment verification so test teardown can retire it.
        task.cleanup_error = None
        task.revision += 1
        self.until(lambda: not self.window.isVisible())


if __name__ == '__main__':
    unittest.main()
