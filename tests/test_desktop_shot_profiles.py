"""Off-thread source-only panel actions and future snapshot fences."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class DesktopShotProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['shot-profile-test', '-platform', 'offscreen'])

    def setUp(self):
        from dataclasses import replace

        import cv2
        import numpy as np
        from PySide6.QtWidgets import QWidget

        from corridorkey.config import InferenceConfig
        from corridorkey.desktop_shot_profiles import ShotProfilesPanel
        from corridorkey.session import JobRequest
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        source = self.root / 'source'
        source.mkdir()
        for frame in range(4):
            cv2.imwrite(str(source / f'{frame}.png'), np.full((16, 32, 3), 90, np.uint8))
        class Workspace(QWidget):
            def job(self, *, preview_only=False):
                return self.current
            def update_settings(self, changes):
                config, writer = self.current.configurations()
                self.current = JobRequest.create(self.current.input_path, self.current.output_dir,
                    replace(config, **changes), writer)
                return True
        self.workspace = Workspace()
        self.workspace.current = JobRequest.create(source, self.root / 'output', InferenceConfig(
            stage2_batch=3, start_frame=1, num_frames=3, output_format='exr', outputs={'alpha'}))
        self.panel = ShotProfilesPanel(self.workspace)

    def tearDown(self):
        self.panel.shutdown()
        if self.panel.io_thread is not None:
            self.panel.io_thread.join(10)
        self.panel.poll()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.workspace)

    def until_idle(self):
        deadline = time.monotonic() + 10
        while self.panel.busy and time.monotonic() < deadline:
            self.app.processEvents()
            self.panel.poll()
            time.sleep(.005)
        self.panel.poll()
        self.assertFalse(self.panel.busy)

    def test_analyze_edit_save_open_apply_disable_preserves_captured_queue(self):
        import json

        from corridorkey.shot_profiles import ProfilePlan
        frozen_queue = self.workspace.current
        self.panel.analyze()
        self.until_idle()
        self.assertIsNone(self.workspace.current.configurations()[0].shot_profiles)
        data = json.loads(self.panel.document.toPlainText())
        data['rows'][0]['stage2_batch'] = 1
        data['rows'][0]['alpha_feather_radius'] = 2
        self.panel.document.setPlainText(json.dumps(data))
        path = self.root / 'profiles.json'
        self.panel.save_path(path)
        self.until_idle()
        self.assertTrue(path.exists())
        original = path.read_bytes()
        self.panel.open_path(path)
        self.until_idle()
        self.assertEqual(ProfilePlan.parse(json.loads(self.panel.document.toPlainText())).to_dict(), data)
        self.panel.apply()
        self.until_idle()
        self.assertEqual(self.workspace.current.configurations()[0].shot_profiles, data)
        self.assertIsNone(frozen_queue.configurations()[0].shot_profiles)
        self.panel.disable()
        self.assertIsNone(self.workspace.current.configurations()[0].shot_profiles)
        self.assertEqual(path.read_bytes(), original)

    def test_stale_cancelled_apply_and_busy_owner_are_rejected(self):
        from corridorkey.shot_profiles import analyze_job, apply_profiles
        plan = analyze_job(self.workspace.current)
        self.panel.document.setPlainText(plan.payload)
        gate, entered = threading.Event(), threading.Event()
        def delayed(job, plan, **kwargs):
            result = apply_profiles(job, plan)
            entered.set()
            gate.wait(5)
            return result
        with patch('corridorkey.desktop_shot_profiles.apply_profiles', side_effect=delayed):
            self.panel.apply()
            self.assertTrue(entered.wait(5))
            self.panel.analyze()
            self.assertIn('retire', self.panel.status.text())
            self.panel.cancel()
            gate.set()
            self.until_idle()
        self.assertIsNone(self.workspace.current.configurations()[0].shot_profiles)
        self.assertIn('rejected', self.panel.status.text())

    def test_window_setting_changes_fence_analysis_without_queued_mutation(self):
        import json
        import sys

        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.shot_profiles import analyze_job
        from corridorkey.worker_client import WorkerClient
        from tests.qt_cleanup import dispose_widget
        fixture = Path(__file__).with_name('worker_fixture.py')
        window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(fixture)])))
        try:
            window.source.setText(self.workspace.current.input_path)
            window.output.setText(self.workspace.current.output_dir)
            window.config.setPlainText(self.workspace.current.config_json)
            window.writer.setPlainText(self.workspace.current.writer_json)
            frozen = window.job()
            window.jobs['existing'] = {'job': frozen, 'status': 'sending', 'event': None}
            panel = window.shot_profiles
            plan = analyze_job(frozen)
            gate, entered = threading.Event(), threading.Event()
            def delayed(*args, **kwargs):
                entered.set()
                gate.wait(5)
                return plan
            with patch('corridorkey.desktop_shot_profiles.analyze_job', side_effect=delayed):
                panel.analyze()
                self.assertTrue(entered.wait(5))
                settings = json.loads(window.config.toPlainText())
                settings['alpha_feather_radius'] = 1
                window.config.setPlainText(json.dumps(settings))
                gate.set()
                deadline = time.monotonic() + 10
                while panel.busy and time.monotonic() < deadline:
                    self.app.processEvents()
                    window.poll()
                    time.sleep(.005)
                self.assertIn('rejected', panel.status.text())
                self.assertEqual(panel.document.toPlainText(), '')
            self.assertEqual(window.jobs['existing']['job'], frozen)
        finally:
            window.close()
            window.bridge.thread.join(10)
            window.poll()
            dispose_widget(window)

    def test_committed_save_survives_late_cancel_and_duplicate_editor_keys_refuse(self):
        from corridorkey.shot_profiles import analyze_job, save_profiles
        plan = analyze_job(self.workspace.current)
        raw = plan.payload.replace('"enabled":true', '"enabled":false,"enabled":true', 1)
        self.panel.document.setPlainText(raw)
        self.panel.apply()
        self.assertFalse(self.panel.busy)
        self.assertIn('Duplicate', self.panel.status.text())
        self.panel.document.setPlainText(plan.payload)
        gate, committed = threading.Event(), threading.Event()
        def delayed(path, plan, job, **kwargs):
            save_profiles(path, plan, job, **kwargs)
            committed.set()
            gate.wait(5)
        destination = self.root / 'retained.json'
        with patch('corridorkey.desktop_shot_profiles.save_profiles', side_effect=delayed):
            self.panel.save_path(destination)
            self.assertTrue(committed.wait(5))
            original = destination.read_bytes()
            self.panel.cancel()
            gate.set()
            self.until_idle()
        self.assertEqual(destination.read_bytes(), original)
        self.assertIn('Saved new local', self.panel.status.text())
        self.assertIsNone(self.workspace.current.configurations()[0].shot_profiles)
