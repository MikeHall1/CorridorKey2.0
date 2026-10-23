"""Foreign archive → verified native draft through actual Qt; no model execution."""
import copy
import importlib.util
import json
import os
import time
import unittest

import test_desktop_window as window_fixtures


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt binding unavailable')
@unittest.skipIf(os.name == 'nt', 'Windows-flavour archive on a POSIX receiver scenario; native Windows acceptance remains unverified')
class ForeignPortableWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        window_fixtures.DesktopWindowTests.setUpClass()

    def setUp(self):
        self.fixture = window_fixtures.DesktopWindowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window, self.root, self.app = self.fixture.window, self.fixture.root.resolve(), self.fixture.app

    def until(self, predicate):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.005)
        panel = self.window.portable_project
        self.fail(panel.status.text() if panel is not None else 'No portable project dialog')

    def prepared_panel(self):
        from test_portable_foreign_project import make_foreign_collected_package

        from corridorkey.desktop_portable_project import PortableProjectDialog
        folder = self.root / 'foreign-fixture'
        folder.mkdir()
        package, metadata = make_foreign_collected_package(folder)
        self.window.bridge.send('status', {})
        self.until(lambda: self.window.backend.text().startswith('Worker PID'))
        panel = PortableProjectDialog(None, self.window)
        panel.loaded.connect(self.window.apply_portable_project)
        self.window.portable_project = panel
        panel.show()
        panel.package_path.setText(str(package))
        panel.inventory_package()
        self.until(lambda: not panel.busy)
        self.assertEqual(panel.mode, 'package', panel.status.text())
        self.assertIsNotNone(panel.report, panel.status.text())
        self.assertIn('Captured source start 1', panel.details.toPlainText())
        self.assertIn(metadata['original_wire']['output_dir'], panel.details.toPlainText())
        panel.read_budget.setValue(64)
        panel.verify()
        self.until(lambda: not panel.busy)
        self.assertTrue(panel.inspection_ready, panel.status.text())
        output, scratch = self.root / 'native-output', self.root / 'native-scratch'
        output.mkdir()
        scratch.mkdir()
        panel.output.setText(str(output))
        panel.scratch.setText(str(scratch))
        return panel, package, metadata, output, scratch

    def test_verified_foreign_dialog_applies_native_paths_preserving_owner_queue_and_originals(self):
        from corridorkey.corrections import load_corrections
        from corridorkey.worker_protocol import job_to_wire
        panel, package, metadata, output, scratch = self.prepared_panel()
        original_manifest = (package / 'project.json').read_bytes()
        bridge, selection, jobs = self.window.bridge, self.window.asset_selection, copy.deepcopy(self.window.jobs)
        panel.apply()
        self.until(lambda: not panel.busy)
        self.assertIsNotNone(panel.last_resolved_job, panel.status.text())
        report = panel.application_report
        assert report is not None
        self.assertTrue(report['accepted'], panel.status.text())
        resolved = panel.last_resolved_job
        assert resolved is not None
        self.assertEqual(job_to_wire(self.window.job()), job_to_wire(resolved))
        self.assertEqual(resolved.output_dir, str(output))
        config, _ = resolved.configurations()
        self.assertEqual(str(config.temp_dir), str(scratch))
        self.assertEqual((config.start_frame, config.num_frames, config.frame_dir_fps), (1, 2, 30.))
        self.assertEqual(load_corrections(config.corrections_path).identity, metadata['correction_identity'])
        self.assertEqual(self.window.frame.value(), metadata['preview_frame'])
        self.assertIs(self.window.bridge, bridge)
        self.assertFalse(bridge.done.is_set())
        self.assertEqual(self.window.jobs, jobs)
        self.assertEqual(self.window.asset_selection, selection)
        self.assertEqual((package / 'project.json').read_bytes(), original_manifest)
        inspection = panel.inspection
        assert inspection is not None
        self.assertEqual(inspection.report['requested_job'], metadata['original_wire'])
        self.assertLessEqual(len(json.dumps(report, allow_nan=False).encode()), 64 * 1024)
        for path, raw in metadata.get('originals', {}).items():
            self.assertEqual(path.read_bytes(), raw)

    def test_foreign_destination_cannot_be_silently_rebased_by_dialog(self):
        panel, package, metadata, _, scratch = self.prepared_panel()
        before = (self.window.source.text(), self.window.config.toPlainText(), self.window.output.text())
        original_manifest = (package / 'project.json').read_bytes()
        # A producing-host path entered on POSIX must not become a local cwd child.
        panel.output.setText(r'C:\archive\previous-export')
        panel.apply()
        self.until(lambda: not panel.busy)
        self.assertIsNone(panel.last_resolved_job)
        self.assertIn('native', panel.status.text().lower())
        self.assertEqual((self.window.source.text(), self.window.config.toPlainText(), self.window.output.text()), before)
        self.assertEqual(list(scratch.iterdir()), [])
        self.assertEqual((package / 'project.json').read_bytes(), original_manifest)
        self.assertFalse(self.window.jobs)
        for path, raw in metadata.get('originals', {}).items():
            self.assertEqual(path.read_bytes(), raw)


if __name__ == '__main__':
    unittest.main()
