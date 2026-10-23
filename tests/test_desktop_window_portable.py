"""Actual window/portable-dialog ownership with the existing fixture worker."""
import copy
import importlib.util
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_desktop_window as window_fixtures

QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt desktop is unavailable')
class PortableWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        window_fixtures.DesktopWindowTests.setUpClass()

    def setUp(self):
        # Compose the fixture; inheriting/importing its TestCase would discover
        # and rerun every unrelated window case in this module.
        self.fixture = window_fixtures.DesktopWindowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window, self.root, self.app = self.fixture.window, self.fixture.root, self.fixture.app

    def job(self, name='portable'):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig, WriterConfig
        from corridorkey.session import JobRequest

        source = self.root / name
        source.mkdir()
        for index in range(6):
            self.assertTrue(cv2.imwrite(str(source / f'frame{index:03d}.png'), np.full((24, 32, 3), index * 20, np.uint8)))
        hint = self.root / f'{name}-hint.png'
        self.assertTrue(cv2.imwrite(str(hint), np.full((24, 32), 255, np.uint8)))
        settings = InferenceConfig(start_frame=2, num_frames=3, frame_dir_fps=48.,
            frame_dir_linear=True, source_colour_profile='linear_bt709_d65', stage2_batch=2,
            quick_key=True, quick_interval=2, foundation_cache_mb=13, outputs={'alpha', 'cutout'},
            output_format='exr', exr_dtype='float', cutout_alpha_mode='premultiplied',
            temp_dir=self.root / f'{name}-scratch')
        writer = WriterConfig(ffmpeg_bin='explicit-local-ffmpeg', crf=17, threads=3)
        return JobRequest.create(source, self.root / f'{name}-outputs', settings, writer,
                                 initial_hint_path=hint, profile_stages=True)

    def form(self):
        return (self.window.source.text(), self.window.output.text(), self.window.first_hint.text(),
                self.window.full_hint.text(), self.window.profile.isChecked(), self.window.config.toPlainText(),
                self.window.writer.toPlainText(), self.window.frame.value())

    def open_dialog(self, job):
        self.window.populate_job(job, frame=1)
        self.window.open_portable_project()
        panel = self.window.portable_project
        assert panel is not None
        return panel

    def test_projects_menu_opens_retained_idle_dialog_without_io_or_queueing(self):
        from corridorkey.asset_inventory import AssetSelection
        from corridorkey.worker_protocol import job_to_wire

        widgets = importlib.import_module('PySide6.QtWidgets')
        job = self.job()
        selection = AssetSelection(self.root / 'selected-assets', self.root / 'selected-weights')
        self.window.asset_selection = selection
        self.window.populate_job(job, frame=1)
        button = next(button for button in self.window.findChildren(widgets.QPushButton) if button.text() == 'Projects…')
        menu = button.menu()
        assert menu is not None
        action = next(action for action in menu.actions() if action.text() == 'Portable projects…')
        bridge = self.window.bridge
        started = []

        def unexpected_io(*args, **kwargs):
            started.append((args, kwargs))
            raise AssertionError('Opening a portable project must not start file I/O')

        # A plain callable avoids native Qt slot-introspection of a class-level
        # MagicMock while the QAction constructs its child dialog.
        with patch('corridorkey.desktop_portable_project.PortableProjectDialog._begin', new=unexpected_io), \
                patch.object(bridge, 'send') as sent:
            action.trigger()
            self.app.processEvents()
            panel = self.window.portable_project
            assert panel is not None
            self.assertTrue(panel.isVisible())
            self.assertIs(panel.selection, selection)
            self.assertEqual(panel.frame, 1)
            self.assertEqual(panel.original_wire, job_to_wire(job))
            self.assertIsNone(panel.io_thread)
            self.assertIsNone(panel.inventory)
            action.trigger()
            self.assertIs(self.window.portable_project, panel)
            self.assertEqual(started, [])
            sent.assert_not_called()
        self.assertIs(self.window.bridge, bridge)
        self.assertFalse(self.window.jobs)

    def test_loaded_signal_preserves_settings_offsets_queue_and_model_owner(self):
        from corridorkey.asset_inventory import AssetSelection
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_to_wire

        old = self.job('old')
        panel = self.open_dialog(old)
        loaded = self.job('loaded')
        queue_entry = {'job': old, 'status': 'queued', 'event': None}
        self.window.jobs['existing-record'] = queue_entry
        bridge = self.window.bridge
        selection = AssetSelection(self.root / 'active-assets', self.root / 'active-weights')
        self.window.asset_selection = selection
        output = str((self.root / 'explicit-output').resolve())
        report = {'model_references': {'asset_root': '/unavailable-original-machine/models'},
                  'compatibility': 'NOT VERIFIED', 'scratch_override': str(loaded.configurations()[0].temp_dir)}
        with patch.object(bridge, 'send') as sent:
            panel.loaded.emit(loaded, 2, output, report)
            sent.assert_not_called()
        settings, writer = loaded.configurations()
        expected = JobRequest.create(loaded.input_path, output, settings, writer,
            initial_hint_path=loaded.initial_hint_path, hint_video_path=loaded.hint_video_path,
            profile_stages=loaded.profile_stages)
        self.assertEqual(job_to_wire(self.window.job()), job_to_wire(expected))
        self.assertEqual(self.window.frame.value(), 2)
        self.assertEqual(self.window.limit.value(), 3)
        self.assertEqual(self.window._settings()['start_frame'], 2)
        self.assertEqual(self.window._settings()['temp_dir'], str(settings.temp_dir.resolve()))
        self.assertIs(self.window.jobs['existing-record'], queue_entry)
        self.assertEqual(list(self.window.jobs), ['existing-record'])
        self.assertIs(self.window.bridge, bridge)
        self.assertIs(self.window.asset_selection, selection)
        self.assertIn('NOT VERIFIED', self.window.statusBar().toolTip())

    def test_invalid_payload_and_dirty_correction_draft_leave_form_unchanged(self):
        from corridorkey.session import JobRequest

        old, loaded = self.job('old'), self.job('loaded')
        panel = self.open_dialog(old)
        before = self.form()
        output = str((self.root / 'explicit-output').resolve())
        invalid = [(JobRequest, 0, output, {}), (loaded, True, output, {}),
                   (loaded, 10**9 + 1, output, {}), (loaded, 2**40, output, {}),
                   (loaded, 0, '', {}), (loaded, 0, output, {'bad': float('nan')}),
                   (replace(loaded, config_json='{'), 0, output, {}),
                   (replace(loaded, writer_json='{'), 0, output, {}),
                   (replace(loaded, config_json='{"hann_chunk":80,"hann_chunk":40}'), 0, output, {}),
                   (replace(loaded, writer_json='{"threads":2,"threads":3}'), 0, output, {}),
                   (loaded, 0, output, {'oversized': 'x' * (64 * 1024)})]
        for payload in invalid:
            with self.subTest(payload_type=type(payload[0]).__name__, frame=payload[1]), \
                    patch.object(self.window, 'error') as error, \
                    patch.object(panel, 'acknowledge_application') as acknowledged:
                self.window.apply_portable_project(*payload)
                error.assert_called_once()
                acknowledged.assert_called_once()
                self.assertIs(acknowledged.call_args.args[0], False)
                self.assertEqual(self.form(), before)
        fake_editor = SimpleNamespace(canvas=SimpleNamespace(dirty=True))
        with patch.object(self.window, 'editor', fake_editor), patch.object(self.window, 'error') as error:
            self.window.apply_portable_project(loaded, 2, output, {})
            error.assert_called_once()
            self.assertEqual(self.form(), before)
        self.assertFalse(self.window.jobs)

    def test_parent_close_waits_for_actual_owned_portable_operation(self):
        panel = self.open_dialog(self.job())
        entered, release = threading.Event(), threading.Event()

        def blocked(check):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture portable operation gate')
            check()
            return {'unexpected': 'cancelled operation must never publish'}

        self.assertTrue(panel._begin('draft_inventory', blocked))
        try:
            self.assertTrue(entered.wait(2))
            self.window.close()
            self.assertTrue(self.window.isVisible())
            self.assertTrue(panel.busy)
            self.assertTrue(panel.cancellation.cancelled)
            self.assertTrue(panel.closing)
            self.assertIsNone(panel.inventory)
        finally:
            release.set()
        self.fixture.until(lambda: not panel.busy and not self.window.isVisible())
        self.assertIsNone(panel.inventory)
        self.assertFalse(self.window.jobs)

    def test_reopening_completed_hidden_dialog_captures_fresh_draft(self):
        from corridorkey.worker_protocol import job_to_wire

        first_job = self.job('first')
        panel = self.open_dialog(first_job)
        captured = copy.deepcopy(panel.original_wire)
        panel.close()
        self.assertFalse(panel.isVisible())
        self.assertFalse(panel.busy)
        second_job = self.job('second')
        self.window.populate_job(second_job, frame=2)
        self.window.open_portable_project()
        fresh = self.window.portable_project
        assert fresh is not None
        self.assertIsNot(fresh, panel)
        self.assertEqual(captured, job_to_wire(first_job))
        self.assertEqual(fresh.original_wire, job_to_wire(second_job))
        self.assertEqual(fresh.frame, 2)
        self.assertIsNone(fresh.inventory)
        self.assertIsNone(fresh.io_thread)
        self.assertFalse(self.window.jobs)

    def test_committed_unemitted_snapshot_survives_parent_reopen_with_bounded_detached_history(self):
        import test_portable_project as core_fixtures

        from corridorkey.corrections import load_corrections
        from corridorkey.portable_project import (
            ProjectInspection,
            inspect_project,
            publish_project,
        )

        core = core_fixtures.PortableProjectTests()
        core.setUp()
        self.addCleanup(core.tearDown)
        job, document = core.corrected_job(4)
        package = publish_project(core.plan(job, ('corrections', 'source')), core.parent)
        inspection = inspect_project(package, read_budget=10**7)
        panel = self.open_dialog(job)
        panel.inspection = inspection
        panel.inspection_ready = True
        panel.inspection_frame = inspection.preview_frame
        panel.output.setText(str(core.output))
        panel.scratch.setText(str(core.scratch))
        emitted = []
        panel.loaded.connect(lambda *payload: emitted.append(payload))
        original_form = self.form()
        committed, release = threading.Event(), threading.Event()
        original_apply = ProjectInspection.apply

        def commit_then_hold(current, *, output_dir, temp_dir, check):
            resolved = original_apply(current, output_dir=output_dir, temp_dir=temp_dir, check=check)
            committed.set()
            if not release.wait(5):
                raise TimeoutError('fixture committed portable snapshot delivery gate')
            return resolved

        with patch.object(ProjectInspection, 'apply', new=commit_then_hold):
            panel.apply()
            try:
                self.fixture.until(committed.is_set)
                panel.cancel()
            finally:
                release.set()
            self.fixture.until(lambda: not panel.busy)
        self.assertEqual(emitted, [])
        self.assertEqual(self.form(), original_form)
        resolved = panel.last_resolved_job
        assert resolved is not None
        config, _ = resolved.configurations()
        assert config.corrections_path is not None
        artifact = Path(config.corrections_path)
        self.assertEqual(load_corrections(artifact, source=resolved.input_path).identity, document.identity)
        old_report = panel.application_report
        assert old_report is not None
        self.assertIs(old_report['emitted'], False)
        self.assertIs(old_report['accepted'], False)
        panel.close()
        self.window.open_portable_project()
        fresh = self.window.portable_project
        assert fresh is not None
        self.assertIsNot(fresh, panel)
        record = self.window.portable_application_records[-1]
        self.assertIs(record['job'], resolved)
        self.assertEqual(record['report']['resolved_correction_snapshot'], str(artifact))
        self.assertIn(str(artifact), fresh.details.toPlainText())
        old_report['accepted'] = True
        self.assertIs(record['report']['accepted'], False)
        self.assertTrue(artifact.is_file())

        # Actual further relocations exercise the session's bounded metadata
        # history. Evicting a record never deletes its committed artifact.
        for _ in range(8):
            additional = inspection.apply(output_dir=core.output, temp_dir=core.scratch)
            fresh._retain_resolution((additional, inspection.preview_frame, str(core.output), {}),
                                     emitted=False, accepted=False, message='Committed fixture snapshot retained without emission')
            self.window.retain_portable_application(fresh)
        self.assertEqual(len(self.window.portable_application_records), 8)
        self.assertTrue(artifact.is_file())
        self.assertEqual(emitted, [])
        self.assertEqual(self.form(), original_form)
        self.assertFalse(self.window.jobs)


if __name__ == '__main__':
    unittest.main()
