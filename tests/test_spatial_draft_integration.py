"""Explicit spatial settings/recipes/cache labels; neural quality is not assessed."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


class SpatialDraftContractTests(unittest.TestCase):
    def test_scale_validation_and_immutable_wire(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_from_wire, job_to_wire
        for scale in (1, 1.0, .5, .125, 1e-100):
            cfg = InferenceConfig(model_scale=scale, temp_dir=Path(tempfile.gettempdir()).resolve() / 'm45-immutable-fixture')
            cfg.validate()
            job = JobRequest.create('.', '.', cfg)
            self.assertEqual(job_from_wire(job_to_wire(job)), job)
            cfg.model_scale = 1
            self.assertEqual(job.configurations()[0].model_scale, scale)
        for scale in (False, True, 0, -.1, 1.001, float('nan'), float('inf'), '0.5', 10**1000):
            invalid_cfg: Any = InferenceConfig()
            invalid_cfg.model_scale = scale
            with self.subTest(scale=str(scale)), self.assertRaises(ValueError):
                invalid_cfg.validate()

    def test_diagnostic_full_quality_resets_scale_preserve_mode_keeps_it(self):
        from corridorkey.diagnostic_ranges import inspect_diagnostics, prepare_range
        from corridorkey.diagnostics import json_value
        from corridorkey.session import JobRequest
        from tests.test_diagnostic_ranges import make_diagnostic_fixture
        with tempfile.TemporaryDirectory() as folder:
            job, record, parent = make_diagnostic_fixture(folder)
            cfg, writer = job.configurations()
            cfg = replace(cfg, model_scale=.5)
            job = JobRequest.create(job.input_path, job.output_dir, cfg, writer)
            record['requested_settings'] = json_value(cfg)
            before = {p: p.read_bytes() for p in Path(folder).rglob('*') if p.is_file()}
            snapshot = inspect_diagnostics(job, record)
            full = prepare_range(snapshot, 3, 5, parent)
            draft = prepare_range(snapshot, 3, 5, parent, full_quality=False)
            self.assertEqual(full.configurations()[0].model_scale, 1)
            self.assertFalse(full.configurations()[0].quick_key)
            self.assertEqual(draft.configurations()[0].model_scale, .5)
            self.assertTrue(draft.configurations()[0].quick_key)
            self.assertFalse(Path(full.output_dir).exists())
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_ofx_spatial_draft_is_separate_from_source_rgb(self):
        from corridorkey.ofx_record import cache_spec
        record: dict[str, Any] = {'schema_version': 1, 'status': 'complete', 'input': 'local', 'diagnostics': {
            'source': {'native_hw': [64, 64], 'start_frame': 0, 'num_frames': 1, 'native_fps': 24},
            'frames_submitted': 1, 'outputs': {'fg': 'fg', 'alpha': 'alpha'}, 'output_contracts': {
                'fg': {'format': 'exr', 'channels': 3, 'rgb_representation': 'scene_linear',
                       'foreground_mode': 'reconstructed'},
                'alpha': {'format': 'exr', 'channels': 1}}}}
        self.assertFalse(cache_spec(record).draft)
        fg = record['diagnostics']['output_contracts']['fg']
        fg['spatial_draft'] = {'enabled': True, 'requested_scale': .5}
        self.assertTrue(cache_spec(record).draft)
        self.assertEqual(fg['foreground_mode'], 'reconstructed')
        fg['spatial_draft']['enabled'] = False
        self.assertFalse(cache_spec(record).draft)


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class SpatialDraftDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['spatial-draft', '-platform', 'offscreen'])

    def test_common_form_validates_without_rounding_or_submission(self):
        from PySide6.QtWidgets import QLineEdit

        from corridorkey.desktop_settings import SettingsPanel
        from corridorkey.desktop_state import pretty_settings
        from corridorkey.session import JobRequest
        from tests.qt_cleanup import dispose_widget
        panel = SettingsPanel()
        try:
            original = JobRequest.create('.', '.')
            panel.load(*pretty_settings(original))
            control = panel.controls['engine', 'model_scale']
            assert isinstance(control, QLineEdit)
            control.setText('0.3333333333333333')
            config, _ = panel.values()
            self.assertEqual(json.loads(config)['model_scale'], .3333333333333333)
            self.assertEqual(original.configurations()[0].model_scale, 1)
            for invalid in ('0', '1.1', 'nan', '-.5'):
                control.setText(invalid)
                with self.assertRaises(ValueError):
                    panel.values()
        finally:
            dispose_widget(panel)

    def test_persistent_warning_and_full_quality_preserve_existing_job(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.worker_client import WorkerClient
        from tests.qt_cleanup import dispose_widget
        window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable,
            str(Path(__file__).with_name('worker_fixture.py'))])))
        try:
            window.source.setText(str(Path.cwd()))
            captured = window.job(preview_only=True)
            window.update_settings({'model_scale': .5})
            self.assertIn('Spatial draft ×0.5', window.quality_note.text())
            window.draft_quality()
            self.assertIn('Spatial draft', window.quality_note.text())
            self.assertIn('Quick Key draft', window.quality_note.text())
            window.full_quality()
            self.assertEqual(window._settings()['model_scale'], 1)
            self.assertFalse(window._settings()['quick_key'])
            self.assertNotIn('Spatial draft', window.quality_note.text())
            # Exercise loading arbitrary persisted settings, independent of JSON whitespace.
            settings = json.loads(captured.config_json)
            settings['model_scale'] = .25
            window.config.setPlainText(json.dumps(settings))
            self.assertIn('Spatial draft ×0.25', window.quality_note.text())
            self.assertEqual(captured.configurations()[0].model_scale, 1)
            self.assertEqual(window.bridge.commands.qsize(), 0)
        finally:
            window.close()
            window.bridge.thread.join(5)
            window.loader.thread.join(5)
            self.assertFalse(window.bridge.thread.is_alive())
            self.assertFalse(window.loader.thread.is_alive())
            window.poll()
            dispose_widget(window)
