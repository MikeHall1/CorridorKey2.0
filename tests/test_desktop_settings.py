"""Common forms preserve advanced settings, precision and immutable queued jobs."""
import importlib.util
import json
import os
import unittest

from tests.qt_cleanup import dispose_widget

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class SettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['settings-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_settings import SettingsPanel
        from corridorkey.desktop_state import pretty_settings
        from corridorkey.session import JobRequest
        self.panel = SettingsPanel()
        self.config, self.writer = pretty_settings(JobRequest.create('.', '.'))
        self.panel.load(self.config, self.writer)

    def tearDown(self):
        dispose_widget(self.panel)

    def test_exact_numbers_and_unexposed_options_survive(self):
        settings = json.loads(self.config)
        settings.update(frame_dir_fps=23.976023976023978, seed=123, alpha_refine_sigma=0.123456789012345)
        self.panel.load(json.dumps(settings), self.writer)
        config, _ = self.panel.values()
        self.assertEqual(json.loads(config)['frame_dir_fps'], settings['frame_dir_fps'])
        self.assertEqual(json.loads(config)['alpha_refine_sigma'], settings['alpha_refine_sigma'])
        self.assertEqual(json.loads(config)['seed'], 123)

    def test_conflicting_controls_and_nonfinite_numbers_rejected(self):
        from PySide6.QtWidgets import QCheckBox, QLineEdit
        low = self.panel.controls['engine', 'low_vram']
        cache = self.panel.controls['engine', 'foundation_cache_mb']
        assert isinstance(low, QCheckBox) and isinstance(cache, QLineEdit)
        low.setChecked(True)
        cache.setText('32')
        with self.assertRaisesRegex(ValueError, 'low_vram'):
            self.panel.values()
        low.setChecked(False)
        fps = self.panel.controls['engine', 'frame_dir_fps']
        assert isinstance(fps, QLineEdit)
        fps.setText('nan')
        with self.assertRaises(ValueError):
            self.panel.values()
        fps.setText('24')
        for checkbox in self.panel.outputs.values():
            checkbox.setChecked(False)
        with self.assertRaisesRegex(ValueError, 'outputs'):
            self.panel.values()

    def test_apply_contains_validated_complete_settings_and_original_token(self):
        from PySide6.QtWidgets import QComboBox, QLineEdit

        from corridorkey.session import JobRequest
        queued = JobRequest.create('.', '.')
        received = []
        self.panel.apply_requested.connect(lambda *args: received.append(args))
        export = self.panel.controls['engine', 'output_format']
        assert isinstance(export, QComboBox)
        export.setCurrentText('exr')
        budget = self.panel.controls['engine', 'temp_map_budget_bytes']
        assert isinstance(budget, QLineEdit)
        budget.setText('32')
        self.panel.apply()
        self.assertEqual(received[0][:2], (self.config, self.writer))
        self.assertEqual(json.loads(received[0][2])['output_format'], 'exr')
        self.assertEqual(json.loads(received[0][2])['temp_map_budget_bytes'], 32)
        self.assertIsNone(json.loads(self.config)['temp_map_budget_bytes'])
        self.assertIsNone(queued.configurations()[0].temp_map_budget_bytes)
        self.assertEqual(json.loads(self.config)['output_format'], 'video')

    def test_managed_scratch_optional_exact_byte_values_and_bounds(self):
        from PySide6.QtWidgets import QLineEdit
        budget = self.panel.controls['engine', 'temp_map_budget_bytes']
        assert isinstance(budget, QLineEdit)
        self.assertEqual(budget.text(), '')
        self.assertIsNone(json.loads(self.panel.values()[0])['temp_map_budget_bytes'])
        for text, expected in [('0', 0), ('9223372036854775807', 2**63 - 1)]:
            budget.setText(text)
            self.assertEqual(json.loads(self.panel.values()[0])['temp_map_budget_bytes'], expected)
        for text in ('-1', '9223372036854775808', '1.5', 'invalid'):
            budget.setText(text)
            with self.assertRaises(ValueError):
                self.panel.values()
        budget.setText('   ')
        self.assertIsNone(json.loads(self.panel.values()[0])['temp_map_budget_bytes'])

    def test_source_witness_control_requires_explicit_linear_exr_contract(self):
        from PySide6.QtWidgets import QCheckBox, QComboBox
        witness = self.panel.controls['engine', 'source_witness']
        linear = self.panel.controls['engine', 'frame_dir_linear']
        export = self.panel.controls['engine', 'output_format']
        assert isinstance(witness, QCheckBox) and isinstance(linear, QCheckBox) and isinstance(export, QComboBox)
        self.assertFalse(witness.isChecked())
        witness.setChecked(True)
        with self.assertRaisesRegex(ValueError, 'source_witness'):
            self.panel.values()
        linear.setChecked(True)
        export.setCurrentText('exr')
        config, _ = self.panel.values()
        self.assertTrue(json.loads(config)['source_witness'])
        self.assertFalse(json.loads(self.config)['source_witness'])

    def test_window_rejects_stale_edits_and_applies_without_enqueue(self):
        from corridorkey.desktop_window import DesktopWindow
        window = DesktopWindow()
        try:
            window.open_common_settings()
            window.update_settings({'despill_strength': 0.2, 'temp_map_budget_bytes': 128})
            window.settings_panel.apply()
            self.assertIn('changed since', window.settings_panel.status.text())
            self.assertEqual(window._settings()['despill_strength'], 0.2)
            self.assertEqual(window._settings()['temp_map_budget_bytes'], 128)
            window.open_common_settings()
            window.settings_panel.apply()
            self.assertIn('Validated', window.settings_panel.status.text())
            self.assertFalse(window.jobs)
            self.assertTrue(window.bridge.commands.empty())
        finally:
            window.close()
            window.bridge.thread.join(10)
            window.poll()
            self.app.processEvents()
            dispose_widget(window)
