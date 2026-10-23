"""Explicit sampled memory stopping survives user settings and job transport."""
import contextlib
import importlib.util
import io
import json
import os
import unittest
from unittest.mock import Mock

from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_from_wire, job_to_wire


class ProcessMemorySettingsTests(unittest.TestCase):
    def test_exact_optional_threshold_validation_and_immutable_wire(self):
        self.assertIsNone(InferenceConfig().process_tree_rss_stop_bytes)
        for value in (None, 0, 1, 2**63 - 1):
            with self.subTest(value=value):
                config = InferenceConfig(process_tree_rss_stop_bytes=value)
                request = JobRequest.create('.', '.', config)
                config.process_tree_rss_stop_bytes = 123
                decoded = job_from_wire(job_to_wire(request))
                self.assertEqual(decoded.configurations()[0].process_tree_rss_stop_bytes, value)
        for value in (True, False, -1, 2**63, 1.5, '1024'):
            invalid = InferenceConfig()
            vars(invalid)['process_tree_rss_stop_bytes'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'process_tree_rss_stop_bytes'):
                invalid.validate()

    def test_cli_preserves_literal_zero_and_rejects_overflow_before_submission(self):
        import infer

        session = Mock(last_result=None)
        arguments = ['--input', '.', '--output_dir', '.', '--process_tree_rss_stop_bytes', '0']
        infer.main(arguments, session=session)
        request = session.run.call_args.args[0]
        config = request.configurations()[0]
        self.assertEqual(config.process_tree_rss_stop_bytes, 0)
        self.assertFalse(config.low_vram)
        self.assertFalse(config.quick_key)
        session.reset_mock()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            infer.main(arguments[:-1] + [str(2**63)], session=session)
        session.run.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class ProcessMemorySettingsQtTests(unittest.TestCase):
    def test_future_job_form_preserves_exact_threshold_and_stale_edit_guard(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication, QLineEdit

        from corridorkey.desktop_window import DesktopWindow
        from tests.qt_cleanup import dispose_widget

        app = QApplication.instance() or QApplication(['memory-settings', '-platform', 'offscreen'])
        window = DesktopWindow()
        try:
            window.open_common_settings()
            panel = window.settings_panel
            field = panel.controls['engine', 'process_tree_rss_stop_bytes']
            assert isinstance(field, QLineEdit)
            for text, expected in [('', None), ('0', 0), (str(2**63 - 1), 2**63 - 1)]:
                field.setText(text)
                self.assertEqual(json.loads(panel.values()[0])['process_tree_rss_stop_bytes'], expected)
            for text in ('-1', str(2**63), '1.5'):
                field.setText(text)
                with self.assertRaises(ValueError):
                    panel.values()
            field.setText('1000000000')
            window.update_settings({'process_tree_rss_stop_bytes': 2000000000})
            panel.apply()
            self.assertIn('changed since', panel.status.text())
            self.assertEqual(window._settings()['process_tree_rss_stop_bytes'], 2000000000)
            window.open_common_settings()
            field.setText('0')
            panel.apply()
            self.assertEqual(window._settings()['process_tree_rss_stop_bytes'], 0)
            self.assertFalse(window.jobs)
            self.assertTrue(window.bridge.commands.empty())
        finally:
            window.close()
            window.bridge.thread.join(10)
            window.poll()
            app.processEvents()
            dispose_widget(window)
