"""First-run preference and deferred intro; no global user settings touched."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from corridorkey.desktop import first_run_readiness

QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt desktop is unavailable')
class FirstRunTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtCore import QSettings
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = str(Path(self.folder.name) / 'preferences.ini')
        self.preferences = QSettings(self.path, QSettings.Format.IniFormat)
        self.window = SimpleNamespace(closing=False, readiness=SimpleNamespace(show=Mock(), inspect=Mock()))
        self.callbacks = []

    def schedule(self, delay, callback):
        self.assertEqual(delay, 0)
        self.callbacks.append(callback)

    def test_first_run_opens_only_intro_then_persists_seen_for_next_launch(self):
        from PySide6.QtCore import QSettings
        self.assertTrue(first_run_readiness(self.window, self.preferences, self.schedule))
        self.window.readiness.show.assert_not_called()
        self.callbacks.pop()()
        self.window.readiness.show.assert_called_once()
        self.window.readiness.inspect.assert_not_called()
        self.preferences.sync()
        reopened = QSettings(self.path, QSettings.Format.IniFormat)
        self.assertFalse(first_run_readiness(self.window, reopened, self.schedule))
        self.assertEqual(self.callbacks, [])
        self.assertTrue(first_run_readiness(self.window, reopened, self.schedule, force=True))

    def test_suppressed_or_closed_launch_does_not_mark_seen_or_present(self):
        self.assertFalse(first_run_readiness(self.window, self.preferences, self.schedule, enabled=False))
        self.assertFalse(self.preferences.value('readiness/introduction_seen', False, type=bool))
        self.assertTrue(first_run_readiness(self.window, self.preferences, self.schedule))
        self.window.closing = True
        self.callbacks.pop()()
        self.window.readiness.show.assert_not_called()
        self.assertFalse(self.preferences.value('readiness/introduction_seen', False, type=bool))


if __name__ == '__main__':
    unittest.main()
