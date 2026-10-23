"""Local inventory is metadata-only; changing assets never reuses a live owner."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.asset_inventory import AssetSelection, inspect_assets, inspect_backend
from corridorkey.runtime_paths import RuntimePaths
from tests.qt_cleanup import dispose_widget

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def populate(root: Path):
    paths = RuntimePaths.from_package(root)
    for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint'):
        path = getattr(paths, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not a model; metadata fixture only')
    paths.cradio_dir.mkdir(parents=True)
    (paths.cradio_dir / 'config.json').write_text('{}')


class InventoryTests(unittest.TestCase):
    def test_missing_empty_wrong_type_and_present_are_reported_without_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            selection = AssetSelection(root)
            missing = inspect_assets(selection)
            self.assertFalse(missing['assets_present'])
            self.assertEqual(len(missing['entries']), 5)
            populate(root)
            with patch.dict('sys.modules', {'torch': None}):
                self.assertTrue(inspect_assets(selection)['assets_present'])
            paths = RuntimePaths.from_package(root)
            paths.weights.write_bytes(b'')
            self.assertFalse(inspect_assets(selection)['assets_present'])
            paths.weights.unlink()
            paths.weights.mkdir()
            self.assertFalse(inspect_assets(selection)['assets_present'])

    def test_backend_probe_reports_unavailable_and_probe_failure_truthfully(self):
        import subprocess
        with patch('corridorkey.asset_inventory.subprocess.run') as run:
            run.return_value.stdout = '{"cuda_available":false,"torch":"fixture","cuda_build":null}'
            self.assertFalse(inspect_backend()['cuda_available'])
            self.assertIn('unsupported', inspect_backend()['status'])
            run.side_effect = subprocess.TimeoutExpired('fixture', 20)
            self.assertIsNone(inspect_backend()['cuda_available'])


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt not installed')
class AssetsPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['assets-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_window import DesktopWindow
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name).resolve()
        populate(self.root)
        self.window = DesktopWindow()

    def tearDown(self):
        self.window.close()
        self.window.bridge.thread.join(10)
        if self.window.assets.audit_panel.task is not None:
            self.window.assets.audit_panel.task.thread.join(5)
        if self.window.assets.download_panel.task is not None:
            self.window.assets.download_panel.task.thread.join(5)
        if self.window.assets.io_thread is not None:
            self.window.assets.io_thread.join(5)
        self.window.poll()
        self.app.processEvents()
        dispose_widget(self.window)
        self.folder.cleanup()

    def test_inventory_completion_and_path_change_cannot_silently_apply(self):
        panel = self.window.assets
        panel.root.setText(str(self.root))
        panel.use()
        self.assertIn('Check', panel.status.text())
        with patch('corridorkey.desktop_assets.inspect_backend', return_value={'status': 'fixture, not backend validation'}):
            panel.inspect()
            assert panel.io_thread is not None
            panel.io_thread.join(10)
        panel.poll()
        self.assertTrue(panel.ready)
        panel.use()
        self.assertIn('Stop worker', panel.status.text())
        panel.weights.setText(str(self.root / 'different'))
        panel.use()
        self.assertIn('Check', panel.status.text())
        self.assertIsNone(self.window.asset_selection)

    def test_clean_stop_changes_future_factory_only_and_keeps_no_job_submitted(self):
        from functools import partial
        self.window.bridge.stop()
        self.window.bridge.thread.join(10)
        selection = AssetSelection(self.root)
        self.window.activate_assets(selection)
        self.assertEqual(self.window.asset_selection, selection)
        assert isinstance(self.window.bridge.factory, partial)
        self.assertEqual(self.window.bridge.factory.keywords['asset_root'], self.root)
        self.assertFalse(self.window.jobs)
        self.assertTrue(self.window.bridge.commands.empty())
        self.assertIn('No model loaded', self.window.backend.text())

    def test_queue_activation_during_refresh_prevents_second_asset_owner(self):
        self.window.bridge.stop()
        self.window.bridge.thread.join(10)
        self.window.queue_panel.run_io('create', self.root)
        assert self.window.queue_panel.io_thread is not None
        self.window.queue_panel.io_thread.join(10)
        self.window.activate_assets(AssetSelection(self.root))
        self.assertIsNone(self.window.asset_selection)
        self.assertIn('ownership changed', self.window.assets.status.text())
        self.assertIsNotNone(self.window.bridge.journal_workspace)

    def test_uncertain_benchmark_cleanup_blocks_asset_and_queue_owners(self):
        from unittest.mock import Mock
        self.window.bridge.stop()
        self.window.bridge.thread.join(10)
        previous = self.window.bridge
        task = Mock()
        task.done.is_set.return_value = True
        task.snapshot.return_value = {'revision': self.window.benchmark.revision, 'cleanup_error': 'fixture uncertainty'}
        self.window.benchmark.task = task
        try:
            self.window.activate_assets(AssetSelection(self.root))
            self.assertIn('cleanup is unverified', self.window.assets.status.text())
            self.assertIs(self.window.bridge, previous)
            self.window.activate_queue(self.root / 'queue')
            self.assertIn('cleanup is unverified', self.window.queue_panel.status.text())
            self.assertIsNone(self.window.pending_queue_root)
        finally:
            self.window.benchmark.task = None
