"""Explicit download UI with injected metadata/payload functions, never remote weights."""
import hashlib
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
PAYLOAD = b'explicit download UI fixture'


def plan_fixture():
    revision = 'a' * 40
    return {'schema_version': 1, 'kind': 'test_only_download_plan', 'token': 'fixture-token',
            'publishers': [{'asset': 'fixture', 'publisher': 'Test fixture publisher', 'repository': 'fixture/repository',
                            'licence': 'test-only fixture', 'licence_url': 'https://example.invalid/licence',
                            'requested_revision': revision, 'resolved_revision': revision}],
            'files': [{'asset': 'fixture', 'path': 'weights/corridorkey_v2.safetensors', 'repository': 'fixture/repository',
                       'revision': revision, 'bytes': len(PAYLOAD),
                       'checksum': {'algorithm': 'sha256', 'value': hashlib.sha256(PAYLOAD).hexdigest()}}],
            'total_bytes': len(PAYLOAD)}


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt binding unavailable')
class DesktopAssetDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['asset-download-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_assets import AssetsPanel
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name).resolve()
        self.current = self.root / 'current-assets'
        self.current.mkdir()
        self.destination = self.root / 'destination'
        self.destination.mkdir()
        self.owner = AssetsPanel()
        self.owner.root.setText(str(self.current))
        self.owner.show()
        self.panel = self.owner.download_panel
        self.used = []
        self.owner.use_requested.connect(self.used.append)

    def tearDown(self):
        self.owner.close()
        for panel in (self.owner.audit_panel, self.panel):
            if panel.task is not None:
                panel.task.thread.join(5)
        if self.owner.io_thread is not None:
            self.owner.io_thread.join(5)
        self.owner.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.owner)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.owner.poll()
            if predicate():
                return
            time.sleep(.005)
        self.fail('Download UI condition not reached')

    def finish(self):
        self.until(lambda: not self.owner.io_busy)

    def prepared(self):
        self.owner.open_download()
        self.finish()
        self.panel.destination_parent.setText(str(self.destination))
        self.panel.budget.setText(str(len(PAYLOAD)))
        with patch('corridorkey.asset_download.resolve_plan', return_value={'status': 'complete', 'plan': plan_fixture(), 'error': None}):
            self.panel.resolve_plan()
            self.finish()
        self.assertTrue(self.panel.download_button.isEnabled())

    def test_open_is_offline_and_explicit_select_fills_paths_without_use(self):
        from corridorkey.asset_inventory import AssetSelection
        with patch('corridorkey.asset_download.resolve_plan', side_effect=AssertionError('No online request on open')) as resolve:
            self.owner.open_download()
            self.finish()
            resolve.assert_not_called()
        self.assertTrue(self.panel.catalog_loaded)
        self.assertIn('Payload sizes unknown', self.panel.status.text())
        self.prepared()
        self.owner.checked = AssetSelection(self.current)
        self.owner.ready = True
        captured = []
        downloaded = self.destination / 'downloaded-fixture'

        def download(plan, parent, budget, *, protected_paths=(), check=None, progress=None):
            captured.append((plan, parent, budget, protected_paths))
            weights = downloaded / 'weights/corridorkey_v2.safetensors'
            weights.parent.mkdir(parents=True)
            weights.write_bytes(PAYLOAD)
            return {'status': 'complete', 'destination': str(downloaded), 'usable': True,
                    'copied_bytes': len(PAYLOAD), 'copied_files': 1,
                    'weights_relative': 'weights/corridorkey_v2.safetensors', 'error': None}

        with patch('corridorkey.asset_download.download_plan', side_effect=download):
            self.panel.download()
            self.finish()
        self.assertEqual(self.owner.root.text(), str(self.current))
        self.assertEqual(self.used, [])
        self.assertTrue(self.panel.select_button.isEnabled())
        self.assertEqual(captured[0][1:], (self.destination, len(PAYLOAD), (self.current,)))
        self.panel.select_downloaded()
        self.assertEqual(self.owner.root.text(), str(downloaded))
        self.assertEqual(self.owner.weights.text(), str(downloaded / 'weights/corridorkey_v2.safetensors'))
        self.assertIsNone(self.owner.checked)
        self.assertFalse(self.owner.ready)
        self.assertEqual(self.used, [])
        self.assertIn('Check local assets/backend', self.owner.status.text())
        self.assertFalse(self.panel.select_button.isEnabled())

    def test_editing_destination_budget_does_not_repeat_online_resolution(self):
        self.prepared()
        plan = self.panel.plan
        self.panel.budget.setText('0')
        self.assertFalse(self.panel.download_button.isEnabled())
        self.assertEqual(self.panel.plan, plan)
        self.panel.budget.setText(str(len(PAYLOAD)))
        self.assertTrue(self.panel.download_button.isEnabled())
        self.panel.destination_parent.clear()
        self.assertFalse(self.panel.download_button.isEnabled())
        self.assertEqual(self.panel.plan, plan)
        self.panel.budget.setText('9223372036854775808')
        self.assertFalse(self.panel.download_button.isEnabled())
        self.assertIn('exact payload byte budget', self.panel.review.text())

    def test_parent_close_waits_unexpected_download_exit_and_excludes_siblings(self):
        self.prepared()
        entered, release = threading.Event(), threading.Event()
        names = []

        def unexpected(*args, **kwargs):
            names.append(threading.current_thread().name)
            entered.set()
            release.wait(5)
            raise SystemExit('unexpected download exit')

        with patch('corridorkey.asset_download.download_plan', side_effect=unexpected):
            self.panel.download()
            self.assertTrue(entered.wait(2))
            assert self.panel.task is not None
            task = self.panel.task
            self.owner.inspect()
            self.assertIsNone(self.owner.io_thread)
            self.owner.audit_panel.start()
            self.assertIsNone(self.owner.audit_panel.task)
            self.assertIn('Wait for current asset', self.owner.audit_panel.status.text())
            self.owner.close()
            self.assertTrue(self.owner.isVisible())
            self.assertTrue(task.cancellation.cancelled)
            release.set()
            self.until(lambda: not self.owner.isVisible())
        self.assertFalse(task.thread.is_alive())
        self.assertTrue(task.done.is_set())
        self.assertIn('SystemExit', self.panel.status.text())
        self.assertEqual(names, ['model-asset-download'])
        self.assertEqual(task.results.maxsize, 1)

    def test_inflight_edits_do_not_retarget_plan_or_enable_selection(self):
        self.prepared()
        entered, release = threading.Event(), threading.Event()
        captured = []

        def committed(plan, parent, budget, *, protected_paths=(), check=None, progress=None):
            captured.append((plan, parent, budget, protected_paths))
            entered.set()
            release.wait(5)
            return {'status': 'complete', 'destination': str(parent / 'committed'), 'usable': True,
                    'copied_files': 1, 'copied_bytes': len(PAYLOAD),
                    'weights_relative': 'weights/corridorkey_v2.safetensors'}

        with patch('corridorkey.asset_download.download_plan', side_effect=committed):
            self.panel.download()
            self.assertTrue(entered.wait(2))
            self.panel.destination_parent.setText(str(self.root))
            self.panel.budget.setText(str(len(PAYLOAD) + 1))
            self.owner.root.setText(str(self.root / 'other-selection'))
            self.panel.cancel()
            release.set()
            self.finish()
        self.assertEqual(captured[0][1:], (self.destination, len(PAYLOAD), (self.current,)))
        self.assertIn('Download complete', self.panel.status.text())
        self.assertEqual(self.panel.result_path.text(), str(self.destination / 'committed'))
        self.assertFalse(self.panel.select_button.isEnabled())
        self.assertEqual(self.used, [])

    def test_partial_cancelled_destination_and_unknown_bytes_are_truthful(self):
        self.prepared()
        partial = self.destination / 'partial-fixture'
        with patch('corridorkey.asset_download.download_plan', return_value={'status': 'cancelled',
                   'destination': str(partial), 'usable': False, 'copied_files': 0, 'copied_bytes': None,
                   'partial_files': [{'path': 'weights/incomplete', 'bytes': 3}], 'error': 'fixture cancellation'}):
            self.panel.download()
            self.finish()
        self.assertEqual(self.panel.result_path.text(), str(partial))
        self.assertIn('Unknown bytes', self.panel.status.text())
        self.assertIn('Partial files retained: 1', self.panel.partial_files.text())
        self.assertIn('weights/incomplete: 3 bytes', self.panel.partial_files.text())
        self.assertFalse(self.panel.select_button.isEnabled())

    def test_unknown_plan_size_never_becomes_zero_byte_download(self):
        self.owner.open_download()
        self.finish()
        plan = plan_fixture()
        plan['total_bytes'] = None
        with patch('corridorkey.asset_download.resolve_plan', return_value={'status': 'complete', 'plan': plan}):
            self.panel.resolve_plan()
            self.finish()
        self.assertIsNone(self.panel.plan)
        self.assertFalse(self.panel.download_button.isEnabled())
        self.assertIn('unknown or invalid', self.panel.status.text())

    def test_actual_owned_http_plan_download_and_explicit_path_selection(self):
        from corridorkey import asset_download
        from corridorkey.asset_inventory import inspect_assets
        from tests.test_asset_download import PublisherFixture

        fixture = PublisherFixture()
        try:
            before = {str(path.relative_to(fixture.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in fixture.root.rglob('*') if path.is_file()}
            resolve = asset_download.resolve_plan
            download = asset_download.download_plan
            self.owner.open_download()
            self.finish()
            self.assertEqual(fixture.requests, [])
            with patch('corridorkey.asset_download.resolve_plan', side_effect=lambda **kwargs: resolve(transport=fixture.transport, **kwargs)), \
                 patch('corridorkey.asset_download.download_plan', side_effect=lambda *args, **kwargs: download(*args, transport=fixture.transport, **kwargs)):
                self.panel.resolve_plan()
                self.finish()
                self.assertIsNotNone(self.panel.plan)
                assert self.panel.plan is not None
                plan = json.loads(self.panel.plan)
                self.assertTrue(fixture.requests)
                self.assertEqual(list(self.destination.iterdir()), [])
                self.panel.destination_parent.setText(str(self.destination))
                self.panel.budget.setText(str(plan['total_bytes']))
                self.assertTrue(self.panel.download_button.isEnabled())
                self.panel.download()
                self.finish()
            downloaded = Path(self.panel.result_path.text())
            self.assertEqual(downloaded.parent, self.destination)
            self.assertTrue(self.panel.select_button.isEnabled(), self.panel.status.text())
            self.assertEqual(self.owner.root.text(), str(self.current))
            self.panel.select_downloaded()
            self.assertEqual(self.owner.root.text(), str(downloaded))
            self.assertTrue(inspect_assets(self.owner.selection())['assets_present'])
            self.assertEqual(self.used, [])
            self.assertFalse(self.owner.ready)
            self.assertEqual(before, {str(path.relative_to(fixture.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in fixture.root.rglob('*') if path.is_file()})
        finally:
            fixture.close()
