"""Local tiny-format audit UI, explicit fresh report export and owned close."""
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


def populate_audit(root: Path):
    import numpy as np
    from safetensors.numpy import save_file

    from corridorkey.runtime_paths import RuntimePaths
    paths = RuntimePaths.from_package(root)
    for path in (paths.weights, paths.foundation_stats):
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {'format': 'corridorkey_v2_runtime_checkpoint', 'model_config': '{}', 'metadata': '{}'} if path == paths.weights else None
        save_file({'fixture': np.zeros((1,), dtype=np.float32)}, path, metadata=metadata)
    paths.rvm_ckpt.parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths.rvm_ckpt, fixture=np.zeros((1,), dtype=np.float32))
    paths.moge_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.moge_checkpoint.write_bytes(b'legacy .pt fixture; never deserialized')
    paths.cradio_dir.mkdir(parents=True)
    (paths.cradio_dir / 'config.json').write_text('{}')
    return paths


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt binding unavailable')
class DesktopAssetAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['asset-audit-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_assets import AssetsPanel
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name).resolve()
        self.paths = populate_audit(self.root)
        self.owner = AssetsPanel()
        self.owner.root.setText(str(self.root))
        self.owner.show()
        self.owner.open_audit()
        self.panel = self.owner.audit_panel
        self.used = []
        self.owner.use_requested.connect(self.used.append)

    def tearDown(self):
        self.owner.close()
        if self.panel.task is not None:
            self.panel.task.thread.join(5)
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
        self.fail('Asset audit condition not reached')

    def finish(self):
        self.until(lambda: not self.owner.io_busy)

    def test_actual_tiny_format_audit_explicit_export_and_stale_source_refusal(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        self.assertIsNone(self.panel.task)
        self.panel.start()
        self.finish()
        self.assertIsNotNone(self.panel.report)
        assert self.panel.report is not None
        report = json.loads(self.panel.report)
        weights = next(entry for entry in report['entries'] if entry['asset'] == 'weights')
        self.assertEqual(weights['format_status'], 'PASS')
        self.assertEqual(weights['sha256'], hashlib.sha256(self.paths.weights.read_bytes()).hexdigest())
        self.assertIn('Model compatibility: NOT VERIFIED', self.panel.details.toPlainText())
        self.assertIn('Publisher authenticity: NOT VERIFIED', self.panel.details.toPlainText())
        self.assertEqual(self.used, [])
        destination = self.root / 'audit.json'
        self.panel.save_report(destination)
        self.finish()
        self.assertEqual(json.loads(destination.read_bytes()), report)
        self.assertEqual(self.panel.export_path.text(), str(destination))
        self.assertIn('Report export complete', self.panel.status.text())
        for name, raw in before.items():
            self.assertEqual((self.root / name).read_bytes(), raw)
        preserved = destination.read_bytes()
        self.paths.weights.write_bytes(self.paths.weights.read_bytes() + b'changed fixture')
        refused = self.root / 'stale.json'
        self.panel.save_report(refused)
        self.finish()
        self.assertFalse(refused.exists())
        self.assertIn('Report export failed', self.panel.status.text())
        self.assertFalse(self.panel.save_button.isEnabled())
        self.assertEqual(destination.read_bytes(), preserved)
        self.assertEqual(self.used, [])

    def test_legacy_checkpoint_unverified_does_not_replace_optional_presence_use(self):
        legacy = self.root / 'legacy.pt'
        legacy.write_bytes(b'legacy fixture; no pickle loading')
        self.owner.weights.setText(str(legacy))
        with patch('corridorkey.desktop_assets.inspect_backend', return_value={'status': 'fixture; backend not validated'}):
            self.owner.inspect()
            self.finish()
        self.assertTrue(self.owner.ready)
        self.owner.open_audit()
        self.panel.start()
        self.finish()
        assert self.panel.report is not None
        data = json.loads(self.panel.report)
        weights = next(entry for entry in data['entries'] if entry['asset'] == 'weights')
        self.assertEqual(weights['format_status'], 'UNVERIFIED')
        self.assertTrue(self.owner.ready)
        self.assertEqual(self.used, [])
        self.owner.use()
        self.assertEqual(len(self.used), 1)
        self.assertEqual(self.used[0].weights, legacy)

    def test_parent_close_waits_unexpected_exit_and_excludes_presence_check(self):
        entered, release = threading.Event(), threading.Event()
        names = []

        def unexpected(selection, *, check=None, progress=None):
            names.append(threading.current_thread().name)
            entered.set()
            release.wait(5)
            raise SystemExit('unexpected audit exit')

        with patch('corridorkey.asset_audit.audit_assets', side_effect=unexpected):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            assert self.panel.task is not None
            task = self.panel.task
            self.owner.inspect()
            self.assertIsNone(self.owner.io_thread)
            self.panel.start()
            self.assertIs(self.panel.task, task)
            self.owner.close()
            self.assertTrue(self.owner.isVisible())
            self.assertTrue(task.cancellation.cancelled)
            release.set()
            self.until(lambda: not self.owner.isVisible())
        self.assertFalse(task.thread.is_alive())
        self.assertTrue(task.done.is_set())
        self.assertIn('SystemExit', self.panel.status.text())
        self.assertEqual(names, ['local-asset-audit'])
        self.assertEqual(task.results.maxsize, 1)

    def test_inflight_selection_edit_invalidates_report_and_coalesces_progress(self):
        from corridorkey.control import ProgressEvent
        entered, release = threading.Event(), threading.Event()
        captured = []

        def delayed(selection, *, check=None, progress=None):
            captured.append(selection)
            assert progress is not None
            for count in range(1001):
                progress(ProgressEvent('asset_hash', count, 1000))
            entered.set()
            release.wait(5)
            return {'status': 'complete', 'entries': [], 'error': None}

        with patch('corridorkey.asset_audit.audit_assets', side_effect=delayed):
            self.panel.start()
            self.assertTrue(entered.wait(2))
            self.owner.root.clear()
            assert self.panel.task is not None
            event = self.panel.task.progress()
            assert event is not None
            self.assertEqual(event.completed, 1000)
            release.set()
            self.finish()
        self.assertEqual(captured[0].root, self.root)
        self.assertIsNone(self.panel.report)
        self.assertFalse(self.panel.save_button.isEnabled())
        self.assertIn('Selection changed', self.panel.status.text())

    def test_saved_commit_survives_late_cancel_and_partial_failure_path_is_visible(self):
        self.panel.start()
        self.finish()
        entered, release = threading.Event(), threading.Event()
        destination = self.root / 'committed.json'
        alias = self.root / 'committed-private-alias'

        def committed(report, output_path, *, check=None):
            destination.write_text('{}')
            entered.set()
            release.wait(5)
            return {'status': 'complete', 'destination': str(destination), 'usable': True,
                    'bytes': 2, 'error': None, 'warning': 'Private stage cleanup failed',
                    'partial_path': str(alias)}

        with patch('corridorkey.asset_audit.export_audit', side_effect=committed):
            self.panel.save_report(destination)
            self.assertTrue(entered.wait(2))
            self.panel.cancel()
            release.set()
            self.finish()
        self.assertIn('Report export complete', self.panel.status.text())
        self.assertEqual(self.panel.export_path.text(), str(destination))
        self.assertIn('Private report alias retained; published report is complete: ' + str(alias),
                      self.panel.details.toPlainText())
        self.assertNotIn('Partial report retained (not complete)', self.panel.details.toPlainText())
        partial = self.root / 'partial-stage'
        with patch('corridorkey.asset_audit.export_audit', return_value={'status': 'failed', 'destination': None,
                    'partial_path': str(partial), 'usable': False, 'error': 'injected write failure'}):
            self.panel.save_report(self.root / 'failed.json')
            self.finish()
        self.assertEqual(self.panel.export_path.text(), str(partial))
        self.assertIn('not complete', self.panel.details.toPlainText())
        self.assertFalse(self.panel.save_button.isEnabled())
