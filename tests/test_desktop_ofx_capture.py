"""Actual Qt ownership and CPU capture/worker/cache pipeline, never host acceptance."""
import hashlib
import importlib.util
import json
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import test_desktop_window as fixtures


def capture_fixture(root):
    import numpy as np
    directory = root / 'capture'
    directory.mkdir()
    frames = []
    for index in range(3):
        rgb = np.arange(12 * 20 * 3, dtype=np.float32).reshape(12, 20, 3) / 200 - 1 + index
        rgb[0, 0] = [-0., 1.25, -2.]
        frames.append(rgb.astype('<f4').tobytes())
    payload = b''.join(frames)
    digests = [hashlib.sha256(b'CorridorKey/source/linear-rgb-f32/v1\0' + struct.pack('<III', i, 20, 12) + raw).hexdigest() for i, raw in enumerate(frames)]
    manifest = {'schema_version': 1, 'type': 'corridorkey_ofx_capture', 'capture_id': 'a' * 32,
                'source_scope': 'ofx_instance_upstream', 'host_start': -9, 'frame_count': 3,
                'width': 20, 'height': 12, 'fps': 24., 'pixel_aspect': 1.,
                'colour_contract': 'scene_linear_bt709_d65_opaque_rgb',
                'payload': {'path': 'frames.rgb32', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()},
                'source_digests': digests}
    (directory / 'frames.rgb32').write_bytes(payload)
    path = directory / 'capture.json'
    path.write_text(json.dumps(manifest))
    (directory / 'COMPLETE').write_bytes(b'')
    return path


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Optional Qt binding unavailable')
class DesktopCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DesktopWindowTests.setUpClass()

    def setUp(self):
        self.fixture = fixtures.DesktopWindowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window, self.root, self.app = self.fixture.window, self.fixture.root.resolve(), self.fixture.app
        self.window.open_ofx_capture()
        panel = self.window.ofx_capture
        assert panel is not None
        self.panel = panel

    def until(self, predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Capture condition not reached: ' + self.panel.status.text())

    def choices(self, *, native=False):
        if native:
            import test_ofx_capture_native as native_fixtures
            native_fixtures.OfxCaptureNativeTests.setUpClass()
            try:
                parent = self.root / 'native-captures'
                parent.mkdir()
                subprocess.run([str(native_fixtures.OfxCaptureNativeTests.binary), str(parent), 'success'],
                               check=True, capture_output=True, timeout=10)
                path = next(parent.iterdir()) / 'capture.json'
            finally:
                native_fixtures.OfxCaptureNativeTests.tearDownClass()
        else:
            path = capture_fixture(self.root)
        self.panel.manifest.setText(str(path))
        (self.root / 'scratch').mkdir()
        (self.root / 'cache').mkdir()
        self.panel.scratch.setText(str(self.root / 'scratch'))
        self.panel.output.setText(str(self.root / 'render'))
        self.panel.destination.setText(str(self.root / 'cache'))
        return path

    def test_opening_does_not_inspect_or_submit_work(self):
        self.assertTrue(self.panel.isVisible())
        self.assertIsNone(self.panel.operation)
        self.assertIsNone(self.panel.plan)
        self.assertFalse(self.window.jobs)
        self.assertFalse(self.window.bridge.done.is_set())
        self.assertIn('No capture inspected', self.panel.status.text())

    def test_reopen_refusals_preserve_dialog_until_successful_replacement(self):
        import shiboken6
        from PySide6.QtCore import QCoreApplication, QEvent

        old = self.panel
        old.close()
        config = self.window.config.toPlainText()
        for refusal in ('hint', 'profile', 'oversized'):
            with self.subTest(refusal=refusal):
                self.window.first_hint.setText('hint.png' if refusal == 'hint' else '')
                self.window.profile.setChecked(refusal == 'profile')
                self.window.config.setPlainText('x' * 65537 if refusal == 'oversized' else config)
                with patch.object(self.window, 'error') as error:
                    self.window.open_ofx_capture()
                    error.assert_called_once()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.assertTrue(shiboken6.isValid(old))
                self.assertIs(self.window.ofx_capture, old)
                self.window.poll()
        self.window.first_hint.clear()
        self.window.profile.setChecked(False)
        self.window.config.setPlainText(config)
        self.window.open_ofx_capture()
        self.assertIsNot(self.window.ofx_capture, old)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertFalse(shiboken6.isValid(old))
        self.window.poll()

    def test_cancel_delivery_failure_is_visible_and_retryable(self):
        self.panel.mark_submitted('capture-job')
        try:
            with patch.object(self.window.bridge, 'send', side_effect=[queue.Full('transient'), 'cancel-ack']) as send:
                self.panel.cancel()
                self.assertFalse(self.panel._cancel_sent)
                self.assertIn('delivery failed', self.panel.status.text())
                self.panel.cancel()
                self.assertTrue(self.panel._cancel_sent)
                self.assertEqual(send.call_count, 2)
                self.assertIn('Cancellation requested', self.panel.status.text())
                self.panel.cancel()
                self.assertEqual(send.call_count, 2)
        finally:
            self.panel.observe({'kind': 'cancelled', 'request_id': 'capture-job', 'payload': {}})

    def test_retained_history_display_is_bounded_without_altering_paths(self):
        publications = [{'type': 'cache_without_attachment', 'cache': f'/capture-{index}/' + 'é' * 16000,
                         'error': 'receipt unavailable'} for index in range(8)]
        self.panel.publications = list(publications)
        self.panel._show_publications()
        shown = self.panel.retained.toPlainText()
        self.assertLessEqual(len(shown.encode('utf-8')), 65536)
        self.assertIn('display truncated', shown)
        self.assertIn('capture-7', shown)
        self.assertEqual(self.panel.publications, publications)

    def test_failed_before_output_can_convert_and_submit_fresh_job(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.worker_client import WorkerClient
        self.choices()
        self.panel.verify()
        self.until(lambda: not self.panel.io_busy)
        self.panel.materialize()
        self.until(lambda: not self.panel.io_busy)
        previous = self.panel.prepared
        self.assertIsNotNone(previous, self.panel.status.text())
        preserved = {str(p.relative_to(previous.workspace)): p.read_bytes()
                     for p in previous.workspace.rglob('*') if p.is_file()}
        # The ordinary test worker requires .worker-control.json. Its absence
        # produces a real child failure before the requested output is created.
        self.panel.submit_render()
        self.until(lambda: not self.panel.rendering)
        previous_id = self.panel.submitted_id
        self.assertIsNotNone(previous_id)
        assert previous_id is not None
        self.assertEqual(self.window.jobs[previous_id]['status'], 'failed')
        self.assertFalse(Path(previous.job.output_dir).exists())
        self.panel.materialize()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNotNone(self.panel.prepared)
        self.assertNotEqual(self.panel.prepared.workspace, previous.workspace)
        self.assertIsNone(self.panel.submitted_id)
        self.assertIsNone(self.panel.record)
        self.assertFalse(self.panel._cancel_sent)
        self.assertTrue(self.panel.render_button.isEnabled())
        status = self.panel.status.text()
        for kind in ('result', 'progress', 'failed'):
            self.panel.observe({'kind': kind, 'request_id': previous_id,
                                'payload': {'record': {'stale': True}, 'stage': 'old job'}})
        self.assertIsNone(self.panel.record)
        self.assertEqual(self.panel.status.text(), status)
        old = self.window.bridge
        old.stop()
        old.thread.join(5)
        self.window.bridge = DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(Path(__file__).with_name('ofx_capture_worker_fixture.py'))]))
        self.panel.submit_render()
        self.assertNotEqual(self.panel.submitted_id, previous_id)
        self.until(lambda: not self.panel.rendering)
        self.assertIsNotNone(self.panel.record, self.panel.status.text())
        self.assertEqual(len(self.window.jobs), 2)
        self.panel.seal()
        self.until(lambda: not self.panel.io_busy)
        receipt = Path(self.panel.attachment.text())
        self.assertTrue(receipt.is_file(), self.panel.status.text())
        successful_id = self.panel.submitted_id
        # Simulate deliberate external removal of this fixture's render only;
        # a new conversion must not inherit its completed record or receipt UI.
        shutil.rmtree(self.panel.prepared.job.output_dir)
        self.panel.materialize()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.record)
        self.assertIsNone(self.panel.submitted_id)
        self.assertFalse(self.panel.seal_button.isEnabled())
        self.assertFalse(self.panel.attachment.text())
        self.panel.observe({'kind': 'result', 'request_id': successful_id,
                            'payload': {'record': {'stale': True}}})
        self.assertIsNone(self.panel.record)
        self.assertTrue(receipt.is_file())
        self.assertEqual(preserved, {str(p.relative_to(previous.workspace)): p.read_bytes()
                                     for p in previous.workspace.rglob('*') if p.is_file()})

    def test_receipt_failure_retains_committed_cache_without_attachment_claim(self):
        from corridorkey.control import CancelledError
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.worker_client import WorkerClient
        old = self.window.bridge
        old.stop()
        old.thread.join(5)
        self.window.bridge = DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(Path(__file__).with_name('ofx_capture_worker_fixture.py'))]))
        source = self.choices(native=True)
        original = {p.name: p.read_bytes() for p in source.parent.iterdir()}
        self.panel.verify()
        self.until(lambda: not self.panel.io_busy)
        self.panel.materialize()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNotNone(self.panel.prepared, self.panel.status.text())
        self.panel.submit_render()
        self.until(lambda: not self.panel.rendering)
        self.assertIsNotNone(self.panel.record, self.panel.status.text())
        for error in (OSError('receipt storage unavailable'), CancelledError('receipt cancelled')):
            with self.subTest(error=type(error).__name__), patch('corridorkey.ofx_capture.attachment_receipt', side_effect=error):
                count = len(self.panel.publications)
                self.panel.seal()
                self.until(lambda: not self.panel.io_busy)
                self.assertEqual(len(self.panel.publications), count + 1)
                publication = self.panel.publications[-1]
                self.assertEqual(publication['type'], 'cache_without_attachment')
                cache = Path(publication['cache'])
                self.assertEqual(cache.read_bytes()[:8], b'CKOFX002')
                self.assertIn(str(cache), self.panel.retained.toPlainText())
                self.assertIn(str(error), self.panel.status.text())
                self.assertIn('attachment unavailable', self.panel.status.text())
                self.assertFalse(self.panel.attachment.text())
                self.assertNotIn('receipt', publication)
        self.assertEqual(original, {p.name: p.read_bytes() for p in source.parent.iterdir()})

    def test_actual_postcommit_conversion_is_retained_after_stale_choices_and_close(self):
        from corridorkey.ofx_capture import materialize_capture
        source = self.choices()
        original = {p.name: p.read_bytes() for p in source.parent.iterdir()}
        for transition in ('stale', 'close'):
            with self.subTest(transition=transition):
                self.panel.manifest.setText(str(source))
                self.panel.verify()
                self.until(lambda: not self.panel.io_busy)
                self.assertIsNotNone(self.panel.plan, self.panel.status.text())
                entered, release = threading.Event(), threading.Event()
                committed = []
                def gated(*args, committed=committed, entered=entered, release=release, **kwargs):
                    prepared = materialize_capture(*args, **kwargs)
                    committed.append(prepared)
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError('postcommit fixture gate')
                    return prepared
                with patch('corridorkey.ofx_capture.materialize_capture', side_effect=gated):
                    self.panel.materialize()
                    operation = self.panel.operation
                    assert operation is not None
                    try:
                        self.assertTrue(entered.wait(3))
                        self.assertTrue((committed[0].workspace / 'COMPLETE').is_file())
                        if transition == 'stale':
                            self.panel.manifest.setText('superseded-capture.json')
                        else:
                            self.assertFalse(self.panel.shutdown())
                    finally:
                        release.set()
                    self.until(lambda: not self.panel.busy)
                self.assertFalse(operation.thread.is_alive())
                self.assertIsNone(self.panel.prepared)
                self.assertFalse(self.panel.render_button.isEnabled())
                self.assertFalse(self.window.jobs)
                self.assertIn(str(committed[0].workspace), self.panel.retained.toPlainText())
                self.assertTrue((committed[0].workspace / 'COMPLETE').is_file())
        publications = list(self.panel.publications)
        self.window.open_ofx_capture()
        reopened = self.window.ofx_capture
        assert reopened is not None
        self.assertEqual(reopened.publications, publications)
        self.assertEqual(original, {p.name: p.read_bytes() for p in source.parent.iterdir()})

    def test_actual_capture_conversion_shared_worker_and_explicit_cache_receipt(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.worker_client import WorkerClient
        old = self.window.bridge
        old.stop()
        old.thread.join(5)
        self.window.bridge = DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(Path(__file__).with_name('ofx_capture_worker_fixture.py'))]))
        source = self.choices(native=True)
        original = {p.name: p.read_bytes() for p in source.parent.iterdir()}
        settings = self.window.config.toPlainText()
        self.panel.inspect()
        self.until(lambda: not self.panel.io_busy)
        self.assertIn('Metadata only', self.panel.status.text())
        self.assertFalse(self.window.jobs)
        self.panel.verify()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNotNone(self.panel.plan, self.panel.status.text())
        self.assertIn('source_witness', self.panel.details.toPlainText())
        self.assertFalse(self.window.jobs)
        self.panel.materialize()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNotNone(self.panel.prepared, self.panel.status.text())
        self.assertFalse(self.window.jobs)
        self.panel.submit_render()
        self.until(lambda: not self.panel.rendering)
        self.assertIsNotNone(self.panel.record, self.panel.status.text())
        record = self.panel.record
        assert record is not None
        self.assertFalse(record['diagnostics']['execution']['neural_inference'])
        self.assertEqual(len(self.window.jobs), 1)
        for protected in (source.parent, self.panel.prepared.workspace):
            before = {str(p.relative_to(protected)): p.read_bytes() for p in protected.rglob('*') if p.is_file()}
            self.panel.destination.setText(str(protected))
            self.panel.seal()
            self.until(lambda: not self.panel.io_busy)
            self.assertIn('protected', self.panel.status.text().lower())
            self.assertFalse(self.panel.attachment.text())
            self.assertEqual(before, {str(p.relative_to(protected)): p.read_bytes() for p in protected.rglob('*') if p.is_file()})
        self.panel.destination.setText(str(self.root / 'cache'))
        self.panel.seal()
        self.until(lambda: not self.panel.io_busy)
        receipt = Path(self.panel.attachment.text())
        self.assertTrue(receipt.is_file(), self.panel.status.text())
        self.assertIn('CacheStart to -2', self.panel.status.text())
        self.assertEqual(settings, self.window.config.toPlainText())
        self.assertEqual(original, {p.name: p.read_bytes() for p in source.parent.iterdir()})

    def test_close_waits_for_blocked_owned_io_and_discards_late_result(self):
        entered, release = threading.Event(), threading.Event()
        def blocked(control):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test gate')
            return 'obsolete metadata'
        self.panel._start('Inspect', blocked)
        operation = self.panel.operation
        assert operation is not None
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(self.panel.shutdown())
            self.assertTrue(self.panel.isVisible())
            self.assertIs(self.panel.operation, operation)
        finally:
            release.set()
        self.until(lambda: not self.panel.busy)
        self.assertFalse(operation.thread.is_alive())
        self.assertNotIn('obsolete metadata', self.panel.details.toPlainText())

    def test_changed_choices_cancel_and_fence_old_metadata(self):
        entered, release = threading.Event(), threading.Event()
        def blocked(control):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test gate')
            return 'obsolete metadata'
        self.panel._start('Inspect', blocked)
        try:
            self.assertTrue(entered.wait(2))
            self.panel.manifest.setText('new-selection.json')
        finally:
            release.set()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.plan)
        self.assertNotIn('obsolete metadata', self.panel.details.toPlainText())

    def test_committed_terminal_survives_late_cancellation(self):
        from corridorkey.desktop_ofx_capture import CaptureOperation
        committed = self.root / 'committed-receipt.json'
        def operation(control):
            committed.write_text('explicit test artifact')
            control.cancellation.cancel('arrived after commit')
            return committed
        task = CaptureOperation('test publication', operation)
        task.thread.join(3)
        self.assertFalse(task.thread.is_alive())
        value, error = task.results.get_nowait()
        self.assertEqual(value, committed)
        self.assertIsNone(error)
        self.assertTrue(committed.is_file())

    def test_asset_switch_refuses_submit_without_enqueuing(self):
        from corridorkey.asset_inventory import AssetSelection
        path = self.choices()
        self.panel.verify()
        self.until(lambda: not self.panel.io_busy)
        self.panel.materialize()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNotNone(self.panel.prepared, self.panel.status.text())
        self.window.asset_selection = AssetSelection(self.root, None)
        self.panel.submit_render()
        self.assertFalse(self.window.jobs)
        self.assertIn('model assets changed', self.panel.status.text())
        self.assertTrue(path.is_file())

    def test_duplicate_original_settings_refuse_before_capture_hashes(self):
        self.choices()
        config, writer = self.panel.config_json, self.panel.writer_json
        for field, raw in (('config_json', '{"hann_chunk":8,"hann_chunk":9}'),
                           ('writer_json', '{"codec":"one","codec":"two"}')):
            with self.subTest(field=field), patch('corridorkey.ofx_capture.verify_capture') as verify:
                self.panel.config_json, self.panel.writer_json = config, writer
                setattr(self.panel, field, raw)
                self.panel.verify()
                self.until(lambda: not self.panel.io_busy)
                verify.assert_not_called()
                self.assertIn('duplicate', self.panel.status.text().lower())
                self.assertIsNone(self.panel.prepared)
                self.assertFalse(self.window.jobs)

    def test_postcommit_oversized_review_retains_workspace_truth(self):
        from corridorkey.ofx_capture import PreparedCapture
        self.choices()
        self.panel.verify()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNotNone(self.panel.plan, self.panel.status.text())
        with patch.object(PreparedCapture, 'summary', new=property(lambda _: {'oversized': 'x' * 65536})):
            self.panel.materialize()
            self.until(lambda: not self.panel.io_busy)
        self.assertIsNotNone(self.panel.prepared, self.panel.status.text())
        self.assertTrue((self.panel.prepared.workspace / 'COMPLETE').is_file())
        self.assertIn(str(self.panel.prepared.workspace), self.panel.status.text())
        self.assertFalse(self.panel.render_button.isEnabled())
        self.assertFalse(self.window.jobs)
