import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
FIXTURE = Path(__file__).with_name('worker_fixture.py')


@unittest.skipUnless(QT_AVAILABLE, 'Optional Qt binding unavailable')
class QueueRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['recovery-ui-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_bridge import DesktopBridge
        from corridorkey.desktop_window import DesktopWindow
        from corridorkey.queue_store import QueueStore
        from corridorkey.worker_client import WorkerClient
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.queue = QueueStore.create(self.root)
        self.window = DesktopWindow(DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(FIXTURE)]),
                                                 journal_workspace=self.queue))
        self.window.show()
        self.window.recovery.show()
        source = self.root / 'source.json'
        source.write_text('{}')
        self.batch = self.root / 'batch.json'
        self.batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch',
                         'jobs': [{'input': 'source.json', 'output_dir': f'out{index}',
                                   'settings': {'output_format': 'exr', 'outputs': ['alpha']}} for index in range(2)]}))
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        panel = self.window.recovery
        panel.batch.setText(str(self.batch))
        panel.workspace.setText(str(self.cache))
        panel.budget.setText('1048576')

    def tearDown(self):
        self.window.close()
        self.window.bridge.thread.join(10)
        if self.window.recovery.io_thread is not None:
            self.window.recovery.io_thread.join(5)
        self.window.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.window)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Recovery condition not reached: ' + self.window.recovery.status.text())

    def run_recovery(self, resume=False):
        prior = set(self.window.jobs)
        self.window.recovery.request(resume)
        added = set(self.window.jobs) - prior
        self.assertEqual(len(added), 1)
        request_id = added.pop()
        self.until(lambda: self.window.jobs[request_id]['status'] in {'result', 'failed', 'rejected', 'lost'})
        self.assertEqual(self.window.jobs[request_id]['status'], 'result', self.window.jobs[request_id]['event'])
        report = json.loads(Path(self.window.recovery.report.text()).read_text())
        return request_id, report

    def test_actual_render_restore_snapshot_and_verified_existing_resume(self):
        from corridorkey.queue_store import QueueStore
        first_id, report = self.run_recovery()
        self.assertEqual([job['mode'] for job in report['jobs']], ['render', 'verified_cache_restore'])
        saved = QueueStore.inspect(self.queue, request_id=first_id)['request']['payload']
        self.assertNotEqual(saved['batch'], str(self.batch))
        self.batch.write_text('{"changed_after_queue":true}')
        self.window.recovery.load_payload(saved)
        self.assertEqual(len(self.window.jobs), 1)
        _, resumed = self.run_recovery(True)
        self.assertEqual([job['mode'] for job in resumed['jobs']], ['verified_existing', 'verified_existing'])
        self.window.recovery.inspect_current()
        self.until(lambda: not self.window.recovery.io_busy)
        self.assertIn('verified_existing', self.window.recovery.details.toPlainText())
        self.assertIn('historical', self.window.recovery.details.toPlainText())

    def test_corrupted_existing_output_is_not_relabelled_complete(self):
        from corridorkey.queue_store import QueueStore
        first_id, report = self.run_recovery()
        original = Path(report['jobs'][0]['output_dir'])
        image = min((original / 'alpha').glob('*.exr'))
        image.write_bytes(b'corrupt native output')
        saved = QueueStore.inspect(self.queue, request_id=first_id)['request']['payload']
        self.window.recovery.load_payload(saved)
        _, resumed = self.run_recovery(True)
        self.assertEqual(resumed['jobs'][0]['mode'], 'verified_cache_restore')
        self.assertNotEqual(resumed['jobs'][0]['output_dir'], str(original))
        self.assertEqual(image.read_bytes(), b'corrupt native output')
        self.assertEqual(resumed['jobs'][1]['mode'], 'verified_existing')

    def test_invalid_controls_reject_without_silent_policy_changes(self):
        panel = self.window.recovery
        self.until(lambda: 'Durable queue:' in self.window.queue_label.text())
        self.assertIn('No recovery request submitted', panel.status.text())
        panel.budget.setText('-1')
        panel.request(False)
        self.assertFalse(self.window.jobs)
        self.assertIn('nonnegative', panel.status.text())
        panel.budget.setText('0')
        self.batch.write_text(self.batch.read_text().replace('"exr"', '"video"'))
        panel.request(False)
        request_id = next(iter(self.window.jobs))
        self.until(lambda: self.window.jobs[request_id]['status'] == 'rejected')
        self.assertIn('EXR', str(self.window.jobs[request_id]['event']))
        self.assertFalse((self.cache / 'workspace.json').exists())

    def test_manifest_mutation_after_queued_acceptance_cannot_change_execution(self):
        blocked = self.root / 'blocked.json'
        gate = self.root / 'gate'
        blocked.write_text(json.dumps({'gate': str(gate)}))
        self.window.set_source(str(blocked))
        self.window.output.setText(str(self.root))
        self.window.update_settings({'output_format': 'exr', 'outputs': ['alpha']})
        self.window.enqueue()
        first = next(iter(self.window.jobs))
        self.until(lambda: self.window.jobs[first]['status'] == 'progress')
        self.window.recovery.request(False)
        recovery = list(self.window.jobs)[1]
        self.until(lambda: self.window.jobs[recovery]['status'] == 'accepted')
        self.batch.write_text('{"invalid_after_admission":true}')
        gate.touch()
        self.until(lambda: self.window.jobs[recovery]['status'] in {'result', 'failed', 'lost'})
        self.assertEqual(self.window.jobs[recovery]['status'], 'result', self.window.jobs[recovery]['event'])
        report = json.loads(Path(self.window.recovery.report.text()).read_text())
        self.assertEqual([job['mode'] for job in report['jobs']], ['render', 'verified_cache_restore'])

    def test_zero_budget_and_durable_flags_reach_actual_controller(self):
        from corridorkey.queue_store import QueueStore
        panel = self.window.recovery
        panel.budget.setText('0')
        panel.foundations.setChecked(True)
        panel.rvm.setChecked(True)
        panel.hann.setChecked(True)
        request_id, report = self.run_recovery()
        self.assertTrue(report['durable_foundations_enabled'])
        self.assertTrue(report['durable_rvm_enabled'])
        self.assertTrue(report['durable_hann_enabled'])
        self.assertEqual([job['mode'] for job in report['jobs']], ['render', 'render'])
        saved = QueueStore.inspect(self.queue, request_id=request_id)['request']['payload']
        panel.load_payload(saved)
        self.assertEqual(panel.budget.text(), '0')
        _, resumed = self.run_recovery(True)
        self.assertEqual([job['mode'] for job in resumed['jobs']], ['verified_existing', 'verified_existing'])
