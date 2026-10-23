import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest
from corridorkey.worker_client import WorkerClient

HAS_STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'cv2'))
FIXTURE = Path(__file__).with_name('worker_fixture.py')


@unittest.skipUnless(HAS_STACK, 'requires explicit CPU native export fixture, never pretrained inference')
class WorkerProcessTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def job(self, name, data=None):
        source = self.root / f'{name}.json'
        source.write_text(json.dumps(data or {}))
        return JobRequest.create(source, self.root / name, InferenceConfig(output_format='exr', outputs={'alpha'}))

    def client(self, capacity=4):
        client = WorkerClient(capacity=capacity, command=[sys.executable, str(FIXTURE), '--capacity', str(capacity)])
        self.addCleanup(client.close)
        return client

    def wait(self, client, kind, name):
        while True:
            event = client.next_event(10)
            if event['kind'] == kind and event['request_id'] == name:
                return event
            if event['kind'] in ('lost', 'failed', 'rejected'):
                self.fail(str(event))

    def test_child_reuses_model_and_keeps_python_native_logs_off_protocol(self):
        client = self.client()
        records = []
        for name in ('first', 'second'):
            request_id = client.submit(self.job(name))
            result = self.wait(client, 'result', request_id)['payload']['record']
            records.append(result)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['diagnostics']['execution']['fixture_loads'], 1)
            self.assertEqual(len(list(Path(result['diagnostics']['outputs']['alpha']).glob('*.exr'))), 2)
        self.assertFalse(records[0]['model_reused'])
        self.assertTrue(records[1]['model_reused'])
        self.assertTrue(client.close())
        self.assertIn('Python engine log', client.log_tail)
        self.assertIn('native fd1 log', client.log_tail)

    def test_controls_cancel_queued_and_running_work_without_blocking(self):
        client = self.client(capacity=1)
        active = client.submit(self.job('blocked', {'gate': str(self.root / 'gate')}))
        self.wait(client, 'progress', active)
        queued = client.submit(self.job('queued'))
        self.wait(client, 'accepted', queued)
        client.cancel(queued)
        self.wait(client, 'cancelled', queued)
        client.cancel(active)
        self.wait(client, 'cancelled', active)
        good = client.submit(self.job('good'))
        result = self.wait(client, 'result', good)['payload']['record']
        self.assertEqual(result['diagnostics']['execution']['fixture_loads'], 2)
        self.assertFalse((self.root / 'queued').exists())

    def test_observation_timeout_never_restarts_a_live_worker(self):
        client = self.client()
        gate = self.root / 'gate'
        request_id = client.submit(self.job('blocked', {'gate': str(gate)}))
        self.wait(client, 'progress', request_id)
        pid = client.process.pid
        with self.assertRaises(RuntimeError):
            client.restart()
        # Progress remains coalesced while the caller temporarily stops consuming.
        self.assertIsNone(client.process.poll())
        self.assertEqual(client.process.pid, pid)
        gate.touch()
        self.wait(client, 'result', request_id)

    def test_actual_hard_child_exit_marks_job_lost_and_explicit_restart_is_new_generation(self):
        client = self.client()
        generation = client.generation
        if os.name != 'nt':
            # Restart requires proven descendant discovery before the leader
            # exits. Keep the ready fixture idle until that prerequisite holds;
            # first discovery after a hard exit correctly remains fail-closed.
            with client.kill_lock:
                client._capture_descendants()
            self.assertIsNotNone(client.containment)
            assert client.containment is not None
            self.assertGreater(client.containment.complete_samples, 0)
            self.assertIsNone(client.containment.failure)
        request_id = client.submit(self.job('crash', {'exit': True}))
        self.wait(client, 'lost', request_id)
        client.process.wait(timeout=3)
        replacement = client.restart()
        self.addCleanup(replacement.close)
        self.assertNotEqual(replacement.generation, generation)
        result_id = replacement.submit(self.job('after'))
        self.wait(replacement, 'result', result_id)

    def manifest(self):
        source = self.root / 'source.json'
        source.write_text('{}')
        manifest = self.root / 'batch.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch',
                                       'jobs': [{'input': str(source), 'output_dir': str(self.root / name),
                                                 'settings': {'output_format': 'exr', 'outputs': ['alpha']}}
                                                for name in ('one', 'two')]}))
        return manifest

    def test_recovery_controller_runs_in_child_and_resume_keeps_completed_outputs(self):
        client = self.client()
        payload = {'batch': str(self.manifest()), 'workspace': str(self.root / 'workspace'), 'max_cache_bytes': 100000}
        request_id = client.send('recover', payload)
        result = self.wait(client, 'result', request_id)['payload']['record']
        report = json.loads(Path(result['report_path']).read_text())
        self.assertEqual([job['mode'] for job in report['jobs']], ['render', 'verified_cache_restore'])
        self.assertEqual(result['fixture_loads'], 1)
        request_id = client.send('recover', {**payload, 'resume': True})
        resumed = self.wait(client, 'result', request_id)['payload']['record']
        report = json.loads(Path(resumed['report_path']).read_text())
        self.assertEqual([job['mode'] for job in report['jobs']], ['verified_existing', 'verified_existing'])
        self.assertEqual(resumed['fixture_loads'], 1)

    def test_cli_client_writes_actual_native_job_and_failure_reports(self):
        from corridorkey.worker_client import run_worker_batch

        def factory(**kwargs):
            return WorkerClient(command=[sys.executable, str(FIXTURE)], **kwargs)

        report_path = self.root / 'report.json'
        report = run_worker_batch(self.manifest(), report_path, client_factory=factory)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(len(report['jobs']), 2)
        self.assertTrue(report['jobs'][1]['record']['model_reused'])
        self.assertEqual(json.loads(report_path.read_text())['status'], 'complete')
        self.assertTrue(report['worker_clean_stop'])
        previous_report = report_path.read_bytes()
        failed = run_worker_batch(self.root / 'missing.json', report_path)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['error']['type'], 'FileNotFoundError')
        self.assertEqual(failed['report_write']['status'], 'not_written')
        self.assertEqual(report_path.read_bytes(), previous_report)

    def test_production_child_reports_unsupported_backend_without_fixture_fallback(self):
        client = WorkerClient()
        self.addCleanup(client.close)
        request_id = client.submit(self.job('native'))
        event = self.wait(client, 'failed', request_id)
        self.assertEqual(event['payload']['record']['status'], 'failed')
        self.assertIn('CUDA', event['payload']['error']['message'])
        self.assertFalse((self.root / 'native' / 'alpha').exists())


    @unittest.skipUnless(os.name == 'posix', 'requires actual POSIX worker signal delivery')
    def test_manual_sigint_cancels_owner_and_flushes_terminal_record(self):
        import signal

        client = self.client()
        request_id = client.submit(self.job('interrupt', {'gate': str(self.root / 'gate')}))
        self.wait(client, 'progress', request_id)
        os.kill(client.process.pid, signal.SIGINT)
        event = self.wait(client, 'cancelled', request_id)
        self.assertEqual(event['payload']['record']['status'], 'cancelled')
        self.assertEqual(client.process.wait(timeout=3), 130)
