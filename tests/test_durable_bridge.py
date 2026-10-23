import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.config import InferenceConfig
from corridorkey.desktop_bridge import DesktopBridge
from corridorkey.queue_store import QueueStore
from corridorkey.session import JobRequest
from corridorkey.worker_client import WorkerClient
from corridorkey.worker_protocol import job_to_wire

FIXTURE = Path(__file__).with_name('worker_fixture.py')


class DurableBridgeTests(unittest.TestCase):
    def wait(self, bridge, kind, request_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for event in bridge.poll():
                if event['kind'] == kind and event['request_id'] == request_id:
                    return event
            time.sleep(.01)
        self.fail(f'No {kind} event: {bridge.failure}')

    def setup_bridge(self, root):
        workspace = QueueStore.create(root)
        bridge = DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(FIXTURE)]), journal_workspace=workspace)
        return workspace, bridge

    def test_real_child_terminal_event_is_persisted_before_frontend_receives_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.json'
            source.write_text('{}')
            workspace, bridge = self.setup_bridge(root)
            try:
                job = JobRequest.create(source, root / 'out', InferenceConfig(output_format='exr'))
                payload = {'job': job_to_wire(job)}
                request_id = bridge.send('render', payload)
                payload['job']['output_dir'] = str(root / 'changed-after-submit')
                self.wait(bridge, 'result', request_id)
                record = QueueStore.inspect(workspace, request_id=request_id)
                self.assertEqual(record['status'], 'result')
                self.assertEqual(record['request']['payload']['job']['output_dir'], job.output_dir)
                self.assertTrue((Path(job.output_dir) / 'alpha').is_dir())
                self.assertFalse((root / 'changed-after-submit').exists())
            finally:
                bridge.stop()
                bridge.thread.join(10)
            self.assertTrue(bridge.done.is_set())
            self.assertIsNone(bridge.failure)
            self.assertEqual(QueueStore.inspect(workspace)['rows'][0]['clean'], 1)

    def test_failed_intent_publication_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            workspace = QueueStore.create(root)
            with patch('corridorkey.desktop_bridge.QueueStore.prepare', side_effect=OSError('disk full')), \
                 patch('corridorkey.desktop_bridge.WorkerClient') as factory:
                bridge = DesktopBridge(factory, journal_workspace=workspace)
                try:
                    request_id = bridge.send('render', {'job': job_to_wire(JobRequest.create(root / 'input', root / 'output'))})
                    event = self.wait(bridge, 'rejected', request_id)
                    self.assertIn('disk full', event['payload']['error'])
                    factory.assert_not_called()
                finally:
                    bridge.stop()
                    bridge.thread.join(10)
            self.assertFalse((root / 'output').exists())

    def test_actual_worker_crash_is_saved_as_lost_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.json'
            source.write_text(json.dumps({'exit': True}))
            workspace, bridge = self.setup_bridge(root)
            try:
                job = JobRequest.create(source, root / 'out', InferenceConfig(output_format='exr'))
                request_id = bridge.send('render', {'job': job_to_wire(job)})
                self.wait(bridge, 'lost', request_id)
                self.assertTrue(bridge.done.wait(10))
                record = QueueStore.inspect(workspace, request_id=request_id)
                self.assertEqual(record['status'], 'lost')
                self.assertTrue(record['event']['payload']['outputs_may_be_partial'])
                self.assertEqual(len(QueueStore.inspect(workspace)['rows']), 1)
            finally:
                bridge.stop()
                bridge.thread.join(10)

    def test_terminal_persistence_failure_during_close_is_explicit(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.json'
            source.write_text(json.dumps({'gate': str(root / 'gate')}))
            workspace, bridge = self.setup_bridge(root)
            job = JobRequest.create(source, root / 'out', InferenceConfig(output_format='exr'))
            request_id = bridge.send('render', {'job': job_to_wire(job)})
            self.wait(bridge, 'progress', request_id)
            observe = QueueStore.observe

            def full_on_cancel(store, event):
                if event['kind'] == 'cancelled':
                    raise sqlite3.OperationalError('injected terminal disk full')
                return observe(store, event)

            with patch('corridorkey.queue_store.QueueStore.observe', new=full_on_cancel):
                bridge.stop()
                bridge.thread.join(10)
            self.assertTrue(bridge.done.is_set())
            self.assertIn('injected terminal disk full', bridge.failure or '')
            snapshot = QueueStore.inspect(workspace)
            self.assertNotEqual(snapshot['rows'][0]['status'], 'cancelled')
            self.assertEqual(snapshot['rows'][0]['clean'], 0)

    def test_post_send_persistence_failure_never_fabricates_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.json'
            source.write_text(json.dumps({'gate': str(root / 'gate')}))
            phase = QueueStore.phase

            def fail_after_send(store, request_id, state):
                if state == 'sent':
                    raise OSError('injected post-send persistence failure')
                return phase(store, request_id, state)

            with patch('corridorkey.queue_store.QueueStore.phase', new=fail_after_send):
                workspace, bridge = self.setup_bridge(root)
                request_id = bridge.send('render', {'job': job_to_wire(JobRequest.create(source, root / 'out', InferenceConfig(output_format='exr')))})
                self.assertTrue(bridge.done.wait(10))
            self.assertIn('post-send', bridge.failure or '')
            events = [event for event in bridge.poll() if event.get('request_id') == request_id]
            self.assertFalse(any(event['kind'] == 'rejected' for event in events))
            record = QueueStore.inspect(workspace, request_id=request_id)
            self.assertIn(record['status'], {'cancelled', 'result', 'lost'})
            self.assertNotEqual(record['status'], 'rejected')
