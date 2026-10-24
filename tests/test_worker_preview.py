"""Shared preview dispatch/ownership; injected CPU runners are not neural evidence."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from corridorkey.config import InferenceConfig
from corridorkey.control import CancellationToken
from corridorkey.desktop_bridge import DesktopBridge
from corridorkey.queue_store import QueueStore
from corridorkey.session import InferenceSession, JobRequest
from corridorkey.worker_client import WorkerClient
from corridorkey.worker_core import SharedEngine, WorkerCore
from corridorkey.worker_protocol import (
    Request,
    decode_frame,
    job_from_wire,
    job_to_wire,
    parse_request,
    request_frame,
)

STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'cv2'))
FIXTURE = Path(__file__).with_name('worker_fixture.py')


class WorkerPreviewTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.source = self.root / 'source.json'
        self.source.write_text('{}')
        self.job = JobRequest.create(self.source, self.root / 'output',
            InferenceConfig(output_format='exr', exr_dtype='float', outputs={'alpha'},
                            start_frame=3, num_frames=2, stage2_batch=2, foundation_cache_mb=13))

    def request(self, identifier, action='preview', payload=None):
        return parse_request(decode_frame(request_frame(identifier, action,
            {'job': job_to_wire(self.job)} if payload is None else payload)))

    def test_preview_wire_uses_exact_render_job_validation_and_detached_snapshot(self):
        wire = job_to_wire(self.job)
        request = self.request('preview')
        self.assertEqual(request.payload(), self.request('render', 'render').payload())
        wire['output_dir'] = str(self.root / 'mutated')
        exported = request.payload()
        exported['job']['config_json'] = '{}'
        self.assertEqual(job_to_wire(job_from_wire(request.payload()['job'])), job_to_wire(self.job))
        invalid = [{}, {'batch': 'not recovery'}, {'job': job_to_wire(self.job), 'extra': True},
                   {'job': JobRequest}, {'job': {'input_path': 'relative'}}]
        duplicate = copy.deepcopy(job_to_wire(self.job))
        duplicate['config_json'] = '{"hann_chunk":80,"hann_chunk":40}'
        invalid.append({'job': duplicate})
        nonfinite = copy.deepcopy(job_to_wire(self.job))
        nonfinite['writer_json'] = '{"timeout_seconds":NaN}'
        invalid.append({'job': nonfinite})
        for payload in invalid:
            with self.subTest(payload=str(payload)[:80]), self.assertRaises((TypeError, ValueError)):
                self.request('invalid', payload=payload)

    def test_shared_engine_reuses_actual_session_and_model_for_render_and_preview(self):
        loads, runs, disposed = [], [], []
        model = object()

        def load():
            loads.append(model)
            return model

        def run(loaded, job, config, writer, diagnostics):
            runs.append((loaded, job, config.stage2_batch))
            diagnostics.execution = {'neural_inference': False, 'evidence': 'injected CPU ownership runner'}

        session = InferenceSession(loader=load, runner=run, disposer=disposed.append)
        engine = SharedEngine()
        with patch('corridorkey.worker_core.InferenceSession', return_value=session) as constructor:
            for identifier, action in (('a', 'preview'), ('b', 'render'), ('c', 'preview')):
                result = engine.run(self.request(identifier, action), CancellationToken(), lambda _: None)
                self.assertEqual(result['status'], 'complete')
                self.assertFalse(result['diagnostics']['execution']['neural_inference'])
            constructor.assert_called_once()
        self.assertEqual(loads, [model])
        self.assertEqual([item[0] for item in runs], [model] * 3)
        self.assertEqual([job_to_wire(item[1]) for item in runs], [job_to_wire(self.job)] * 3)
        self.assertEqual([item[2] for item in runs], [2] * 3)
        with self.assertRaisesRegex(ValueError, 'supports render, preview'):
            engine.run(Request('control', 'status', '{}'), CancellationToken(), lambda _: None)
        self.assertIs(engine.session, session)  # No accidental recovery close/fallback.
        engine.close()
        self.assertEqual(disposed, [model])

    def test_core_preview_active_pending_cancel_seen_capacity_and_shutdown_terminals(self):
        entered, release = threading.Event(), threading.Event()
        runs, events = [], []
        condition = threading.Condition()

        def runner(_model, job, config, writer, diagnostics):
            runs.append(job)
            entered.set()
            while not release.wait(.01):
                diagnostics.control.checkpoint('fixture_wait', 0, 1)
            diagnostics.control.checkpoint('fixture_done', 1, 1)

        def emit(kind, identifier, payload):
            with condition:
                events.append((kind, identifier, payload))
                condition.notify_all()

        session = InferenceSession(loader=object, runner=runner, disposer=lambda _: None)
        engine = SharedEngine()
        with patch('corridorkey.worker_core.InferenceSession', return_value=session):
            core = WorkerCore(engine, emit, capacity=1)
            try:
                core.dispatch(self.request('active'))
                self.assertTrue(entered.wait(2))
                core.dispatch(self.request('waiting'))
                core.dispatch(self.request('full'))
                core.dispatch(self.request('cancel-waiting', 'cancel', {'target': 'waiting'}))
                core.dispatch(self.request('waiting'))
                core.dispatch(self.request('shutdown-pending'))
                core.stop()
                self.assertTrue(core.finish())
            finally:
                release.set()
                core.stop()
                self.assertTrue(core.finish())
        self.assertEqual(len(runs), 1)
        terminals = {(kind, identifier) for kind, identifier, _ in events if kind in ('cancelled', 'result', 'failed')}
        self.assertEqual(terminals, {('cancelled', 'active'), ('cancelled', 'waiting'), ('cancelled', 'shutdown-pending')})
        self.assertIn(('rejected', 'full', {'code': 'queue_full'}), events)
        self.assertIn(('rejected', 'waiting', {'code': 'duplicate_id'}), events)
        self.assertFalse(any(kind == 'started' and identifier != 'active' for kind, identifier, _ in events))
        self.assertIsNone(core.fatal)
        self.assertIsNone(engine.session)

    def test_pending_desktop_preview_cancels_without_durable_queue_prepare(self):
        entered, release = threading.Event(), threading.Event()

        class Client:
            process = SimpleNamespace(pid=123)
            generation = 'fixture-generation'
            cleanup_error = failure = None

            def send(self, action, payload, *, request_id):
                return request_id

            def next_event(self, timeout=0):
                raise TimeoutError('no fixture event')

            def close(self):
                return True

        def factory():
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture worker constructor gate')
            return cast(WorkerClient, Client())

        journal = QueueStore.create(self.root)
        bridge = DesktopBridge(factory, journal_workspace=journal)
        try:
            bridge.send('status', {})
            self.assertTrue(entered.wait(2))
            identifier = bridge.send('preview', {'job': job_to_wire(self.job)})
            bridge.stop()
            release.set()
            self.assertTrue(bridge.done.wait(5))
            events = bridge.poll()
            cancelled = [event for event in events if event['request_id'] == identifier]
            self.assertEqual([event['kind'] for event in cancelled], ['cancelled'])
            self.assertIs(cancelled[0]['payload']['executed'], False)
            self.assertEqual(QueueStore.inspect(journal)['rows'], [])
            self.assertIsNone(bridge.failure)
        finally:
            release.set()
            bridge.stop()
            bridge.thread.join(5)

    def terminal(self, client, identifier):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            event = client.next_event(max(0, deadline - time.monotonic()))
            if event['request_id'] == identifier and event['kind'] in ('result', 'failed', 'cancelled', 'lost', 'rejected'):
                return event
        self.fail('Missing preview terminal')

    @unittest.skipUnless(STACK, 'requires optional CPU image adapters')
    def test_actual_child_preview_render_share_model_and_track_seen_terminal(self):
        with WorkerClient(command=[sys.executable, str(FIXTURE)]) as client:
            for identifier, action in (('preview-a', 'preview'), ('render', 'render'), ('preview-b', 'preview')):
                destination = self.root / identifier
                job = replace(self.job, output_dir=str(destination))
                if action == 'preview':
                    client.submit_preview(job, request_id=identifier)
                else:
                    client.submit(job, request_id=identifier)
                event = self.terminal(client, identifier)
                self.assertEqual(event['kind'], 'result')
                record = event['payload']['record']
                self.assertEqual(record['diagnostics']['execution']['fixture_loads'], 1)
                self.assertFalse(record['diagnostics']['execution']['neural_inference'])
                self.assertEqual(len(list((destination / 'alpha').glob('*.exr'))), 2)
                self.assertNotIn(identifier, client.outstanding)
            with self.assertRaisesRegex(ValueError, 'already used'):
                client.submit_preview(self.job, request_id='preview-a')
        self.assertTrue(client.clean_stop)
        self.assertIsNone(client.cleanup_error)

    @unittest.skipUnless(STACK, 'requires optional CPU image adapters')
    def test_actual_child_preview_crash_becomes_lost_not_ignored(self):
        self.source.write_text(json.dumps({'exit': True}))
        client = WorkerClient(command=[sys.executable, str(FIXTURE)])
        try:
            if os.name != 'nt':
                # Prove discovery while the idle leader is alive, before requesting a hard exit.
                with client.kill_lock:
                    client._capture_descendants()
                self.assertIsNotNone(client.containment)
                assert client.containment is not None
                self.assertGreater(client.containment.complete_samples, 0)
                self.assertIsNone(client.containment.failure)
            identifier = client.submit_preview(self.job)
            event = self.terminal(client, identifier)
            self.assertEqual(event['kind'], 'lost')
            self.assertTrue(event['payload']['outputs_may_be_partial'])
            self.assertFalse(client.outstanding)
        finally:
            client.close()
        self.assertIsNotNone(client.failure)
        self.assertIsNone(client.cleanup_error)

    @unittest.skipUnless(STACK, 'requires optional CPU image adapters')
    def test_actual_child_preview_progress_capacity_and_cancellation_retire_model(self):
        self.source.write_text(json.dumps({'gate': str(self.root / 'closed-gate')}))
        with WorkerClient(command=[sys.executable, str(FIXTURE), '--capacity', '1']) as client:
            active = client.submit_preview(self.job, request_id='active-preview')
            deadline = time.monotonic() + 5
            while True:
                event = client.next_event(max(0, deadline - time.monotonic()))
                if event['kind'] == 'progress' and event['request_id'] == active:
                    break
                if time.monotonic() >= deadline:
                    self.fail('Preview progress was not delivered')
            waiting = client.submit_preview(self.job, request_id='waiting-preview')
            with self.assertRaises(BufferError):
                client.submit_preview(self.job, request_id='full-preview')
            self.assertNotIn('full-preview', client.seen)
            client.cancel(waiting)
            self.assertEqual(self.terminal(client, waiting)['kind'], 'cancelled')
            client.cancel(active)
            self.assertEqual(self.terminal(client, active)['kind'], 'cancelled')
            self.assertFalse(client.outstanding)
            self.source.write_text('{}')
            identifier = client.submit(self.job, request_id='after-cancel')
            result = self.terminal(client, identifier)
            self.assertEqual(result['kind'], 'result')
            self.assertEqual(result['payload']['record']['diagnostics']['execution']['fixture_loads'], 2)
        self.assertTrue(client.clean_stop)


if __name__ == '__main__':
    unittest.main()
