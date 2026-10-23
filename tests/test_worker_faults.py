import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest
from corridorkey.worker_client import WorkerClient
from corridorkey.worker_protocol import encode_frame, job_from_wire, job_to_wire


class WorkerFaultTests(unittest.TestCase):
    def test_inner_job_json_rejects_duplicates_and_missing_required_settings(self):
        wire = job_to_wire(JobRequest.create('/tmp/source', '/tmp/output'))
        settings = wire['config_json']
        assert isinstance(settings, str)
        for config in ('{}', '{"output_format":"video",' + settings[1:]):
            with self.subTest(config=config[:40]), self.assertRaises((ValueError, TypeError)):
                job_from_wire({**wire, 'config_json': config})

    def test_nonreading_peer_cannot_block_submitter_thread(self):
        frame = encode_frame({'schema_version': 1, 'generation': 'a' * 32, 'sequence': 0, 'kind': 'ready',
                              'request_id': None, 'payload': {'capacity': 4}})
        code = f'import os,time;os.write(1,{frame!r});time.sleep(60)'
        client = WorkerClient(command=[sys.executable, '-c', code], shutdown_timeout=0.1)
        self.addCleanup(client.close)
        finished = threading.Event()

        def submit():
            try:
                for _ in range(10000):
                    client.send('status', {})
            except (OSError, BufferError, RuntimeError):
                pass
            finally:
                finished.set()

        sender = threading.Thread(target=submit, daemon=True)
        sender.start()
        responsive = finished.wait(0.5)
        client._terminate_owned()
        sender.join(2)
        self.assertTrue(responsive, 'submitter blocked on an owned peer that stopped reading')

    def test_batch_observation_timeout_polls_same_live_child(self):
        import importlib.util
        if not all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'cv2')):
            self.skipTest('requires explicit CPU output fixture')
        from corridorkey.worker_client import run_worker_batch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.json'
            source.write_text('{}')
            manifest = root / 'batch.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
                {'input': str(source), 'output_dir': str(root / 'output'),
                 'settings': {'output_format': 'exr', 'outputs': ['alpha']}}]}))
            observed = []

            def factory(**kwargs):
                client = WorkerClient(command=[sys.executable, str(Path(__file__).with_name('worker_fixture.py'))], **kwargs)
                original = client.next_event

                def wait(timeout=10.0):
                    observed.append(client.process.pid)
                    if len(observed) == 1:
                        self.assertIsNone(client.process.poll())
                        raise TimeoutError('injected observation timeout')
                    return original(timeout)

                client.next_event = wait
                return client

            report = run_worker_batch(manifest, root / 'report.json', client_factory=factory)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(set(observed)), 1)
            self.assertEqual(len(report['jobs']), 1)


    def test_nonfinite_timeouts_reject_before_spawn_and_observation(self):
        from unittest.mock import patch

        for value in (float('nan'), float('inf'), -1, True):
            with patch('subprocess.Popen') as spawn, self.assertRaises(ValueError):
                WorkerClient(shutdown_timeout=value)
            spawn.assert_not_called()
        with WorkerClient() as client, self.assertRaises(ValueError):
            client.next_event(float('nan'))

    def test_unsupported_recovery_mode_rejects_before_missing_assets(self):
        from corridorkey.control import CancellationToken
        from corridorkey.worker_core import SharedEngine
        from corridorkey.worker_protocol import parse_request

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.write_text('fixture source')
            batch = root / 'batch.json'
            batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
                {'input': str(source), 'output_dir': str(root / 'output')}]}))
            request = parse_request({'schema_version': 1, 'request_id': 'r', 'action': 'recover', 'payload': {
                'batch': str(batch), 'workspace': str(root / 'cache'), 'max_cache_bytes': 0}})
            with self.assertRaisesRegex(ValueError, 'EXR'):
                SharedEngine().run(request, CancellationToken(), lambda _: None)
            self.assertFalse((root / 'cache').exists())

    @unittest.skipUnless(os.name == 'posix', 'requires actual POSIX owned process-group validation')
    def test_force_close_stops_descendant_that_ignores_sigterm(self):
        import importlib.util
        import signal
        if not all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'cv2')):
            self.skipTest('requires explicit CPU worker fixture')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker, heartbeat = root / 'pid', root / 'heartbeat'
            source = root / 'source.json'
            source.write_text(json.dumps({'descendant': str(marker), 'heartbeat': str(heartbeat)}))
            client = WorkerClient(command=[sys.executable, str(Path(__file__).with_name('worker_fixture.py'))], shutdown_timeout=0.1)
            self.addCleanup(client.close)
            job = JobRequest.create(source, root / 'output', InferenceConfig(output_format='exr', outputs={'alpha'}))
            request_id = client.submit(job)
            while True:
                event = client.next_event(10)
                if (event['kind'] == 'progress' and event['request_id'] == request_id and
                        event['payload']['stage'] == 'fixture_descendant_ready'):
                    break
            pid = int(marker.read_text())
            try:
                self.assertFalse(client.close())
                self.assertIsNotNone(client.process.poll())
                value = heartbeat.read_text()
                time.sleep(0.1)
                self.assertEqual(heartbeat.read_text(), value, 'owned descendant still executing after force close')
            finally:
                try:
                    if os.getpgid(pid) == client.process.pid:
                        os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
