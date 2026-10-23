import sys
import tempfile
import time
import unittest
from pathlib import Path

from corridorkey.config import InferenceConfig
from corridorkey.desktop_bridge import DesktopBridge
from corridorkey.session import JobRequest
from corridorkey.worker_client import WorkerClient
from corridorkey.worker_protocol import job_to_wire

FIXTURE = Path(__file__).with_name('worker_fixture.py')


class DesktopBridgeTests(unittest.TestCase):
    def wait(self, bridge, kind, request_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for event in bridge.poll():
                if event['kind'] == kind and event['request_id'] == request_id:
                    return event
            if bridge.done.is_set():
                self.fail(bridge.failure)
            time.sleep(0.01)
        self.fail(f'No {kind} event')

    def test_actual_child_render_then_cancel_without_blocking_frontend(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source.json'
            source.write_text('{}')
            bridge = DesktopBridge(lambda: WorkerClient(command=[sys.executable, str(FIXTURE)]))
            try:
                job = JobRequest.create(source, Path(root) / 'output', InferenceConfig(output_format='exr'))
                request_id = bridge.send('render', {'job': job_to_wire(job)})
                result = self.wait(bridge, 'result', request_id)
                self.assertEqual(result['payload']['record']['status'], 'complete')
                self.assertEqual(len(list((Path(job.output_dir) / 'alpha').glob('*.exr'))), 2)
                source.write_text('{"gate":"' + str(Path(root) / 'gate') + '"}')
                request_id = bridge.send('render', {'job': job_to_wire(job)})
                self.wait(bridge, 'progress', request_id)
                bridge.send('cancel', {'target': request_id})
                self.wait(bridge, 'cancelled', request_id)
            finally:
                bridge.stop()
                bridge.thread.join(10)
            self.assertTrue(bridge.done.is_set())
            self.assertIsNone(bridge.failure)
