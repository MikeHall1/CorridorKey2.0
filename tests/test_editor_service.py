import queue
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor_service import EditorService
from corridorkey.session import JobRequest


class EditorServiceTests(unittest.TestCase):
    def event(self, service, kind):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                event = service.events.get(timeout=.1)
            except queue.Empty:
                continue
            if event['kind'] == 'error':
                self.fail(event['error'])
            if event['kind'] == kind:
                return event
        self.fail('No editor event')

    def test_background_binding_edits_and_saved_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source'
            source.mkdir()
            cv2.imwrite(str(source / 'frame.png'), np.zeros((6, 8, 3), np.uint8))
            service = EditorService(JobRequest.create(source, root / 'out', InferenceConfig(output_format='exr')), root)
            try:
                ready = self.event(service, 'ready')
                self.assertEqual(ready['snapshot']['document']['source']['width'], 8)
                service.request('brush', mask=np.full((6, 8), 255, np.uint8), kind='remove', frame=0,
                                strength=.5, direction='both', span=[0, 1])
                self.assertEqual(self.event(service, 'result')['snapshot']['invalidated'], [(0, 1)])
                service.request('save')
                result = self.event(service, 'result')
                self.assertTrue(Path(result['snapshot']['saved_path']).is_file())
                service.request('view', frame=0)
                result = self.event(service, 'result')
                self.assertEqual(result['view'][0].pixels.shape, (6, 8, 3))
            finally:
                service.stop()
                service.thread.join(10)
            self.assertTrue(service.done.is_set())
