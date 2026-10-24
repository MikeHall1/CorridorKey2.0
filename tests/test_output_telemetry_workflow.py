"""Actual owned CPU worker/export telemetry; no pretrained or GPU timing claims."""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'cv2', 'safetensors'))
FIXTURE = Path(__file__).with_name('output_telemetry_worker_fixture.py')


def hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(folder).glob('*.exr')}


@unittest.skipUnless(STACK, 'Requires explicit CPU native export fixture; not GPU inference')
class OutputTelemetryWorkflowTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(9):
            self.assertTrue(cv2.imwrite(str(self.source / f'{frame:04}.png'), np.full((64, 64, 3), 80 + frame * 4, np.uint8)))
        self.originals = {path.name: path.read_bytes() for path in self.source.iterdir()}
        from corridorkey.worker_client import WorkerClient
        self.client = WorkerClient(command=[sys.executable, str(FIXTURE)])
        self.addCleanup(self.close_client)

    def close_client(self):
        self.client.close()
        self.assertIsNotNone(self.client.process.poll())
        for thread in (self.client.reader, self.client.writer, self.client.log_reader):
            self.assertFalse(thread.is_alive(), 'Owned worker transport thread survived close')

    def job(self, name, *, profile, quick=False):
        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest
        config = InferenceConfig(output_format='exr', outputs={'alpha', 'fg'}, exr_dtype='float',
            auto_vram=False, hann_chunk=8, hann_stride=4, shape_quantum=64, stage2_batch=2,
            start_frame=2, num_frames=5, temp_dir=self.root / 'temp', despill_strength=0,
            quick_key=quick, quick_interval=4, quick_context_radius=1, quick_fallback='neural')
        return JobRequest.create(self.source, self.root / name, config, profile_stages=profile)

    def terminal(self, request_id):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            event = self.client.next_event(timeout=5)
            if event['request_id'] == request_id and event['kind'] in ('result', 'failed', 'cancelled', 'lost', 'rejected'):
                return event
        self.fail('No terminal result from the same owned worker')

    def assert_completed(self, record):
        self.assertEqual(record['status'], 'complete')
        diagnostics = record['diagnostics']
        telemetry = diagnostics['output_telemetry']
        self.assertEqual(telemetry['status'], 'complete')
        self.assertEqual(telemetry['frame_index_space'], 'output-relative')
        self.assertEqual(telemetry['expected_frames'], 5)
        for key in ('registered_frames', 'all_queues_accepted_frames', 'all_native_completed_frames'):
            self.assertEqual(telemetry[key], 5, key)
        for key in ('missing_registered_frames', 'missing_queue_frames', 'missing_native_frames'):
            self.assertEqual(telemetry[key], 0, key)
        self.assertEqual(diagnostics['source']['start_frame'], 2)
        self.assertEqual(diagnostics['frames_submitted'], 5)
        for output in ('alpha', 'fg'):
            self.assertEqual(telemetry['per_output'][output]['submitted_frames'], 5)
            self.assertEqual(telemetry['per_output'][output]['native_completed_frames'], 5)
            self.assertEqual(len(hashes(diagnostics['outputs'][output])), 5)
        for key in ('render_ready_offsets', 'queue_accept_latency', 'native_write_latency', 'native_completion_offsets'):
            distribution = telemetry[key]
            self.assertEqual(distribution['count'], 5, key)
            self.assertEqual(distribution['expected_count'], 5, key)
            values = [distribution[name] for name in ('min_seconds', 'p50_seconds', 'p95_seconds', 'p99_seconds', 'max_seconds')]
            self.assertEqual(values, sorted(values), key)
            self.assertTrue(all(value >= 0 for value in values), key)
        self.assertLessEqual(telemetry['completion_spacing']['count'], 4)
        self.assertIsNone(diagnostics['resources']['peak_allocated_vram_bytes'])
        self.assertIn('CPU fixture', diagnostics['resources']['scope'])

    def test_profile_regular_and_quick_final_outputs_match_unprofiled_and_preserve_sources(self):
        import cv2
        import numpy as np
        for quick in (False, True):
            with self.subTest(quick=quick):
                unprofiled = self.terminal(self.client.submit(self.job(f'oracle{quick}', profile=False, quick=quick)))
                self.assertEqual(unprofiled['kind'], 'result', unprofiled)
                before = unprofiled['payload']['record']
                self.assertIsNone(before['diagnostics']['output_telemetry'])
                measured = self.terminal(self.client.submit(self.job(f'profile{quick}', profile=True, quick=quick)))
                self.assertEqual(measured['kind'], 'result', measured)
                after = measured['payload']['record']
                self.assertTrue(after['diagnostics']['profiling_enabled'])
                self.assertTrue(after['diagnostics']['stages'])
                self.assertTrue(all(stage['method'] == 'host_wall_explicit_cpu_fixture' for stage in after['diagnostics']['stages']))
                self.assert_completed(after)
                for output in ('alpha', 'fg'):
                    self.assertEqual(hashes(after['diagnostics']['outputs'][output]), hashes(before['diagnostics']['outputs'][output]))
                for path in Path(after['diagnostics']['outputs']['alpha']).glob('*.exr'):
                    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    self.assertEqual(image.dtype, np.float32)
                    np.testing.assert_array_equal(image, np.full((64, 64), np.float32(.4)))
        self.assertEqual({path.name: path.read_bytes() for path in self.source.iterdir()}, self.originals)

    def test_cancelled_profile_keeps_exact_partial_writer_counts_and_reaps_owner(self):
        ready, release = self.root / 'ready', self.root / 'release'
        (self.root / '.stage2-worker.json').write_text(json.dumps({'ready': str(ready), 'gate': str(release)}))
        request_id = self.client.submit(self.job('cancelled', profile=True))
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(ready.exists())
        self.client.cancel(request_id)
        event = self.terminal(request_id)
        self.assertEqual(event['kind'], 'cancelled', event)
        record = event['payload']['record']
        self.assertEqual(record['status'], 'cancelled')
        telemetry = record['diagnostics']['output_telemetry']
        self.assertEqual(telemetry['status'], 'partial')
        self.assertEqual(telemetry['expected_frames'], 5)
        self.assertGreater(telemetry['registered_frames'], 0)
        self.assertLess(telemetry['registered_frames'], 5)
        self.assertIsNone(record['end_to_end_fps'])
        for output in ('alpha', 'fg'):
            paths = list(Path(record['diagnostics']['outputs'][output]).glob('*.exr'))
            self.assertEqual(telemetry['per_output'][output]['native_completed_frames'], len(paths))
        self.close_client()
        self.assertEqual({path.name: path.read_bytes() for path in self.source.iterdir()}, self.originals)
