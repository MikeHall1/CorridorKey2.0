"""Declared sequence primaries across configuration, transport, replay and quality binding."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from corridorkey.batch import load_jobs
from corridorkey.config import InferenceConfig
from corridorkey.recovery_identity import job_identity
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_from_wire, job_to_wire

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


class ColourProfileWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / '0000.exr').write_bytes(b'identity fixture only; not decoded')

    def config(self, profile):
        return InferenceConfig(frame_dir_linear=True, source_colour_profile=profile,
                               output_format='exr', outputs={'alpha', 'fg'}, auto_vram=False)

    def test_legacy_default_and_explicit_profile_validation(self):
        self.assertEqual(InferenceConfig().source_colour_profile, 'legacy')
        for value in ('unknown', None, [], True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                InferenceConfig(**json.loads(json.dumps({'source_colour_profile': value}))).validate()
        for profile in ('linear_bt709_d65', 'linear_bt2020_d65'):
            self.config(profile).validate()
            with self.assertRaisesRegex(ValueError, 'scene-linear'):
                InferenceConfig(source_colour_profile=profile).validate()
            cfg = self.config(profile)
            cfg.source_witness = True
            with self.assertRaisesRegex(ValueError, 'source_witness'):
                cfg.validate()

    def test_video_rejected_before_processing_after_wire_roundtrip(self):
        video = self.root / 'video.mov'
        video.write_bytes(b'not decoded')
        job = JobRequest.create(video, self.root / 'output', self.config('linear_bt2020_d65'))
        cfg, _ = job.configurations()
        with self.assertRaisesRegex(ValueError, 'video is unsupported'):
            job.validate_source(cfg)
        with self.assertRaisesRegex(ValueError, 'video is unsupported'):
            restored = job_from_wire(job_to_wire(job))
            restored.validate_source(restored.configurations()[0])
        self.assertFalse((self.root / 'output').exists())

    def test_batch_and_wire_preserve_profile_exactly(self):
        batch = self.root / 'batch.json'
        batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [{
            'input': str(self.source), 'output_dir': str(self.root / 'output'),
            'settings': {'frame_dir_linear': True, 'source_colour_profile': 'linear_bt2020_d65',
                         'output_format': 'exr', 'auto_vram': False, 'temp_dir': str(self.root / 'temp')}}]}))
        jobs, _ = load_jobs(batch)
        restored = job_from_wire(job_to_wire(jobs[0]))
        self.assertEqual(restored, jobs[0])
        self.assertEqual(restored.configurations()[0].source_colour_profile, 'linear_bt2020_d65')

    def test_profile_change_invalidates_whole_job_and_child_cache_parent(self):
        code = self.root / 'implementation/corridorkey'
        code.mkdir(parents=True)
        (code / 'runtime.py').write_text('# synthetic fixture implementation')
        assets = {}
        for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir'):
            path = self.root / name
            path.write_bytes(name.encode())
            assets[name] = path
        environment = {'backend': 'cpu-fixture', 'precision': 'float32',
                       'dependencies': {'fixture': '1'}, 'runtime': {'fixture': True}}
        keys = []
        for profile in ('legacy', 'linear_bt709_d65', 'linear_bt2020_d65'):
            job = JobRequest.create(self.source, self.root / 'output', self.config(profile))
            identity = job_identity(job, assets=assets, environment=environment,
                                    implementation_root=self.root / 'implementation')
            keys.append(identity.key)
            self.assertEqual(identity.to_dict()['settings']['source_colour_profile'], profile)
        self.assertEqual(len(set(keys)), 3)

    def test_quality_binding_accepts_only_exact_declared_working_contract(self):
        from corridorkey.benchmark_quality import _working_primaries_match
        from corridorkey.colour_contract import describe_source_colour
        legacy = {'rgb_primaries': 'assumed sRGB/BT.709; no automatic colour management'}
        self.assertTrue(_working_primaries_match(legacy))
        for profile in ('linear_bt709_d65', 'linear_bt2020_d65'):
            contract = {'rgb_primaries': 'BT.709/D65 working output',
                        'source_colour_contract': describe_source_colour(profile, scene_linear=True)}
            self.assertTrue(_working_primaries_match(contract))
            changed = copy.deepcopy(contract)
            changed['source_colour_contract']['working_primaries'] = 'bt2020_d65'
            self.assertFalse(_working_primaries_match(changed))
            changed = copy.deepcopy(contract)
            changed['source_colour_contract']['matrix'][0][0] += .01
            self.assertFalse(_working_primaries_match(changed))
        self.assertFalse(_working_primaries_match({'rgb_primaries': 'BT.2020'}))

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2')),
                         'Requires explicit CPU runtime fixture')
    def test_actual_worker_exports_declared_profile_and_known_working_contract(self):
        import cv2
        import numpy as np

        from corridorkey.benchmark_quality import _working_primaries_match
        from corridorkey.colour_contract import describe_source_colour
        from corridorkey.worker_client import WorkerClient
        path = self.source / '0000.exr'
        self.assertTrue(cv2.imwrite(str(path), np.full((64, 64, 3), .25, np.float32)))
        original = path.read_bytes()
        cfg = self.config('linear_bt2020_d65')
        cfg.hann_chunk, cfg.hann_stride, cfg.shape_quantum = 8, 4, 64
        cfg.temp_dir = self.root / 'scratch'
        job = JobRequest.create(self.source, self.root / 'export', cfg)
        fixture = Path(__file__).with_name('output_telemetry_worker_fixture.py')
        client = WorkerClient(command=[sys.executable, str(fixture), '--fixture-root', str(self.root / 'fixture-assets')])
        try:
            request = client.submit(job)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                event = client.next_event(timeout=1)
                if event['request_id'] == request and event['kind'] in ('result', 'failed', 'cancelled', 'lost', 'rejected'):
                    self.assertEqual(event['kind'], 'result', event)
                    diagnostics = event['payload']['record']['diagnostics']
                    expected = describe_source_colour('linear_bt2020_d65', scene_linear=True)
                    self.assertEqual(diagnostics['source']['source_colour_contract'], expected)
                    self.assertEqual(diagnostics['execution']['source_colour_contract'], expected)
                    self.assertTrue(_working_primaries_match(diagnostics['output_contracts']['fg']))
                    self.assertEqual(diagnostics['frames_submitted'], 1)
                    break
            else:
                self.fail('No terminal event from the owned CPU fixture worker')
        finally:
            client.close()
            self.assertIsNotNone(client.process.poll())
            for thread in (client.reader, client.writer, client.log_reader):
                self.assertFalse(thread.is_alive())
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
