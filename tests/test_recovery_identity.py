import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from corridorkey.config import InferenceConfig
from corridorkey.recovery_identity import ReplayIdentity, canonical, job_identity
from corridorkey.session import JobRequest


class ReplayIdentityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'source fixture')
        code = self.root / 'implementation/corridorkey'
        code.mkdir(parents=True)
        (code / 'runtime.py').write_text('# synthetic implementation identity')
        self.assets = {}
        for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint'):
            path = self.root / name
            path.write_bytes(name.encode())
            self.assets[name] = path
        directory = self.root / 'cradio'
        directory.mkdir()
        (directory / 'config.json').write_text('{}')
        self.assets['cradio_dir'] = directory
        self.environment = {'backend': 'cpu-test-fixture', 'precision': 'fixture-float32',
                            'dependencies': {'fixture': '1'}, 'runtime': {'device': 'test only'}}
        self.job = JobRequest.create(self.source, self.root / 'output', InferenceConfig(output_format='exr'))

    def identity(self, job=None):
        return job_identity(job or self.job, assets=self.assets, environment=self.environment,
                            implementation_root=self.root / 'implementation')

    def test_destination_independent_canonical_and_detached(self):
        first = self.identity()
        second = self.identity(replace(self.job, output_dir=str(self.root / 'other')))
        self.assertEqual(first.key, second.key)
        data = first.to_dict()
        data['environment']['precision'] = 'changed'
        self.assertNotEqual(data, first.to_dict())
        self.assertEqual(first, ReplayIdentity(first.payload))

    def test_settings_hints_assets_environment_and_code_invalidate(self):
        first = self.identity().key
        config, writer = self.job.configurations()
        altered = JobRequest.create(self.source, self.root / 'other', replace(config, despill_strength=0.2), writer)
        self.assertNotEqual(first, self.identity(altered).key)
        hint = self.root / 'hint.png'
        hint.write_bytes(b'identity-only hint fixture')
        self.assertNotEqual(first, self.identity(replace(self.job, initial_hint_path=str(hint))).key)
        self.assets['weights'].write_bytes(b'changed weights')
        self.assertNotEqual(first, self.identity().key)
        before = self.identity().key
        self.environment['runtime']['device'] = 'other'
        self.assertNotEqual(before, self.identity().key)
        before = self.identity().key
        (self.root / 'implementation/corridorkey/runtime.py').write_text('# changed implementation')
        self.assertNotEqual(before, self.identity().key)

    def test_source_and_source_sequence_order_invalidate(self):
        first = self.identity().key
        self.source.write_bytes(b'changed source')
        self.assertNotEqual(first, self.identity().key)
        source = self.root / 'frames'
        source.mkdir()
        (source / 'frame1.png').write_bytes(b'first')
        (source / 'frame2.png').write_bytes(b'second')
        job = replace(self.job, input_path=str(source))
        before = self.identity(job).key
        (source / 'frame1.png').rename(source / 'frame3.png')
        self.assertNotEqual(before, self.identity(job).key)

    def test_unsupported_scheduling_and_missing_context_reject(self):
        config, writer = self.job.configurations()
        for changed in (replace(config, output_format='video'), replace(config, auto_vram=True)):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'EXR output'):
                self.identity(JobRequest.create(self.source, self.root / 'output', changed, writer))
        with self.assertRaises(ValueError):
            job_identity(self.job, assets={}, environment=self.environment, implementation_root=self.root / 'implementation')
        with self.assertRaises(ValueError):
            ReplayIdentity(json.dumps({'schema_version': 1, 'kind': 'wrong'}))

    def test_hashing_cancellation_propagates(self):
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            job_identity(self.job, assets=self.assets, environment=self.environment,
                         implementation_root=self.root / 'implementation',
                         check=lambda: (_ for _ in ()).throw(RuntimeError('cancel')))

    def test_version_types_are_strict(self):
        for version in (True, 1.0, '1'):
            with self.subTest(version=version), self.assertRaises(ValueError):
                ReplayIdentity(canonical({'schema_version': version, 'kind': 'whole_job_exr_replay'}))
