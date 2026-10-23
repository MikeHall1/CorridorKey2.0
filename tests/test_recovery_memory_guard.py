"""The sampled stop policy also covers actual nonrender recovery workflows."""
import importlib.util
import itertools
import json
import unittest
from unittest.mock import patch

HAS_STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'cv2', 'torch'))


@unittest.skipUnless(HAS_STACK, 'requires optional CPU export adapters')
class RecoveryMemoryGuardTests(unittest.TestCase):
    def setUp(self):
        from corridorkey.memory_guard import ProcessTreeRssGuard
        from tests.test_recovery_runner import RecoveryRunnerTests

        self.guard_type = ProcessTreeRssGuard
        self.fixture = RecoveryRunnerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.guards = []
        self.phase = None
        self.refuse_restore = False

    def guard_factory(self, limit, **kwargs):
        owner = self

        class Observer:
            def sample(self, check=None):
                if check is not None:
                    check()
                value = 20 if owner.refuse_restore and owner.phase == 'recovery_restore' else 1
                return {'status': 'available', 'current_rss_bytes': value,
                        'provider': 'injected deterministic observation; no native RSS claim'}

        guard = self.guard_type(limit, observer=Observer(), clock=itertools.count().__next__, **kwargs)
        self.guards.append(guard)
        return guard

    def jobs(self, limit):
        from corridorkey.session import JobRequest

        result = []
        for job in self.fixture.jobs:
            config, writer = job.configurations()
            config.process_tree_rss_stop_bytes = limit
            result.append(JobRequest.create(job.input_path, job.output_dir, config, writer))
        return result

    def run_jobs(self, *, resume=False, limit=10):
        from corridorkey import memory_guard
        from corridorkey.recovery import run_recoverable

        def progress(event):
            self.phase = event.stage

        f = self.fixture
        with patch.object(memory_guard, 'ProcessTreeRssGuard', side_effect=self.guard_factory):
            return run_recoverable(self.jobs(limit), workspace=f.workspace, max_cache_bytes=100000,
                                   assets=f.assets, environment=f.environment,
                                   implementation_root=f.root / 'implementation', resume=resume,
                                   session_factory=f.factory, progress=progress)

    def test_render_restore_and_verified_existing_share_each_job_observation(self):
        report = self.run_jobs()
        self.assertEqual([job['mode'] for job in report['jobs']], ['render', 'verified_cache_restore'])
        self.assertEqual(len(self.guards), 2)
        self.assertEqual(self.fixture.loads, 1)
        self.assertTrue(all(item['policy']['samples'] > 0 for item in report['resource_observations']))
        self.assertEqual(report['jobs'][0]['origin_record']['diagnostics']['resource_policy']['limit_bytes'], 10)
        resumed = self.run_jobs(resume=True)
        self.assertEqual([job['mode'] for job in resumed['jobs']], ['verified_existing', 'verified_existing'])
        self.assertEqual(len(self.guards), 4)
        self.assertEqual(self.fixture.loads, 1)

    def test_zero_threshold_refuses_before_dependency_hashing_or_model_load(self):
        from corridorkey.memory_guard import ResourceLimitError

        with (patch('corridorkey.recovery.job_identity', side_effect=AssertionError('hashing must not start')),
              self.assertRaises(ResourceLimitError)):
            self.run_jobs(limit=0)
        self.assertEqual(self.fixture.loads, 0)
        self.assertFalse(self.fixture.renders)
        records = list((self.fixture.workspace / 'reports').glob('run-*.json'))
        self.assertEqual(len(records), 1)
        payload = json.loads(records[0].read_text())
        # Recovery reports are strict sealed envelopes.
        report = payload.get('payload', payload)
        self.assertEqual(report['status'], 'failed')
        policy = report['resource_observations'][0]['policy']
        self.assertEqual(policy['failure']['type'], 'ResourceLimitError')

    def test_restore_stop_preserves_sealed_cache_and_completed_original_export(self):
        from pathlib import Path

        from corridorkey.memory_guard import ResourceLimitError

        self.refuse_restore = True
        with self.assertRaises(ResourceLimitError):
            self.run_jobs()
        self.assertEqual(self.fixture.loads, 1)
        self.assertEqual(len(self.fixture.renders), 1)
        export = self.fixture.root / 'first/synthetic_alpha_exr/alpha_000000.exr'
        original = export.read_bytes()
        self.refuse_restore = False
        resumed = self.run_jobs(resume=True)
        self.assertEqual([job['mode'] for job in resumed['jobs']], ['verified_existing', 'verified_cache_restore'])
        self.assertEqual(export.read_bytes(), original)
        self.assertEqual(self.fixture.loads, 1)
        self.assertTrue(Path(resumed['jobs'][1]['outputs']['alpha']).is_dir())
