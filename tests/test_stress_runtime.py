"""CPU-owned worker evidence, never production model or fault-volume acceptance."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.config import InferenceConfig
from corridorkey.control import CancelledError
from corridorkey.session import JobRequest
from corridorkey.stress_runtime import execute_scenario
from corridorkey.worker_client import WorkerCleanupError, WorkerClient

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2'))


class StressRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.source = self.parent / 'source.json'
        self.source.write_text('{}')
        self.original = self.parent / 'original-output'
        self.original.mkdir()
        (self.original / 'sentinel').write_bytes(b'untouched')
        self.job = JobRequest.create(self.source, self.original,
                                     InferenceConfig(output_format='exr', outputs={'alpha'}, temp_dir=self.parent / 'original-temp'))
        self.clients = []

    def factory(self, **kwargs):
        client = WorkerClient(capacity=kwargs['capacity'],
                              command=[sys.executable, str(Path(__file__).with_name('worker_fixture.py')), '--capacity', '1'])
        self.clients.append(client)
        return client

    def run_case(self, scenario, **kwargs):
        case = {'scenario': scenario, 'repetitions': 2, 'timeout_seconds': 30, **kwargs}
        return execute_scenario(case, [self.job], None, self.parent / 'fresh', worker_factory=self.factory)

    def test_uncertain_worker_constructor_never_reports_cleanup_verified(self):
        def factory(**kwargs):
            raise WorkerCleanupError('Owned startup reader is still live')

        result = execute_scenario({'scenario': 'repeated', 'repetitions': 2},
                                 [self.job], None, self.parent / 'fresh', worker_factory=factory)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['cleanup']['verified'])
        self.assertIn('Owned startup reader', result['cleanup']['startup_error'])
        self.assertEqual(result['cleanup']['owners'], [])

    @unittest.skipUnless(STACK, 'requires explicit CPU export fixture')
    def test_repeated_real_owned_worker_preserves_templates_and_reaps(self):
        result = self.run_case('repeated')
        self.assertEqual(result['status'], 'measured', result)
        self.assertTrue(result['cleanup']['verified'])
        self.assertEqual(len(result['raw_records']), 2)
        records = [r['payload']['record'] for r in result['raw_records']]
        self.assertFalse(records[0]['model_reused'])
        self.assertTrue(records[1]['model_reused'])
        self.assertEqual((self.original / 'sentinel').read_bytes(), b'untouched')
        self.assertFalse((self.parent / 'original-temp').exists())
        self.assertEqual(self.source.read_text(), '{}')
        self.assertTrue(all(c.closed and c.process.poll() is not None for c in self.clients))
        self.assertIsNone(result['acceptance_verdict'])

    @unittest.skipUnless(STACK, 'requires explicit CPU export fixture')
    def test_memory_warmup_and_ten_current_samples_one_worker(self):
        class Observer:
            def __init__(self, *, root_pid):
                self.pid = root_pid
                self.index = 0

            def sample(self, *, check):
                check()
                self.index += 1
                return {'status': 'available', 'current_rss_bytes': 100000,
                        'processes': [{'pid': self.pid, 'birth_identity': 'fixture'}]}
        result = execute_scenario({'scenario': 'memory_growth', 'repetitions': 10, 'warmup_jobs': 1},
                                  [self.job], None, self.parent / 'fresh', worker_factory=self.factory,
                                  observer_factory=Observer)
        self.assertEqual(result['status'], 'measured', result)
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(len(result['resources']), 11)
        self.assertTrue(result['resources'][0]['warmup'])
        self.assertAlmostEqual(result['memory_screen']['slope_bytes_per_job'], 0)

    @unittest.skipUnless(STACK, 'requires explicit CPU export fixture')
    def test_concurrent_uses_actual_owned_engine_intervals(self):
        result = self.run_case('concurrent_isolated')
        self.assertIn(result['status'], ('measured', 'not_verified'), result)
        self.assertEqual(len(self.clients), 2)
        self.assertTrue(result['cleanup']['verified'])
        intervals = [event['payload']['execution_interval'] for event in result['raw_records']]
        self.assertEqual({item['pid'] for item in intervals}, {c.process.pid for c in self.clients})
        overlap = min(i['finished_monotonic_ns'] for i in intervals) - max(i['started_monotonic_ns'] for i in intervals)
        self.assertEqual(result['coverage']['overlap_verified'], overlap > 0)
        self.assertEqual(result['status'] == 'measured', overlap > 0)

    def test_nonfinite_deadline_is_bounded_terminal_failure(self):
        result = self.run_case('repeated', timeout_seconds=float('inf'))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('deadline', result['error'])
        self.assertFalse(self.clients)

    @unittest.skipUnless(STACK, 'requires CPU image decoding')
    def test_tiny_real_clip_cannot_count_as_ten_minute_clip(self):
        import cv2
        import numpy as np
        source = self.parent / 'frames'
        source.mkdir()
        self.assertTrue(cv2.imwrite(str(source / 'frame_000000.png'), np.zeros((4, 4, 3), np.uint8)))
        job = JobRequest.create(source, self.original, InferenceConfig(output_format='exr'))
        result = execute_scenario({'scenario': 'long_clip'}, [job], None, self.parent / 'fresh', worker_factory=self.factory)
        self.assertEqual(result['status'], 'not_verified', result)
        self.assertAlmostEqual(result['coverage']['nominal_selected_seconds'], 1 / 24)
        self.assertFalse(self.clients)

    @unittest.skipUnless(STACK, 'requires explicit CPU export fixture')
    def test_incomplete_rss_is_not_zero_or_memory_success(self):
        class Observer:
            def __init__(self, *, root_pid):
                self.pid = root_pid

            def sample(self, *, check):
                return {'status': 'unavailable', 'current_rss_bytes': None, 'reason': 'denied live descendant'}
        result = execute_scenario({'scenario': 'memory_growth', 'repetitions': 10}, [self.job], None,
                                  self.parent / 'fresh', worker_factory=self.factory, observer_factory=Observer)
        self.assertEqual(result['status'], 'not_verified', result)
        self.assertNotIn('memory_screen', result)
        self.assertIsNone(result['resources'][0]['current_rss_bytes'])
        self.assertTrue(result['cleanup']['verified'])

    @unittest.skipUnless(STACK, 'requires explicit CPU export fixture')
    def test_positive_trend_and_threshold_exceedance_cannot_pass_memory(self):
        for growth, expected in ((100, 'not_verified'), (256 * 1024 ** 2, 'failed')):
            with self.subTest(growth=growth):
                class Observer:
                    def __init__(self, *, root_pid, increment=growth):
                        self.index = 0
                        self.increment = increment

                    def sample(self, *, check):
                        self.index += 1
                        return {'status': 'available', 'current_rss_bytes': 100000 + self.index * self.increment}
                result = execute_scenario({'scenario': 'memory_growth', 'repetitions': 10}, [self.job], None,
                                          self.parent / f'growth-{growth}', worker_factory=self.factory,
                                          observer_factory=Observer)
                self.assertEqual(result['status'], expected, result)
                self.assertFalse(result['success'])
                self.assertTrue(result['memory_screen']['review_required'])
                self.assertTrue(result['cleanup']['verified'])

    @unittest.skipUnless(STACK and importlib.util.find_spec('safetensors'), 'requires actual CPU shared-runtime recovery fixture')
    def test_readable_cancel_and_interrupted_resume_match_independent_oracle(self):
        import cv2
        import numpy as np
        source = self.parent / 'actual-frames'
        source.mkdir()
        for index in range(3):
            self.assertTrue(cv2.imwrite(str(source / f'frame_{index:06d}.png'),
                                        np.full((64, 64, 3), index * 20 + 80, np.uint8)))
        original = {p.name: p.read_bytes() for p in source.iterdir()}
        cfg = InferenceConfig(output_format='exr', outputs={'alpha', 'fg'}, exr_dtype='float',
                              hann_chunk=8, hann_stride=4, shape_quantum=64, stage2_batch=2)
        job = JobRequest.create(source, self.original, cfg)
        assets = self.parent / 'fixture-assets'
        assets.mkdir()
        fixture = Path(__file__).with_name('stress_runtime_worker_fixture.py')
        for scenario in ('cancel', 'interruption_recovery'):
            creations = []

            def factory(_creations=creations, **kwargs):
                command = [sys.executable, str(fixture), '--fixture-root', str(assets), '--capacity', '1']
                if not _creations:
                    command.append('--gate-after-readable')
                client = WorkerClient(capacity=1, command=command)
                _creations.append(client)
                return client
            result = execute_scenario({'scenario': scenario, 'timeout_seconds': 60, 'cancel_after_readable_frames': 2}, [job], None,
                                      self.parent / scenario, worker_factory=factory)
            self.assertEqual(result['status'], 'measured', result)
            self.assertTrue(result['cleanup']['verified'], result)
            self.assertTrue(all(c.closed and c.process.poll() is not None for c in creations))
            if scenario == 'interruption_recovery':
                self.assertEqual(len(creations), 3)
                self.assertTrue(result['deliberate_interruption']['partial_sha256'])
                self.assertEqual(result['deliberate_interruption']['readable_frame_indices'], [0, 1])
                self.assertEqual(result['deliberate_interruption']['required_readable_frames'], 2)
                self.assertEqual(result['coverage']['oracle_exact_files'], 6)
                self.assertTrue(result['coverage']['original_partial_preserved'])
            else:
                self.assertEqual(result['fault_observation']['readable_frame_indices'], [0, 1])
            self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, original)

    @unittest.skipUnless(STACK, 'requires explicit CPU export fixture')
    def test_evidence_budget_refuses_before_raw_record_append(self):
        result = execute_scenario({'scenario': 'repeated'}, [self.job], None, self.parent / 'fresh',
                                  worker_factory=self.factory, limits={'max_evidence_bytes': 1})
        self.assertEqual(result['status'], 'failed')
        self.assertIn('record byte bound', result['error'])
        self.assertEqual(result['raw_records'], [])
        self.assertTrue(result['cleanup']['verified'])

    def test_unavailable_faults_never_start_workers_or_fill_volume(self):
        result = self.run_case('disk_full')
        self.assertEqual(result['status'], 'not_verified')
        self.assertFalse(self.clients)
        self.assertEqual(list((self.parent / 'fresh').iterdir()), [])

    def test_batch50_not_repeated_single_clip(self):
        result = self.run_case('batch50')
        self.assertEqual(result['status'], 'not_verified')
        self.assertFalse(self.clients)

    def test_refused_nonempty_root_not_scanned_or_changed(self):
        root = self.parent / 'fresh'
        root.mkdir()
        (root / 'unowned').write_text('preserve')
        with patch('corridorkey.stress_runtime._inventory', side_effect=AssertionError('must not scan')):
            result = self.run_case('repeated')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual((root / 'unowned').read_text(), 'preserve')

    def test_inventory_vanished_queued_scratch_is_absent_but_real_errors_propagate(self):
        from corridorkey.stress_runtime import _inventory

        root = self.parent / 'inventory'
        root.mkdir()
        scratch = root / 'job-scratch'
        scratch.mkdir()
        (root / 'retained').write_bytes(b'owned')
        original = os.scandir
        removed = []

        def remove_before_open(directory):
            if Path(directory) == scratch:
                scratch.rmdir()
                removed.append(scratch)
            return original(directory)

        with patch('corridorkey.stress_runtime.os.scandir', remove_before_open):
            files, size = _inventory(root, lambda: None, 5)
        self.assertEqual(removed, [scratch])
        self.assertEqual(files, {'retained': 5})
        self.assertEqual(size, 5)
        with patch('corridorkey.stress_runtime.os.scandir', side_effect=PermissionError('denied live subtree')), self.assertRaises(PermissionError):
            _inventory(root, lambda: None, 5)
        with self.assertRaises(FileNotFoundError):
            _inventory(root / 'missing-root', lambda: None, 5)
        with self.assertRaisesRegex(RuntimeError, 'byte ceiling'):
            _inventory(root, lambda: None, 4)

    def test_external_cancellation_is_terminal_and_no_owner_leak(self):
        def cancel():
            raise CancelledError('explicit stop')
        result = execute_scenario({'scenario': 'repeated'}, [self.job], None, self.parent / 'fresh',
                                  check=cancel, worker_factory=self.factory)
        self.assertEqual(result['status'], 'cancelled')
        self.assertFalse(self.clients)

    @unittest.skipUnless(STACK, 'requires explicit CPU export fixture')
    def test_failed_render_remains_failure_and_cleanup_verified(self):
        self.source.write_text(json.dumps({'fail': True}))
        result = self.run_case('repeated')
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(result['cleanup']['verified'])
        self.assertEqual(result['raw_records'][0]['kind'], 'failed')

    @unittest.skipUnless(STACK, 'requires explicit CPU export fixture')
    def test_artifact_cap_stops_and_reaps_owned_worker(self):
        result = execute_scenario({'scenario': 'repeated'}, [self.job], None, self.parent / 'fresh',
                                  worker_factory=self.factory, limits={'max_artifact_bytes': 1})
        self.assertEqual(result['status'], 'failed')
        self.assertIn('ceiling', result['error'])
        self.assertTrue(all(c.closed and c.process.poll() is not None for c in self.clients))


if __name__ == '__main__':
    unittest.main()
