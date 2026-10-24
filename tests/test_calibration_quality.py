"""Analytic reference-agreement policies; CPU fixtures never produce CUDA recommendations."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.calibration import (
    CalibrationLimits,
    inventory_calibration,
    plan_calibration,
    revalidate_plan,
    summarize_trials,
)
from corridorkey.calibration_task import evaluate_trial_quality, run_calibration
from corridorkey.control import CancellationToken
from tests.test_calibration import calibration_fixture
from tests.test_calibration_task import FakeClient


class QualityFixtureClient(FakeClient):
    """Actual native analytic exports, explicitly non-neural test provider."""
    def send(self, action, payload):
        request = super().send(action, payload)
        if action == 'cancel':
            return request
        record = self.event['payload']['record']
        record['job_id'] = f'fixture-{id(self)}-{self.jobs}'
        diagnostics = record['diagnostics']
        diagnostics['source']['output_hw'] = [8, 12]
        diagnostics['output_contracts']['alpha']['alpha_representation'] = 'coverage (no transfer function)'
        destination = Path(diagnostics['outputs']['alpha'])
        destination.mkdir(parents=True)
        error = 0.125 if self.jobs > 1 else 0.25
        for index in range(diagnostics['frames_submitted']):
            if not cv2.imwrite(str(destination / f'alpha_{index:06d}.exr'), np.full((8, 12), 0.5 + error, np.float32)):
                raise AssertionError('Native analytic EXR fixture export failed')
        return request


class CalibrationQualityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.job, self.options = calibration_fixture(self.root)
        self.reference = self.root / 'reference.npy'
        np.save(self.reference, np.full((8, 12), 0.5, np.float32))
        self.manifest = self.root / 'reference.json'
        self.manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'alpha_sequence_evaluation',
            'fps': 24, 'source': self.job.input_path,
            'reference_provenance': {'origin': 'analytic fixture', 'author': 'test', 'permitted_use': 'local', 'split': 'synthetic'},
            'frames': [{'frame': i, 'candidate': 'replaced.exr', 'reference': str(self.reference)} for i in range(2)]}))
        self.policy = {'schema_version': 1, 'kind': 'calibration_quality_policy',
                       'objective': 'fastest_warm_reference_agreement', 'manifest': self.manifest.name,
                       'limits': [{'metric': 'alpha.mae', 'units': 'alpha/pixel', 'maximum': 0.125}],
                       'accepted_tradeoffs': ['Reference agreement only; independent visual review still required.']}
        self.path = self.root / 'quality-policy.json'
        self.save()
        self.enterContext(patch.dict(os.environ, {'OPENCV_IO_ENABLE_OPENEXR': '1'}))
        FakeClient.instances = []
        FakeClient.live = 0

    def save(self):
        self.path.write_text(json.dumps(self.policy))

    def plan(self):
        return plan_calibration(self.job, **self.options, quality_policy_path=self.path)

    def test_policy_pinned_immutable_and_stale_references_rejected(self):
        plan = self.plan()
        self.assertEqual(plan.to_dict()['quality']['policy'], self.policy)
        self.assertIn(str(self.reference), [pin.path for pin in plan.pins])
        np.save(self.reference, np.full((8, 12), 0.4, np.float32))
        with self.assertRaises(ValueError):
            revalidate_plan(plan)

    def test_malformed_or_unsupported_policy_never_falls_back_to_timing(self):
        for defect in ('unknown', 'units', 'negative', 'memory', 'missing_metric'):
            with self.subTest(defect=defect):
                policy = copy.deepcopy(self.policy)
                if defect == 'unknown':
                    policy['extra'] = True
                elif defect == 'units':
                    policy['limits'][0]['units'] = 'seconds'
                elif defect == 'negative':
                    policy['limits'][0]['maximum'] = -1
                elif defect == 'memory':
                    policy['objective'] = 'minimum_memory'
                else:
                    policy['limits'][0]['metric'] = 'colour.composite_mae'
                    policy['limits'][0]['units'] = 'linear RGB/channel'
                self.path.write_text(json.dumps(policy))
                with self.assertRaises(ValueError):
                    self.plan()

    def test_read_budget_identity_bounds_and_reference_artifact_collisions(self):
        with self.assertRaisesRegex(ValueError, 'read budget'):
            plan_calibration(self.job, **dict(self.options, limits=CalibrationLimits(read_budget=1)), quality_policy_path=self.path)
        summary = inventory_calibration(self.job, **self.options, quality_policy_path=self.path)
        self.assertEqual(summary['quality_objective'], 'fastest_warm_reference_agreement')
        planned = self.plan()
        self.path.write_text(self.path.read_text() + ' ')
        with self.assertRaises(ValueError):
            revalidate_plan(planned)
        self.save()
        copied = self.options['scratch_parent'] / 'reference.npy'
        copied.write_bytes(self.reference.read_bytes())
        raw = json.loads(self.manifest.read_text())
        raw['frames'][0]['reference'] = str(copied)
        self.manifest.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'artifact parents'):
            self.plan()

    def test_actual_export_metrics_threshold_exclusions_and_fixture_never_recommends(self):
        report = run_calibration(self.plan(), worker_factory=QualityFixtureClient).to_dict()
        self.assertEqual(report['status'], 'complete', report.get('error'))
        self.assertEqual(report['summary']['recommendations'], [])
        for trial in report['trials']:
            expected = 'pass' if trial['state'] == 'warm_session' else 'fail'
            self.assertEqual(trial['quality']['status'], expected, trial)
            self.assertEqual(trial['quality']['metrics'][0]['value'], 0.125 if expected == 'pass' else 0.25)
            self.assertTrue(Path(trial['quality']['report']['path']).is_file())
            self.assertFalse(trial['eligible'])
            self.assertIn('excluded from cold/warm', trial['quality_timing_scope'])
        self.policy['limits'][0]['maximum'] = 0.124
        self.save()
        report = run_calibration(self.plan(), worker_factory=QualityFixtureClient).to_dict()
        self.assertTrue(all(t['quality']['status'] == 'fail' for t in report['trials']))

    def test_evaluation_cost_excluded_from_trial_timers_and_cancel_fail_closed(self):
        ticks = [0.0]
        def clock():
            ticks[0] += 0.01
            return ticks[0]
        def slow(quality, record, **kwargs):
            outcome = evaluate_trial_quality(quality, record, **kwargs)
            ticks[0] += 10
            return outcome
        with patch('corridorkey.calibration_task.evaluate_trial_quality', side_effect=slow):
            report = run_calibration(self.plan(), worker_factory=QualityFixtureClient, clock=clock).to_dict()
        self.assertTrue(all(t['elapsed_seconds'] < 10 for t in report['trials']))
        self.assertTrue(all(t['quality_evaluation_seconds'] >= 10 for t in report['trials']))
        token = CancellationToken()
        def cancel_after(quality, record):
            outcome = original(quality, record)
            token.cancel('quality cancellation fixture')
            return outcome
        from corridorkey.benchmark_quality import evaluate_job_quality as original
        with patch('corridorkey.benchmark_quality.evaluate_job_quality', side_effect=cancel_after):
            report = run_calibration(self.plan(), worker_factory=QualityFixtureClient, cancellation=token).to_dict()
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['summary']['recommendations'], [])
        self.assertEqual(report['trials'][0]['quality']['status'], 'not_verified')
        self.assertTrue(all(t['eligible'] is False for t in report['trials']))

    def test_missing_measured_metric_remains_unavailable_with_raw_report(self):
        from corridorkey.benchmark_quality import evaluate_job_quality
        plan = self.plan()
        client = QualityFixtureClient()
        wire = plan.to_dict()['profiles'][0]['job']
        wire['output_dir'] = str(self.root / 'analytic-missing-metric')
        try:
            client.send('preview', {'job': wire})
            record = client.event['payload']['record']
            quality = plan.to_dict()['quality']
            native = evaluate_job_quality(quality['template'], record)
            del native['candidate']['mae']
            with patch('corridorkey.benchmark_quality.evaluate_job_quality', return_value=native):
                outcome, raw = evaluate_trial_quality(quality, record)
            self.assertEqual(outcome['status'], 'not_verified')
            self.assertIn('reason', outcome['metrics'][0])
            self.assertIs(raw, native)
        finally:
            client.close()

    def test_quality_all_repetitions_gate_recommendation_and_legacy_stays_timing(self):
        plan = self.plan().to_dict()
        trials = [{'profile': 'baseline', 'state': 'warm_session', 'eligible': True, 'status': 'complete',
                   'repetition': i, 'elapsed_seconds': 1 + i, 'quality': {'status': 'pass'},
                   'measurement': {'neural_gpu_verified': True, 'frames': 2, 'processing_seconds': 1}} for i in range(3)]
        trials += [dict(row, state='cold_process') for row in copy.deepcopy(trials)]
        data = {'plan': plan, 'trials': trials, 'status': 'complete'}
        self.assertIn('reference', summarize_trials(data)['recommendations'][0]['basis'])
        trials[1]['quality']['status'] = 'not_verified'
        self.assertEqual(summarize_trials(data)['recommendations'], [])
        del plan['quality']
        self.assertIn('quality', summarize_trials(data)['recommendations'][0]['basis'])


if __name__ == '__main__':
    unittest.main()
