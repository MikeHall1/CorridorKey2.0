"""Portable analytic report fixtures; these are not neural performance measurements."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from corridorkey import regression_gate as gate


class RegressionGateTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.report_path, self.policy_path = self.root / 'pair.json', self.root / 'policy.json'
        self.report = {'schema_version': 1, 'kind': 'inference_pair', 'status': 'complete',
                       'summary': {'success': True, 'speedup': 1}, 'pairs': [],
                       'conditions': dict.fromkeys(('order', 'selection', 'filesystem', 'output', 'warm_session', 'timing', 'quality'), 'fixture'),
                       'provenance': {}}
        for side, cache in (('baseline', 0), ('candidate', 1)):
            settings = {'auto_vram': False, 'foundation_cache_mb': cache}
            self.report['provenance'][side] = {
                'normalized_batch': {'weights': 'fixture', 'jobs': [
                    {'input': 'local-source', 'output_dir': side, 'settings': settings, 'writer': {}}]},
                'frozen_manifest': {'model_state': 'warm_session'}}
        for index in range(3):
            samples = {}
            for side in gate.SIDES:
                settings = copy.deepcopy(self.report['provenance'][side]['normalized_batch']['jobs'][0]['settings'])
                samples[side] = {'summary': {'success': True}, 'inference': {'status': 'measured'},
                    'repository': {'commit': 'synthetic'}, 'assets': {'source': 'fixture'},
                    'weights': {'fixture': '0' * 64}, 'corrections': [None],
                    'environment': {'python': 'fixture', 'os': 'fixture', 'dependencies': {'test': '1'},
                                    'hardware': {'cpu': 'synthetic'}},
                    'runs': [{'elapsed_seconds': 1.0, 'jobs': [{'status': 'complete',
                        'requested_settings': settings, 'diagnostics': {'effective_settings': copy.deepcopy(settings)}}]}]}
            for side, sample in samples.items():
                sample.update(model_state='warm_session', manifest=copy.deepcopy(self.report['provenance'][side]['frozen_manifest']),
                              cache={'filesystem': 'uncontrolled', 'intermediate': 'per_job_temporary_only', 'output': 'fresh_destination_per_run'})
                job = sample['runs'][0]['jobs'][0]
                job.update(input='local-source', writer_settings={}, hints={'first_frame': None, 'full': None},
                           model_reused=True, job_id=f'{side}-{index}', output_dir=str(self.root / side / str(index)))
                warmup = copy.deepcopy(job)
                warmup.update(model_reused=False, job_id=f'warmup-{side}-{index}')
                sample['warmup'] = [warmup]
            self.report['pairs'].append({'index': index, 'order': list(gate.SIDES if index % 2 == 0 else reversed(gate.SIDES)), 'samples': samples})
        self.policy = {'schema_version': 1, 'kind': 'regression_policy',
                       'context_sha256': gate.digest(gate.context(self.report)), 'repetitions': 3,
                       'performance': {'max_elapsed_regression': 0.25, 'max_relative_spread': 0.1},
                       'quality': [], 'required_evidence': [], 'visual_review_required': False,
                       'accepted_tradeoffs': ['Timing screen only; no quality acceptance.']}

    def save(self):
        self.report_path.write_text(json.dumps(self.report))
        self.policy_path.write_text(json.dumps(self.policy))
        return hashlib.sha256(self.policy_path.read_bytes()).hexdigest()

    def run_gate(self):
        return gate.evaluate(self.report_path, self.policy_path, self.save())

    def times(self, values):
        for pair, value in zip(self.report['pairs'], values, strict=True):
            pair['samples']['candidate']['runs'][0]['elapsed_seconds'] = value

    def test_boundary_and_deliberate_synthetic_regression_derive_raw_not_summary(self):
        for duration, expected in ((1, 'pass'), (1.25, 'pass'), (1.2501, 'fail'), (2, 'fail')):
            with self.subTest(duration=duration):
                self.times([duration] * 3)
                result = self.run_gate()
                self.assertEqual(result['status'], expected, result)
                self.assertIn('NOT VERIFIED', result['predeclaration'])
                self.assertEqual(result['source']['sha256'], hashlib.sha256(self.report_path.read_bytes()).hexdigest())

    def test_noise_and_threshold_crossing_are_inconclusive(self):
        for times in ([1, 1.5, 2], [1.24, 1.25, 1.26]):
            self.times(times)
            self.assertEqual(self.run_gate()['status'], 'inconclusive')

    def test_missing_mismatch_cancelled_and_tamper_never_pass(self):
        original = copy.deepcopy(self.report)
        for defect in ('settings', 'hardware', 'cache', 'missing', 'cancelled', 'repetitions'):
            with self.subTest(defect=defect):
                self.report = copy.deepcopy(original)
                sample = self.report['pairs'][1]['samples']['candidate']
                if defect == 'settings':
                    sample['runs'][0]['jobs'][0]['diagnostics']['effective_settings']['auto_vram'] = True
                elif defect == 'hardware':
                    sample['environment']['hardware']['cpu'] = 'other'
                elif defect == 'cache':
                    self.report['conditions']['filesystem'] = 'changed'
                elif defect == 'missing':
                    del self.report['provenance']
                elif defect == 'cancelled':
                    self.report.update(status='cancelled', summary={'success': False, 'speedup': None})
                else:
                    self.report['pairs'].pop()
                self.assertEqual(self.run_gate()['status'], 'not_verified')
        self.report = original
        pinned = self.save()
        self.policy_path.write_text(self.policy_path.read_text() + ' ')
        self.assertEqual(gate.evaluate(self.report_path, self.policy_path, pinned)['status'], 'not_verified')

    def test_missing_quality_numerical_memory_visual_stay_not_verified(self):
        self.policy['quality'] = [{'job': 0, 'metric': 'alpha.mae', 'units': 'alpha/pixel', 'max_increase': 0.01}]
        self.policy['required_evidence'] = ['numerical_parity', 'observed_memory']
        self.policy['visual_review_required'] = True
        result = self.run_gate()
        self.assertEqual(result['status'], 'not_verified')
        self.assertEqual(sum(c['status'] == 'not_verified' for c in result['checks']), 4)

    def test_real_analytic_quality_regression_and_reference_mismatch(self):
        import os
        from unittest.mock import patch

        import cv2
        import numpy as np

        from corridorkey.evaluation_media import evaluate_sequence_spec

        source = self.root / 'source.bin'
        source.write_bytes(b'analytic source')
        reference = self.root / 'reference.npy'
        np.save(reference, np.full((2, 2), 0.5, np.float32))
        quality = {}
        with patch.dict(os.environ, {'OPENCV_IO_ENABLE_OPENEXR': '1'}):
            for side, error in (('baseline', 0.0), ('candidate', 0.125)):
                image = self.root / f'{side}.exr'
                self.assertTrue(cv2.imwrite(str(image), np.full((2, 2), 0.5 + error, np.float32)))
                spec = {'schema_version': 1, 'kind': 'alpha_sequence_evaluation', 'fps': 24,
                        'source': str(source), 'frames': [{'frame': 0, 'candidate': str(image), 'reference': str(reference)}],
                        'reference_provenance': {'origin': 'analytic fixture', 'author': 'test', 'permitted_use': 'local', 'split': 'synthetic'}}
                q = evaluate_sequence_spec(spec, manifest_path=self.root / 'template.json', manifest_sha256='0' * 64)
                q['candidate_binding'] = {'method': 'completed native EXR exports, output-relative frame IDs',
                    'source_start_frame': 0, 'frames_submitted': 1, 'selected_frames': 1, 'known_shot_boundaries': None}
                quality[side] = q
        for pair in self.report['pairs']:
            for side in gate.SIDES:
                pair['samples'][side]['inference']['quality'] = {'jobs': [
                    {'run': 0, 'job': 0, 'status': 'measured', 'report': copy.deepcopy(quality[side])}]}
        for side in gate.SIDES:
            self.report['provenance'][side]['normalized_batch']['jobs'][0]['input'] = str(source)
        for pair in self.report['pairs']:
            for side in gate.SIDES:
                sample = pair['samples'][side]
                job = sample['runs'][0]['jobs'][0]
                job.update(input=str(source), output_dir=str(self.root))
                sample['warmup'][0]['input'] = str(source)
                job['diagnostics'].update(source={'start_frame': 0, 'native_fps': 24}, frames_submitted=1)
                binding = sample['inference']['quality']['jobs'][0]['report']['candidate_binding']
                binding.update(job_id=job['job_id'], output_dir=job['output_dir'])
        self.policy['context_sha256'] = gate.digest(gate.context(self.report))
        self.policy['quality'] = [{'job': 0, 'metric': 'alpha.mae', 'units': 'alpha/pixel', 'max_increase': 0.125}]
        self.assertEqual(self.run_gate()['status'], 'pass')
        self.policy['quality'][0]['max_increase'] = 0.124
        self.assertEqual(self.run_gate()['status'], 'fail')
        original = copy.deepcopy(self.report)
        for pair in self.report['pairs']:
            for sample in pair['samples'].values():
                q = sample['inference']['quality']['jobs'][0]['report']
                q['reference_provenance']['origin'] = 'changed'
                for asset in q['assets']:
                    if asset['role'] == 'reference':
                        asset['sha256'] = 'e' * 64
        self.assertEqual(self.run_gate()['status'], 'not_verified')
        self.report = original
        self.report['pairs'][0]['samples']['candidate']['inference']['quality']['jobs'][0]['report']['source']['sha256'] = 'f' * 64
        self.assertEqual(self.run_gate()['status'], 'not_verified')

    def test_actual_lifecycle_input_writer_hints_cache_and_warmup_are_bound(self):
        original = copy.deepcopy(self.report)
        for defect in ('model_state', 'input', 'writer', 'hints', 'cache', 'warmup'):
            with self.subTest(defect=defect):
                self.report = copy.deepcopy(original)
                for pair in self.report['pairs']:
                    for sample in pair['samples'].values():
                        job = sample['runs'][0]['jobs'][0]
                        if defect == 'model_state':
                            sample['model_state'] = 'cold_process'
                        elif defect == 'input':
                            job['input'] = 'other'
                        elif defect == 'writer':
                            job['writer_settings'] = {'other': True}
                        elif defect == 'hints':
                            job['hints']['full'] = 'other'
                        elif defect == 'cache':
                            sample['cache']['filesystem'] = 'changed'
                        else:
                            sample['warmup'] = []
                self.assertEqual(self.run_gate()['status'], 'not_verified')

    def test_derived_arithmetic_extremes_are_not_verified_and_finite_json(self):
        for baseline, candidate in ((1e308, 1e-308), (1e-308, 1e308)):
            for pair in self.report['pairs']:
                pair['samples']['baseline']['runs'][0]['elapsed_seconds'] = baseline
                pair['samples']['candidate']['runs'][0]['elapsed_seconds'] = candidate
            result = self.run_gate()
            self.assertEqual(result['status'], 'not_verified')
            json.dumps(result, allow_nan=False)

    def test_cooperative_cancellation(self):
        pinned = self.save()
        def cancel():
            raise gate.CancelledError('fixture')
        self.assertEqual(gate.evaluate(self.report_path, self.policy_path, pinned, check=cancel)['status'], 'cancelled')

    def test_cli_exclusive_alias_safe_output_and_source_preservation(self):
        pinned = self.save()
        original = self.report_path.read_bytes()
        for destination in (self.report_path, self.policy_path, self.root / 'PAIR.JSON'):
            with self.subTest(destination=destination), self.assertRaises((ValueError, FileExistsError)):
                gate.main(['--report', str(self.report_path), '--policy', str(self.policy_path),
                           '--policy-sha256', pinned, '--output', str(destination)])
        alias = self.root / 'alias.json'
        alias.symlink_to(self.report_path)
        with self.assertRaises(ValueError):
            gate.main(['--report', str(self.report_path), '--policy', str(self.policy_path),
                       '--policy-sha256', pinned, '--output', str(alias)])
        output = self.root / 'gate.json'
        args = ['--report', str(self.report_path), '--policy', str(self.policy_path),
                '--policy-sha256', pinned, '--output', str(output)]
        self.assertEqual(gate.main(args), 0)
        with self.assertRaises(FileExistsError):
            gate.main(args)
        self.assertEqual(self.report_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
