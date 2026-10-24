"""Native EXR-derived paired evidence, aggregation and refusal contracts."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'cv2'))


@unittest.skipUnless(HAS_STACK, 'requires native CPU EXR metric stack')
class EvaluationDossierTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'OPENCV_IO_ENABLE_OPENEXR': '1'}))
        import cv2
        import numpy as np

        from corridorkey.evaluation_media import evaluate_sequence_spec
        from corridorkey.identities import file_sha256
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'generated local analytic fixture; no neural inference')
        self.reference = self.root / 'reference.npy'
        np.save(self.reference, np.full((2, 2), 0.5, np.float32))
        self.evaluate = evaluate_sequence_spec
        self.hash = file_sha256
        self.cv2, self.np = cv2, np

    def quality(self, name, error, *, mask=None):
        path = self.root / f'{name}.exr'
        self.assertTrue(self.cv2.imwrite(str(path), self.np.full((2, 2), 0.5 + error, self.np.float32)))
        spec = {'schema_version': 1, 'kind': 'alpha_sequence_evaluation', 'fps': 24,
            'source': str(self.source), 'frames': [{'frame': 0, 'candidate': str(path), 'reference': str(self.reference)}],
            'reference_provenance': {'origin': 'analytic fixture', 'author': 'test', 'permitted_use': 'local', 'split': 'synthetic'}}
        if mask is not None:
            spec['frames'][0]['mask'] = str(mask)
        report = self.evaluate(spec, manifest_path=self.root / 'quality-template.json', manifest_sha256='0' * 64)
        report['candidate_binding'] = {'method': 'completed native EXR exports, output-relative frame IDs',
            'source_start_frame': 0, 'frames_submitted': 1, 'selected_frames': 1,
            'known_shot_boundaries': None, 'job_id': 'generated-fixture', 'output_dir': str(self.root)}
        return report

    def sample(self, qualities):
        return {'schema_version': 1, 'kind': 'inference', 'summary': {'success': True},
            'repository': {'working_tree_sha256': '0' * 64}, 'assets': {'fixture': 'synthetic'},
            'weights': {'fixture': '0' * 64}, 'corrections': [None],
            'environment': {'python': 'fixture', 'os': 'fixture', 'dependencies': {'fixture': '1'},
                            'hardware': {'cpu': 'fixture', 'gpu': None}},
            'inference': {'status': 'measured', 'quality': {'status': 'measured', 'jobs': [
                {'run': 0, 'job': i, 'status': q['candidate']['status'], 'report': q} for i, q in enumerate(qualities)]}},
            'runs': [{'elapsed_seconds': 1.0, 'jobs': [{'status': 'complete'} for _ in qualities]}]}

    def pair(self, errors, *, repetitions=3, masks=None):
        baseline = self.sample([self.quality(f'b{i}', before, mask=(masks or [None] * len(errors))[i])
                                for i, (before, _) in enumerate(errors)])
        candidate = self.sample([self.quality(f'c{i}', after, mask=(masks or [None] * len(errors))[i])
                                 for i, (_, after) in enumerate(errors)])
        return {'schema_version': 1, 'kind': 'inference_pair', 'status': 'complete',
            'summary': {'success': True, 'speedup': 1.0}, 'pairs': [
                {'index': i, 'order': ['baseline', 'candidate'] if i % 2 == 0 else ['candidate', 'baseline'],
                 'samples': {'baseline': copy.deepcopy(baseline), 'candidate': copy.deepcopy(candidate)}}
                for i in range(repetitions)]}

    def manifest(self, report, *, policy=None):
        source = self.root / 'pair.json'
        source.write_text(json.dumps(report, allow_nan=False))
        count = len(report['pairs'][0]['samples']['baseline']['runs'][0]['jobs'])
        manifest = {'schema_version': 1, 'kind': 'evaluation_dossier', 'reports': [{'id': 'pair', 'path': source.name}],
            'cases': [{'id': f'clip{i}', 'report_id': 'pair', 'job_index': i, 'categories': ['synthetic'],
                       'split': 'fixture'} for i in range(count)]}
        if policy is not None:
            manifest['policy'] = policy
        path = self.root / 'dossier.json'
        path.write_text(json.dumps(manifest))
        return path

    def build(self, report, **kwargs):
        from corridorkey.evaluation_dossier import build_dossier
        terminal = build_dossier(self.manifest(report, **kwargs))
        self.assertEqual(terminal['status'], 'complete', terminal)
        return terminal['report']

    def test_native_category_median_unchanged_but_worst20percent_exposed(self):
        from corridorkey.evaluation_dossier import revalidate_inputs
        report = self.build(self.pair([(0.1, 0.09), (0.1, 0.1), (0.1, 0.12)]))
        metric = report['categories'][0]['metrics']['alpha.mae']
        self.assertAlmostEqual(metric['relative_delta'], 0.0)
        self.assertEqual(metric['worst_case_median']['case_id'], 'clip2')
        self.assertAlmostEqual(metric['worst_case_median']['relative_delta'], 0.2, places=5)
        self.assertEqual(metric['screen'], 'regression')
        self.assertEqual(metric['worst_candidate_error']['case_id'], 'clip2')
        self.assertAlmostEqual(metric['worst_candidate_error']['value'], 0.12, places=6)
        self.assertEqual(report['acceptance_verdict'], None)
        self.assertEqual(report['review_status'], 'pending')
        self.assertEqual(len(report['review_media']), 6)
        self.assertEqual(revalidate_inputs(report)['token'], report['token'])
        before = self.source.read_bytes()
        self.assertEqual(before, b'generated local analytic fixture; no neural inference')

    def test_population_pool_and_equal_case_median_are_distinct(self):
        # 1 versus3 selected pixels in the native2x2 fixture.
        masks = []
        for count in (1, 3):
            path = self.root / f'mask{count}.npy'
            mask = self.np.zeros((2, 2), bool)
            mask.flat[:count] = True
            self.np.save(path, mask)
            masks.append(path)
        report = self.build(self.pair([(0.2, 0.2), (0.1, 0.1)], masks=masks))
        metric = report['categories'][0]['metrics']['alpha.mae']
        self.assertAlmostEqual(metric['baseline_median'], 0.15, places=6)
        self.assertAlmostEqual(metric['pooled']['baseline'], 0.125, places=6)
        self.assertEqual(metric['coverage']['assessed'], 2)

    def test_zero_baseline_uses_only_explicit_absolute_tolerance(self):
        pairs = self.pair([(0, 0.001)])
        report = self.build(pairs)
        metric = report['cases'][0]['metrics']['alpha.mae']
        self.assertIsNone(metric['relative_delta'])
        self.assertEqual(metric['screen'], 'unassessed')
        report = self.build(pairs, policy={'zero_baseline_absolute_tolerances': {
            'alpha.mae': {'units': 'alpha/pixel', 'value': 0.0005}}})
        self.assertEqual(report['cases'][0]['metrics']['alpha.mae']['screen'], 'regression')
        self.assertIsNone(report['cases'][0]['metrics']['alpha.mae']['relative_delta'])

    def test_reference_units_coverage_and_conditions_mismatch_are_unassessed(self):
        for change in ('reference', 'units', 'coverage', 'conditions'):
            pairs = self.pair([(0.1, 0.11)])
            sample = pairs['pairs'][0]['samples']['candidate']
            quality = sample['inference']['quality']['jobs'][0]['report']
            if change == 'reference':
                next(a for a in quality['assets'] if a['role'] == 'reference')['sha256'] = 'f' * 64
            elif change == 'units':
                quality['candidate']['mae']['units'] = 'linear RGB/channel'
            elif change == 'coverage':
                quality['frame_ids'] = [4]
            else:
                sample['weights']['fixture'] = 'f' * 64
            report = self.build(pairs)
            metric = report['cases'][0]['metrics']['alpha.mae']
            self.assertIn(metric['status'], ('partial', 'unassessed'))
            self.assertGreaterEqual(metric['coverage']['unassessed'], 1)
            self.assertIn(metric['screen'], ('unassessed', 'regression'))

    def test_repeated_contract_changes_on_both_sides_are_not_pooled(self):
        for field in ('decode_contract', 'shape_hw', 'frames', 'population'):
            with self.subTest(field=field):
                pairs = self.pair([(0.1, 0.1)])
                for side in ('baseline', 'candidate'):
                    quality = pairs['pairs'][1]['samples'][side]['inference']['quality']['jobs'][0]['report']
                    if field == 'decode_contract':
                        quality[field] = 'different normalization for both timed sides'
                    elif field == 'population':
                        metric = quality['candidate']['mae']
                        metric['population'] = 2
                        metric['error_sum'] = metric['value'] * 2
                    else:
                        quality['candidate'][field] = [1, 4] if field == 'shape_hw' else 2
                report = self.build(pairs)
                metric = report['cases'][0]['metrics']['alpha.mae']
                self.assertEqual(metric['coverage'], {'expected': 3, 'assessed': 2, 'unassessed': 1})
                self.assertEqual(metric['status'], 'partial')
                self.assertEqual(metric['screen'], 'unassessed')
                self.assertIn('contracts changed', metric['repetitions'][1]['reason'])

    def test_temporal_repeat_contract_retains_validity_but_not_observed_error(self):
        from corridorkey.evaluation_dossier import _measurement_contract
        measured = {'shape_hw': [2, 2], 'frames': 2, 'flow_convention': 'declared',
            'consistency_pixels': 1.5, 'pairs': [{'previous_frame': 0, 'frame': 1,
                'valid_pixels': 4, 'cut_excluded': False, 'temporal_residual_mae': 0.1}],
            'temporal_residual_mae': {'value': 0.1, 'error_sum': 0.4, 'population': 4,
                'units': 'alpha/pixel pair', 'status': 'measured'}}
        changed_error = copy.deepcopy(measured)
        changed_error['pairs'][0]['temporal_residual_mae'] = 0.2
        changed_error['temporal_residual_mae']['value'] = 0.2
        changed_error['temporal_residual_mae']['error_sum'] = 0.8
        self.assertEqual(_measurement_contract(measured), _measurement_contract(changed_error))
        changed_population = copy.deepcopy(changed_error)
        changed_population['temporal_residual_mae']['population'] = 3
        self.assertNotEqual(_measurement_contract(measured), _measurement_contract(changed_population))
        changed_error['pairs'][0]['valid_pixels'] = 3
        self.assertNotEqual(_measurement_contract(measured), _measurement_contract(changed_error))

    def test_external_quality_baseline_and_quality_pairs_not_double_counted(self):
        pairs = self.pair([(0.1, 0.12)])
        for pair in pairs['pairs']:
            for side in ('baseline', 'candidate'):
                quality = pair['samples'][side]['inference']['quality']['jobs'][0]['report']
                quality['baseline'] = {'mae': {'value': 999}}
        pairs['quality_pairs'] = [{'not': 'canonical evidence'}] * 99
        report = self.build(pairs)
        metric = report['cases'][0]['metrics']['alpha.mae']
        self.assertEqual(metric['coverage']['expected'], 3)
        self.assertAlmostEqual(metric['relative_delta'], 0.2, places=5)

    def test_unavailable_references_empty_masks_and_failed_pair_stay_unassessed(self):
        mask = self.root / 'empty.npy'
        self.np.save(mask, self.np.zeros((2, 2), bool))
        pairs = self.pair([(0.1, 0.1)], masks=[mask])
        report = self.build(pairs)
        self.assertEqual(report['cases'][0]['status'], 'unassessed')
        self.assertIsNone(report['cases'][0]['metrics']['alpha.mae']['candidate_median'])
        pairs = self.pair([(0.1, 0.1)])
        pairs['status'] = 'cancelled'
        pairs['summary'] = {'success': False, 'speedup': None}
        report = self.build(pairs)
        self.assertEqual(report['coverage']['unassessed_cases'], 1)

    def test_colour_background_contract_and_legacy_are_explicit(self):
        from corridorkey.evaluation_colour_media import evaluate_colour_spec
        reference = self.root / 'reference-rgb.npy'
        candidate = self.root / 'candidate-rgb.exr'
        mask = self.root / 'foreground-mask.npy'
        self.np.save(reference, self.np.full((2, 2, 3), 0.25, self.np.float32))
        self.np.save(mask, self.np.ones((2, 2), bool))
        self.assertTrue(self.cv2.imwrite(str(candidate), self.np.full((2, 2, 3), 0.35, self.np.float32)))
        alpha = self.quality('colour-alpha', 0.1)
        alpha_path = next(a['path'] for a in alpha['assets'] if a['role'] == 'candidate')
        spec = {'schema_version': 1, 'kind': 'colour_sequence_evaluation', 'fps': 24,
            'source': str(self.source), 'colour_space': 'linear_srgb_bt709',
            'associations': {'candidate': 'straight', 'reference': 'straight'}, 'background': [0, 0, 0],
            'reference_provenance': alpha['reference_provenance'], 'frames': [{'frame': 0,
                'candidate_rgb': str(candidate), 'candidate_alpha': alpha_path, 'reference_rgb': str(reference),
                'reference_alpha': str(self.reference), 'foreground_mask': str(mask)}]}
        colour = evaluate_colour_spec(spec, manifest_path=self.root/'colour.json', manifest_sha256='0'*64)
        colour['candidate_binding'] = alpha['candidate_binding']
        pairs = self.pair([(0.1, 0.1)])
        for pair in pairs['pairs']:
            for side in ('baseline', 'candidate'):
                entry = pair['samples'][side]['inference']['quality']['jobs'][0]
                entry['status'], entry['report'] = 'measured', copy.deepcopy(colour)
        report = self.build(pairs)
        self.assertEqual(report['cases'][0]['metrics']['colour.composite_mae']['status'], 'assessed')
        for pair in pairs['pairs']:
            pair['samples']['candidate']['inference']['quality']['jobs'][0]['report']['evaluation_contract']['constant_background'] = [1, 1, 1]
        report = self.build(pairs)
        self.assertEqual(report['cases'][0]['metrics']['colour.composite_mae']['status'], 'unassessed')
        for pair in pairs['pairs']:
            for side in ('baseline', 'candidate'):
                pair['samples'][side]['inference']['quality']['jobs'][0]['report'].pop('evaluation_contract', None)
        report = self.build(pairs)
        self.assertEqual(report['cases'][0]['metrics']['colour.composite_mae']['status'], 'unassessed')

    def test_input_staleness_token_tamper_and_cancellation(self):
        from corridorkey.control import CancellationToken
        from corridorkey.evaluation_dossier import build_dossier, revalidate_inputs
        pairs = self.pair([(0.1, 0.1)])
        report = self.build(pairs)
        bad = copy.deepcopy(report)
        bad['review_status'] = 'approved'
        with self.assertRaises(ValueError):
            revalidate_inputs(bad)
        path = self.root / 'pair.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            revalidate_inputs(report)
        token = CancellationToken()
        token.cancel()
        result = build_dossier(self.root / 'dossier.json', check=token.check)
        self.assertEqual(result['status'], 'cancelled')
        self.assertIsNone(result['report'])

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_fifo_manifest_report_and_final_fence_refuse_without_blocking(self):
        fifo = self.root / 'blocked.json'
        mkfifo = getattr(os, 'mkfifo', None)
        assert callable(mkfifo)
        mkfifo(fifo)
        script = ('import json,sys; from corridorkey.evaluation_dossier import build_dossier; '
                  'print(json.dumps(build_dossier(sys.argv[1])))')
        for target in ('manifest', 'report'):
            if target == 'manifest':
                path = fifo
            else:
                path = self.manifest(self.pair([(0.1, 0.1)]))
                manifest = json.loads(path.read_text())
                manifest['reports'][0]['path'] = fifo.name
                path.write_text(json.dumps(manifest))
            child = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True,
                                   text=True, timeout=3, check=False)
            self.assertEqual(child.returncode, 0, child.stderr)
            result = json.loads(child.stdout)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('regular', result['error'].lower())
        from corridorkey.evaluation_dossier import build_dossier
        path = self.manifest(self.pair([(0.1, 0.1)]))
        report_path = self.root / 'pair.json'
        def substitute_after_read(event):
            report_path.rename(self.root / 'original-pair.json')
            mkfifo = getattr(os, 'mkfifo', None)
            assert callable(mkfifo)
            mkfifo(report_path)
        terminal = build_dossier(path, progress=substitute_after_read)
        self.assertEqual(terminal['status'], 'failed')
        self.assertIn('regular', terminal['error'].lower())

    def test_strict_json_mapping_bounds_and_no_model_import(self):
        from corridorkey.evaluation_dossier import build_dossier
        manifest = self.manifest(self.pair([(0.1, 0.1)]))
        data = json.loads(manifest.read_text())
        data['cases'].append(copy.deepcopy(data['cases'][0]))
        manifest.write_text(json.dumps(data))
        self.assertEqual(build_dossier(manifest)['status'], 'failed')
        manifest.write_text('{"schema_version":1,"schema_version":1}')
        self.assertEqual(build_dossier(manifest)['status'], 'failed')
        manifest = self.manifest(self.pair([(0.1, 0.1)]))
        with patch('corridorkey.evaluation_dossier.MAX_INPUT_BYTES', 1):
            self.assertEqual(build_dossier(manifest)['status'], 'failed')
        source = Path(__file__).parents[1] / 'corridorkey/evaluation_dossier.py'
        self.assertNotIn('import torch', source.read_text())


def make_dossier_fixture(root):
    """Persist tiny EXR-derived reports for actual desktop/CLI tests; no model run."""
    with patch.dict(os.environ, {'OPENCV_IO_ENABLE_OPENEXR': '1'}):
        import cv2
        import numpy as np

        from corridorkey.evaluation_media import evaluate_sequence_spec
        from corridorkey.identities import file_sha256
        helper = EvaluationDossierTests()
        helper.root = Path(root)
        helper.root.mkdir(parents=True, exist_ok=True)
        helper.source = helper.root / 'source.bin'
        helper.source.write_bytes(b'generated local analytic fixture; no neural inference')
        helper.reference = helper.root / 'reference.npy'
        np.save(helper.reference, np.full((2, 2), 0.5, np.float32))
        helper.evaluate, helper.hash = evaluate_sequence_spec, file_sha256
        helper.cv2, helper.np = cv2, np
        return helper.manifest(helper.pair([(0.1, 0.09), (0.1, 0.1), (0.1, 0.12)]))
