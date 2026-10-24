"""Manifest/orchestrator contracts; injected outcomes do not establish real stress acceptance."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.benchmark_report import read_report
from corridorkey.control import CancelledError
from corridorkey.stress_suite import SCENARIOS, load_stress_manifest, run_stress_suite


class StressSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source.mov'
        self.source.write_bytes(b'local source fixture')
        self.old = self.root / 'old-output'
        self.old.mkdir()
        (self.old / 'sentinel').write_bytes(b'preserve')
        self.batch = self.root / 'batch.json'
        self.batch_data = {'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
            {'input': 'source.mov', 'output_dir': 'old-output',
             'settings': {'output_format': 'exr', 'outputs': ['alpha']}}]}
        self.batch.write_text(json.dumps(self.batch_data))
        self.manifest = self.root / 'stress.json'
        self.output = self.root / 'report.json'
        self.data = {'schema_version': 1, 'kind': 'production_stress', 'cases': [
            {'id': 'case.' + scenario, 'scenario': scenario, 'batch_manifest': 'batch.json'}
            for scenario in SCENARIOS]}
        provenance = patch('corridorkey.stress_suite._provenance', return_value={'fixture': True})
        provenance.start()
        self.addCleanup(provenance.stop)
        self.calls = []

    def driver(self, case, jobs, weights, root, **kwargs):
        self.calls.append((case, jobs, weights, root, kwargs))
        (root / 'render.exr').write_bytes(b'fixture export')
        return {'status': 'measured', 'success': True, 'reason': None, 'error': None,
                'cleanup': {'verified': True, 'owners': []}, 'scope': 'Injected fixture',
                'coverage': {'jobs': len(jobs)}}

    def run_suite(self, driver=None, **kwargs):
        self.manifest.write_text(json.dumps(self.data))
        return run_stress_suite(self.manifest, self.output,
                                scenario_runner=driver or self.driver, **kwargs)

    def test_complete_inventory_frozen_jobs_evidence_and_originals(self):
        original = {p: p.read_bytes() for p in (self.source, self.batch, self.old / 'sentinel')}
        report = self.run_suite()
        self.assertEqual(report['status'], 'complete', report)
        self.assertTrue(report['summary']['success'])
        self.assertEqual(report['summary']['required_measured_cases'], 9)
        self.assertEqual(read_report(self.output, regular_file=True)[0], report)
        for row in report['cases']:
            evidence = row['evidence']
            raw = Path(evidence['path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), evidence['sha256'])
            self.assertEqual(len(raw), evidence['bytes'])
        plan = json.loads((Path(report['artifact_dir']) / 'frozen-plan.json').read_text())
        self.assertEqual(plan['provenance'], {'fixture': True})
        self.assertEqual(len(plan['requested_jobs']), 9)
        for path, raw in original.items():
            self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(len({str(call[3]) for call in self.calls}), 9)

    def test_missing_scenarios_remain_required_and_unverified(self):
        self.data['cases'] = self.data['cases'][2:3]
        report = self.run_suite()
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(report['summary']['success'])
        self.assertEqual(report['summary']['expected_cases'], 9)
        self.assertEqual(report['summary']['not_verified_cases'], 8)
        self.assertEqual(len(self.calls), 1)
        read_report(self.output)

    def test_missing_batch_is_not_verified_without_driver(self):
        self.batch.unlink()
        report = self.run_suite()
        self.assertEqual(report['summary']['not_verified_cases'], 9)
        self.assertFalse(self.calls)
        self.assertEqual(report['status'], 'complete')

    def test_driver_abort_keeps_denominator_and_cleanup_uncertain(self):
        def abort(*args, **kwargs):
            raise CancelledError('cancel during active scenario')
        report = self.run_suite(abort)
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['summary']['expected_cases'], 9)
        self.assertEqual(report['summary']['cancelled_cases'], 9)
        self.assertFalse(report['summary']['cleanup_verified'])
        read_report(self.output)

    def test_cancel_before_case_has_no_claimed_owner(self):
        def progress(event):
            raise CancelledError('after first terminal case')
        report = self.run_suite(progress=progress)
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['summary']['measured_cases'], 1)
        self.assertEqual(report['summary']['cancelled_cases'], 8)
        self.assertTrue(report['summary']['cleanup_verified'])
        read_report(self.output)

    def test_source_or_manifest_mutation_fails_final_fence(self):
        def mutate(*args, **kwargs):
            result = self.driver(*args, **kwargs)
            self.source.write_bytes(b'changed during execution')
            return result
        report = self.run_suite(mutate)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('changed during execution', report['error'])
        self.assertFalse(report['summary']['success'])
        read_report(self.output)

    def test_cumulative_artifact_limit_stops_next_case(self):
        self.data['limits'] = {'max_artifact_bytes': 120000}
        def large(*args, **kwargs):
            result = self.driver(*args, **kwargs)
            root = args[3]
            (root / 'large').write_bytes(b'x' * kwargs['limits']['max_artifact_bytes'])
            return result
        report = self.run_suite(large)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['summary']['expected_cases'], 9)
        self.assertEqual(report['summary']['not_verified_cases'], 8)
        read_report(self.output)

    def test_protect_declared_output_even_when_another_job_is_missing(self):
        self.batch_data['jobs'].append({'input': 'missing.mov', 'output_dir': 'other'})
        self.batch.write_text(json.dumps(self.batch_data))
        self.output = self.old / 'report.json'
        report = self.run_suite()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('protected', report['error'])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls)

    def test_existing_report_is_never_replaced(self):
        self.output.write_bytes(b'user report')
        report = self.run_suite()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(self.output.read_bytes(), b'user report')
        self.assertFalse(self.calls)

    def test_duplicate_fields_unknown_commands_and_nonfinite_limits_rejected(self):
        self.manifest.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ValueError):
            load_stress_manifest(self.manifest)
        for extra in ({'command': 'bad'}, {'limits': {'deadline_seconds': float('inf')}},
                      {'limits': {'max_artifact_bytes': True}}):
            with self.subTest(extra=extra):
                self.manifest.write_text(json.dumps({**self.data, **extra}))
                with self.assertRaises(ValueError):
                    load_stress_manifest(self.manifest)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO admission')
    def test_fifo_batch_refused_without_open_block(self):
        self.batch.unlink()
        mkfifo = getattr(os, 'mkfifo', None)
        assert callable(mkfifo)
        mkfifo(self.batch)
        report = self.run_suite()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['summary']['failed_cases'], 9)
        self.assertFalse(self.calls)
        read_report(self.output)

    def test_actual_selected_weights_are_used_for_prerequisite_gate(self):
        self.batch_data['weights'] = 'alternate.safetensors'
        self.batch.write_text(json.dumps(self.batch_data))
        self.manifest.write_text(json.dumps(self.data))
        with patch('corridorkey.stress_runtime.production_prerequisites', return_value=['fixture missing target GPU']) as prerequisite:
            report = run_stress_suite(self.manifest, self.output)
        self.assertEqual(prerequisite.call_count, 1)
        self.assertEqual(prerequisite.call_args.kwargs['weights'], self.root / 'alternate.safetensors')
        self.assertEqual(report['summary']['not_verified_cases'], 9)
        self.assertEqual(report['status'], 'complete')
        read_report(self.output)

    def test_default_model_asset_tree_is_protected(self):
        assets = self.root / 'default-assets'
        assets.mkdir()
        self.output = assets / 'report.json'
        with patch('corridorkey.stress_suite._asset_paths', return_value=[assets]):
            report = self.run_suite()
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls)

    def test_production_model_hashes_and_final_mutation_fence(self):
        assets = self.root / 'assets'
        assets.mkdir()
        model = assets / 'model.bin'
        model.write_bytes(b'model before')
        self.manifest.write_text(json.dumps(self.data))
        def mutate(*args, **kwargs):
            result = self.driver(*args, **kwargs)
            model.write_bytes(b'model changed')
            return result
        with patch('corridorkey.stress_suite._asset_paths', return_value=[assets]), \
                patch('corridorkey.stress_runtime.production_prerequisites', return_value=[]), \
                patch('corridorkey.stress_runtime.execute_scenario', side_effect=mutate):
            report = run_stress_suite(self.manifest, self.output)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('changed during execution', report['error'])
        plan = json.loads((Path(report['artifact_dir']) / 'frozen-plan.json').read_text())
        self.assertEqual(plan['model_assets']['files'][0]['sha256'],
                         hashlib.sha256(b'model before').hexdigest())

    def test_model_directory_added_member_fails_final_fence(self):
        assets = self.root / 'assets'
        assets.mkdir()
        (assets / 'model.bin').write_bytes(b'model before')
        self.manifest.write_text(json.dumps(self.data))
        def add(*args, **kwargs):
            result = self.driver(*args, **kwargs)
            (assets / 'preferred-config.json').write_text('{}')
            return result
        with patch('corridorkey.stress_suite._asset_paths', return_value=[assets]), \
                patch('corridorkey.stress_runtime.production_prerequisites', return_value=[]), \
                patch('corridorkey.stress_runtime.execute_scenario', side_effect=add):
            report = run_stress_suite(self.manifest, self.output)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('membership changed', report['error'])
        self.assertEqual(report['summary']['expected_cases'], 9)
        self.assertFalse(report['summary']['success'])


if __name__ == '__main__':
    unittest.main()
