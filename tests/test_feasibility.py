"""Portable projection tests use existing report schema, not hardware measurements."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from corridorkey import feasibility


def inference_fixture():
    return {'schema_version': 1, 'kind': 'inference', 'summary': {'success': True},
            'model_state': 'warm_session', 'inference': {'status': 'measured'},
            'cache': {'filesystem': 'uncontrolled'}, 'runs': [{'index': 0, 'elapsed_seconds': 2,
            'jobs': [{'status': 'complete', 'processing_seconds': 1.5, 'model_load_seconds': 0,
            'diagnostics': {'frames_submitted': 48, 'first_rendered_batch_seconds': 0.5,
                            'profiling_enabled': True, 'stages': [
                                {'name': 'stage1_stream_including_stage2_render_submit', 'seconds': 1.4, 'status': 'complete', 'method': 'synchronized_wall'},
                                {'name': 'stage2_inference', 'seconds': 0.8, 'status': 'complete', 'method': 'synchronized_wall'},
                                {'name': 'render', 'seconds': 0.2, 'status': 'complete', 'method': 'host_wall'}],
                            'resources': {'peak_allocated_vram_bytes': 1024, 'scope': 'processing after model load'}}}]}]}


class FeasibilityTests(unittest.TestCase):
    def test_actual_report_schema_bound_identity_target_and_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'inference.json'
            raw = json.dumps(inference_fixture()).encode()
            path.write_bytes(raw)
            result = feasibility.inspect(path, 24)
        self.assertEqual(result['frame_budget_seconds'], 1 / 24)
        self.assertEqual(result['source']['sha256'], hashlib.sha256(raw).hexdigest())
        sample = result['samples'][0]
        self.assertEqual(sample['offline_fps'], 24)
        self.assertEqual(sample['offline_target_comparison'], 'at_or_above_target')
        self.assertEqual(sample['jobs'][0]['processing_fps'], 32)
        self.assertEqual(sample['jobs'][0]['first_rendered_batch_seconds'], 0.5)
        self.assertEqual(len(sample['jobs'][0]['stages']), 3)
        self.assertEqual(sample['jobs'][0]['stage_lanes']['decode'], 'NOT VERIFIED')
        self.assertEqual(result['sustained_presentation'], 'NOT VERIFIED')
        self.assertEqual(result['instrumentation_overhead'], 'NOT VERIFIED')
        self.assertNotIn('stage_total_seconds', feasibility.view(result))

    def test_missing_failed_cpu_never_inference_claim_and_cold_scope(self):
        report = inference_fixture()
        report['model_state'] = 'cold_process'
        self.assertIn('interpreter', feasibility.project(report)['samples'][0]['timing_scope'])
        report['runs'][0]['jobs'][0]['diagnostics'] = {}
        sample = feasibility.project(report)['samples'][0]
        self.assertIsNone(sample['offline_fps'])
        self.assertEqual(sample['status'], 'NOT VERIFIED')
        self.assertEqual(sample['jobs'][0]['stage_lanes']['transfer'], 'NOT VERIFIED')
        report['summary']['success'] = False
        self.assertEqual(feasibility.project(report)['status'], 'NOT VERIFIED')
        cpu = {'kind': 'cpu_correctness', 'summary': {'success': True}, 'runs': [{'elapsed_seconds': 0.001}]}
        self.assertEqual(feasibility.project(cpu)['samples'], [])

    def test_pairs_separate_repetitions_lifecycle_and_no_speedup_synthesis(self):
        from tests.test_benchmark_report import PairedReportTests
        report = PairedReportTests().report(True)
        for pair in report['pairs']:
            pair['samples'] = {'baseline': inference_fixture(), 'candidate': copy.deepcopy(inference_fixture())}
            pair['samples']['candidate']['model_state'] = 'cold_process'
        projection = feasibility.project(report, 30)
        self.assertEqual(len(projection['samples']), 6)
        self.assertEqual(projection['samples'][1]['model_state'], 'cold_process')
        self.assertEqual(projection['samples'][0]['offline_target_comparison'], 'below_target')
        self.assertNotIn('speedup', projection)

    def test_invalid_targets_and_bounded_nested_observations(self):
        for target in (0, -1, True, float('nan'), float('inf'), 1001, 1e-320):
            with self.subTest(target=target), self.assertRaises(ValueError):
                feasibility.project(inference_fixture(), target)
        report = inference_fixture()
        report['runs'] *= 257
        with self.assertRaises(ValueError):
            feasibility.project(report)
        report = inference_fixture()
        report['runs'][0]['jobs'][0]['diagnostics']['stages'][0]['seconds'] = -1
        with self.assertRaises(ValueError):
            feasibility.project(report)

    def test_unrepresentable_rates_are_unavailable_not_infinite(self):
        report = inference_fixture()
        report['runs'][0]['elapsed_seconds'] = 1e-308
        report['runs'][0]['jobs'][0]['diagnostics']['frames_submitted'] = 10 ** 308
        projection = feasibility.project(report)
        self.assertIsNone(projection['samples'][0]['offline_fps'])
        json.dumps(projection, allow_nan=False)

    def test_failed_pair_rejected_raw_sample_has_no_invented_fps(self):
        from tests.test_benchmark_report import PairedReportTests
        report = PairedReportTests().report(False)
        report['status'] = 'failed'
        report['pairs'][0]['samples']['baseline'] = {'rejected_report_repr': 'nonfinite raw report'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'failed.json'
            path.write_text(json.dumps(report))
            projection = feasibility.inspect(path)
        self.assertEqual(projection['status'], 'NOT VERIFIED')
        self.assertIsNone(projection['samples'][0]['offline_fps'])
        self.assertIn('failed/rejected', projection['samples'][0]['reason'])

    def test_malformed_diagnostic_observations_are_not_recorded(self):
        for key, value in (('resources', False), ('resource_policy', 0), ('output_telemetry', 'broken'),
                           ('resources', {'nonsense': 2}), ('resource_policy', {'samples': 'many'}),
                           ('output_telemetry', {'status': 'complete', 'native_completion': 'yes'})):
            with self.subTest(key=key, value=value):
                report = inference_fixture()
                report['runs'][0]['jobs'][0]['diagnostics'][key] = value
                observation = feasibility.project(report)['samples'][0]['jobs'][0][key]
                self.assertEqual(observation['status'], 'NOT VERIFIED')
                self.assertIn('reason', observation)

    def test_partial_output_telemetry_retains_scope_and_no_delivery_claim(self):
        report = inference_fixture()
        telemetry = {'status': 'partial', 'native_completion': False,
                     'queue_accept_latency': {'status': 'complete', 'p95_seconds': 0.02},
                     'native_write_latency': {'status': 'unavailable', 'reason': 'no per-frame completion'}}
        report['runs'][0]['jobs'][0]['diagnostics']['output_telemetry'] = telemetry
        projection = feasibility.project(report)
        job = projection['samples'][0]['jobs'][0]
        self.assertEqual(job['output_telemetry']['value'], telemetry)
        self.assertEqual(job['stage_lanes']['delivery'], 'NOT VERIFIED')
        self.assertIn('not fsync or playback', job['output_telemetry']['scope'])


if __name__ == '__main__':
    unittest.main()
