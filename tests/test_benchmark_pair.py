import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey import benchmark_pair as pair


class PairTests(unittest.TestCase):
    def setup_files(self, root, state='cold_process'):
        source = root / 'source'
        source.mkdir()
        (source / 'frame.png').write_bytes(b'fixture only; not decoded')
        for side, budget in (('baseline', 0), ('candidate', 64)):
            batch = {'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
                {'input': str(source), 'output_dir': str(root / side),
                 'settings': {'foundation_cache_mb': budget}, 'writer': {}}]}
            (root / f'{side}-batch.json').write_text(json.dumps(batch))
            (root / f'{side}.json').write_text(json.dumps({
                'schema_version': 1, 'kind': 'inference', 'repetitions': 3,
                'model_state': state, 'batch_manifest': f'{side}-batch.json'}))
        manifest = root / 'pair.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_pair',
                                       'baseline': 'baseline.json', 'candidate': 'candidate.json',
                                       'changed_settings': ['foundation_cache_mb'], 'repetitions': 3}))
        return manifest

    def sample(self, path, manifest):
        batch = json.loads(Path(manifest['batch_manifest']).read_text())
        settings = batch['jobs'][0]['settings']
        return {'summary': {'success': True}, 'inference': {'status': 'measured', 'quality': None},
                'runs': [{'elapsed_seconds': 2 if path.stem == 'baseline' else 1,
                          'jobs': [{'status': 'complete', 'requested_settings': settings,
                                    'diagnostics': {'effective_settings': copy.deepcopy(settings)}}]}], 'weights': {},
                'repository': {'commit': 'fixture'}, 'environment': {'hardware': {'cpu': 'fixture'}}}

    def test_alternates_actual_samples_and_retains_raw_quality(self):
        for state in ('cold_process', 'warm_session'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = self.setup_files(root, state)
                order = []

                def measure(path, value, order=order, state=state):
                    order.append(path.stem)
                    self.assertEqual(value['repetitions'], 1)
                    self.assertEqual(value['model_state'], state)
                    return self.sample(path, value)

                with patch.object(pair.benchmark, 'run_inference_benchmark', side_effect=measure):
                    result = pair.run(manifest)
                self.assertEqual(result['status'], 'complete', result.get('error'))
                self.assertEqual(order, ['baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate'])
                self.assertEqual(result['summary']['speedup'], 2)
                self.assertEqual(result['summary']['baseline']['raw_seconds'], [2, 2, 2])
                self.assertIsNone(result['pairs'][0]['samples']['candidate']['inference']['quality'])

    def test_partial_failed_cancelled_malformed_and_drift_never_speedup(self):
        for defect in ('failure', 'cancel', 'nan', 'missing', 'drift', 'mutate'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = self.setup_files(root)
                calls = 0

                def measure(path, value, defect=defect, root=root):
                    nonlocal calls
                    calls += 1
                    result = self.sample(path, value)
                    if calls == 2:
                        if defect == 'failure':
                            result['summary']['success'] = False
                        elif defect == 'cancel':
                            result['inference']['status'] = 'cancelled'
                        elif defect == 'nan':
                            result['runs'][0]['elapsed_seconds'] = float('nan')
                        elif defect == 'missing':
                            result['runs'][0]['jobs'] = []
                        elif defect == 'drift':
                            result['repository']['commit'] = 'changed'
                        else:
                            (root / 'source' / 'frame.png').write_bytes(b'changed')
                    return result

                with patch.object(pair.benchmark, 'run_inference_benchmark', side_effect=measure):
                    result = pair.run(manifest)
                self.assertEqual(result['status'], 'cancelled' if defect == 'cancel' else 'failed')
                self.assertFalse(result['summary']['success'])
                self.assertIsNone(result['summary']['speedup'])
                self.assertEqual(calls, 2)
                self.assertEqual(len(result['pairs'][0]['samples']), 2)

    def test_rejects_unmatched_settings_and_short_protocol_before_execution(self):
        for defect in ('repeats', 'workload', 'undeclared', 'same', 'state'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = self.setup_files(root)
                data = json.loads(manifest.read_text())
                if defect == 'repeats':
                    data['repetitions'] = True
                elif defect == 'workload':
                    data['changed_settings'] = ['outputs']
                else:
                    path = root / ('candidate.json' if defect == 'state' else 'candidate-batch.json')
                    candidate = json.loads(path.read_text())
                    if defect == 'state':
                        candidate['model_state'] = 'warm_session'
                    else:
                        candidate['jobs'][0]['settings']['num_frames' if defect == 'undeclared' else 'foundation_cache_mb'] = 1 if defect == 'undeclared' else 0
                    path.write_text(json.dumps(candidate))
                manifest.write_text(json.dumps(data))
                with patch.object(pair.benchmark, 'run_inference_benchmark') as runner:
                    result = pair.run(manifest)
                self.assertEqual(result['status'], 'failed')
                runner.assert_not_called()

    def test_cli_preserves_existing_destination_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.setup_files(root)
            output = root / 'report.json'
            output.write_text('keep')
            with self.assertRaises(FileExistsError):
                pair.main(['--manifest', str(manifest), '--output', str(output)])
            self.assertEqual(output.read_text(), 'keep')
            output.unlink()
            with patch.object(pair.benchmark, 'run_inference_benchmark', side_effect=RuntimeError('fixture error')):
                self.assertEqual(pair.main(['--manifest', str(manifest), '--output', str(output)]), 1)
            self.assertEqual(json.loads(output.read_text())['error']['message'], 'fixture error')

    def test_manifest_snapshot_survives_original_settings_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.setup_files(root)
            original = (root / 'source' / 'frame.png').read_bytes()
            budgets = []

            def measure(path, value):
                batch = json.loads(Path(value['batch_manifest']).read_text())
                budgets.append(batch['jobs'][0]['settings']['foundation_cache_mb'])
                (root / 'candidate-batch.json').write_text('{}')
                return copy.deepcopy(self.sample(path, value))

            with patch.object(pair.benchmark, 'run_inference_benchmark', side_effect=measure):
                result = pair.run(manifest)
            self.assertEqual(result['status'], 'complete', result.get('error'))
            self.assertEqual(budgets, [0, 64, 64, 0, 0, 64])
            self.assertEqual((root / 'source' / 'frame.png').read_bytes(), original)

    def test_real_cli_child_rejects_invalid_protocol_with_terminal_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'invalid.json'
            manifest.write_text('{"schema_version":1,"kind":"inference_pair","repetitions":2}')
            output = root / 'result.json'
            result = subprocess.run([sys.executable, '-m', 'corridorkey.benchmark_pair',
                                     '--manifest', str(manifest), '--output', str(output)],
                                    capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 1, result.stderr)
            evidence = json.loads(output.read_text())
            self.assertEqual(evidence['status'], 'failed')
            self.assertIsNone(evidence['summary']['speedup'])
            self.assertEqual(evidence['pairs'], [])

    def test_automatic_selection_refused_before_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.setup_files(root)
            for side in ('baseline', 'candidate'):
                path = root / f'{side}-batch.json'
                data = json.loads(path.read_text())
                data['jobs'][0]['settings']['auto_vram'] = True
                path.write_text(json.dumps(data))
            with patch.object(pair.benchmark, 'run_inference_benchmark') as runner:
                evidence = pair.run(manifest)
            self.assertEqual(evidence['status'], 'failed')
            self.assertIn('auto_vram=false', evidence['error']['message'])
            self.assertIsNone(evidence['summary']['speedup'])
            runner.assert_not_called()

    def test_effective_settings_must_match_workload_and_remain_stable(self):
        for defect in ('cross_side', 'same_side', 'missing', 'requested'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest = self.setup_files(root)
                calls = 0

                def measure(path, value, defect=defect):
                    nonlocal calls
                    calls += 1
                    raw = self.sample(path, value)
                    job = raw['runs'][0]['jobs'][0]
                    if (defect == 'cross_side' and calls == 2) or (defect == 'same_side' and calls == 3):
                        settings = job['diagnostics']['effective_settings']
                        # Simulates actual automatic/runtime adjustment despite matched requested settings.
                        if defect == 'same_side':
                            # Even the explicitly varied field must remain stable within one side.
                            settings['foundation_cache_mb'] += 1
                        else:
                            settings.update(hann_chunk=20, hann_stride=10, low_vram=True)
                    elif defect == 'missing' and calls == 2:
                        job['diagnostics']['effective_settings'] = {}
                    elif defect == 'requested' and calls == 2:
                        job['requested_settings'] = dict(job['requested_settings'], hann_chunk=20)
                    return raw

                with patch.object(pair.benchmark, 'run_inference_benchmark', side_effect=measure):
                    evidence = pair.run(manifest)
                self.assertEqual(evidence['status'], 'failed')
                self.assertIsNone(evidence['summary']['speedup'])
                self.assertEqual(calls, 3 if defect == 'same_side' else 2)
