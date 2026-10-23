"""Benchmark orchestration fixtures, not actual inference/performance evidence."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from corridorkey import benchmark

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2'))


@unittest.skipUnless(STACK, 'requires optional CPU runtime imports')
class BenchmarkSafetyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'synthetic source; no inference')
        self.weights = self.root / 'weights.bin'
        self.weights.write_bytes(b'no model loaded')
        self.cradio = self.root / 'cradio'
        self.cradio.mkdir()
        self.batch = self.root / 'batch.json'
        self.batch_data = {'schema_version': 1, 'kind': 'inference_batch', 'jobs': [{
            'input': self.source.name, 'output_dir': 'user-output',
            'settings': {'output_format': 'exr', 'outputs': ['alpha']},
            'writer': {'crf': 12}, 'profile_stages': False}]}
        self.batch.write_text(json.dumps(self.batch_data))
        self.manifest = self.root / 'benchmark.json'
        self.manifest.write_text(json.dumps({
            'schema_version': 1, 'kind': 'inference', 'batch_manifest': self.batch.name,
            'model_state': 'cold_process', 'repetitions': 3, 'quality_manifests': [None]}))
        self.children = []

    def child(self, command, **kwargs):
        from corridorkey.batch import load_jobs

        manifest = Path(command[3])
        jobs, weights = load_jobs(manifest)
        self.children.append((manifest, jobs, weights))
        report = Path(command[command.index('--report') + 1])
        report.write_text(json.dumps({'jobs': [{'status': 'complete', 'diagnostics': {}} for _ in jobs]}))
        return subprocess.CompletedProcess(command, 0, 'synthetic child', '')

    def run_fixture(self, *, progress=None, quality_error=None):
        paths = SimpleNamespace(weights=self.weights, foundation_stats=self.weights,
                                rvm_ckpt=self.weights, moge_checkpoint=self.weights,
                                cradio_dir=self.cradio, validate=lambda: None)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(benchmark, 'ROOT', self.root))
            stack.enter_context(patch.object(benchmark, 'repository_identity', return_value={'fixture': True}))
            stack.enter_context(patch.object(benchmark, 'environment', return_value={'hardware': {}}))
            stack.enter_context(patch.object(benchmark, 'command', side_effect=RuntimeError('no test GPU')))
            stack.enter_context(patch('corridorkey.runtime.RuntimePaths.from_package', return_value=paths))
            stack.enter_context(patch('corridorkey.benchmark_quality.prepare_quality_templates', return_value=[None]))
            stack.enter_context(patch('corridorkey.benchmark_quality.evaluate_run_quality',
                                      side_effect=quality_error, return_value={'success': True}))
            stack.enter_context(patch.object(benchmark.subprocess, 'run', side_effect=self.child))
            return benchmark.run(self.manifest, progress=progress)

    def test_late_quality_failure_and_cancellation_cannot_report_success(self):
        for error, status, exit_code in [(RuntimeError('late quality failure'), 'unavailable_or_failed', 1),
                                        (KeyboardInterrupt(), 'cancelled', 130)]:
            with self.subTest(error=type(error).__name__):
                report = self.run_fixture(quality_error=error)
                self.assertFalse(report['summary']['success'])
                self.assertEqual(report['inference']['status'], status)
                self.assertEqual(report['error']['type'], type(error).__name__)
                self.assertEqual(len(report['runs']), 3)
                self.assertIsNotNone(report['summary']['median_seconds'])
                output = self.root / 'report.json'
                with patch.object(benchmark, 'run', return_value=report), \
                        patch('sys.argv', ['benchmark', '--output', str(output)]), \
                        contextlib.redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(benchmark.main(), exit_code)
                self.assertIn('failed', stdout.getvalue())
                self.assertFalse(json.loads(output.read_text())['summary']['success'])

    def test_cold_jobs_and_provenance_are_frozen_before_original_batch_mutation(self):
        original_hash = hashlib.sha256(self.batch.read_bytes()).hexdigest()
        original_output = self.root / 'user-output'
        original_output.mkdir()
        sentinel = original_output / 'keep.txt'
        sentinel.write_bytes(b'user output must remain untouched')

        def mutate(stage, completed, total):
            self.assertEqual(stage, 'cold_repetition')
            changed = copy.deepcopy(self.batch_data)
            changed['jobs'][0]['writer']['crf'] = 50
            changed['jobs'][0]['profile_stages'] = True
            changed['jobs'].append({'input': str(self.source), 'output_dir': str(original_output)})
            self.batch.write_text(json.dumps(changed))

        report = self.run_fixture(progress=mutate)
        self.assertTrue(report['summary']['success'], report.get('error'))
        self.assertEqual(report['batch_manifest_sha256'], original_hash)
        self.assertEqual(len(self.children), 3)
        for index, (manifest, jobs, weights) in enumerate(self.children):
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].input_path, str(self.source))
            self.assertEqual(jobs[0].output_dir, str(manifest.parent / str(index) / '0'))
            self.assertFalse(jobs[0].profile_stages)
            self.assertEqual(jobs[0].configurations()[1].crf, 12)
            self.assertEqual(weights, self.weights)
        self.assertEqual(sentinel.read_bytes(), b'user output must remain untouched')
