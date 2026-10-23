import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.benchmark import load_manifest, run


class BenchmarkControlTests(unittest.TestCase):
    def test_cpu_timeout_default_and_explicit_value_reach_each_child_and_report(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'suite.json'
            for requested, effective in [(None, 600), (240.5, 240.5)]:
                manifest = {'schema_version': 1, 'kind': 'cpu_correctness', 'repetitions': 3}
                if requested is not None:
                    manifest['timeout_seconds'] = requested
                path.write_text(json.dumps(manifest))
                with self.subTest(requested=requested), \
                        patch('corridorkey.benchmark.repository_identity', return_value={}), \
                        patch('corridorkey.benchmark.environment', return_value={}), \
                        patch('corridorkey.benchmark.subprocess.run',
                              return_value=subprocess.CompletedProcess([], 0, 'fixture', '')) as child:
                    report = run(path)
                self.assertTrue(report['summary']['success'])
                self.assertEqual(child.call_count, 3)
                self.assertTrue(all(call.kwargs['timeout'] == effective for call in child.call_args_list))
                self.assertEqual(report['cpu_timeout']['requested_seconds'], requested)
                self.assertEqual(report['cpu_timeout']['effective_seconds'], effective)

    def test_cpu_timeout_requires_bounded_positive_finite_number(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'suite.json'
            for value in (None, True, 0, -1, '600', float('nan'), float('inf'), 3600.1, 10**100):
                path.write_text(json.dumps({'schema_version': 1, 'kind': 'cpu_correctness',
                                            'timeout_seconds': value}))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_manifest(path)

    def test_strict_manifest_rejects_ambiguous_or_unknown_fields(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'suite.json'
            for raw in ['{"schema_version":true,"kind":"cpu_correctness"}',
                        '{"schema_version":1,"schema_version":1,"kind":"cpu_correctness"}',
                        '{"schema_version":1,"kind":"cpu_correctness","description":NaN}',
                        '{"schema_version":1,"kind":"cpu_correctness","description":1e309}',
                        '{"schema_version":1,"kind":"cpu_correctness","invented":1}']:
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_cancelled_and_timed_out_cpu_runs_keep_only_real_partial_records(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'suite.json'
            path.write_text(json.dumps({'schema_version': 1, 'kind': 'cpu_correctness', 'repetitions': 3}))
            for error, status in [(KeyboardInterrupt(), 'cancelled'), (subprocess.TimeoutExpired('test', 120), 'failed')]:
                with patch('corridorkey.benchmark.repository_identity', return_value={}), \
                     patch('corridorkey.benchmark.environment', return_value={}), \
                     patch('corridorkey.benchmark.subprocess.run', side_effect=[subprocess.CompletedProcess([], 0, 'actual', ''), error]):
                    report = run(path)
                self.assertEqual(report['status'], status)
                self.assertFalse(report['summary']['success'])
                self.assertIsNone(report['summary']['median_seconds'])
                self.assertEqual(len(report['runs']), 1)
                self.assertEqual(report['runs'][0]['stdout'], 'actual')
                self.assertIsNone(report['inference']['fps'])

    def test_raw_oversized_manifest_cannot_hide_in_trailing_newlines(self):
        from corridorkey.worker_protocol import MAX_FRAME_BYTES
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'suite.json'
            path.write_bytes(b'{"schema_version":1,"kind":"cpu_correctness"}' + b'\n' * MAX_FRAME_BYTES)
            with self.assertRaisesRegex(ValueError, 'size limit'):
                load_manifest(path)
