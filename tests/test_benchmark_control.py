import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.benchmark import load_manifest, run


class BenchmarkControlTests(unittest.TestCase):
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
