"""Real owned benchmark child and strict report flow; no neural performance claim."""
import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'cv2'))


def manifest(parent):
    return {'schema_version': 1, 'kind': 'decode_reuse', 'artifact_parent': str(parent),
            'repetitions': 3, 'cache_bytes': 16384, 'frame_count': 8, 'shape_hw': [16, 16],
            'chunk': 6, 'stride': 3, 'quick_interval': 3}


class DecodeReuseManifestTests(unittest.TestCase):
    def test_explicit_research_manifest_and_bounds_without_loading_a_model(self):
        from corridorkey.benchmark import load_manifest
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            original = manifest(folder)
            path.write_text(json.dumps(original))
            self.assertEqual(load_manifest(path), original)
            for update in ({'repetitions': 2}, {'repetitions': 21}, {'cache_bytes': True},
                           {'shape_hw': [512, 512], 'frame_count': 128},
                           {'arbitrary_media': '/tmp/source.mov'}, {'artifact_parent': ''}):
                with self.subTest(update=update):
                    path.write_text(json.dumps({**original, **update}))
                    with self.assertRaises((ValueError, TypeError)):
                        load_manifest(path)

    def test_changed_manifest_refuses_before_artifact_creation_or_provenance_work(self):
        from corridorkey.benchmark import run_decode_reuse_benchmark
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'manifest.json'
            original = manifest(root)
            path.write_text(json.dumps({**original, 'cache_bytes': 0}))
            with patch('corridorkey.benchmark.repository_identity') as repository, \
                 patch('corridorkey.benchmark.environment') as environment, \
                 self.assertRaisesRegex(ValueError, 'manifest changed'):
                run_decode_reuse_benchmark(path, original)
            repository.assert_not_called()
            environment.assert_not_called()
            self.assertEqual({p.name for p in root.iterdir()}, {'manifest.json'})


@unittest.skipUnless(STACK, 'Requires actual local CPU decode/preprocessing stack')
class DecodeReuseRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from corridorkey.benchmark_report import read_report
        from corridorkey.benchmark_task import BenchmarkTask
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.root = Path(cls.folder.name).resolve()
        cls.manifest_path = cls.root / 'manifest.json'
        cls.manifest_path.write_text(json.dumps(manifest(cls.root)))
        cls.original_manifest = cls.manifest_path.read_bytes()
        cls.task = BenchmarkTask(cls.manifest_path, cls.root)
        if not cls.task.done.wait(45):
            cls.task.cancel()
            cls.task.thread.join(10)
            raise AssertionError('Owned decode investigation did not terminate')
        cls.task.thread.join(10)
        if cls.task.snapshot()['status'] != 'complete':
            raise AssertionError(cls.task.snapshot())
        cls.report, cls.raw = read_report(cls.task.report_path)

    def test_actual_shared_cli_report_is_paired_cpu_only_and_owner_is_terminal(self):
        report = self.report
        self.assertEqual(report['kind'], 'decode_reuse')
        self.assertEqual(report['status'], 'complete')
        self.assertTrue(report['summary']['success'])
        self.assertEqual(report['inference'], {'status': 'not_measured', 'fps': None, 'quality': None})
        self.assertEqual(report['manifest'], manifest(self.root))
        self.assertRegex(report['manifest_sha256'], '^[0-9a-f]{64}$')
        self.assertIn('repository', report)
        self.assertIn('environment', report)
        self.assertEqual(len(report['results']), 9)
        for result in report['results']:
            self.assertEqual(len(result['pairs']), 3)
            self.assertTrue(result['exact_equal'])
            for index, pair in enumerate(result['pairs']):
                self.assertEqual(pair['index'], index)
                self.assertEqual(pair['order'], ['reference', 'candidate'] if index % 2 == 0 else ['candidate', 'reference'])
                self.assertTrue(pair['validation']['exact_equal'])
                self.assertLessEqual(pair['candidate']['peak_retained_bytes'], report['settings']['cache_bytes'])
        self.assertEqual(self.manifest_path.read_bytes(), self.original_manifest)
        self.assertFalse(self.task.thread.is_alive())
        self.assertIsNotNone(self.task.process)
        assert self.task.process is not None
        self.assertIsNotNone(self.task.process.poll())
        if self.task.reader is not None:
            self.assertFalse(self.task.reader.is_alive())

    def test_inspection_and_export_preserve_exact_report_bytes(self):
        from corridorkey.benchmark_report import export_report, read_report, report_view
        target = self.root / 'export.json'
        export_report(self.task.report_path, target)
        self.assertEqual(target.read_bytes(), self.raw)
        self.assertEqual(read_report(target)[0], self.report)
        text = report_view(self.report)
        self.assertIn('decode_reuse', text)
        self.assertIn('not_measured', text)
        with self.assertRaises(FileExistsError):
            export_report(self.task.report_path, target)

    def test_report_inspection_rejects_fabricated_success_parity_and_budget(self):
        from corridorkey.benchmark_report import read_report
        path = self.root / 'invalid.json'
        for defect in ('fps', 'short_pairs', 'order', 'parity', 'budget'):
            data = copy.deepcopy(self.report)
            if defect == 'fps':
                data['inference']['fps'] = 1000
            elif defect == 'short_pairs':
                data['results'][0]['pairs'].pop()
            elif defect == 'order':
                data['results'][0]['pairs'][1]['order'].reverse()
            elif defect == 'parity':
                data['results'][0]['pairs'][0]['validation']['exact_equal'] = False
            else:
                data['results'][0]['pairs'][0]['candidate']['peak_retained_bytes'] = data['settings']['cache_bytes'] + 1
            with self.subTest(defect=defect):
                path.write_text(json.dumps(data))
                with self.assertRaises((ValueError, TypeError)):
                    read_report(path)
