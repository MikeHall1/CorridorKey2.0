"""Readiness observations must preserve inputs and never imply inference."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.asset_inventory import AssetSelection
from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.control import CancellationToken
from corridorkey.readiness import check_readiness, export_report, prepare_short_render
from corridorkey.session import JobRequest


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'frame.png').write_bytes(b'local fixture, not decoded')
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        self.output = self.root / 'output'
        self.job = JobRequest.create(self.source, self.output, InferenceConfig(output_format='exr'))
        self.selection = AssetSelection(self.assets)

    def stages(self, report):
        return {item['stage']: item for item in report['stages']}

    def test_read_only_stages_and_no_implicit_expensive_operations(self):
        with patch('corridorkey.session.InferenceSession.run', side_effect=AssertionError('No inference')):
            report = check_readiness(self.job, self.selection)
        stages = self.stages(report)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(stages['assets']['status'], 'FAIL')
        self.assertEqual(stages['source']['status'], 'PASS')
        self.assertEqual(stages['destination']['status'], 'PASS')
        self.assertEqual(stages['ffmpeg']['status'], 'NOT REQUIRED')
        for name in ('backend', 'model_load', 'inference', 'output_decode', 'display'):
            self.assertEqual(stages[name]['status'], 'NOT VERIFIED')
        self.assertIsNone(report['effective_settings'])
        self.assertFalse(self.output.exists())
        self.assertEqual((self.source / 'frame.png').read_bytes(), b'local fixture, not decoded')

    def test_source_free_check_and_pre_cancelled_report(self):
        report = check_readiness(None, self.selection)
        self.assertEqual(self.stages(report)['source']['status'], 'NOT SELECTED')
        token = CancellationToken()
        token.cancel('user stop')
        report = check_readiness(self.job, self.selection, cancellation=token)
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['stages'], [])

    def test_cancel_during_inventory_returns_interrupted_stage(self):
        token = CancellationToken()

        def stop(*args, **kwargs):
            token.cancel()
            token.check()

        with patch('corridorkey.readiness.validate_output_paths', side_effect=stop):
            report = check_readiness(self.job, self.selection, cancellation=token)
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(self.stages(report)['destination']['status'], 'CANCELLED')
        self.assertNotIn('resources', self.stages(report))

    def test_numpy2_is_not_compatible_and_native_imports_not_claimed(self):
        from importlib.metadata import version

        def installed(name):
            return '2.0.0' if name == 'numpy' else version(name)

        with patch('corridorkey.readiness.importlib.metadata.version', side_effect=installed):
            report = check_readiness(None, self.selection)
        dependencies = self.stages(report)['dependencies']
        self.assertEqual(dependencies['status'], 'FAIL')
        numpy = next(row for row in dependencies['details']['packages'] if row['package'] == 'numpy')
        self.assertFalse(numpy['compatible_metadata'])

    def test_destination_protection_and_existing_export_warning(self):
        bad = JobRequest.create(self.source, self.source / 'nested', InferenceConfig(output_format='exr'))
        self.assertEqual(self.stages(check_readiness(bad, self.selection))['destination']['status'], 'FAIL')
        self.output.mkdir()
        (self.output / 'source_alpha_exr').mkdir()
        report = check_readiness(self.job, self.selection)
        self.assertEqual(self.stages(report)['destination']['status'], 'WARNING')

    def test_report_exclusive_and_protected_from_aliases(self):
        report = check_readiness(self.job, self.selection)
        destination = self.root / 'report.json'
        export_report(report, destination)
        self.assertEqual(json.loads(destination.read_bytes()), report)
        with self.assertRaises(FileExistsError):
            export_report(report, destination)
        for protected in (self.source, self.assets):
            with self.assertRaises(ValueError):
                export_report(report, protected / 'report.json')
        alias = self.root / 'alias'
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            export_report(report, alias / 'report.json')
        self.assertFalse((self.source / 'report.json').exists())

    def test_explicit_subrange_preserves_requested_quality_and_writer(self):
        cfg = InferenceConfig(output_format='exr', exr_dtype='float', model_scale=0.5,
                              quick_key=True, start_frame=10, quick_cut_frames=[0, 5, 19],
                              quick_reset_frames=[2, 8], stage2_batch=3)
        original = JobRequest.create(self.source, self.output, cfg, WriterConfig(crf=9))
        prepared = prepare_short_render(original, 14, 8)
        actual, writer = prepared.configurations()
        self.assertEqual(actual.quick_cut_frames, [1])
        self.assertEqual(actual.quick_reset_frames, [4])
        self.assertEqual((actual.start_frame, actual.num_frames), (14, 8))
        self.assertEqual((actual.model_scale, actual.exr_dtype, actual.stage2_batch), (0.5, 'float', 3))
        self.assertEqual(writer.crf, 9)
        self.assertEqual(original.configurations()[0].start_frame, 10)
        self.assertEqual(Path(prepared.output_dir).parent, self.output.resolve())
        self.assertFalse(Path(prepared.output_dir).exists())
        self.assertNotEqual(prepared.output_dir, prepare_short_render(original, 14, 8).output_dir)

    def test_strict_limits_and_hint_correction_semantics(self):
        for start, count in ((True, 1), (-1, 2), (0, True), (0, 0), (0, 121)):
            with self.assertRaises(ValueError):
                prepare_short_render(self.job, start, count)
        hinted = JobRequest.create(self.source, self.output, initial_hint_path=self.source / 'frame.png')
        with self.assertRaisesRegex(ValueError, 'hint'):
            prepare_short_render(hinted, 1, 1)
        self.assertEqual(prepare_short_render(hinted, 0, 1).initial_hint_path, hinted.initial_hint_path)
        corrected = JobRequest.create(self.source, self.output,
                                      InferenceConfig(corrections_path=self.root / 'corrections.json'))
        with self.assertRaisesRegex(ValueError, 'rebasing'):
            prepare_short_render(corrected, 0, 1)


if __name__ == '__main__':
    unittest.main()
