"""Reports must never replace protected media, manifests, assets or outputs."""
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from corridorkey.batch import run_batch
from corridorkey.config import InferenceConfig
from corridorkey.output_paths import output_layout, validate_report_destination
from corridorkey.session import InferenceSession, JobRequest
from corridorkey.worker_client import run_worker_batch


class ReportPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source.mov'
        self.source.write_bytes(b'original footage')
        self.output = self.root / 'out'
        self.job = JobRequest.create(self.source, self.output, InferenceConfig(output_format='exr'))
        self.manifest = self.root / 'batch.json'
        self.manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch',
                                            'jobs': [{'input': str(self.source), 'output_dir': str(self.output)}]}))

    def test_worker_refuses_source_and_manifest_before_child_and_preserves_bytes(self):
        factory = Mock(side_effect=AssertionError('child must not start'))
        for path in (self.source, self.manifest):
            with self.subTest(path=path):
                before = path.read_bytes()
                report = run_worker_batch(self.manifest, path, client_factory=factory)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(report['report_write']['status'], 'not_written')
        factory.assert_not_called()

    def test_batch_refuses_source_before_session_and_preserves_bytes(self):
        factory_calls = Mock(side_effect=AssertionError('session must not start'))
        class ForbiddenSession(InferenceSession):
            def __init__(self, **kwargs):
                factory_calls(**kwargs)
                super().__init__(**kwargs)
        with self.assertRaisesRegex(ValueError, 'protected'):
            run_batch([self.job], report_path=self.source, session_factory=ForbiddenSession)
        factory_calls.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b'original footage')

    def test_infer_diagnostics_refused_before_model_or_failure_write(self):
        import infer
        loader = Mock(side_effect=RuntimeError('missing model'))
        with InferenceSession(loader=loader) as session, self.assertRaisesRegex(ValueError, 'protected'):
            infer._run_job(session, self.job, self.source)
        loader.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b'original footage')

    def test_invalid_manifest_preserves_existing_report(self):
        self.manifest.write_text('{invalid')
        report_path = self.root / 'report.json'
        report_path.write_bytes(b'previous report')
        result = run_worker_batch(self.manifest, report_path)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['report_write']['status'], 'not_written')
        self.assertEqual(report_path.read_bytes(), b'previous report')

    def test_guard_protects_all_jobs_hints_hardlinks_and_directory_members(self):
        frames = self.root / 'frames'
        frames.mkdir()
        frame = frames / '0001.png'
        frame.write_bytes(b'frame')
        hint = self.root / 'hint.png'
        hint.write_bytes(b'hint')
        job = JobRequest.create(frames, self.root / 'out2', initial_hint_path=hint)
        alias = self.root / 'alias.json'
        os.link(frame, alias)
        for target in (self.source, hint, frame, frames / 'new.json', alias):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'protected'):
                validate_report_destination(target, (self.job, job))

    def test_guard_protects_generated_outputs_but_allows_report_sibling(self):
        config, _ = self.job.configurations()
        layout = output_layout(self.job.input_path, config)
        targets = [self.output / name for name in layout['metadata']]
        targets += [self.output / item['path'] / 'diagnostics.json' for item in layout['outputs'].values()]
        for target in targets:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'protected'):
                validate_report_destination(target, (self.job,))
        self.assertEqual(validate_report_destination(self.output / 'report.json', (self.job,)), self.output / 'report.json')

    def test_guard_protects_actual_asset_root_missing_names_and_model_hardlinks(self):
        assets = self.root / 'selected-assets'
        model_dir = assets / 'vfms/cradio/C-RADIOv4-SO400M/nested'
        model_dir.mkdir(parents=True)
        model = model_dir / 'model.bin'
        model.write_bytes(b'weights')
        alias = self.root / 'model-report.json'
        os.link(model, alias)
        for target in (alias, assets / 'weights/foundation_stats.safetensors', model_dir / 'report.json'):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'protected'):
                validate_report_destination(target, (self.job,), asset_root=assets)

    def test_missing_protected_names_refuse_case_and_unicode_equivalents(self):
        default = self.root / 'weights/CORRIDORKEY_V2.SAFETENSORS'
        with self.assertRaisesRegex(ValueError, 'protected'):
            validate_report_destination(default, (), package_root=self.root)
        declared = self.root / 'Módél.safetensors'
        variant = self.root / unicodedata.normalize('NFD', declared.name.upper())
        with self.assertRaisesRegex(ValueError, 'protected'):
            validate_report_destination(variant, (), weights=declared, package_root=self.root)
        with self.assertRaisesRegex(ValueError, 'protected'):
            validate_report_destination(self.root / 'RECOVERY/queue.sqlite', (),
                                        protected_directories=(self.root / 'recovery',), package_root=self.root)
        # Record an actual spelling alias where the local filesystem supports it;
        # admission remains conservatively case-insensitive on other platforms.
        probe = self.root / 'case-probe'
        probe.write_bytes(b'unchanged')
        if (self.root / 'CASE-PROBE').exists():
            self.assertTrue(os.path.samefile(probe, self.root / 'CASE-PROBE'))
        self.assertEqual(probe.read_bytes(), b'unchanged')

    def test_safe_failure_report_replaces_unrelated_report_with_missing_assets(self):
        import infer
        report_path = self.root / 'report.json'
        report_path.write_bytes(b'old report')
        with (InferenceSession(asset_root=self.root / 'absent-assets', loader=Mock(side_effect=RuntimeError('missing model'))) as session,
              self.assertRaisesRegex(RuntimeError, 'missing model')):
            infer._run_job(session, self.job, report_path)
        self.assertEqual(json.loads(report_path.read_text())['status'], 'failed')

    def test_actual_session_asset_root_is_protected_before_loading(self):
        import infer
        assets = self.root / 'selected-assets'
        weights = assets / 'weights/corridorkey_v2.safetensors'
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b'actual selected model')
        loader = Mock(side_effect=AssertionError('model must not load'))
        with InferenceSession(asset_root=assets, loader=loader) as session, self.assertRaisesRegex(ValueError, 'protected'):
            infer._run_job(session, self.job, weights)
        loader.assert_not_called()
        self.assertEqual(weights.read_bytes(), b'actual selected model')

    def test_correction_assets_and_invalid_inventory_fail_closed(self):
        correction, mask = self.root / 'correction.json', self.root / 'mask.png'
        mask.write_bytes(b'authored mask')
        data = {'schema_version': 1, 'revision': 1,
                'source': {'sha256': '0' * 64, 'start_frame': 0, 'frame_count': 1,
                           'fps': 24., 'width': 1, 'height': 1}, 'shots': [[0, 1]],
                'entries': [{'id': 'mask', 'revision': 1, 'frame': 0, 'kind': 'matte',
                             'asset': {'path': str(mask), 'sha256': '1' * 64}, 'strength': 1.,
                             'direction': 'both', 'range': [0, 1]}]}
        correction.write_text(json.dumps(data))
        job = JobRequest.create(self.source, self.output, InferenceConfig(corrections_path=str(correction)))
        for target in (correction, mask):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'protected'):
                validate_report_destination(target, (job,))
        correction.write_text('{invalid')
        with self.assertRaises(ValueError):
            validate_report_destination(self.root / 'report.json', (job,))

    def test_bounded_inventory_and_cancellation_do_not_create_report(self):
        from corridorkey.control import CancelledError
        target = self.root / 'report.json'
        with patch('corridorkey.output_paths.MAX_SCAN_ENTRIES', 1), self.assertRaisesRegex(ValueError, 'entry limit'):
            validate_report_destination(target, (self.job,))
        with self.assertRaises(CancelledError):
            validate_report_destination(target, (self.job,), check=Mock(side_effect=CancelledError('stop')))
        self.assertFalse(target.exists())

    def test_final_report_rechecks_alias_and_preserves_original_render_failure(self):
        import infer
        report = self.root / 'report.json'
        def fail(*_):
            report.symlink_to(self.source)
            raise OSError('original render failure')
        with (InferenceSession(loader=lambda: object(), runner=fail, disposer=lambda _: None) as session,
              self.assertRaisesRegex(OSError, 'original render failure') as caught):
            infer._run_job(session, self.job, report)
        self.assertIn('alias', caught.exception.__notes__[0])
        self.assertTrue(report.is_symlink())
        self.assertEqual(self.source.read_bytes(), b'original footage')

    def test_real_cli_rejects_protected_reports_without_gpu_or_modification(self):
        commands = [([sys.executable, '-m', 'corridorkey.batch', str(self.manifest), '--report', str(self.manifest)], self.manifest),
                    ([sys.executable, '-m', 'corridorkey.worker', 'run', '--batch', str(self.manifest), '--report', str(self.source)], self.source)]
        for command, target in commands:
            with self.subTest(command=command):
                before = target.read_bytes()
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(target.read_bytes(), before)
                if 'corridorkey.worker' in command:
                    self.assertIn('NOT WRITTEN', result.stdout)

    def test_guard_refuses_leaf_alias_special_and_recovery_tree(self):
        report = self.root / 'existing.json'
        report.write_bytes(b'old report')
        linked = self.root / 'linked.json'
        linked.symlink_to(report)
        with self.assertRaisesRegex(ValueError, 'alias'):
            validate_report_destination(linked, (self.job,))
        with self.assertRaisesRegex(ValueError, 'regular'):
            validate_report_destination(self.root, (self.job,))
        workspace = self.root / 'recovery'
        with self.assertRaisesRegex(ValueError, 'protected'):
            validate_report_destination(workspace / 'queue.sqlite', (self.job,), protected_directories=(workspace,))


if __name__ == '__main__':
    unittest.main()
