"""Regression proofs for batch preservation and retained report admission."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from corridorkey import output_paths
from corridorkey.batch import run_batch
from corridorkey.config import InferenceConfig
from corridorkey.session import InferenceSession, JobRequest
from corridorkey.worker_client import run_worker_batch


class AuditSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.first = self.root / 'plate.png'
        self.second = self.root / 'second.png'
        self.first.write_bytes(b'first source')
        self.second.write_bytes(b'second source')
        self.config = InferenceConfig(output_format='exr', outputs={'alpha'})
        self.output = self.root / 'out'
        self.hint = self.output / 'plate_alpha_exr/alpha_000000.exr'
        self.hint.parent.mkdir(parents=True)
        self.hint.write_bytes(b'artist hint')
        self.jobs = (JobRequest.create(self.first, self.output, self.config),
                     JobRequest.create(self.second, self.root / 'out2', self.config,
                                       initial_hint_path=self.hint))

    def test_cross_job_hint_refused_before_session_or_mutation(self):
        factory = Mock(side_effect=AssertionError('session must not start'))

        class ForbiddenSession(InferenceSession):
            def __init__(self, **kwargs):
                factory(**kwargs)
                super().__init__(**kwargs)

        with self.assertRaisesRegex(ValueError, 'protected'):
            run_batch(self.jobs, session_factory=ForbiddenSession)
        factory.assert_not_called()
        self.assertEqual(self.hint.read_bytes(), b'artist hint')

    def test_cross_job_hardlink_refused_and_noncolliding_batch_allowed(self):
        self.hint.unlink()
        os.link(self.second, self.hint)
        plain_jobs = (self.jobs[0], JobRequest.create(self.second, self.root / 'out2', self.config))
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            output_paths.validate_batch_output_paths(plain_jobs)
        self.hint.unlink()
        self.hint.write_bytes(b'unrelated rerender')
        jobs = (self.jobs[0], JobRequest.create(self.second, self.root / 'out2', self.config))
        output_paths.validate_batch_output_paths(jobs)
        self.assertEqual(self.hint.read_bytes(), b'unrelated rerender')

    def test_worker_batch_refuses_cross_job_hint_before_child(self):
        import json
        manifest = self.root / 'batch.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [
            {'input': str(self.first), 'output_dir': str(self.output),
             'settings': {'output_format': 'exr', 'outputs': ['alpha']}},
            {'input': str(self.second), 'output_dir': str(self.root / 'out2'),
             'hint_first_frame': str(self.hint),
             'settings': {'output_format': 'exr', 'outputs': ['alpha']}}]}))
        factory = Mock(side_effect=AssertionError('child must not start'))
        report = run_worker_batch(manifest, self.root / 'report.json', client_factory=factory)
        self.assertEqual(report['status'], 'failed')
        factory.assert_not_called()
        self.assertEqual(self.hint.read_bytes(), b'artist hint')

    def test_retained_report_recheck_does_not_rescan_outputs(self):
        job = self.jobs[0]
        target = self.root / 'report.json'
        admit = output_paths.prepare_report_destination(target, (job,))
        for index in range(100):
            (self.hint.parent / f'alpha_{index:06d}.exr').write_bytes(b'new output')
        with patch.object(output_paths.os, 'scandir', side_effect=AssertionError('full rescan')):
            self.assertEqual(admit(), target)
        target.symlink_to(self.first)
        with self.assertRaisesRegex(ValueError, 'alias'):
            admit()
        self.assertEqual(self.first.read_bytes(), b'first source')

    def test_new_export_hardlink_still_protected_by_retained_guard(self):
        target = self.root / 'report.json'
        admit = output_paths.prepare_report_destination(target, (self.jobs[0],))
        new_export = self.hint.parent / 'alpha_000001.exr'
        new_export.write_bytes(b'new export')
        os.link(new_export, target)
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            admit()
        self.assertEqual(new_export.read_bytes(), b'new export')

    def test_selected_nested_model_hardlink_refused_before_recovery_workspace(self):
        from corridorkey.recovery import run_recoverable

        selected = self.root / 'selected-cradio'
        nested = selected / 'nested/weights.bin'
        nested.parent.mkdir(parents=True)
        nested.write_bytes(b'approved model')
        self.hint.unlink()
        os.link(nested, self.hint)
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            output_paths.validate_batch_output_paths(
                (self.jobs[0],), protected_directories=(selected,))
        workspace = self.root / 'recovery'
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            run_recoverable((self.jobs[0],), workspace=workspace, max_cache_bytes=0,
                            assets={'cradio_dir': selected}, environment={})
        self.assertFalse(workspace.exists())
        self.assertEqual(nested.read_bytes(), b'approved model')

    def test_linked_model_children_refused_for_default_and_selected_roots(self):
        outside = self.root / 'outside/deep/weights.bin'
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b'approved model')
        self.hint.unlink()
        os.link(outside, self.hint)
        default = self.root / 'vfms/cradio/C-RADIOv4-SO400M'
        selected = self.root / 'selected-model'
        for directory in (default, selected):
            directory.mkdir(parents=True)
            try:
                (directory / 'linked').symlink_to(outside.parent.parent, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f'local directory symlinks unavailable: {exc}')
        with self.assertRaisesRegex(ValueError, 'linked|reparse'):
            output_paths.validate_batch_output_paths((self.jobs[0],), package_root=self.root)
        (default / 'linked').unlink()
        with self.assertRaisesRegex(ValueError, 'linked|reparse'):
            output_paths.validate_batch_output_paths(
                (self.jobs[0],), package_root=self.root, protected_directories=(selected,))
        self.assertEqual(outside.read_bytes(), b'approved model')
