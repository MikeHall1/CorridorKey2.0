"""Saved recommendation input artifacts cannot become inference outputs/reports."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class ShotProfilePathTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        import infer
        from corridorkey.session import JobRequest
        from corridorkey.shot_profiles import analyze_job, save_profiles
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        source = self.root / 'source'
        source.mkdir()
        cv2.imwrite(str(source / '0.png'), np.full((16, 32, 3), 90, np.uint8))
        self.args = ['--input', str(source), '--output_dir', str(self.root / 'output'),
                     '--output_format', 'exr', '--outputs', 'alpha']
        self.session = SimpleNamespace(weights=None, package_root=None, asset_root=None, last_result=None)
        self.session.run = Mock(side_effect=self.render)
        base = infer.main(self.args, session=SimpleNamespace(weights=None, last_result=None, run=lambda job: job))
        assert isinstance(base, JobRequest)
        self.plan = analyze_job(base)
        self.base = base
        self.profile = self.root / 'profiles.json'
        save_profiles(self.profile, self.plan, base)
        self.original = self.profile.read_bytes()

    def render(self, job):
        from corridorkey.session import JobResult
        result = JobResult(json.dumps({'status': 'complete', 'fixture': 'not neural inference'}))
        self.session.last_result = result
        return result

    def test_cli_diagnostics_cannot_overwrite_loaded_profile_or_hardlink(self):
        import infer
        for same in (True, False):
            with self.subTest(same=same):
                report = self.profile if same else self.root / 'hardlink.json'
                if not same:
                    os.link(self.profile, report)
                self.session.run.reset_mock()
                caught = None
                try:
                    infer.main([*self.args, '--shot_profiles', str(self.profile), '--diagnostics', str(report)],
                               session=self.session)
                except ValueError as exc:
                    caught = exc
                self.assertEqual(self.profile.read_bytes(), self.original)
                self.assertIsNotNone(caught, 'profile aliases must be refused before rendering')
                self.session.run.assert_not_called()

    def test_cli_writer_targets_and_hardlink_aliases_preserve_profile(self):
        import infer
        from corridorkey.output_paths import output_layout
        output = self.root / 'new-output'
        output.mkdir()
        config, _ = self.base.configurations()
        layout = output_layout(self.base.input_path, config)
        metadata = output / layout['metadata'][0]
        metadata.write_bytes(self.original)
        with self.assertRaisesRegex(ValueError, 'protected'):
            infer.main([*self.args, '--output_dir', str(output), '--shot_profiles', str(metadata)], session=self.session)
        self.session.run.assert_not_called()
        self.assertEqual(metadata.read_bytes(), self.original)
        metadata.unlink()
        planned = layout['outputs']['alpha']
        leaf = output / planned['path']
        leaf.mkdir()
        alias = leaf / f"{planned['frame_name']}_000000.exr"
        os.link(self.profile, alias)
        with self.assertRaisesRegex(ValueError, 'hard-link'):
            infer.main([*self.args, '--output_dir', str(output), '--shot_profiles', str(self.profile)], session=self.session)
        self.session.run.assert_not_called()
        self.assertEqual(self.profile.read_bytes(), self.original)

    def test_final_report_fence_preserves_input_and_original_render_error(self):
        import infer
        for fail in (False, True):
            report = self.root / f'late-report-{fail}.json'
            def render(job, *, report=report, fail=fail):
                os.link(self.profile, report)
                result = self.render(job)
                if fail:
                    raise OSError('original render failure')
                return result
            self.session.run.side_effect = render
            expected = OSError if fail else ValueError
            with self.subTest(fail=fail), self.assertRaises(expected) as caught:
                infer.main([*self.args, '--shot_profiles', str(self.profile), '--diagnostics', str(report)],
                           session=self.session)
            if fail:
                self.assertEqual(str(caught.exception), 'original render failure')
                self.assertTrue(any('hard-link' in note for note in caught.exception.__notes__))
            self.assertEqual(self.profile.read_bytes(), self.original)
