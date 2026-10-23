"""Report publication cannot overwrite benchmark inputs, even on failed runs."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey import benchmark


class BenchmarkReportSafetyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.manifest = self.root / 'suite.json'
        self.manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'cpu_correctness'}))
        self.failure = {'kind': 'cpu_correctness', 'summary': {'success': False}, 'status': 'failed'}

    def invoke(self, output, effect=None):
        with patch.object(benchmark, 'ROOT', self.root), \
                patch('sys.argv', ['benchmark', '--manifest', str(self.manifest), '--output', str(output)]), \
                patch.object(benchmark, 'run', side_effect=effect, return_value=self.failure) as runner, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                code = benchmark.main()
            except SystemExit as exc:
                code = exc.code
        return code, runner

    def inference(self, kind='alpha_sequence_evaluation'):
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'local source never decoded by admission')
        self.second_source = self.root / 'second-source.bin'
        self.second_source.write_bytes(b'second local source')
        self.hint = self.root / 'hint.bin'
        self.hint.write_bytes(b'local hint')
        self.weights = self.root / 'model.bin'
        self.weights.write_bytes(b'local model never loaded by admission')
        self.reference = self.root / 'reference.npy'
        self.reference.write_bytes(b'reference inventory only')
        self.mask = self.root / 'mask.npy'
        self.mask.write_bytes(b'mask inventory only')
        self.quality = self.root / 'quality.json'
        provenance = {'author': 'fixture', 'origin': 'local', 'permitted_use': 'test', 'split': 'synthetic'}
        frame = {'frame': 0, 'candidate': '@render', 'reference': self.reference.name, 'mask': self.mask.name}
        quality = {'schema_version': 1, 'kind': kind, 'fps': 24, 'source': self.source.name,
                   'reference_provenance': provenance, 'frames': [frame]}
        if kind == 'colour_sequence_evaluation':
            quality.update(colour_space='linear_srgb_bt709', associations={'candidate': 'straight', 'reference': 'straight'})
            quality['frames'] = [{'frame': 0, 'candidate_rgb': '@render', 'candidate_alpha': '@render',
                                  'reference_rgb': self.reference.name, 'reference_alpha': self.reference.name,
                                  'foreground_mask': self.mask.name}]
        elif kind == 'temporal_sequence_evaluation':
            quality['flow_convention'] = 'current_to_previous_xy_pixels'
            quality['flow_provenance'] = {'method': 'fixture', 'origin': 'local',
                                          'visibility_definition': 'fixture', 'independent_of_candidate': True}
            quality['frames'].append({**frame, 'frame': 1, 'backward_flow': self.mask.name, 'flow_previous_frame': 0})
        self.quality.write_text(json.dumps(quality))
        self.batch = self.root / 'batch.json'
        self.batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch',
            'weights': self.weights.name, 'jobs': [{
                'input': self.source.name, 'output_dir': 'render', 'hint_first_frame': self.hint.name,
                'settings': {'output_format': 'exr', 'outputs': ['alpha', 'fg']}},
                {'input': self.second_source.name, 'output_dir': 'second-render'}]}))
        self.manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'inference',
            'batch_manifest': self.batch.name, 'model_state': 'cold_process', 'quality_manifests': [self.quality.name, None]}))

    def test_manifest_and_hardlink_are_rejected_before_any_cpu_run(self):
        alias = self.root / 'alias.json'
        os.link(self.manifest, alias)
        before = self.manifest.read_bytes()
        for target in (self.manifest, alias):
            with self.subTest(target=target):
                code, runner = self.invoke(target)
                self.assertNotEqual(code, 0)
                runner.assert_not_called()
                self.assertEqual(self.manifest.read_bytes(), before)
                self.assertEqual(alias.read_bytes(), before)

    def test_inference_inputs_batch_weights_and_quality_assets_are_protected(self):
        self.inference()
        for target in (self.source, self.second_source, self.hint, self.weights, self.batch,
                       self.quality, self.reference, self.mask):
            with self.subTest(target=target):
                before = target.read_bytes()
                code, runner = self.invoke(target)
                self.assertNotEqual(code, 0)
                runner.assert_not_called()
                self.assertEqual(target.read_bytes(), before)

    def test_colour_and_temporal_asset_roles_are_inventoried_without_decoding(self):
        for kind in ('colour_sequence_evaluation', 'temporal_sequence_evaluation'):
            self.inference(kind)
            for target in (self.reference, self.mask):
                with self.subTest(kind=kind, target=target):
                    before = target.read_bytes()
                    _, runner = self.invoke(target)
                    runner.assert_not_called()
                    self.assertEqual(target.read_bytes(), before)

    def test_safe_existing_report_can_be_replaced_with_truthful_failure(self):
        report = self.root / 'report.json'
        report.write_bytes(b'previous report')
        code, runner = self.invoke(report)
        self.assertEqual(code, 1)
        runner.assert_called_once()
        self.assertEqual(json.loads(report.read_bytes()), self.failure)

    def test_actual_cli_rejects_suite_and_reference_collisions_without_execution(self):
        for inference in (False, True):
            if inference:
                self.inference()
            target = self.reference if inference else self.manifest
            before = target.read_bytes()
            result = subprocess.run([sys.executable, '-m', 'corridorkey.benchmark', '--manifest',
                                     str(self.manifest), '--output', str(target)],
                                    capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn('protected', result.stderr)
            self.assertEqual(target.read_bytes(), before)
            self.assertNotIn('repetition:', result.stdout)

    def test_destination_alias_created_during_run_is_refused_before_publication(self):
        self.inference()
        output = self.root / 'report.json'
        before = self.source.read_bytes()

        def mutate(*args, **kwargs):
            output.symlink_to(self.source)
            return self.failure

        _, runner = self.invoke(output, mutate)
        runner.assert_called_once()
        self.assertTrue(output.is_symlink())
        self.assertEqual(self.source.read_bytes(), before)

    def test_suite_symlink_retargeting_refuses_publication_even_when_old_bytes_remain(self):
        original = self.manifest
        alternate = self.root / 'alternate-suite.json'
        alternate.write_bytes(original.read_bytes())
        alias = self.root / 'selected-suite.json'
        alias.symlink_to(original)
        self.manifest = alias
        output = self.root / 'report.json'
        output.write_bytes(b'previous report')

        def mutate(*args, **kwargs):
            alias.unlink()
            alias.symlink_to(alternate)
            return self.failure

        _, runner = self.invoke(output, mutate)
        runner.assert_called_once()
        self.assertEqual(output.read_bytes(), b'previous report')
        self.assertEqual(original.read_bytes(), alternate.read_bytes())

    def test_changed_batch_after_run_refuses_publication_and_preserves_old_report(self):
        self.inference()
        output = self.root / 'report.json'
        output.write_bytes(b'previous report')

        def mutate(*args, **kwargs):
            self.batch.write_text('{}')
            return self.failure

        code, runner = self.invoke(output, mutate)
        self.assertNotEqual(code, 0)
        runner.assert_called_once()
        self.assertEqual(output.read_bytes(), b'previous report')

    def test_incomplete_quality_inventory_refuses_before_run_without_overwriting_report(self):
        self.inference()
        self.quality.write_text('{"kind":"unknown"}')
        output = self.root / 'report.json'
        output.write_bytes(b'previous report')
        _, runner = self.invoke(output)
        runner.assert_not_called()
        self.assertEqual(output.read_bytes(), b'previous report')

    def test_decode_reuse_protects_generated_root_but_allows_sibling_report(self):
        self.manifest.write_text(json.dumps({'schema_version': 1, 'kind': 'decode_reuse',
                                            'artifact_parent': str(self.root)}))
        generated = self.root / 'generated'
        generated.mkdir()
        protected = generated / 'frame.npy'
        protected.write_bytes(b'generated pixels')
        result = {**self.failure, 'kind': 'decode_reuse', 'artifact_dir': str(generated)}
        _, runner = self.invoke(protected, lambda *args, **kwargs: result)
        runner.assert_called_once()
        self.assertEqual(protected.read_bytes(), b'generated pixels')
        output = self.root / 'report.json'
        code, runner = self.invoke(output, lambda *args, **kwargs: result)
        self.assertEqual(code, 1)
        runner.assert_called_once()
        self.assertEqual(json.loads(output.read_bytes()), result)
