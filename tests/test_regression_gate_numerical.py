"""Actual local native exports; no neural/GPU or visual acceptance claim."""
import copy
import hashlib
import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey import regression_gate as gate
from corridorkey import regression_numerical as numerical
from corridorkey.identities import source_identity
from tests import test_regression_gate as fixtures


class NumericalGateTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.RegressionGateTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.report, self.policy = fixture.root.resolve(), fixture.report, fixture.policy
        self.report_path, self.policy_path = self.root / 'pair.json', self.root / 'policy.json'
        self.evidence_path = self.root / 'evidence.json'
        self.enterContext(patch.dict(os.environ, {'OPENCV_IO_ENABLE_OPENEXR': '1'}))
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'owned analytic fixture source; not inference')
        for side in gate.SIDES:
            frozen = self.report['provenance'][side]['normalized_batch']['jobs'][0]
            frozen['input'] = str(self.source)
            frozen['settings'].update(output_format='exr', exr_dtype='float', frame_dir_fps=24, start_frame=0, low_vram=False, outputs=['alpha', 'fg'])
        for pair in self.report['pairs']:
            for side in gate.SIDES:
                sample = pair['samples'][side]
                frozen = self.report['provenance'][side]['normalized_batch']['jobs'][0]
                job = sample['runs'][0]['jobs'][0]
                job['output_dir'] = str(Path(job['output_dir']).resolve())
                job.update(input=str(self.source), requested_settings=copy.deepcopy(frozen['settings']))
                job['diagnostics'].update(effective_settings=copy.deepcopy(frozen['settings']), profiling_enabled=False,
                    source={'start_frame': 0, 'num_frames': 1, 'native_fps': 24, 'native_hw': [2, 2], 'output_hw': [2, 2]},
                    frames_submitted=1, execution={'backend': 'pytorch', 'requested_device': 'cuda', 'gpu': 'analytic fixture; NOT hardware evidence',
                        'device_index': 0, 'cuda_version': 'fixture', 'autocast_dtype': 'bfloat16',
                        'rvm_cache_dtype': 'float16', 'baton_dtype': 'float16', 'cached_playback': False,
                        'source_colour': 'display_coded_assumed_srgb', 'model_rgb': 'signed_asinh_scene_linear',
                        'alpha': 'coverage', 'temporal_strategy': 'full_rate_with_chronological_rvm_prepass',
                        'nonfinite_policy': 'fail; no NaN/Inf repair', 'rgb_primaries': 'assumed sRGB/BT.709',
                        'foundation_cache_dtype': None if side == 'baseline' else 'float32',
                        'foundation_extraction_policy': 'full_windows' if side == 'baseline' else 'missing_iid_frames',
                        'stage2_frame_evaluations': 1, 'stage2_calls': 1, 'completed_prediction_frames': 1},
                    outputs={}, output_contracts={})
                sample['assets'] = {str(self.source): source_identity(self.source)}
                warm = sample['warmup'][0]
                warm.update(input=job['input'], requested_settings=copy.deepcopy(frozen['settings']))
                warm['diagnostics']['effective_settings'] = copy.deepcopy(frozen['settings'])
                for kind in ('alpha', 'fg'):
                    folder = Path(job['output_dir']) / kind
                    folder.mkdir(parents=True)
                    image = np.array([[0, 1e-8], [0.5, 1]], np.float32) if kind == 'alpha' else np.full((2, 2, 3), -2.0, np.float32)
                    self.assertTrue(cv2.imwrite(str(folder / f'{kind}_000000.exr'), image))
                    job['diagnostics']['outputs'][kind] = str(folder)
                    job['diagnostics']['output_contracts'][kind] = ({'format': 'exr', 'channels': 1, 'alpha_representation': 'coverage (no transfer function)'} if kind == 'alpha' else
                        {'format': 'exr', 'channels': 3, 'rgb_representation': 'scene_linear', 'alpha_association': None, 'rgb_primaries': 'assumed sRGB/BT.709; no automatic colour management'})
        self.policy.update(schema_version=2, required_evidence=['numerical_parity'],
                           numerical={'constraints': [{'job': 0, 'metric': 'alpha.mae', 'units': 'alpha/pixel', 'maximum': 0.001},
                               {'job': 0, 'metric': 'colour.composite_mae', 'units': 'linear RGB/channel', 'maximum': 0.001}],
                                      'budgets': {'max_files': 64, 'max_bytes': 16 * 1024**2, 'max_pixels': 4}})
        self.policy['context_sha256'] = gate.digest(gate.context(self.report))
        self.report_path.write_text(json.dumps(self.report))
        self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()

    def capture(self):
        self.report_path.write_text(json.dumps(self.report))
        result = gate.capture_numerical(self.report_path, self.policy_path, self.policy_sha)
        self.evidence_path.write_text(json.dumps(result))
        return hashlib.sha256(self.evidence_path.read_bytes()).hexdigest()

    def evaluate(self):
        return gate.evaluate(self.report_path, self.policy_path, self.policy_sha,
                             numerical_evidence_path=self.evidence_path,
                             numerical_evidence_sha256=hashlib.sha256(self.evidence_path.read_bytes()).hexdigest())

    def test_predeclared_policy_close_exports_pass_and_real_wrong_exports_fail_before_capture(self):
        original_policy = self.policy_path.read_bytes()
        self.capture()
        self.assertEqual(self.evaluate()['status'], 'pass')
        for kind in ('alpha', 'fg'):
            with self.subTest(kind=kind):
                job = self.report['pairs'][0]['samples']['candidate']['runs'][0]['jobs'][0]
                path = Path(job['diagnostics']['outputs'][kind]) / f'{kind}_000000.exr'
                original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                wrong = np.full_like(original, 0.8 if kind == 'alpha' else 20)
                self.assertTrue(cv2.imwrite(str(path), wrong))
                self.capture()  # New honest integrity capture, unchanged predeclared tolerances.
                result = self.evaluate()
                self.assertEqual(result['status'], 'fail', result)
                self.assertTrue(cv2.imwrite(str(path), original))
        self.assertEqual(self.policy_path.read_bytes(), original_policy)

    def test_missing_tampered_cancelled_and_budget_evidence_never_pass(self):
        self.assertEqual(gate.evaluate(self.report_path, self.policy_path, self.policy_sha)['status'], 'not_verified')
        self.capture()
        captured = self.evidence_path.read_bytes()
        self.evidence_path.write_bytes(captured + b' ')
        wrong_sha = hashlib.sha256(captured).hexdigest()
        self.assertEqual(gate.evaluate(self.report_path, self.policy_path, self.policy_sha,
            numerical_evidence_path=self.evidence_path, numerical_evidence_sha256=wrong_sha)['status'], 'not_verified')
        def cancel():
            raise gate.CancelledError('numerical fixture')
        self.assertEqual(gate.evaluate(self.report_path, self.policy_path, self.policy_sha, check=cancel)['status'], 'cancelled')

    def cli_args(self, output):
        return ['--report', str(self.report_path), '--policy', str(self.policy_path), '--policy-sha256', self.policy_sha,
                '--numerical-evidence', str(self.evidence_path), '--numerical-evidence-sha256',
                hashlib.sha256(self.evidence_path.read_bytes()).hexdigest(), '--output', str(output)]

    def test_cli_protects_all_inputs_export_roots_hardlinks_and_late_aliases(self):
        from corridorkey import regression_numerical as numerical
        self.capture()
        job = self.report['pairs'][0]['samples']['candidate']['runs'][0]['jobs'][0]
        export = Path(job['diagnostics']['outputs']['alpha']) / 'alpha_000000.exr'
        protected = (self.source, self.report_path, self.policy_path, self.evidence_path, export,
                     export.parent / 'new-report.json')
        for output in protected:
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                gate.main(self.cli_args(output))
        destination = self.root / 'late-hardlink.json'
        original = numerical.fence
        def late(evidence, **kwargs):
            os.link(export, destination)
            original(evidence, **kwargs)
        before = export.read_bytes()
        try:
            with patch.object(numerical, 'fence', side_effect=late), self.assertRaises(ValueError):
                gate.main(self.cli_args(destination))
            self.assertEqual(export.read_bytes(), before)
        finally:
            destination.unlink(missing_ok=True)
        # A late parent redirect is also refused before opening any destination.
        self.capture()
        parent = self.root / 'late-parent'
        def redirect(evidence, **kwargs):
            original(evidence, **kwargs)
            parent.rmdir()
            parent.symlink_to(export.parent, target_is_directory=True)
        try:
            with patch.object(numerical, 'fence', side_effect=redirect), self.assertRaises(ValueError):
                gate.main(self.cli_args(parent / 'new.json'))
            self.assertFalse((export.parent / 'new.json').exists())
        finally:
            if parent.is_symlink():
                parent.unlink()

    def test_overbudget_corrupt_native_and_late_content_changes_are_unverified(self):
        from corridorkey.regression_numerical import snapshot
        with self.assertRaisesRegex(ValueError, 'budget'):
            snapshot(self.report_path, {'max_files': 64, 'max_bytes': 1, 'max_pixels': 4})
        self.capture()
        job = self.report['pairs'][0]['samples']['candidate']['runs'][0]['jobs'][0]
        path = Path(job['diagnostics']['outputs']['alpha']) / 'alpha_000000.exr'
        raw = path.read_bytes()
        path.write_bytes(b'corrupt native export')
        self.assertEqual(self.evaluate()['status'], 'not_verified')
        with self.assertRaises(ValueError):
            self.capture()
        path.write_bytes(raw)
        self.capture()
        def corrupt(template, record, **kwargs):
            from corridorkey.benchmark_quality import evaluate_job_quality
            result = evaluate_job_quality(template, record, **kwargs)
            path.write_bytes(raw + b'changed')
            return result
        with patch('corridorkey.regression_numerical.evaluate_job_quality', side_effect=corrupt):
            self.assertEqual(self.evaluate()['status'], 'not_verified')

    def test_cancellation_after_native_evaluation_and_receipt_linkage(self):
        self.capture()
        from corridorkey.benchmark_quality import evaluate_job_quality
        cancelled = [False]
        def native(template, record, **kwargs):
            result = evaluate_job_quality(template, record, **kwargs)
            cancelled[0] = True
            return result
        def check():
            if cancelled[0]:
                raise gate.CancelledError('after native evaluation')
        with patch('corridorkey.regression_numerical.evaluate_job_quality', side_effect=native):
            result = gate.evaluate(self.report_path, self.policy_path, self.policy_sha, check=check,
                numerical_evidence_path=self.evidence_path,
                numerical_evidence_sha256=hashlib.sha256(self.evidence_path.read_bytes()).hexdigest())
        self.assertEqual(result['status'], 'cancelled')
        evidence = json.loads(self.evidence_path.read_text())
        evidence['policy_sha256'] = '0' * 64
        self.evidence_path.write_text(json.dumps(evidence))
        self.assertEqual(self.evaluate()['status'], 'not_verified')

    def test_actual_cli_capture_then_gate_uses_unchanged_policy_and_exclusive_reports(self):
        before = self.policy_path.read_bytes()
        captured = self.root / 'captured-cli.json'
        self.assertEqual(gate.main(['--report', str(self.report_path), '--policy', str(self.policy_path),
            '--policy-sha256', self.policy_sha, '--capture-numerical', '--output', str(captured)]), 0)
        self.evidence_path = captured
        output = self.root / 'gate-result.json'
        self.assertEqual(gate.main(self.cli_args(output)), 0)
        result = json.loads(output.read_text())
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['checks'][-1]['name'], 'numerical_parity')
        self.assertEqual(len(result['checks'][-1]['observations']), 6)
        with self.assertRaises(FileExistsError):
            gate.main(self.cli_args(output))
        self.assertEqual(self.policy_path.read_bytes(), before)

    def two_frames(self):
        for pair in self.report['pairs']:
            for side in gate.SIDES:
                job = pair['samples'][side]['runs'][0]['jobs'][0]
                job['diagnostics']['frames_submitted'] = 2
                job['diagnostics']['source']['num_frames'] = 2
                for kind in ('alpha', 'fg'):
                    folder = Path(job['diagnostics']['outputs'][kind])
                    (folder / f'{kind}_000001.exr').write_bytes((folder / f'{kind}_000000.exr').read_bytes())

    def test_actual_video_fps_and_allowed_cache_batch_execution_variants(self):
        # File/video FPS comes from the measured source, not the image-sequence option.
        self.two_frames()
        for pair in self.report['pairs']:
            for side in gate.SIDES:
                sample = pair['samples'][side]
                job = sample['runs'][0]['jobs'][0]
                job['diagnostics']['source']['native_fps'] = 30
                batch = 1 if side == 'baseline' else 2
                frozen = self.report['provenance'][side]['normalized_batch']['jobs'][0]['settings']
                frozen['stage2_batch'] = batch
                for measured in (job, sample['warmup'][0]):
                    measured['requested_settings']['stage2_batch'] = batch
                    measured['diagnostics']['effective_settings']['stage2_batch'] = batch
                job['diagnostics']['execution'].update(stage2_frame_evaluations=2, completed_prediction_frames=2,
                                                     stage2_calls=2 if side == 'baseline' else 1)
        self.policy['context_sha256'] = gate.digest(gate.context(self.report))
        self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        self.capture()
        result = self.evaluate()
        self.assertEqual(result['status'], 'pass', result)
        evidence = json.loads(self.evidence_path.read_text())
        sides = evidence['jobs'][0]['sides']
        self.assertEqual(sides['baseline']['execution']['stage2_calls'], 2)
        self.assertEqual(sides['candidate']['execution']['stage2_calls'], 1)
        self.assertEqual(sides['candidate']['source']['native_fps'], 30)
        for field, changed in (('backend', 'other'), ('requested_device', 'cpu'), ('device_index', 1), ('autocast_dtype', 'float16')):
            with self.subTest(invariant=field):
                job = self.report['pairs'][0]['samples']['candidate']['runs'][0]['jobs'][0]
                original = job['diagnostics']['execution'][field]
                job['diagnostics']['execution'][field] = changed
                with self.assertRaises(ValueError):
                    self.capture()
                job['diagnostics']['execution'][field] = original

    def test_sequence_fps_must_still_match_declared_rate(self):
        source = self.root / 'sequence'
        source.mkdir()
        self.assertTrue(cv2.imwrite(str(source / '0.png'), np.zeros((2, 2, 3), np.uint8)))
        for side in gate.SIDES:
            self.report['provenance'][side]['normalized_batch']['jobs'][0]['input'] = str(source)
        for pair in self.report['pairs']:
            for sample in pair['samples'].values():
                job = sample['runs'][0]['jobs'][0]
                job['input'] = str(source)
                sample['warmup'][0]['input'] = str(source)
                sample['assets'] = {str(source): source_identity(source)}
        self.policy['context_sha256'] = gate.digest(gate.context(self.report))
        self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        self.capture()
        self.assertEqual(self.evaluate()['status'], 'pass')
        evidence = json.loads(self.evidence_path.read_text())
        added = source / '2.png'
        self.assertTrue(cv2.imwrite(str(added), np.zeros((2, 2, 3), np.uint8)))
        with self.assertRaisesRegex(ValueError, 'membership'):
            numerical.fence(evidence)
        added.unlink()
        numerical.fence(evidence)
        for pair in self.report['pairs']:
            for sample in pair['samples'].values():
                sample['runs'][0]['jobs'][0]['diagnostics']['source']['native_fps'] = 30
        with self.assertRaisesRegex(ValueError, 'Sequence FPS'):
            self.capture()

    def test_cache_policy_variation_preserves_full_per_side_binding(self):
        self.capture()
        self.assertEqual(self.evaluate()['status'], 'pass')

    def test_cache_dtype_must_match_each_sides_effective_policy(self):
        self.capture()
        self.assertEqual(self.evaluate()['status'], 'pass')
        for side, invalid in (('candidate', 'float16'), ('candidate', None), ('baseline', 'float32')):
            with self.subTest(side=side, invalid=invalid):
                execution = self.report['pairs'][0]['samples'][side]['runs'][0]['jobs'][0]['diagnostics']['execution']
                original = execution['foundation_cache_dtype']
                execution['foundation_cache_dtype'] = invalid
                with self.assertRaisesRegex(ValueError, 'cache dtype'):
                    self.capture()
                execution['foundation_cache_dtype'] = original
        # Low-VRAM's disk prepass also requires float32, even with zero RAM cache.
        for side in gate.SIDES:
            frozen = self.report['provenance'][side]['normalized_batch']['jobs'][0]['settings']
            frozen['low_vram'] = True
        for pair in self.report['pairs']:
            for sample in pair['samples'].values():
                job = sample['runs'][0]['jobs'][0]
                for measured in (job, sample['warmup'][0]):
                    measured['requested_settings']['low_vram'] = True
                    measured['diagnostics']['effective_settings']['low_vram'] = True
                job['diagnostics']['execution'].update(foundation_cache_dtype='float32', foundation_extraction_policy='disk_prepass')
        self.policy['context_sha256'] = gate.digest(gate.context(self.report))
        self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        self.capture()
        self.assertEqual(self.evaluate()['status'], 'pass')
        job = self.report['pairs'][0]['samples']['baseline']['runs'][0]['jobs'][0]
        job['diagnostics']['execution']['foundation_cache_dtype'] = None
        with self.assertRaisesRegex(ValueError, 'cache dtype'):
            self.capture()

    def test_cancellation_stops_before_later_frame_decode(self):
        self.two_frames()
        self.capture()
        from corridorkey.evaluation_colour_media import read_rgb_frame
        decoded = []
        pending = [False]
        def read(path, **kwargs):
            value = read_rgb_frame(path, **kwargs)
            decoded.append(Path(path).name)
            pending[0] = True
            return value
        def check():
            if pending[0]:
                raise gate.CancelledError('after first native RGB frame')
        with patch('corridorkey.evaluation_colour_media.read_rgb_frame', side_effect=read):
            result = gate.evaluate(self.report_path, self.policy_path, self.policy_sha, check=check,
                numerical_evidence_path=self.evidence_path,
                numerical_evidence_sha256=hashlib.sha256(self.evidence_path.read_bytes()).hexdigest())
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(decoded, ['fg_000000.exr'])

    def test_source_colour_range_dtype_coverage_contracts_reject(self):
        self.capture()
        job = self.report['pairs'][0]['samples']['candidate']['runs'][0]['jobs'][0]
        original = copy.deepcopy(job)
        for defect in ('colour', 'association', 'range', 'dtype', 'coverage', 'model'):
            with self.subTest(defect=defect):
                job.clear(); job.update(copy.deepcopy(original))
                if defect == 'colour':
                    job['diagnostics']['output_contracts']['fg']['rgb_primaries'] = 'unknown'
                elif defect == 'association':
                    job['diagnostics']['output_contracts']['fg']['alpha_association'] = 'premultiplied'
                elif defect == 'range':
                    job['diagnostics']['source']['start_frame'] = 5
                elif defect == 'dtype':
                    job['requested_settings']['exr_dtype'] = 'half'
                elif defect == 'coverage':
                    job['diagnostics']['frames_submitted'] = 2
                else:
                    job['diagnostics']['execution']['backend'] = 'changed'
                with self.assertRaises(ValueError):
                    self.capture()

    def test_source_growth_rejects_without_redundant_content_read(self):
        self.capture()
        original_read = numerical.read_file
        original_open = Path.open
        reopened = []

        def grow_after_read(pin, **kwargs):
            result = original_read(pin, **kwargs)
            if pin.path == str(self.source):
                self.source.write_bytes(b'x' * 2048)
            return result

        def watch_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get('mode', 'r')
            if path == self.source and mode == 'rb':
                reopened.append(path.stat().st_size)
            return original_open(path, *args, **kwargs)

        with (patch.object(numerical, 'read_file', grow_after_read),
              patch.object(Path, 'open', watch_open), self.assertRaises(ValueError)):
            self.capture()
        self.assertEqual(reopened, [], 'Growth must reject before a redundant source content read')

    def test_verified_source_identity_preserves_file_sequence_wire_and_natural_order(self):
        self.capture()
        evidence = json.loads(self.evidence_path.read_text())
        assets = {a['path']: a for a in evidence['assets']}
        expected = source_identity(self.source)
        current, _ = numerical.snapshot(self.report_path, self.policy['numerical']['budgets'])
        # The old direct-source reader yields byte-identical receipt JSON for stable inputs.
        def legacy_identity(path, _assets, _maximum, *, check=None):
            return source_identity(path, check=check)
        with patch.object(numerical, '_source_identity', legacy_identity):
            legacy, _ = numerical.snapshot(self.report_path, self.policy['numerical']['budgets'])
        self.assertEqual(json.dumps(current), json.dumps(legacy))
        with patch.object(Path, 'open', side_effect=AssertionError('No duplicate content reads')):
            actual = numerical._source_identity(self.source, assets, 64)
        self.assertEqual(json.dumps(actual), json.dumps(expected))

        sequence = self.root / 'ordered-source'
        sequence.mkdir()
        for name in ('10.png', '2.png', '1.png'):
            self.assertTrue(cv2.imwrite(str(sequence / name), np.zeros((2, 2, 3), np.uint8)))
        assets = {}
        for path in sequence.iterdir():
            pin = numerical.admit_file(path)
            sha, _ = numerical.read_file(pin)
            assets[pin.path] = {**asdict(pin), 'sha256': sha}
        expected = source_identity(sequence)
        with patch.object(Path, 'open', side_effect=AssertionError('No duplicate content reads')):
            actual = numerical._source_identity(sequence, assets, 64)
        self.assertEqual(json.dumps(actual), json.dumps(expected))
        self.assertEqual([f['name'] for f in actual['files']], ['1.png', '2.png', '10.png'])
        with self.assertRaises(ValueError):
            numerical._source_identity(sequence, assets, 2)
        added = sequence / '3.png'
        self.assertTrue(cv2.imwrite(str(added), np.zeros((2, 2, 3), np.uint8)))
        with self.assertRaisesRegex(ValueError, 'membership'):
            numerical._source_identity(sequence, assets, 64)
        added.unlink()
        (sequence / '2.png').unlink()
        with self.assertRaisesRegex(ValueError, 'membership'):
            numerical._source_identity(sequence, assets, 64)

    def test_source_identity_cancel_and_final_membership_fence(self):
        self.capture()
        evidence = json.loads(self.evidence_path.read_text())
        assets = {a['path']: a for a in evidence['assets']}

        def cancel():
            raise gate.CancelledError('source identity fixture')

        with self.assertRaises(gate.CancelledError):
            numerical._source_identity(self.source, assets, 64, check=cancel)
        # A valid nonzero selected range still binds the complete original source.
        for side in gate.SIDES:
            self.report['provenance'][side]['normalized_batch']['jobs'][0]['settings']['start_frame'] = 3
        for pair in self.report['pairs']:
            for sample in pair['samples'].values():
                job = sample['runs'][0]['jobs'][0]
                job['requested_settings']['start_frame'] = 3
                job['diagnostics']['effective_settings']['start_frame'] = 3
                job['diagnostics']['source']['start_frame'] = 3
                sample['warmup'][0]['requested_settings']['start_frame'] = 3
                sample['warmup'][0]['diagnostics']['effective_settings']['start_frame'] = 3
        self.policy['context_sha256'] = gate.digest(gate.context(self.report))
        self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        self.capture()
        self.assertEqual(self.evaluate()['status'], 'pass')

    def test_generic_visual_resource_fences_preserve_asset_guards(self):
        self.capture()
        captured = json.loads(self.evidence_path.read_text())
        generic = {'assets': captured['assets'], 'protected_directories': captured['protected_directories']}
        for extra in ({}, {'kind': 'regression_visual_evidence', 'bindings': []}):
            with self.subTest(producer=extra.get('kind', 'resource_inputs')):
                numerical.fence({**generic, **extra})
        # Explicitly typed numerical receipts must not silently omit required metadata.
        with self.assertRaises(KeyError):
            numerical.fence({**generic, 'kind': 'regression_numerical_evidence'})
        self.source.write_bytes(b'changed')
        for extra in ({}, {'kind': 'regression_visual_evidence', 'bindings': []}):
            with self.subTest(changed_producer=extra.get('kind', 'resource_inputs')), self.assertRaises(ValueError):
                numerical.fence({**generic, **extra})
