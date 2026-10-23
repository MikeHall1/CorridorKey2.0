import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'cv2', 'torch'))
if HAS_STACK or TYPE_CHECKING:
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    import cv2
    import numpy as np

    from corridorkey.benchmark_quality import (
        evaluate_job_quality,
        evaluate_run_quality,
        prepare_quality_templates,
    )
    from corridorkey.config import InferenceConfig
    from corridorkey.session import JobRequest


@unittest.skipUnless(HAS_STACK, 'requires CPU runtime dependencies')
class BenchmarkQualityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'local synthetic source identity; no neural inference')
        self.output = self.root / 'render'
        self.alpha = self.output / 'source_alpha_exr'
        self.alpha.mkdir(parents=True)
        self.job = JobRequest.create(self.source, self.output,
                                     InferenceConfig(output_format='exr', outputs={'alpha'}))
        self.path = self.root / 'quality.json'
        data = {'schema_version': 1, 'kind': 'alpha_sequence_evaluation', 'fps': 24,
                'source': self.source.name, 'reference_provenance': {
                    'author': 'fixture', 'origin': 'test generator', 'permitted_use': 'local test', 'split': 'synthetic'},
                'frames': []}
        for index in range(2):
            self.assertTrue(cv2.imwrite(str(self.alpha / f'alpha_{index:06d}.exr'),
                                        np.array([[0., 0.5]], dtype=np.float32)))
            np.save(self.root / f'reference{index}.npy', np.array([[0., 1.]]))
            np.save(self.root / f'baseline{index}.npy', np.array([[1., 0.]]))
            data['frames'].append({'frame': index, 'candidate': '@render',
                                   'reference': f'reference{index}.npy', 'baseline': f'baseline{index}.npy'})
        self.path.write_text(json.dumps(data))
        self.record = {'status': 'complete', 'input': str(self.source), 'output_dir': str(self.output),
                       'job_id': 'synthetic-fixture', 'diagnostics': {
                           'source': {'native_fps': 24., 'output_hw': [1, 2]}, 'frames_submitted': 2,
                           'outputs': {'alpha': str(self.alpha)}, 'output_contracts': {'alpha': {
                               'format': 'exr', 'channels': 1, 'alpha_representation': 'coverage (no transfer function)'}}}}

    def template(self):
        templates = prepare_quality_templates([str(self.path)], [self.job], manifest_dir=self.root)
        assert templates is not None
        return templates[0]

    def test_binds_actual_exr_exports_with_source_and_output_frame_identity(self):
        template = self.template()
        report = evaluate_job_quality(template, self.record)
        self.assertEqual(report['candidate']['mae']['value'], 0.25)
        self.assertEqual(report['candidate_minus_baseline']['mae'], -0.75)
        self.assertEqual(report['candidate_binding']['frames_submitted'], 2)
        self.assertTrue(all(item['path'].startswith(str(self.alpha.resolve()))
                            for item in report['assets'] if item['role'] == 'candidate'))
        self.assertNotIn(str(self.alpha), json.dumps(template))
        for file in self.alpha.iterdir():
            file.rename(file.with_name(file.name.replace('alpha_', 'alpha_source_rgb_draft_')))
        self.assertEqual(evaluate_job_quality(template, self.record)['candidate']['mae']['value'], 0.25)

    def test_rejects_mismatched_fps_contract_count_source_and_geometry(self):
        template = self.template()
        for section, key, value in [('source', 'native_fps', 25), ('source', 'output_hw', [2, 2]),
                                     (None, 'frames_submitted', 3)]:
            record = copy.deepcopy(self.record)
            target = record['diagnostics'][section] if section else record['diagnostics']
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate_job_quality(template, record)
        record = copy.deepcopy(self.record)
        record['diagnostics']['output_contracts']['alpha']['format'] = 'video'
        with self.assertRaisesRegex(ValueError, 'coverage EXR'):
            evaluate_job_quality(template, record)
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            evaluate_job_quality(template, self.record)

    def test_spatial_and_combined_drafts_bind_actual_native_exports_and_keep_provenance(self):
        template = self.template()
        spatial = {'enabled': True, 'requested_scale': 0.5, 'version': 'spatial-draft-v1',
                   'model_native_hw': [1, 1], 'output_hw': [1, 2]}
        self.record['diagnostics']['output_contracts']['alpha']['spatial_draft'] = spatial
        for suffix in ('_spatial_draft', '_source_rgb_draft_spatial_draft'):
            for path in self.alpha.iterdir():
                index = path.stem.rsplit('_', 1)[1]
                path.rename(path.with_name(f'alpha{suffix}_{index}.exr'))
            with self.subTest(suffix=suffix):
                report = evaluate_job_quality(template, self.record)
                self.assertEqual(report['candidate']['mae']['value'], 0.25)
                self.assertEqual(report['candidate']['shape_hw'], [1, 2])
                self.assertEqual(report['candidate_binding']['spatial_draft'], spatial)
                report['candidate_binding']['spatial_draft']['requested_scale'] = 0.25
                self.assertEqual(spatial['requested_scale'], 0.5)
        duplicate = self.alpha / 'alpha_spatial_draft_000000.exr'
        duplicate.write_bytes(b'duplicate output-relative frame; must not bind twice')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluate_job_quality(template, self.record)

    def test_preflight_validates_explicit_job_count_output_and_source_binding(self):
        for paths in ([], [None, None], [True]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                prepare_quality_templates(paths, [self.job], manifest_dir=self.root)
        video_job = JobRequest.create(self.source, self.output, InferenceConfig(outputs={'alpha'}))
        with self.assertRaisesRegex(ValueError, 'EXR alpha'):
            prepare_quality_templates([str(self.path)], [video_job], manifest_dir=self.root)
        data = json.loads(self.path.read_text())
        data['source'] = 'other.bin'
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'source must match'):
            self.template()

    def test_quality_failure_is_explicit_and_does_not_change_job_timing(self):
        template = self.template()
        self.record['processing_seconds'] = 10.0
        runs = [{'index': 0, 'jobs': [self.record]}]
        report = evaluate_run_quality([template], runs)
        self.assertTrue(report['success'])
        (self.alpha / 'alpha_000001.exr').unlink()
        report = evaluate_run_quality([template], runs)
        self.assertFalse(report['success'])
        self.assertEqual(report['jobs'][0]['status'], 'failed')
        self.assertEqual(self.record['processing_seconds'], 10)
        self.assertEqual(self.record['status'], 'complete')
        self.assertEqual(evaluate_run_quality([None], runs)['status'], 'unavailable')

    def test_unexpected_duplicate_or_out_of_timeline_exports_are_rejected(self):
        template = self.template()
        (self.alpha / 'extra.txt').touch()
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            evaluate_job_quality(template, self.record)
        (self.alpha / 'extra.txt').unlink()
        (self.alpha / 'alpha_source_rgb_draft_000000.exr').write_bytes(b'duplicate index')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluate_job_quality(template, self.record)

    def test_warm_benchmark_integrates_real_exr_metrics_after_all_timed_jobs(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from corridorkey import benchmark
        from corridorkey.benchmark_quality import evaluate_run_quality
        from corridorkey.session import InferenceSession
        from corridorkey.video_io import ExrSequenceWriter

        batch = self.root / 'batch.json'
        batch.write_text(json.dumps({'schema_version': 1, 'kind': 'inference_batch', 'jobs': [{
            'input': self.source.name, 'output_dir': 'requested-output',
            'settings': {'output_format': 'exr', 'outputs': ['alpha']}}]}))
        manifest_path = self.root / 'benchmark.json'
        manifest = {'schema_version': 1, 'kind': 'inference', 'batch_manifest': batch.name,
                    'model_state': 'warm_session', 'repetitions': 3, 'quality_manifests': [self.path.name]}
        manifest_path.write_text(json.dumps(manifest))
        dummy = self.root / 'test-weights.bin'
        dummy.write_bytes(b'test placeholder; no model loaded')
        cradio = self.root / 'test-cradio'
        cradio.mkdir()
        paths = SimpleNamespace(weights=dummy, foundation_stats=dummy, rvm_ckpt=dummy,
                                moge_checkpoint=dummy, cradio_dir=cradio, validate=lambda: None)
        events = []

        def render(model, job, config, writer_config, diagnostics):
            events.append('render')
            target = Path(job.output_dir) / 'source_alpha_exr'
            writer = ExrSequenceWriter(str(target), 24., (2, 1), channels=1, dtype='float', frame_name='alpha')
            try:
                writer.write_many(np.tile(np.array([0., 0.5], dtype=np.float32).reshape(1, 1, 2, 1), (2, 1, 1, 1)))
            finally:
                writer.release()
            diagnostics.outputs['alpha'] = str(target)
            diagnostics.output_contracts['alpha'] = {
                'format': 'exr', 'channels': 1, 'alpha_representation': 'coverage (no transfer function)'}
            diagnostics.source = {'native_fps': 24., 'output_hw': [1, 2]}
            diagnostics.frames_submitted = 2

        def factory(**kwargs):
            return InferenceSession(loader=lambda: object(), runner=render, disposer=lambda model: None, **kwargs)

        corrupt_output = False

        def quality(templates, runs):
            self.assertEqual(events, ['render'] * 4)  # warmup plus3 measured synthetic renders
            events.append('quality')
            if corrupt_output:
                alpha = Path(runs[0]['jobs'][0]['diagnostics']['outputs']['alpha'])
                (alpha / 'alpha_000001.exr').unlink()
            return evaluate_run_quality(templates, runs)

        # This is CPU export/report integration with a neural substitute, never
        # GPU throughput, model quality or available CUDA evidence.
        for corrupt_output in (False, True):
            events.clear()
            with patch.object(benchmark, 'ROOT', self.root), \
                    patch.object(benchmark, 'repository_identity', return_value={'scope': 'test substitute'}), \
                    patch.object(benchmark, 'environment', return_value={'hardware': {}}), \
                    patch.object(benchmark, 'command', side_effect=RuntimeError('test has no driver')), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch('corridorkey.runtime.RuntimePaths.from_package', return_value=paths), \
                    patch('corridorkey.session.InferenceSession', side_effect=factory), \
                    patch('corridorkey.benchmark_quality.evaluate_run_quality', side_effect=quality):
                report = benchmark.run(manifest_path)
            self.assertEqual(report['summary']['success'], not corrupt_output, report.get('error'))
            self.assertEqual(report['inference']['status'], 'measured')
            self.assertEqual(events, ['render'] * 4 + ['quality'])
            self.assertEqual(len(report['inference']['quality']['jobs']), 3)
            if corrupt_output:
                self.assertEqual(report['inference']['quality']['status'], 'failed')
            else:
                for entry in report['inference']['quality']['jobs']:
                    self.assertEqual(entry['report']['candidate']['mae']['value'], 0.25)
            self.assertFalse((self.root / 'requested-output').exists())
    def test_benchmark_manifest_validates_quality_field_without_requiring_runtime_imports(self):
        from corridorkey.benchmark import load_manifest

        path = self.root / 'benchmark.json'
        value = {'schema_version': 1, 'kind': 'inference', 'batch_manifest': 'batch.json',
                 'model_state': 'cold_process'}
        for setting in ([], 'quality.json', [False], ['']):
            path.write_text(json.dumps({**value, 'quality_manifests': setting}))
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                load_manifest(path)
        path.write_text(json.dumps({**value, 'quality_manifests': ['quality.json', None]}))
        self.assertEqual(load_manifest(path)['quality_manifests'], ['quality.json', None])

    def test_reference_only_accuracy_is_measured_while_comparison_is_unavailable(self):
        data = json.loads(self.path.read_text())
        for frame in data['frames']:
            frame.pop('baseline')
        self.path.write_text(json.dumps(data))
        report = evaluate_run_quality([self.template()], [{'index': 0, 'jobs': [self.record]}])
        self.assertEqual(report['status'], 'measured')
        self.assertEqual(report['jobs'][0]['comparison_status'], 'unavailable')
        self.assertEqual(report['jobs'][0]['report']['candidate']['mae']['value'], 0.25)
        mixed = evaluate_run_quality([self.template(), None], [{'index': 0, 'jobs': [self.record, self.record]}])
        self.assertEqual(mixed['status'], 'partial')

    def test_reference_hashes_are_pinned_across_repetitions(self):
        template = self.template()
        np.save(self.root / 'reference0.npy', np.array([[1., 1.]]))
        report = evaluate_run_quality([template], [{'index': i, 'jobs': [self.record]} for i in range(3)])
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(all('changed since quality preflight' in job['error']['message'] for job in report['jobs']))

    def test_invalid_export_count_is_rejected_before_allocating_a_range(self):
        record = copy.deepcopy(self.record)
        record['diagnostics']['frames_submitted'] = 10**30
        with self.assertRaisesRegex(ValueError, 'every submitted output frame'):
            evaluate_job_quality(self.template(), record)

    def colour_template(self, *, output='fg', association='straight'):
        data = {'schema_version': 1, 'kind': 'colour_sequence_evaluation', 'colour_space': 'linear_srgb_bt709',
                'fps': 24, 'candidate_output': output, 'source': self.source.name, 'background': [0., 0., 0.],
                'associations': {'candidate': association, 'reference': 'straight', 'baseline': 'straight'},
                'rgb_layouts': {'candidate_rgb': 'rgba' if output == 'cutout' else 'rgb'},
                'reference_provenance': {'author': 'fixture', 'origin': 'synthetic',
                                         'permitted_use': 'local test', 'split': 'synthetic'}, 'frames': []}
        target = self.output / f'colour_{output}'
        target.mkdir()
        for index in range(2):
            rgb = np.full((1, 2, 3), 0.25, dtype=np.float32)
            alpha = np.array([[0., 0.5]], dtype=np.float32)
            if association == 'premultiplied':
                rgb *= alpha[..., None]
            if output == 'cutout':
                rgb = np.concatenate((rgb, alpha[..., None]), axis=-1)
            prefix = output + ('_premultiplied' if association == 'premultiplied' else '')
            self.assertTrue(cv2.imwrite(str(target / f'{prefix}_{index:06d}.exr'), rgb))
            fields = {'frame': index, 'candidate_rgb': '@render', 'candidate_alpha': '@render'}
            for role, array in [('reference_rgb', np.zeros((1, 2, 3))), ('reference_alpha', np.array([[0., 1.]])),
                                 ('baseline_rgb', np.ones((1, 2, 3))), ('baseline_alpha', np.ones((1, 2))),
                                 ('foreground_mask', np.ones((1, 2), dtype=bool))]:
                name = f'{role}-{index}.npy'
                np.save(self.root / name, array)
                fields[role] = name
            data['frames'].append(fields)
        self.path.write_text(json.dumps(data))
        config = InferenceConfig(output_format='exr', outputs={'alpha', output}, cutout_linear=True,
                                 cutout_alpha_mode=association if output == 'cutout' else 'straight')
        job = JobRequest.create(self.source, self.output, config)
        templates = prepare_quality_templates([str(self.path)], [job], manifest_dir=self.root)
        assert templates is not None
        self.record['diagnostics']['outputs'][output] = str(target)
        self.record['diagnostics']['output_contracts'][output] = {
            'format': 'exr', 'channels': 4 if output == 'cutout' else 3,
            'alpha_association': association if output == 'cutout' else None,
            'rgb_representation': 'scene_linear',
            'rgb_primaries': 'assumed sRGB/BT.709; no automatic colour management'}
        return templates[0]

    def test_colour_binding_scores_actual_foreground_and_premultiplied_cutout_exports(self):
        for output, association in [('fg', 'straight'), ('cutout', 'premultiplied')]:
            template = self.colour_template(output=output, association=association)
            report = evaluate_job_quality(template, self.record)
            self.assertEqual(report['candidate']['foreground_mae']['value'], 0.25)
            self.assertEqual(report['candidate']['composite_mae']['value'], 0.0625)
            self.assertEqual(report['candidate_binding']['rgb_output'], output)
            self.assertEqual(report['candidate']['candidate_association'], association)
            directory = Path(self.record['diagnostics']['outputs'][output])
            for path in directory.iterdir():
                prefix, index = path.stem.rsplit('_', 1)
                path.rename(path.with_name(f'{prefix}_source_rgb_draft_{index}.exr'))
            self.assertEqual(evaluate_job_quality(template, self.record)['candidate']['foreground_mae']['value'], 0.25)

    def test_colour_binding_rejects_wrong_primaries_association_and_missing_outputs(self):
        template = self.colour_template()
        contract = self.record['diagnostics']['output_contracts']['fg']
        contract['rgb_primaries'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'primaries'):
            evaluate_job_quality(template, self.record)
        contract['rgb_primaries'] = 'assumed sRGB/BT.709; no automatic colour management'
        directory = Path(self.record['diagnostics']['outputs']['fg'])
        (directory / 'fg_000001.exr').unlink()
        with self.assertRaisesRegex(ValueError, 'every submitted output frame'):
            evaluate_job_quality(template, self.record)

    def test_colour_preflight_requires_requested_rgb_and_declared_association(self):
        self.colour_template(output='cutout', association='premultiplied')
        with self.assertRaisesRegex(ValueError, 'requested RGB'):
            prepare_quality_templates([str(self.path)], [self.job], manifest_dir=self.root)
        template = self.colour_template(output='fg')
        self.record['diagnostics']['output_contracts']['fg']['rgb_representation'] = 'display_coded'
        with self.assertRaisesRegex(ValueError, 'contract'):
            evaluate_job_quality(template, self.record)

    def temporal_template(self):
        from corridorkey.evaluation_temporal import FLOW_CONVENTION

        data = json.loads(self.path.read_text())
        data.update(kind='temporal_sequence_evaluation', flow_convention=FLOW_CONVENTION,
                    flow_provenance={'method': 'synthetic identity', 'origin': 'known fixture geometry',
                                     'visibility_definition': 'all pixels valid current correspondences',
                                     'independent_of_candidate': True})
        entry = data['frames'][1]
        for role, value in [('backward_flow', np.zeros((1, 2, 2))), ('forward_flow', np.zeros((1, 2, 2))),
                            ('visibility', np.ones((1, 2), dtype=bool))]:
            name = f'{role}.npy'
            np.save(self.root / name, value)
            entry[role] = name
        entry['flow_previous_frame'] = 0
        self.path.write_text(json.dumps(data))
        return self.template()

    def test_temporal_binding_preserves_spatial_accuracy_and_pins_flow_assets(self):
        template = self.temporal_template()
        report = evaluate_job_quality(template, self.record)
        self.assertEqual(report['candidate']['temporal_residual_mae']['value'], 0)
        self.assertEqual(report['candidate']['spatial']['mae']['value'], 0.25)
        self.assertEqual(report['candidate_minus_baseline']['spatial_mae'], -0.75)
        np.save(self.root / 'backward_flow.npy', np.ones((1, 2, 2)))
        with self.assertRaisesRegex(ValueError, 'changed since quality preflight'):
            evaluate_job_quality(template, self.record)

    def test_temporal_binding_requires_exclusion_of_known_runtime_cuts_and_resets(self):
        template = self.temporal_template()
        for metadata in ({'quick_key': {'effective_cuts': [1]}},
                         {'corrections': {'reset_frames': [1], 'shots': [[0, 2]]}}):
            record = copy.deepcopy(self.record)
            record['diagnostics'].update(metadata)
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, 'known shot cut/reset'):
                evaluate_job_quality(template, record)
        data = json.loads(self.path.read_text())
        for role in ('backward_flow', 'forward_flow', 'visibility', 'flow_previous_frame'):
            data['frames'][1].pop(role)
        data['frames'][1]['cut'] = True
        data.pop('flow_provenance')
        self.path.write_text(json.dumps(data))
        self.record['diagnostics']['quick_key'] = {'effective_cuts': [1]}
        report = evaluate_job_quality(self.template(), self.record)
        self.assertEqual(report['candidate']['cut_pairs'], 1)
        self.assertIsNone(report['candidate']['temporal_residual_mae']['value'])
        self.assertEqual(report['candidate_binding']['known_shot_boundaries'], [1])
