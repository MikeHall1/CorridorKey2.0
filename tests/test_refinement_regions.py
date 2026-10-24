"""Authored final-only region mechanics; not neural quality/performance proof."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.correction_rasters import CorrectionRasters
from corridorkey.corrections import (
    CorrectionDocument,
    load_corrections,
    save_corrections,
)
from corridorkey.refinement import finish_alpha


class RefinementRegionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.shape = (7, 9)
        self.meta = {'native_hw': self.shape, 'start_frame': 12, 'num_frames': 3, 'native_fps': 24.}

    def document(self, masks, *, strengths=None):
        entries = []
        for index, mask in enumerate(masks):
            path = self.root / f'region{index}.png'
            self.assertTrue(cv2.imwrite(str(path), mask))
            entries.append({'id': str(index), 'revision': 1, 'frame': 0, 'kind': 'region',
                            'purpose': 'refinement', 'target': 'edge_finishing', 'direction': 'both',
                            'strength': strengths[index] if strengths else 1., 'range': [0, 2],
                            'asset': {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}})
        return CorrectionDocument.parse({'schema_version': 5, 'revision': 1,
            'source': {'sha256': 'a' * 64, 'start_frame': 12, 'frame_count': 3, 'fps': 24., 'width': 9, 'height': 7},
            'shots': [[0, 3]], 'entries': entries, 'tracks': [], 'guidance': []}, self.root)

    def test_regions_are_not_constraints_hints_temporal_effects_or_anchor_frames(self):
        doc = self.document([np.full(self.shape, 255, np.uint8)])
        rasters = CorrectionRasters(doc, self.meta)
        self.addCleanup(rasters.close)
        self.assertEqual(doc.frames, ())
        self.assertFalse(rasters.has_effect(0))
        self.assertFalse(rasters.has_effect(0, temporal=True))
        self.assertFalse(rasters.has_condition(0))
        self.assertEqual(rasters.condition(0), (None, None))
        alpha = np.full(self.shape, .25, np.float32)
        np.testing.assert_array_equal(rasters.apply(alpha, 0), alpha)
        region = rasters.refinement_region(0)
        assert region is not None
        np.testing.assert_array_equal(region, np.ones(self.shape))
        self.assertIsNone(rasters.refinement_region(2))

    def test_exact_outside_identity_fraction_overlap_order_and_disabled(self):
        mask = np.zeros(self.shape, np.uint8)
        mask[:, 3:6] = 255
        second = np.full(self.shape, 128, np.uint8)
        doc = self.document([mask, second], strengths=[1., .5])
        rasters = CorrectionRasters(doc, self.meta)
        self.addCleanup(rasters.close)
        weight = rasters.refinement_region(0)
        assert weight is not None
        np.testing.assert_array_equal(weight, np.maximum(mask / np.float32(255), second / np.float32(255) * .5))
        data = doc.to_dict()
        data['entries'].reverse()
        other = CorrectionRasters(CorrectionDocument.parse(data, self.root), self.meta)
        self.addCleanup(other.close)
        other_weight = other.refinement_region(0)
        assert other_weight is not None
        np.testing.assert_array_equal(other_weight, weight)
        alpha = np.random.default_rng(7).random(self.shape, dtype=np.float32)
        full = finish_alpha(alpha, expansion=1, feather=2)
        for region in (np.zeros(self.shape, np.float32), np.ones(self.shape, np.float32), mask / np.float32(255), weight):
            result = finish_alpha(alpha, expansion=1, feather=2, region=region)
            np.testing.assert_array_equal(result[region == 0], alpha[region == 0])
            np.testing.assert_array_equal(result[region == 1], full[region == 1])
            fractional = (region > 0) & (region < 1)
            np.testing.assert_allclose(result[fractional], (alpha * (1 - region) + full * region)[fractional], atol=0, rtol=0)
        data['entries'][0]['strength'] = data['entries'][1]['strength'] = 0
        disabled = CorrectionRasters(CorrectionDocument.parse(data, self.root), self.meta)
        self.addCleanup(disabled.close)
        self.assertIsNone(disabled.refinement_region(0))

    def test_schema_scope_migration_identity_relocation_and_unknown_kind(self):
        doc = self.document([np.zeros(self.shape, np.uint8)])
        for key, value in [('purpose', 'neural'), ('target', 'alpha_refine'), ('direction', 'forward')]:
            data = copy.deepcopy(doc.to_dict())
            data['entries'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CorrectionDocument.parse(data, self.root)
        data = doc.to_dict()
        data['schema_version'] = 4
        with self.assertRaises(ValueError):
            CorrectionDocument.parse(data, self.root)
        destination = self.root / 'nested' / 'saved.json'
        destination.parent.mkdir()
        save_corrections(destination, doc)
        restored = load_corrections(destination)
        self.assertEqual(restored.identity, doc.identity)
        self.assertEqual(restored.asset_paths(), doc.asset_paths())
        rasters = CorrectionRasters(doc, self.meta)
        self.addCleanup(rasters.close)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            rasters._operation(np.zeros(self.shape, np.float32), {**data['entries'][0], 'kind': 'future'})

    def test_region_admission_before_decode_hash_geometry_bytes_and_alias(self):
        from corridorkey.trimap_refinement import load_coverage_asset

        doc = self.document([np.full(self.shape, 127, np.uint8)])
        path = doc.asset_paths()[0]
        loaded = load_coverage_asset(path, expected_hw=self.shape)
        np.testing.assert_array_equal(loaded.coverage, np.full(self.shape, 127 / 255, dtype=np.float32))
        with patch.object(cv2, 'imdecode', side_effect=AssertionError('decoder ran before admission')):
            for kwargs in ({'expected_hw': (3, 3)}, {'expected_hw': self.shape, 'expected_sha256': '0' * 64}):
                with self.assertRaises(ValueError):
                    load_coverage_asset(path, **kwargs)
            with patch('corridorkey.trimap_refinement.MAX_ENCODED_BYTES', 1), self.assertRaises(ValueError):
                load_coverage_asset(path, expected_hw=self.shape)
        alias = self.root / 'alias.png'
        alias.symlink_to(path)
        with self.assertRaises((OSError, ValueError)):
            load_coverage_asset(alias, expected_hw=self.shape)
        for bad in (np.zeros((1, 1)), np.full(self.shape, np.nan), np.full(self.shape, 1.1)):
            with self.assertRaises(ValueError):
                finish_alpha(np.zeros(self.shape), expansion=1, region=bad)

    def test_editor_brush_soft_import_view_save_undo_and_no_kind_conversion(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.session import JobRequest

        source = self.root / 'source'
        source.mkdir()
        for frame in range(3):
            self.assertTrue(cv2.imwrite(str(source / f'{frame}.png'), np.full((*self.shape, 3), 110, np.uint8)))
        job = JobRequest.create(source, self.root / 'output', InferenceConfig(output_format='exr'))
        editor = CorrectionEditor(job, self.root)
        old = editor.document.payload
        mask = np.zeros(self.shape, np.uint8)
        mask[1:4, 2:6] = 255
        editor.commit_brush(mask, kind='region', frame=0, strength=1., direction='both', span=[0, 2])
        data = editor.document.to_dict()
        self.assertEqual(data['schema_version'], 5)
        self.assertEqual((data['tracks'], data['guidance']), ([], []))
        identifier = data['entries'][0]['id']
        with self.assertRaisesRegex(ValueError, 'reinterpreted'):
            editor.edit(identifier, {'kind': 'remove'})
        view, overlay = editor.view(0, identifier=identifier)
        self.assertEqual((view.height, view.width), self.shape)
        assert overlay is not None
        np.testing.assert_array_equal(overlay, mask)
        saved = editor.save()
        self.assertEqual(load_corrections(saved, source=source).identity, editor.document.identity)
        original = saved.read_bytes()
        editor.undo()
        self.assertEqual(editor.document.payload, old)
        editor.redo()
        self.assertEqual(editor.document.to_dict()['entries'][0]['id'], identifier)
        soft = self.root / 'soft.png'
        self.assertTrue(cv2.imwrite(str(soft), np.full(self.shape, 18000, np.uint16)))
        editor.import_mask(soft, kind='region', frame=1, strength=.4, direction='both', span=[1, 2])
        self.assertEqual(saved.read_bytes(), original)
        self.assertIsNone(job.configurations()[0].corrections_path)
        reopened = CorrectionEditor(job, self.root, existing=editor.save())
        self.assertEqual(reopened.document.identity, editor.document.identity)

    def test_region_dependencies_collect_protect_and_invalidate_replay(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.output_paths import validate_report_destination
        from corridorkey.portable_project import (
            inspect_project,
            plan_project,
            publish_project,
        )
        from corridorkey.recovery_identity import ASSETS, job_identity
        from corridorkey.session import JobRequest
        from corridorkey.stage2_identity import stage2_finishing_context

        source = self.root / 'source'
        source.mkdir()
        for frame in range(3):
            self.assertTrue(cv2.imwrite(str(source / f'{frame}.png'), np.full((*self.shape, 3), 100, np.uint8)))
        cfg = InferenceConfig(output_format='exr', auto_vram=False)
        editor = CorrectionEditor(JobRequest.create(source, self.root / 'out', cfg), self.root)
        editor.commit_brush(np.full(self.shape, 255, np.uint8), kind='region', frame=0,
                            strength=1., direction='both', span=[0, 3])
        cfg.corrections_path = editor.save()
        original = editor.document.identity
        job = JobRequest.create(source, self.root / 'out', cfg)
        region_path = editor.document.asset_paths()[0]
        before = region_path.read_bytes()
        with self.assertRaises(ValueError):
            validate_report_destination(region_path, [job])
        self.assertEqual(region_path.read_bytes(), before)
        packages = self.root / 'packages'
        packages.mkdir()
        plan = plan_project(job, 0, collect_ids=('source', 'corrections'), read_budget=1024**2, copy_budget=1024**2)
        package = publish_project(plan, packages)
        (self.root / 'collected-output').mkdir()
        (self.root / 'scratch').mkdir()
        collected = inspect_project(package, read_budget=1024**2).apply(
            output_dir=self.root / 'collected-output', temp_dir=self.root / 'scratch')
        collected_path = collected.configurations()[0].corrections_path
        assert collected_path is not None
        self.assertEqual(load_corrections(collected_path, source=collected.input_path).identity, original)
        implementation = self.root / 'implementation'
        (implementation / 'corridorkey').mkdir(parents=True)
        (implementation / 'corridorkey/runtime.py').write_text('# identity fixture only\n')
        assets = {}
        for name in ASSETS:
            path = self.root / f'{name}.bin'
            path.write_bytes(b'identity fixture; never load')
            assets[name] = path
        environment = {'backend': 'cpu-fixture', 'precision': 'float32',
                       'dependencies': {'fixture': '1'}, 'runtime': {'fixture': True}}
        old = job_identity(job, assets=assets, environment=environment, implementation_root=implementation)
        editor.edit(editor.document.to_dict()['entries'][0]['id'], {'strength': .5})
        cfg.corrections_path = editor.save()
        changed = job_identity(JobRequest.create(source, self.root / 'new-output', cfg), assets=assets,
                               environment=environment, implementation_root=implementation)
        self.assertNotEqual(old.key, changed.key)
        self.assertEqual(stage2_finishing_context(old).key, stage2_finishing_context(changed).key)
        from corridorkey.recovery_identity import upstream_finishing_context
        self.assertNotEqual(upstream_finishing_context(old).key, upstream_finishing_context(changed).key)

    def test_v5_guidance_attachment_preserves_regions_and_legacy_versions(self):
        from tests.test_guidance_provenance import GuidanceProvenanceTests

        fixture = GuidanceProvenanceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for version in (1, 2, 3, 4):
            data = copy.deepcopy(fixture.data)
            data['schema_version'] = version
            if version >= 3:
                data['tracks'] = []
            if version >= 4:
                data['guidance'] = []
            self.assertEqual(CorrectionDocument.parse(data, fixture.root).to_dict(), data)
        data['schema_version'] = 5
        data['entries'].append({**data['entries'][0], 'id': 'region', 'kind': 'region',
                                'purpose': 'refinement', 'target': 'edge_finishing'})
        document = CorrectionDocument.parse(data, fixture.root)
        guided = fixture.attach(document)
        self.assertEqual(guided.to_dict()['schema_version'], 5)
        self.assertEqual(guided.to_dict()['entries'][1], data['entries'][1])

    def test_actual_export_all_paths_regions_are_final_only_and_aligned(self):
        import torch

        from corridorkey import draft
        from corridorkey.identities import source_identity
        from corridorkey.motion import Propagation
        from tests import test_spatial_draft as fixture

        helper = fixture.SpatialDraftTests()
        helper.root = self.root
        source = helper.source(exr=True)
        preserved = {path: path.read_bytes() for path in source.iterdir()}
        mask = np.zeros((65, 97), np.uint8)
        mask[:, 15:60] = 255
        path = self.root / 'runtime-region.png'
        self.assertTrue(cv2.imwrite(str(path), mask))
        document = CorrectionDocument.parse({'schema_version': 5, 'revision': 1,
            'source': {'sha256': source_identity(source)['sha256'], 'start_frame': 0, 'frame_count': 5,
                       'fps': 24., 'width': 97, 'height': 65}, 'shots': [[0, 5]], 'tracks': [], 'guidance': [],
            'entries': [{'id': 'region', 'revision': 1, 'frame': 1, 'kind': 'region', 'purpose': 'refinement',
                         'target': 'edge_finishing', 'direction': 'both', 'strength': .5, 'range': [1, 4],
                         'asset': {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}]}, self.root)
        manifest = self.root / 'runtime-regions.json'
        save_corrections(manifest, document)
        offset_data = document.to_dict()
        offset_data['source'].update(start_frame=1, frame_count=4)
        offset_data['shots'] = [[0, 4]]
        offset_manifest = self.root / 'runtime-offset.json'
        save_corrections(offset_manifest, CorrectionDocument.parse(offset_data, self.root))
        factory, original_frames = fixture.make_spatial_cpu_model, draft.offline_frames

        def patterned_model(**kwargs):
            model = factory(**kwargs)
            original = model.stage2.forward
            def forward(rgb, baton):
                logits = original(rgb, baton)
                values = torch.arange(logits.shape[-1]).remainder(13).float() / 12
                logits[:, :, 0] = values.view(1, 1, 1, -1)
                return logits
            model.stage2.forward = forward
            return model

        def uncertain(*args, **kwargs):
            alpha = np.tile(np.arange(97, dtype=np.float32) % 13 / 12, (65, 1))
            return Propagation(alpha, np.zeros_like(alpha), np.ones_like(alpha, dtype=bool),
                               {'flow_evaluations': 4, 'motion_seconds': 0., 'uncertain_fraction': 1., 'motion_fraction': 0.})

        def forced(*args, **kwargs):
            return original_frames(*args, interpolate=uncertain, **kwargs)

        with patch.object(fixture, 'make_spatial_cpu_model', patterned_model), patch.object(draft, 'offline_frames', forced):
            for scale, low, quick, start in ((1., False, False, 0), (1., True, False, 0), (.5, True, False, 0),
                                             (1., False, True, 0), (.5, True, True, 1)):
                with self.subTest(scale=scale, low=low, quick=quick, start=start):
                    kwargs = {'scale': scale, 'low': low, 'quick': quick, 'refine': True,
                              'outputs': {'alpha', 'fg', 'cutout'},
                              'overrides': {'cutout_alpha_mode': 'premultiplied', 'start_frame': start}}
                    baseline_model, baseline, _ = helper.run_job(source, **kwargs)
                    kwargs['overrides'].update(alpha_expand_radius=1, alpha_feather_radius=2)
                    model, actual, _ = helper.run_job(source, correction=offset_manifest if start else manifest, **kwargs)
                    assert actual.alpha_refinement is not None
                    self.assertEqual(actual.alpha_refinement['edge_finishing']['frames_evaluated'], 5 - start)
                    self.assertEqual(model.stage2.shapes, baseline_model.stage2.shapes)
                    self.assertEqual(len(model.stage1.seen_hints), len(baseline_model.stage1.seen_hints))
                    for a, b in zip(model.stage1.seen_hints, baseline_model.stage1.seen_hints):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    for index, (a, b) in enumerate(zip(sorted(Path(baseline.outputs['alpha']).glob('*.exr')),
                                                       sorted(Path(actual.outputs['alpha']).glob('*.exr')))):
                        raw = cv2.imread(str(a), cv2.IMREAD_UNCHANGED)
                        result = cv2.imread(str(b), cv2.IMREAD_UNCHANGED)
                        region = mask.astype(np.float32) / 255 * .5 if 1 <= index < 4 else None
                        np.testing.assert_array_equal(result, finish_alpha(raw, expansion=1, feather=2, region=region))
                    for a, b in zip(sorted(Path(baseline.outputs['fg']).glob('*.exr')),
                                    sorted(Path(actual.outputs['fg']).glob('*.exr'))):
                        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(preserved, {path: path.read_bytes() for path in source.iterdir()})
