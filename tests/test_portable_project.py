"""Actual small local byte collections, moved packages and typed corrections."""
import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from corridorkey.config import InferenceConfig
from corridorkey.control import CancelledError
from corridorkey.corrections import CorrectionDocument
from corridorkey.desktop_state import load_project, save_project
from corridorkey.identities import source_identity
from corridorkey.output_paths import raw_clip_name
from corridorkey.portable_project import (
    ProjectInspection,
    ProjectPlan,
    _metadata,
    inspect_project,
    inventory_package,
    inventory_project,
    plan_project,
    publish_project,
)
from corridorkey.project_collection import admit_file
from corridorkey.session import JobRequest
from corridorkey.worker_protocol import job_to_wire
from tests.test_project_collection import windows_file_clocks


class PortableProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'named-shot' / 'frames'
        self.source.mkdir(parents=True)
        for name in ('frame10.png', 'frame2.png', 'frame1.png'):
            (self.source / name).write_bytes(name.encode())
        (self.source / 'private.env').write_bytes(b'not collected')
        self.parent = self.root / 'packages'
        self.parent.mkdir()
        self.scratch = self.root / 'scratch'
        self.scratch.mkdir()
        self.output = self.root / 'new-output'
        self.job = JobRequest.create(self.source, self.root / 'old-output', InferenceConfig(start_frame=1, num_frames=2, frame_dir_fps=30, output_format='exr'))
        self.before = job_to_wire(self.job)

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, job=None, collect_ids=()):
        return plan_project(job or self.job, 1, collect_ids=collect_ids, read_budget=10**7, copy_budget=10**7)

    def test_windows_clock_domains_metadata_reader(self):
        path = self.root / 'metadata.json'
        path.write_bytes(b'{"value":1}')
        with windows_file_clocks():
            pin = admit_file(path)
            self.assertEqual(_metadata(pin, 128, None), b'{"value":1}')
            self.assertEqual(admit_file(path).fingerprint, pin.fingerprint)

    def test_metadata_preview_no_payload_hash_missing_precise_and_models_unverified(self):
        with patch('corridorkey.portable_project.read_file', side_effect=AssertionError('no payload read')):
            report = inventory_project(self.job, 1, asset_root=self.root / 'missing-assets')
        self.assertFalse(report['ready'])
        self.assertEqual([row['relative_path'] for row in report['dependencies'][0]['files']], ['frame1.png', 'frame2.png', 'frame10.png'])
        self.assertEqual(len(report['model_references']['entries']), 5)
        missing = JobRequest.create(self.root / 'missing', self.output)
        self.assertIn('unavailable', inventory_project(missing, 0)['dependencies'][0]['status'])
        with self.assertRaises(ValueError):
            self.plan(missing)

    def test_reference_only_default_immutable_and_legacy_unchanged(self):
        plan = plan_project(self.job, 1, read_budget=1000, copy_budget=0)
        self.assertEqual(plan.summary['collect_ids'], [])
        self.assertEqual(plan.summary['copy_bytes'], 0)
        invalid_mutation: Any = plan
        with self.assertRaises(FrozenInstanceError):
            invalid_mutation.manifest_json = b'bad'
        summary = plan.summary
        summary['dependencies'].clear()
        self.assertTrue(plan.summary['dependencies'])
        package = publish_project(plan, self.parent)
        self.assertEqual({item.name for item in package.iterdir()}, {'project.json', 'COMPLETE'})
        inspection = inspect_project(package, read_budget=1000)
        result = inspection.apply(output_dir=self.output, temp_dir=self.scratch)
        self.assertEqual(result.input_path, self.job.input_path)
        self.assertEqual(inspection.preview_frame, 1)
        self.assertEqual(inspection.report['requested_job'], self.before)
        self.assertEqual(job_to_wire(self.job), self.before)
        legacy = self.root / 'legacy.json'
        save_project(legacy, self.job, 1)
        legacy_job, legacy_frame = load_project(legacy)
        self.assertEqual(job_to_wire(legacy_job), job_to_wire(self.job))
        self.assertEqual(legacy_frame, 1)
        with self.assertRaises(ValueError):
            load_project(package / 'project.json')

    def test_collect_move_preserves_full_sequence_and_clip_prefix(self):
        package = publish_project(self.plan(collect_ids=('source',)), self.parent)
        moved = self.root / 'moved'
        shutil.move(package, moved)
        shutil.rmtree(self.source.parent)
        metadata = inventory_package(moved)
        self.assertFalse(metadata['ready'])
        self.assertEqual(metadata['dependencies'][0]['file_count'], 3)
        inspection = inspect_project(moved, read_budget=1000)
        self.assertTrue(inspection.ready)
        result = inspection.apply(output_dir=self.output, temp_dir=self.scratch)
        self.assertEqual(raw_clip_name(result.input_path), 'named-shot')
        self.assertEqual(Path(result.input_path).name, 'frames')
        self.assertEqual(len(list(Path(result.input_path).iterdir())), 3)
        cfg, _ = result.configurations()
        self.assertEqual((cfg.start_frame, cfg.num_frames, cfg.frame_dir_fps), (1, 2, 30))
        self.assertFalse(any(path.name == 'private.env' for path in moved.rglob('*')))

    def test_explicit_exact_relink_and_changed_content_rejected(self):
        package = publish_project(self.plan(), self.parent)
        moved_source = self.root / 'new-location' / 'frames'
        moved_source.parent.mkdir()
        shutil.move(self.source, moved_source)
        missing = inspect_project(package, read_budget=1000)
        self.assertFalse(missing.ready)
        self.assertIn('unavailable', missing.report['dependencies'][0]['status'])
        linked = inspect_project(package, relinks={'source': moved_source}, read_budget=1000)
        self.assertTrue(linked.ready)
        (moved_source / 'frame2.png').write_bytes(b'x' * len(b'frame2.png'))
        self.assertFalse(inspect_project(package, relinks={'source': moved_source}, read_budget=1000).ready)
        with self.assertRaises(ValueError):
            linked.apply(output_dir=self.output, temp_dir=self.scratch)
        with self.assertRaises(ValueError):
            inventory_package(package, relinks={'unknown': self.source})

    def test_budgets_checked_before_payloads_and_zero_inspection_not_ready(self):
        with patch('corridorkey.portable_project.read_file', side_effect=AssertionError('admit first')):
            with self.assertRaises(ValueError):
                plan_project(self.job, 0, read_budget=0, copy_budget=0)
            with self.assertRaises(ValueError):
                plan_project(self.job, 0, collect_ids=['source'], read_budget=1000, copy_budget=0)
        package = publish_project(self.plan(), self.parent)
        with patch('corridorkey.portable_project.read_file', side_effect=AssertionError('zero budget')):
            report = inspect_project(package, read_budget=0)
        self.assertFalse(report.ready)
        self.assertIn('budget', report.report['dependencies'][0]['status'])

    def corrected_job(self, version):
        asset_dir = self.root / 'authored'
        asset_dir.mkdir(exist_ok=True)
        asset = asset_dir / 'mask.png'
        asset.write_bytes(b'mask fixture immutable bytes')
        data = {'schema_version': version, 'revision': 2, 'source': {'sha256': source_identity(self.source)['sha256'], 'start_frame': 1, 'frame_count': 2, 'fps': 30, 'width': 2, 'height': 2}, 'shots': [[0,2]], 'entries': [{'id':'artist', 'revision':0,'kind':'keep','frame':0,'asset':{'path':'mask.png','sha256':hashlib.sha256(asset.read_bytes()).hexdigest()},'strength':.75,'direction':'both','range':[0,2]}]}
        if version >= 3:
            data['tracks'] = []
        if version == 4:
            data['guidance'] = []
        doc = CorrectionDocument.parse(data, asset_dir)
        path = asset_dir / 'corrections.json'
        path.write_text(doc.payload)
        config, writer = self.job.configurations()
        config.corrections_path = str(path)
        return JobRequest.create(self.source, self.job.output_dir, config, writer), doc

    def test_corrections_all_versions_typed_relocation_and_closure(self):
        for version in range(1,5):
            with self.subTest(version=version):
                job, doc = self.corrected_job(version)
                plan = self.plan(job, ('corrections', 'source'))
                self.assertIn('correction_asset_000000', plan.summary['collect_ids'])
                package = publish_project(plan, self.parent)
                original_config, _ = job.configurations()
                assert original_config.corrections_path is not None
                original_doc = Path(original_config.corrections_path)
                original_bytes = original_doc.read_bytes()
                inspection = inspect_project(package, read_budget=10**7)
                applied = inspection.apply(output_dir=self.output, temp_dir=self.scratch)
                cfg, _ = applied.configurations()
                assert cfg.corrections_path is not None
                relocated_path = Path(cfg.corrections_path)
                relocated = CorrectionDocument.parse(json.loads(relocated_path.read_bytes()), relocated_path.parent)
                self.assertEqual(relocated.identity, doc.identity)
                self.assertEqual(relocated.to_dict()['source'], doc.to_dict()['source'])
                self.assertTrue(all(path.exists() for path in relocated.asset_paths()))
                self.assertEqual(original_doc.read_bytes(), original_bytes)
                self.assertNotEqual(str(relocated_path), str(original_doc))

    def test_moved_schema4_tracks_guidance_resources_and_exact_final_constraints(self):
        import cv2
        import numpy as np

        from corridorkey.correction_rasters import CorrectionRasters
        from corridorkey.corrections import attach_guidance, load_corrections
        from tests.test_guidance_provenance import GuidanceProvenanceTests
        from tests.test_tracked_corrections import TrackedCorrectionTests

        tracked = TrackedCorrectionTests()
        guided = GuidanceProvenanceTests()
        tracked.setUp()
        tracked.root = tracked.root.resolve()
        for entry in tracked.data['entries']:
            entry['asset']['path'] = str(Path(entry['asset']['path']).resolve())
        guided.setUp()
        try:
            for path in self.source.glob('*.png'):
                self.assertTrue(cv2.imwrite(str(path), np.full((2,4,3), 120, np.uint8)))
            tracked.data['source'].update(sha256=source_identity(self.source)['sha256'], start_frame=1, frame_count=2, fps=30.)
            tracked.data['shots'] = [[0,2]]
            tracked.data['entries'][0]['range'] = [0,2]
            for resource in guided.resources:
                resource['asset']['path'] = str(guided.root / resource['asset']['path'])
            guided.observation['source_samples'][0]['source_frame'] = 2
            guided.observation['result_asset_sha256'] = tracked.data['entries'][0]['asset']['sha256']
            origin = attach_guidance(CorrectionDocument.parse(tracked.data, tracked.root), 'anchor',
                                     guidance_id='origin', kind='clean_plate', resources=guided.resources,
                                     parameters=guided.parameters, decode=guided.decode, observation=guided.observation)
            data = tracked.tracked(data=origin.to_dict(), span=(0,2))
            # tracked() returns schema3; retain the independently generated schema4 origin.
            data['schema_version'] = 4
            document = CorrectionDocument.parse(data, tracked.root)
            doc_path = tracked.root.resolve() / 'corrections.json'
            doc_path.write_text(document.payload)
            cfg, writer = self.job.configurations()
            cfg.corrections_path = doc_path
            job = JobRequest.create(self.source, self.job.output_dir, cfg, writer)
            package = publish_project(self.plan(job, ('source','corrections')), self.parent)
            original = {path: path.read_bytes() for path in document.asset_paths()}
            moved = self.root / 'schema4-moved'
            shutil.move(package, moved)
            inspection = inspect_project(moved, read_budget=10**7)
            self.assertTrue(inspection.ready, inspection.report)
            applied = inspection.apply(output_dir=self.output, temp_dir=self.scratch)
            settings, _ = applied.configurations()
            reopened = load_corrections(settings.corrections_path)
            self.assertEqual(reopened.identity, document.identity)
            self.assertEqual(len(reopened.to_dict()['tracks'][0]['frames']), 2)
            self.assertEqual(len(reopened.to_dict()['guidance'][0]['resources']), 2)
            self.assertEqual({path: path.read_bytes() for path in original}, original)
            meta = {'native_hw': (2,4), 'start_frame': 1, 'num_frames': 2, 'native_fps': 30.}
            before, after = CorrectionRasters(document, meta), CorrectionRasters(reopened, meta)
            try:
                for frame in range(2):
                    np.testing.assert_array_equal(before.apply(np.full((2,4), .2, np.float32), frame),
                                                  after.apply(np.full((2,4), .2, np.float32), frame))
            finally:
                before.close()
                after.close()
        finally:
            tracked.doCleanups()
            guided.doCleanups()

    def test_missing_asset_and_source_mismatch_refuse_plan(self):
        job, _ = self.corrected_job(1)
        (self.root / 'authored' / 'mask.png').unlink()
        report = inventory_project(job, 0)
        self.assertIn('unavailable', report['dependencies'][-1]['status'])
        with self.assertRaises(ValueError):
            self.plan(job)
        job, _ = self.corrected_job(1)
        (self.source / 'frame1.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'source content identity'):
            self.plan(job)

    def test_publish_mutation_and_cancel_cleanup_completion_last(self):
        plan = self.plan(collect_ids=('source',))
        (self.source / 'frame1.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            publish_project(plan, self.parent)
        self.assertEqual(list(self.parent.iterdir()), [])
        plan = self.plan(collect_ids=('source',))
        def cancel_after_create():
            if list(self.parent.iterdir()):
                raise CancelledError('cancel owned collection')
        with self.assertRaises(CancelledError):
            publish_project(plan, self.parent, check=cancel_after_create)
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_filename_collisions_aliases_and_untrusted_manifest_rejected(self):
        from corridorkey.portable_project import _names
        for names in [('frame1.png', 'FRAME1.png'), ('e\u0301.png', '\u00e9.png')]:
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'collision'):
                _names([{'relative_path': name} for name in names])
        package = publish_project(self.plan(collect_ids=('source',)), self.parent)
        manifest = package / 'project.json'
        data = json.loads(manifest.read_bytes())
        data['dependencies'][0]['package_path'] = '../escape'
        manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            inventory_package(package)

    def test_final_publish_fence_and_apply_cancellation_own_cleanup(self):
        from corridorkey.project_collection import read_file as actual_read
        plan = self.plan(collect_ids=('source',))
        modified = False
        def mutate_after_copy(pin, **options):
            nonlocal modified
            result = actual_read(pin, **options)
            if options.get('destination') is not None and not modified:
                modified = True
                (self.source / 'new-frame.png').write_bytes(b'new member')
            return result
        with patch('corridorkey.portable_project.read_file', side_effect=mutate_after_copy), self.assertRaises(ValueError):
            publish_project(plan, self.parent)
        self.assertEqual(list(self.parent.iterdir()), [])
        (self.source / 'new-frame.png').unlink()
        job, _ = self.corrected_job(1)
        package = publish_project(self.plan(job), self.parent)
        inspection = inspect_project(package, read_budget=10**7)
        def cancel_after_workspace():
            if list(self.scratch.iterdir()):
                raise CancelledError('cancel after correction relocation')
        with self.assertRaises(CancelledError):
            inspection.apply(output_dir=self.output, temp_dir=self.scratch, check=cancel_after_workspace)
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertTrue((package / 'project.json').is_file())

    def test_model_refs_never_collected_and_protected_from_writes(self):
        assets = self.root / 'model-assets'
        assets.mkdir()
        model = assets / 'weight.pt'
        model.write_bytes(b'not a tensor fixture')
        plan = plan_project(self.job, 1, read_budget=1000, copy_budget=0, asset_root=assets, weights=model)
        self.assertEqual(plan.summary['copy_bytes'], 0)
        self.assertEqual(len(plan.summary['dependencies']), 1)
        package = publish_project(plan, self.parent)
        inspection = inspect_project(package, read_budget=1000)
        self.assertEqual(inspection.model_references['weights'], str(model))
        with self.assertRaises(ValueError):
            inspection.apply(output_dir=assets / 'output', temp_dir=self.scratch)
        with self.assertRaises(ValueError):
            publish_project(plan, assets)
        self.assertEqual(model.read_bytes(), b'not a tensor fixture')

    def test_strict_manifest_types_duplicate_fields_and_bounds(self):
        package = publish_project(self.plan(), self.parent)
        path = package / 'project.json'
        original = path.read_bytes()
        for transform in (
            lambda data: data.update(dependencies=[None]),
            lambda data: data['dependencies'][0].update(files=[None]),
            lambda data: data['dependencies'][0]['files'][0].update(relative_path='C:escape'),
            lambda data: data['dependencies'][0].update(file_count=True),
        ):
            data = json.loads(original)
            transform(data)
            path.write_text(json.dumps(data))
            with self.assertRaises((ValueError, TypeError)):
                inventory_package(package)
        path.write_bytes(b'{"schema_version":2,"schema_version":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            inventory_package(package)
        path.write_bytes(original)
        with patch('corridorkey.portable_project.MAX_FILES', 2):
            self.assertIn('unavailable', inventory_project(self.job, 0)['dependencies'][0]['status'])

    def test_apply_normalizes_dotdot_and_refuses_same_casefold_and_nested_locations(self):
        package = publish_project(self.plan(), self.parent)
        inspection = inspect_project(package, read_budget=1000)
        shared = self.root / 'shared'
        shared.mkdir()
        for output, scratch in (
            (self.scratch / '..' / 'shared', shared),
            (self.root / 'SHARED', shared),
            (shared / 'nested', shared),
            (shared, shared / 'nested'),
        ):
            with self.subTest(output=output, scratch=scratch), self.assertRaisesRegex(ValueError, 'separate'):
                inspection.apply(output_dir=output, temp_dir=scratch)
        with self.assertRaises(ValueError):
            inspection.apply(output_dir=self.root / 'NAMED-SHOT' / 'FRAMES' / 'output', temp_dir=self.scratch)
        applied = inspection.apply(output_dir=self.root / 'unused' / '..' / 'actual-output',
                                   temp_dir=self.scratch / 'unused' / '..')
        self.assertEqual(applied.output_dir, str(self.root / 'actual-output'))
        self.assertEqual(applied.configurations()[0].temp_dir, self.scratch)

    def test_original_duplicate_config_and_writer_json_refused_before_inventory_io(self):
        invalid = [replace(self.job, config_json='{"hann_chunk":70,' + self.job.config_json[1:]),
                   replace(self.job, writer_json='{"crf":17,' + self.job.writer_json[1:])]
        # Use an existing writer field so this exercises duplicate-key rejection.
        writer_key = next(iter(json.loads(self.job.writer_json)))
        invalid[1] = replace(self.job, writer_json=json.dumps({writer_key: json.loads(self.job.writer_json)[writer_key]})[:-1] + ',' + self.job.writer_json[1:])
        with patch('corridorkey.portable_project._scan', side_effect=AssertionError('I/O before strict validation')):
            for bad in invalid:
                with self.subTest(field=bad.config_json[:30]), self.assertRaisesRegex(ValueError, 'duplicate'):
                    inventory_project(bad, 0)
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    plan_project(bad, 0, read_budget=1000, copy_budget=0)

    def test_failed_fsync_and_cleanup_retains_only_uncommitted_unopenable_package(self):
        with patch('corridorkey.portable_project.os.fsync', side_effect=OSError('fsync fault')), patch('corridorkey.portable_project.cleanup_owned', side_effect=OSError('cleanup fault')), self.assertRaisesRegex(OSError, 'fsync fault') as caught:
            publish_project(self.plan(), self.parent)
        self.assertIn('cleanup fault', str(caught.exception.__notes__))
        retained, = self.parent.iterdir()
        self.assertTrue((retained / 'project.json').is_file())
        self.assertFalse((retained / 'COMPLETE').exists())
        with self.assertRaises(FileNotFoundError):
            inspect_project(retained, read_budget=1000)
        with self.assertRaises(FileNotFoundError):
            inventory_package(retained)

    def test_precommit_cancel_and_marker_conflict_no_overwrite(self):
        def cancel_after_metadata():
            for path in self.parent.iterdir():
                if (path / 'project.json').exists():
                    raise CancelledError('cancel before commit')
        with patch('corridorkey.portable_project.cleanup_owned', side_effect=OSError('retained')), self.assertRaises(CancelledError):
            publish_project(self.plan(), self.parent, check=cancel_after_metadata)
        retained, = self.parent.iterdir()
        self.assertFalse((retained / 'COMPLETE').exists())
        with self.assertRaises(FileNotFoundError):
            inventory_package(retained)
        shutil.rmtree(retained)
        original = __import__('os').open
        def conflicting(path, flags, *args, **kwargs):
            if Path(path).name == 'COMPLETE':
                Path(path).write_bytes(b'conflicting marker never overwritten')
            return original(path, flags, *args, **kwargs)
        with patch('corridorkey.portable_project.os.open', side_effect=conflicting), self.assertRaises(FileExistsError):
            publish_project(self.plan(), self.parent)
        self.assertFalse(list(self.parent.iterdir()))

    def test_marker_type_alias_and_postinspection_mutation_fenced(self):
        package = publish_project(self.plan(), self.parent)
        inspection = inspect_project(package, read_budget=1000)
        marker = package / 'COMPLETE'
        marker.write_bytes(b'not empty')
        with self.assertRaises(ValueError):
            inventory_package(package)
        with self.assertRaises(ValueError):
            inspection.apply(output_dir=self.output, temp_dir=self.scratch)
        marker.unlink()
        other = self.root / 'empty'
        other.write_bytes(b'')
        marker.symlink_to(other)
        with self.assertRaises(ValueError):
            inventory_package(package)
        marker.unlink()
        marker.write_bytes(b'')
        with self.assertRaises(ValueError):
            inspection.apply(output_dir=self.output, temp_dir=self.scratch)

    def test_apply_fsync_error_no_return_and_primary_cleanup_error_preserved(self):
        job, _ = self.corrected_job(1)
        package = publish_project(self.plan(job), self.parent)
        inspection = inspect_project(package, read_budget=10**7)
        with patch('corridorkey.portable_project.os.fsync', side_effect=OSError('apply fsync fault')), patch('corridorkey.portable_project.cleanup_owned', side_effect=OSError('apply cleanup fault')), self.assertRaisesRegex(OSError, 'apply fsync fault') as caught:
            inspection.apply(output_dir=self.output, temp_dir=self.scratch)
        self.assertIn('apply cleanup fault', str(caught.exception.__notes__))
        retained, = self.scratch.iterdir()
        self.assertTrue((retained / '.corrections.pending').is_file())
        self.assertFalse((retained / 'corrections.json').exists())
        self.assertTrue(inspection.ready)
        self.assertEqual(job_to_wire(job)['output_dir'], self.job.output_dir)

    def test_apply_write_protection_and_classes_rejected_before_io(self):
        package = publish_project(self.plan(), self.parent)
        inspection = inspect_project(package, read_budget=1000)
        for destination in (self.source, self.source / 'nested', package / 'nested'):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                inspection.apply(output_dir=destination, temp_dir=self.scratch)
        invalid_plan: Any = ProjectPlan
        invalid_job: Any = JobRequest
        invalid_inspection: Any = ProjectInspection
        with patch('corridorkey.portable_project._inventory', side_effect=AssertionError('no I/O')), self.assertRaises(TypeError):
            publish_project(invalid_plan, self.parent)
        with self.assertRaises(TypeError):
            inventory_project(invalid_job, 0)
        with self.assertRaises(TypeError):
            ProjectInspection.apply(invalid_inspection, output_dir=self.output, temp_dir=self.scratch)
