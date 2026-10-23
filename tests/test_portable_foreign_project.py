"""Moved Windows archival packages on POSIX; no native Windows/runtime claim."""
import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.control import CancellationToken, CancelledError
from corridorkey.correction_rasters import CorrectionRasters
from corridorkey.corrections import (
    CorrectionDocument,
    attach_guidance,
    load_corrections,
)
from corridorkey.identities import source_identity
from corridorkey.output_paths import raw_clip_name
from corridorkey.portable_project import (
    ProjectPlan,
    inspect_project,
    inventory_package,
    inventory_project,
    plan_project,
    publish_project,
)
from corridorkey.project_collection import FilePin
from corridorkey.scheduling import frame_paths
from corridorkey.session import JobRequest


def _write_manifest(package, data):
    (package / 'project.json').write_text(json.dumps(data, sort_keys=True, allow_nan=False))


def make_foreign_collected_package(root: Path) -> tuple[Path, dict]:
    """Actual local payloads with Windows archival paths, for focused Qt tests.

    Model references and accepted-guidance producers are provenance fixtures,
    never pretrained-model compatibility or native-Windows execution evidence.
    """
    from tests.test_guidance_provenance import GuidanceProvenanceTests
    from tests.test_tracked_corrections import TrackedCorrectionTests

    root = root.resolve()
    source = root / 'named-shot' / 'frames'
    hints = root / 'hint-shot' / 'frames'
    authored = root / 'authored'
    for directory in (source, hints, authored, root / 'packages'):
        directory.mkdir(parents=True)
    for index, name in enumerate(('frame1.png', 'frame2.png', 'frame10.png')):
        if not cv2.imwrite(str(source / name), np.full((2, 4, 3), 80 + index * 20, np.uint8)):
            raise AssertionError('Source PNG fixture write failed')
        if not cv2.imwrite(str(hints / name), np.full((2, 4), 100 + index, np.uint8)):
            raise AssertionError('Hint PNG fixture write failed')
    first = root / 'hint-first.png'
    if not cv2.imwrite(str(first), np.full((2, 4), 180, np.uint8)):
        raise AssertionError('First hint fixture write failed')
    tracked, guided = TrackedCorrectionTests(), GuidanceProvenanceTests()
    tracked.setUp()
    guided.setUp()
    try:
        for entry in tracked.data['entries']:
            old = Path(entry['asset']['path'])
            target = authored / old.name
            target.write_bytes(old.read_bytes())
            entry['asset']['path'] = str(target)
        tracked.root = authored
        tracked.data['source'].update(sha256=source_identity(source)['sha256'], start_frame=1, frame_count=2, fps=30.)
        tracked.data['shots'] = [[0, 2]]
        tracked.data['entries'][0]['range'] = [0, 2]
        for resource in guided.resources:
            old = guided.root / resource['asset']['path']
            target = authored / old.name
            target.write_bytes(old.read_bytes())
            resource['asset']['path'] = str(target)
        guided.observation['source_samples'][0]['source_frame'] = 2
        guided.observation['result_asset_sha256'] = tracked.data['entries'][0]['asset']['sha256']
        origin = attach_guidance(CorrectionDocument.parse(tracked.data, authored), 'anchor', guidance_id='origin',
                                 kind='clean_plate', resources=guided.resources, parameters=guided.parameters,
                                 decode=guided.decode, observation=guided.observation)
        data = tracked.tracked(data=origin.to_dict(), span=(0, 2))
        data['schema_version'] = 4
        document = CorrectionDocument.parse(data, authored)
        correction_file = authored / 'corrections.json'
        correction_file.write_text(document.payload)
    finally:
        tracked.doCleanups()
        guided.doCleanups()
    job = JobRequest.create(source, root / 'original-output',
                            InferenceConfig(start_frame=1, num_frames=2, frame_dir_fps=30., output_format='exr',
                                            exr_dtype='float', corrections_path=correction_file,
                                            temp_dir=root / 'original-temp', despill_strength=.31),
                            WriterConfig(ffmpeg_bin='/usr/bin/ffmpeg'), initial_hint_path=first, hint_video_path=hints)
    inventory = inventory_project(job, 1)
    ids = tuple(dep['id'] for dep in inventory['dependencies'])
    plan = plan_project(job, 1, collect_ids=ids, read_budget=10**7, copy_budget=10**7)
    package = publish_project(plan, root / 'packages')
    manifest = json.loads((package / 'project.json').read_bytes())
    mapping = {dep['id']: dep for dep in manifest['dependencies']}
    for dep in manifest['dependencies']:
        if dep['id'] == 'source':
            dep['path'] = r'C:\Archive\named-shot\frames'
        elif dep['id'] == 'full_hint':
            dep['path'] = r'C:\Archive\hint-shot\frames'
        elif dep['id'] == 'initial_hint':
            dep['path'] = r'C:\Archive\hint-first.png'
        elif dep['id'] == 'corrections':
            dep['path'] = r'C:\Archive\authored\corrections.json'
        else:
            dep['path'] = 'C:\\Archive\\resources\\' + dep['id'] + '\\' + dep['files'][0]['relative_path']
    wire = manifest['original_job']
    wire['input_path'], wire['output_dir'] = mapping['source']['path'], r'D:\Exports\original-output'
    wire['initial_hint_path'], wire['hint_video_path'] = mapping['initial_hint']['path'], mapping['full_hint']['path']
    config = json.loads(wire['config_json'])
    config.update(temp_dir=r'E:\Scratch\original-temp', corrections_path=mapping['corrections']['path'])
    wire['config_json'] = json.dumps(config, sort_keys=True)
    writer = json.loads(wire['writer_json'])
    writer['ffmpeg_bin'] = r'C:\Tools\ffmpeg.exe'
    wire['writer_json'] = json.dumps(writer, sort_keys=True)
    foreign_document = manifest['corrections']['document']
    for asset, identifier in zip(CorrectionDocument.parse(foreign_document).asset_bindings(),
                                 manifest['corrections']['asset_ids'], strict=True):
        # asset_bindings is detached: mutate the matching typed document positions below.
        if asset['sha256'] != mapping[identifier]['files'][0]['sha256']:
            raise AssertionError('Fixture resource hash closure differs')
    from corridorkey.portable_project import _assets
    for asset, identifier in zip(_assets(foreign_document), manifest['corrections']['asset_ids'], strict=True):
        asset['path'] = mapping[identifier]['path']
    if CorrectionDocument.parse(foreign_document).identity != document.identity:
        raise AssertionError('Archival resource path relocation changed semantic identity')
    correction_dep = mapping['corrections']
    raw = json.dumps(foreign_document, sort_keys=True, separators=(',', ':')).encode()
    correction_payload = package / correction_dep['package_path'] / correction_dep['files'][0]['relative_path']
    correction_payload.write_bytes(raw)
    correction_dep['bytes'] = len(raw)
    correction_dep['files'][0].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    refs = manifest['model_references']
    refs['asset_root'], refs['weights'] = r'C:\Models', r'D:\Weights\corridorkey.safetensors'
    for row in refs['entries']:
        row['path'] = 'C:\\Models\\' + row['id']
    _write_manifest(package, manifest)
    originals = {path: path.read_bytes() for directory in (source, hints, authored) for path in directory.iterdir() if path.is_file()}
    originals[first] = first.read_bytes()
    return package, {'original_wire': copy.deepcopy(wire), 'correction_identity': document.identity,
                     'preview_frame': 1, 'originals': originals, 'original_document': document,
                     'source': source, 'hints': hints, 'job': job}


@unittest.skipIf(os.name == 'nt', 'Windows archival packages on a POSIX receiver; native Windows NOT VERIFIED')
class PortableForeignProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.package, self.metadata = make_foreign_collected_package(self.root)
        self.output = self.root / 'fresh-output'
        self.scratch = self.root / 'fresh-scratch'
        self.scratch.mkdir()
        self.before_package = {path.relative_to(self.package): path.read_bytes()
                               for path in self.package.rglob('*') if path.is_file()}

    def tearDown(self):
        self.assertEqual({path: path.read_bytes() for path in self.metadata['originals']}, self.metadata['originals'])
        self.temp.cleanup()

    def inspect(self):
        result = inspect_project(self.package, read_budget=10**7)
        self.assertTrue(result.ready, result.report)
        return result

    def test_public_inspection_records_reject_forged_metadata_and_pins_before_io(self):
        inspection = self.inspect()
        report = inspection.report
        cases = []
        for ready in (1, 'yes', [], None):
            changed = copy.deepcopy(report)
            changed['ready'] = ready
            cases.append(replace(inspection, report_json=json.dumps(changed).encode()))
        for field, value in (('path', r'C:\Archive\foreign'), ('depends_on', ['unknown']),
                             ('kind', 'file'), ('status', 'available'), ('required', 1)):
            changed = copy.deepcopy(report)
            changed['dependencies'][0][field] = value
            cases.append(replace(inspection, report_json=json.dumps(changed).encode()))
        changed = copy.deepcopy(report)
        changed['preview_frame'] = float(changed['preview_frame'])
        cases.append(replace(inspection, report_json=json.dumps(changed).encode()))
        pins = inspection.pins
        first = pins[0][1][0]
        for bad in (FilePin, replace(first, path=r'C:\Archive\frame.png'),
                    replace(first, fingerprint=(0, 0, -1, 0, 0)),
                    replace(first, fingerprint=(0, 0, True, 0, 0)),
                    replace(first, sha256='0' * 64)):
            cases.append(replace(inspection, pins=((pins[0][0], (bad, *pins[0][1][1:])), *pins[1:])))
        cases.extend((replace(inspection, pins=set()), replace(inspection, pins=pins[:-1]),
                      replace(inspection, pins=(("wrong", pins[0][1]), *pins[1:])),
                      replace(inspection, package_path=r'C:\Archive\package'),
                      replace(inspection, manifest_pin=FilePin),
                      replace(inspection, manifest_pin=replace(inspection.manifest_pin, path=r'C:\Archive\project.json')),
                      replace(inspection, marker_pin=replace(inspection.marker_pin, path=str(self.package / 'other'))),
                      replace(inspection, marker_pin=replace(inspection.marker_pin, fingerprint=(0, 0, 1, 0, 0)))))
        for index, forged in enumerate(cases):
            with self.subTest(index=index), patch('corridorkey.portable_project.assert_pin', side_effect=AssertionError('no pin IO')), patch('corridorkey.portable_project._scan', side_effect=AssertionError('no scan')), patch('corridorkey.portable_project.real_path', side_effect=AssertionError('no native admission')), self.assertRaises((TypeError, ValueError)):
                forged.apply(output_dir=self.output, temp_dir=self.scratch)
        self.assertEqual(list(self.scratch.iterdir()), [])

    def test_consistent_detached_manifest_and_report_forgery_refused_without_artifacts(self):
        inspection = self.inspect()
        data = json.loads(inspection.manifest_json)
        report = inspection.report
        config = json.loads(data['original_job']['config_json'])
        self.assertEqual(config['model_scale'], 1)
        config['model_scale'] = 0.5
        data['original_job']['config_json'] = json.dumps(config)
        report['requested_job'] = copy.deepcopy(data['original_job'])
        forged = replace(inspection, manifest_json=json.dumps(data).encode(), report_json=json.dumps(report).encode())
        before = set(self.root.iterdir())
        with patch('corridorkey.portable_project._protect', side_effect=AssertionError('must authenticate package first')), self.assertRaisesRegex(ValueError, 'pinned package manifest'):
            forged.apply(output_dir=self.output, temp_dir=self.scratch)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertEqual(json.loads((self.package / 'project.json').read_bytes()), json.loads(inspection.manifest_json))

    def test_native_publication_pin_association_rejects_before_destination_io(self):
        job = self.metadata['job']
        plan = plan_project(job, 0, read_budget=10**7, copy_budget=0)
        first_id, first_pins = plan.pins[0]
        for pins in (set(), plan.pins[:-1],
                     ((first_id, (FilePin, *first_pins[1:])), *plan.pins[1:]),
                     ((first_id, (replace(first_pins[0], path=r'C:\Archive\foreign.png'), *first_pins[1:])), *plan.pins[1:])):
            with self.subTest(pins_type=type(pins)), patch('corridorkey.portable_project.real_path', side_effect=AssertionError('no destination IO')), patch('corridorkey.portable_project.assert_pin', side_effect=AssertionError('no pin IO')), self.assertRaises((TypeError, ValueError)):
                publish_project(replace(plan, pins=pins), self.root)

    def test_actual_moved_all_collected_windows_package_apply_offsets_hints_and_schema4_rasters(self):
        moved = self.root / 'moved-package'
        shutil.move(self.package, moved)
        self.package = moved
        original_wire = copy.deepcopy(self.metadata['original_wire'])
        report = inventory_package(moved)
        self.assertEqual(report['archive_path_flavour'], 'windows')
        self.assertTrue(report['foreign_paths'])
        self.assertEqual(report['requested_job'], original_wire)
        result = self.inspect().apply(output_dir=self.output, temp_dir=self.scratch)
        config, writer = result.configurations()
        self.assertEqual((config.start_frame, config.num_frames, config.frame_dir_fps), (1, 2, 30.))
        self.assertEqual((config.exr_dtype, config.despill_strength), ('float', .31))
        self.assertEqual(writer.ffmpeg_bin, r'C:\Tools\ffmpeg.exe')  # Unused for EXR; never rewritten.
        self.assertEqual(raw_clip_name(result.input_path), raw_clip_name(self.metadata['job'].input_path))
        self.assertTrue(Path(result.input_path).is_relative_to(moved))
        assert result.initial_hint_path is not None
        assert result.hint_video_path is not None
        self.assertTrue(Path(result.initial_hint_path).is_relative_to(moved))
        self.assertTrue(Path(result.hint_video_path).is_relative_to(moved))
        self.assertEqual([Path(path).name for path in frame_paths(result.input_path)], ['frame1.png', 'frame2.png', 'frame10.png'])
        reopened = load_corrections(config.corrections_path)
        self.assertEqual(reopened.identity, self.metadata['correction_identity'])
        self.assertEqual(len(reopened.to_dict()['tracks'][0]['frames']), 2)
        self.assertEqual(len(reopened.to_dict()['guidance'][0]['resources']), 2)
        self.assertTrue(all(path.is_relative_to(moved) for path in reopened.asset_paths()))
        meta = {'native_hw': (2, 4), 'start_frame': 1, 'num_frames': 2, 'native_fps': 30.}
        before, after = CorrectionRasters(self.metadata['original_document'], meta), CorrectionRasters(reopened, meta)
        try:
            for frame in range(2):
                np.testing.assert_array_equal(before.apply(np.full((2, 4), .2, np.float32), frame),
                                              after.apply(np.full((2, 4), .2, np.float32), frame))
        finally:
            before.close()
            after.close()
        self.assertEqual({path.relative_to(moved): path.read_bytes() for path in moved.rglob('*') if path.is_file()}, self.before_package)
        self.assertEqual(inventory_package(moved)['requested_job'], original_wire)

    def test_no_foreign_reference_ever_reaches_native_io_or_job_codec_before_rebind(self):
        from corridorkey import portable_project
        original_real, original_job = portable_project.real_path, portable_project.job_from_wire

        def real(path, **kwargs):
            self.assertNotIn('C:\\', str(path))
            self.assertNotIn('D:\\', str(path))
            self.assertNotIn('E:\\', str(path))
            return original_real(path, **kwargs)

        def native(wire):
            self.assertTrue(Path(wire['input_path']).is_absolute())
            self.assertNotIn('C:\\', wire['input_path'])
            self.assertEqual(json.loads(wire['writer_json'])['ffmpeg_bin'], r'C:\Tools\ffmpeg.exe')
            return original_job(wire)

        with patch('corridorkey.portable_project.real_path', side_effect=real), patch('corridorkey.portable_project.job_from_wire', side_effect=native) as codec:
            inspection = self.inspect()
            self.assertEqual(codec.call_count, 0)
            result = inspection.apply(output_dir=self.output, temp_dir=self.scratch)
            self.assertTrue(Path(result.input_path).is_relative_to(self.package))
            self.assertEqual(codec.call_count, 1)

    def test_foreign_reference_only_or_partial_refuse_before_any_dependency_probe(self):
        for identifier in ('source', 'correction_asset_000000'):
            data = json.loads((self.package / 'project.json').read_bytes())
            dep = next(dep for dep in data['dependencies'] if dep['id'] == identifier)
            dep['collected'] = False
            dep.pop('package_path')
            _write_manifest(self.package, data)
            with patch('corridorkey.portable_project._scan', side_effect=AssertionError('must refuse before any foreign dependency scan')), self.assertRaisesRegex(ValueError, 'every render dependency'):
                inventory_package(self.package)
            _write_manifest(self.package, json.loads(self.before_package[Path('project.json')]))

    def test_foreign_plan_publication_refuses_before_native_destination_or_dependency_io(self):
        inspection = self.inspect()
        foreign = ProjectPlan(inspection.manifest_json, inspection.pins, inspection.report_json)
        destination = self.root / 'not-created'
        before = set(self.root.iterdir())
        with patch('corridorkey.portable_project.real_path', side_effect=AssertionError('no destination admission')), patch('corridorkey.portable_project._protect', side_effect=AssertionError('no foreign protection resolve')), patch('corridorkey.portable_project._fence', side_effect=AssertionError('no foreign fence')), patch('corridorkey.portable_project._scan', side_effect=AssertionError('no foreign scan')), self.assertRaisesRegex(ValueError, 'Foreign archival plans cannot be published'):
            publish_project(foreign, destination)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertFalse(destination.exists())

    def test_foreign_encoder_video_actionable_refusal_exr_preserved_and_native_destinations_required(self):
        inspection = self.inspect()
        for output, scratch in ((Path('relative'), self.scratch), (self.output, Path(r'C:\Scratch')), (Path(r'D:\Output'), self.scratch)):
            with self.subTest(output=output, scratch=scratch), self.assertRaises(ValueError):
                inspection.apply(output_dir=output, temp_dir=scratch)
            self.assertEqual(list(self.scratch.iterdir()), [])
        data = json.loads((self.package / 'project.json').read_bytes())
        config = json.loads(data['original_job']['config_json'])
        config['output_format'] = 'video'
        data['original_job']['config_json'] = json.dumps(config)
        _write_manifest(self.package, data)
        video = self.inspect()
        with self.assertRaisesRegex(ValueError, 'foreign absolute FFmpeg'):
            video.apply(output_dir=self.output, temp_dir=self.scratch)
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertFalse(self.output.exists())

    def test_malformed_archival_wire_collection_traversal_and_case_collision_refuse(self):
        original = json.loads((self.package / 'project.json').read_bytes())
        mutations = []
        duplicate = copy.deepcopy(original)
        duplicate['original_job']['config_json'] = duplicate['original_job']['config_json'][:-1] + ',"model_scale":0.5}'
        mutations.append(duplicate)
        relative = copy.deepcopy(original)
        relative['original_job']['output_dir'] = 'C:relative'
        mutations.append(relative)
        mixed = copy.deepcopy(original)
        mixed['model_references']['asset_root'] = '/tmp/native-models'
        mutations.append(mixed)
        traversal = copy.deepcopy(original)
        traversal['dependencies'][0]['package_path'] = '../escape'
        mutations.append(traversal)
        collision = copy.deepcopy(original)
        files = collision['dependencies'][0]['files']
        files[1]['relative_path'] = files[0]['relative_path'].upper()
        mutations.append(collision)
        for data in mutations:
            _write_manifest(self.package, data)
            with self.assertRaises((ValueError, TypeError)):
                inventory_package(self.package)
        _write_manifest(self.package, original)

    def test_hash_drift_cancel_and_fresh_correction_commit_preserve_all_sources(self):
        inspection = self.inspect()
        token = CancellationToken()
        token.cancel('explicit test cancellation')
        with self.assertRaises(CancelledError):
            inspection.apply(output_dir=self.output, temp_dir=self.scratch, check=token.check)
        self.assertEqual(list(self.scratch.iterdir()), [])
        with patch('corridorkey.portable_project.os.fsync', side_effect=OSError('commit sync failed')), self.assertRaises(OSError):
            inspection.apply(output_dir=self.output, temp_dir=self.scratch)
        self.assertEqual(list(self.scratch.iterdir()), [])
        data = json.loads((self.package / 'project.json').read_bytes())
        source = data['dependencies'][0]
        leaf = self.package / source['package_path'] / source['files'][0]['relative_path']
        original = leaf.read_bytes()
        leaf.write_bytes(original + b'changed')
        with self.assertRaises(ValueError):
            inspection.apply(output_dir=self.output, temp_dir=self.scratch)
        self.assertFalse(inspect_project(self.package, read_budget=10**7).ready)


if __name__ == '__main__':
    unittest.main()
