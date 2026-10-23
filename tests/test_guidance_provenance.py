"""Accepted guidance is an existing raster with immutable generation provenance."""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.corrections import (
    CorrectionDocument,
    CorrectionHistory,
    attach_guidance,
    canonical,
    guidance_binding,
    invalidated_ranges,
    load_corrections,
    save_corrections,
)


class GuidanceProvenanceTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.data = {'schema_version': 1, 'revision': 0,
                     'source': {'sha256': 'a' * 64, 'start_frame': 7, 'frame_count': 5,
                                'fps': 24., 'width': 4, 'height': 2},
                     'shots': [[0, 3], [3, 5]],
                     'entries': [{'id': 'generated', 'revision': 0, 'frame': 1, 'kind': 'matte',
                                  'asset': self.asset('mask.bin', b'accepted raster fixture'),
                                  'strength': .5, 'direction': 'both', 'range': [0, 3]}]}
        self.native_decode = {'contract': 'corridorkey-native-rgb-v1', 'source_kind': 'frames',
                              'frame_dir_linear': False, 'frame_dir_alpha': 'ignore',
                              'source_colour_profile': 'legacy', 'linear_brightness': 1.,
                              'linear_contrast': 1., 'linear_contrast_pivot': .18}
        self.decode = {'source': copy.deepcopy(self.native_decode), 'reference': copy.deepcopy(self.native_decode)}
        self.resources = [{'id': 'plate', 'role': 'reference', 'asset': self.asset('plate.bin', b'reference fixture')},
                          {'id': 'background', 'role': 'source_sample', 'asset': self.asset('background.bin', b'binary fixture')}]
        self.parameters = {'contract': 'corridorkey-guidance-v1',
                           'producer': {'id': 'fixture-not-quality-evidence', 'version': '1', 'weights_sha256': []},
                           'settings': {'threshold': .1, 'mapping': 'single_image'}}
        self.observation = {'source_samples': [{'frame': 1, 'source_frame': 8, 'rgb_sha256': 'b' * 64}],
                            'reference_samples': [{'resource_id': 'plate', 'frame': 0, 'rgb_sha256': 'c' * 64}],
                            'result_asset_sha256': self.data['entries'][0]['asset']['sha256'],
                            'diagnostics': {'quality_verified': False, 'evidence': 'persistence fixture'}}

    def asset(self, name, content):
        path = self.root / name
        path.write_bytes(content)
        return {'path': name, 'sha256': hashlib.sha256(content).hexdigest()}

    def document(self):
        return CorrectionDocument.parse(self.data, self.root)

    def attach(self, document=None, **overrides):
        options = {'guidance_id': 'origin', 'kind': 'clean_plate', 'resources': self.resources,
                   'parameters': self.parameters, 'decode': self.decode, 'observation': self.observation}
        options.update(overrides)
        return attach_guidance(document or self.document(), 'generated', **options)

    def rebind(self, data):
        origin = data['guidance'][0]
        origin['binding'] = guidance_binding(data, origin['entry_id'], kind=origin['kind'],
                                             resources=origin['resources'], parameters=origin['parameters'],
                                             decode=origin['decode'], observation=origin['observation'])
        origin['binding_sha256'] = hashlib.sha256(canonical(origin['binding']).encode()).hexdigest()
        return CorrectionDocument.parse(data, self.root)

    def test_attachment_detaches_inputs_and_preserves_native_kinds_and_legacy_schemas(self):
        before = copy.deepcopy((self.data, self.resources, self.parameters, self.decode, self.observation))
        for version in (1, 2, 3):
            for kind in ('keep', 'remove', 'matte', 'trimap'):
                if version == 1 and kind == 'trimap':
                    continue
                with self.subTest(version=version, kind=kind):
                    data = copy.deepcopy(self.data)
                    data['schema_version'] = version
                    data['entries'][0]['kind'] = kind
                    if version == 3:
                        data['tracks'] = []
                    original = CorrectionDocument.parse(data, self.root)
                    guided = self.attach(original)
                    self.assertEqual(guided.to_dict()['schema_version'], 4)
                    self.assertEqual(guided.to_dict()['entries'][0]['kind'], kind)
                    self.assertEqual(guided.to_dict()['tracks'], [])
                    self.assertEqual(original.to_dict(), data)
        document = self.attach()
        self.parameters['settings']['threshold'] = .9
        self.assertEqual(document.to_dict()['guidance'][0]['parameters']['settings']['threshold'], .1)
        self.parameters['settings']['threshold'] = .1
        self.assertEqual(before, (self.data, self.resources, self.parameters, self.decode, self.observation))
        with self.assertRaises(TypeError):
            attach_guidance(CorrectionDocument, 'generated', guidance_id='x', kind='clean_plate',
                            resources=self.resources, parameters=self.parameters, decode=self.decode, observation=self.observation)

    def test_application_controls_and_reordering_preserve_origin_but_generation_edits_refuse(self):
        document = self.attach()
        data = document.to_dict()
        data['revision'] += 1
        entry = data['entries'][0]
        entry.update(revision=5, strength=.2, direction='forward', range=[1, 3])
        other = copy.deepcopy(entry)
        other.update(id='ordinary', frame=0, direction='both', range=[0, 3])
        other.pop('guidance_id')
        data['entries'].insert(0, other)
        edited = CorrectionDocument.parse(data, self.root)
        self.assertEqual(edited.to_dict()['guidance'], document.to_dict()['guidance'])
        for field, value in (('frame', 2), ('kind', 'keep'), ('id', 'renamed'),
                             ('asset', {'path': 'mask.bin', 'sha256': 'f' * 64})):
            bad = document.to_dict()
            bad['entries'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                CorrectionDocument.parse(bad, self.root)
        bad = document.to_dict()
        bad['source']['start_frame'] += 1
        with self.assertRaises(ValueError):
            CorrectionDocument.parse(bad, self.root)
        with self.assertRaisesRegex(ValueError, 'existing origin'):
            self.attach(document)

    def test_strict_schema_origin_bijection_and_versions(self):
        good = self.attach().to_dict()
        mutations = [lambda d: d.pop('tracks'), lambda d: d.pop('guidance'),
                     lambda d: d['guidance'][0].update(extra=True),
                     lambda d: d['entries'][0].pop('guidance_id'),
                     lambda d: d['entries'][0].update(guidance_id='missing'),
                     lambda d: d['entries'][0].update(guidance_id=[]),
                     lambda d: d.update(guidance=[]),
                     lambda d: d['guidance'].append(copy.deepcopy(d['guidance'][0])),
                     lambda d: d['guidance'][0].update(binding_sha256='f' * 64),
                     lambda d: d.update(schema_version=3),
                     lambda d: d.update(schema_version=5)]
        for index, mutate in enumerate(mutations):
            data = copy.deepcopy(good)
            mutate(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                CorrectionDocument.parse(data, self.root)
        for version in (1, 2, 3):
            data = copy.deepcopy(self.data)
            data['schema_version'] = version
            if version == 3:
                data['tracks'] = []
            data['entries'][0]['guidance_id'] = 'not-supported'
            with self.assertRaises(ValueError):
                CorrectionDocument.parse(data, self.root)

    def test_source_offset_samples_resource_roles_and_finite_settings(self):
        for name in ('source_offset', 'generation_frame_missing', 'duplicate_samples', 'unknown_reference',
                     'reference_missing', 'bad_result', 'nan', 'unknown_parameters', 'unknown_decode', 'set_value'):
            observation, parameters = copy.deepcopy(self.observation), copy.deepcopy(self.parameters)
            decode = copy.deepcopy(self.decode)
            if name == 'source_offset':
                observation['source_samples'][0]['source_frame'] = 1
            elif name == 'generation_frame_missing':
                observation['source_samples'][0].update(frame=0, source_frame=7)
            elif name == 'duplicate_samples':
                observation['source_samples'] *= 2
            elif name == 'unknown_reference':
                observation['reference_samples'][0]['resource_id'] = 'background'
            elif name == 'reference_missing':
                observation['reference_samples'] = []
            elif name == 'bad_result':
                observation['result_asset_sha256'] = 'd' * 64
            elif name == 'nan':
                parameters['settings']['threshold'] = float('nan')
            elif name == 'unknown_parameters':
                parameters['path'] = 'provider weights must stay external'
            elif name == 'unknown_decode':
                decode['reference']['unknown'] = True
            else:
                parameters['settings']['bad'] = {1, 'heterogeneous'}
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.attach(observation=observation, parameters=parameters, decode=decode)

    def test_live_source_decode_guard_and_reference_contract_persistence(self):
        document = self.attach()
        document.validate_guidance_context(decode=self.native_decode)
        changed = {**self.native_decode, 'linear_brightness': 2.}
        with self.assertRaisesRegex(ValueError, 'decode changed'):
            document.validate_guidance_context(decode=changed)
        data = document.to_dict()
        data['guidance'][0]['decode']['reference']['linear_contrast'] = 2.
        with self.assertRaises(ValueError):
            CorrectionDocument.parse(data, self.root)
        rebound = self.rebind(data)
        self.assertNotEqual(rebound.identity, document.identity)
        self.assertEqual(invalidated_ranges(document, rebound), [(0, 3)])

    def test_path_free_identity_inventory_relocation_and_hash_recheck(self):
        document = self.attach()
        before = {path: path.read_bytes() for path in self.root.iterdir()}
        inventory = document.asset_bindings()
        self.assertEqual(len(inventory), 3)
        inventory[1]['sha256'] = 'f' * 64
        self.assertNotEqual(document.asset_bindings()[1]['sha256'], 'f' * 64)
        self.assertEqual(set(document.asset_paths()), {self.root / name for name in ('mask.bin', 'plate.bin', 'background.bin')})
        child = self.root / 'saved'
        child.mkdir()
        destination = child / 'corrections.json'
        save_corrections(destination, document)
        restored = load_corrections(destination)
        self.assertEqual(restored.identity, document.identity)
        self.assertEqual(restored.asset_paths(), document.asset_paths())
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        binding = restored.to_dict()['guidance'][0]['binding']
        self.assertNotIn('path', canonical(binding))
        (self.root / 'plate.bin').write_bytes(b'mutated reference')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_corrections(destination)

    def test_reference_mutation_cancellation_and_preflight_bounds(self):
        document = self.attach()
        original = (self.root / 'plate.bin').read_bytes()
        calls = 0

        def mutate():
            nonlocal calls
            calls += 1
            if calls == 4:  # two preflight checks, before open, first hash chunk
                (self.root / 'plate.bin').write_bytes(b'x' * len(original))

        with self.assertRaisesRegex(ValueError, 'changed|hash mismatch'):
            document.verify_assets(check=mutate)
        (self.root / 'plate.bin').write_bytes(original)

        def cancel():
            raise RuntimeError('cancelled by owner')

        with self.assertRaisesRegex(RuntimeError, 'cancelled by owner'):
            document.verify_assets(check=cancel)
        with patch('corridorkey.guidance_provenance.MAX_TOTAL_RESOURCE_BYTES', 1), patch('os.open') as opened:
            with self.assertRaisesRegex(ValueError, 'resource bytes'):
                document.verify_assets()
            opened.assert_not_called()
        large = self.root / 'large.bin'
        with large.open('wb') as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
        data = document.to_dict()
        data['guidance'][0]['resources'][0]['asset']['path'] = 'large.bin'
        large_document = CorrectionDocument.parse(data, self.root)
        with patch('os.open') as opened:
            with self.assertRaisesRegex(ValueError, 'bounded regular'):
                large_document.verify_assets()
            opened.assert_not_called()

    def test_aliases_and_missing_resources_refuse_before_use(self):
        document = self.attach()
        plate = self.root / 'plate.bin'
        alias = self.root / 'alias.bin'
        os.link(plate, alias)
        with self.assertRaisesRegex(ValueError, 'single-link'):
            document.verify_assets()
        alias.unlink()
        alias.symlink_to(plate)
        data = document.to_dict()
        data['guidance'][0]['resources'][0]['asset']['path'] = 'alias.bin'
        with self.assertRaisesRegex(ValueError, 'single-link'):
            CorrectionDocument.parse(data, self.root).verify_assets()
        plate.unlink()
        with self.assertRaises(FileNotFoundError):
            document.verify_assets()

    def test_metadata_bounds_and_origin_only_invalidation_history(self):
        document = self.attach()
        data = document.to_dict()
        data['guidance'] *= 33
        with self.assertRaisesRegex(ValueError, 'bounded list'):
            CorrectionDocument.parse(data, self.root)
        parameters = copy.deepcopy(self.parameters)
        parameters['settings']['too_large'] = ['x' * 4096] * 5
        with self.assertRaisesRegex(ValueError, '16 KiB'):
            self.attach(parameters=parameters)
        resources = [dict(self.resources[1], id=f'sample-{index}') for index in range(301)]
        with self.assertRaisesRegex(ValueError, 'bounded list'):
            self.attach(resources=resources)
        data = document.to_dict()
        data['revision'] += 1
        data['guidance'][0]['parameters']['settings']['threshold'] = .2
        changed = self.rebind(data)
        self.assertEqual(invalidated_ranges(document, changed), [(0, 3)])
        history = CorrectionHistory(document)
        self.assertEqual(history.apply(changed), [(0, 3)])
        history.undo()
        self.assertEqual(history.current, document)
        history.redo()
        self.assertEqual(history.current, changed)

    def test_optional_provider_weights_are_hashes_only_and_import_is_headless(self):
        parameters = copy.deepcopy(self.parameters)
        parameters['producer']['weights_sha256'] = ['f' * 64]
        observation = {**self.observation, 'reference_samples': []}
        document = self.attach(kind='subject_clicks', resources=[], parameters=parameters,
                               decode={'source': self.native_decode, 'reference': None}, observation=observation)
        document.verify_assets()
        self.assertEqual(len(document.asset_paths()), 1)
        script = ('import sys; from corridorkey.corrections import guidance_binding, attach_guidance; '
                  'assert not any(name in sys.modules for name in ("numpy", "cv2", "torch", "PySide6"))')
        result = subprocess.run([sys.executable, '-S', '-c', script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        bad = copy.deepcopy(parameters)
        bad['producer']['weights_sha256'] = ['relative/provider.bin']
        with self.assertRaises(ValueError):
            self.attach(parameters=bad)
        # The persisted field is canonical ordinary JSON, never a dataclass/class
        # or a heterogeneous set serialized by a permissive fallback.
        self.assertIsInstance(json.loads(document.payload)['guidance'], list)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('numpy', 'cv2')), 'requires optional CPU image adapters')
    def test_native_raster_and_trimap_results_unchanged_by_provenance(self):
        import cv2
        import numpy as np

        from corridorkey.correction_rasters import CorrectionRasters

        alpha = np.full((2, 4), .4, dtype=np.float32)
        rgb = np.zeros((2, 4, 3), dtype=np.float32)
        meta = {'native_hw': (2, 4), 'start_frame': 7, 'num_frames': 5, 'native_fps': 24.}
        mask = np.array([[0, 0, 65535, 65535], [0, 0, 65535, 65535]], dtype=np.uint16)
        path = self.root / 'native.png'
        self.assertTrue(cv2.imwrite(str(path), mask))
        for kind in ('keep', 'remove', 'matte', 'trimap'):
            data = copy.deepcopy(self.data)
            data['schema_version'] = 2
            data['entries'][0].update(kind=kind, asset={'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            observation = {**self.observation, 'result_asset_sha256': data['entries'][0]['asset']['sha256']}
            ordinary = CorrectionDocument.parse(data, self.root)
            guided = self.attach(ordinary, observation=observation)
            guided.verify_assets()
            left, right = CorrectionRasters(ordinary, meta), CorrectionRasters(guided, meta)
            try:
                with self.subTest(kind=kind):
                    np.testing.assert_array_equal(left.apply(alpha, 1, guidance=rgb), right.apply(alpha, 1, guidance=rgb))
                    np.testing.assert_array_equal(left.apply(alpha, 1, temporal=True), right.apply(alpha, 1, temporal=True))
                    self.assertEqual(left.has_condition(1), right.has_condition(1))
                    self.assertEqual(left.requires_guidance(1), right.requires_guidance(1))
            finally:
                left.close()
                right.close()
        np.testing.assert_array_equal(alpha, np.full((2, 4), .4, dtype=np.float32))
        np.testing.assert_array_equal(rgb, np.zeros((2, 4, 3), dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
