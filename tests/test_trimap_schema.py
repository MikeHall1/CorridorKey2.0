"""Explicit schema admission and immutable trimap dependency identity."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from corridorkey.corrections import (
    CorrectionDocument,
    CorrectionHistory,
    invalidated_ranges,
    load_corrections,
    save_corrections,
)
from corridorkey.identities import file_sha256


def data():
    return {'schema_version': 1, 'revision': 0,
            'source': {'sha256': 'a' * 64, 'start_frame': 7, 'frame_count': 4,
                       'fps': 24, 'width': 5, 'height': 3},
            'shots': [[0, 2], [2, 4]], 'entries': []}


def entry(kind='trimap', *, identifier='t', frame=1):
    return {'id': identifier, 'revision': 1, 'frame': frame, 'kind': kind,
            'asset': {'path': 'classes.png', 'sha256': 'b' * 64},
            'strength': .5, 'direction': 'both', 'range': [0, 2]}


class TrimapSchemaTests(unittest.TestCase):
    def test_new_semantics_require_version_two_and_do_not_promote_old_masks(self):
        legacy = data()
        legacy['entries'] = [entry('matte')]
        original = CorrectionDocument.parse(legacy)
        self.assertEqual(original.to_dict(), legacy)
        self.assertEqual(original.to_dict()['schema_version'], 1)
        proposed = copy.deepcopy(legacy)
        proposed['entries'][0]['kind'] = 'trimap'
        with self.assertRaises(ValueError):
            CorrectionDocument.parse(proposed)
        proposed['schema_version'] = 2
        trimap = CorrectionDocument.parse(proposed)
        self.assertEqual(trimap.to_dict()['entries'][0]['kind'], 'trimap')
        self.assertNotEqual(trimap.identity, original.identity)
        mixed = trimap.to_dict()
        mixed['entries'].append(entry('keep', identifier='keep'))
        self.assertEqual(len(CorrectionDocument.parse(mixed).to_dict()['entries']), 2)
        for version in (True, 0, 3, '2'):
            with self.subTest(version=version), self.assertRaises(ValueError):
                CorrectionDocument.parse({**proposed, 'schema_version': version})

    def test_strength_order_ranges_and_unknown_fields_remain_strict(self):
        original = data()
        original.update(schema_version=2, entries=[entry()])
        for mutation in ({'strength': True}, {'strength': float('nan')},
                         {'strength': -.1}, {'range': [0, 3]},
                         {'direction': 'forward'}, {'radius': 4}, {'kind': 'unknown'}):
            candidate = copy.deepcopy(original)
            candidate['entries'][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                CorrectionDocument.parse(candidate)
        after = copy.deepcopy(original)
        after['entries'][0]['asset']['sha256'] = 'c' * 64
        self.assertEqual(invalidated_ranges(CorrectionDocument.parse(original),
                                            CorrectionDocument.parse(after)), [(0, 2)])

    def test_version_upgrade_undo_save_reopen_and_exact_asset_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asset = root / 'classes.png'
            asset.write_bytes(b'Identity-only fixture; class decoding is tested by the raster boundary.')
            original = CorrectionDocument.parse(data(), root)
            upgraded = original.to_dict()
            upgraded.update(schema_version=2, revision=1, entries=[entry()])
            upgraded['entries'][0]['asset']['sha256'] = file_sha256(asset)
            upgraded_doc = CorrectionDocument.parse(upgraded, root)
            history = CorrectionHistory(original)
            self.assertEqual(history.apply(upgraded_doc), [(0, 2)])
            history.undo()
            self.assertEqual(history.current.to_dict()['schema_version'], 1)
            history.redo()
            path = root / 'saved' / 'document.json'
            save_corrections(path, history.current)
            # This schema/identity fixture deliberately has no raster pixels.
            # Exercise dispatch to the separate bounded raster boundary without
            # treating opaque bytes as a valid decoded trimap.
            calls = []

            def admitted(asset_path, *, expected_hw, expected_sha256, check=None):
                calls.append(Path(asset_path))
                self.assertEqual(expected_hw, (3, 5))
                if file_sha256(asset_path, check=check) != expected_sha256:
                    raise ValueError('correction asset hash mismatch')

            boundary = SimpleNamespace(load_trimap_asset=admitted)
            with patch.dict('sys.modules', {'corridorkey.trimap_refinement': boundary}):
                loaded = load_corrections(path)
            self.assertEqual(loaded.identity, upgraded_doc.identity)
            self.assertEqual(loaded.to_dict()['schema_version'], 2)
            self.assertEqual(loaded.asset_path(loaded.to_dict()['entries'][0]).read_bytes(), asset.read_bytes())
            relocated = loaded.to_dict()
            relocated['entries'][0]['asset']['path'] = '/different/classes.png'
            self.assertEqual(CorrectionDocument.parse(relocated).identity, loaded.identity)
            asset.write_bytes(b'Changed class assignment')
            with patch.dict('sys.modules', {'corridorkey.trimap_refinement': boundary}), \
                    self.assertRaisesRegex(ValueError, 'asset hash'):
                load_corrections(path)
            self.assertEqual(len(calls), 2)

    def test_trimap_preflight_delegates_before_hashing_and_preserves_lexical_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            target = root / 'target.png'
            target.write_bytes(b'identity-only fixture')
            alias = root / 'classes.png'
            alias.symlink_to(target.name)
            original = data()
            original.update(schema_version=2, entries=[entry()])
            document = CorrectionDocument.parse(original, root)
            admission = Mock(side_effect=ValueError('bounded refusal'))
            with patch.dict('sys.modules', {'corridorkey.trimap_refinement': SimpleNamespace(
                    load_trimap_asset=admission)}), \
                    patch('corridorkey.corrections.file_sha256', side_effect=AssertionError('unbounded hash')), \
                    self.assertRaisesRegex(ValueError, 'bounded refusal'):
                document.verify_assets()
            admission.assert_called_once_with(alias, expected_hw=(3, 5),
                                              expected_sha256='b' * 64, check=None)
