import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY or TYPE_CHECKING:
    import numpy as np

    from corridorkey.evaluation_temporal import FLOW_CONVENTION
    from corridorkey.evaluation_temporal_media import (
        evaluate_temporal_manifest,
        load_temporal_manifest,
    )


@unittest.skipUnless(HAS_NUMPY, 'requires NumPy')
class TemporalMediaTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / 'temporal.json'
        frames = []
        for index in range(3):
            entry: dict[str, int | str] = {'frame': index}
            for role, value in [('candidate', np.full((1, 2), 0. if index == 0 else 1.)),
                                ('reference', np.zeros((1, 2))), ('baseline', np.zeros((1, 2)))]:
                name = f'{role}-{index}.npy'
                np.save(self.root / name, value)
                entry[role] = name
            if index:
                for role, value in [('backward_flow', np.zeros((1, 2, 2))), ('forward_flow', np.zeros((1, 2, 2))),
                                    ('visibility', np.ones((1, 2), dtype=bool))]:
                    name = f'{role}-{index}.npy'
                    np.save(self.root / name, value)
                    entry[role] = name
                entry['flow_previous_frame'] = index - 1
            frames.append(entry)
        self.data = {'schema_version': 1, 'kind': 'temporal_sequence_evaluation', 'fps': 24,
                     'flow_convention': FLOW_CONVENTION, 'frames': frames,
                     'reference_provenance': {'author': 'fixture', 'origin': 'synthetic',
                                              'permitted_use': 'local test', 'split': 'synthetic'},
                     'flow_provenance': {'method': 'synthetic zero displacement', 'origin': 'test source geometry',
                                         'visibility_definition': 'true means visible current correspondence',
                                         'independent_of_candidate': True}}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def test_real_array_assets_have_paired_spatial_temporal_scores_and_hashes(self):
        report = evaluate_temporal_manifest(self.path)
        self.assertEqual(report['candidate']['temporal_residual_mae']['value'], 0.5)
        self.assertAlmostEqual(report['candidate']['spatial']['mae']['value'], 2 / 3)
        self.assertEqual(report['candidate_minus_baseline']['temporal_residual_mae'], 0.5)
        self.assertEqual(len(report['assets']), 15)
        np.save(self.root / 'visibility-1.npy', np.zeros((1, 2), dtype=bool))
        after = evaluate_temporal_manifest(self.path)
        self.assertNotEqual(report['assets'], after['assets'])
        self.assertEqual(after['candidate']['temporal_residual_mae']['value'], 0)
        self.assertEqual(after['candidate']['status'], 'partial')

    def test_schema_rejects_flow_direction_wrong_links_and_candidate_dependent_provenance(self):
        for change in ({'flow_convention': 'forward_xy'}, {'flow_provenance': {}}, {'unknown': 1}):
            self.path.write_text(json.dumps({**self.data, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_temporal_manifest(self.path)
        self.data['frames'][1]['flow_previous_frame'] = 99
        self.save()
        with self.assertRaisesRegex(ValueError, 'preceding'):
            load_temporal_manifest(self.path)
        self.data['frames'][1]['flow_previous_frame'] = 0
        self.data['flow_provenance']['independent_of_candidate'] = False
        self.save()
        with self.assertRaisesRegex(ValueError, 'independent'):
            load_temporal_manifest(self.path)

    def test_cut_excludes_transition_and_missing_flow_remains_unavailable(self):
        for entry in self.data['frames'][1:]:
            for field in ('backward_flow', 'forward_flow', 'visibility', 'flow_previous_frame'):
                entry.pop(field)
            entry['cut'] = True
        self.data.pop('flow_provenance')
        self.save()
        report = evaluate_temporal_manifest(self.path)
        self.assertEqual(report['candidate']['cut_pairs'], 2)
        self.assertIsNone(report['candidate']['temporal_residual_mae']['value'])
        self.assertAlmostEqual(report['candidate']['spatial']['mae']['value'], 2 / 3)

    def test_flow_geometry_and_binary_visibility_are_validated(self):
        np.save(self.root / 'backward_flow-1.npy', np.zeros((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, 'native HxWx2'):
            evaluate_temporal_manifest(self.path)
        np.save(self.root / 'backward_flow-1.npy', np.zeros((1, 2, 2)))
        np.save(self.root / 'visibility-1.npy', np.full((1, 2), 0.5))
        with self.assertRaisesRegex(ValueError, 'binary'):
            evaluate_temporal_manifest(self.path)

    def test_real_cli_protects_flow_and_old_report_after_failure(self):
        output = self.root / 'report.json'
        command = [sys.executable, '-m', 'corridorkey.evaluation_temporal_media', str(self.path), '--output']
        process = subprocess.run([*command, str(output)], check=False, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        before = output.read_bytes()
        process = subprocess.run([*command, str(self.root / 'backward_flow-1.npy')], check=False,
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)
        (self.root / 'backward_flow-1.npy').write_bytes(b'corrupt')
        process = subprocess.run([*command, str(output)], check=False, capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(output.read_bytes(), before)
