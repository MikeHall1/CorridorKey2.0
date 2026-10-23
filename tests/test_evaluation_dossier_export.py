import hashlib
import json
import os
import tempfile
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

from corridorkey.control import CancelledError
from corridorkey.evaluation_dossier_export import export_dossier


class Revalidator(types.ModuleType):
    revalidate_inputs: Callable[..., Any]


class DossierExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sources = self.root / 'source'
        self.sources.mkdir()
        self.report = {'token': 'fixture', 'protected_paths': [str(self.sources)], 'cases': [
            {'id': '=bad', 'metrics': {'alpha.mae': {'units': 'coverage', 'candidate_median': .2}}}],
            'categories': [], 'review_media': []}
        for side in ('baseline', 'candidate'):
            path = self.sources / f'{side}.npy'
            path.write_bytes(b'exact fixture bytes ' + side.encode())
            self.report['review_media'].append({'case_id': '=bad', 'side': side, 'frame': 4,
                'role': 'candidate', 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size, 'fps': 24., 'view_contract': {}})
        self.original = {path: path.read_bytes() for path in self.sources.iterdir()}
        self.fake = Revalidator('corridorkey.evaluation_dossier')
        self.fake.revalidate_inputs = lambda report, **kwargs: report
        self.module_patch = patch.dict('sys.modules', {'corridorkey.evaluation_dossier': self.fake})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_metadata_export_is_fresh_escaped_and_pending(self):
        first = export_dossier(self.report, self.root)
        second = export_dossier(self.report, self.root)
        self.assertTrue(first['usable'])
        self.assertNotEqual(first['destination'], second['destination'])
        result = Path(first['destination'])
        self.assertTrue((result / 'COMPLETE.json').is_file())
        self.assertIn("'=bad", (result / 'analysis.csv').read_text())
        packet = json.loads((result / 'review/packet.json').read_text())
        self.assertEqual(packet['media_status'], 'not_prepared')
        self.assertEqual(packet['review_status'], 'pending')
        self.assertEqual(packet['media'], [])
        self.assertEqual(len(packet['cases']), 1)
        self.assertEqual(len((result / 'review/ratings.csv').read_text().splitlines()), 2)

    def test_budget_exact_bytes_and_assignment_concealment(self):
        total = sum(row['bytes'] for row in self.report['review_media'])
        refused = export_dossier(self.report, self.root, include_media=True, max_media_bytes=total - 1)
        self.assertFalse(refused['usable'])
        self.assertIsNone(refused['partial_path'])
        progress = []
        result = export_dossier(self.report, self.root, include_media=True, max_media_bytes=total,
                                progress=progress.append)
        self.assertTrue(result['usable'], result)
        destination = Path(result['destination'])
        packet = json.loads((destination / 'review/packet.json').read_text())
        self.assertEqual({row['assignment'] for row in packet['media']}, {'A', 'B'})
        self.assertEqual(len({row['case'] for row in packet['media']}), 1)
        self.assertEqual([row['assignment'] for row in packet['media']], ['A', 'B'])
        self.assertEqual(progress[-1].stage, 'copy_review_media')
        self.assertEqual(progress[-1].completed, 2)
        public = (destination / 'review/packet.json').read_text()
        self.assertNotIn(str(self.sources), public)
        self.assertNotIn('=bad', public)
        for row in packet['media']:
            self.assertIn((destination / 'review' / row['media']).read_bytes(), self.original.values())
        self.assertEqual(self.original, {path: path.read_bytes() for path in self.sources.iterdir()})

    def test_protected_tree_and_missing_side_refused(self):
        result = export_dossier(self.report, self.sources)
        self.assertFalse(result['usable'])
        self.assertIsNone(result['partial_path'])
        self.report['review_media'].pop()
        result = export_dossier(self.report, self.root, include_media=True, max_media_bytes=1000)
        self.assertFalse(result['usable'])
        self.assertIsNone(result['partial_path'])

    def test_stale_media_is_unsealed_partial(self):
        path = next(iter(self.original))
        path.write_bytes(b'x' * len(self.original[path]))
        result = export_dossier(self.report, self.root, include_media=True, max_media_bytes=1000)
        self.assertFalse(result['usable'])
        self.assertIsNotNone(result['partial_path'])
        self.assertFalse((Path(result['partial_path']) / 'COMPLETE.json').exists())

    def test_final_revalidation_failure_never_seals(self):
        calls = []
        def validate(report, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError('changed input')
            return report
        self.fake.revalidate_inputs = validate
        result = export_dossier(self.report, self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((Path(result['partial_path']) / 'COMPLETE.json').exists())

    def test_cancellation_and_media_growth_do_not_seal(self):
        def cancel():
            raise CancelledError('requested')
        result = export_dossier(self.report, self.root, check=cancel)
        self.assertEqual(result['status'], 'cancelled')
        self.assertIsNone(result['partial_path'])
        path = next(iter(self.original))
        path.write_bytes(path.read_bytes() + b'x')
        result = export_dossier(self.report, self.root, include_media=True, max_media_bytes=1000)
        self.assertFalse(result['usable'])
        self.assertFalse((Path(result['partial_path']) / 'COMPLETE.json').exists())

    def test_post_commit_exception_reports_actual_usable_export(self):
        original = os.rename
        def late_error(source, destination):
            original(source, destination)
            raise KeyboardInterrupt('after actual commit')
        with patch('corridorkey.evaluation_dossier_export.os.rename', side_effect=late_error):
            result = export_dossier(self.report, self.root)
        self.assertEqual(result['status'], 'complete', result)
        self.assertTrue(result['usable'])
        self.assertIn('late KeyboardInterrupt', result['warning'])
        self.assertTrue((Path(result['destination']) / 'COMPLETE.json').is_file())

    def test_source_fence_follows_pending_marker_flush(self):
        from corridorkey import evaluation_dossier_export as exporter
        original = exporter._write
        flushed = []
        def write(path, data):
            original(path, data)
            if path.name == '.complete-pending.json':
                flushed.append(True)
        def validate(report, **kwargs):
            if flushed:
                raise ValueError('changed while staging marker')
            return report
        self.fake.revalidate_inputs = validate
        with patch.object(exporter, '_write', side_effect=write):
            result = export_dossier(self.report, self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((Path(result['partial_path']) / 'COMPLETE.json').exists())
