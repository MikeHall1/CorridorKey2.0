"""Actual near-limit manifests fit bounded terminal records; no neural model."""
import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from corridorkey.correction_reporting import (
    MAX_CORRECTION_REPORT_BYTES,
    correction_report,
)
from corridorkey.corrections import (
    MAX_CORRECTION_BYTES,
    CorrectionDocument,
    canonical,
    track_binding,
)

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
STACK = all(importlib.util.find_spec(name) for name in ('numpy', 'cv2'))


class ReportingImportTests(unittest.TestCase):
    def test_metadata_summary_import_remains_stdlib_only(self):
        output = subprocess.check_output([sys.executable, '-S', '-c',
            ('import sys; import corridorkey.correction_reporting; '
             'assert not any(name in sys.modules for name in ("numpy","cv2","torch","PySide6")); print("PASS")')],
            cwd=Path(__file__).resolve().parents[1], text=True)
        self.assertEqual(output.strip(), 'PASS')


@unittest.skipUnless(STACK, 'requires optional CPU image adapters')
class CorrectionReportingTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.correction_tracking_context import tracking_decode
        from corridorkey.identities import source_identity
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for index in range(80):
            self.assertTrue(cv2.imwrite(str(self.source / f'{index:03}.png'), np.full((2, 4, 3), index, np.uint8)))
        self.source_hash = source_identity(self.source)['sha256']
        mask = np.zeros((2, 4), np.float32)
        mask[:, 1] = 1
        authored, derived = self.root / 'authored.png', self.root / 'derived.exr'
        self.assertTrue(cv2.imwrite(str(authored), np.rint(mask * 65535).astype(np.uint16)))
        self.assertTrue(cv2.imwrite(str(derived), mask, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]))
        self.authored = {'path': str(authored), 'sha256': hashlib.sha256(authored.read_bytes()).hexdigest()}
        self.derived = {'path': str(derived), 'sha256': hashlib.sha256(derived.read_bytes()).hexdigest()}
        self.mask_hash = hashlib.sha256(mask.astype('<f4').tobytes()).hexdigest()
        self.decode = tracking_decode(InferenceConfig(), self.source)
        self.settings = {'kernel_version': 'correction-tracking-v1', 'direction': 'both',
                         'unreliable_policy': 'request_anchor', 'max_uncertain_fraction': .25,
                         'motion': {'max_side': 640, 'consistency_pixels': 1.5, 'photometric_scale': .15,
                                    'min_reliability': .2, 'max_disagreement': .25},
                         'max_frames': 300, 'max_pixels': 4 * 1024**2, 'max_anchor_bytes': 64 * 1024**2,
                         'max_anchors': 32, 'estimator_identity': 'fixture-metadata-no-inference'}
        self.data = {'schema_version': 1, 'revision': 1,
                     'source': {'sha256': self.source_hash, 'start_frame': 0, 'frame_count': 80,
                                'fps': 24., 'width': 4, 'height': 2},
                     'shots': [[0, 80]], 'entries': [self.entry('artist', 0, [0, 80])]}

    def entry(self, identifier, frame, span):
        return {'id': identifier, 'revision': 1, 'frame': frame, 'kind': 'keep', 'strength': .5,
                'direction': 'both', 'range': span, 'asset': self.authored.copy()}

    def track(self, data, identifier='tracked', anchor_id='artist', span=(0, 80)):
        binding = track_binding(data, [anchor_id], cuts=[], decode=self.decode, tracking=self.settings)
        return {'id': identifier, 'owner_id': anchor_id, 'anchor_ids': [anchor_id], 'range': list(span), 'direction': 'both',
                'binding': binding, 'binding_sha256': hashlib.sha256(canonical(binding).encode()).hexdigest(),
                'frames': [{'frame': frame, 'asset': self.derived.copy()} for frame in range(*span)],
                'observations': [{'frame': frame, 'source_frame': frame,
                                  'status': 'authored_anchor' if frame == binding['anchors'][0]['frame'] else 'preview',
                                  'candidate_identity': 'b' * 64, 'source_proxy_sha256': {str(frame): 'c' * 64},
                                  'derived_mask_sha256': self.mask_hash,
                                  'diagnostics': {'fixture': True, 'reliability_calibrated': False}}
                                 for frame in range(*span)]}

    def tracked(self):
        data = copy.deepcopy(self.data)
        data.update(schema_version=3, tracks=[self.track(data)])
        return data

    def near_limit(self, tracked):
        data = self.tracked() if tracked else copy.deepcopy(self.data)
        target = MAX_CORRECTION_BYTES - 64
        if tracked:
            rows = data['tracks'][0]['observations']
            for row in rows:
                row['diagnostics']['padding'] = ['x' * 4000] * 12 + ['']
            quotient, remainder = divmod(target - len(canonical(data).encode()), len(rows))
            self.assertTrue(0 <= quotient < 4096)
            for index, row in enumerate(rows):
                row['diagnostics']['padding'][-1] = 'x' * (quotient + (index < remainder))
        else:
            data['entries'][0]['id'] = ''
            data['entries'][0]['id'] = 'x' * (target - len(canonical(data).encode()))
        document = CorrectionDocument.parse(data, self.root)
        self.assertEqual(len(document.payload.encode()), target)
        return document

    def test_small_legacy_detail_fields_preserved_and_detached(self):
        document = CorrectionDocument.parse(self.data, self.root)
        report = correction_report(document, self.root / 'manifest.json')
        for key in ('entries', 'shots', 'source'):
            self.assertEqual(report[key], self.data[key])
        self.assertEqual(report['authored_frames'], [0])
        self.assertEqual(report['reset_frames'], [])
        self.assertEqual(report['tracks'], [])
        report['entries'][0]['strength'] = 0
        self.assertEqual(document.to_dict()['entries'][0]['strength'], .5)

    def test_tracks_keep_compact_bindings_and_counts_without_observation_payloads(self):
        data = self.tracked()
        data['tracks'][0]['binding']['tracking']['unreliable_policy'] = 'retain_approximation'
        data['tracks'][0]['binding_sha256'] = hashlib.sha256(canonical(data['tracks'][0]['binding']).encode()).hexdigest()
        data['tracks'][0]['observations'][1]['status'] = 'approximation'
        report = correction_report(CorrectionDocument.parse(data, self.root), self.root / 'manifest.json')
        track = report['tracks'][0]
        self.assertEqual((track['id'], track['range'], track['direction']), ('tracked', [0, 80], 'both'))
        self.assertEqual(track['binding_sha256'], data['tracks'][0]['binding_sha256'])
        self.assertEqual((track['frame_count'], track['observation_count'], track['approximation_count']), (80, 80, 1))
        for key in ('frames', 'observations', 'binding'):
            self.assertNotIn(key, track)
            self.assertIn(f'tracks[].{key}', report['omitted_fields'])

    def test_near_limit_valid_tracked_and_legacy_session_failures_fit_worker_frames(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.identities import source_identity
        from corridorkey.session import InferenceSession, JobRequest
        from corridorkey.worker_protocol import encode_frame
        for tracked in (True, False):
            with self.subTest(tracked=tracked):
                document = self.near_limit(tracked)
                path = self.root / ('tracked.json' if tracked else 'legacy.json')
                path.write_text(document.payload)
                def unavailable_model():
                    raise RuntimeError('fixture model-load failure; no inference')
                job = JobRequest.create(self.source, self.root / 'output',
                                        InferenceConfig(output_format='exr', num_frames=80, corrections_path=path))
                with InferenceSession(loader=unavailable_model) as session:
                    with self.assertRaisesRegex(RuntimeError, 'fixture model-load'):
                        session.run(job)
                    assert session.last_result is not None
                    record = session.last_result.to_dict()
                encoded = encode_frame({'kind': 'failed', 'record': record})
                self.assertLess(len(encoded), MAX_CORRECTION_BYTES)
                metadata = record['diagnostics']['corrections']
                self.assertEqual(metadata['identity'], document.identity)
                self.assertEqual(metadata['document_path'], str(path.resolve()))
                self.assertEqual(record['failure_checkpoint']['stage'], 'model_load')
                self.assertLessEqual(len(canonical(metadata).encode()), MAX_CORRECTION_REPORT_BYTES)
                if not tracked:
                    self.assertNotIn('entries', metadata)
                    self.assertIn('entries', metadata['omitted_fields'])
                self.assertEqual(source_identity(self.source)['sha256'], self.source_hash)

    def test_max_tracks_and_large_labels_stay_bounded_with_explicit_omissions(self):
        data = copy.deepcopy(self.data)
        data['entries'] = [self.entry('artist-' + 'x' * 2000 + str(frame), frame, [frame, frame + 1]) for frame in range(32)]
        data.update(schema_version=3, tracks=[self.track(data, identifier='😀' * 250 + str(frame),
                    anchor_id=entry['id'], span=(frame, frame + 1)) for frame, entry in enumerate(data['entries'])])
        document = CorrectionDocument.parse(data, self.root)
        report = correction_report(document, 'x' * 10000)
        self.assertEqual(len(report['tracks']), 32)
        self.assertEqual(report['tracks_count'], 32)
        self.assertIsNone(report['document_path'])
        self.assertIn('document_path', report['omitted_fields'])
        self.assertTrue(all(row['owner_id'] is None and 'owner_id' in row['omitted_fields'] for row in report['tracks']))
        self.assertLessEqual(len(canonical(report).encode()), MAX_CORRECTION_REPORT_BYTES)

    def test_runtime_counters_retained_and_reserved_provenance_cannot_be_replaced(self):
        document = CorrectionDocument.parse(self.tracked(), self.root)
        fields = {'output_frames_refined': 0, 'model_conditioning': 'both_direction_authored_hints_and_auxiliary_carry_only'}
        report = correction_report(document, 'manifest.json', runtime_fields=fields)
        self.assertEqual(report['output_frames_refined'], 0)
        self.assertEqual(report['model_conditioning'], fields['model_conditioning'])
        with self.assertRaises(ValueError):
            correction_report(document, 'manifest.json', runtime_fields={'identity': 'wrong'})


if __name__ == '__main__':
    unittest.main()
