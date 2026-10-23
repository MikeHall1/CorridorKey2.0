"""Connected native subject guidance with an explicit provider-interface fixture."""
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import test_subject_guidance

from corridorkey.config import InferenceConfig
from corridorkey.control import CancelledError
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.correction_rasters import CorrectionRasters
from corridorkey.corrections import load_corrections
from corridorkey.session import JobRequest
from corridorkey.subject_guidance import ExclusiveResourceLease, GuidanceSession


class FixtureProcess:
    """Adapter test double only; the production UI cannot select this provider."""
    def __init__(self, config, checkpoint, *, workspace):
        self.provider = test_subject_guidance.FixtureProvider()
        self.session = GuidanceSession(self.provider, ExclusiveResourceLease('cpu', 'editor-fixture', lambda: True))
        self.identity = None
        self.closed = False
        self.close_error = False

    def predict(self, frame, clicks, *, check):
        check()
        if frame.identity != self.identity:
            self.session.bind(frame)
            self.identity = frame.identity
        result = self.session.predict(self.session.request(clicks))
        check()
        return result

    def clear(self):
        self.session.clear()

    def close(self):
        if self.close_error:
            raise OSError('fixture provider reap pending')
        self.session.close()
        self.closed = True


class SubjectGuidanceEditorTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(4):
            cv2.imwrite(str(self.source / f'{frame}.png'), np.full((24, 32, 3), (20, 50 + frame, 180), np.uint8))
        self.originals = {path: path.read_bytes() for path in self.source.iterdir()}
        self.config = self.root / 'provider.yaml'
        self.weights = self.root / 'provider.pt'
        self.config.write_text('fixture only')
        self.weights.write_bytes(b'fixture only; not a real model')
        self.job = JobRequest.create(self.source, self.root / 'output', InferenceConfig(start_frame=1, output_format='exr'))
        self.editor = CorrectionEditor(self.job, self.root)
        self.addCleanup(self.editor.discard_subject)
        self.patch = patch('corridorkey.subject_guidance_editor.SubjectGuidanceProcess', FixtureProcess)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.clicks = [{'x': 10, 'y': 12, 'positive': True}]

    def prepare(self):
        self.editor.prepare_subject(frame=0, config_path=self.config, checkpoint_path=self.weights)

    def test_explicit_source_predict_inspect_keep_save_reopen_without_provider_weights(self):
        self.prepare()
        summary = self.editor.subject_summary()
        assert summary is not None
        self.assertEqual((summary['state'], summary['frame'], summary['source_frame']), ('prepared', 0, 1))
        self.assertEqual(self.editor.document.to_dict()['entries'], [])
        self.editor.predict_subject(clicks=self.clicks)
        self.clicks[0]['x'] = 2  # Caller input is detached before later publication/acceptance.
        with self.assertRaisesRegex(ValueError, 'inspect'):
            self.editor.accept_subject()
        display, overlay = self.editor.subject_view()
        assert overlay is not None
        self.assertEqual(overlay[12, 10], 255)
        display.pixels[:] = 0
        overlay[:] = 0
        _, again = self.editor.subject_view()
        assert again is not None
        self.assertEqual(again[12, 10], 255)
        self.editor.accept_subject()
        self.assertIsNone(self.editor.subject_preview)
        self.assertEqual(self.editor.revision, 1)
        data = self.editor.document.to_dict()
        self.assertEqual((data['schema_version'], data['entries'][0]['kind']), (4, 'keep'))
        origin = data['guidance'][0]
        self.assertIn('fixture', origin['observation']['diagnostics']['evidence'])
        self.assertEqual(origin['parameters']['settings']['provider_execution']['clicks'][0]['x'], 10)
        self.assertEqual(origin['observation']['source_samples'][0]['source_frame'], 1)
        rasters = CorrectionRasters(self.editor.document, self.editor.meta)
        try:
            expected = np.full((24, 32), .4, np.float32)
            alpha = rasters.apply(expected.copy(), 0)
            self.assertEqual(alpha[12, 10], 1.)
            alpha[12, 10] = .4
            np.testing.assert_array_equal(alpha, expected)
            np.testing.assert_array_equal(rasters.apply(expected.copy(), 1), expected)
        finally:
            rasters.close()
        path = self.editor.save()
        self.weights.unlink()
        self.config.unlink()
        self.assertEqual(load_corrections(path, source=self.source).identity, self.editor.document.identity)
        self.editor.undo()
        self.assertFalse(self.editor.document.to_dict().get('guidance'))
        self.editor.redo()
        self.assertEqual(self.editor.document.to_dict()['entries'][0]['kind'], 'keep')
        self.assertEqual(self.originals, {path: path.read_bytes() for path in self.source.iterdir()})

    def test_refinement_reuses_embedding_and_clear_restarts_logits(self):
        self.prepare()
        preview = self.editor.subject_preview
        assert preview is not None
        owner = preview['owner']
        self.editor.predict_subject(clicks=self.clicks)
        self.editor.subject_view()
        self.editor.predict_subject(clicks=[*self.clicks, {'x': 10, 'y': 12, 'positive': False}, {'x': 14, 'y': 12, 'positive': True}])
        summary = self.editor.subject_summary()
        assert summary is not None
        self.assertFalse(summary['inspected'])
        self.assertEqual(owner.provider.embeddings, 1)
        self.assertIsNotNone(owner.provider.previous[-1])
        self.editor.clear_subject()
        self.editor.predict_subject(clicks=self.clicks)
        self.assertIsNone(owner.provider.previous[-1])
        self.assertEqual(owner.provider.embeddings, 1)

    def test_accepted_keep_is_a_trackable_schema4_anchor_without_losing_origin(self):
        self.prepare()
        self.editor.predict_subject(clicks=self.clicks)
        self.editor.subject_view()
        self.editor.accept_subject()
        origin = self.editor.document.to_dict()['guidance']
        entry = self.editor.document.to_dict()['entries'][0]
        self.editor.prepare_tracking(anchor_ids=[entry['id']], span=[0, 3], direction='forward')
        summary = self.editor.tracking_summary()
        assert summary is not None
        self.assertTrue(summary['complete'])
        self.editor.tracking_view(2)
        self.editor.accept_tracking()
        self.assertEqual(self.editor.document.to_dict()['schema_version'], 4)
        self.assertEqual(self.editor.document.to_dict()['guidance'], origin)
        self.assertEqual(len(self.editor.document.to_dict()['tracks']), 1)
        self.assertEqual(load_corrections(self.editor.save(), source=self.source).identity, self.editor.document.identity)

    def test_failed_save_keeps_preview_without_copy_growth_and_new_request_removes_old_result(self):
        self.prepare()
        self.editor.predict_subject(clicks=self.clicks)
        self.editor.subject_view()
        before = self.editor.document.identity
        for _ in range(3):
            with patch('corridorkey.correction_editor.save_corrections', side_effect=OSError('fixture disk full')), \
                    self.assertRaisesRegex(OSError, 'disk full'):
                self.editor.accept_subject()
            self.assertFalse(list((self.editor.workspace / 'assets').iterdir()))
            self.assertFalse(list(self.editor.workspace.glob('subject-guidance-*')))
        self.assertEqual(self.editor.document.identity, before)
        assert self.editor.subject_preview is not None
        with patch.object(self.editor.subject_preview['owner'], 'predict', side_effect=ValueError('fixture failed prediction')), \
                self.assertRaisesRegex(ValueError, 'failed prediction'):
            self.editor.predict_subject(clicks=self.clicks)
        with self.assertRaisesRegex(ValueError, 'Predict'):
            self.editor.accept_subject()

    def test_source_mutation_refuses_acceptance_and_owner_cleanup_retains_reachability(self):
        self.prepare()
        preview = self.editor.subject_preview
        assert preview is not None
        owner = preview['owner']
        self.editor.predict_subject(clicks=self.clicks)
        self.editor.subject_view()
        path = self.source / '1.png'
        cv2.imwrite(str(path), np.zeros((24, 32, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.editor.accept_subject()
        owner.close_error = True
        self.editor.discard_subject()
        assert self.editor.subject_preview is not None
        assert self.editor.subject_cleanup_error is not None
        self.assertIs(self.editor.subject_preview['owner'], owner)
        self.assertIn('reap pending', self.editor.subject_cleanup_error)
        with self.assertRaisesRegex(OSError, 'cleanup'):
            self.prepare()
        owner.close_error = False
        self.editor.discard_subject()
        self.assertIsNone(self.editor.subject_preview)
        self.assertTrue(owner.closed)

    def test_postcommit_close_failure_reports_committed_history_and_retains_owner(self):
        self.prepare()
        preview = self.editor.subject_preview
        assert preview is not None
        owner = preview['owner']
        self.editor.predict_subject(clicks=self.clicks)
        self.editor.subject_view()
        owner.close_error = True
        self.editor.accept_subject()
        self.assertEqual(self.editor.revision, 1)
        self.assertTrue(self.editor.document.to_dict()['guidance'])
        assert self.editor.subject_cleanup_error is not None
        self.assertIn('reap pending', self.editor.subject_cleanup_error)
        owner.close_error = False
        self.editor.discard_subject()

    def test_source_observation_binds_exact_native_rgb_and_explicit_proxy(self):
        from corridorkey.video_io import _read_frame_dir_views
        self.prepare()
        preview = self.editor.subject_preview
        assert preview is not None
        _, linear = _read_frame_dir_views(str(self.source / '1.png'), exr_linear=False)
        self.assertEqual(preview['native_hash'], hashlib.sha256(linear.astype('<f4').tobytes()).hexdigest())
        self.assertIn('no resize', preview['source'].decode_identity)
        with self.assertRaisesRegex(ValueError, '128'):
            self.editor.predict_subject(clicks=self.clicks * 129)
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.editor.predict_subject(clicks=[{'x': 1, 'y': 1, 'positive': False}])

    def test_invalid_newer_prompts_always_revoke_inspected_previous_result(self):
        self.prepare()
        for invalid in ([], self.clicks * 129, [{'x': 32, 'y': 0, 'positive': True}],
                        [{'x': 1, 'y': 1, 'positive': False}]):
            self.editor.predict_subject(clicks=self.clicks)
            self.editor.subject_view()
            with self.assertRaises(ValueError):
                self.editor.predict_subject(clicks=invalid)
            with self.assertRaisesRegex(ValueError, 'Predict'):
                self.editor.accept_subject()

    def test_service_cancellation_before_dispatch_revokes_previous_inspected_mask(self):
        from corridorkey.correction_editor_service import EditorService
        service = EditorService(self.job, self.root)
        try:
            self.assertTrue(service.ready.wait(5))
            service.events.get(timeout=5)
            service.request('subject_prepare', frame=0, config_path=self.config, checkpoint_path=self.weights)
            self.assertEqual(service.events.get(timeout=5)['kind'], 'result')
            service.request('subject_predict', clicks=self.clicks)
            self.assertEqual(service.events.get(timeout=5)['kind'], 'result')
            service.request('subject_view')
            self.assertTrue(service.events.get(timeout=5)['snapshot']['subject_preview']['inspected'])
            with patch.object(service, 'check', side_effect=CancelledError('fixture cancelled before dispatch')):
                service.request('subject_predict', clicks=self.clicks)
                event = service.events.get(timeout=5)
            self.assertEqual(event['kind'], 'error')
            self.assertIn('cancelled before dispatch', event['error'])
            self.assertIsNone(event['snapshot']['subject_preview']['result'])
            self.assertFalse(event['snapshot']['subject_preview']['inspected'])
        finally:
            service.stop()
            service.thread.join(5)
            self.assertTrue(service.done.is_set())

    def test_busy_service_request_cannot_clear_active_cancellation(self):
        from corridorkey.correction_editor_service import EditorService
        entered, release = threading.Event(), threading.Event()
        def blocked(owner, frame, clicks, *, check):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('fixture release deadline')
            check()
            raise AssertionError('cancelled operation must not publish')
        service = EditorService(self.job, self.root)
        try:
            self.assertTrue(service.ready.wait(5))
            service.events.get(timeout=5)
            service.request('subject_prepare', frame=0, config_path=self.config, checkpoint_path=self.weights)
            self.assertEqual(service.events.get(timeout=5)['kind'], 'result')
            with patch.object(FixtureProcess, 'predict', blocked):
                service.request('subject_predict', clicks=self.clicks)
                self.assertTrue(entered.wait(5))
                service.cancel_current.set()
                with self.assertRaisesRegex(RuntimeError, 'retire'):
                    service.request('subject_view')
                self.assertTrue(service.cancel_current.is_set())
                release.set()
                event = service.events.get(timeout=5)
            self.assertEqual(event['kind'], 'error')
            self.assertIn('cancelled', event['error'])
        finally:
            release.set()
            service.stop()
            service.thread.join(5)
            self.assertTrue(service.done.is_set())
