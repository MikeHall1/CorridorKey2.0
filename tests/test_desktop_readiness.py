"""Qt readiness workflow fixtures, never evidence of real model inference."""
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(QT_AVAILABLE, 'Optional requirements-desktop.txt is not installed')
class DesktopReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['readiness-ui-test', '-platform', 'offscreen'])

    def setUp(self):
        from corridorkey.desktop_window import DesktopWindow
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.window = DesktopWindow()
        self.window.show()
        self.panel = self.window.readiness
        self.panel.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        if self.panel.io_thread is not None:
            self.panel.io_thread.join(5)
        self.window.bridge.thread.join(5)
        self.window.loader.thread.join(5)
        self.window.poll()
        self.app.processEvents()
        from tests.qt_cleanup import dispose_widget
        dispose_widget(self.window)
        self.folder.cleanup()

    def until(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Readiness UI condition not reached')

    def select_source(self):
        source = self.root / 'local.mp4'
        source.write_bytes(b'local fixture source, not encoded media')
        self.window.source.setText(str(source))
        self.window.output.setText(str(self.root))
        return source

    def test_opening_does_not_run_checks_and_missing_form_allows_source_free_checks(self):
        report = {'stages': [{'id': 'inference', 'status': 'NOT VERIFIED', 'message': 'Explicit render required'}]}
        with patch('corridorkey.desktop_readiness.check_readiness', return_value=report) as check:
            self.assertFalse(check.called)
            self.panel.inspect()
            self.until(lambda: not self.panel.io_busy)
            check.assert_called_once()
            self.assertIsNone(check.call_args.args[0])
            self.assertEqual(self.panel.report, report)
            self.assertIn('NOT VERIFIED', self.panel.details.toPlainText())
            self.assertFalse(self.window.jobs)

    def test_prepare_is_immutable_and_run_is_explicit_single_use(self):
        from corridorkey.worker_protocol import job_to_wire
        source = self.select_source()
        original = self.window.job()
        self.panel.start.setValue(3)
        self.panel.count.setValue(4)
        self.panel.prepare()
        self.until(lambda: not self.panel.io_busy)
        prepared = self.panel.prepared
        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertEqual(self.window.job(), original)
        self.assertFalse(self.window.jobs)
        self.assertIn(prepared.output_dir, self.panel.details.toPlainText())
        self.window.source.setText(str(self.root / 'changed.mp4'))
        with patch.object(self.window.bridge, 'send', return_value='readiness-job') as send:
            self.panel.run()
            send.assert_called_once_with('render', {'job': job_to_wire(prepared)})
            self.assertEqual(self.window.jobs['readiness-job']['job'], prepared)
            self.panel.run()
            self.assertEqual(send.call_count, 1)
            self.panel.cancel()
            self.assertEqual(send.call_args.args[0], 'cancel')
            self.assertEqual(send.call_args.args[1]['target'], 'readiness-job')
        self.assertIsNone(self.panel.prepared)
        self.assertEqual(source.read_bytes(), b'local fixture source, not encoded media')

    def test_repeated_renders_keep_only_current_protection_without_mutating_scan(self):
        source = self.select_source()
        original = self.window.job()
        self.panel.report = {'schema_version': 1, 'protected_paths': [str(source)], 'stages': []}
        for index in range(100):
            self.panel.prepared = original
            self.panel.prepared_selection = None
            self.panel.mark_submitted(f'local-fixture-{index}')
            self.panel.observe({'request_id': f'local-fixture-{index}', 'kind': 'rejected',
                                'payload': {'error': 'fixture only; not submitted to worker'}})
        self.assertEqual(self.panel.report['protected_paths'], [str(source)])
        self.assertLessEqual(len(self.panel.render_protected), 3)
        assert self.panel.render_observation is not None
        self.assertEqual(self.panel.render_observation['request_id'], 'local-fixture-99')
        self.assertEqual(source.read_bytes(), b'local fixture source, not encoded media')

    def test_submission_respects_ownership_guards_and_history_capacity(self):
        from corridorkey.session import JobRequest
        prepared = JobRequest.create('.', str(self.root / 'fresh'))
        self.panel.prepared = prepared
        with patch.object(self.window.bridge, 'send') as send:
            for mode in ('closing', 'queue', 'benchmark', 'full'):
                self.window.closing = mode == 'closing'
                self.window.pending_queue_root = self.root if mode == 'queue' else None
                self.window.benchmark.waiting = mode == 'benchmark'
                self.window.jobs = {str(i): {} for i in range(128)} if mode == 'full' else {}
                self.window.submit_readiness(prepared)
                self.assertFalse(send.called, mode)
            self.window.jobs = {}
            self.window.closing = False
            self.window.pending_queue_root = None
            self.window.benchmark.waiting = False

    def test_checks_cancel_without_accepting_late_success_and_window_waits(self):
        entered, release = threading.Event(), threading.Event()

        def delayed_check(job, selection, *, cancellation):
            entered.set()
            release.wait(5)
            return {'stages': [], 'fixture': True}

        with patch('corridorkey.desktop_readiness.check_readiness', side_effect=delayed_check):
            try:
                self.panel.inspect()
                self.assertTrue(entered.wait(2))
                self.window.close()
                self.assertTrue(self.window.isVisible())
                self.assertTrue(self.panel.cancellation.cancelled)
            finally:
                release.set()
            self.until(lambda: not self.window.isVisible())
        self.assertIsNone(self.panel.report)

    def test_export_is_background_owned_and_does_not_overwrite(self):
        self.panel.report = {'stages': [], 'actual_inference': 'NOT VERIFIED'}
        target = self.root / 'report.json'
        with patch('corridorkey.desktop_readiness.QFileDialog.getSaveFileName', return_value=(str(target), 'JSON')):
            self.panel.choose_export()
            self.until(lambda: not self.panel.io_busy)
            self.assertEqual(json.loads(target.read_text()), self.panel.report)
            before = target.read_bytes()
            self.panel.choose_export()
            self.until(lambda: not self.panel.io_busy)
            self.assertEqual(target.read_bytes(), before)
            self.assertIn('Error', self.panel.status.text())

    def test_range_edit_invalidates_snapshot_and_terminal_releases_submission(self):
        from corridorkey.session import JobRequest
        self.panel.prepared = JobRequest.create('.', str(self.root / 'fresh'))
        self.panel.count.setValue(2)
        self.assertIsNone(self.panel.prepared)
        self.panel.prepared = JobRequest.create('.', str(self.root / 'fresh'))
        self.panel.mark_submitted('fixture')
        self.panel.observe({'request_id': 'other', 'kind': 'result', 'payload': {}})
        self.assertEqual(self.panel.submitted_id, 'fixture')
        self.panel.observe({'request_id': 'fixture', 'kind': 'failed', 'payload': {'error': 'fixture failure'}})
        self.assertIsNone(self.panel.submitted_id)
        self.assertIn('fixture failure', self.panel.status.text())

    def test_checks_replace_display_and_invalidate_prepared_job(self):
        from corridorkey.session import JobRequest
        self.panel.prepared = JobRequest.create('.', str(self.root / 'fresh'))
        with patch('corridorkey.desktop_readiness.check_readiness', return_value={'stages': []}):
            self.panel.inspect()
            self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.prepared)
        self.assertFalse(self.panel.run_button.isEnabled())

    def test_changed_assets_require_new_snapshot(self):
        from corridorkey.asset_inventory import AssetSelection
        from corridorkey.session import JobRequest
        self.panel.prepared = JobRequest.create('.', str(self.root / 'fresh'))
        self.window.asset_selection = AssetSelection(self.root)
        with patch.object(self.window.bridge, 'send') as send:
            self.panel.run()
            self.assertFalse(send.called)
        self.assertIn('assets changed', self.panel.status.text())

    def test_late_preparation_cannot_reactivate_changed_range(self):
        from corridorkey.readiness import prepare_short_render
        entered, release = threading.Event(), threading.Event()
        self.select_source()

        def delayed_prepare(job, start, count):
            entered.set()
            release.wait(5)
            return prepare_short_render(job, start, count)

        with patch('corridorkey.desktop_readiness.prepare_short_render', side_effect=delayed_prepare):
            try:
                self.panel.prepare()
                self.assertTrue(entered.wait(2))
                self.panel.start.setValue(7)
            finally:
                release.set()
            self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.prepared)
        self.assertIn('Range changed', self.panel.status.text())

    def completed_fixture(self, *, request_id='completed-fixture', count=2, requested_count=2):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.output_paths import output_layout
        from corridorkey.session import JobRequest
        source = self.root / request_id
        source.mkdir()
        for index in range(4):
            self.assertTrue(cv2.imwrite(str(source / f'{index:03}.png'), np.full((2, 3, 3), 30 + index * 20, np.uint8)))
        config = InferenceConfig(output_format='exr', outputs={'alpha', 'fg'}, start_frame=1, num_frames=requested_count)
        from corridorkey.worker_protocol import job_from_wire, job_to_wire
        job = job_from_wire(job_to_wire(JobRequest.create(source, self.root / f'{request_id}-output', config)))
        outputs = {}
        for role, item in output_layout(job.input_path, config)['outputs'].items():
            path = Path(job.output_dir) / item['path']
            path.mkdir(parents=True)
            outputs[role] = str(path)
            for index in range(count):
                pixels = np.full((2, 3) if role == 'alpha' else (2, 3, 3), .2 + index * .4 if role == 'alpha' else .25, np.float32)
                self.assertTrue(cv2.imwrite(str(path / f'{index:03}.exr'), pixels))
        record = {'status': 'complete', 'input': job.input_path, 'output_dir': job.output_dir,
                  'requested_settings': json.loads(job.config_json), 'writer_settings': json.loads(job.writer_json),
                  'hints': {'first_frame': None, 'full': None}, 'model_reused': False,
                  'model_initialization_seconds': .01, 'model_load_seconds': None,
                  'diagnostics': {'source': {'start_frame': 1, 'num_frames': count, 'native_hw': [2, 3], 'native_fps': 24.},
                                  'frames_submitted': count, 'outputs': outputs,
                                  'effective_settings': json.loads(job.config_json), 'execution': None}}
        self.panel.prepared = job
        self.panel.mark_submitted(request_id)
        return job, record

    def complete_fixture(self, record):
        self.panel.observe({'request_id': self.panel.latest_request_id, 'kind': 'result', 'payload': {'record': record}})

    def test_completed_output_explicit_decode_uses_captured_snapshot_and_image_free_report(self):
        job, record = self.completed_fixture()
        original = {path: path.read_bytes() for path in Path(job.input_path).iterdir()}
        self.window.source.setText(str(self.root / 'different-future-source'))
        self.complete_fixture(record)
        self.assertIsNone(self.panel.io_thread)  # completion did not trigger decoding
        self.assertTrue(self.panel.inspect_button.isEnabled())
        self.assertEqual(self.panel.latest_job, job)
        record['diagnostics']['outputs']['alpha'] = 'mutated outside record'  # captured copy is independent
        self.panel.inspection_frame.setValue(2)
        self.panel.inspect_completed()
        self.until(lambda: not self.panel.io_busy)
        pixmap = self.panel.thumbnail_pixmap
        assert pixmap is not None
        self.assertAlmostEqual(pixmap.toImage().pixelColor(0, 0).red(), 153, delta=1)
        report = self.panel.render_observation
        assert report is not None
        self.assertEqual(report['actual_inference'], 'NOT VERIFIED')
        self.assertEqual(report['output_decode']['status'], 'PASS')
        self.assertEqual(report['output_decode']['source_frame'], 2)
        self.assertEqual(report['output_decode']['export_frame'], 1)
        self.assertEqual(report['display']['status'], 'PASS')
        self.assertIn('NOT VERIFIED', report['pretrained_model_authenticity_and_quality'])
        target = self.root / 'inspection-report.json'
        with patch('corridorkey.desktop_readiness.QFileDialog.getSaveFileName', return_value=(str(target), 'JSON')):
            self.panel.choose_export()
            self.until(lambda: not self.panel.io_busy)
        exported = json.loads(target.read_text())['render_observation']
        self.assertEqual(exported['requested_job']['input_path'], job.input_path)
        self.assertNotIn('pixels', json.dumps(exported))
        self.assertEqual({path: path.read_bytes() for path in Path(job.input_path).iterdir()}, original)
        self.panel.inspection_mode.setCurrentText('Foreground')
        self.panel.inspect_completed()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNotNone(self.panel.thumbnail_pixmap)

    def test_old_completion_and_mismatched_settings_range_and_outputs_cannot_activate(self):
        _, record = self.completed_fixture()
        self.panel.observe({'request_id': 'stale-other-request', 'kind': 'result', 'payload': {'record': record}})
        self.assertIsNone(self.panel.completed_record)
        for field in ('requested_settings', 'range', 'output'):
            broken = json.loads(json.dumps(record))
            if field == 'requested_settings':
                broken['requested_settings']['model_scale'] = .5
            elif field == 'range':
                broken['diagnostics']['source']['start_frame'] = 0
            else:
                broken['diagnostics']['outputs']['alpha'] = str(self.root / 'unrelated')
            self.panel.prepared = self.panel.latest_job
            self.panel.mark_submitted('completed-fixture')
            self.complete_fixture(broken)
            self.assertIsNone(self.panel.completed_output)
            self.assertFalse(self.panel.inspect_button.isEnabled())
            self.assertIn('unavailable', self.panel.status.text())
        self.panel.inspect_completed()
        self.assertIn('requires this latest guided job', self.panel.status.text())

    def test_decode_failure_and_cancelled_late_success_remain_separate_from_inference(self):
        from corridorkey.desktop_preview import preview
        _, record = self.completed_fixture()
        self.complete_fixture(record)
        alpha = Path(record['diagnostics']['outputs']['alpha']) / '000.exr'
        original = alpha.read_bytes()
        alpha.write_bytes(b'not an image')
        self.panel.inspect_completed()
        self.until(lambda: not self.panel.io_busy)
        assert self.panel.render_observation is not None
        self.assertEqual(self.panel.render_observation['output_decode']['status'], 'FAIL')
        self.assertEqual(self.panel.render_observation['actual_inference'], 'NOT VERIFIED')
        alpha.write_bytes(original)
        entered, release = threading.Event(), threading.Event()

        def held_preview(**options):
            entered.set()
            release.wait(5)
            return preview(**options)

        with patch('corridorkey.desktop_readiness.preview', side_effect=held_preview):
            try:
                self.panel.inspect_completed()
                self.assertTrue(entered.wait(2))
                self.panel.cancel()
            finally:
                release.set()
            self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.thumbnail_pixmap)
        self.assertIn('CANCELLED', self.panel.render_observation['output_decode'])
        self.assertEqual(self.panel.render_observation['display'], 'NOT VERIFIED')

    def test_new_submission_and_close_reject_late_inspection_results(self):
        from corridorkey.desktop_preview import preview
        job, record = self.completed_fixture()
        self.complete_fixture(record)
        entered, release = threading.Event(), threading.Event()

        def held_preview(**options):
            entered.set()
            release.wait(5)
            return preview(**options)

        with patch('corridorkey.desktop_readiness.preview', side_effect=held_preview):
            try:
                self.panel.inspect_completed()
                self.assertTrue(entered.wait(2))
                self.panel.prepared = job
                self.panel.mark_submitted('newer-request')
            finally:
                release.set()
            self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.thumbnail_pixmap)
        self.assertIsNone(self.panel.completed_record)
        assert self.panel.render_observation is not None
        self.assertEqual(self.panel.render_observation['request_id'], 'newer-request')
        self.complete_fixture(record)
        entered.clear()
        release.clear()
        with patch('corridorkey.desktop_readiness.preview', side_effect=held_preview):
            try:
                self.panel.inspect_completed()
                self.assertTrue(entered.wait(2))
                self.panel.close()
                self.assertTrue(self.panel.isVisible())
            finally:
                release.set()
            self.until(lambda: not self.panel.io_busy and not self.panel.isVisible())
        self.assertIsNone(self.panel.thumbnail_pixmap)
        self.assertIn('CANCELLED', self.panel.render_observation['output_decode'])

    def test_shorter_count_and_engine_reported_cuda_are_explicit_evidence_only(self):
        job, record = self.completed_fixture(count=1, requested_count=2)
        execution = {'backend': 'pytorch', 'requested_device': 'cuda', 'stage2_calls': 1,
                     'stage2_frame_evaluations': 1, 'completed_prediction_frames': 1}
        record['diagnostics']['execution'] = execution
        self.complete_fixture(record)
        assert self.panel.render_observation is not None
        self.assertEqual(self.panel.render_observation['actual_inference'], 'NOT VERIFIED')
        self.assertEqual(self.panel.render_observation['engine_reported']['range']['observed_count'], 1)
        note = self.panel.render_observation['engine_reported']['range']['note']
        self.assertIn('reason/source-end truncation and exact requested count NOT VERIFIED', note)
        execution.update({'gpu': 'fixture metadata, no actual GPU execution', 'device_index': 0, 'cuda_version': 'fixture'})
        self.panel.prepared = job
        self.panel.mark_submitted('metadata-fixture')
        self.complete_fixture(record)
        self.assertEqual(self.panel.render_observation['actual_inference'], 'ENGINE REPORTED CUDA INFERENCE')
        self.assertIn('NOT VERIFIED', self.panel.render_observation['pretrained_model_authenticity_and_quality'])
        self.assertEqual(self.panel.inspection_frame.maximum(), 1)

    def test_inspecting_old_output_invalidates_newer_displayed_render_snapshot(self):
        job, record = self.completed_fixture()
        self.complete_fixture(record)
        self.window.populate_job(job)
        self.panel.prepare()
        self.until(lambda: not self.panel.io_busy)
        prepared = self.panel.prepared
        assert prepared is not None
        self.assertIn(prepared.output_dir, self.panel.details.toPlainText())
        self.assertTrue(self.panel.run_button.isEnabled())
        self.panel.inspect_completed()
        self.until(lambda: not self.panel.io_busy)
        self.assertIsNone(self.panel.prepared)
        self.assertFalse(self.panel.run_button.isEnabled())
        self.assertNotIn(prepared.output_dir, self.panel.details.toPlainText())
        with patch.object(self.window.bridge, 'send') as send:
            self.panel.run()
        send.assert_not_called()

    def test_qt_presentation_failure_preserves_successful_decode_evidence(self):
        from PySide6.QtGui import QPixmap
        _, record = self.completed_fixture()
        self.complete_fixture(record)
        null_pixmap = QPixmap()
        with patch('corridorkey.desktop_readiness.QPixmap') as pixmap_type:
            pixmap_type.fromImage.return_value = null_pixmap
            self.panel.inspect_completed()
            self.until(lambda: not self.panel.io_busy)
        assert self.panel.render_observation is not None
        self.assertEqual(self.panel.render_observation['output_decode']['status'], 'PASS')
        self.assertIn('FAIL', self.panel.render_observation['display'])
        self.assertIsNone(self.panel.thumbnail_pixmap)

    def test_oversized_record_and_failed_job_keep_inspection_unavailable(self):
        from corridorkey.worker_protocol import MAX_FRAME_BYTES
        job, record = self.completed_fixture()
        record['oversized'] = 'x' * MAX_FRAME_BYTES
        self.complete_fixture(record)
        self.assertIsNone(self.panel.completed_record)
        self.assertIn('4 MiB', self.panel.status.text())
        self.panel.prepared = job
        self.panel.mark_submitted('failure-fixture')
        self.panel.observe({'request_id': 'failure-fixture', 'kind': 'failed', 'payload': {'error': 'fixture failure'}})
        self.assertFalse(self.panel.inspect_button.isEnabled())
        self.panel.inspect_completed()
        self.assertIn('requires this latest guided job', self.panel.status.text())

    def test_capabilities_are_explicit_owned_and_invalidate_hidden_render_snapshot(self):
        from corridorkey.session import JobRequest
        self.select_source()
        job = self.window.job()
        self.panel.prepared = job
        fixture = {'status': 'complete', 'requested_backend': 'cuda', 'effective_backend': None,
                   'scope': 'test-only capability fixture; no GPU/model executed',
                   'stages': [{'stage': 'backend', 'status': 'FAIL', 'details': 'unavailable fixture'}]}
        with patch('corridorkey.desktop_readiness.check_capabilities', return_value=fixture) as probe:
            self.assertFalse(probe.called)
            self.panel.probe_capabilities()
            self.until(lambda: not self.panel.io_busy)
            probe.assert_called_once()
            self.assertEqual(probe.call_args.args[0], job)
        self.assertIsNone(self.panel.prepared)
        self.assertFalse(self.panel.run_button.isEnabled())
        assert self.panel.capability_observation is not None
        self.assertEqual(self.panel.capability_observation['requested_job']['input_path'], job.input_path)
        self.assertIsNone(self.panel.capability_observation['result']['effective_backend'])
        self.assertIn('actual inference', self.panel.status.text())
        self.assertFalse(self.window.jobs)
        self.assertEqual(self.window.job(), job)
        self.panel.prepared = JobRequest.create('.', str(self.root / 'unused'))
        with patch('corridorkey.desktop_readiness.check_readiness', return_value={'stages': []}):
            self.panel.inspect()
            self.until(lambda: not self.panel.io_busy)
        target = self.root / 'capabilities.json'
        with patch('corridorkey.desktop_readiness.QFileDialog.getSaveFileName', return_value=(str(target), 'JSON')):
            self.panel.choose_export()
            self.until(lambda: not self.panel.io_busy)
        self.assertEqual(json.loads(target.read_text())['capabilities']['result'], fixture)

    def test_capability_probe_missing_job_and_cancelled_partial_report(self):
        with patch('corridorkey.desktop_readiness.check_capabilities') as probe:
            self.panel.probe_capabilities()
            self.assertFalse(probe.called)
            self.assertIn('requires a valid captured job', self.panel.status.text())
        self.select_source()
        entered, release = threading.Event(), threading.Event()

        def held_probe(job, *, cancellation):
            entered.set()
            release.wait(5)
            self.assertTrue(cancellation.cancelled)
            return {'status': 'cancelled', 'stages': [], 'interrupted_stage': 'backend',
                    'scope': 'fixture; no subprocess or model executed'}

        with patch('corridorkey.desktop_readiness.check_capabilities', side_effect=held_probe):
            try:
                self.panel.probe_capabilities()
                self.assertTrue(entered.wait(2))
                self.panel.cancel()
            finally:
                release.set()
            self.until(lambda: not self.panel.io_busy)
        assert self.panel.capability_observation is not None
        self.assertEqual(self.panel.capability_observation['result']['status'], 'cancelled')
        self.assertEqual(self.panel.capability_observation['result']['interrupted_stage'], 'backend')

    def fail_fixture(self, record, *, kind='failed', error=None):
        self.panel.observe({'request_id': self.panel.latest_request_id, 'kind': kind,
                            'payload': {'record': record, 'error': error or {'type': 'FixtureFailure', 'message': 'fixture failure'}}})

    def test_failed_model_load_uses_terminal_checkpoint_without_coalesced_progress(self):
        job, record = self.completed_fixture()
        original = {path: path.read_bytes() for path in Path(job.input_path).iterdir()}
        record.update({'status': 'failed', 'failure_checkpoint': {'stage': 'model_load', 'completed': 0, 'total': 1},
                       'error': {'type': 'RuntimeError', 'message': 'local checkpoint incompatible',
                                 'notes': ['fixture note before cleanup', 'fixture cleanup note']},
                       'end_to_end_fps': 9999})
        record['diagnostics'].update({'source': None, 'outputs': {}, 'effective_settings': None, 'frames_submitted': 0,
                                      'last_checkpoint': record['failure_checkpoint']})
        self.fail_fixture(record)
        observation = self.panel.render_observation
        assert observation is not None
        self.assertEqual(observation['failure_checkpoint'], {'stage': 'model_load', 'completed': 0, 'total': 1})
        self.assertIn('not proof', observation['failure_checkpoint_scope'])
        self.assertEqual(observation['engine_reported']['error']['notes'], ['fixture note before cleanup', 'fixture cleanup note'])
        self.assertEqual(observation['engine_reported']['requested_settings'], json.loads(job.config_json))
        self.assertIsNone(observation['engine_reported']['effective_settings'])
        self.assertIsNone(observation['engine_reported']['end_to_end_fps'])
        self.assertIn('NOT VERIFIED', observation['engine_reported']['model_load_status'])
        self.assertEqual(observation['actual_inference'], 'NOT VERIFIED')
        self.assertIsNone(self.panel.completed_record)
        self.assertFalse(self.panel.inspect_button.isEnabled())
        self.assertIsNone(self.panel.io_thread)
        self.assertIn('model_load 0/1', self.panel.status.text())
        record['error']['notes'].append('outside mutation')
        self.assertEqual(len(observation['engine_reported']['error']['notes']), 2)
        self.assertEqual({path: path.read_bytes() for path in Path(job.input_path).iterdir()}, original)

    def test_early_validation_cancel_and_no_record_have_precise_unverified_state(self):
        job, record = self.completed_fixture()
        record.update({'status': 'cancelled', 'requested_settings': None, 'writer_settings': None,
                       'failure_checkpoint': {'stage': 'validate', 'completed': 0, 'total': 1},
                       'error': {'type': 'CancelledError', 'message': 'fixture cancelled before validation', 'notes': []}})
        record['diagnostics'] = {'last_checkpoint': record['failure_checkpoint'], 'frames_submitted': 0,
                                 'effective_settings': None, 'source': None, 'execution': None, 'outputs': {}}
        self.fail_fixture(record, kind='cancelled')
        observation = self.panel.render_observation
        assert observation is not None
        self.assertEqual(observation['terminal_kind'], 'cancelled')
        self.assertEqual(observation['failure_checkpoint']['stage'], 'validate')
        self.assertIsNone(observation['engine_reported']['requested_settings'])
        self.assertEqual(observation['requested_job']['config_json'], job.config_json)
        self.assertFalse(self.panel.inspect_button.isEnabled())
        self.panel.prepared = job
        self.panel.mark_submitted('cancelled-before-start')
        self.fail_fixture(None, kind='cancelled')
        observation = self.panel.render_observation
        assert observation is not None
        self.assertEqual(observation['record_validation']['status'], 'NOT VERIFIED')
        self.assertIn('No engine record', observation['record_validation']['error'])
        self.assertIsNone(observation['failure_checkpoint'])
        self.assertNotIn('engine_reported', observation)
        self.assertEqual(observation['actual_inference'], 'NOT VERIFIED')

    def test_partial_cuda_failure_is_reported_without_success_or_fps_and_exports(self):
        _, record = self.completed_fixture()
        record.update({'status': 'failed', 'failure_checkpoint': {'stage': 'stage2', 'completed': 1, 'total': 2},
                       'error': {'type': 'WriterError', 'message': 'fixture output failure', 'notes': ['partial files remain']}})
        record['diagnostics']['execution'] = {'backend': 'pytorch', 'requested_device': 'cuda', 'gpu': 'fixture metadata only',
                                              'device_index': 0, 'cuda_version': 'fixture', 'stage2_calls': 1,
                                              'stage2_frame_evaluations': 1, 'completed_prediction_frames': 1}
        record['diagnostics']['frames_submitted'] = 1
        self.fail_fixture(record)
        observation = self.panel.render_observation
        assert observation is not None
        self.assertEqual(observation['actual_inference'], 'NOT VERIFIED')
        self.assertIn('BEFORE FAILURE', observation['engine_reported_partial_cuda_work'])
        self.assertEqual(observation['engine_reported']['frames_submitted'], 1)
        self.assertIsNone(observation['engine_reported']['end_to_end_fps'])
        self.assertEqual(observation['output_decode'], 'NOT VERIFIED')
        target = self.root / 'failed-report.json'
        with patch('corridorkey.desktop_readiness.QFileDialog.getSaveFileName', return_value=(str(target), 'JSON')):
            self.panel.choose_export()
            self.until(lambda: not self.panel.io_busy)
        exported = json.loads(target.read_text())['render_observation']
        self.assertEqual(exported['engine_reported']['error']['notes'], ['partial files remain'])
        self.assertNotIn('pixels', json.dumps(exported))

    def test_failed_record_binding_mismatch_old_request_and_oversize_are_not_adopted(self):
        from corridorkey.worker_protocol import MAX_FRAME_BYTES
        job, record = self.completed_fixture()
        record['status'] = 'failed'
        self.panel.observe({'request_id': 'old-request', 'kind': 'failed', 'payload': {'record': record}})
        self.assertEqual(self.panel.submitted_id, 'completed-fixture')
        self.assertIsNone(self.panel.failed_record)
        for field in ('input', 'output_dir', 'requested_settings', 'writer_settings', 'hints', 'source_range', 'frames_submitted', 'oversized'):
            broken = json.loads(json.dumps(record))
            if field in {'input', 'output_dir'}:
                broken[field] = str(self.root / 'wrong')
            elif field == 'requested_settings':
                broken[field]['model_scale'] = .5
            elif field == 'writer_settings':
                broken[field]['codec'] = 'different'
            elif field == 'hints':
                broken[field]['first_frame'] = str(self.root / 'different-hint')
            elif field == 'source_range':
                broken['diagnostics']['source']['start_frame'] = 99
            elif field == 'frames_submitted':
                broken['diagnostics']['frames_submitted'] = 999
            else:
                broken['oversized'] = 'x' * MAX_FRAME_BYTES
            self.panel.prepared = job
            self.panel.mark_submitted('failure-binding-fixture')
            self.fail_fixture(broken)
            observation = self.panel.render_observation
            assert observation is not None
            self.assertIsNone(self.panel.failed_record, field)
            self.assertEqual(observation['record_validation']['status'], 'NOT VERIFIED', field)
            self.assertNotIn('engine_reported', observation, field)
            self.assertFalse(self.panel.inspect_button.isEnabled(), field)

    def test_failed_last_checkpoint_fallback_and_missing_stage_never_guess_from_progress(self):
        job, record = self.completed_fixture()
        record['status'] = 'failed'
        record['diagnostics']['last_checkpoint'] = {'stage': 'encoder_preflight', 'completed': 0, 'total': 1}
        self.fail_fixture(record)
        assert self.panel.render_observation is not None
        self.assertEqual(self.panel.render_observation['failure_checkpoint']['stage'], 'encoder_preflight')
        self.panel.prepared = job
        self.panel.mark_submitted('no-failure-stage')
        record['diagnostics']['last_checkpoint'] = None
        self.fail_fixture(record)
        self.assertIsNone(self.panel.render_observation['failure_checkpoint'])
        self.assertIn('no engine checkpoint', self.panel.status.text())
