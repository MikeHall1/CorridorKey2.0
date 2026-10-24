"""Native EXR host service scopes; no encoder-only CPU/GPU performance claim."""
import os
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from corridorkey.control import CancelledError
from corridorkey.diagnostics import RunDiagnostics
from corridorkey.feasibility import project, view
from corridorkey.video_io import ExrSequenceWriter
from tests import test_output_telemetry_workflow as workflow
from tests.test_feasibility import inference_fixture

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
NAME = 'native_image_encode_and_write'


class NativeWriteObservationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.frames = np.array([[[[-2., 0., 3.], [1e-8, .5, 2.]]]], np.float32)

    def writer(self, name, **kwargs):
        writer = ExrSequenceWriter(str(self.root / name), 24, (2, 1), dtype='float', **kwargs)
        self.addCleanup(lambda thread=writer.worker.thread: self.assertFalse(thread.is_alive()))
        return writer

    def test_real_exr_profile_off_on_bytes_pixels_and_callback_order(self):
        diagnostics = RunDiagnostics(profile_stages=True)
        diagnostics.synchronize = lambda: self.fail('Host write must not synchronize CUDA')
        completed = []
        oracle = self.writer('off')
        oracle.write_many(self.frames)
        oracle.release()
        profiled = self.writer('on', host_observer=diagnostics,
            on_frame_written=lambda index, at: completed.append((index, len(diagnostics.stages))))
        profiled.write_many(self.frames)
        profiled.release()
        self.assertIsNone(profiled.host_observer)
        profiled.release()  # A repeated successful close remains retired and does not write again.
        paths = [self.root / side / 'frame_000000.exr' for side in ('off', 'on')]
        self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
        np.testing.assert_array_equal(cv2.imread(str(paths[1]), cv2.IMREAD_UNCHANGED), self.frames[0])
        self.assertEqual(completed, [(0, 1)])
        row = diagnostics.stages[0]
        self.assertEqual((row['name'], row['method'], row['status'], row['count']), (NAME, 'host_wall', 'complete', 1))

    def test_half_alpha_and_disabled_observer_keep_original_storage_and_clock_contract(self):
        alpha = np.array([[[[0.], [1.]]]], np.float32)
        clock = Mock(return_value=0.)
        disabled = RunDiagnostics(clock=clock)
        for name, observer in (('default', None), ('disabled', disabled)):
            writer = ExrSequenceWriter(str(self.root / name), 24, (2, 1), channels=1,
                dtype='half', host_observer=observer, on_frame_written=lambda *args: False)
            writer.write_many(alpha)
            writer.release()
            self.assertFalse(writer.worker.thread.is_alive())
            self.assertIsNone(writer.observation_error)
        a, b = (self.root / name / 'frame_000000.exr' for name in ('default', 'disabled'))
        self.assertEqual(a.read_bytes(), b.read_bytes())
        np.testing.assert_array_equal(cv2.imread(str(b), cv2.IMREAD_UNCHANGED), alpha[0, :, :, 0])
        clock.assert_called_once()
        self.assertEqual(disabled.stages, [])

    def test_native_false_and_exception_remain_primary_failed_observations(self):
        for outcome in (False, RuntimeError('native primary')):
            with self.subTest(outcome=outcome):
                diagnostics = RunDiagnostics(profile_stages=True)
                writer = self.writer(str(type(outcome)), host_observer=diagnostics)
                with patch('corridorkey.video_io.cv2.imwrite', return_value=False,
                           side_effect=outcome if isinstance(outcome, Exception) else None):
                    writer.write_many(self.frames)
                    with self.assertRaises(RuntimeError) as error:
                        writer.release()
                if isinstance(outcome, Exception):
                    self.assertIs(error.exception, outcome)
                else:
                    self.assertIn('cv2.imwrite failed', str(error.exception))
                self.assertEqual(diagnostics.stages[0]['status'], 'failed')
                self.assertFalse(writer.worker.thread.is_alive())
                self.assertIsNone(writer.host_observer)

    def test_worker_body_is_lock_free_no_array_copy_and_retains_native_parameters(self):
        def exercise():
            diagnostics = RunDiagnostics(profile_stages=True)
            owner = threading.get_ident()
            called = []
            writer = self.writer('thread', host_observer=diagnostics)
            stage_lock = diagnostics._stage_lock
            self.assertIsNotNone(stage_lock)
            assert stage_lock is not None

            def native(path, frame, parameters):
                self.assertNotEqual(threading.get_ident(), owner)
                self.assertTrue(np.shares_memory(frame, self.frames))
                self.assertIs(parameters, writer.imwrite_params)
                self.assertTrue(stage_lock.acquire(blocking=False))
                stage_lock.release()
                called.append(path)
                return True

            with patch('corridorkey.video_io.cv2.imwrite', side_effect=native):
                writer.write_many(self.frames)
                writer.release()
            self.assertEqual(len(called), 1)
            return weakref.ref(diagnostics)

        reference = exercise()
        self.assertIsNone(reference(), 'Retired observer must not create a diagnostics ownership cycle')

    def test_release_retains_live_observer_then_clears_on_confirmed_retirement(self):
        diagnostics = RunDiagnostics(profile_stages=True)
        entered, unblock = threading.Event(), threading.Event()
        writer = self.writer('live', host_observer=diagnostics, timeout_seconds=.02)

        def native(*args):
            entered.set()
            if not unblock.wait(2):
                raise AssertionError('Test must release its owned native writer')
            return True

        with patch('corridorkey.video_io.cv2.imwrite', side_effect=native):
            try:
                writer.write_many(self.frames)
                self.assertTrue(entered.wait(1))
                with self.assertRaises(TimeoutError):
                    writer.release()
                self.assertTrue(writer.worker.thread.is_alive())
                self.assertIs(writer.host_observer, diagnostics)
            finally:
                unblock.set()
                writer.worker.thread.join(1)
            self.assertFalse(writer.worker.thread.is_alive())
            with self.assertRaises(TimeoutError):
                writer.release()
        self.assertIsNone(writer.host_observer)

    def test_cancel_and_callback_error_preserve_existing_native_boundaries(self):
        diagnostics = RunDiagnostics(profile_stages=True)
        cancelled = threading.Event()

        def check():
            if cancelled.is_set():
                raise CancelledError('cancel after first native write')

        writer = self.writer('cancel', host_observer=diagnostics, check=check)
        def native(*args):
            cancelled.set()
            return True
        with patch('corridorkey.video_io.cv2.imwrite', side_effect=native) as writes:
            writer.write_many(np.concatenate([self.frames, self.frames]))
            with self.assertRaises(CancelledError):
                writer.release()
        self.assertEqual(writes.call_count, 1)
        self.assertEqual(diagnostics.stages[0]['count'], 1)
        diagnostics = RunDiagnostics(profile_stages=True)
        def callback(*args):
            raise ValueError('diagnostic callback failure')
        writer = self.writer('callback', host_observer=diagnostics, on_frame_written=callback)
        writer.write_many(self.frames)
        writer.release()
        self.assertIn('diagnostic callback failure', writer.observation_error or '')
        self.assertEqual(diagnostics.stages[0]['status'], 'complete')

    def test_feasibility_maps_combined_native_service_with_explicit_scope(self):
        diagnostics = RunDiagnostics(profile_stages=True)
        with diagnostics.stage(NAME, gpu=False):
            pass
        report = inference_fixture()
        report['runs'][0]['jobs'][0]['diagnostics'].update(diagnostics.snapshot())
        projection = project(report)
        row = projection['samples'][0]['jobs'][0]
        self.assertEqual(row['stage_lanes']['encoding'][0]['name'], NAME)
        self.assertIn('not pure encoding', view(projection))

    def test_actual_runtime_wires_only_profiled_writers_and_retires_them(self):
        fixture = workflow.OutputTelemetryWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for profile in (False, True):
            event = fixture.terminal(fixture.client.submit(fixture.job(str(profile), profile=profile)))
            self.assertEqual(event['kind'], 'result', event)
            record = event['payload']['record']
            stages = [s for s in record['diagnostics']['stages'] if s['name'] == NAME]
            self.assertEqual(sum(s['count'] for s in stages), 10 if profile else 0)
        fixture.close_client()
