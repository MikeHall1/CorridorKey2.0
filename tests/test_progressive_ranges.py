"""Bounded captured-range scheduling and existing desktop-owner integration."""
import hashlib
import json
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from corridorkey.progressive_preview import (
    PreviewArtifact,
    PreviewLimits,
    ProgressivePreview,
)
from tests.test_progressive_preview import context, stages


def retire(scheduler, request):
    return scheduler.complete(request, PreviewArtifact(request.identity, b'x'))


def drain(scheduler, *, idle=False):
    observed = []
    while (request := scheduler.next_request(idle=idle)) is not None:
        observed.append(request)
        retire(scheduler, request)
        if len(observed) > 200:
            raise AssertionError('Unbounded/repeated background work')
    return observed


class ProgressiveRangeTests(unittest.TestCase):
    def test_priority_idle_selection_then_remaining_and_final_coverage(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True,
                                       limits=PreviewLimits(surround_radius=1, upcoming_frames=1))
        scheduler.set_ranges(selected=(25, 27), remaining=True)
        scheduler.focus(15, playing=True)
        immediate = drain(scheduler)
        self.assertEqual([r.priority for r in immediate[:3]], ['current'] * 3)
        self.assertEqual([r.source_frame for r in immediate if r.priority == 'surround'], [16, 14])
        self.assertEqual([r.source_frame for r in immediate if r.priority == 'upcoming'], [17])
        self.assertFalse(any(r.priority in ('selected', 'remaining') for r in immediate))
        background = drain(scheduler, idle=True)
        selected = [r for r in background if r.priority == 'selected']
        self.assertEqual([r.source_frame for r in selected], [25, 26])
        self.assertTrue(all(r.stage.name == 'final' for r in selected))
        bands = [r.priority for r in background]
        self.assertLess(bands.index('selected'), bands.index('remaining'))
        state = scheduler.range_state
        self.assertEqual((state.total_frames, state.completed_frames, state.failed_frames), (20, 20, 0))
        self.assertIsNone(scheduler.next_request(idle=True))
        scheduler.focus(19)
        drain(scheduler, idle=True)
        self.assertIsNone(scheduler.next_request(idle=True))

    def test_eviction_does_not_repeat_background_but_visible_focus_may_rerender(self):
        limits = PreviewLimits(surround_radius=0, upcoming_frames=0, max_cache_entries=2,
                               max_result_bytes=1, max_cache_bytes=2)
        scheduler = ProgressivePreview(context(), (stages()[-1],), enabled=True, limits=limits)
        scheduler.set_ranges(selected=(20, 25))
        observed = drain(scheduler, idle=True)
        self.assertEqual([r.source_frame for r in observed], [10, 20, 21, 22, 23, 24])
        self.assertEqual(scheduler.range_state.completed_frames, 5)
        self.assertLessEqual(scheduler.snapshot().cache_entries, 2)
        scheduler.focus(12)
        self.assertEqual([r.source_frame for r in drain(scheduler, idle=True)], [12])

        scheduler.focus(20)
        self.assertEqual([r.source_frame for r in drain(scheduler, idle=True)], [20])
        self.assertEqual(scheduler.range_state.completed_frames, 5)

    def test_large_context_is_lazy_and_unsupported_work_is_refused_before_admission(self):
        scheduler = ProgressivePreview(context(source_end=2**50), (stages()[-1],), enabled=True,
                                       limits=PreviewLimits(surround_radius=0, upcoming_frames=0))
        scheduler.set_ranges(selected=(2**40, 2**40 + 3))
        self.assertEqual(scheduler.range_state.total_frames, 3)
        self.assertLessEqual(scheduler.snapshot().queued, scheduler.limits.max_queued)
        self.assertEqual([r.source_frame for r in drain(scheduler, idle=True)],
                         [10, 2**40, 2**40 + 1, 2**40 + 2])
        generation = scheduler.generation
        invalid: tuple[dict[str, Any], ...] = ({'remaining': True}, {'selected': (9, 10)}, {'selected': (20, 20)},
                        {'selected': (10, 267)}, {'selected': (True, 20)}, {'remaining': 1})
        for options in invalid:
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                scheduler.set_ranges(**options)
        self.assertEqual(scheduler.generation, generation)
        self.assertEqual(scheduler.range_state.completed_frames, 3)

    def test_seek_rebind_cancel_and_attempt_budget_keep_stale_work_owned(self):
        scheduler = ProgressivePreview(context(), (stages()[-1],), enabled=True,
                                       limits=PreviewLimits(surround_radius=0, upcoming_frames=0,
                                                            max_range_requests=1))
        scheduler.set_ranges(selected=(20, 22))
        drain(scheduler)
        old = scheduler.next_request(idle=True)
        assert old is not None
        scheduler.focus(11)
        self.assertIsNone(scheduler.next_request(idle=True))
        self.assertFalse(retire(scheduler, old))
        self.assertEqual(scheduler.range_state.completed_frames, 0)
        drain(scheduler)
        self.assertIsNone(scheduler.next_request(idle=True))
        self.assertTrue(scheduler.range_state.budget_exhausted)
        scheduler.set_ranges(selected=(20, 21))
        pending = scheduler.next_request(idle=True)
        assert pending is not None
        scheduler.cancel()
        self.assertFalse(retire(scheduler, pending))
        self.assertIsNone(scheduler.next_request(idle=True))
        scheduler.bind(context(source_identity='new'))
        self.assertEqual(scheduler.range_state.total_frames, 0)

    def test_failure_is_counted_not_retried_after_eviction_or_seek(self):
        scheduler = ProgressivePreview(context(), (stages()[-1],), enabled=True,
                                       limits=PreviewLimits(surround_radius=0, upcoming_frames=0))
        scheduler.set_ranges(selected=(20, 22))
        drain(scheduler)
        failed = scheduler.next_request(idle=True)
        assert failed is not None
        self.assertTrue(scheduler.fail(failed, 'explicit provider failure'))
        observed = drain(scheduler, idle=True)
        self.assertEqual([r.source_frame for r in observed], [21])
        self.assertEqual((scheduler.range_state.completed_frames, scheduler.range_state.failed_frames), (1, 1))
        scheduler.focus(12)
        self.assertEqual([r.source_frame for r in drain(scheduler, idle=True)], [12])


class ProgressiveRangeDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_desktop_window import DesktopWindowTests
        DesktopWindowTests.setUpClass()

    def setUp(self):
        from tests.test_desktop_window import DesktopWindowTests
        self.fixture = DesktopWindowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.window, self.root, self.app = self.fixture.window, self.fixture.root, self.fixture.app
        self.panel = self.window.progressive

    def until(self, predicate):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.window.poll()
            self.panel.poll()
            if predicate():
                return
            time.sleep(.01)
        self.fail(self.panel.status.text())

    def configure(self, *, gate=None):
        import cv2
        import numpy as np

        from corridorkey.config import InferenceConfig
        from corridorkey.session import JobRequest

        source = self.root / 'frames'
        source.mkdir()
        for index in range(12):
            self.assertTrue(cv2.imwrite(str(source / f'{index:03}.png'), np.full((24, 32, 3), index * 15, np.uint8)))
        controls: dict[str, object] = {'export_fg': True}
        if gate is not None:
            controls['gate'] = str(gate)
        (source / '.worker-control.json').write_text(json.dumps(controls))
        job = JobRequest.create(source, self.root / 'exports',
                                InferenceConfig(output_format='exr', outputs={'alpha', 'fg'}, auto_vram=False,
                                                start_frame=2, num_frames=10))
        self.window.populate_job(job, frame=3)
        scratch = self.root / 'scratch'
        scratch.mkdir()
        self.panel.scratch.setText(str(scratch))
        self.panel.selected_range.setChecked(True)
        self.panel.selected_in.setText('9')
        self.panel.selected_out.setText('11')
        return job, scratch, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}

    def test_actual_selected_then_remaining_source_mapping_progress_and_no_export(self):
        from corridorkey.worker_protocol import job_to_wire
        job, scratch, before = self.configure()
        original = job_to_wire(self.window.job())
        self.panel.remaining_range.setChecked(True)
        observed = []
        next_request = ProgressivePreview.next_request
        def observe(scheduler, **kwargs):
            request = next_request(scheduler, **kwargs)
            if request is not None:
                observed.append((request.source_frame, request.priority))
            return request
        with patch.object(ProgressivePreview, 'next_request', observe):
            self.panel.enabled.setChecked(True)
            self.panel.begin()
            self.until(lambda: self.panel.scheduler is not None and self.panel.scheduler.range_state.completed_frames == 10)
        self.assertEqual([frame for frame, band in observed if band == 'selected'], [9, 10])
        remaining = [frame for frame, band in observed if band == 'remaining']
        self.assertEqual(remaining, [6, 7, 8, 11])
        self.assertTrue(all(2 <= frame < 12 for frame, _ in observed))
        self.assertEqual(len(observed), 10)
        self.panel._display()
        self.assertIn('10/10 completed', self.panel.range_status.text())
        self.assertEqual(json.loads(self.panel.details.toPlainText())['range_work']['selected_source_range'], [9, 11])
        self.assertEqual(job_to_wire(self.window.job()), original)
        scheduler = self.panel.scheduler
        assert scheduler is not None
        for result in scheduler._cache.values():
            self.assertEqual(json.loads(result.artifact.metadata_json)['execution_observations']['fixture_loads'], 1)
        self.panel.enabled.setChecked(False)
        self.until(lambda: not self.panel.busy)
        self.assertEqual(list(scratch.iterdir()), [])
        self.assertFalse(self.window.jobs)
        self.assertFalse(Path(job.output_dir).exists())
        self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}, before)

    def test_invalid_absolute_range_refuses_before_model_work(self):
        _, scratch, _ = self.configure()
        self.panel.selected_in.setText('1')
        self.panel.enabled.setChecked(True)
        self.panel.begin()
        self.until(lambda: self.panel.failure is not None)
        self.assertIn('Selected source In', self.panel.status.text())
        self.assertIsNone(self.panel.worker_id)
        self.assertIsNone(self.panel.recipe)
        self.assertEqual(list(scratch.iterdir()), [])
        self.assertFalse(self.window.jobs)

    def test_scope_edit_cancels_owned_worker_and_requires_explicit_recapture(self):
        gate = self.root / 'worker-gate'
        _, scratch, _ = self.configure(gate=gate)
        self.panel.enabled.setChecked(True)
        self.panel.begin()
        self.until(lambda: self.panel.worker_id is not None)
        request = self.panel.request
        assert request is not None
        self.panel.selected_out.setText('12')
        self.assertFalse(self.panel.enabled.isChecked())
        self.assertIsNone(self.panel.recipe)
        gate.write_text('allow')
        self.until(lambda: not self.panel.busy)
        scheduler = self.panel.scheduler
        assert scheduler is not None
        self.assertIsNone(scheduler.snapshot().pending)
        self.assertGreater(scheduler.snapshot().stale_completions, 0)
        self.panel.enabled.setChecked(True)
        self.panel.poll()
        self.assertIsNone(self.panel.recipe)
        self.assertIsNone(self.panel.worker_id)
        self.panel.enabled.setChecked(False)
        self.assertEqual(list(scratch.iterdir()), [])
