"""Deterministic scheduler-only fixtures; no neural/render performance claims."""
import json
import threading
import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Any

from corridorkey.control import CancelledError
from corridorkey.progressive_preview import (
    PreviewArtifact,
    PreviewContext,
    PreviewLimits,
    PreviewRequest,
    PreviewResult,
    PreviewStage,
    ProgressivePreview,
)


def context(**overrides: Any):
    settings = {'outputs': ['alpha', 'fg'], 'model_scale': 1., 'quick_key': False}
    provenance = {'source_sha256': 'a' * 64, 'model_sha256': 'b' * 64,
                  'correction_identity': 'c' * 64, 'decoder': 'fixture-v1',
                  'purpose': 'preview-only', 'temporal_context': 'explicit caller recipe'}
    values: dict[str, Any] = {'source_identity': 'a' * 64, 'requested_settings': settings, 'provenance': provenance,
                              'source_start': 10, 'source_end': 30}
    values.update(overrides)
    return PreviewContext.capture(**values)


def stages():
    return tuple(PreviewStage.capture(name, {'max_side': side, 'scope': 'preview-only', 'temporal_context': 'restarted'},
                                     estimated_work_bytes=side)
                 for name, side in (('draft', 128), ('refined', 256), ('final', 512)))


def finish(scheduler, request=None, payload=b'RGB', metadata=b'{}'):
    request = request if request is not None else scheduler.next_request()
    assert request is not None
    return scheduler.complete(request, PreviewArtifact(request.identity, payload, metadata))


def displayed(scheduler: ProgressivePreview) -> PreviewResult:
    result = scheduler.snapshot().result
    assert result is not None
    return result


def next_work(scheduler: ProgressivePreview) -> PreviewRequest:
    request = scheduler.next_request()
    assert request is not None
    return request


class ProgressivePreviewTests(unittest.TestCase):
    def test_disabled_default_and_explicit_current_ladder_no_export_mutation(self):
        captured, configured = context(), stages()
        scheduler = ProgressivePreview(captured, configured)
        self.assertIsNone(scheduler.next_request())
        self.assertEqual(scheduler.snapshot().status, 'disabled')
        scheduler.set_enabled(True)
        for expected in ('draft', 'refined', 'final'):
            request = scheduler.next_request()
            assert request is not None
            self.assertEqual(request.stage.name, expected)
            self.assertEqual(request.priority, 'current')
            self.assertEqual(request.context.requested_settings_json, captured.requested_settings_json)
            self.assertEqual(request.stage.effective_settings_json, configured[('draft','refined','final').index(expected)].effective_settings_json)
            self.assertTrue(finish(scheduler, request))
            self.assertEqual(scheduler.snapshot().status, expected)
        self.assertEqual(json.loads(captured.requested_settings_json)['model_scale'], 1.)
        self.assertFalse(json.loads(captured.requested_settings_json)['quick_key'])
        self.assertEqual(displayed(scheduler).request.stage.name, 'final')

    def test_nearby_before_upcoming_and_idle_refinements_only_explicit_idle(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        scheduler.focus(15, playing=True)
        observed = []
        while (request := scheduler.next_request()) is not None:
            observed.append((request.source_frame, request.priority, request.stage.name))
            finish(scheduler, request)
        self.assertEqual(observed[:3], [(15,'current','draft'), (15,'current','refined'), (15,'current','final')])
        self.assertEqual([frame for frame, band, _ in observed if band == 'surround'], [16,14,17,13])
        self.assertEqual([frame for frame, band, _ in observed if band == 'upcoming'], [18,19,20,21])
        self.assertEqual(scheduler.snapshot().status, 'final')
        self.assertEqual(displayed(scheduler).request.source_frame, 15)
        request = scheduler.next_request(idle=True)
        assert request is not None
        self.assertEqual((request.priority, request.stage.name), ('idle', 'refined'))
        self.assertEqual(displayed(scheduler).request.source_frame, 15)

    def test_reverse_direction_boundaries_and_global_frame_coordinates(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        scheduler.focus(11, direction=-1, playing=True)
        frames = []
        while (request := scheduler.next_request()) is not None:
            frames.append(request.source_frame)
            finish(scheduler, request)
        self.assertTrue(all(10 <= frame < 30 for frame in frames))
        self.assertEqual(frames[:3], [11,11,11])
        self.assertEqual(frames[3:6], [10,12,13])
        self.assertEqual(scheduler.snapshot(10).source_frame, 10)
        invalid_frames: list[Any] = [9,30,True,11.0]
        for invalid in invalid_frames:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scheduler.focus(invalid)

    def test_seek_keeps_native_owner_then_rejects_old_completion(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        old = scheduler.next_request()
        assert old is not None
        scheduler.focus(20)
        self.assertIsNone(scheduler.next_request())
        self.assertIs(scheduler.snapshot().pending, old)
        self.assertEqual(scheduler.snapshot().reserved_work_bytes, old.stage.estimated_work_bytes)
        with self.assertRaises(CancelledError):
            scheduler.check(old)
        self.assertFalse(finish(scheduler, old))
        current = scheduler.next_request()
        assert current is not None
        self.assertEqual(current.source_frame, 20)
        self.assertGreater(current.generation, old.generation)
        self.assertEqual(scheduler.snapshot().stale_completions, 1)
        self.assertEqual(scheduler.snapshot().cache_bytes, 0)
        self.assertFalse(finish(scheduler, old))
        self.assertIs(scheduler.snapshot().pending, current)

    def test_rebind_settings_source_corrections_and_decode_invalidate_exact_reuse(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        finish(scheduler)
        original = scheduler.snapshot().result
        assert original is not None
        changes = [context(source_identity='different-source'),
                   context(requested_settings={'model_scale': .5}),
                   context(provenance={'correction_identity': 'changed'}),
                   context(provenance={'decoder': 'changed'})]
        for changed in changes:
            with self.subTest(identity=changed.identity):
                scheduler.bind(changed)
                stale = scheduler.snapshot()
                self.assertEqual(stale.status, 'stale')
                self.assertIs(stale.result, original)
                request = scheduler.next_request()
                assert request is not None
                self.assertNotEqual(request.identity, original.request.identity)
                scheduler.fail(request, 'fixture unavailable, no fallback')
        scheduler.bind(context())
        self.assertIs(scheduler.snapshot().result, original)
        self.assertEqual(next_work(scheduler).stage.name, 'refined')

    def test_exact_cached_final_satisfies_lower_stages_without_downgrade(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        for _ in range(3):
            finish(scheduler)
        cached = scheduler.snapshot().result
        scheduler.focus(10)
        self.assertIs(scheduler.snapshot().result, cached)
        next_work = scheduler.next_request()
        assert next_work is not None
        self.assertNotEqual(next_work.source_frame, 10)
        self.assertEqual(scheduler.snapshot().status, 'final')

    def test_disable_and_cancel_retain_owner_until_terminal_no_overlap(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        old = scheduler.next_request()
        assert old is not None
        scheduler.set_enabled(False)
        scheduler.set_enabled(True)
        self.assertIsNone(scheduler.next_request())
        self.assertFalse(scheduler.fail(old, 'cancelled native call retired'))
        request = scheduler.next_request()
        assert request is not None
        scheduler.cancel()
        with self.assertRaises(CancelledError):
            scheduler.check(request)
        self.assertIsNone(scheduler.next_request())
        self.assertFalse(finish(scheduler, request))
        self.assertEqual(scheduler.snapshot().status, 'cancelled')
        scheduler.focus(12)
        self.assertEqual(next_work(scheduler).source_frame, 12)

    def test_bad_provenance_or_oversized_result_release_owner_without_publication(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True,
                                       limits=PreviewLimits(max_result_bytes=3, max_cache_bytes=3))
        request = scheduler.next_request()
        assert request is not None
        self.assertFalse(scheduler.complete(request, PreviewArtifact('different', b'RGB')))
        self.assertEqual(scheduler.snapshot().status, 'error')
        self.assertEqual(scheduler.snapshot().cache_bytes, 0)
        request = scheduler.next_request()
        assert request is not None
        self.assertFalse(finish(scheduler, request, b'over'))
        self.assertIsNone(scheduler.snapshot().pending)
        self.assertEqual(scheduler.snapshot().cache_entries, 0)
        error = scheduler.snapshot().error
        assert error is not None
        self.assertIn('byte budget', error)

    def test_failure_never_retries_or_changes_recipe_implicitly(self):
        scheduler = ProgressivePreview(context(), (stages()[2],), enabled=True,
                                       limits=PreviewLimits(surround_radius=0, upcoming_frames=0))
        request = scheduler.next_request()
        assert request is not None
        self.assertTrue(scheduler.fail(request, 'actual provider unavailable'))
        self.assertIsNone(scheduler.next_request(idle=True))
        self.assertEqual(scheduler.snapshot().status, 'error')
        self.assertEqual(scheduler.snapshot().error, 'actual provider unavailable')
        scheduler.focus(10)
        self.assertEqual(next_work(scheduler).stage.name, 'final')

    def test_queue_cache_entries_and_payload_bytes_remain_bounded(self):
        limits = PreviewLimits(max_queued=5, max_cache_entries=2, max_cache_bytes=6, max_result_bytes=3,
                               surround_radius=16, upcoming_frames=64)
        scheduler = ProgressivePreview(context(), stages(), enabled=True, limits=limits)
        scheduler.focus(20, playing=True)
        self.assertEqual(scheduler.snapshot().queued, 5)
        count = 0
        while (request := scheduler.next_request(idle=True)) is not None:
            self.assertLessEqual(request.stage.estimated_work_bytes, limits.max_work_bytes)
            self.assertIsNone(scheduler.next_request(idle=True))
            finish(scheduler, request)
            state = scheduler.snapshot()
            self.assertLessEqual(state.cache_entries, 2)
            self.assertLessEqual(state.cache_bytes, 6)
            count += 1
        self.assertEqual(count, 5)

    def test_prefetch_eviction_preserves_focused_final_under_tight_cache(self):
        for count in (1,2):
            with self.subTest(entries=count):
                scheduler = ProgressivePreview(context(), stages(), enabled=True,
                                               limits=PreviewLimits(max_cache_entries=count, max_cache_bytes=count*3, max_result_bytes=3))
                for _ in range(3):
                    finish(scheduler)
                focused = displayed(scheduler)
                self.assertEqual(focused.request.stage.name, 'final')
                while (request := scheduler.next_request(idle=True)) is not None:
                    accepted = finish(scheduler, request)
                    if count == 1:
                        self.assertFalse(accepted)
                    self.assertIs(displayed(scheduler), focused)
                    self.assertEqual(scheduler.snapshot().status, 'final')
                    self.assertLessEqual(scheduler.snapshot().cache_entries, count)

    def test_attempt_identity_rejects_old_same_recipe_artifact_but_keeps_cache_key(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        original = next_work(scheduler)
        artifact = PreviewArtifact(original.identity, b'old')
        scheduler.fail(original, 'first attempt failed')
        scheduler.focus(original.source_frame)
        current = next_work(scheduler)
        self.assertNotEqual(current.identity, original.identity)
        self.assertEqual(current.cache_identity, original.cache_identity)
        self.assertFalse(scheduler.complete(current, artifact))
        self.assertEqual(scheduler.snapshot().cache_entries, 0)
        self.assertIn('identity differs', scheduler.snapshot().error or '')

    def test_worker_thread_check_cooperative_cancel_and_single_terminal(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        request = scheduler.next_request()
        assert request is not None
        entered, cancelled = threading.Event(), threading.Event()
        errors = []
        def worker():
            try:
                scheduler.check(request)
                entered.set()
                if not cancelled.wait(3):
                    raise RuntimeError('fixture gate timeout')
                scheduler.check(request)
            except CancelledError:
                errors.append('cancelled')
            finally:
                scheduler.fail(request, 'worker retired')
        owner = threading.Thread(target=worker)
        owner.start()
        try:
            self.assertTrue(entered.wait(3))
            scheduler.focus(22)
            self.assertIsNone(scheduler.next_request())
        finally:
            cancelled.set()
            owner.join(3)
        self.assertFalse(owner.is_alive())
        self.assertEqual(errors, ['cancelled'])
        self.assertEqual(next_work(scheduler).source_frame, 22)

    def test_captures_detach_mutable_inputs_and_results_keep_requested_effective_provenance(self):
        requested = {'quality': {'model_scale': 1.}}
        provenance = {'source': 'source-v1', 'corrections': ['artist-v1']}
        captured = context(requested_settings=requested, provenance=provenance)
        requested['quality']['model_scale'] = .25
        provenance['corrections'].append('mutated')
        self.assertEqual(json.loads(captured.requested_settings_json), {'quality': {'model_scale': 1.}})
        self.assertEqual(json.loads(captured.provenance_json)['corrections'], ['artist-v1'])
        scheduler = ProgressivePreview(captured, stages(), enabled=True)
        request = scheduler.next_request()
        assert request is not None
        metadata = b'{"actual_range":[10,11],"temporal_context":"restarted","quality_verified":false}'
        finish(scheduler, request, metadata=metadata)
        result = scheduler.snapshot().result
        assert result is not None
        self.assertEqual(json.loads(result.artifact.metadata_json)['actual_range'], [10,11])
        self.assertEqual(result.request.stage.name, 'draft')
        self.assertEqual(result.request.context, captured)
        invalid: Any = result.artifact
        with self.assertRaises(FrozenInstanceError):
            invalid.payload = b'changed'
        for attribute, value in (('context', context()), ('enabled', False), ('generation', 0), ('stages', stages()), ('limits', PreviewLimits())):
            with self.subTest(attribute=attribute), self.assertRaises(AttributeError):
                setattr(scheduler, attribute, value)

    def test_strict_json_limits_classes_sets_and_invalid_recipes(self):
        for invalid in ({'set': {1,'heterogeneous'}}, {'class': PreviewContext}, {'instance': context()}, {'float': float('nan')}):
            with self.subTest(invalid_type=type(next(iter(invalid.values())))), self.assertRaises((TypeError, ValueError)):
                context(provenance=invalid)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            replace(context(), provenance_json=b'{"key":1,"key":2}')
        with self.assertRaises(TypeError):
            replace(context(), provenance_json=bytearray(b'{}'))
        invalid_class: Any = PreviewContext
        with self.assertRaises(TypeError):
            ProgressivePreview(invalid_class, stages())
        invalid_stage: Any = PreviewStage
        with self.assertRaises(TypeError):
            ProgressivePreview(context(), (invalid_stage,))
        with self.assertRaises(ValueError):
            ProgressivePreview(context(), stages()[::-1])
        with self.assertRaises(ValueError):
            ProgressivePreview(context(), stages(), limits=PreviewLimits(max_work_bytes=256))
        with self.assertRaises(ValueError):
            ProgressivePreview(context(), stages(), limits=PreviewLimits(max_queued=1))
        with self.assertRaises(ValueError):
            PreviewArtifact('identity', b'RGB', json.dumps({'large':'x'*17000}).encode())

    def test_stale_display_carries_original_frame_never_claims_new_frame_alignment(self):
        scheduler = ProgressivePreview(context(), stages(), enabled=True)
        finish(scheduler)
        scheduler.focus(18)
        state = scheduler.snapshot()
        self.assertEqual(state.status, 'stale')
        self.assertEqual(state.source_frame, 18)
        assert state.result is not None
        self.assertEqual(state.result.request.source_frame, 10)
        self.assertNotEqual(state.result.request.generation, state.generation)
