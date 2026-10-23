"""Deterministic playback scheduling; no decoder, Qt or inference fixtures."""
import unittest
from dataclasses import replace
from typing import Any

from corridorkey.desktop_comparison import RenderView
from corridorkey.playback import Playback


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.view = RenderView('/source', 10, 10, 20, 30, 10.0,
                               (('alpha', '/alpha'),), True, False, 'ignore')
        self.player = self.create()

    def create(self, **changes):
        args: dict[str, Any] = {'source_identity': 'source-sha256', 'timebase': 'sequence_declared',
                                'timing_evidence': 'User-declared 10fps image sequence', 'clock': self.clock}
        args.update(changes)
        return Playback(self.view, **args)

    def finish(self):
        request = self.player.tick()
        self.assertIsNotNone(request)
        assert request is not None
        self.assertTrue(self.player.complete(request))
        return request

    def test_still_frame_source_time_and_no_duplicate_decode(self):
        request = self.finish()
        self.assertEqual((request.source_frame, request.source_seconds), (10, 1.0))
        self.assertEqual(request.options['index'], 10)
        self.assertIsNone(self.player.tick())
        self.clock.now = 100
        self.assertIsNone(self.player.tick())

    def test_play_pause_and_speed_preserve_source_position(self):
        self.player.play()
        self.finish()
        self.clock.now = .31
        self.assertEqual(self.finish().source_frame, 13)
        self.player.set_speed(2)
        self.clock.now = .51
        self.assertEqual(self.finish().source_frame, 17)
        self.player.pause()
        self.clock.now = 2
        self.assertEqual(self.finish().source_frame, 17)
        self.assertFalse(self.player.playing)

    def test_pending_bounds_work_and_counts_buffering(self):
        self.player.play()
        first = self.player.tick()
        assert first is not None
        self.clock.now = .31
        for _ in range(20):
            self.assertIsNone(self.player.tick())
        self.assertIs(self.player.pending, first)
        self.assertEqual(self.player.buffered_opportunities, 3)
        self.assertEqual(self.player.dropped_opportunities, 3)
        self.assertTrue(self.player.complete(first))
        self.assertEqual(self.finish().source_frame, 13)

    def test_seek_fences_pending_but_does_not_release_its_owner(self):
        first = self.player.tick()
        assert first is not None
        self.player.seek(15)
        self.assertIsNone(self.player.tick())
        self.assertFalse(self.player.complete(replace(first)))
        self.assertIs(self.player.pending, first)
        self.assertFalse(self.player.complete(first))
        self.assertEqual(self.player.stale_completions, 1)
        self.assertEqual(self.finish().source_frame, 15)
        self.assertFalse(self.player.complete(first))

    def test_rebinding_source_fences_same_frame_old_pixels(self):
        first = self.player.tick()
        assert first is not None
        self.player.bind(self.view, source_identity='replacement-content',
                         timebase='cfr_verified', timing_evidence='Explicit ffprobe CFR analysis')
        self.assertIsNone(self.player.tick())
        self.assertFalse(self.player.complete(first))
        self.assertEqual(self.finish().source_identity, 'replacement-content')

    def test_loop_range_and_terminal_frame(self):
        self.player.set_range(12, 15)
        self.player.set_loop(True)
        self.player.play()
        self.assertEqual(self.finish().source_frame, 12)
        self.clock.now = .41
        self.assertEqual(self.finish().source_frame, 13)
        self.player.set_loop(False)
        self.clock.now = 5
        self.assertEqual(self.finish().source_frame, 14)
        self.assertFalse(self.player.playing)
        dropped = self.player.dropped_opportunities
        self.clock.now = 10
        self.assertIsNone(self.player.tick())
        self.assertEqual(self.player.dropped_opportunities, dropped)

    def test_comparison_reuses_recorded_offsets_and_common_range(self):
        reference = replace(self.view, start=12, count=10, outputs=(('alpha', '/other'),))
        self.player = self.create(reference=reference, mode='Alpha')
        request = self.finish()
        self.assertEqual((self.player.start, self.player.end), (12, 20))
        self.assertEqual(request.options['pair']['a']['index'], 0)
        self.assertEqual(request.options['pair']['b']['index'], 2)
        self.assertEqual(request.source_seconds, 1.2)
        with self.assertRaises(ValueError):
            self.create(reference=replace(reference, source='/different'))

    def test_independent_reference_mode_uses_existing_view_mapping(self):
        reference = replace(self.view, start=12, count=10)
        self.player = self.create(reference=reference, mode='Alpha', reference_mode='Source')
        request = self.finish()
        self.assertEqual(request.options['pair']['a']['index'], 12)
        self.assertEqual(request.options['pair']['b']['index'], 2)
        with self.assertRaises(ValueError):
            self.create(reference=reference, reference_mode='Unavailable')

    def test_rejects_unknown_vfr_or_unsubstantiated_timebase(self):
        for changes in ({'timebase': 'vfr'}, {'timebase': 'unknown'}, {'timing_evidence': ''}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.create(**changes)

    def test_failed_completion_releases_owner_without_busy_retry(self):
        request = self.player.tick()
        assert request is not None
        self.assertFalse(self.player.complete(request, succeeded=False))
        self.assertIsNone(self.player.pending)
        self.assertIsNone(self.player.tick())
        self.player.seek(10)
        self.finish()

    def test_invalid_controls_preserve_configuration(self):
        for value in (True, 0, float('nan'), 9):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.player.set_speed(value)
        for frame in (True, -1, 20):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                self.player.seek(frame)
        for span in ((9, 12), (12, 12), (12, 21)):
            with self.subTest(span=span), self.assertRaises(ValueError):
                self.player.set_range(*span)
        with self.assertRaises(ValueError):
            invalid: Any = 1
            self.player.set_loop(invalid)
        self.assertEqual((self.player.frame, self.player.speed), (10, 1))

    def test_clock_failures_are_explicit(self):
        self.clock.now = -1
        with self.assertRaisesRegex(ValueError, 'backwards'):
            self.player.tick()
        self.clock.now = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.player.tick()


if __name__ == '__main__':
    unittest.main()
