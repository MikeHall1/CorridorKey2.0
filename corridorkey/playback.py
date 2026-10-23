"""Bounded, timer-driven playback planning; no decoding, threads or Qt.

A caller supplies verified CFR evidence (or an explicitly declared sequence
rate), calls tick(), submits its options to PreviewLoader, and calls complete()
for *every* terminal decode, including errors and superseded results. Only a
True completion may be displayed. In-flight native decode cannot be cancelled;
seek/pause/rebind fence its result and retain its ownership until completion.
Average decoder FPS alone is not CFR evidence. Source time is frame/fps in the
explicit constant-rate source coordinate system, not a measured video PTS.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .desktop_comparison import RenderView, common_range


@dataclass(frozen=True)
class FrameRequest:
    serial: int
    generation: int
    source_identity: str
    source_frame: int
    source_seconds: float
    options: dict[str, Any]


class Playback:
    """One outstanding frame or atomic A/B pair; caller owns actual decoding.

    bind() changes the source/view configuration, retaining any outstanding
    decode until its terminal callback. Counts are presentation opportunities,
    not claims about decoder internals or render performance.
    """

    def __init__(self, view: RenderView, *, source_identity: str, timebase: str,
                 timing_evidence: str, reference: RenderView | None = None,
                 mode: str = 'Source', reference_mode: str | None = None, background=None,
                 clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.generation = 0
        self.serial = 0
        self.pending: FrameRequest | None = None
        self.playing = False
        self.speed = 1.0
        self.loop = False
        self.dropped_opportunities = 0
        self.buffered_opportunities = 0
        self.stale_completions = 0
        self._last_clock: float | None = None
        self.bind(view, source_identity=source_identity, timebase=timebase,
                  timing_evidence=timing_evidence, reference=reference,
                  mode=mode, reference_mode=reference_mode, background=background)

    def _now(self) -> float:
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError('Playback clock must return finite monotonic seconds.')
        if self._last_clock is not None and now < self._last_clock:
            raise ValueError('Playback clock moved backwards.')
        self._last_clock = now
        return now

    def bind(self, view: RenderView, *, source_identity: str, timebase: str,
             timing_evidence: str, reference: RenderView | None = None,
             mode: str = 'Source', reference_mode: str | None = None, background=None) -> None:
        if timebase not in {'cfr_verified', 'sequence_declared'} or not isinstance(timing_evidence, str) or not timing_evidence.strip():
            raise ValueError('Playback requires explicit CFR evidence or a declared sequence rate; VFR/unknown timing is unsupported.')
        if not isinstance(source_identity, str) or not source_identity.strip():
            raise ValueError('A caller-bound source identity is required.')
        if not math.isfinite(view.fps) or view.fps <= 0:
            raise ValueError('Playback requires a positive finite source rate.')
        start, end = common_range(view, reference) if reference is not None else (view.start, view.end)
        if start < 0 or end <= start:
            raise ValueError('Playback range must be nonempty.')
        # Validate display modes before changing the current configuration.
        view.options(mode, start, background)
        if reference is not None:
            reference.options(reference_mode or mode, start, background)
        now = self._now()
        self.view, self.reference = view, reference
        self.source_identity, self.timebase = source_identity, timebase
        self.timing_evidence = timing_evidence
        self.mode, self.background = mode, background
        self.reference_mode = reference_mode or mode
        self.start, self.end = start, end
        self.frame = start
        self.playing = False
        self._reset(now)

    def _reset(self, now: float) -> None:
        self.generation += 1
        self._anchor_frame, self._anchor_time = self.frame, now
        self._last_slot = 0
        self._last_requested: tuple[int, int] | None = None

    def seek(self, frame: int) -> None:
        if type(frame) is not int or not self.start <= frame < self.end:
            raise ValueError('Seek frame lies outside the selected source range.')
        now = self._now()
        self.frame = frame
        self._reset(now)

    def set_range(self, start: int, end: int) -> None:
        lower, upper = common_range(self.view, self.reference) if self.reference is not None else (self.view.start, self.view.end)
        if type(start) is not int or type(end) is not int or not lower <= start < end <= upper:
            raise ValueError('Loop range must lie within the available common source range.')
        now = self._now()
        self.start, self.end = start, end
        self.frame = min(max(self.frame, start), end - 1)
        self._reset(now)

    def _position(self, now: float) -> tuple[int, int]:
        slots = math.floor(max(0.0, now - self._anchor_time) * self.view.fps * self.speed) if self.playing else 0
        if not self.loop:
            slots = min(slots, self.end - 1 - self._anchor_frame)
        position = self._anchor_frame + slots
        if self.loop:
            position = self.start + (position - self.start) % (self.end - self.start)
        else:
            position = min(position, self.end - 1)
        return position, slots

    def play(self) -> None:
        if not self.playing:
            now = self._now()
            self.playing = True
            self._reset(now)

    def pause(self) -> None:
        now = self._now()
        self.frame, _ = self._position(now)
        self.playing = False
        self._reset(now)

    def set_speed(self, speed: float) -> None:
        if type(speed) not in (int, float) or not math.isfinite(speed) or not 0.05 <= speed <= 8:
            raise ValueError('Playback speed must be finite in [0.05, 8].')
        now = self._now()
        self.frame, _ = self._position(now)
        self.speed = float(speed)
        self._reset(now)

    def set_loop(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError('Loop must be boolean.')
        now = self._now()
        self.frame, _ = self._position(now)
        self.loop = enabled
        self._reset(now)

    def tick(self) -> FrameRequest | None:
        now = self._now()
        self.frame, slots = self._position(now)
        advanced = max(0, slots - self._last_slot)
        self._last_slot = slots
        if self.pending is not None:
            self.buffered_opportunities += advanced
            self.dropped_opportunities += advanced
            return None
        key = (self.generation, self.frame)
        if key == self._last_requested:
            self.dropped_opportunities += advanced
            return None
        self.dropped_opportunities += max(0, advanced - 1)
        options = self.view.options(self.mode, self.frame, self.background)
        if self.reference is not None:
            options = {'pair': {'a': self.reference.options(self.reference_mode, self.frame, self.background), 'b': options}}
        self.serial += 1
        request = FrameRequest(self.serial, self.generation, self.source_identity,
                               self.frame, self.frame / self.view.fps, options)
        self.pending = request
        self._last_requested = key
        if self.playing and not self.loop and self.frame == self.end - 1:
            self.playing = False
            self._anchor_frame = self.frame
        return request

    def complete(self, request: FrameRequest, *, succeeded: bool = True) -> bool:
        """Release the matching owner; return whether its pixels may be shown.

        Pass the original token, never a reconstructed request. Errors release
        ownership too; retrying a failed frame requires an explicit seek/play.
        """
        if request is not self.pending:
            return False
        self.pending = None
        valid = (request.generation == self.generation
                 and request.source_identity == self.source_identity)
        if not valid:
            self.stale_completions += 1
        return valid and succeeded
