"""Explicit bounded preview scheduling; no Qt, decoding, inference or exports.

Every stage is a caller-declared recipe. ``final`` means final for that preview
recipe, never full-clip/export equivalence. Source-global frame coordinates and
all source/model/correction/decode/temporal identities belong in PreviewContext.
The core does not authenticate footage or infer missing provenance.

One outstanding request remains owned until complete()/fail(), including after
seek, rebind, cancellation or disable. Worker threads call check(request) at
cooperative boundaries; this cannot interrupt native/GPU work. Limits bound
resident immutable payloads/entries and declared work admission, not native RSS
or the memory of a caller retaining evicted results. No fallback changes recipes.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .control import CancelledError

MAX_CAPTURE_BYTES = 64 * 1024
MAX_ARTIFACT_METADATA_BYTES = 16 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
QUALITIES = ('draft', 'refined', 'final')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate preview JSON field')
        result[key] = value
    return result


def _primitive(value, depth=0):
    if depth > 16:
        raise ValueError('Preview metadata nesting exceeds 16')
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, str) and len(value.encode()) <= MAX_CAPTURE_BYTES:
        return
    if isinstance(value, list) and len(value) <= 1024:
        for item in value:
            _primitive(item, depth + 1)
        return
    if isinstance(value, dict) and len(value) <= 256 and all(isinstance(key, str) for key in value):
        for item in value.values():
            _primitive(item, depth + 1)
        return
    raise TypeError('Preview metadata requires finite JSON primitives; dataclass classes/instances and sets are unsupported')


def _canonical(value: Any, maximum: int = MAX_CAPTURE_BYTES) -> bytes:
    _primitive(value)
    result = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(result) > maximum:
        raise ValueError('Preview captured metadata byte limit exceeded')
    if not isinstance(value, dict):
        raise TypeError('Preview settings/provenance metadata must be a JSON object')
    return result


def _json_bytes(raw: bytes, maximum: int = MAX_CAPTURE_BYTES) -> bytes:
    if not isinstance(raw, bytes):
        raise TypeError('Captured preview settings/provenance require immutable bytes')
    if len(raw) > maximum:
        raise ValueError('Preview captured metadata byte limit exceeded')
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except RecursionError as exc:
        raise ValueError('Preview metadata nesting exceeds parser limits') from exc
    return _canonical(value, maximum)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} requires an exact integer in [{low},{high}]')


def _identity(value, name):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > 256:
        raise ValueError(f'{name} requires a nonempty immutable identity of at most 256 UTF-8 bytes')


@dataclass(frozen=True)
class PreviewContext:
    source_identity: str
    requested_settings_json: bytes
    provenance_json: bytes
    source_start: int
    source_end: int

    def __post_init__(self):
        _identity(self.source_identity, 'source_identity')
        _integer(self.source_start, 'source_start', 0, 2**63 - 2)
        _integer(self.source_end, 'source_end', self.source_start + 1, 2**63 - 1)
        object.__setattr__(self, 'requested_settings_json', _json_bytes(self.requested_settings_json))
        object.__setattr__(self, 'provenance_json', _json_bytes(self.provenance_json))

    @classmethod
    def capture(cls, *, source_identity: str, requested_settings: dict, provenance: dict,
                source_start: int, source_end: int) -> PreviewContext:
        return cls(source_identity, _canonical(requested_settings), _canonical(provenance), source_start, source_end)

    @property
    def identity(self) -> str:
        digest = hashlib.sha256()
        for part in (self.source_identity.encode(), self.requested_settings_json, self.provenance_json,
                     str(self.source_start).encode(), str(self.source_end).encode()):
            digest.update(len(part).to_bytes(8, 'big'))
            digest.update(part)
        return digest.hexdigest()


@dataclass(frozen=True)
class PreviewStage:
    name: str
    effective_settings_json: bytes
    estimated_work_bytes: int

    def __post_init__(self):
        if self.name not in QUALITIES:
            raise ValueError('Preview stage must be draft, refined or final (for this preview recipe)')
        object.__setattr__(self, 'effective_settings_json', _json_bytes(self.effective_settings_json))
        _integer(self.estimated_work_bytes, 'estimated_work_bytes', 1, 2**31)

    @classmethod
    def capture(cls, name: str, effective_settings: dict, *, estimated_work_bytes: int) -> PreviewStage:
        return cls(name, _canonical(effective_settings), estimated_work_bytes)


@dataclass(frozen=True)
class PreviewLimits:
    max_queued: int = 64
    max_cache_entries: int = 16
    max_cache_bytes: int = 32 * 1024 * 1024
    max_result_bytes: int = 8 * 1024 * 1024
    max_work_bytes: int = 64 * 1024 * 1024
    surround_radius: int = 2
    upcoming_frames: int = 4
    max_range_frames: int = 256
    max_range_requests: int = 1024

    def __post_init__(self):
        for name, low, high in (('max_queued', 1, 512), ('max_cache_entries', 1, 256),
                                ('max_cache_bytes', 1, 512 * 1024 * 1024),
                                ('max_result_bytes', 1, MAX_PAYLOAD_BYTES), ('max_work_bytes', 1, 2**31),
                                ('surround_radius', 0, 16), ('upcoming_frames', 0, 64),
                                ('max_range_frames', 1, 1024), ('max_range_requests', 1, 1024)):
            _integer(getattr(self, name), name, low, high)
        if self.max_result_bytes > self.max_cache_bytes:
            raise ValueError('Result payload bound must fit the cache byte budget')


@dataclass(frozen=True)
class PreviewRequest:
    serial: int
    generation: int
    identity: str
    context_identity: str
    source_frame: int
    priority: str
    stage: PreviewStage
    context: PreviewContext

    @property
    def cache_identity(self) -> str:
        return hashlib.sha256(self.context_identity.encode() + b'\0' + str(self.source_frame).encode() + b'\0' +
                              self.stage.name.encode() + b'\0' + self.stage.effective_settings_json).hexdigest()


@dataclass(frozen=True)
class PreviewArtifact:
    request_identity: str
    payload: bytes
    metadata_json: bytes = b'{}'

    def __post_init__(self):
        _identity(self.request_identity, 'request_identity')
        if not isinstance(self.payload, bytes):
            raise TypeError('Preview artifact payload requires immutable bytes')
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ValueError('Preview artifact exceeds the global payload limit')
        object.__setattr__(self, 'metadata_json', _json_bytes(self.metadata_json, MAX_ARTIFACT_METADATA_BYTES))


@dataclass(frozen=True)
class PreviewResult:
    request: PreviewRequest
    artifact: PreviewArtifact


@dataclass(frozen=True)
class PreviewState:
    generation: int
    source_frame: int
    status: str
    result: PreviewResult | None
    pending: PreviewRequest | None
    queued: int
    cache_entries: int
    cache_bytes: int
    reserved_work_bytes: int
    stale_completions: int
    error: str | None


@dataclass(frozen=True)
class _Task:
    source_frame: int
    stage_index: int
    band: int
    distance: int


@dataclass(frozen=True)
class PreviewRangeState:
    selected: tuple[int, int] | None
    remaining: bool
    total_frames: int
    completed_frames: int
    failed_frames: int
    attempts: int
    max_frames: int
    max_requests: int
    budget_exhausted: bool


class ProgressivePreview:
    """One-owner scheduler; public methods/check() are thread-safe.

    Current frame gets its configured stage ladder first. Nearby frames get the
    cheapest configured stage, then upcoming playback frames. Their upgrades run
    only when next_request(idle=True). A successful exact-provenance cached
    stage satisfies all lower stages, so reuse never downgrades the display.
    focus()/bind()/set_enabled()/cancel() fence old work without releasing it.
    A caller MUST deliver exactly one terminal complete()/fail() per request.
    """

    def __init__(self, context: PreviewContext, stages: tuple[PreviewStage, ...], *,
                 enabled: bool = False, limits: PreviewLimits | None = None):
        self._lock = threading.RLock()
        self._limits = limits if limits is not None else PreviewLimits()
        if not isinstance(self._limits, PreviewLimits):
            raise TypeError('PreviewLimits requires a dataclass instance')
        self._validate_recipe(context, stages)
        if type(enabled) is not bool:
            raise TypeError('Preview enabled must be boolean')
        self._context, self._stages = context, stages
        self._context_identity = context.identity
        self._enabled = enabled
        self._generation = 1
        self._serial = 0
        self._pending: PreviewRequest | None = None
        self._cache: OrderedDict[str, PreviewResult] = OrderedDict()
        self._cache_bytes = 0
        self._last_visible: str | None = None
        self._errors: dict[str, str] = {}
        self._seen: set[str] = set()
        self._queue: list[_Task] = []
        self._focused = True
        self._frame = context.source_start
        self._direction = 1
        self._playing = False
        self._stale_completions = 0
        self._reset_ranges()
        self._rebuild()

    @property
    def context(self) -> PreviewContext:
        with self._lock:
            return self._context

    @property
    def stages(self) -> tuple[PreviewStage, ...]:
        with self._lock:
            return self._stages

    @property
    def limits(self) -> PreviewLimits:
        return self._limits

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def _reset_ranges(self):
        self._selected: tuple[int, int] | None = None
        self._remaining = False
        self._range_done: dict[int, bool] = {}
        self._selected_cursor = self._remaining_cursor = self._context.source_start
        self._range_attempts = 0

    @property
    def range_state(self) -> PreviewRangeState:
        with self._lock:
            total = (self._context.source_end - self._context.source_start if self._remaining else
                     self._selected[1] - self._selected[0] if self._selected else 0)
            return PreviewRangeState(self._selected, self._remaining, total,
                                     sum(self._range_done.values()), len(self._range_done) - sum(self._range_done.values()),
                                     self._range_attempts, self._limits.max_range_frames, self._limits.max_range_requests,
                                     self._range_attempts >= self._limits.max_range_requests and len(self._range_done) < total)

    def set_ranges(self, *, selected: tuple[int, int] | None = None, remaining: bool = False) -> None:
        """Explicit absolute [In,Out) coverage; final recipe only, admitted at idle.

        Bounded cursors/terminal acknowledgements survive focus and eviction.
        Changing range/context resets progress and fences the outstanding owner.
        """
        with self._lock:
            if type(remaining) is not bool:
                raise TypeError('Remaining preview range must be boolean')
            if selected is not None:
                if not isinstance(selected, tuple) or len(selected) != 2:
                    raise ValueError('Selected preview range requires an immutable (In,Out) pair')
                _integer(selected[0], 'Selected source In', self._context.source_start, self._context.source_end - 1)
                _integer(selected[1], 'Selected source Out (exclusive)', selected[0] + 1, self._context.source_end)
            total = (self._context.source_end - self._context.source_start if remaining else
                     selected[1] - selected[0] if selected else 0)
            if total > self._limits.max_range_frames:
                raise ValueError(f'Preview range has {total} frames; limit {self._limits.max_range_frames}. Capture a smaller requested/selected range; no automatic worker restart.')
            self._reset_ranges()
            self._selected, self._remaining = selected, remaining
            self._selected_cursor = selected[0] if selected else self._context.source_start
            self._generation += 1
            self._rebuild()

    def _range_includes(self, frame):
        return self._remaining or (self._selected is not None and self._selected[0] <= frame < self._selected[1])

    def _ack_range(self, request, success):
        if request.stage == self._stages[-1] and self._range_includes(request.source_frame):
            self._range_done[request.source_frame] = success

    def _range_task(self):
        if self._range_attempts >= self._limits.max_range_requests:
            return None
        for band, stop in ((4, self._selected[1] if self._selected else self._selected_cursor),
                           (5, self._context.source_end if self._remaining else self._remaining_cursor)):
            while (frame := self._selected_cursor if band == 4 else self._remaining_cursor) < stop:
                if frame not in self._range_done:
                    best, _ = self._best(frame)
                    if best == len(self._stages) - 1:
                        self._range_done[frame] = True
                    elif self._key(frame, len(self._stages) - 1) in self._seen:
                        self._range_done[frame] = False
                    else:
                        return _Task(frame, len(self._stages) - 1, band, 0)
                if band == 4:
                    self._selected_cursor += 1
                else:
                    self._remaining_cursor += 1
        return None

    def _validate_recipe(self, context, stages):
        if not isinstance(context, PreviewContext):
            raise TypeError('PreviewContext requires a dataclass instance')
        if not isinstance(stages, tuple) or not 1 <= len(stages) <= 3 or any(not isinstance(stage, PreviewStage) for stage in stages):
            raise TypeError('Stages require an immutable tuple of one to three PreviewStage instances')
        ranks = [QUALITIES.index(stage.name) for stage in stages]
        if ranks != sorted(set(ranks)):
            raise ValueError('Stages must be unique and ordered draft/refined/final')
        if self._limits.max_queued < len(stages):
            raise ValueError('Queue budget must admit the full explicitly configured current-frame stage ladder')
        if any(stage.estimated_work_bytes > self._limits.max_work_bytes for stage in stages):
            raise ValueError('Declared stage work reservation exceeds the explicit work admission budget')

    def _key(self, frame: int, stage_index: int) -> str:
        stage = self._stages[stage_index]
        return hashlib.sha256(self._context_identity.encode() + b'\0' + str(frame).encode() + b'\0' +
                              stage.name.encode() + b'\0' + stage.effective_settings_json).hexdigest()

    def _rebuild(self):
        self._queue = []
        self._seen.clear()
        self._errors.clear()
        if not self._enabled or not self._focused:
            return
        tasks = [_Task(self._frame, index, 0, 0) for index in range(len(self._stages))]
        surrounding = []
        for distance in range(1, self._limits.surround_radius + 1):
            for sign in (self._direction, -self._direction):
                frame = self._frame + sign * distance
                if self._context.source_start <= frame < self._context.source_end:
                    surrounding.append((frame, distance))
                    tasks.append(_Task(frame, 0, 1, distance))
        upcoming = []
        if self._playing:
            for offset in range(1, self._limits.upcoming_frames + 1):
                distance = self._limits.surround_radius + offset
                frame = self._frame + self._direction * distance
                if self._context.source_start <= frame < self._context.source_end:
                    upcoming.append((frame, distance))
                    tasks.append(_Task(frame, 0, 2, distance))
        for frame, distance in (*surrounding, *upcoming):
            tasks.extend(_Task(frame, index, 3, distance) for index in range(1, len(self._stages)))
        tasks.sort(key=lambda task: (task.band, task.distance, task.stage_index))
        self._queue = tasks[:self._limits.max_queued]

    def bind(self, context: PreviewContext, stages: tuple[PreviewStage, ...] | None = None) -> None:
        with self._lock:
            stages = self._stages if stages is None else stages
            self._validate_recipe(context, stages)
            self._context, self._stages = context, stages
            self._context_identity = context.identity
            self._frame = min(max(self._frame, context.source_start), context.source_end - 1)
            self._generation += 1
            self._focused = True
            self._reset_ranges()
            self._rebuild()

    def focus(self, source_frame: int, *, direction: int = 1, playing: bool = False) -> None:
        with self._lock:
            _integer(source_frame, 'source_frame', self._context.source_start, self._context.source_end - 1)
            if type(direction) is not int or direction not in (-1, 1) or type(playing) is not bool:
                raise ValueError('Direction must be +/-1 and playing must be boolean')
            self._frame, self._direction, self._playing = source_frame, direction, playing
            self._focused = True
            self._generation += 1
            self._rebuild()

    def set_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError('Preview enabled must be boolean')
        with self._lock:
            if self._enabled == enabled:
                return
            self._enabled = enabled
            self._generation += 1
            if not enabled:
                self._cache.clear()
                self._cache_bytes = 0
                self._last_visible = None
            self._rebuild()

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1
            self._focused = False
            self._rebuild()

    def check(self, request: PreviewRequest) -> None:
        """Worker cooperative boundary; stale/native work keeps its owner slot."""
        with self._lock:
            if not isinstance(request, PreviewRequest) or self._pending is not request or not self._enabled or request.generation != self._generation:
                raise CancelledError('Progressive preview request was cancelled, superseded or disabled')

    def _best(self, frame: int) -> tuple[int, PreviewResult | None]:
        for index in reversed(range(len(self._stages))):
            key = self._key(frame, index)
            if key in self._cache:
                return index, self._cache[key]
        return -1, None

    def next_request(self, *, idle: bool = False) -> PreviewRequest | None:
        if type(idle) is not bool:
            raise TypeError('idle must be boolean')
        with self._lock:
            if not self._enabled or not self._focused or self._pending is not None:
                return None
            while True:
                if self._queue:
                    task = self._queue[0]
                    if task.band == 3 and not idle:
                        return None
                    self._queue.pop(0)
                else:
                    task = self._range_task() if idle else None
                    if task is None:
                        return None
                key = self._key(task.source_frame, task.stage_index)
                best, _ = self._best(task.source_frame)
                if task.band != 0 and task.source_frame in self._range_done:
                    continue
                if key in self._seen or best >= task.stage_index:
                    continue
                self._seen.add(key)
                self._serial += 1
                attempt = hashlib.sha256(f'{self._generation}:{self._serial}:{key}'.encode()).hexdigest()
                request = PreviewRequest(self._serial, self._generation, attempt, self._context_identity,
                                         task.source_frame, ('current', 'surround', 'upcoming', 'idle', 'selected', 'remaining')[task.band],
                                         self._stages[task.stage_index], self._context)
                if task.band >= 4:
                    self._range_attempts += 1
                self._pending = request
                return request

    def complete(self, request: PreviewRequest, artifact: PreviewArtifact) -> bool:
        """Terminal completion; True accepts into cache. Present snapshot().result.

        Neighbour/upcoming successes are cached without replacing the focused
        display. The snapshot preserves source_frame and stale/result identity.
        """
        if not isinstance(request, PreviewRequest) or not isinstance(artifact, PreviewArtifact):
            raise TypeError('Completion requires PreviewRequest and PreviewArtifact instances')
        with self._lock:
            if self._pending is not request:
                self._stale_completions += 1
                return False
            self._pending = None
            if not self._enabled or request.generation != self._generation or request.context_identity != self._context_identity:
                self._stale_completions += 1
                return False
            if artifact.request_identity != request.identity:
                self._errors[request.cache_identity] = 'Result request/provenance identity differs from the captured request'
                self._ack_range(request, False)
                return False
            if len(artifact.payload) > self._limits.max_result_bytes:
                self._errors[request.cache_identity] = 'Result payload exceeds the explicit result byte budget'
                self._ack_range(request, False)
                return False
            key = request.cache_identity
            _, focused = self._best(self._frame)
            protected = focused.request.cache_identity if focused is not None and request.source_frame != self._frame else None
            while len(self._cache) >= self._limits.max_cache_entries or self._cache_bytes + len(artifact.payload) > self._limits.max_cache_bytes:
                victim = next((candidate for candidate in self._cache if candidate != protected), None)
                if victim is None:
                    self._errors[key] = 'Preview cache admission refused: byte/entry budget preserves the focused result'
                    self._ack_range(request, False)
                    return False
                evicted = self._cache.pop(victim)
                self._cache_bytes -= len(evicted.artifact.payload)
            self._cache[key] = PreviewResult(request, artifact)
            self._cache_bytes += len(artifact.payload)
            self._ack_range(request, True)
            if request.source_frame == self._frame:
                self._last_visible = key
            return True

    def fail(self, request: PreviewRequest, error: str) -> bool:
        """Retire failed/cancelled owned work once; never retry/fallback implicitly."""
        if not isinstance(request, PreviewRequest) or not isinstance(error, str) or not error.strip():
            raise TypeError('Terminal failure requires a request instance and explanatory text')
        with self._lock:
            if self._pending is not request:
                return False
            self._pending = None
            if not self._enabled or request.generation != self._generation:
                self._stale_completions += 1
                return False
            self._errors[request.cache_identity] = error[:2048]
            self._ack_range(request, False)
            return True

    def snapshot(self, source_frame: int | None = None) -> PreviewState:
        with self._lock:
            frame = self._frame if source_frame is None else source_frame
            _integer(frame, 'source_frame', self._context.source_start, self._context.source_end - 1)
            _, result = self._best(frame)
            if result is not None:
                self._cache.move_to_end(result.request.cache_identity)
                if frame == self._frame:
                    self._last_visible = result.request.cache_identity
            stale = self._cache.get(self._last_visible or '') if result is None else None
            error = next((self._errors[self._key(frame, index)] for index in reversed(range(len(self._stages))) if self._key(frame, index) in self._errors), None)
            if not self._enabled:
                status, result = 'disabled', None
            elif not self._focused:
                status = 'cancelled'
            elif result is not None:
                status = result.request.stage.name
            elif error is not None:
                status = 'error'
            elif stale is not None:
                status, result = 'stale', stale
            elif self._focused:
                status = 'pending'
            else:
                status = 'cancelled'
            return PreviewState(self._generation, frame, status, result, self._pending, len(self._queue),
                                len(self._cache), self._cache_bytes,
                                self._pending.stage.estimated_work_bytes if self._pending is not None else 0,
                                self._stale_completions, error)
