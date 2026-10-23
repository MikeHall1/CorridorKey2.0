"""Streaming authored-mask tracking previews; no persistence or neural channels.

Timeline bounds/anchor indices are correction-local. Reader callbacks receive
absolute source indices (source_start + local index). Candidates must be
explicitly inspected/accepted by a higher-level authoring workflow. Pixel/frame
and anchor-byte admission bounds retained work, not OpenCV native allocations.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .control import JobControl
from .motion import (
    MotionSettings,
    conservative_approximation,
    farneback,
    interpolate_alpha,
    propagate_alpha,
)

KERNEL_VERSION = 'correction-tracking-v1'


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')


def _source_identity(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Source identity must be a lowercase SHA256 string.')


@dataclass(frozen=True)
class TrackingAnchor:
    identifier: str
    revision: int
    frame: int
    kind: str
    strength: float
    height: int
    width: int
    mask_bytes: bytes

    @classmethod
    def create(cls, identifier: str, revision: int, frame: int, kind: str, strength: float,
               mask: NDArray[np.float32]) -> TrackingAnchor:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.float32 or mask.ndim != 2:
            raise ValueError('Anchor must be a native float32 coverage mask.')
        if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError('Anchor mask must be finite coverage in [0,1].')
        anchor = cls(identifier, revision, frame, kind, strength, mask.shape[0], mask.shape[1], mask.astype('<f4', copy=False).tobytes())
        anchor.validate()
        return anchor

    def validate(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier or len(self.identifier) > 256:
            raise ValueError('Anchor identifier must be nonempty and at most 256 characters.')
        for value, name in ((self.revision, 'anchor revision'), (self.frame, 'anchor frame')):
            _integer(value, name)
        _integer(self.height, 'anchor height', 2)
        _integer(self.width, 'anchor width', 2)
        if self.kind not in {'matte', 'keep', 'remove'}:
            raise ValueError('Tracking supports matte/keep/remove coverage only, not trimap classes.')
        if type(self.strength) not in (int, float) or not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError('Anchor strength must be finite in [0,1].')
        if type(self.mask_bytes) is not bytes or len(self.mask_bytes) != self.height * self.width * 4:
            raise ValueError('Anchor bytes must be immutable native float32 coverage.')
        mask = self.mask()
        if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError('Anchor bytes contain invalid coverage.')

    def mask(self) -> NDArray[np.float32]:
        return np.frombuffer(self.mask_bytes, dtype='<f4').reshape(self.height, self.width)

    def provenance(self) -> dict[str, Any]:
        return {'id': self.identifier, 'revision': self.revision, 'frame': self.frame,
                'kind': self.kind, 'strength': self.strength, 'shape': [self.height, self.width],
                'mask_sha256': _digest(self.mask_bytes)}


@dataclass(frozen=True)
class TrackingRequest:
    source_identity: str
    start: int
    end: int
    source_start: int = 0
    source_decode_identity: str = 'caller-declared-display-rgb-float32-v1'
    direction: str = 'both'
    cuts: tuple[int, ...] = ()
    resets: tuple[int, ...] = ()
    allow_cross_boundaries: bool = False
    unreliable_policy: str = 'request_anchor'
    max_uncertain_fraction: float = .25
    motion: MotionSettings = field(default_factory=MotionSettings)
    max_frames: int = 300
    max_pixels: int = 4 * 1024 * 1024
    max_anchor_bytes: int = 64 * 1024 * 1024
    max_anchors: int = 32

    def validate(self) -> None:
        _source_identity(self.source_identity)
        if (not isinstance(self.source_decode_identity, str) or not self.source_decode_identity.strip()
                or len(self.source_decode_identity) > 512):
            raise ValueError('A bounded immutable source-decode contract identity is required.')
        for name in ('start', 'end', 'source_start'):
            _integer(getattr(self, name), name)
        for name in ('max_frames', 'max_pixels', 'max_anchor_bytes', 'max_anchors'):
            _integer(getattr(self, name), name, 1)
        if not self.start < self.end or self.end - self.start > self.max_frames:
            raise ValueError('Tracking range must be nonempty and within the explicit frame budget.')
        if self.direction not in {'forward', 'backward', 'both'}:
            raise ValueError('Tracking direction must be forward/backward/both.')
        if self.unreliable_policy not in {'stop', 'request_anchor', 'retain_approximation'}:
            raise ValueError('Unknown tracking reliability policy.')
        if type(self.allow_cross_boundaries) is not bool:
            raise ValueError('Cross-boundary tracking requires an explicit boolean.')
        for boundaries in (self.cuts, self.resets):
            if (type(boundaries) is not tuple or len(boundaries) > self.max_frames + 1
                    or any(type(v) is not int or not 0 <= v <= self.end for v in boundaries)):
                raise ValueError('Boundaries must be immutable tuples of nonnegative local frames.')
            if tuple(sorted(set(boundaries))) != boundaries:
                raise ValueError('Boundaries must be sorted and unique.')
        if type(self.max_uncertain_fraction) not in (int, float) or not math.isfinite(self.max_uncertain_fraction) or not 0 <= self.max_uncertain_fraction <= 1:
            raise ValueError('Uncertain fraction threshold must be finite in [0,1].')
        if not isinstance(self.motion, MotionSettings):
            raise TypeError('Tracking requires a MotionSettings instance, not its class.')
        self.motion.validate()


@dataclass(frozen=True)
class TrackingFrame:
    source_identity: str
    source_frame: int
    proxy: NDArray[np.float32]


@dataclass(frozen=True)
class TrackingCandidate:
    local_frame: int
    source_frame: int
    height: int
    width: int
    status: str
    mask_bytes: bytes | None
    reliability_bytes: bytes | None
    uncertain_bytes: bytes | None
    provenance_json: str
    diagnostics_json: str
    requires_acceptance: bool = True

    @property
    def identity(self) -> str:
        return _digest(self.provenance_json.encode())

    def mask(self) -> NDArray[np.float32] | None:
        return None if self.mask_bytes is None else np.frombuffer(self.mask_bytes, dtype='<f4').reshape(self.height, self.width)


def track_corrections(request: TrackingRequest, anchors: tuple[TrackingAnchor, ...],
                      read_frame: Callable[[int], TrackingFrame], *, control: JobControl | None = None,
                      estimator=farneback, estimator_identity: str | None = None) -> Generator[TrackingCandidate, None, None]:
    """Yield one bounded preview/event at a time; caller must bound retained results.

    Each target is warped directly from the nearest authored anchor(s), never
    from an approximated derived mask. Bidirectional interior targets use both
    surrounding anchors; directional tracking never reads an opposite anchor.
    Unreliable propagation blocks its interval until another artist anchor.
    No inference or files are changed, including when iteration is cancelled.
    """
    if not isinstance(request, TrackingRequest):
        raise TypeError('Tracking requires a request instance, not its class.')
    request.validate()
    control = control if control is not None else JobControl()
    control.check()
    if type(anchors) is not tuple or not 1 <= len(anchors) <= request.max_anchors:
        raise ValueError('Provide an immutable, bounded tuple of artist anchors.')
    if any(not isinstance(a, TrackingAnchor) for a in anchors):
        raise ValueError('Artist anchors must be TrackingAnchor instances.')
    if any(type(a.mask_bytes) is not bytes for a in anchors):
        raise ValueError('Artist mask storage must be immutable bytes.')
    if sum(len(a.mask_bytes) for a in anchors) > request.max_anchor_bytes:
        raise ValueError('Artist masks exceed the explicit anchor-byte budget.')
    for anchor in anchors:
        control.check()
        if anchor.height * anchor.width > request.max_pixels:
            raise ValueError('Native tracking geometry exceeds the pixel budget.')
        anchor.validate()
    ordered = sorted(anchors, key=lambda a: a.frame)
    shape = (ordered[0].height, ordered[0].width)
    if (len({a.frame for a in ordered}) != len(ordered) or len({a.identifier for a in ordered}) != len(ordered)
            or any(not request.start <= a.frame < request.end or (a.height, a.width) != shape for a in ordered)):
        raise ValueError('Anchors require unique IDs/frames, matching native geometry and frames within the requested range.')
    if len({(a.kind, a.strength) for a in ordered}) != 1:
        raise ValueError('One track must contain the same correction operation and strength at every artist anchor.')
    if estimator_identity is None:
        if estimator is not farneback:
            raise ValueError('Injected motion estimators require an explicit provenance identity.')
        estimator_identity = f'opencv-{cv2.__version__}-farneback-v1'
    if not isinstance(estimator_identity, str) or not estimator_identity.strip() or len(estimator_identity) > 512:
        raise ValueError('Motion estimator provenance identity is required and bounded.')
    base = {'kernel': KERNEL_VERSION, 'request': asdict(request),
            'anchors': [a.provenance() for a in ordered], 'estimator': estimator_identity,
            'mask_contract': 'float32-coverage-derived-final-correction-preview-not-alpha-matting'}
    base_hash = _digest(_canonical(base).encode())
    boundaries = sorted(set(request.cuts + request.resets))
    by_frame = {a.frame: a for a in ordered}
    blocked: tuple[int, ...] | None = None

    def read(local: int) -> tuple[NDArray[np.float32], str]:
        control.check()
        actual = request.source_start + local
        frame = read_frame(actual)
        if (not isinstance(frame, TrackingFrame) or frame.source_identity != request.source_identity
                or type(frame.source_frame) is not int or frame.source_frame != actual):
            raise ValueError('Tracking reader returned a mismatched source/frame identity.')
        proxy = frame.proxy
        if (not isinstance(proxy, np.ndarray) or proxy.dtype != np.float32 or proxy.shape != (*shape, 3)
                or not np.isfinite(proxy).all() or np.any((proxy < 0) | (proxy > 1))):
            raise ValueError('Tracking reader must return finite native float32 display RGB in [0,1].')
        owned = proxy.copy()
        control.check()
        return owned, _digest(owned.astype('<f4', copy=False).tobytes())

    frames = range(request.end - 1, request.start - 1, -1) if request.direction == 'backward' else range(request.start, request.end)
    for completed, local in enumerate(frames):
        control.checkpoint('tracking', completed, request.end - request.start)
        permitted = ordered if request.allow_cross_boundaries else [a for a in ordered if not any(min(a.frame, local) < b <= max(a.frame, local) for b in boundaries)]
        left = next((a for a in reversed(permitted) if a.frame <= local), None)
        right = next((a for a in permitted if a.frame >= local), None)
        selected = ([left] if request.direction == 'forward' else [right]) if request.direction != 'both' else [a for a in (left, right) if a is not None]
        selected = list({a.frame: a for a in selected if a is not None}.values())
        dependency = tuple(a.frame for a in selected)
        anchor = by_frame.get(local)
        diagnostics: dict[str, Any] = {'reliability_calibrated': False,
            'uncertainty_meaning': 'flow/photometric heuristic; possible occlusion, not proof of occlusion',
            'approximation_retained': False}
        hashes: dict[str, str] = {}
        mask = reliability = uncertain = None
        if anchor is not None:
            _, frame_hash = read(local)
            hashes[str(local)] = frame_hash
            mask = anchor.mask()
            reliability = np.ones(shape, np.float32)
            uncertain = np.zeros(shape, np.bool_)
            status = 'authored_anchor'
            blocked = None
        elif not selected:
            status = 'anchor_required'
            diagnostics['reason'] = 'No permitted directional anchor in this cut/reset segment.'
        elif blocked == dependency:
            status = 'stopped' if request.unreliable_policy == 'stop' else 'anchor_required'
            diagnostics['reason'] = 'Propagation interval stopped; another artist anchor is required.'
        else:
            target, target_hash = read(local)
            hashes[str(local)] = target_hash
            first, first_hash = read(selected[0].frame)
            hashes[str(selected[0].frame)] = first_hash
            if len(selected) == 2:
                second, second_hash = read(selected[1].frame)
                hashes[str(selected[1].frame)] = second_hash
                fraction = (local - selected[0].frame) / (selected[1].frame - selected[0].frame)
                propagation = interpolate_alpha(first, second, target, selected[0].mask(), selected[1].mask(), fraction,
                                                settings=request.motion, control=control, estimator=estimator)
            else:
                propagation = propagate_alpha(first, target, selected[0].mask(), settings=request.motion,
                                              control=control, estimator=estimator)
            diagnostics.update(propagation.diagnostics)
            uncertain_fraction = float(propagation.uncertain.mean())
            diagnostics['uncertain_fraction'] = uncertain_fraction
            unreliable = uncertain_fraction > request.max_uncertain_fraction
            if unreliable and request.unreliable_policy != 'retain_approximation':
                blocked = dependency
                status = 'stopped' if request.unreliable_policy == 'stop' else 'anchor_required'
                diagnostics['reason'] = 'Motion reliability threshold exceeded; candidate not accepted.'
            else:
                status = 'approximation' if unreliable else 'preview'
                mask = propagation.alpha
                if unreliable:
                    nearest = min(selected, key=lambda a: abs(a.frame - local))
                    mask = conservative_approximation(propagation, nearest.mask())
                    diagnostics['approximation_retained'] = True
            reliability, uncertain = propagation.reliability, propagation.uncertain
        mask_bytes = None if mask is None else np.asarray(mask, dtype='<f4').tobytes()
        provenance = {'track_sha256': base_hash, 'track': base, 'source_identity': request.source_identity,
                      'local_frame': local, 'source_frame': request.source_start + local,
                      'source_proxy_sha256': hashes, 'anchor_frames': list(dependency), 'status': status,
                      'derived_mask_sha256': None if mask_bytes is None else _digest(mask_bytes)}
        control.check()
        yield TrackingCandidate(local, request.source_start + local, *shape, status, mask_bytes,
            None if reliability is None else np.asarray(reliability, dtype='<f4').tobytes(),
            None if uncertain is None else np.asarray(uncertain, dtype=np.bool_).tobytes(),
            _canonical(provenance), _canonical(diagnostics))
    control.checkpoint('tracking', request.end - request.start, request.end - request.start)
