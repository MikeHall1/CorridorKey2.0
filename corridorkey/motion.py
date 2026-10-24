"""CPU draft alpha propagation. Reliability is diagnostic, not confidence."""
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .control import JobControl


@dataclass(frozen=True)
class MotionSettings:
    max_side: int = 640
    consistency_pixels: float = 1.5
    photometric_scale: float = 0.15
    min_reliability: float = 0.2
    max_disagreement: float = 0.25

    def validate(self):
        if type(self.max_side) is not int or self.max_side < 16:
            raise ValueError("motion max_side must be an integer >= 16")
        for name in ("consistency_pixels", "photometric_scale", "min_reliability", "max_disagreement"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_reliability > 1 or self.max_disagreement > 1:
            raise ValueError("reliability/disagreement thresholds must be <= 1")


@dataclass
class Propagation:
    alpha: np.ndarray
    reliability: np.ndarray
    uncertain: np.ndarray
    diagnostics: dict
    disagreement_bytes: bytes | None = None


def coverage(alpha, shape):
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.shape != shape or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError(f"alpha must be finite coverage in [0,1] with shape {shape}")
    return alpha


def _proxy(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError("motion proxy must be HWC RGB with both dimensions >= 2")
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("motion proxy must be explicit finite display RGB in [0,1]")
    return rgb


def farneback(previous, following):
    return cv2.calcOpticalFlowFarneback(previous, following, None, 0.5, 3, 21, 5, 7, 1.5, 0)


def _gray(rgb, shape):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape != shape:
        gray = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.round(gray * 255).astype(np.uint8)


def _flow(estimator, previous, following, native_shape, control):
    control.check()
    flow = np.asarray(estimator(previous, following), dtype=np.float32)
    if flow.shape != (*previous.shape, 2) or not np.isfinite(flow).all():
        raise ValueError("motion estimator returned an invalid displacement field")
    h, w = native_shape
    if previous.shape != native_shape:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow *= np.array([w / previous.shape[1], h / previous.shape[0]], dtype=np.float32)
    control.check()
    return flow


def warp_anchor(alpha, anchor_rgb, target_rgb, backward, forward, settings):
    """Sample anchor using target->anchor flow; reverse flow checks consistency."""
    h, w = alpha.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    coords = np.stack((x, y), axis=-1) + backward
    valid = ((coords[..., 0] >= 0) & (coords[..., 0] <= w - 1)
             & (coords[..., 1] >= 0) & (coords[..., 1] <= h - 1))
    def sample(array):
        return cv2.remap(array, coords[..., 0], coords[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped = sample(alpha)
    error = np.linalg.norm(backward + sample(forward), axis=-1)
    photo = np.mean(np.abs(sample(anchor_rgb) - target_rgb), axis=-1)
    reliability = (valid * np.exp(-error / settings.consistency_pixels)
                   * np.exp(-photo / settings.photometric_scale)).astype(np.float32)
    return warped, reliability


def interpolate_alpha(left_rgb, right_rgb, target_rgb, left_alpha, right_alpha, fraction,
                      *, settings=None, control=None, estimator=farneback, capture_disagreement=False):
    """Bidirectional offline propagation toward the actual target source frame.

    Callers must keep all three frames in the same shot/reset segment. The
    uncertain mask requires an explicit fallback policy before exporting.
    """
    if type(capture_disagreement) is not bool:
        raise TypeError('Spatial disagreement capture requires an explicit boolean')
    settings = settings if settings is not None else MotionSettings()
    settings.validate()
    control = control if control is not None else JobControl()
    control.check()
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("offline fraction must be strictly between anchor times")
    left, right, target = map(_proxy, (left_rgb, right_rgb, target_rgb))
    if left.shape != target.shape or right.shape != target.shape:
        raise ValueError("motion proxies must have matching dimensions")
    shape = target.shape[:2]
    la, ra = coverage(left_alpha, shape), coverage(right_alpha, shape)
    ratio = min(1, settings.max_side / max(shape))
    small_shape = tuple(max(2, round(n * ratio)) for n in shape)
    started = time.perf_counter()
    lg, rg, tg = (_gray(rgb, small_shape) for rgb in (left, right, target))
    left_backward = _flow(estimator, tg, lg, shape, control)
    lwarp, lr = warp_anchor(la, left, target, left_backward,
                           _flow(estimator, lg, tg, shape, control), settings)
    right_backward = _flow(estimator, tg, rg, shape, control)
    rwarp, rr = warp_anchor(ra, right, target, right_backward,
                           _flow(estimator, rg, tg, shape, control), settings)
    lw, rw = (1 - fraction) * lr, fraction * rr
    denominator = lw + rw
    alpha = (lw * lwarp + rw * rwarp) / np.maximum(denominator, 1e-12)
    reliability = np.maximum(lr, rr)
    disagreement = np.abs(lwarp - rwarp)
    uncertain = ((reliability < settings.min_reliability)
                 | ((np.minimum(lr, rr) >= settings.min_reliability)
                    & (disagreement > settings.max_disagreement)))
    control.check()
    return Propagation(np.clip(alpha, 0, 1).astype(np.float32), reliability, uncertain, {
        "flow_evaluations": 4, "motion_seconds": time.perf_counter() - started,
        "flow_shape_hw": list(small_shape), "mean_reliability": float(reliability.mean()),
        "uncertain_fraction": float(uncertain.mean()),
        "mean_anchor_disagreement": float(disagreement.mean()),
        "motion_fraction": float(max(np.percentile(np.linalg.norm(left_backward, axis=-1), 95),
                                     np.percentile(np.linalg.norm(right_backward, axis=-1), 95)) / math.hypot(*shape)),
        "reliability_calibrated": False,
    }, disagreement_bytes=disagreement.astype('<f4', copy=False).tobytes() if capture_disagreement else None)


def conservative_approximation(propagation, nearest_alpha):
    """Explicit approximation: retain known coverage; hold where both flows fail."""
    nearest = coverage(nearest_alpha, propagation.alpha.shape)
    return np.where(propagation.uncertain, np.maximum(propagation.alpha, nearest),
                    propagation.alpha).astype(np.float32)


def propagate_alpha(anchor_rgb, target_rgb, anchor_alpha, *, settings=None, control=None, estimator=farneback):
    """Past-anchor-to-current propagation; neither flow reads a future frame."""
    settings = settings if settings is not None else MotionSettings()
    settings.validate()
    control = control if control is not None else JobControl()
    control.check()
    anchor, target = map(_proxy, (anchor_rgb, target_rgb))
    if anchor.shape != target.shape:
        raise ValueError("causal motion proxies must have matching dimensions")
    shape = target.shape[:2]
    alpha = coverage(anchor_alpha, shape)
    ratio = min(1, settings.max_side / max(shape))
    small_shape = tuple(max(2, round(n * ratio)) for n in shape)
    started = time.perf_counter()
    ag, tg = (_gray(rgb, small_shape) for rgb in (anchor, target))
    backward = _flow(estimator, tg, ag, shape, control)
    forward = _flow(estimator, ag, tg, shape, control)
    warped, reliability = warp_anchor(alpha, anchor, target, backward, forward, settings)
    control.check()
    return Propagation(warped, reliability, reliability < settings.min_reliability, {
        "flow_evaluations": 2, "motion_seconds": time.perf_counter() - started,
        "flow_shape_hw": list(small_shape), "mean_reliability": float(reliability.mean()),
        "uncertain_fraction": float((reliability < settings.min_reliability).mean()),
        "motion_fraction": float(np.percentile(np.linalg.norm(backward, axis=-1), 95) / math.hypot(*shape)),
        "drift_diagnostic": float(1 - reliability.mean()),
        "drift_metric": "one_minus_mean_reliability_not_calibrated_geometric_error",
        "reliability_calibrated": False,
    })
