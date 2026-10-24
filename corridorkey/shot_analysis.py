"""Streaming CPU shot diagnostics; heuristic boundaries are not ground truth."""
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .control import JobControl
from .motion import _proxy


@dataclass(frozen=True)
class ShotSettings:
    max_side: int = 128
    cut_threshold: float = 0.35
    min_pixel_change: float = 0.1

    def validate(self):
        if type(self.max_side) is not int or self.max_side < 16:
            raise ValueError("analysis max_side must be an integer >=16")
        for name in ("cut_threshold", "min_pixel_change"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0,1]")


def _small(proxy, max_side):
    proxy = _proxy(proxy)
    h, w = proxy.shape[:2]
    scale = min(1, max_side / max(h, w))
    return cv2.resize(proxy, (max(2, round(w * scale)), max(2, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _histogram(proxy):
    bins = np.minimum((proxy * 16).astype(np.int32), 15)
    return np.stack([np.bincount(bins[..., channel].ravel(), minlength=16)
                     for channel in range(3)]).astype(np.float64) / (proxy.shape[0] * proxy.shape[1])


def analyze_shots(frame_count, read_source, *, settings=None, control=None, emit=None):
    """Read every native frame; retain only adjacent reduced proxies and scores.

    read_source returns SourceFrame. emit receives one JSON-compatible diagnostic
    per source frame, allowing a bounded-memory JSONL log rather than retaining
    full-clip arrays. Flashes/exposure shifts can be flagged; cuts can be missed.
    """
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("analysis frame count must be a positive integer")
    settings = settings if settings is not None else ShotSettings()
    settings.validate()
    control = control if control is not None else JobControl()
    previous = previous_histogram = None
    cuts = []
    max_score = 0.0
    started = time.perf_counter()
    native_shape = None
    for frame in range(frame_count):
        control.checkpoint("shot_analysis", frame, frame_count)
        source = read_source(frame)
        control.check()
        proxy = _proxy(source.proxy)
        if native_shape is None:
            native_shape = proxy.shape
        elif proxy.shape != native_shape:
            raise ValueError("source dimensions changed during shot analysis")
        current = _small(proxy, settings.max_side)
        histogram = _histogram(current)
        delta = distance = score = 0.0
        cut = False
        if previous is not None:
            delta = float(np.abs(current - previous).mean())
            distance = float(np.abs(histogram - previous_histogram).sum(axis=1).mean() / 2)
            score = max(delta, distance)
            cut = score >= settings.cut_threshold and delta >= settings.min_pixel_change
            if cut:
                cuts.append(frame)
        detail = float((np.abs(np.diff(current, axis=0)).mean()
                        + np.abs(np.diff(current, axis=1)).mean()) / 2)
        record = {"edge_detail": detail, "frame": frame, "cut": cut, "score": score,
                  "mean_rgb_change": delta, "histogram_distance": distance,
                  "calibrated": False}
        if emit is not None:
            emit(record)
        max_score = max(max_score, score)
        previous, previous_histogram = current, histogram
    control.checkpoint("shot_analysis", frame_count, frame_count)
    return {"detected_cuts": cuts, "frames_analyzed": frame_count,
            "proxy_shape_hw": list(previous.shape[:2]), "max_cut_score": max_score,
            "analysis_seconds": time.perf_counter() - started,
            "method": "rgb_histogram_tv_and_adjacent_mean_absolute_rgb",
            "cut_threshold": settings.cut_threshold, "min_pixel_change": settings.min_pixel_change,
            "calibrated": False, "semantic_cut_guarantee": False}
