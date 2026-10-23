"""Bounded offline Quick Key frame assembly, shared by runtime and CPU fixtures."""
from dataclasses import dataclass
import math
import numpy as np

from .control import JobControl
from .motion import MotionSettings, coverage, interpolate_alpha, conservative_approximation, propagate_alpha


@dataclass
class SourceFrame:
    rgb: np.ndarray  # Current source RGB; can retain signed/HDR linear values.
    proxy: np.ndarray  # Explicit display RGB for motion, in [0,1].


@dataclass
class DraftFrame:
    index: int
    rgb: np.ndarray
    alpha: np.ndarray
    diagnostics: dict


class UncertainMotionError(RuntimeError):
    pass


def offline_frames(plan, read_frame, read_anchor, read_alpha, *, settings=None,
                   fallback="stop", neural_fallback=None, control=None, stats=None,
                   interpolate=interpolate_alpha, adaptive=False, motion_threshold=0.05):
    """Yield native-rate source-RGB drafts. Anchor/source reads use separate cursors.

    Only two anchor sources/mattes are retained. No source frame interpolation
    or reconstructed-foreground propagation occurs. This offline algorithm uses
    future anchors; it has look-ahead and is not a live/causal mode.
    """
    if fallback not in {"stop", "approximate", "neural"}:
        raise ValueError("fallback must be stop, approximate or neural")
    if fallback == "neural" and neural_fallback is None:
        raise ValueError("neural fallback requires an actual inference callback")
    if type(adaptive) is not bool or (adaptive and fallback != "neural"):
        raise ValueError("adaptive anchors require explicit neural fallback")
    if type(motion_threshold) not in (int, float) or not math.isfinite(motion_threshold) or not 0 < motion_threshold <= 1:
        raise ValueError("motion threshold must be finite and in (0,1]")
    settings = settings if settings is not None else MotionSettings()
    settings.validate()
    control = control if control is not None else JobControl()
    stats = stats if stats is not None else {}
    stats.update({"mode": "offline_motion_guided", "foreground": "source_rgb_draft",
                  "anchors": list(plan.anchors), "frames_emitted": 0,
                  "flow_evaluations": 0, "motion_seconds": 0.0,
                  "fallback_frames": [], "extra_neural_frames": [],
                  "max_look_ahead_frames": 0, "anchor_source_reads": 0,
                  "peak_retained_anchor_sources": 0, "reliability_calibrated": False})
    stats.update({"adaptive": adaptive, "planned_anchors": list(plan.anchors),
                  "promoted_anchors": [], "promotion_reasons": []})
    cached = {}
    promoted = None
    for frame in range(plan.frame_count):
        control.checkpoint("quick_key_output", frame, plan.frame_count)
        source = read_frame(frame)
        rgb = np.asarray(source.rgb, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
            raise ValueError("draft source must be finite HWC RGB")
        left, right = plan.bracket(frame)
        if promoted is not None and left < promoted < frame and plan.segment(promoted) == plan.segment(frame):
            left = promoted
        needed = {left, right}
        for old in list(cached):
            if old not in needed:
                del cached[old]
        diagnostic = {"source_rgb_draft": True, "anchors": [left, right],
                      "look_ahead_frames": right - frame, "fallback": None}
        if left == right:
            alpha = coverage(read_alpha(frame), rgb.shape[:2]).copy()
        else:
            for anchor in (left, right):
                if anchor not in cached:
                    anchor_source = read_anchor(anchor)
                    cached[anchor] = (np.asarray(anchor_source.proxy, dtype=np.float32).copy(),
                                      coverage(read_alpha(anchor), rgb.shape[:2]).copy())
                    stats["anchor_source_reads"] += 1
                    stats["peak_retained_anchor_sources"] = max(stats["peak_retained_anchor_sources"], len(cached))
            result = interpolate(cached[left][0], cached[right][0], source.proxy,
                                 cached[left][1], cached[right][1], (frame - left) / (right - left),
                                 settings=settings, control=control)
            diagnostic.update(result.diagnostics)
            stats["flow_evaluations"] += result.diagnostics["flow_evaluations"]
            stats["motion_seconds"] += result.diagnostics["motion_seconds"]
            alpha = result.alpha
            fast_motion = adaptive and result.diagnostics.get("motion_fraction", 0) >= motion_threshold
            if result.uncertain.any() or fast_motion:
                diagnostic["fallback"] = fallback
                stats["fallback_frames"].append(frame)
                if fallback == "stop":
                    raise UncertainMotionError(
                        f"Quick Key frame {frame}: {result.uncertain.mean():.1%} uncertain pixels; "
                        "choose an explicit approximation or additional-neural fallback")
                if fallback == "neural":
                    control.check()
                    alpha = coverage(neural_fallback(frame), rgb.shape[:2]).copy()
                    stats["extra_neural_frames"].append(frame)
                    if adaptive:
                        promoted = frame
                        # Replace the past anchor, retaining at most two sources.
                        cached.pop(left, None)
                        cached[frame] = (np.asarray(source.proxy, dtype=np.float32).copy(), alpha.copy())
                        stats["promoted_anchors"].append(frame)
                        stats["anchors"] = sorted(set(plan.anchors) | set(stats["promoted_anchors"]))
                        stats["promotion_reasons"].append({"frame": frame,
                            "reason": "motion_threshold" if fast_motion else "uncertain_flow_or_matte_disagreement"})
                        diagnostic["promoted_anchor"] = True
                else:
                    nearest = left if frame - left <= right - frame else right
                    alpha = conservative_approximation(result, cached[nearest][1])
            stats["max_look_ahead_frames"] = max(stats["max_look_ahead_frames"], right - frame)
        control.check()
        yield DraftFrame(frame, rgb, alpha, diagnostic)
        stats["frames_emitted"] += 1
    control.checkpoint("quick_key_output", plan.frame_count, plan.frame_count)


def past_frames(plan, read_frame, read_alpha, *, approach="causal", settings=None,
                fallback="stop", neural_fallback=None, control=None, stats=None,
                adaptive=False, motion_threshold=0.05, propagate=propagate_alpha):
    """Hold or causal matte assembly. Neural callbacks must also use past-only inputs.

    Causal propagation always promotes an explicitly requested neural fallback;
    adaptive additionally requests anchors for the motion threshold. This is not
    a claim that the runtime's whole-clip precompute/export is a live pipeline.
    """
    if approach not in {"hold", "causal"}:
        raise ValueError("past approach must be hold or causal")
    if fallback not in {"stop", "approximate", "neural"}:
        raise ValueError("fallback must be stop, approximate or neural")
    if type(adaptive) is not bool or (adaptive and (approach == "hold" or fallback != "neural")):
        raise ValueError("adaptive past propagation requires causal mode and neural fallback")
    if fallback == "neural" and neural_fallback is None:
        raise ValueError("neural fallback requires an actual callback")
    if type(motion_threshold) not in (int, float) or not math.isfinite(motion_threshold) or not 0 < motion_threshold <= 1:
        raise ValueError("motion threshold must be finite and in (0,1]")
    settings = settings if settings is not None else MotionSettings()
    settings.validate()
    control = control if control is not None else JobControl()
    stats = stats if stats is not None else {}
    stats.update({"mode": "hold_last_matte" if approach == "hold" else "causal_motion_guided",
                  "foreground": "source_rgb_draft", "anchors": list(plan.anchors),
                  "planned_anchors": list(plan.anchors), "promoted_anchors": [], "promotion_reasons": [],
                  "adaptive": adaptive, "frames_emitted": 0, "flow_evaluations": 0, "motion_seconds": 0.0,
                  "fallback_frames": [], "extra_neural_frames": [], "max_look_ahead_frames": 0,
                  "anchor_source_reads": 0, "peak_retained_anchor_sources": 0,
                  "reliability_calibrated": False})
    scheduled = set(plan.anchors)
    anchor_frame = anchor_proxy = anchor_alpha = None
    for frame in range(plan.frame_count):
        control.checkpoint("quick_key_output", frame, plan.frame_count)
        source = read_frame(frame)
        rgb = np.asarray(source.rgb, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
            raise ValueError("draft source must be finite HWC RGB")
        diagnostic = {"source_rgb_draft": True, "look_ahead_frames": 0, "fallback": None}
        if frame in scheduled:
            anchor_frame = frame
            anchor_alpha = coverage(read_alpha(frame), rgb.shape[:2]).copy()
            anchor_proxy = np.asarray(source.proxy, dtype=np.float32).copy() if approach == "causal" else None
            stats["peak_retained_anchor_sources"] = 1 if approach == "causal" else 0
            alpha = anchor_alpha.copy()
        elif anchor_frame is None or plan.segment(anchor_frame) != plan.segment(frame):
            raise RuntimeError("past draft has no anchor in this shot/reset segment")
        elif approach == "hold":
            alpha = anchor_alpha.copy()
            diagnostic["approximation"] = "hold_last_matte"
        else:
            diagnostic["propagated_from_frame"] = anchor_frame
            result = propagate(anchor_proxy, source.proxy, anchor_alpha, settings=settings, control=control)
            diagnostic.update(result.diagnostics)
            stats["flow_evaluations"] += result.diagnostics["flow_evaluations"]
            stats["motion_seconds"] += result.diagnostics["motion_seconds"]
            alpha = result.alpha
            fast_motion = adaptive and result.diagnostics.get("motion_fraction", 0) >= motion_threshold
            if result.uncertain.any() or fast_motion:
                diagnostic["fallback"] = fallback
                stats["fallback_frames"].append(frame)
                if fallback == "stop":
                    raise UncertainMotionError(f"Causal Quick Key frame {frame}: uncertain propagation; choose an explicit fallback")
                if fallback == "approximate":
                    alpha = conservative_approximation(result, anchor_alpha)
                else:
                    alpha = coverage(neural_fallback(frame), rgb.shape[:2]).copy()
                    stats["extra_neural_frames"].append(frame)
                    stats["promoted_anchors"].append(frame)
                    stats["promotion_reasons"].append({"frame": frame,
                        "reason": "motion_threshold" if fast_motion else "causal_uncertain_propagation"})
                    stats["anchors"] = sorted(scheduled | set(stats["promoted_anchors"]))
                    diagnostic["promoted_anchor"] = True
                    anchor_frame, anchor_proxy, anchor_alpha = frame, np.asarray(source.proxy, dtype=np.float32).copy(), alpha.copy()
        diagnostic["anchors"] = [anchor_frame, anchor_frame]
        diagnostic["age_since_anchor_frames"] = frame - anchor_frame
        control.check()
        yield DraftFrame(frame, rgb, alpha, diagnostic)
        stats["frames_emitted"] += 1
    control.checkpoint("quick_key_output", plan.frame_count, plan.frame_count)
