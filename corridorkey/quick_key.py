"""Draft anchor scheduling on the native timeline; no neural approximation here."""
from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorPlan:
    frame_count: int
    interval: int
    # Half-open shot/reset ranges; propagation cannot cross these boundaries.
    segments: tuple[tuple[int, int], ...]
    anchors: tuple[int, ...]
    reasons: tuple[tuple[int, tuple[str, ...]], ...]

    @property
    def full_rate(self):
        return self.interval == 1

    def segment(self, frame):
        if type(frame) is not int or not 0 <= frame < self.frame_count:
            raise ValueError("frame must be an integer inside the source timeline")
        index = bisect_right(self.segments, (frame, self.frame_count + 1)) - 1
        return self.segments[index]

    def bracket(self, frame):
        """Offline anchors for a frame; an anchor brackets itself."""
        start, stop = self.segment(frame)
        index = bisect_right(self.anchors, frame) - 1
        previous = self.anchors[index]
        if previous != frame and (index + 1 >= len(self.anchors) or self.anchors[index + 1] >= stop):
            raise ValueError("no future anchor in this segment; a past-only plan cannot bracket this frame")
        following = previous if previous == frame else self.anchors[index + 1]
        if not start <= previous <= following < stop:
            raise RuntimeError("anchor plan crosses a segment boundary")
        return previous, following

    def context(self, anchor, radius, *, past_only=False):
        """Contiguous native frame indices, clipped to the shot/reset segment."""
        if type(radius) is not int or radius < 0:
            raise ValueError("context radius must be a nonnegative integer")
        if anchor not in self.anchors:
            raise ValueError("context is defined only for scheduled anchors")
        start, stop = self.segment(anchor)
        return range(max(start, anchor - radius), anchor + 1 if past_only else min(stop, anchor + radius + 1))


def plan_anchors(frame_count, interval, *, cuts=(), resets=(), corrections=(), max_gap=None, include_final=True):
    """Cuts/reset indices identify the first frame of a new segment.

    Correction frames are mandatory anchors, without restarting the regular
    interval. This is a schedule, not cut detection or correction conditioning.
    """
    if any(type(value) is not int or value < 1 for value in (frame_count, interval)):
        raise ValueError("frame count and interval must be positive integers")
    if max_gap is not None and (type(max_gap) is not int or max_gap < 1):
        raise ValueError("maximum anchor gap must be a positive integer or None")
    if type(include_final) is not bool:
        raise ValueError("include_final must be boolean")
    points = {}
    for name, values in (("cut", cuts), ("reset", resets), ("correction", corrections)):
        values = tuple(values)
        if any(type(value) is not int or not 0 <= value < frame_count for value in values):
            raise ValueError(f"{name} points must be integer source-frame indices")
        points[name] = set(values)
    boundaries = sorted({0, frame_count} | points["cut"] | points["reset"])
    segments = tuple(zip(boundaries, boundaries[1:]))
    reasons = {}
    def add(frame, reason):
        reasons.setdefault(frame, set()).add(reason)
    for start, stop in segments:
        for frame in range(start, stop, interval):
            add(frame, "regular")
        add(start, "segment_start")
        if include_final:
            add(stop - 1, "segment_final")
    for reason, frames in points.items():
        for frame in frames:
            add(frame, reason)
    if max_gap is not None:
        for left, right in zip(sorted(reasons), sorted(reasons)[1:]):
            for frame in range(left + max_gap, right, max_gap):
                add(frame, "max_gap")
        if not include_final:
            for start, stop in segments:
                last = max(frame for frame in reasons if start <= frame < stop)
                for frame in range(last + max_gap, stop, max_gap):
                    add(frame, "max_gap")
    anchors = tuple(sorted(reasons))
    return AnchorPlan(frame_count, interval, segments, anchors,
                      tuple((frame, tuple(sorted(reasons[frame]))) for frame in anchors))
