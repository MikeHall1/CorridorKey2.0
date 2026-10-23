"""Bounded-frame reference evaluation; clip-length image buffers are never retained."""
from __future__ import annotations

import numpy as np

from .evaluation import METRICS, _metric, alpha_metrics, comparison_report


class AlphaAccumulator:
    """Aggregate exact error sums/populations, not the mean of per-frame means."""

    def __init__(self, *, boundary_radius=1):
        self.boundary_radius = boundary_radius
        self.template = None
        self.reference_available = None
        self.frames = 0
        self.sums = dict.fromkeys(METRICS, 0.0)
        self.counts = dict.fromkeys(METRICS, 0)

    def add(self, prediction, reference=None, *, mask=None):
        if np.asarray(prediction).ndim != 2:
            raise ValueError('streamed alpha frames must be HxW arrays')
        report = alpha_metrics(prediction, reference, mask=mask, boundary_radius=self.boundary_radius)
        if self.template is not None:
            if report['shape_hw'] != self.template['shape_hw']:
                raise ValueError('sequence geometry changed')
            if (reference is not None) != self.reference_available:
                raise ValueError('reference availability must be constant across the sequence')
        else:
            self.template = report
            self.reference_available = reference is not None
        self.frames += 1
        for name in METRICS:
            metric = report[name]
            self.counts[name] += metric['population']
            self.sums[name] += (metric['value'] if name == 'sad' else metric['error_sum']) or 0.0

    def snapshot(self):
        if self.template is None:
            raise ValueError('cannot evaluate an empty sequence')
        result = {**self.template, 'frames': self.frames}
        for name in METRICS:
            result[name] = _metric(self.sums[name], self.counts[name], self.template[name]['units'])
        result['sad']['value'] = self.sums['sad'] if self.counts['sad'] else None
        if self.counts['mae']:
            result.update(status='measured', reason=None)
        else:
            result.update(status='unavailable', reason='reference missing' if not self.reference_available
                          else 'evaluation mask is empty')
        return result


def compare_stream(candidate_frames, reference_frames=None, *, baseline_frames=None,
                   masks=None, boundary_radius=1, check=None):
    """Consume aligned iterables once and reject unequal lengths or changed geometry."""
    candidate_acc = AlphaAccumulator(boundary_radius=boundary_radius)
    baseline_acc = AlphaAccumulator(boundary_radius=boundary_radius) if baseline_frames is not None else None
    others = {name: iter(values) for name, values in
              (('reference', reference_frames), ('baseline', baseline_frames), ('mask', masks))
              if values is not None}
    sentinel = object()
    for frame in candidate_frames:
        if check is not None:
            check()
        values = {name: next(iterator, sentinel) for name, iterator in others.items()}
        if any(value is sentinel for value in values.values()):
            raise ValueError('evaluation sequence lengths differ')
        reference, mask = values.get('reference'), values.get('mask')
        candidate_acc.add(frame, reference, mask=mask)
        if baseline_acc is not None:
            baseline = values['baseline']
            if np.asarray(baseline).shape != np.asarray(frame).shape:
                raise ValueError('candidate and baseline geometry differs')
            baseline_acc.add(baseline, reference, mask=mask)
    if any(next(iterator, sentinel) is not sentinel for iterator in others.values()):
        raise ValueError('evaluation sequence lengths differ')
    if check is not None:
        check()
    report = comparison_report(candidate_acc.snapshot(),
                               baseline_acc.snapshot() if baseline_acc is not None else None)
    report['processing'] = 'single chronological pass; bounded image buffers; metadata scales with frame count'
    return report
