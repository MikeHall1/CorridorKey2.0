"""Bounded-frame aggregation of declared linear RGB and compositing metrics."""
from __future__ import annotations

import math

import numpy as np

from .evaluation import _metric
from .evaluation_colour import colour_metrics

COLOUR_METRICS = ('foreground_mae', 'composite_mae')
ALPHA_METRICS = ('mae', 'normalized_sad', 'boundary_mae', 'forward_gradient_mae')


def availability(report):
    count = sum(report[name]['value'] is not None for name in COLOUR_METRICS)
    return 'measured' if count == 2 else 'partial' if count else 'unavailable'


class ColourAccumulator:
    def __init__(self, **contract):
        self.contract = contract
        self.template = None
        self.reference_available = None
        self.frames = 0
        self.sums = dict.fromkeys((*COLOUR_METRICS, *ALPHA_METRICS), 0.0)
        self.counts = dict.fromkeys(self.sums, 0)
        self.pixel_counts = dict.fromkeys(('foreground_pixels', 'candidate_undefined_foreground_pixels',
                                         'reference_excluded_foreground_pixels'), 0)
        self.rgb_min, self.rgb_max = float('inf'), -float('inf')

    def add(self, candidate_rgb, candidate_alpha, reference_rgb=None, reference_alpha=None, **options):
        if np.asarray(candidate_rgb).ndim != 3:
            raise ValueError('streamed RGB frames must be HxWx3')
        report = colour_metrics(candidate_rgb, candidate_alpha, reference_rgb, reference_alpha,
                                **self.contract, **options)
        available = reference_rgb is not None and reference_alpha is not None
        if self.template is not None:
            if self.template['shape_hw'] != report['shape_hw']:
                raise ValueError('colour sequence geometry changed')
            if available != self.reference_available:
                raise ValueError('reference availability must be constant across the colour sequence')
        else:
            self.template = report
            self.reference_available = available
        next_sums = {}
        for name in self.sums:
            metric = report[name] if name in COLOUR_METRICS else report['alpha'][name]
            next_sums[name] = self.sums[name] + (metric['error_sum'] or 0.0)
        if not all(math.isfinite(value) for value in next_sums.values()):
            raise ValueError('colour sequence error accumulation overflow')
        self.frames += 1
        self.sums = next_sums
        for name in self.sums:
            metric = report[name] if name in COLOUR_METRICS else report['alpha'][name]
            self.counts[name] += metric['population']
        for name in self.pixel_counts:
            self.pixel_counts[name] += report[name]
        self.rgb_min = min(self.rgb_min, report['candidate_rgb_range'][0])
        self.rgb_max = max(self.rgb_max, report['candidate_rgb_range'][1])

    def snapshot(self):
        if self.template is None:
            raise ValueError('cannot evaluate an empty colour sequence')
        result = {**self.template, 'frames': self.frames,
                  'candidate_rgb_range': [self.rgb_min, self.rgb_max], **self.pixel_counts}
        result['alpha'] = {**self.template['alpha'], 'frames': self.frames}
        for name in self.sums:
            metric = self.template[name] if name in COLOUR_METRICS else self.template['alpha'][name]
            merged = _metric(self.sums[name], self.counts[name], metric['units'])
            if name in COLOUR_METRICS:
                result[name] = merged
                if merged['value'] is not None:
                    result[name.removesuffix('_mae') + '_reason'] = None
            else:
                result['alpha'][name] = merged
        result['alpha']['sad'] = {'value': self.sums['mae'] if self.counts['mae'] else None,
                                  'population': self.counts['mae'], 'units': 'alpha sum',
                                  'status': 'measured' if self.counts['mae'] else 'unavailable'}
        result['alpha']['status'] = 'measured' if self.counts['mae'] else 'unavailable'
        result['alpha']['reason'] = None if self.counts['mae'] else (
            'reference missing' if not self.reference_available else 'evaluation mask is empty')
        result['status'] = availability(result)
        return result


def compare_colour_reports(candidate, baseline=None):
    deltas = {}
    for name in COLOUR_METRICS:
        after = candidate[name]['value']
        before = baseline[name]['value'] if baseline is not None else None
        deltas[name] = after - before if after is not None and before is not None else None
    return {'schema_version': 1, 'kind': 'linear_colour_comparison', 'candidate': candidate,
            'baseline': baseline, 'candidate_minus_baseline': deltas,
            'status': availability(candidate),
            'comparison_status': 'measured' if any(v is not None for v in deltas.values()) else 'unavailable',
            'note': 'Identical references/backgrounds/masks; negative deltas mean lower declared linear error only.'}


def compare_colour_stream(frames, *, colour_space, candidate_association, reference_association,
                          baseline_association=None, unpremultiply_epsilon=1e-6, check=None):
    """Frames are explicit aligned dictionaries; optional roles stay constant."""
    contract = {'colour_space': colour_space, 'candidate_association': candidate_association,
                'reference_association': reference_association, 'unpremultiply_epsilon': unpremultiply_epsilon}
    candidate = ColourAccumulator(**contract)
    baseline = None
    expected_roles = None
    for frame in frames:
        if check is not None:
            check()
        if expected_roles is not None and set(frame) != expected_roles:
            raise ValueError('colour frame roles must remain constant')
        expected_roles = set(frame)
        options = {name: frame[name] for name in ('foreground_mask', 'composite_mask', 'background') if name in frame}
        candidate.add(frame['candidate_rgb'], frame['candidate_alpha'], frame.get('reference_rgb'),
                      frame.get('reference_alpha'), **options)
        if 'baseline_rgb' in frame:
            if baseline_association is None:
                raise ValueError('baseline association must be declared')
            if baseline is None:
                baseline = ColourAccumulator(**{**contract, 'candidate_association': baseline_association})
            baseline.add(frame['baseline_rgb'], frame['baseline_alpha'], frame.get('reference_rgb'),
                         frame.get('reference_alpha'), **options)
    if check is not None:
        check()
    report = compare_colour_reports(candidate.snapshot(), baseline.snapshot() if baseline is not None else None)
    report['processing'] = 'bounded decoded image buffers; summed channel errors/populations across frames'
    return report
