"""Bounded two-frame temporal scoring with independently owned preceding alpha."""
from __future__ import annotations

import math

import numpy as np

from .evaluation import _metric, alpha_metrics
from .evaluation_temporal import (
    FLOW_CONVENTION,
    temporal_pair,
    validate_temporal_settings,
)

SPATIAL = ('mae', 'normalized_sad', 'boundary_mae', 'forward_gradient_mae')
TEMPORAL = ('temporal_residual_mae', 'candidate_change_mae', 'reference_change_mae')


class TemporalAccumulator:
    def __init__(self, *, flow_convention=FLOW_CONVENTION, consistency_pixels=1.5):
        validate_temporal_settings(flow_convention, consistency_pixels)
        self.flow_convention = flow_convention
        self.consistency_pixels = consistency_pixels
        self.previous = None
        self.template = None
        self.reference_available = None
        self.frames = 0
        self.sums = dict.fromkeys((*SPATIAL, *TEMPORAL), 0.0)
        self.counts = dict.fromkeys(self.sums, 0)
        self.pairs = []

    def add(self, frame, alpha, reference=None, *, mask=None, backward_flow=None, forward_flow=None,
            visibility=None, cut=False, flow_previous_frame=None):
        if type(cut) is not bool:
            raise TypeError('cut must be boolean')
        if flow_previous_frame is not None and (type(flow_previous_frame) is not int or flow_previous_frame < 0):
            raise ValueError('flow previous frame must be a nonnegative integer')
        if type(frame) is not int or frame < 0 or (self.previous is not None and frame <= self.previous['frame']):
            raise ValueError('temporal frame IDs must be increasing nonnegative integers')
        if np.asarray(alpha).ndim != 2:
            raise ValueError('temporal stream alpha must be HxW')
        spatial = alpha_metrics(alpha, reference, mask=mask)
        if self.template is not None:
            if spatial['shape_hw'] != self.template['shape_hw']:
                raise ValueError('temporal stream geometry changed')
            if (reference is not None) != self.reference_available:
                raise ValueError('temporal reference availability must be constant')
        pair = None
        if self.previous is not None:
            previous_id = self.previous['frame']
            if flow_previous_frame is not None and flow_previous_frame != previous_id:
                raise ValueError('flow previous frame does not match preceding selected frame')
            if not cut and backward_flow is not None and frame != previous_id + 1 and flow_previous_frame != previous_id:
                raise ValueError('nonconsecutive flow requires explicit previous frame ID')
            pair = temporal_pair(self.previous['alpha'], alpha, self.previous['reference'], reference,
                                 backward_flow=backward_flow, forward_flow=forward_flow, visibility=visibility,
                                 previous_mask=self.previous['mask'], current_mask=mask, cut=cut,
                                 flow_convention=self.flow_convention, consistency_pixels=self.consistency_pixels)
        elif any(value is not None for value in (backward_flow, forward_flow, visibility, flow_previous_frame)):
            raise ValueError('first selected frame cannot declare a preceding transition')
        proposed = dict(self.sums)
        for name in SPATIAL:
            proposed[name] += spatial[name]['error_sum'] or 0.0
        if pair is not None:
            for name in TEMPORAL:
                proposed[name] += pair[name]['error_sum'] or 0.0
        if not all(math.isfinite(value) for value in proposed.values()):
            raise ValueError('temporal metric accumulation overflow')
        # Own copies: a generator/decoder may reuse its input buffer on the next pull.
        saved = {'frame': frame, 'alpha': np.array(alpha, copy=True),
                 'reference': np.array(reference, copy=True) if reference is not None else None,
                 'mask': np.array(mask, copy=True) if mask is not None else None}
        self.sums = proposed
        self.frames += 1
        if self.template is None:
            self.template = spatial
            self.reference_available = reference is not None
        for name in SPATIAL:
            self.counts[name] += spatial[name]['population']
        if pair is not None and self.previous is not None:
            for name in TEMPORAL:
                self.counts[name] += pair[name]['population']
            self.pairs.append({'previous_frame': self.previous['frame'], 'frame': frame,
                               'cut_excluded': pair['cut_excluded'], 'reason': pair['reason'],
                               'valid_pixels': pair['valid_pixels'], 'out_of_bounds_pixels': pair['out_of_bounds_pixels'],
                               'consistency_excluded_pixels': pair.get('consistency_excluded_pixels'),
                               'visibility_excluded_pixels': pair.get('visibility_excluded_pixels'),
                               'mask_excluded_pixels': pair.get('mask_excluded_pixels'),
                               'forward_flow_provided': pair['forward_flow_provided'],
                               'consistency_check': pair['consistency_check'],
                               'temporal_residual_mae': pair['temporal_residual_mae']['value']})
        self.previous = saved

    def snapshot(self):
        if self.template is None:
            raise ValueError('cannot evaluate an empty temporal sequence')
        spatial = {**self.template, 'frames': self.frames}
        spatial['shape_hw'] = list(self.template['shape_hw'])
        for name in SPATIAL:
            spatial[name] = _metric(self.sums[name], self.counts[name], self.template[name]['units'])
        spatial['sad'] = {'value': self.sums['mae'] if self.counts['mae'] else None, 'population': self.counts['mae'],
                           'units': 'alpha sum', 'status': 'measured' if self.counts['mae'] else 'unavailable'}
        spatial['status'] = 'measured' if self.counts['mae'] else 'unavailable'
        spatial['reason'] = None if self.counts['mae'] else ('reference missing' if not self.reference_available
                                                          else 'evaluation mask is empty')
        metrics = {name: _metric(self.sums[name], self.counts[name], 'alpha/pixel pair') for name in TEMPORAL}
        missing = sum(not pair['cut_excluded'] and pair['temporal_residual_mae'] is None for pair in self.pairs)
        temporal_available = self.counts['temporal_residual_mae'] > 0
        spatial_available = self.counts['mae'] > 0
        status = ('measured' if temporal_available and spatial_available and not missing else
                  'partial' if temporal_available or spatial_available else 'unavailable')
        return {'schema_version': 1, 'kind': 'temporal_alpha_accuracy', 'frames': self.frames,
                'shape_hw': list(self.template['shape_hw']), 'spatial': spatial, **metrics,
                'flow_convention': self.flow_convention, 'consistency_pixels': self.consistency_pixels,
                'pairs': [dict(pair) for pair in self.pairs], 'cut_pairs': sum(p['cut_excluded'] for p in self.pairs),
                'unavailable_noncut_pairs': missing, 'status': status,
                'note': 'Spatial accuracy and temporal residual are complementary; stable blank/frozen mattes can be wrong.'}


def compare_temporal_stream(frames, *, flow_convention=FLOW_CONVENTION, consistency_pixels=1.5, check=None):
    candidate = TemporalAccumulator(flow_convention=flow_convention, consistency_pixels=consistency_pixels)
    baseline = None
    expected = None
    for frame in frames:
        if check is not None:
            check()
        present = {name for name in ('reference', 'baseline', 'mask') if name in frame}
        if expected is not None and present != expected:
            raise ValueError('reference/baseline/mask roles must remain constant')
        expected = present
        if 'baseline' in frame and np.asarray(frame['candidate']).shape != np.asarray(frame['baseline']).shape:
            raise ValueError('candidate and baseline temporal geometry differs')
        options = {name: frame[name] for name in ('mask', 'backward_flow', 'forward_flow', 'visibility', 'cut',
                                                'flow_previous_frame') if name in frame}
        candidate.add(frame['frame'], frame['candidate'], frame.get('reference'), **options)
        if 'baseline' in frame:
            if baseline is None:
                baseline = TemporalAccumulator(flow_convention=flow_convention, consistency_pixels=consistency_pixels)
            baseline.add(frame['frame'], frame['baseline'], frame.get('reference'), **options)
    if check is not None:
        check()
    candidate_report = candidate.snapshot()
    baseline_report = baseline.snapshot() if baseline is not None else None
    deltas = {}
    for name in ('spatial_mae', 'temporal_residual_mae'):
        after = candidate_report['spatial']['mae']['value'] if name == 'spatial_mae' else candidate_report[name]['value']
        before = (baseline_report['spatial']['mae']['value'] if name == 'spatial_mae' else baseline_report[name]['value']) if baseline_report is not None else None
        deltas[name] = after - before if after is not None and before is not None else None
    return {'schema_version': 1, 'kind': 'temporal_alpha_comparison', 'candidate': candidate_report,
            'baseline': baseline_report, 'candidate_minus_baseline': deltas, 'status': candidate_report['status'],
            'comparison_status': 'measured' if any(value is not None for value in deltas.values()) else 'unavailable',
            'processing': 'owned preceding alpha/reference/mask only; scalar pair metadata grows with frame count'}
