"""Reference temporal alpha residuals with declared flow/visibility and spatial accuracy."""
from __future__ import annotations

import numpy as np

from .evaluation import _alpha, _metric, alpha_metrics

FLOW_CONVENTION = 'current_to_previous_xy_pixels'


def validate_temporal_settings(flow_convention, consistency_pixels):
    if flow_convention != FLOW_CONVENTION:
        raise ValueError('flow convention must be current_to_previous_xy_pixels; no conversion is implicit')
    if not isinstance(consistency_pixels, (int, float)) or isinstance(consistency_pixels, bool):
        raise TypeError('consistency_pixels must be numeric')
    if not 0 <= consistency_pixels <= float(np.finfo(np.float64).max):
        raise ValueError('consistency_pixels must be finite and nonnegative')


def _frame(value, name):
    if np.asarray(value).ndim != 2:
        raise ValueError(f'{name} must be HxW alpha')
    return _alpha(value, name)[0]


def _validity(value, shape, name):
    if value is None:
        return np.ones(shape, dtype=bool)
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f'{name} must be boolean HxW with the alpha geometry')
    return array


def _flow(value, shape, name):
    array = np.asarray(value)
    if array.shape != (*shape, 2) or array.dtype.kind not in 'uif':
        raise ValueError(f'{name} must be native HxWx2 real pixel displacement')
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def _sampler(shape, flow):
    height, width = shape
    y, x = np.mgrid[:height, :width]
    x, y = x + flow[..., 0], y + flow[..., 1]
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    # Clipping makes indexing safe only. Outside samples are always excluded.
    x, y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    stencil = ((y0, x0, (1 - dy) * (1 - dx)), (y0, x1, (1 - dy) * dx),
               (y1, x0, dy * (1 - dx)), (y1, x1, dy * dx))

    def sample(array):
        result = np.zeros_like(array, dtype=np.float64)
        for row, column, weight in stencil:
            result += array[row, column] * (weight[..., None] if array.ndim == 3 else weight)
        return result

    def source_valid(mask):
        return np.logical_and.reduce([(weight == 0) | mask[row, column] for row, column, weight in stencil])

    return sample, source_valid, inside


def temporal_pair(previous, current, reference_previous=None, reference_current=None, *,
                  backward_flow=None, forward_flow=None, visibility=None, previous_mask=None,
                  current_mask=None, cut=False, flow_convention, consistency_pixels=1.5):
    """Score |(pred_now-warp(pred_prev))-(ref_now-warp(ref_prev))|.

    Backward flow is on the current grid, sampling the previous frame at x+dx,
    y+dy. Optional forward flow is previous->current and only checks consistency.
    Visibility must explicitly exclude occluded/disoccluded current pixels.
    """
    validate_temporal_settings(flow_convention, consistency_pixels)
    if type(cut) is not bool:
        raise TypeError('cut must be boolean')
    before, now = _frame(previous, 'previous'), _frame(current, 'current')
    if before.shape != now.shape:
        raise ValueError('temporal pair geometry differs')
    shape = now.shape
    previous_valid = _validity(previous_mask, shape, 'previous_mask')
    current_valid = _validity(current_mask, shape, 'current_mask')
    reference_available = reference_previous is not None and reference_current is not None
    if (reference_previous is None) != (reference_current is None):
        raise ValueError('both reference frames are required or both absent')
    target_before = _frame(reference_previous, 'reference_previous') if reference_available else None
    target_now = _frame(reference_current, 'reference_current') if reference_available else None
    if target_before is not None and target_now is not None and (target_before.shape != shape or target_now.shape != shape):
        raise ValueError('reference temporal geometry differs')
    empty = _metric(0, 0, 'alpha/pixel pair')
    report = {'schema_version': 1, 'kind': 'temporal_alpha_pair', 'shape_hw': list(shape),
              'flow_convention': flow_convention, 'sampling': 'float64 bilinear, outside excluded',
              'consistency_pixels': consistency_pixels, 'consistency_check': False,
              'forward_flow_provided': forward_flow is not None,
              'temporal_residual_mae': dict(empty), 'candidate_change_mae': dict(empty),
              'reference_change_mae': dict(empty), 'status': 'unavailable', 'reason': None,
              'spatial_previous': alpha_metrics(before, target_before, mask=previous_valid),
              'spatial_current': alpha_metrics(now, target_now, mask=current_valid),
              'cut_excluded': cut, 'valid_pixels': 0, 'out_of_bounds_pixels': 0,
              'note': 'Temporal residual is not accuracy or calibrated flow confidence; inspect paired spatial errors.'}
    if cut:
        report['reason'] = 'shot cut/reset excluded'
        return report
    if backward_flow is None:
        if forward_flow is not None:
            raise ValueError('forward consistency flow requires backward sampling flow')
        report['reason'] = 'backward flow missing'
        return report
    backward = _flow(backward_flow, shape, 'backward_flow')
    visible = _validity(visibility, shape, 'visibility')
    if visibility is None:
        report['reason'] = 'explicit visibility mask missing'
        return report
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            sample, source_valid, inside = _sampler(shape, backward)
            report['out_of_bounds_pixels'] = int((~inside).sum())
            previous_valid_sample = source_valid(previous_valid)
            valid = current_valid & visible & inside & previous_valid_sample
            report['mask_excluded_pixels'] = int((~current_valid | ~previous_valid_sample).sum())
            report['visibility_excluded_pixels'] = int((~visible).sum())
            if forward_flow is not None:
                forward = _flow(forward_flow, shape, 'forward_flow')
                error = backward[valid] + sample(forward)[valid]
                consistent = np.ones(shape, dtype=bool)
                consistent[valid] = np.hypot(error[:, 0], error[:, 1]) <= consistency_pixels
                report['consistency_check'] = True
                report['consistency_excluded_pixels'] = int((valid & ~consistent).sum())
                valid &= consistent
            else:
                report['consistency_excluded_pixels'] = 0
            count = int(valid.sum())
            report['valid_pixels'] = count
            candidate_delta = now[valid] - sample(before)[valid]
            report['candidate_change_mae'] = _metric(np.abs(candidate_delta).sum(dtype=np.float64), count, 'alpha/pixel pair')
            if target_before is not None and target_now is not None:
                target_delta = target_now[valid] - sample(target_before)[valid]
                report['reference_change_mae'] = _metric(np.abs(target_delta).sum(dtype=np.float64), count, 'alpha/pixel pair')
                total = np.abs(candidate_delta - target_delta).sum(dtype=np.float64)
                report['temporal_residual_mae'] = _metric(total, count, 'alpha/pixel pair')
                report['status'] = 'measured' if count else 'unavailable'
                report['reason'] = None if count else 'no valid visible flow correspondences'
            else:
                report['reason'] = 'reference frames missing; change is diagnostic only'
    except FloatingPointError as exc:
        raise ValueError('temporal flow arithmetic overflow/invalid operation') from exc
    return report
