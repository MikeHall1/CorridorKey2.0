"""Declared linear-light foreground/composite accuracy, without colour inference."""
from __future__ import annotations

import numpy as np

from .evaluation import _alpha, _metric, alpha_metrics

COLOUR_SPACE = 'linear_srgb_bt709'


def _rgb(value, name):
    array = np.asarray(value)
    if (array.ndim not in (3, 4) or array.shape[-1] != 3 or 0 in array.shape or
            array.dtype.kind not in 'uif'):
        raise ValueError(f'{name} must be HxWx3 or TxHxWx3 real RGB')
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite; HDR/negative RGB is permitted without clipping')
    return array[None] if array.ndim == 3 else array


def _mask(value, shape, name):
    if value is None:
        return np.ones(shape, dtype=bool)
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ValueError(f'{name} must be boolean')
    array = array[None] if array.ndim == 2 else array
    if array.shape != shape:
        raise ValueError(f'{name} geometry must match RGB/alpha')
    return array


def _representations(rgb, alpha, association, epsilon):
    if association == 'straight':
        return rgb, rgb * alpha[..., None]
    if np.any(rgb[alpha == 0] != 0):
        raise ValueError('premultiplied RGB must be zero at zero alpha')
    straight = np.zeros_like(rgb)
    np.divide(rgb, alpha[..., None], out=straight, where=alpha[..., None] > epsilon)
    return straight, rgb


def colour_metrics(candidate_rgb, candidate_alpha, reference_rgb=None, reference_alpha=None, *,
                   colour_space, candidate_association, reference_association, background=None,
                   foreground_mask=None, composite_mask=None, unpremultiply_epsilon=1e-6):
    """Score channel-weighted absolute errors in the explicitly declared linear space.

    Foreground MAE requires a supplied valid-reference mask. Premultiplied RGB
    unpremultiplies only above epsilon; candidate colour is zero otherwise and
    those selected pixels are counted. Composite MAE uses the same background.
    """
    if colour_space != COLOUR_SPACE:
        raise ValueError('only explicit linear_srgb_bt709 is supported; no colour transform is inferred')
    if candidate_association not in ('straight', 'premultiplied') or reference_association not in ('straight', 'premultiplied'):
        raise ValueError('association must be straight or premultiplied')
    epsilon = unpremultiply_epsilon
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise TypeError('unpremultiply_epsilon must be numeric')
    if not 0 <= epsilon < 1 or not np.isfinite(epsilon):
        raise ValueError('unpremultiply_epsilon must be finite in [0,1)')
    rgb = _rgb(candidate_rgb, 'candidate_rgb')
    alpha = _alpha(candidate_alpha, 'candidate_alpha')
    if rgb.shape[:-1] != alpha.shape:
        raise ValueError('candidate RGB/alpha geometry differs')
    composite_valid = _mask(composite_mask, alpha.shape, 'composite_mask')
    foreground_valid = _mask(foreground_mask, alpha.shape, 'foreground_mask')
    missing = _metric(0, 0, 'linear RGB/channel')
    result = {'schema_version': 1, 'kind': 'linear_colour_accuracy', 'colour_space': colour_space,
              'frames': int(alpha.shape[0]), 'shape_hw': list(alpha.shape[1:]),
              'candidate_association': candidate_association, 'reference_association': reference_association,
              'unpremultiply_epsilon': epsilon, 'undefined_straight_colour_policy': 'zero at alpha<=epsilon for premultiplied input',
              'foreground_mae': dict(missing), 'composite_mae': dict(missing),
              'foreground_reason': 'reference RGB/alpha missing', 'composite_reason': 'reference RGB/alpha missing',
              'foreground_pixels': 0, 'candidate_undefined_foreground_pixels': 0,
              'reference_excluded_foreground_pixels': 0,
              'alpha': alpha_metrics(alpha, reference_alpha, mask=composite_valid),
              'candidate_rgb_range': [float(rgb.min()), float(rgb.max())],
              'note': 'Absolute channel errors in declared linear units, not perceptual/peak-normalized scores. '
                      'No clipping, automatic transfer/primaries conversion or visual-quality claim.'}
    # Validate the candidate association even when no reference exists.
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            straight, premultiplied = _representations(rgb, alpha, candidate_association, epsilon)
            if reference_rgb is None or reference_alpha is None:
                return result
            target = _rgb(reference_rgb, 'reference_rgb')
            target_alpha = _alpha(reference_alpha, 'reference_alpha')
            if target.shape != rgb.shape or target_alpha.shape != alpha.shape:
                raise ValueError('reference RGB/alpha geometry must match candidate')
            target_straight, target_premultiplied = _representations(target, target_alpha, reference_association, epsilon)
            if foreground_mask is not None:
                selected = foreground_valid & (target_alpha > epsilon)
                count = int(selected.sum())
                total = np.abs(straight[selected] - target_straight[selected]).sum(dtype=np.float64)
                result['foreground_mae'] = _metric(total, count * 3, 'linear RGB/channel')
                result['foreground_pixels'] = count
                result['candidate_undefined_foreground_pixels'] = (int((selected & (alpha <= epsilon)).sum())
                                                                   if candidate_association == 'premultiplied' else 0)
                result['reference_excluded_foreground_pixels'] = int((foreground_valid & ~selected).sum())
                result['foreground_reason'] = None if count else 'no valid foreground pixels above reference alpha epsilon'
            else:
                result['foreground_reason'] = 'valid-reference foreground mask missing'
            if background is not None:
                bg = np.asarray(background)
                if bg.shape == (3,):
                    bg = np.broadcast_to(bg, rgb.shape)
                else:
                    bg = _rgb(bg, 'background')
                if bg.shape != rgb.shape or bg.dtype.kind not in 'uif' or not np.isfinite(bg).all():
                    raise ValueError('background must be finite RGB of matching geometry or a constant RGB triplet')
                rendered = premultiplied[composite_valid] + bg[composite_valid] * (1 - alpha[composite_valid, None])
                target_rendered = (target_premultiplied[composite_valid] + bg[composite_valid] *
                                   (1 - target_alpha[composite_valid, None]))
                total = np.abs(rendered - target_rendered).sum(dtype=np.float64)
                result['composite_mae'] = _metric(total, int(composite_valid.sum()) * 3, 'linear RGB/channel')
                result['composite_reason'] = None if composite_valid.any() else 'composite mask is empty'
            else:
                result['composite_reason'] = 'linear background missing'
    except FloatingPointError as exc:
        raise ValueError('colour metric arithmetic overflow/invalid operation') from exc
    return result
