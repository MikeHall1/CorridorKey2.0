"""Reference alpha accuracy metrics; no inference or perceptual quality claims."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _alpha(value, name):
    array = np.asarray(value)
    if array.ndim not in (2, 3) or 0 in array.shape or array.dtype.kind not in 'buif':
        raise ValueError(f'{name} must be a nonempty HxW or TxHxW real alpha array')
    array = array.astype(np.float64)
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError(f'{name} must contain finite alpha in [0,1]; no implicit normalization')
    return array[None] if array.ndim == 2 else array


def _metric(total, count, units):
    return {'value': float(total / count) if count else None, 'population': int(count),
            'error_sum': float(total) if count else None,
            'units': units, 'status': 'measured' if count else 'unavailable'}


def _boundary(reference, radius):
    # All partial-coverage pixels plus both sides of a four-connected 0.5 contour.
    inside = reference >= 0.5
    band = (reference > 0) & (reference < 1)
    horizontal = inside[..., 1:] != inside[..., :-1]
    vertical = inside[..., 1:, :] != inside[..., :-1, :]
    band[..., 1:] |= horizontal
    band[..., :-1] |= horizontal
    band[..., 1:, :] |= vertical
    band[..., :-1, :] |= vertical
    for _ in range(radius):
        padded = np.pad(band, ((0, 0), (1, 1), (1, 1)), constant_values=False)
        band = np.logical_or.reduce([padded[:, y:y + band.shape[1], x:x + band.shape[2]]
                                     for y in range(3) for x in range(3)])
    return band


def alpha_metrics(prediction, reference=None, *, mask=None, boundary_radius=1):
    """Pixel-weighted metrics across frames; masked gradients require both endpoints.

    Mask is boolean with the original prediction shape. Radius is a nonnegative
    Chebyshev dilation of the declared contour/soft-alpha seeds, in pixels.
    """
    if type(boundary_radius) is not int or not 0 <= boundary_radius <= 64:
        raise ValueError('boundary_radius must be an integer in [0,64]')
    predicted = _alpha(prediction, 'prediction')
    if mask is None:
        evaluated = np.ones(predicted.shape, dtype=bool)
    else:
        raw_mask = np.asarray(mask)
        if raw_mask.dtype != np.bool_ or raw_mask.shape != np.asarray(prediction).shape:
            raise ValueError('mask must be boolean with the prediction shape')
        evaluated = raw_mask[None] if raw_mask.ndim == 2 else raw_mask
    result = {'schema_version': 1, 'kind': 'reference_alpha_accuracy',
              'frames': int(predicted.shape[0]), 'shape_hw': list(predicted.shape[1:]),
              'boundary_radius_pixels': boundary_radius,
              'boundary_definition': 'soft coverage or both sides of 4-connected >=0.5 contour; Chebyshev dilation',
              'gradient_definition': 'absolute difference of signed forward x/y differences; valid endpoint pairs only',
              'aggregation': 'pixel/pair weighted across frames; no transfer function on alpha',
              'status': 'unavailable' if reference is None else 'measured',
              'reason': 'reference missing' if reference is None else None}
    if reference is None:
        for name in ('mae', 'normalized_sad', 'boundary_mae', 'forward_gradient_mae'):
            result[name] = _metric(0, 0, 'alpha/pixel' if name != 'forward_gradient_mae' else 'alpha/pair')
        result['sad'] = {'value': None, 'population': 0, 'units': 'alpha sum', 'status': 'unavailable'}
        return result
    target = _alpha(reference, 'reference')
    if target.shape != predicted.shape:
        raise ValueError('prediction and reference shapes must match; no resizing/broadcasting')
    error = np.abs(predicted - target)
    population = int(evaluated.sum())
    total = float(error[evaluated].sum(dtype=np.float64))
    result['mae'] = _metric(total, population, 'alpha/pixel')
    result['normalized_sad'] = dict(result['mae'])
    result['sad'] = {'value': total if population else None, 'population': population,
                     'units': 'alpha sum', 'status': 'measured' if population else 'unavailable'}
    boundary = _boundary(target, boundary_radius) & evaluated
    result['boundary_mae'] = _metric(error[boundary].sum(dtype=np.float64),
                                     boundary.sum(), 'alpha/pixel')
    gradient_total = 0.0
    gradient_count = 0
    for axis in (-1, -2):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        valid = evaluated[tuple(first)] & evaluated[tuple(second)]
        differences = np.abs(np.diff(predicted, axis=axis) - np.diff(target, axis=axis))
        gradient_total += float(differences[valid].sum(dtype=np.float64))
        gradient_count += int(valid.sum())
    result['forward_gradient_mae'] = _metric(gradient_total, gradient_count, 'alpha/pair')
    if not population:
        result.update(status='unavailable', reason='evaluation mask is empty')
    return result


METRICS = ('mae', 'normalized_sad', 'sad', 'boundary_mae', 'forward_gradient_mae')


def compare_alpha(candidate, reference=None, *, baseline=None, mask=None, boundary_radius=1):
    """Compare accuracy using identical references/masks; negative deltas mean less error."""
    candidate_metrics = alpha_metrics(candidate, reference, mask=mask, boundary_radius=boundary_radius)
    baseline_metrics = (alpha_metrics(baseline, reference, mask=mask, boundary_radius=boundary_radius)
                        if baseline is not None else None)
    if baseline is not None and np.asarray(candidate).shape != np.asarray(baseline).shape:
        raise ValueError('candidate and baseline shapes must match')
    return comparison_report(candidate_metrics, baseline_metrics)


def comparison_report(candidate_metrics, baseline_metrics=None):
    """Build deltas from alpha reports evaluated over the same frames/reference/mask."""
    deltas = {}
    for name in METRICS:
        after = candidate_metrics[name]['value']
        before = baseline_metrics[name]['value'] if baseline_metrics is not None else None
        deltas[name] = after - before if after is not None and before is not None else None
    return {'schema_version': 1, 'kind': 'alpha_quality_comparison',
            'candidate': candidate_metrics, 'baseline': baseline_metrics,
            'candidate_minus_baseline': deltas,
            'comparison_status': 'measured' if deltas['mae'] is not None else 'unavailable',
            'note': 'Negative error deltas mean lower error on this reference/mask only. '
                    'Raw SAD comparisons require equal populations. No GPU, visual or perceptual claim.',
            'unavailable_metrics': ['foreground_colour', 'composite', 'motion_compensated_temporal',
                                    'blinded_visual_review']}


def evaluate_files(candidate, *, reference=None, baseline=None, mask=None, boundary_radius=1):
    """Evaluate local normalized .npy arrays without pickle or implicit colour conversion."""
    from .identities import file_sha256

    arrays = {}
    assets = {}
    for name, source in (('candidate', candidate), ('reference', reference),
                         ('baseline', baseline), ('mask', mask)):
        if source is None:
            arrays[name] = None
            assets[name] = None
            continue
        path = Path(source).resolve()
        if path.suffix.lower() != '.npy':
            raise ValueError('evaluation inputs must be local .npy arrays; no image/video decoder is implicit')
        digest = file_sha256(path)
        with path.open('rb') as stream:
            value = np.lib.format.read_array(stream, allow_pickle=False)
        if file_sha256(path) != digest:
            raise ValueError(f'evaluation input changed while reading: {path}')
        arrays[name] = value
        assets[name] = {'path': str(path), 'sha256': digest, 'shape': list(value.shape),
                        'dtype': str(value.dtype)}
    report = compare_alpha(arrays['candidate'], arrays['reference'], baseline=arrays['baseline'],
                           mask=arrays['mask'], boundary_radius=boundary_radius)
    report['assets'] = assets
    report['array_contract'] = 'normalized alpha [0,1], HxW or TxHxW; boolean mask; no alpha transfer function'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--boundary-radius', type=int, default=1)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve() in {p.resolve() for p in
                               (args.candidate, args.reference, args.baseline, args.mask) if p is not None}:
        parser.error('report output must differ from all input assets')
    report = evaluate_files(args.candidate, reference=args.reference, baseline=args.baseline,
                            mask=args.mask, boundary_radius=args.boundary_radius)
    from .diagnostics import write_json_atomic

    write_json_atomic(args.output, report)
    print(f"alpha comparison: {report['comparison_status']}; report: {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
