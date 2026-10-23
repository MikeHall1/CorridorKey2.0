"""Offline, bounded paired quality evidence; screens are not product acceptance."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from .benchmark_report import MAX_REPORT_BYTES, read_report
from .control import CancelledError, ProgressEvent
from .local_json import read_object, read_regular_bytes

MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_CASES = 4096
MAX_REPETITIONS = 128
METRICS = {
    'alpha.mae': 'alpha/pixel', 'alpha.boundary_mae': 'alpha/pixel',
    'alpha.forward_gradient_mae': 'alpha/pair', 'colour.foreground_mae': 'linear RGB/channel',
    'colour.composite_mae': 'linear RGB/channel', 'temporal.temporal_residual_mae': 'alpha/pixel pair',
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _check(check):
    if check is not None:
        check()


def _text(value):
    return isinstance(value, str) and 0 < len(value) <= 128 and not any(ord(c) < 32 for c in value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _identity(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _manifest(path):
    value, raw = read_object(path, max_bytes=1024 * 1024, max_depth=32, regular_file=True)
    if (set(value) - {'schema_version', 'kind', 'reports', 'cases', 'policy'}
            or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('kind') != 'evaluation_dossier'):
        raise ValueError('Unsupported dossier manifest schema')
    reports, cases = value.get('reports'), value.get('cases')
    if not isinstance(reports, list) or not 1 <= len(reports) <= 32:
        raise ValueError('Dossier requires1..32 report mappings')
    ids = set()
    for item in reports:
        if (not isinstance(item, dict) or set(item) != {'id', 'path'} or not _text(item['id'])
                or item['id'] in ids or not isinstance(item['path'], str) or not item['path']):
            raise ValueError('Invalid or duplicate report mapping')
        ids.add(item['id'])
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError('Dossier requires1..4096 explicitly mapped cases')
    seen = set()
    groups = set()
    memberships = 0
    selectors = set()
    for case in cases:
        if (not isinstance(case, dict) or set(case) - {'id', 'report_id', 'job_index', 'categories', 'split', 'required', 'review_pair_index'}
                or not _text(case.get('id')) or case['id'] in seen or case.get('report_id') not in ids
                or type(case.get('job_index')) is not int or case['job_index'] < 0
                or not _text(case.get('split')) or type(case.get('required', True)) is not bool
                or type(case.get('review_pair_index', 0)) is not int or case.get('review_pair_index', 0) < 0):
            raise ValueError('Invalid or duplicate case mapping')
        categories = case.get('categories')
        if (not isinstance(categories, list) or not 1 <= len(categories) <= 32
                or any(not _text(category) for category in categories) or len(set(categories)) != len(categories)):
            raise ValueError('Case categories must be explicitly distinct')
        groups.update((category, case['split']) for category in categories)
        memberships += len(categories)
        if len(groups) > 128 or memberships > 8192:
            raise ValueError('Dossier category/membership bound exceeded')
        selector = (case['report_id'], case['job_index'])
        if selector in selectors:
            raise ValueError('A report job cannot be counted as multiple cases')
        selectors.add(selector)
        seen.add(case['id'])
    policy = value.get('policy', {})
    if not isinstance(policy, dict) or set(policy) - {'relative_regression_limit', 'zero_baseline_absolute_tolerances'}:
        raise ValueError('Unknown threshold policy field')
    limit = policy.get('relative_regression_limit', 0.05)
    if not _number(limit) or not 0 <= limit <= 1:
        raise ValueError('Relative regression limit must be finite in0..1')
    absolute = policy.get('zero_baseline_absolute_tolerances', {})
    if not isinstance(absolute, dict) or set(absolute) - set(METRICS):
        raise ValueError('Unknown absolute-tolerance metric')
    for name, tolerance in absolute.items():
        if (not isinstance(tolerance, dict) or set(tolerance) != {'units', 'value'}
                or tolerance['units'] != METRICS[name] or not _number(tolerance['value']) or tolerance['value'] < 0):
            raise ValueError('Absolute tolerances require the exact metric units and nonnegative value')
    value['policy'] = {'relative_regression_limit': limit, 'zero_baseline_absolute_tolerances': absolute,
        'scope': 'Regression screen only; declared policy is recorded, chronological predeclaration is not verified.'}
    return value, raw


def _sample_identity(sample):
    required = ('repository', 'assets', 'weights', 'corrections', 'environment')
    if any(name not in sample for name in required):
        raise ValueError('Sample lacks comparable repository/asset/environment evidence')
    if any(not isinstance(sample[name], dict) or not sample[name] for name in ('repository', 'assets', 'weights')):
        raise ValueError('Sample identity evidence is empty')
    env = sample['environment']
    if not isinstance(env, dict) or any(name not in env for name in ('python', 'os', 'dependencies', 'hardware')):
        raise ValueError('Sample lacks comparable environment evidence')
    hardware = env['hardware']
    return {name: sample[name] for name in required if name != 'environment'} | {
        'environment': {name: env[name] for name in ('python', 'os', 'dependencies')},
        'hardware': {name: hardware.get(name) for name in ('cpu', 'ram_bytes', 'gpu', 'cuda_version', 'driver', 'device_index')}}


def _quality(sample, job):
    if sample.get('inference', {}).get('status') != 'measured':
        raise ValueError('Sample inference is not measured')
    entries = sample.get('inference', {}).get('quality', {}).get('jobs', [])
    if not isinstance(entries, list):
        raise TypeError('Missing quality job records')
    matches = [entry for entry in entries if isinstance(entry, dict) and type(entry.get('run')) is int and entry['run'] == 0 and type(entry.get('job')) is int and entry['job'] == job]
    if len(matches) != 1:
        raise ValueError('Quality run0/job selection is missing or duplicated')
    entry = matches[0]
    if entry.get('status') not in ('measured', 'partial') or not isinstance(entry.get('report'), dict):
        raise ValueError('Selected quality record is failed or unavailable')
    report = entry['report']
    if report.get('kind') not in ('alpha_sequence_evaluation', 'colour_sequence_evaluation', 'temporal_sequence_evaluation'):
        raise ValueError('Unsupported native sequence quality report')
    if type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise ValueError('Unsupported quality report schema')
    return report


def _metric(report, name):
    candidate = report.get('candidate', {})
    if name.startswith('alpha.'):
        if report['kind'] == 'colour_sequence_evaluation':
            candidate = candidate.get('alpha', {})
        elif report['kind'] == 'temporal_sequence_evaluation':
            candidate = candidate.get('spatial', {})
        return candidate.get(name.split('.')[1]), candidate
    if name.startswith('colour.') and report['kind'] == 'colour_sequence_evaluation':
        return candidate.get(name.split('.')[1]), candidate
    if name.startswith('temporal.') and report['kind'] == 'temporal_sequence_evaluation':
        return candidate.get(name.split('.')[1]), candidate
    return None, candidate


def _asset_contract(report):
    assets = report.get('assets')
    if not isinstance(assets, list) or len(assets) > 65536:
        raise ValueError('Quality asset inventory is missing or oversized')
    result = []
    seen = set()
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get('role'), str):
            raise TypeError('Invalid quality asset role')
        if asset['role'].startswith('candidate') or asset['role'].startswith('baseline'):
            continue
        if (type(asset.get('frame')) is not int or asset['frame'] < 0
                or not isinstance(asset.get('sha256'), str) or re.fullmatch('[0-9a-f]{64}', asset['sha256']) is None):
            raise ValueError('Invalid reference asset frame/hash')
        key = (asset.get('frame'), asset['role'])
        if key in seen:
            raise ValueError('Duplicate reference asset role/frame')
        seen.add(key)
        result.append({name: item for name, item in asset.items() if name != 'path'})
    return sorted(result, key=lambda item: (item.get('frame', -1), item['role']))


def _relevant(value):
    if isinstance(value, dict):
        return {key: _relevant(item) for key, item in value.items() if key != 'baseline' and not key.startswith('baseline_')}
    return value


def _compatibility(left, right, name):
    if left.get('kind') != right.get('kind'):
        raise ValueError('Metric families differ')
    for field in ('frame_ids', 'fps', 'decode_contract', 'reference_provenance'):
        if field not in left or left[field] != right.get(field):
            raise ValueError(f'Quality {field} differs or is missing')
    ids = left['frame_ids']
    if (not isinstance(ids, list) or not ids or len(ids) > 65536
            or any(type(frame) is not int or frame < 0 for frame in ids) or len(set(ids)) != len(ids)):
        raise ValueError('Invalid selected frame coverage')
    if not _number(left['fps']) or left['fps'] <= 0:
        raise ValueError('Invalid declared FPS')
    for report in (left, right):
        source = report.get('source')
        if not isinstance(source, dict) or not isinstance(source.get('sha256'), str) or re.fullmatch('[0-9a-f]{64}', source['sha256']) is None:
            raise ValueError('Missing source identity')
    if left['source']['sha256'] != right['source']['sha256']:
        raise ValueError('Source content differs')
    binding_left, binding_right = left.get('candidate_binding', {}), right.get('candidate_binding', {})
    required_binding = {'method', 'source_start_frame', 'frames_submitted', 'selected_frames', 'known_shot_boundaries'}
    for binding in (binding_left, binding_right):
        if (not isinstance(binding, dict) or not required_binding <= set(binding)
                or binding['method'] != 'completed native EXR exports, output-relative frame IDs'
                or type(binding['source_start_frame']) is not int or binding['source_start_frame'] < 0
                or type(binding['frames_submitted']) is not int or binding['frames_submitted'] <= 0
                or type(binding['selected_frames']) is not int or binding['selected_frames'] != len(ids)):
            raise ValueError('Completed native output/source range binding is missing or invalid')
    for field in ('source_start_frame', 'frames_submitted', 'selected_frames', 'known_shot_boundaries'):
        if binding_left.get(field) != binding_right.get(field):
            raise ValueError('Source/output range binding differs')
    asset_a, asset_b = _asset_contract(left), _asset_contract(right)
    required_reference = 'reference_alpha' if left['kind'] == 'colour_sequence_evaluation' else 'reference'
    if {asset['frame'] for asset in asset_a if asset['role'] == required_reference} != set(ids):
        raise ValueError('Reference asset coverage is incomplete')
    if asset_a != asset_b:
        raise ValueError('Reference/mask/background/flow asset identities differ')
    a, contract_a = _metric(left, name)
    b, contract_b = _metric(right, name)
    mandatory = (('shape_hw', 'frames', 'boundary_radius_pixels', 'boundary_definition', 'gradient_definition', 'aggregation')
                 if name.startswith('alpha.') else ('shape_hw', 'frames', 'colour_space', 'candidate_association',
                     'reference_association', 'unpremultiply_epsilon', 'undefined_straight_colour_policy')
                 if name.startswith('colour.') else ('shape_hw', 'frames', 'flow_convention', 'consistency_pixels', 'pairs'))
    if any(field not in contract_a or field not in contract_b for field in mandatory):
        raise ValueError('Metric definition/geometry contract is incomplete')
    for field in ('shape_hw', 'frames', 'boundary_radius_pixels', 'boundary_definition', 'gradient_definition', 'aggregation',
                  'colour_space', 'candidate_association', 'reference_association', 'unpremultiply_epsilon',
                  'undefined_straight_colour_policy', 'flow_convention', 'consistency_pixels', 'pairs'):
        if field != 'pairs' and contract_a.get(field) != contract_b.get(field):
            raise ValueError(f'Metric {field} differs')
    if name.startswith('colour.'):
        if contract_a.get('colour_space') != 'linear_srgb_bt709':
            raise ValueError('Colour primaries/transfer contract is unsupported')
        evaluation_a, evaluation_b = left.get('evaluation_contract'), right.get('evaluation_contract')
        if not isinstance(evaluation_a, dict) or not isinstance(evaluation_b, dict):
            raise ValueError('Legacy colour report lacks complete background/evaluation contract')
        if _relevant(evaluation_a) != _relevant(evaluation_b):
            raise ValueError('Declared colour evaluation contracts differ')
    if name.startswith('temporal.'):
        for field in ('flow_provenance',):
            if left.get(field) != right.get(field):
                raise ValueError('Temporal provenance differs')
        pairs_a, pairs_b = contract_a.get('pairs'), contract_b.get('pairs')
        if not isinstance(pairs_a, list) or not isinstance(pairs_b, list):
            raise ValueError('Missing temporal transition coverage')
        clean = lambda pairs: [{k: v for k, v in pair.items() if k != 'temporal_residual_mae'} for pair in pairs]
        if clean(pairs_a) != clean(pairs_b):
            raise ValueError('Temporal transitions/cut/validity coverage differs')
    for metric in (a, b):
        if (not isinstance(metric, dict) or metric.get('units') != METRICS[name]
                or metric.get('status') != 'measured' or type(metric.get('population')) is not int
                or metric['population'] <= 0 or not _number(metric.get('value')) or metric['value'] < 0
                or not _number(metric.get('error_sum')) or metric['error_sum'] < 0
                or not math.isclose(metric['value'], metric['error_sum'] / metric['population'], rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError('Metric unavailable or normalized arithmetic/units invalid')
    assert isinstance(a, dict) and isinstance(b, dict)
    if a['population'] != b['population']:
        raise ValueError('Compared metric populations differ')
    return a, b


def _definition(report, name):
    _, contract = _metric(report, name)
    fields = ('boundary_radius_pixels', 'boundary_definition', 'gradient_definition', 'aggregation') if name.startswith('alpha.') else (
        'colour_space', 'candidate_association', 'reference_association', 'unpremultiply_epsilon',
        'undefined_straight_colour_policy') if name.startswith('colour.') else ('flow_convention', 'consistency_pixels')
    return {'decode_contract': report.get('decode_contract'), **{field: contract[field] for field in fields}}


def _delta(baseline, candidate, name, policy):
    absolute = candidate - baseline
    relative = absolute / baseline if baseline > 0 else None
    if not math.isfinite(absolute) or (relative is not None and not math.isfinite(relative)):
        raise ValueError('Metric delta overflow')
    tolerance = policy['zero_baseline_absolute_tolerances'].get(name)
    if relative is not None:
        screen = 'within_limit' if relative <= policy['relative_regression_limit'] else 'regression'
    elif tolerance is not None:
        screen = 'within_limit' if absolute <= tolerance['value'] else 'regression'
    else:
        screen = 'unassessed'
    return {'absolute_delta': absolute, 'relative_delta': relative, 'screen': screen,
            'relative_reason': 'zero baseline; no relative percentage' if relative is None else None}


def _summary(rows, name, policy):
    valid = [row for row in rows if row['status'] == 'assessed']
    coverage = {'expected': len(rows), 'assessed': len(valid), 'unassessed': len(rows) - len(valid)}
    result = {'units': METRICS[name], 'coverage': coverage, 'repetitions': rows,
              'status': 'assessed' if len(valid) == len(rows) and valid else 'partial' if valid else 'unassessed',
              'baseline_median': None, 'candidate_median': None, 'absolute_delta': None,
              'relative_delta': None, 'screen': 'unassessed', 'pooled': None}
    definitions = {_json(row.get('definition')) for row in valid}
    if len(definitions) > 1:
        result.update(status='unassessed', reason='Metric definitions differ across contributors; separate their categories', definition=None)
        return result
    result['definition'] = valid[0].get('definition') if valid else None
    if valid:
        baseline = statistics.median(row['baseline'] for row in valid)
        candidate = statistics.median(row['candidate'] for row in valid)
        population = sum(row['population'] for row in valid)
        result.update(baseline_median=baseline, candidate_median=candidate,
                      **_delta(baseline, candidate, name, policy),
                      pooled={'baseline': math.fsum(row['baseline_sum'] for row in valid) / population,
                              'candidate': math.fsum(row['candidate_sum'] for row in valid) / population,
                              'population': valid[0]['population'], 'repetitions': len(valid)})
        if result['status'] != 'assessed' and result['screen'] == 'within_limit':
            result['screen'] = 'unassessed'
    return result


def _coverage(cases):
    return {'expected_cases': len(cases),
            'assessed_cases': sum(case['status'] == 'assessed' for case in cases),
            'partial_cases': sum(case['status'] == 'partial' for case in cases),
            'unassessed_cases': sum(case['status'] == 'unassessed' for case in cases),
            'required_cases': sum(case['required'] for case in cases),
            'required_assessed_cases': sum(case['required'] and case['status'] == 'assessed' for case in cases)}


def _measurement_contract(value):
    """Keep sampling/geometry/definitions, excluding observed prediction errors."""
    if isinstance(value, list):
        return [_measurement_contract(item) for item in value]
    if not isinstance(value, dict):
        return value
    measured = {'value', 'population', 'units', 'status'} <= set(value)
    omitted = {'value', 'error_sum'} if measured else set()
    omitted |= {'candidate_rgb_range', 'candidate_undefined_foreground_pixels'}
    return {key: _measurement_contract(item) for key, item in value.items()
            if key not in omitted and not (key == 'temporal_residual_mae' and not isinstance(item, dict))}


def _repetition_contract(report):
    return {
        'kind': report.get('kind'), 'frame_ids': report.get('frame_ids'), 'fps': report.get('fps'),
        'source': report.get('source', {}).get('sha256'), 'assets': _asset_contract(report),
        'decode_contract': report.get('decode_contract'),
        'reference_provenance': report.get('reference_provenance'),
        'flow_provenance': report.get('flow_provenance'),
        'evaluation_contract': _relevant(report.get('evaluation_contract')),
        'binding': {key: value for key, value in report.get('candidate_binding', {}).items()
                    if key not in ('job_id', 'output_dir')},
        'measurement_contract': _measurement_contract(report.get('candidate')),
    }


def _case(mapping, report, policy, check):
    pairs = report.get('pairs', [])
    if len(pairs) > MAX_REPETITIONS:
        raise ValueError('Paired repetition bound exceeded')
    metrics = {name: [] for name in METRICS}
    selected = []
    prior = None
    family = None
    for index, pair in enumerate(pairs):
        _check(check)
        quality = {}
        reason = None
        try:
            if report.get('status') != 'complete' or report.get('summary', {}).get('success') is not True:
                raise ValueError('Paired experiment is incomplete/failed/cancelled')
            samples = pair['samples']
            if _sample_identity(samples['baseline']) != _sample_identity(samples['candidate']):
                raise ValueError('Hardware/repository/assets/weights/corrections differ')
            quality = {side: _quality(samples[side], mapping['job_index']) for side in ('baseline', 'candidate')}
            family = quality['baseline'].get('kind')
            signature = {'identity': _sample_identity(samples['baseline']),
                         'reports': {side: _repetition_contract(q) for side, q in quality.items()}}
            if prior is not None and signature != prior:
                raise ValueError('Repeated case conditions/reference/frame contracts changed')
            prior = signature
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            reason = str(exc)[:256]
        for name, rows in metrics.items():
            row = {'pair_index': index, 'status': 'unassessed', 'reason': reason}
            if reason is None:
                try:
                    baseline, candidate = _compatibility(quality['baseline'], quality['candidate'], name)
                    row.update(status='assessed', reason=None, baseline=baseline['value'], candidate=candidate['value'],
                        population=baseline['population'], baseline_sum=baseline['error_sum'], candidate_sum=candidate['error_sum'],
                        definition=_definition(quality['baseline'], name),
                        **_delta(baseline['value'], candidate['value'], name, policy))
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    row['reason'] = str(exc)[:256]
            rows.append(row)
        if index == mapping.get('review_pair_index', 0) and quality:
            selected = [(side, quality[side]) for side in ('baseline', 'candidate')]
    if not pairs:
        for rows in metrics.values():
            rows.append({'pair_index': None, 'status': 'unassessed', 'reason': 'No completed paired observations'})
    summaries = {name: _summary(rows, name, policy) for name, rows in metrics.items()}
    any_valid = any(metric['coverage']['assessed'] for metric in summaries.values())
    # A case can carry only one report family. Do not require other families to exist.
    available_family = family
    expected = [name for name in METRICS if name.startswith('alpha.') or
                (name.startswith('colour.') and available_family == 'colour_sequence_evaluation') or
                (name.startswith('temporal.') and available_family == 'temporal_sequence_evaluation')]
    complete = bool(expected) and all(summaries[name]['status'] == 'assessed' for name in expected)
    result = {**mapping, 'required': mapping.get('required', True), 'review_pair_index': mapping.get('review_pair_index', 0),
              'status': 'assessed' if complete else 'partial' if any_valid else 'unassessed', 'metrics': summaries,
              'review_selection_status': 'available' if selected else 'unavailable',
              'review_selection_reason': None if selected else 'Selected paired observation lacks both quality reports'}
    return result, selected


def _categories(cases, policy):
    groups = {}
    for case in cases:
        for category in case['categories']:
            groups.setdefault((category, case['split']), []).append(case)
    result = []
    encoded_bytes = 0
    for (category, split), members in sorted(groups.items()):
        metrics = {}
        for name, units in METRICS.items():
            rows = []
            observations = []
            for case in members:
                metric = case['metrics'][name]
                rows.append({'case_id': case['id'], 'status': 'assessed' if metric['status'] == 'assessed' else 'unassessed',
                    'baseline': metric['baseline_median'], 'candidate': metric['candidate_median'],
                    'population': 1, 'baseline_sum': metric['baseline_median'], 'candidate_sum': metric['candidate_median'],
                    'definition': metric.get('definition')})
                observations.extend({'case_id': case['id'], **row} for row in metric['repetitions'] if row['status'] == 'assessed')
            summary = _summary(rows, name, policy)
            summary.pop('repetitions')
            valid = [row for row in rows if row['status'] == 'assessed']
            paired = [{'case_id': row['case_id'], **_delta(row['baseline'], row['candidate'], name, policy)} for row in valid]
            relative = [row for row in paired if row['relative_delta'] is not None]
            observed = [row for row in observations if row['relative_delta'] is not None]
            summary['worst_case_median'] = max(relative, key=lambda row: row['relative_delta']) if relative else None
            summary['worst_observed'] = max(observed, key=lambda row: row['relative_delta']) if observed else None
            summary['zero_baseline_cases'] = [row for row in paired if row['relative_delta'] is None]
            worst_candidate = max(observations, key=lambda row: row['candidate']) if observations else None
            summary['worst_candidate_error'] = ({'case_id': worst_candidate['case_id'],
                'pair_index': worst_candidate['pair_index'], 'value': worst_candidate['candidate'],
                'baseline': worst_candidate['baseline'], 'units': units} if worst_candidate else None)
            # Pooled pixels are descriptive; each case contributes once after repetitions.
            pools = [case['metrics'][name]['pooled'] for case in members if case['metrics'][name]['status'] == 'assessed' and not summary.get('reason')]
            population = sum(pool['population'] for pool in pools)
            summary['pooled'] = ({'baseline': math.fsum(pool['baseline'] * pool['population'] for pool in pools) / population,
                                  'candidate': math.fsum(pool['candidate'] * pool['population'] for pool in pools) / population,
                                  'population': population, 'scope': 'descriptive pooled evaluated pixels/pairs/channels across repetitions'} if population else None)
            if any(row['screen'] == 'regression' for row in paired + observations):
                summary['screen'] = 'regression'
            metrics[name] = summary
        category_row = {'category': category, 'split': split, 'coverage': _coverage(members), 'metrics': metrics}
        encoded_bytes += len(_json(category_row))
        if encoded_bytes > MAX_OUTPUT_BYTES // 4:
            raise ValueError('Dossier category detail budget exceeded')
        result.append(category_row)
    return result


def _media_and_paths(case, selected, protected, *, collect_media=True):
    media = []
    for side, report in selected:
        source_path = report.get('source', {}).get('path')
        if isinstance(source_path, str):
            protected.add(str(Path(source_path).resolve()))
        binding = report.get('candidate_binding', {})
        if isinstance(binding.get('output_dir'), str):
            protected.add(str(Path(binding['output_dir']).resolve()))
        for asset in report.get('assets', []):
            if not isinstance(asset, dict) or not isinstance(asset.get('path'), str):
                continue
            path = Path(asset['path']).resolve()
            protected.add(str(path))
            if not str(asset.get('role', '')).startswith('candidate'):
                continue
            protected.add(str(path.parent))
            if not collect_media:
                continue
            try:
                size = path.stat().st_size if path.is_file() else None
            except OSError:
                size = None
            media.append({'case_id': case['id'], 'side': side, 'frame': asset.get('frame'), 'role': asset['role'],
                'path': str(path), 'sha256': asset.get('sha256'), 'bytes': size, 'fps': report.get('fps'),
                'view_contract': report.get('evaluation_contract', {'kind': report.get('kind'), 'decode_contract': report.get('decode_contract')})})
    return media


def _build(path, check, progress):
    path = Path(path).resolve()
    before = _identity(path)
    manifest, raw = _manifest(path)
    captured = {'manifest': {'path': str(path), 'sha256': _sha(raw), 'bytes': len(raw), 'identity': before}, 'reports': []}
    protected = {str(path)}
    cases = []
    review_media = []
    total = 0
    source_paths = set()
    source_hashes = set()
    detail_bytes = 0
    for index, entry in enumerate(manifest['reports']):
        _check(check)
        source = (path.parent / entry['path']).resolve()
        if str(source) in source_paths:
            raise ValueError('Duplicate report path mapping')
        source_paths.add(str(source))
        identity = _identity(source)
        if identity[2] > MAX_REPORT_BYTES or total + identity[2] > MAX_INPUT_BYTES:
            raise ValueError('Dossier input report byte budget exceeded')
        report, report_raw = read_report(source, regular_file=True)
        if report['kind'] != 'inference_pair':
            raise ValueError('Dossier accepts selected existing inference_pair reports only')
        total += len(report_raw)
        if total > MAX_INPUT_BYTES or _identity(source) != identity:
            raise ValueError('Input report grew/changed during dossier read')
        if _sha(report_raw) in source_hashes:
            raise ValueError('Duplicate report bytes cannot count as independent evidence')
        source_hashes.add(_sha(report_raw))
        captured['reports'].append({'id': entry['id'], 'path': str(source), 'sha256': _sha(report_raw),
                                    'bytes': len(report_raw), 'identity': identity})
        protected.add(str(source))
        for pair in report.get('pairs', []):
            _check(check)
            for sample in pair.get('samples', {}).values():
                for run in sample.get('runs', []):
                    for job in run.get('jobs', []):
                        for field in ('input', 'output_dir'):
                            if isinstance(job.get(field), str):
                                protected.add(str(Path(job[field]).resolve()))
                for quality in sample.get('inference', {}).get('quality', {}).get('jobs', []):
                    evidence = quality.get('report')
                    if isinstance(evidence, dict):
                        _media_and_paths({'id': ''}, [('baseline', evidence)], protected, collect_media=False)
        for mapping in manifest['cases']:
            if mapping['report_id'] == entry['id']:
                case, selected = _case(mapping, report, manifest['policy'], check)
                media = _media_and_paths(case, selected, protected)
                detail_bytes += len(_json(case)) + len(_json(media))
                if detail_bytes > MAX_OUTPUT_BYTES // 2:
                    raise ValueError('Dossier case/media metadata budget exceeded')
                cases.append(case)
                review_media.extend(media)
        if progress is not None:
            progress(ProgressEvent('dossier_reports', index + 1, len(manifest['reports'])))
        del report, report_raw
    if _identity(path) != before:
        raise ValueError('Dossier manifest changed during build')
    # Recheck hashes with bounded reads; no media is decoded, hashed or copied.
    for item in [captured['manifest'], *captured['reports']]:
        _check(check)
        observed = read_regular_bytes(Path(item['path']), max_bytes=item['bytes'])
        if len(observed) != item['bytes'] or _sha(observed) != item['sha256'] or _identity(Path(item['path'])) != item['identity']:
            raise ValueError('Dossier input changed before completion')
    categories = _categories(cases, manifest['policy'])
    result = {'schema_version': 1, 'kind': 'evaluation_dossier', 'inputs': captured, 'policy': manifest['policy'],
        'cases': sorted(cases, key=lambda item: item['id']), 'categories': categories, 'coverage': _coverage(cases),
        'protected_paths': sorted(protected), 'review_media': review_media, 'review_status': 'pending',
        'acceptance_verdict': None,
        'scope': 'Paired report-derived regression screens only; no model run, media copy, blinded review, professional quality or GPU acceptance. Categories/splits/policy are explicit user assertions; predeclaration is not independently verified.',
        'aggregation': 'paired repetitions → case median errors → equal-case category medians; pooled population errors are descriptive, raw SAD and cross-unit combined scores excluded',
        'review_selection': 'Explicit review_pair_index; default0 selects both timed sides, not an external template baseline.'}
    result['token'] = _sha(_json(result))
    if len(_json(result)) > MAX_OUTPUT_BYTES:
        raise ValueError('Dossier report exceeds16MiB output bound')
    return result


def build_dossier(manifest_path, *, check=None, progress=None):
    try:
        _check(check)
        report = _build(manifest_path, check, progress)
        return {'status': 'complete', 'report': report, 'error': None}
    except (CancelledError, KeyboardInterrupt) as exc:
        return {'status': 'cancelled', 'report': None, 'error': str(exc)[:512]}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        return {'status': 'failed', 'report': None, 'error': f'{type(exc).__name__}: {exc}'[:512]}


def revalidate_inputs(report, *, check=None):
    if not isinstance(report, dict) or not isinstance(report.get('token'), str):
        raise TypeError('A complete immutable dossier report is required')
    payload = copy.deepcopy(report)
    token = payload.pop('token')
    if _sha(_json(payload)) != token:
        raise ValueError('Dossier report token differs from contents')
    rebuilt = _build(report['inputs']['manifest']['path'], check, None)
    if rebuilt['token'] != token:
        raise ValueError('Dossier inputs or semantic report changed; rebuild before export')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('build', 'export'):
        child = commands.add_parser(command)
        child.add_argument('--manifest', type=Path, required=True)
        if command == 'export':
            child.add_argument('--output-parent', type=Path, required=True)
            child.add_argument('--include-review-media', action='store_true')
            child.add_argument('--max-media-bytes', type=int, default=0)
    args = parser.parse_args(argv)
    terminal = build_dossier(args.manifest)
    if args.command == 'export' and terminal['status'] == 'complete':
        from .evaluation_dossier_export import export_dossier
        terminal = export_dossier(terminal['report'], args.output_parent,
                                  include_media=args.include_review_media, max_media_bytes=args.max_media_bytes)
    print(json.dumps(terminal, allow_nan=False))
    return 0 if terminal['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
