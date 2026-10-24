"""Captured completed-export agreement using the existing native quality evaluator."""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .benchmark_quality import (
    _export_files,
    _working_primaries_match,
    evaluate_job_quality,
)
from .benchmark_report import read_report
from .evaluation_dossier import METRICS, _compatibility
from .identities import sequence_identity
from .native_image_admission import MAX_PIXELS, admit_image
from .preview_sequence import scan_sequence
from .project_collection import (
    FilePin,
    admit_file,
    assert_pin,
    check_now,
    read_file,
    real_path,
)
from .scheduling import frame_path_key

METRIC_UNITS = {k: METRICS[k] for k in ('alpha.mae', 'colour.composite_mae')}
DEFAULT_BUDGETS = {'max_files': 4096, 'max_bytes': 256 * 1024**2, 'max_pixels': MAX_PIXELS}


def validate_budgets(value):
    if (not isinstance(value, dict) or set(value) != set(DEFAULT_BUDGETS)
            or any(type(value[k]) is not int or not 1 <= value[k] <= high for k, high in
                   (('max_files', 4096), ('max_bytes', 1024**3), ('max_pixels', MAX_PIXELS)))):
        raise ValueError('Numerical budgets require bounded exact max_files/max_bytes/max_pixels')


def _source_identity(path, assets, maximum, *, check=None):
    """Reuse charged source hashes; rescan only bounded membership/stat metadata.

    This removes NUM-owned duplicate content reads, not the existing evaluator's
    independent source/reference validation or a global decoder I/O sandbox.
    """
    check_now(check)
    source = real_path(path)
    if source.is_dir():
        expected = []
        for name in assets:
            check_now(check)
            candidate = Path(name)
            if candidate.parent == source and candidate.suffix.lower() in {'.png', '.jpg', '.jpeg', '.exr'}:
                expected.append(candidate)
        expected.sort(key=frame_path_key)
        actual = scan_sequence(source, check=check, max_entries=maximum, max_frames=maximum).paths
        if actual != tuple(str(p) for p in expected):
            raise ValueError('Numerical source sequence membership changed')
    else:
        expected = [source]
    files = []
    for item in expected:
        check_now(check)
        asset = assets.get(str(item))
        if asset is None:
            raise ValueError('Source is absent from admitted numerical assets')
        assert_pin(FilePin(asset['path'], tuple(asset['fingerprint']), asset['sha256']))
        files.append({'name': item.name, 'sha256': asset['sha256']})
    check_now(check)
    if source.is_dir():
        return sequence_identity(path, files)
    return {'path': str(Path(path)), 'sha256': files[0]['sha256']}


def snapshot(report_path, budgets, *, check=None):
    """Derive paths only from admitted actual job records; never from evidence assets."""
    from .regression_gate import context, digest

    validate_budgets(budgets)
    report_pin = admit_file(report_path)
    report, raw = read_report(Path(report_pin.path), regular_file=True)
    assert_pin(report_pin)
    if report.get('kind') != 'inference_pair' or report.get('status') != 'complete':
        raise ValueError('Numerical evidence requires a completed actual inference pair')
    matching = context(report)
    jobs = []
    paths = {report_pin.path: report_pin}
    roots = set()
    seen_jobs, seen_outputs = set(), set()
    for pair in report['pairs']:
        for index, (left, right) in enumerate(zip(pair['samples']['baseline']['runs'][0]['jobs'],
                                                pair['samples']['candidate']['runs'][0]['jobs'], strict=True)):
            if len(jobs) >= 256:
                raise ValueError('Numerical pair/job bound exceeded')
            sides = {}
            for side, record in (('baseline', left), ('candidate', right)):
                check_now(check)
                diagnostics = record['diagnostics']
                source = diagnostics.get('source')
                settings = record['requested_settings']
                count = diagnostics.get('frames_submitted')
                if (not record.get('job_id') or record.get('status') != 'complete'
                        or not isinstance(source, dict) or type(count) is not int or not 1 <= count <= 120
                        or source.get('num_frames') != count or source.get('start_frame') != (settings.get('start_frame') or 0)
                        or settings.get('output_format') != 'exr' or settings.get('exr_dtype') not in ('half', 'float')
                        or diagnostics.get('profiling_enabled') is not False):
                    raise ValueError('Missing completed source/range/FPS/dtype/unprofiled export binding')
                effective = diagnostics['effective_settings']
                expected_cache_dtype = 'float32' if effective.get('low_vram') or effective.get('foundation_cache_mb', 0) else None
                execution = diagnostics.get('execution')
                if (not isinstance(execution, dict) or 'foundation_cache_dtype' not in execution
                        or execution['foundation_cache_dtype'] != expected_cache_dtype):
                    raise ValueError('Foundation cache dtype differs from its effective per-side policy')
                fps = source.get('native_fps')
                requested_outputs = settings.get('outputs')
                if (type(source.get('num_frames')) is not int or type(source.get('start_frame')) is not int
                        or not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0
                        or not isinstance(requested_outputs, list) or not requested_outputs
                        or set(requested_outputs) != set(diagnostics.get('outputs', {}))
                        or set(requested_outputs) != set(diagnostics.get('output_contracts', {}))):
                    raise ValueError('Incomplete actual requested output/FPS/frame coverage binding')
                shape = source.get('output_hw')
                if (not isinstance(shape, list) or len(shape) != 2 or source.get('native_hw') != shape
                        or any(type(v) is not int or v <= 0 for v in shape) or shape[0] * shape[1] > budgets['max_pixels']):
                    raise ValueError('Unsupported native/output geometry or pixel budget')
                output = real_path(record['output_dir'], directory=True)
                input_path = real_path(record['input'])
                if record['job_id'] in seen_jobs or str(output) in seen_outputs:
                    raise ValueError('Repeated measurement reuses a job identity or output destination')
                seen_jobs.add(record['job_id'])
                seen_outputs.add(str(output))
                roots.add(str(output))
                if input_path.is_dir():
                    if source['native_fps'] != settings.get('frame_dir_fps'):
                        raise ValueError('Sequence FPS differs from the declared image-sequence rate')
                    roots.add(str(input_path))
                    inputs = scan_sequence(input_path, check=check, max_entries=budgets['max_files'], max_frames=budgets['max_files']).paths
                else:
                    inputs = [str(input_path)]
                for path in inputs:
                    pin = admit_file(path)
                    paths.setdefault(pin.path, pin)
                contracts = diagnostics.get('output_contracts')
                if not isinstance(contracts, dict) or 'alpha' not in contracts:
                    raise ValueError('Missing native alpha contract')
                kinds = ['alpha'] + (['fg'] if 'fg' in contracts else [])
                exports = {}
                for kind in kinds:
                    contract = contracts[kind]
                    if (contract.get('format') != 'exr' or contract.get('channels') != (1 if kind == 'alpha' else 3)
                            or kind == 'alpha' and contract.get('alpha_representation') != 'coverage (no transfer function)'
                            or kind == 'fg' and (contract.get('rgb_representation') != 'scene_linear'
                                                or not _working_primaries_match(contract)
                                                or contract.get('alpha_association') not in (None, 'straight'))):
                        raise ValueError('Unsupported native coverage/straight linear RGB export contract')
                    directory = real_path(diagnostics['outputs'][kind], directory=True)
                    if not directory.is_relative_to(output) or directory == output:
                        raise ValueError('Export escapes owned measured job output')
                    roots.add(str(directory))
                    with os.scandir(directory) as entries:
                        for number, entry in enumerate(entries, 1):
                            check_now(check)
                            if number > count:
                                raise ValueError('Export directory exceeds exact frame coverage bound')
                            admit_file(entry.path)
                    exports[kind] = {str(i): str(path) for i, path in _export_files(directory, output, count, kind).items()}
                    for path in exports[kind].values():
                        pin = admit_file(path)
                        paths.setdefault(pin.path, pin)
                if len(paths) > budgets['max_files']:
                    raise ValueError('Numerical file-count budget exceeded')
                sides[side] = {'job_id': record['job_id'], 'input': record['input'], 'output_dir': record['output_dir'],
                               'source': source, 'execution': diagnostics.get('execution'), 'contracts': contracts,
                               'exports': exports, 'dtype': settings['exr_dtype']}
            if any(sides['baseline'][k] != sides['candidate'][k] for k in ('input', 'source', 'contracts', 'dtype')):
                raise ValueError('Numerical source/range/geometry/colour/execution/dtype contracts differ')
            invariant_fields = ('backend', 'requested_device', 'gpu', 'device_index', 'cuda_version', 'autocast_dtype',
                                'rvm_cache_dtype', 'baton_dtype', 'cached_playback', 'source_colour', 'model_rgb', 'alpha',
                                'temporal_strategy', 'nonfinite_policy', 'rgb_primaries', 'source_colour_contract')
            executions = [sides[side]['execution'] for side in ('baseline', 'candidate')]
            if (any(not isinstance(e, dict) for e in executions)
                    or {k: executions[0].get(k) for k in invariant_fields} != {k: executions[1].get(k) for k in invariant_fields}):
                raise ValueError('Numerical backend/device/precision/semantic execution identity differs')
            jobs.append({'pair': pair['index'], 'job': index, 'sides': sides})
    # Conservative charge includes evaluator hashing/decoding and all final fences.
    required = sum(p.size for p in paths.values()) * (8 + 16 * len(jobs))
    if required > budgets['max_bytes']:
        raise ValueError('Numerical conservative read/evaluation byte budget exceeded')
    assets = []
    for pin in sorted(paths.values(), key=lambda p: p.path):
        sha, _ = read_file(pin, check=check)
        assets.append({**asdict(pin), 'fingerprint': list(pin.fingerprint), 'sha256': sha})
    by_path = {a['path']: a for a in assets}
    for row in jobs:
        for side in ('baseline', 'candidate'):
            binding = row['sides'][side]
            sample = report['pairs'][row['pair']]['samples'][side]
            if _source_identity(binding['input'], by_path, budgets['max_files'], check=check) != sample['assets'].get(binding['input']):
                raise ValueError('Actual source bytes differ from measured report asset identity')
            for kind, files in binding['exports'].items():
                for path in files.values():
                    admitted = admit_image(path, expected_hw=tuple(binding['source']['output_hw']),
                                           max_pixels=budgets['max_pixels'], check=check)
                    if (admitted.format != 'exr' or admitted.channels != (1 if kind == 'alpha' else 3)
                            or admitted.bit_depth != (16 if binding['dtype'] == 'half' else 32)
                            or admitted.sha256 != by_path[path]['sha256']):
                        raise ValueError('Actual native export header/bytes differ from declared contract')
    for pin in paths.values():
        assert_pin(pin)
    for path in {s['input'] for row in jobs for s in row['sides'].values()}:
        _source_identity(path, by_path, budgets['max_files'], check=check)
    return {'schema_version': 1, 'kind': 'regression_numerical_evidence', 'report_sha256': hashlib.sha256(raw).hexdigest(),
            'context_sha256': digest(matching), 'budgets': budgets, 'conservative_read_bytes': required,
            'jobs': jobs, 'assets': assets, 'protected_directories': sorted(roots),
            'scope': 'Local captured bytes/record bindings, not measurement authenticity or chronological approval.'}, report


def evaluate(evidence, report, constraints, *, check=None):
    """Baseline native exports are references; metrics remain existing evaluator definitions."""
    assets = {a['path']: a for a in evidence['assets']}
    by_path = {p: a['sha256'] for p, a in assets.items()}
    rows = []
    for binding in evidence['jobs']:
        check_now(check)
        baseline, candidate = (binding['sides'][s] for s in ('baseline', 'candidate'))
        selected = [c for c in constraints if c['job'] == binding['job']]
        if not selected:
            continue
        colour = any(c['metric'].startswith('colour.') for c in selected)
        if colour and any('fg' not in s['exports'] for s in (baseline, candidate)):
            raise ValueError('Black-composite agreement requires completed straight foreground exports')
        count = len(baseline['exports']['alpha'])
        frames = []
        for i in range(count):
            frame: dict[str, Any] = {'frame': i}
            if colour:
                frame.update(candidate_alpha='@completed', candidate_rgb='@completed',
                             reference_alpha=baseline['exports']['alpha'][str(i)], reference_rgb=baseline['exports']['fg'][str(i)])
            else:
                frame.update(candidate='@completed', reference=baseline['exports']['alpha'][str(i)])
            frames.append(frame)
        spec = {'schema_version': 1, 'kind': 'colour_sequence_evaluation' if colour else 'alpha_sequence_evaluation',
                'source': baseline['input'], 'fps': baseline['source']['native_fps'], 'frames': frames,
                'reference_provenance': {'author': 'local paired baseline', 'origin': 'completed measured exports',
                                         'permitted_use': 'local numerical agreement', 'split': 'development'}}
        if colour:
            spec.update(colour_space='linear_srgb_bt709', associations={'candidate': 'straight', 'reference': 'straight'},
                        background=[0, 0, 0], candidate_output='fg')
        references = {v for f in frames for k, v in f.items() if k.startswith('reference')}
        template = {'path': 'derived-from-captured-completed-exports', 'sha256': evidence['report_sha256'], 'data': spec,
                    'source': _source_identity(baseline['input'], assets, evidence['budgets']['max_files'], check=check), 'asset_hashes': {p: by_path[p] for p in references}}
        record = report['pairs'][binding['pair']]['samples']['candidate']['runs'][0]['jobs'][binding['job']]
        raw = evaluate_job_quality(template, record, check=check)
        check_now(check)
        for constraint in selected:
            metric, _ = _compatibility(raw, raw, constraint['metric'])
            rows.append({**constraint, 'pair': binding['pair'], 'value': metric['value'], 'population': metric['population'],
                         'status': 'pass' if metric['value'] <= constraint['maximum'] else 'fail',
                         'binding': raw['candidate_binding']})
    if len(rows) != len(report['pairs']) * len(constraints):
        raise ValueError('Missing numerical constraint job/repetition coverage')
    return rows


def fence(evidence, *, check=None):
    """Repeat lexical/inode/link checks immediately before publication."""
    for asset in evidence['assets']:
        check_now(check)
        assert_pin(FilePin(asset['path'], tuple(asset['fingerprint']), asset['sha256']))
    if evidence.get('kind') == 'regression_numerical_evidence':
        assets = {a['path']: a for a in evidence['assets']}
        for path in {s['input'] for row in evidence['jobs'] for s in row['sides'].values()}:
            _source_identity(path, assets, evidence['budgets']['max_files'], check=check)
    for directory in evidence['protected_directories']:
        real_path(directory, directory=True)
