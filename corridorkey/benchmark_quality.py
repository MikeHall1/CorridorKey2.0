"""Bind reference manifests to completed native alpha exports, outside timed runs."""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from .evaluation_colour_media import evaluate_colour_spec, load_colour_manifest
from .evaluation_media import evaluate_sequence_spec, load_sequence_manifest
from .evaluation_temporal_media import evaluate_temporal_spec, load_temporal_manifest
from .identities import file_sha256, source_identity


def prepare_quality_templates(paths, jobs, *, manifest_dir):
    """Validate explicit per-job requests before loading models; None means no reference request."""
    if paths is None:
        return None
    if not isinstance(paths, list) or len(paths) != len(jobs):
        raise ValueError('quality_manifests must have one path or null per batch job')
    templates = []
    for path, job in zip(paths, jobs, strict=True):
        if path is None:
            templates.append(None)
            continue
        if not isinstance(path, str) or not path.strip():
            raise ValueError('quality manifest entries must be paths or null')
        config, _ = job.configurations()
        if config.output_format != 'exr' or 'alpha' not in config.outputs:
            raise ValueError('reference evaluation requires explicit EXR alpha output; no video fallback')
        resolved = (Path(manifest_dir) / path).resolve()
        header = json.loads(resolved.read_bytes())
        colour = isinstance(header, dict) and header.get('kind') == 'colour_sequence_evaluation'
        temporal = isinstance(header, dict) and header.get('kind') == 'temporal_sequence_evaluation'
        loader = load_colour_manifest if colour else load_temporal_manifest if temporal else load_sequence_manifest
        data, digest = loader(resolved)
        if 'source' not in data or Path(data['source']).resolve() != Path(job.input_path).resolve():
            raise ValueError('quality manifest source must match the batch job input')
        channel = (data.get('alpha_channels', {}).get('candidate_alpha', 'gray') if colour else
                   data.get('channels', {}).get('candidate', 'gray'))
        if channel != 'gray':
            raise ValueError('native EXR alpha requires candidate channel=gray')
        if colour:
            output = data.get('candidate_output', 'fg')
            association = config.cutout_alpha_mode if output == 'cutout' else 'straight'
            layout = 'rgba' if output == 'cutout' else 'rgb'
            if (output not in config.outputs or data['associations']['candidate'] != association or
                    data.get('rgb_layouts', {}).get('candidate_rgb', 'rgb') != layout):
                raise ValueError('colour template must match requested RGB output/association/layout')
        asset_hashes = {}
        for frame in data['frames']:
            for role in set(frame) - {'frame', 'candidate', 'candidate_rgb', 'candidate_alpha', 'cut', 'flow_previous_frame'}:
                if not Path(frame[role]).is_file():
                    raise FileNotFoundError(frame[role])
                if frame[role] not in asset_hashes:
                    asset_hashes[frame[role]] = file_sha256(frame[role])
        templates.append({'path': str(resolved), 'sha256': digest, 'data': data,
                          'source': source_identity(job.input_path), 'asset_hashes': asset_hashes})
    return templates


def _export_files(directory, output_dir, count, prefix, *, check=None):
    directory = Path(directory).resolve()
    if not directory.is_dir() or not directory.is_relative_to(output_dir):
        raise ValueError('export must be inside the completed job output directory')
    files = {}
    for path in directory.iterdir():
        if check is not None:
            check()
        match = re.fullmatch(re.escape(prefix) + r'(?:_source_rgb_draft)?(?:_spatial_draft)?_(\d{6,})\.exr', path.name)
        if not path.is_file() or match is None:
            raise ValueError('unexpected file in export directory')
        index = int(match.group(1))
        if index in files:
            raise ValueError('duplicate export index')
        if not path.resolve().is_relative_to(directory):
            raise ValueError('export links must stay inside its output directory')
        files[index] = path
    if type(count) is not int or count <= 0 or len(files) != count or set(files) != set(range(count)):
        raise ValueError('exports must contain every submitted output frame exactly once')
    return files


def _working_primaries_match(contract):
    if contract.get('rgb_primaries') == 'assumed sRGB/BT.709; no automatic colour management':
        return 'source_colour_contract' not in contract
    if contract.get('rgb_primaries') != 'BT.709/D65 working output':
        return False
    declared = contract.get('source_colour_contract')
    if not isinstance(declared, dict) or declared.get('profile') not in ('linear_bt709_d65', 'linear_bt2020_d65'):
        return False
    from .colour_contract import describe_source_colour
    return declared == describe_source_colour(declared['profile'], scene_linear=True)


def _bind_rgb(data, diagnostics, output_dir, count, *, check=None):
    output = data.get('candidate_output', 'fg')
    contract = diagnostics['output_contracts'][output]
    association = contract.get('alpha_association') if output == 'cutout' else 'straight'
    if (contract.get('format') != 'exr' or contract.get('channels') != (4 if output == 'cutout' else 3) or
            contract.get('rgb_representation') != 'scene_linear' or
            not _working_primaries_match(contract) or
            association != data['associations']['candidate']):
        raise ValueError('colour export contract/association/primaries does not match reference declaration')
    prefix = output + ('_premultiplied' if output == 'cutout' and association == 'premultiplied' else '')
    files = _export_files(diagnostics['outputs'][output], output_dir, count, prefix, check=check)
    for frame in data['frames']:
        frame['candidate_rgb'] = str(files[frame['frame']])


def _check_temporal_boundaries(data, record):
    diagnostics = record['diagnostics']
    settings = record.get('requested_settings') or {}
    quick = diagnostics.get('quick_key') or {}
    corrections = diagnostics.get('corrections') or {}
    boundaries = (set(settings.get('quick_cut_frames', ())) | set(settings.get('quick_reset_frames', ())) |
                  set(quick.get('effective_cuts', ())) | set(corrections.get('reset_frames', ())) |
                  {shot[0] for shot in corrections.get('shots', ()) if shot[0] > 0})
    previous = None
    for frame in data['frames']:
        if previous is not None and not frame.get('cut', False) and any(previous < cut <= frame['frame'] for cut in boundaries):
            raise ValueError('temporal reference pair crosses known shot cut/reset; declare cut=true')
        previous = frame['frame']
    return sorted(boundaries)


def evaluate_job_quality(template, record, *, check=None):
    """Only completed native coverage exports with known geometry/timing can bind frames."""
    if check is not None:
        check()
    if record.get('status') != 'complete':
        raise ValueError('quality requires a completed inference job')
    data = copy.deepcopy(template['data'])
    if Path(record['input']).resolve() != Path(data['source']).resolve():
        raise ValueError('quality source does not match completed job')
    diagnostics = record['diagnostics']
    contract = diagnostics['output_contracts']['alpha']
    if (contract.get('format') != 'exr' or contract.get('channels') != 1 or
            contract.get('alpha_representation') != 'coverage (no transfer function)'):
        raise ValueError('quality requires a declared single-channel coverage EXR export')
    source = diagnostics['source']
    fps = source['native_fps']
    if not isinstance(fps, (int, float)) or not math.isclose(fps, data['fps'], rel_tol=1e-9):
        raise ValueError('quality manifest FPS differs from actual output FPS')
    output_dir = Path(record['output_dir']).resolve()
    # Full-rate, Quick Key and spatial draft names all have contiguous
    # output-relative indices. Bind suffix IDs, never lexical ordering or source offsets.
    count = diagnostics['frames_submitted']
    files = _export_files(diagnostics['outputs']['alpha'], output_dir, count, 'alpha', check=check)
    colour = data['kind'] == 'colour_sequence_evaluation'
    temporal = data['kind'] == 'temporal_sequence_evaluation'
    for entry in data['frames']:
        if entry['frame'] not in files:
            raise ValueError('reference frame ID is outside the completed output timeline')
        entry['candidate_alpha' if colour else 'candidate'] = str(files[entry['frame']])
    if colour:
        _bind_rgb(data, diagnostics, output_dir, count, check=check)
    boundaries = _check_temporal_boundaries(data, record) if temporal else None
    before = source_identity(data['source'], check=check)
    if before['sha256'] != template['source']['sha256']:
        raise ValueError('source changed since quality preflight')
    for path, expected in template['asset_hashes'].items():
        if file_sha256(path, check=check) != expected:
            raise ValueError('reference/baseline/mask changed since quality preflight')
    evaluate = evaluate_colour_spec if colour else evaluate_temporal_spec if temporal else evaluate_sequence_spec
    report = evaluate(data, manifest_path=template['path'], manifest_sha256=template['sha256'], check=check)
    for asset in report['assets']:
        if not asset['role'].startswith('candidate') and asset['sha256'] != template['asset_hashes'][asset['path']]:
            raise ValueError('reference/baseline/mask changed during quality evaluation')
    if report['source']['sha256'] != before['sha256']:
        raise ValueError('source changed during quality evaluation')
    if tuple(report['candidate']['shape_hw']) != tuple(source['output_hw']):
        raise ValueError('decoded export dimensions differ from job output geometry')
    report['candidate_binding'] = {'method': 'completed native EXR exports, output-relative frame IDs',
                                   'rgb_output': data.get('candidate_output', 'fg') if colour else None,
                                   'known_shot_boundaries': boundaries,
                                   'job_id': record['job_id'], 'output_dir': str(output_dir),
                                   'source_start_frame': source.get('start_frame'),
                                   'frames_submitted': count, 'selected_frames': len(data['frames'])}
    if 'spatial_draft' in contract:
        report['candidate_binding']['spatial_draft'] = copy.deepcopy(contract['spatial_draft'])
    return report


def evaluate_run_quality(templates, runs):
    """Quality errors remain distinct from measured inference status/timings."""
    reports = []
    for run in runs:
        jobs = run['jobs']
        if len(jobs) != len(templates):
            raise ValueError('completed job count differs from quality template count')
        for index, (template, job) in enumerate(zip(templates, jobs, strict=True)):
            entry = {'run': run['index'], 'job': index}
            if template is None:
                entry.update(status='unavailable', reason='no quality manifest supplied', report=None)
            else:
                try:
                    report = evaluate_job_quality(template, job)
                    entry.update(status=report['candidate']['status'],
                                 comparison_status=report['comparison_status'], report=report)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    entry.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)}, report=None)
            reports.append(entry)
    failed = any(entry['status'] == 'failed' for entry in reports)
    measured = any(entry['status'] in ('measured', 'partial') for entry in reports)
    unavailable = any(entry['status'] in ('unavailable', 'partial') for entry in reports)
    return {'status': 'failed' if failed else ('partial' if measured and unavailable else
                                              'measured' if measured else 'unavailable'),
            'success': not failed, 'jobs': reports,
            'timing_scope': 'After all timed inference repetitions; reference decode/hash/metrics excluded.'}
