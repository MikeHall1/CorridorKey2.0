"""Explicit local alpha/flow/visibility sequences; no flow estimation or alignment inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation_media import _unique_object, read_alpha_frame, validate_sequence_data
from .evaluation_temporal import FLOW_CONVENTION, _flow
from .evaluation_temporal_sequence import compare_temporal_stream
from .identities import file_sha256, source_identity

ALPHA_ROLES = {'candidate', 'reference', 'baseline', 'mask'}
TRANSITION_ROLES = {'backward_flow', 'forward_flow', 'visibility'}
META = {'frame', 'cut', 'flow_previous_frame'}


def load_temporal_manifest(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_unique_object)
    allowed = {'schema_version', 'kind', 'fps', 'frames', 'channels', 'source', 'reference_provenance',
               'flow_convention', 'flow_provenance', 'consistency_pixels'}
    if not isinstance(data, dict) or set(data) - allowed or data.get('kind') != 'temporal_sequence_evaluation':
        raise ValueError('expected temporal_sequence_evaluation with known fields')
    if data.get('flow_convention') != FLOW_CONVENTION:
        raise ValueError('declare current_to_previous_xy_pixels flow convention')
    frames = data.get('frames')
    if not isinstance(frames, list) or not frames or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError('frames must be nonempty explicit objects')
    alpha_data = {key: data[key] for key in ('schema_version', 'fps', 'channels', 'source', 'reference_provenance') if key in data}
    alpha_data.update(kind='alpha_sequence_evaluation', frames=[{key: value for key, value in frame.items()
                                                               if key in ALPHA_ROLES | {'frame'}} for frame in frames])
    alpha_data = validate_sequence_data(alpha_data, path)
    if 'source' in alpha_data:
        data['source'] = alpha_data['source']
    previous = None
    has_flow = False
    for entry, alpha_entry in zip(frames, alpha_data['frames'], strict=True):
        if set(entry) - (ALPHA_ROLES | TRANSITION_ROLES | META):
            raise ValueError('unknown temporal frame fields')
        entry.update(alpha_entry)
        if type(entry.get('cut', False)) is not bool:
            raise TypeError('cut must be boolean')
        transitions = set(entry) & TRANSITION_ROLES
        if previous is None or entry.get('cut', False):
            if transitions or 'flow_previous_frame' in entry:
                raise ValueError('first/cut frame must not declare an unused transition')
        elif transitions:
            if 'backward_flow' not in entry:
                raise ValueError('transition assets require backward sampling flow')
            if type(entry.get('flow_previous_frame')) is not int or entry['flow_previous_frame'] != previous:
                raise ValueError('flow_previous_frame must identify the preceding selected frame')
        elif 'flow_previous_frame' in entry:
            raise ValueError('flow_previous_frame requires flow assets')
        for role in transitions:
            if not isinstance(entry[role], str) or not entry[role].strip():
                raise ValueError('transition paths must be nonempty strings')
            entry[role] = str((path.parent / entry[role]).resolve())
        has_flow |= 'backward_flow' in entry
        previous = entry['frame']
    threshold = data.get('consistency_pixels', 1.5)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError('consistency_pixels must be numeric')
    if not 0 <= threshold <= np.finfo(np.float64).max:
        raise ValueError('consistency_pixels must be finite and nonnegative')
    provenance = data.get('flow_provenance')
    fields = {'method', 'origin', 'visibility_definition', 'independent_of_candidate'}
    if has_flow:
        if (not isinstance(provenance, dict) or set(provenance) != fields or
                provenance['independent_of_candidate'] is not True or
                any(not isinstance(provenance[name], str) or not provenance[name].strip()
                    for name in fields - {'independent_of_candidate'})):
            raise ValueError('flow requires method/origin/visibility_definition and independent_of_candidate=true')
    elif provenance is not None:
        raise ValueError('flow provenance requires flow assets')
    return data, hashlib.sha256(raw).hexdigest()


def read_flow_frame(path, shape, *, check=None):
    path = Path(path)
    if path.suffix.lower() != '.npy':
        raise ValueError('flow must be native HxWx2 .npy pixel displacements; no estimator/resize fallback')
    before = file_sha256(path, check=check)
    with path.open('rb') as stream:
        value = np.lib.format.read_array(stream, allow_pickle=False)
    value = _flow(value, shape, 'flow asset')
    if before != file_sha256(path, check=check):
        raise ValueError('flow asset changed during read')
    return value, {'path': str(path.resolve()), 'sha256': before, 'shape': list(value.shape),
                   'decoded_dtype': str(value.dtype), 'units': 'native xy pixel displacement'}


def evaluate_temporal_manifest(path, *, check=None):
    data, digest = load_temporal_manifest(path)
    return evaluate_temporal_spec(data, manifest_path=path, manifest_sha256=digest, check=check)


def evaluate_temporal_spec(data, *, manifest_path, manifest_sha256, check=None):
    identities = []

    def frames():
        for entry in data['frames']:
            frame = {key: entry[key] for key in META if key in entry}
            shape = None
            for role in sorted(set(entry) & ALPHA_ROLES):
                value, identity = read_alpha_frame(entry[role], channel=data.get('channels', {}).get(role, 'gray'),
                                                   mask=role == 'mask', check=check)
                frame[role] = value
                if role == 'candidate':
                    shape = value.shape
                identities.append({'frame': entry['frame'], 'role': role, **identity})
            for role in sorted(set(entry) & TRANSITION_ROLES):
                if role == 'visibility':
                    value, identity = read_alpha_frame(entry[role], mask=True, check=check)
                else:
                    value, identity = read_flow_frame(entry[role], shape, check=check)
                frame[role] = value
                identities.append({'frame': entry['frame'], 'role': role, **identity})
            yield frame

    report = compare_temporal_stream(frames(), flow_convention=data['flow_convention'],
                                     consistency_pixels=data.get('consistency_pixels', 1.5), check=check)
    report.update(kind='temporal_sequence_evaluation', manifest_path=str(Path(manifest_path).resolve()),
                  evaluation_contract={'schema_version': 1, 'family': 'temporal',
                                       'channels': {role: data.get('channels', {}).get(role, 'gray')
                                                    for role in ALPHA_ROLES if role in data['frames'][0]},
                                       'flow_convention': data['flow_convention'],
                                       'consistency_pixels': data.get('consistency_pixels', 1.5)},
                  manifest_sha256=manifest_sha256, assets=identities, fps=data['fps'],
                  frame_ids=[entry['frame'] for entry in data['frames']], reference_provenance=data.get('reference_provenance'),
                  flow_provenance=data.get('flow_provenance'),
                  decode_contract='Alpha coverage normalized by255/65535 for integer rasters; floating alpha unchanged. '
                                  'Flow is declared current-to-previous XY pixels; no resize or timing inference.',
                  source=source_identity(data['source'], check=check) if 'source' in data else None,
                  identity_scope='Stable individual reads and declared correspondence/provenance; not independent flow/occlusion validation.',
                  timing_note='Declared selected frame IDs/FPS; residuals are per pair, not per-second or encoded/VFR proof.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    data, _ = load_temporal_manifest(args.manifest)
    protected = {Path(frame[role]).resolve() for frame in data['frames']
                 for role in ALPHA_ROLES | TRANSITION_ROLES if role in frame}
    protected.add(args.manifest.resolve())
    if 'source' in data:
        source = Path(data['source']).resolve()
        protected.add(source)
        if source.is_dir() and args.output.resolve().is_relative_to(source):
            parser.error('report must be outside the source frame directory')
    if args.output.resolve() in protected:
        parser.error('report must differ from every input asset/manifest')
    report = evaluate_temporal_manifest(args.manifest)
    from .diagnostics import write_json_atomic

    write_json_atomic(args.output, report)
    print(f"temporal sequence: {report['status']}; report: {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
