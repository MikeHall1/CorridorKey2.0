"""Local streaming RGB/composite reference manifests with explicit linear contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .evaluation_colour import COLOUR_SPACE
from .evaluation_colour_sequence import compare_colour_stream
from .evaluation_media import PROVENANCE, _unique_object, read_alpha_frame
from .identities import file_sha256, source_identity

RGB_ROLES = {'candidate_rgb', 'reference_rgb', 'baseline_rgb', 'background'}
ALPHA_ROLES = {'candidate_alpha', 'reference_alpha', 'baseline_alpha'}
MASK_ROLES = {'foreground_mask', 'composite_mask'}
ROLES = RGB_ROLES | ALPHA_ROLES | MASK_ROLES


def load_colour_manifest(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_unique_object)
    allowed = {'schema_version', 'kind', 'colour_space', 'associations', 'fps', 'frames', 'background',
               'rgb_layouts', 'alpha_channels', 'reference_provenance', 'source', 'unpremultiply_epsilon',
               'candidate_output'}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError('invalid or unknown colour manifest fields')
    if (type(data.get('schema_version')) is not int or data['schema_version'] != 1 or
            data.get('kind') != 'colour_sequence_evaluation' or data.get('colour_space') != COLOUR_SPACE):
        raise ValueError('expected colour_sequence_evaluation v1 with explicit linear_srgb_bt709')
    fps = data.get('fps')
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        raise TypeError('fps must be numeric and not boolean')
    try:
        finite = math.isfinite(fps)
    except OverflowError as exc:
        raise ValueError('fps is outside finite range') from exc
    if not finite or fps <= 0:
        raise ValueError('fps must be finite and positive')
    epsilon = data.get('unpremultiply_epsilon', 1e-6)
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise TypeError('unpremultiply_epsilon must be numeric')
    if not 0 <= epsilon < 1:
        raise ValueError('unpremultiply_epsilon must be finite in [0,1)')
    if data.get('candidate_output', 'fg') not in ('fg', 'cutout'):
        raise ValueError('candidate_output must be fg or cutout')
    frames = data.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('frames must be a nonempty explicit list')
    roles = None
    previous = -1
    for entry in frames:
        if not isinstance(entry, dict) or set(entry) - {'frame', *ROLES}:
            raise ValueError('invalid colour frame fields')
        index = entry.get('frame')
        if type(index) is not int or index <= previous:
            raise ValueError('frame IDs must be increasing nonnegative integers')
        previous = index
        current = set(entry) - {'frame'}
        if not {'candidate_rgb', 'candidate_alpha'} <= current or (roles is not None and roles != current):
            raise ValueError('candidate RGB/alpha required; roles must remain constant')
        roles = current
        for prefix in ('reference', 'baseline'):
            if (prefix + '_rgb' in current) != (prefix + '_alpha' in current):
                raise ValueError('RGB/alpha reference and baseline roles must occur in pairs')
        for role in current:
            if not isinstance(entry[role], str) or not entry[role].strip():
                raise ValueError('frame assets must be nonempty path strings')
            entry[role] = str((path.parent / entry[role]).resolve())
    associations = data.get('associations')
    expected = {'candidate', 'reference'} | ({'baseline'} if 'baseline_rgb' in (roles or ()) else set())
    if (not isinstance(associations, dict) or set(associations) != expected or
            any(v not in ('straight', 'premultiplied') for v in associations.values())):
        raise ValueError('declare candidate/reference and optional baseline associations')
    layouts, channels = data.get('rgb_layouts', {}), data.get('alpha_channels', {})
    if (not isinstance(layouts, dict) or set(layouts) - RGB_ROLES or
            any(v not in ('rgb', 'rgba') for v in layouts.values()) or
            not isinstance(channels, dict) or set(channels) - ALPHA_ROLES or
            any(v not in ('gray', 'alpha', 'replicated_gray') for v in channels.values())):
        raise ValueError('invalid explicit RGB layouts/alpha channels')
    provenance = data.get('reference_provenance')
    if 'reference_rgb' in (roles or ()):
        if (not isinstance(provenance, dict) or set(provenance) != PROVENANCE or
                any(not isinstance(v, str) or not v.strip() for v in provenance.values()) or
                provenance['split'] not in ('synthetic', 'development', 'held_out')):
            raise ValueError('references require declared author/origin/permitted_use/split')
    elif provenance is not None:
        raise ValueError('reference provenance requires reference assets')
    if 'background' in data:
        bg = np.asarray(data['background'])
        if (bg.shape != (3,) or bg.dtype.kind not in 'uif' or not np.isfinite(bg).all() or
                'background' in (roles or ())):
            raise ValueError('background must be a finite constant RGB triplet or per-frame asset, not both')
    if 'source' in data:
        if not isinstance(data['source'], str) or not data['source'].strip():
            raise ValueError('source must be a nonempty path')
        data['source'] = str((path.parent / data['source']).resolve())
    return data, hashlib.sha256(raw).hexdigest()


def read_rgb_frame(path, *, layout='rgb', check=None):
    path = Path(path)
    before = file_sha256(path, check=check)
    if path.suffix.lower() == '.npy':
        with path.open('rb') as stream:
            value = np.lib.format.read_array(stream, allow_pickle=False)
    elif path.suffix.lower() in ('.png', '.exr'):
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
        import cv2

        try:
            value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except cv2.error as exc:
            raise ValueError(f'RGB decoder failed for {path}: {exc}') from exc
        if value is None:
            raise ValueError(f'cannot decode RGB frame: {path}')
        if value.dtype == np.uint8:
            value = value.astype(np.float64) / 255
        elif value.dtype == np.uint16:
            value = value.astype(np.float64) / 65535
        if value.ndim == 3 and value.shape[-1] in (3, 4):
            value = value[..., [2, 1, 0, 3] if value.shape[-1] == 4 else [2, 1, 0]]
    else:
        raise ValueError('RGB frames must be .npy/.png/.exr; no video/lossy/colour transform fallback')
    expected = 3 if layout == 'rgb' else 4 if layout == 'rgba' else None
    if value.ndim != 3 or 0 in value.shape or value.shape[-1] != expected or value.dtype.kind not in 'uif':
        raise ValueError('RGB frame layout differs from declared rgb/rgba')
    rgb = value[..., :3]
    if not np.isfinite(rgb).all():
        raise ValueError('RGB frame contains nonfinite values')
    if before != file_sha256(path, check=check):
        raise ValueError('RGB frame changed during read')
    return rgb, {'path': str(path.resolve()), 'sha256': before, 'shape': list(rgb.shape),
                 'decoded_dtype': str(rgb.dtype), 'layout': layout, 'interpretation': COLOUR_SPACE}


def evaluate_colour_manifest(path, *, check=None):
    data, digest = load_colour_manifest(path)
    return evaluate_colour_spec(data, manifest_path=path, manifest_sha256=digest, check=check)


def evaluate_colour_spec(data, *, manifest_path, manifest_sha256, check=None):
    identities = []

    def frames():
        for entry in data['frames']:
            result = {}
            for role in sorted(set(entry) - {'frame'}):
                if role in RGB_ROLES:
                    value, identity = read_rgb_frame(entry[role], layout=data.get('rgb_layouts', {}).get(role, 'rgb'), check=check)
                else:
                    value, identity = read_alpha_frame(entry[role], channel=data.get('alpha_channels', {}).get(role, 'gray'),
                                                       mask=role in MASK_ROLES, check=check)
                identities.append({'frame': entry['frame'], 'role': role, **identity})
                result[role] = value
            if 'background' in data:
                result['background'] = data['background']
            yield result

    association = data['associations']
    report = compare_colour_stream(frames(), colour_space=data['colour_space'],
                                   candidate_association=association['candidate'], reference_association=association['reference'],
                                   baseline_association=association.get('baseline'),
                                   unpremultiply_epsilon=data.get('unpremultiply_epsilon', 1e-6), check=check)
    report.update(kind='colour_sequence_evaluation', manifest_sha256=manifest_sha256,
                  evaluation_contract={'schema_version': 1, 'family': 'colour',
                                       'colour_space': data['colour_space'],
                                       'associations': data['associations'],
                                       'rgb_layouts': {role: data.get('rgb_layouts', {}).get(role, 'rgb')
                                                       for role in RGB_ROLES if role in data['frames'][0]},
                                       'alpha_channels': {role: data.get('alpha_channels', {}).get(role, 'gray')
                                                          for role in ALPHA_ROLES | MASK_ROLES if role in data['frames'][0]},
                                       'unpremultiply_epsilon': data.get('unpremultiply_epsilon', 1e-6),
                                       'constant_background': data.get('background')},
                  manifest_path=str(Path(manifest_path).resolve()), assets=identities,
                  source=source_identity(data['source'], check=check) if 'source' in data else None,
                  frame_ids=[entry['frame'] for entry in data['frames']], fps=data['fps'],
                  reference_provenance=data.get('reference_provenance'),
                  decode_contract='PNG integer RGB divides by255/65535; EXR/NPY stay in declared linear units; '
                                  'RGB order explicit, alpha unchanged. No transfer/primaries transform or clipping.',
                  identity_scope='Stable individual reads, not globally atomic snapshots or encoded timing proof.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    data, _ = load_colour_manifest(args.manifest)
    protected = {Path(entry[role]).resolve() for entry in data['frames'] for role in ROLES if role in entry}
    protected.add(args.manifest.resolve())
    if 'source' in data:
        source = Path(data['source']).resolve()
        protected.add(source)
        if source.is_dir() and args.output.resolve().is_relative_to(source):
            parser.error('report must be outside the source frame directory')
    if args.output.resolve() in protected:
        parser.error('report must differ from all input assets/manifest')
    report = evaluate_colour_manifest(args.manifest)
    from .diagnostics import write_json_atomic

    write_json_atomic(args.output, report)
    print(f"colour sequence: {report['status']}; report: {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
