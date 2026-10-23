"""Evaluate explicit aligned alpha-frame manifests locally, one frame at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .evaluation_sequence import compare_stream
from .identities import file_sha256, source_identity

ROLES = ('candidate', 'reference', 'baseline', 'mask')
PROVENANCE = {'author', 'origin', 'permitted_use', 'split'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate manifest key: {key}')
        result[key] = value
    return result


def load_sequence_manifest(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_unique_object)
    return validate_sequence_data(data, path), hashlib.sha256(raw).hexdigest()


def validate_sequence_data(data, path):
    """Validate/resolve alpha frame data shared by alpha and temporal manifests."""
    path = Path(path).resolve()
    allowed = {'schema_version', 'kind', 'fps', 'boundary_radius', 'channels', 'frames',
               'reference_provenance', 'source'}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError('invalid or unknown alpha sequence manifest fields')
    if (type(data.get('schema_version')) is not int or data['schema_version'] != 1 or
            data.get('kind') != 'alpha_sequence_evaluation'):
        raise ValueError('expected schema_version=1, kind=alpha_sequence_evaluation')
    fps = data.get('fps')
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        raise TypeError('fps must be numeric and not boolean')
    try:
        finite = math.isfinite(fps)
    except OverflowError as exc:
        raise ValueError('fps is outside the supported finite range') from exc
    if not finite or fps <= 0:
        raise ValueError('fps must be finite and positive; no inferred timing')
    radius = data.get('boundary_radius', 1)
    if type(radius) is not int or not 0 <= radius <= 64:
        raise ValueError('boundary_radius must be an integer in [0,64]')
    frames = data.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('frames must be a nonempty explicit list')
    previous = -1
    roles = None
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) - {'frame', *ROLES} or 'candidate' not in frame:
            raise ValueError('each frame requires a candidate and known fields')
        index = frame.get('frame')
        if type(index) is not int or index <= previous:
            raise ValueError('frame IDs must be increasing nonnegative integers')
        previous = index
        current = set(frame) - {'frame'}
        if roles is not None and roles != current:
            raise ValueError('asset roles must be present for every frame or absent for the whole clip')
        roles = current
        for role in current:
            if not isinstance(frame[role], str) or not frame[role].strip():
                raise ValueError('frame asset paths must be nonempty strings')
            frame[role] = str((path.parent / frame[role]).resolve())
    channels = data.get('channels', {})
    if (not isinstance(channels, dict) or set(channels) - set(ROLES) or
            any(value not in ('gray', 'alpha', 'replicated_gray') for value in channels.values()) or
            channels.get('mask', 'gray') != 'gray'):
        raise ValueError('channels must declare gray/alpha/replicated_gray; mask must be gray')
    provenance = data.get('reference_provenance')
    if 'reference' in (roles or ()):
        if (not isinstance(provenance, dict) or set(provenance) != PROVENANCE or
                any(not isinstance(v, str) or not v.strip() for v in provenance.values()) or
                provenance['split'] not in ('synthetic', 'development', 'held_out')):
            raise ValueError('references require author/origin/permitted_use/split provenance')
    elif provenance is not None:
        raise ValueError('reference provenance requires reference assets')
    if 'source' in data:
        if not isinstance(data['source'], str) or not data['source'].strip():
            raise ValueError('source must be a path string')
        data['source'] = str((path.parent / data['source']).resolve())
    return data


def read_alpha_frame(path, *, channel='gray', mask=False, check=None):
    """Decode data coverage: uint8/uint16 scale by full range; floats stay normalized."""
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
            raise ValueError(f'alpha decoder failed for {path}: {exc}') from exc
        if value is None:
            raise ValueError(f'cannot decode alpha frame: {path}')
        if value.dtype == np.uint8:
            value = value.astype(np.float64) / 255
        elif value.dtype == np.uint16:
            value = value.astype(np.float64) / 65535
        elif value.dtype.kind != 'f':
            raise ValueError('unsupported raster alpha dtype')
    else:
        raise ValueError('alpha frames must be .npy, .png or .exr; no lossy/video decoding is implicit')
    if channel == 'alpha' and value.ndim == 3 and value.shape[-1] == 4:
        value = value[..., 3]
    elif channel == 'replicated_gray' and value.ndim == 3 and value.shape[-1] == 3:
        if not (np.array_equal(value[..., 0], value[..., 1]) and
                np.array_equal(value[..., 0], value[..., 2])):
            raise ValueError('replicated_gray channels differ; no RGB-to-alpha conversion')
        value = value[..., 0]
    elif channel != 'gray' or value.ndim != 2:
        raise ValueError('alpha channel layout does not match explicit channel declaration')
    if value.dtype.kind not in 'buif' or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('decoded alpha must be finite normalized coverage in [0,1]')
    if mask:
        if not np.all((value == 0) | (value == 1)):
            raise ValueError('evaluation mask must be binary; no threshold is implicit')
        value = value.astype(bool)
    if before != file_sha256(path, check=check):
        raise ValueError(f'alpha frame changed during read: {path}')
    return value, {'path': str(path.resolve()), 'sha256': before,
                   'shape': list(value.shape), 'decoded_dtype': str(value.dtype), 'channel': channel}


def evaluate_sequence_manifest(path, *, check=None):
    data, digest = load_sequence_manifest(path)
    return evaluate_sequence_spec(data, manifest_path=path, manifest_sha256=digest, check=check)


def evaluate_sequence_spec(data, *, manifest_path, manifest_sha256, check=None):
    """Evaluate an already validated spec; benchmark binding may replace candidate paths."""
    identities = []

    def frames(role):
        for entry in data['frames']:
            array, identity = read_alpha_frame(entry[role], channel=data.get('channels', {}).get(role, 'gray'),
                                               mask=role == 'mask', check=check)
            identities.append({'frame': entry['frame'], 'role': role, **identity})
            yield array

    available = data['frames'][0]
    report = compare_stream(frames('candidate'), frames('reference') if 'reference' in available else None,
                            baseline_frames=frames('baseline') if 'baseline' in available else None,
                            masks=frames('mask') if 'mask' in available else None,
                            boundary_radius=data.get('boundary_radius', 1), check=check)
    report.update(kind='alpha_sequence_evaluation', manifest_sha256=manifest_sha256,
                  evaluation_contract={'schema_version': 1, 'family': 'alpha',
                                       'boundary_radius': data.get('boundary_radius', 1),
                                       'channels': {role: data.get('channels', {}).get(role, 'gray')
                                                    for role in ROLES if role in available}},
                  manifest_path=str(Path(manifest_path).resolve()), assets=identities,
                  frame_ids=[entry['frame'] for entry in data['frames']], fps=data['fps'],
                  timing_note='Declared frame IDs/fps; no encoded/VFR timing validation.',
                  reference_provenance=data.get('reference_provenance'),
                  source=source_identity(data['source'], check=check) if 'source' in data else None,
                  decode_contract='Data coverage only; uint8/uint16 rasters divide by255/65535; floating/npy '
                                  'alpha is already normalized. No colour transfer, resize or alignment inference.',
                  identity_scope='Manifest bytes and per-frame stable reads; no globally atomic media snapshot.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    data, _ = load_sequence_manifest(args.manifest)
    protected = {Path(entry[role]).resolve() for entry in data['frames'] for role in ROLES if role in entry}
    protected.add(args.manifest.resolve())
    if 'source' in data:
        source = Path(data['source']).resolve()
        if source.is_dir() and args.output.resolve().is_relative_to(source):
            parser.error('report must be outside the source frame directory')
        protected.add(source)
    if args.output.resolve() in protected:
        parser.error('report output must differ from all input assets/manifest')
    report = evaluate_sequence_manifest(args.manifest)
    from .diagnostics import write_json_atomic

    write_json_atomic(args.output, report)
    print(f"alpha sequence: {report['comparison_status']}; report: {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
