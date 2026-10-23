"""Conservative content identities for explicit whole-job EXR replay."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .diagnostics import json_value
from .identities import file_sha256, source_identity

ASSETS = {'weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir'}


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def tree_identity(path, *, check=None):
    """Hash every asset file; directory symlinks cannot silently omit dependencies."""
    path = Path(path).resolve()
    if path.is_file():
        return {'path': str(path), 'sha256': file_sha256(path, check=check), 'bytes': path.stat().st_size}
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for file in sorted(path.rglob('*')):
        if check is not None:
            check()
        if file.is_symlink() and file.is_dir():
            raise ValueError('asset directory symlinks require an explicit expanded asset tree')
        if file.is_file():
            files.append({'name': file.relative_to(path).as_posix(), 'sha256': file_sha256(file, check=check),
                          'bytes': file.stat().st_size})
    if not files:
        raise ValueError(f'asset directory is empty: {path}')
    return {'path': str(path), 'files': files, 'sha256': hashlib.sha256(canonical(files).encode()).hexdigest()}


def implementation_identity(root, *, check=None):
    root = Path(root).resolve()
    # Source-only bundles may import collected moge or the vendored tree that
    # runtime prepends. Both must participate; development has no root/moge.
    files = sorted({file for name in ('corridorkey', 'third_party', 'moge') for file in (root / name).rglob('*.py')
                    if file.is_file()})
    if not files or not (root / 'corridorkey/runtime.py').is_file():
        raise ValueError('implementation root must contain the shared runtime source')
    records = [{'name': file.relative_to(root).as_posix(), 'sha256': file_sha256(file, check=check)} for file in files]
    return {'files': records, 'sha256': hashlib.sha256(canonical(records).encode()).hexdigest()}


@dataclass(frozen=True)
class ReplayIdentity:
    payload: str

    def __post_init__(self):
        data = json.loads(self.payload)
        if (not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != 1 or
                data.get('kind') != 'whole_job_exr_replay' or self.payload != canonical(data)):
            raise ValueError('invalid canonical replay identity')

    @property
    def key(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


@dataclass(frozen=True)
class FeatureFrameIdentity:
    payload: str

    def __post_init__(self):
        data = json.loads(self.payload)
        fields = {'schema_version', 'kind', 'parent_key', 'frame', 'proxy_sha256', 'input_shape', 'input_dtype', 'channels', 'grid_hw'}
        if (not isinstance(data, dict) or set(data) != fields or type(data['schema_version']) is not int or
                data['schema_version'] != 1 or data['kind'] != 'iid_foundation_frame' or self.payload != canonical(data)):
            raise ValueError('invalid canonical IID frame identity')
        if any(not isinstance(data[name], str) or not re.fullmatch('[0-9a-f]{64}', data[name])
               for name in ('parent_key', 'proxy_sha256')):
            raise ValueError('IID identity requires parent/input SHA256')
        if type(data['frame']) is not int or data['frame'] < 0:
            raise ValueError('IID frame index must be nonnegative integer')
        if data['input_dtype'] not in ('torch.float32', 'torch.bfloat16'):
            raise ValueError('IID input precision must be explicit float32 or bfloat16')
        for name, length in (('input_shape', 3), ('grid_hw', 2)):
            values = data[name]
            if (not isinstance(values, list) or len(values) != length or
                    any(type(value) is not int or value <= 0 for value in values)):
                raise ValueError('IID input/grid geometry must be positive integers')
        channels = data['channels']
        if (data['input_shape'][0] != 3 or not isinstance(channels, dict) or not channels or
                set(channels) - {'moge', 'cradio'} or 'moge' not in channels or
                any(type(value) is not int or value <= 0 for value in channels.values())):
            raise ValueError('only explicit positive MoGe/C-RADIO feature channels are supported')

    @property
    def key(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


def feature_identity(parent: ReplayIdentity, frame, proxy_sha256, input_shape, channels, grid_hw, input_dtype='torch.float32'):
    if not isinstance(channels, dict) or set(channels) - {'moge', 'cradio'}:
        raise ValueError('RVM/unknown stages cannot enter the IID cache')
    if any(type(value) is not int or value < 0 for value in channels.values()):
        raise ValueError('feature channel counts must be nonnegative integers')
    return FeatureFrameIdentity(canonical({'schema_version': 1, 'kind': 'iid_foundation_frame',
                                           'parent_key': parent.key, 'frame': frame, 'proxy_sha256': proxy_sha256,
                                           'input_shape': list(input_shape), 'input_dtype': input_dtype, 'grid_hw': list(grid_hw),
                                           'channels': {name: value for name, value in channels.items() if value > 0}}))


def job_identity(job, *, assets, environment, implementation_root, check=None):
    """No model imports. Paths/settings conservatively participate except destination."""
    config, writer = job.configurations()
    job.validate_source(config)
    if config.output_format != 'exr' or config.auto_vram:
        raise ValueError('durable replay requires EXR output and explicit scheduling (auto_vram=false)')
    if set(assets) != ASSETS:
        raise ValueError('all runtime/foundation/stat asset identities are required')
    required = {'backend', 'precision', 'dependencies', 'runtime'}
    if not isinstance(environment, dict) or not required <= set(environment):
        raise ValueError('explicit backend/precision/dependencies/runtime context is required')
    if any(not isinstance(environment[name], str) or not environment[name] for name in ('backend', 'precision')):
        raise ValueError('backend and precision must be explicit names')
    if not isinstance(environment['runtime'], dict) or not environment['runtime'] or not isinstance(environment['dependencies'], dict) or not environment['dependencies']:
        raise ValueError('runtime and dependency context must be nonempty objects')
    corrections = None
    if config.corrections_path is not None:
        from .corrections import load_corrections

        document = load_corrections(config.corrections_path, source=job.input_path, check=check)
        corrections = {'manifest_sha256': file_sha256(config.corrections_path, check=check),
                       'identity': document.identity, 'document': document.to_dict()}
    data = {'schema_version': 1, 'kind': 'whole_job_exr_replay',
            'source': source_identity(job.input_path, check=check),
            'hints': {name: source_identity(path, check=check) if path is not None else None
                      for name, path in (('first', job.initial_hint_path), ('full', job.hint_video_path))},
            'corrections': corrections, 'settings': config, 'writer': writer,
            'profile_stages': job.profile_stages,
            'assets': {name: tree_identity(path, check=check) for name, path in sorted(assets.items())},
            'environment': environment, 'implementation': implementation_identity(implementation_root, check=check),
            'scope': 'completed whole job, fresh recurrent state; not an intermediate/recurrent checkpoint'}
    if check is not None:
        check()
    return ReplayIdentity(canonical(data))
