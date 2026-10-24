"""Conservative content identities for explicit whole-job EXR replay."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .diagnostics import json_value
from .identities import file_sha256, sequence_identity, source_identity
from .scheduling import frame_path_key

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


_PURE_REGION_FIELDS = frozenset({'kind', 'schema_version', 'document_schema_version', 'present', 'source', 'shots'})


def _validate_pure_region_projection(data):
    from .corrections import CorrectionDocument

    projected = data['corrections']
    if (not isinstance(projected, dict) or set(projected) != _PURE_REGION_FIELDS
            or projected['kind'] != 'pure_render_region_schedule'
            or type(projected['schema_version']) is not int or projected['schema_version'] != 1
            or type(projected['document_schema_version']) is not int or projected['document_schema_version'] != 5
            or projected['present'] is not True or 'corrections_path' in data['settings']):
        raise ValueError('Invalid pure render-region dependency projection')
    # Reuse strict source/shot validation without reading any mask assets.
    CorrectionDocument.parse({'schema_version': 5, 'revision': 0, 'source': projected['source'],
                              'shots': projected['shots'], 'entries': [], 'tracks': [], 'guidance': []})
    if data['source'].get('sha256') != projected['source']['sha256']:
        raise ValueError('Projected correction source differs from full source binding')


def _is_render_region(entry):
    return entry['kind'] == 'region' and (entry['purpose'], entry['target']) in (
        ('composite', 'despill'), ('diagnostic', 'tracking_overlay'))


def _validate_mixed_region_projection(data):
    from .corrections import CorrectionDocument

    projected = data['corrections']
    if (not isinstance(projected, dict) or set(projected) != {'kind', 'schema_version', 'present', 'document'}
            or projected['kind'] != 'mixed_finishing_dependencies'
            or type(projected['schema_version']) is not int or projected['schema_version'] != 1
            or projected['present'] is not True or 'corrections_path' in data['settings']):
        raise ValueError('Invalid mixed finishing dependency projection')
    document = CorrectionDocument.parse(projected['document']).to_dict()
    if (document['schema_version'] != 5 or document['revision'] != 0 or not document['entries']
            or document['tracks'] or any(_is_render_region(entry) for entry in document['entries'])
            or data['source'].get('sha256') != document['source']['sha256']):
        raise ValueError('Mixed finishing projection must retain exact nonfinal dependencies only')


def _original_correction_document(data):
    """Validate the complete original wrapper before any private projection."""
    from .corrections import CorrectionDocument

    wrapper = data.get('corrections')
    if wrapper is None:
        return None
    path = data['settings'].get('corrections_path')
    source = data.get('source')
    if (not isinstance(source, dict) or not isinstance(wrapper, dict) or set(wrapper) != {'manifest_sha256', 'identity', 'document'}
            or not isinstance(path, str) or not path
            or any(not isinstance(wrapper[name], str) or re.fullmatch('[0-9a-f]{64}', wrapper[name]) is None
                   for name in ('manifest_sha256', 'identity'))):
        raise ValueError('Finishing reuse requires an original complete correction wrapper')
    document = CorrectionDocument.parse(wrapper['document'])
    if document.identity != wrapper['identity'] or source.get('sha256') != document.to_dict()['source']['sha256']:
        raise ValueError('Correction wrapper identity/source differs from its document')
    return document


def _finishing_correction_version(data):
    """Choose a private policy only after strict complete original-document validation."""
    from .corrections import CorrectionDocument

    document = _original_correction_document(data)
    if document is None:
        return 2
    authored = document.to_dict()
    if authored['schema_version'] != 5 or not authored['entries']:
        return 2
    retained = [entry for entry in authored['entries'] if not _is_render_region(entry)]
    if not retained:
        data['corrections'] = {'kind': 'pure_render_region_schedule', 'schema_version': 1,
                               'document_schema_version': 5, 'present': True,
                               'source': authored['source'], 'shots': authored['shots']}
        version = 3
    else:
        # Accepted derived tracks are downstream-only; validate originals above,
        # never rebind their entry_order/SHA against this private filtered document.
        filtered = {**authored, 'revision': 0, 'entries': retained, 'tracks': []}
        CorrectionDocument.parse(filtered)
        data['corrections'] = {'kind': 'mixed_finishing_dependencies', 'schema_version': 1,
                               'present': True, 'document': filtered}
        version = 4
    del data['settings']['corrections_path']
    return version



def _validate_hint_identity(record):
    """Admit only existing file/ordered-sequence identity forms; do no I/O."""
    if record is None:
        return
    if (not isinstance(record, dict) or set(record) not in ({'path', 'sha256'}, {'path', 'files', 'sha256'})
            or not isinstance(record['path'], str) or not record['path']
            or not isinstance(record['sha256'], str) or re.fullmatch('[0-9a-f]{64}', record['sha256']) is None):
        raise ValueError('Stage2 reuse requires an original hint file/sequence identity')
    if 'files' not in record:
        return
    rows = record['files']
    if type(rows) is not list:
        raise ValueError('Hint sequence requires ordered file identities')
    names = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'name', 'sha256'}
                or not isinstance(row['name'], str) or not row['name'] or row['name'] in ('.', '..')
                or Path(row['name']).name != row['name']
                or not isinstance(row['sha256'], str) or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None):
            raise ValueError('Invalid hint sequence file identity')
        names.append(row['name'])
    if (len(set(names)) != len(names) or names != sorted(names, key=lambda name: frame_path_key(Path(name)))
            or sequence_identity(record['path'], rows)['sha256'] != record['sha256']):
        raise ValueError('Hint sequence order or aggregate differs')



def _validate_conditioning_projection(data, *, prefix):
    if 'corrections_path' in data['settings']:
        raise ValueError('Exact-input context cannot retain the private correction path')
    for record, kind in ((data['corrections'], prefix + '_exact_call_corrections'),
                         (data['hints']['first'], prefix + '_exact_call_hint'),
                         (data['hints']['full'], prefix + '_exact_call_hint')):
        if (not isinstance(record, dict) or set(record) != {'kind', 'schema_version'}
                or type(record['schema_version']) is not int
                or record != {'kind': kind, 'schema_version': 1}):
            raise ValueError('Invalid closed exact-input dependency projection')


def _project_conditioning_inputs(data, *, prefix):
    document = _original_correction_document(data)
    if document is None and data['settings'].get('corrections_path') is not None:
        raise ValueError('Correction path requires an original correction document')
    for name in ('first', 'full'):
        _validate_hint_identity(data['hints'][name])
    # Only raw exact-input child entries use this projection. Original full-job
    # validation and current postprocessing remain mandatory at their boundaries.
    data['corrections'] = {'kind': prefix + '_exact_call_corrections', 'schema_version': 1}
    for name in ('first', 'full'):
        data['hints'][name] = {'kind': prefix + '_exact_call_hint', 'schema_version': 1}
    data['settings'].pop('corrections_path', None)

_UPSTREAM_LEGACY_FINAL_SETTINGS = frozenset({'alpha_expand_radius', 'alpha_feather_radius'})
_UPSTREAM_FINAL_SETTINGS = _UPSTREAM_LEGACY_FINAL_SETTINGS | frozenset({
    'despill_strength', 'despill_screen', 'despill_space', 'despill_preserve_luma', 'despill_edges_only'})
_UPSTREAM_CONTEXT_FIELDS = frozenset({'schema_version', 'kind', 'source', 'hints', 'corrections',
    'settings', 'writer', 'profile_stages', 'assets', 'environment', 'implementation', 'scope', 'reuse_scope'})


@dataclass(frozen=True)
class UpstreamFinishingContext:
    """Domain-separated IID/RVM/Hann dependencies minus versioned final-only controls."""
    payload: str

    def __post_init__(self):
        data = json.loads(self.payload)
        if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
                or data['schema_version'] not in (1, 2, 3, 4, 5) or data.get('kind') != 'upstream_finishing_context'
                or not _UPSTREAM_CONTEXT_FIELDS <= data.keys()
                or data.get('reuse_scope') != ('rvm_hann_exact_inputs' if data['schema_version'] == 5 else 'iid_rvm_hann_raw_inputs')
                or any(not isinstance(data[name], dict) for name in
                       ('source', 'hints', 'settings', 'writer', 'assets', 'environment', 'implementation'))
                or not {'first', 'full'} <= data['hints'].keys()
                or (data['corrections'] is not None and not isinstance(data['corrections'], dict))
                or type(data['profile_stages']) is not bool or not isinstance(data['scope'], str)
                or (_UPSTREAM_LEGACY_FINAL_SETTINGS if data['schema_version'] == 1 else _UPSTREAM_FINAL_SETTINGS).intersection(data['settings'])
                or self.payload != canonical(data)):
            raise ValueError('Invalid canonical upstream finishing context')
        if data['schema_version'] == 3:
            _validate_pure_region_projection(data)
        elif data['schema_version'] == 4:
            _validate_mixed_region_projection(data)
        elif data['schema_version'] == 5:
            _validate_conditioning_projection(data, prefix='temporal')
        object.__setattr__(self, '_reuse_scope', data['reuse_scope'])

    @property
    def reuse_scope(self):
        return self.__dict__['_reuse_scope']

    @property
    def key(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


def upstream_finishing_context(parent: ReplayIdentity):
    if not isinstance(parent, ReplayIdentity):
        raise TypeError('Upstream finishing context requires a replay identity instance')
    data = parent.to_dict()
    settings = data.get('settings')
    if not isinstance(settings, dict) or not _UPSTREAM_FINAL_SETTINGS <= settings.keys():
        raise ValueError('Upstream finishing reuse requires all seven final coverage/composite settings')
    data['schema_version'] = _finishing_correction_version(data)
    data['kind'] = 'upstream_finishing_context'
    data['reuse_scope'] = 'iid_rvm_hann_raw_inputs'
    data['settings'] = {name: value for name, value in settings.items() if name not in _UPSTREAM_FINAL_SETTINGS}
    return UpstreamFinishingContext(canonical(data))


def temporal_finishing_context(parent: ReplayIdentity):
    """RVM/Hann-only parent; every raw child still binds complete actual state."""
    if not isinstance(parent, ReplayIdentity):
        raise TypeError('Temporal context requires a replay identity instance')
    data = parent.to_dict()
    settings = data.get('settings')
    if (not isinstance(settings, dict) or not _UPSTREAM_FINAL_SETTINGS <= settings.keys()
            or not (_UPSTREAM_CONTEXT_FIELDS - {'reuse_scope'}) <= data.keys()
            or not isinstance(data.get('hints'), dict) or not {'first', 'full'} <= data['hints'].keys()):
        raise ValueError('Temporal reuse requires the complete original dependency inventory')
    _project_conditioning_inputs(data, prefix='temporal')
    data['schema_version'] = 5
    data['kind'] = 'upstream_finishing_context'
    data['reuse_scope'] = 'rvm_hann_exact_inputs'
    data['settings'] = {name: value for name, value in settings.items() if name not in _UPSTREAM_FINAL_SETTINGS}
    return UpstreamFinishingContext(canonical(data))


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


def feature_identity(parent: ReplayIdentity | UpstreamFinishingContext, frame, proxy_sha256, input_shape, channels, grid_hw, input_dtype='torch.float32'):
    if not isinstance(parent, (ReplayIdentity, UpstreamFinishingContext)):
        raise TypeError('IID parent must be an explicit replay or upstream finishing context instance')
    if isinstance(parent, UpstreamFinishingContext) and parent.reuse_scope != 'iid_rvm_hann_raw_inputs':
        raise ValueError('Temporal-only context cannot enter the IID cache')
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
