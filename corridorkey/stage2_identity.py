"""Typed exact-call identity for a complete selected Stage2 prediction batch."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .recovery_identity import (
    ReplayIdentity,
    _project_conditioning_inputs,
    _validate_conditioning_projection,
    _validate_mixed_region_projection,
    _validate_pure_region_projection,
    canonical,
)

MAX_IDENTITY_BYTES = 65536
DTYPE_NAMES = ('torch.float32', 'torch.bfloat16', 'torch.float16')


_LEGACY_FINISHING_SETTINGS = frozenset({'alpha_expand_radius', 'alpha_feather_radius'})
FINISHING_SETTINGS = _LEGACY_FINISHING_SETTINGS | frozenset({
    'despill_strength', 'despill_screen', 'despill_space', 'despill_preserve_luma', 'despill_edges_only'})
_CONTEXT_FIELDS = frozenset({'schema_version', 'kind', 'source', 'hints', 'corrections',
    'settings', 'writer', 'profile_stages', 'assets', 'environment', 'implementation', 'scope'})




@dataclass(frozen=True)
class Stage2FinishingContext:
    """Versioned exact-call dependencies; legacy final-only policies stay readable."""
    payload: str

    def __post_init__(self):
        data = json.loads(self.payload)
        if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
                or data['schema_version'] not in (1, 2, 3, 4, 5) or data.get('kind') != 'stage2_finishing_context'
                or not _CONTEXT_FIELDS <= data.keys()
                or any(not isinstance(data[name], dict) for name in
                       ('source', 'hints', 'settings', 'writer', 'assets', 'environment', 'implementation'))
                or not {'first', 'full'} <= data['hints'].keys()
                or (data['corrections'] is not None and not isinstance(data['corrections'], dict))
                or type(data['profile_stages']) is not bool or not isinstance(data['scope'], str)
                or (_LEGACY_FINISHING_SETTINGS if data['schema_version'] == 1 else FINISHING_SETTINGS).intersection(data['settings'])
                or self.payload != canonical(data)):
            raise ValueError('Invalid canonical Stage2 finishing context')
        if data['schema_version'] == 3:
            _validate_pure_region_projection(data)
        elif data['schema_version'] == 4:
            _validate_mixed_region_projection(data)
        elif data['schema_version'] == 5:
            _validate_conditioning_projection(data, prefix='stage2')

    @property
    def key(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


def stage2_finishing_context(parent: ReplayIdentity):
    if not isinstance(parent, ReplayIdentity):
        raise TypeError('Stage2 finishing context requires a replay identity instance')
    data = parent.to_dict()
    settings = data.get('settings')
    if not isinstance(settings, dict) or not FINISHING_SETTINGS <= settings.keys():
        raise ValueError('Stage2 finishing reuse requires all seven final coverage/composite settings')
    if (not _CONTEXT_FIELDS <= data.keys() or not isinstance(data.get('hints'), dict)
            or not {'first', 'full'} <= data['hints'].keys()):
        raise ValueError('Stage2 reuse requires the complete original dependency inventory')
    _project_conditioning_inputs(data, prefix='stage2')
    data['schema_version'] = 5
    data['kind'] = 'stage2_finishing_context'
    data['settings'] = {name: value for name, value in settings.items() if name not in FINISHING_SETTINGS}
    return Stage2FinishingContext(canonical(data))


@dataclass(frozen=True)
class Stage2BatchIdentity:
    payload: str

    def __post_init__(self):
        if not isinstance(self.payload, str) or len(self.payload.encode()) > MAX_IDENTITY_BYTES:
            raise ValueError('Stage2 identity exceeds its metadata bound')
        try:
            data = json.loads(self.payload)
        except (ValueError, RecursionError) as exc:
            raise ValueError('Invalid Stage2 identity JSON') from exc
        fields = {'schema_version', 'kind', 'parent_key', 'frames', 'policy', 'execution', 'inputs'}
        if (not isinstance(data, dict) or set(data) != fields or type(data['schema_version']) is not int
                or data['schema_version'] != 1 or data['kind'] != 'stage2_prediction_batch'
                or self.payload != canonical(data)):
            raise ValueError('Invalid canonical Stage2 batch identity')
        if not isinstance(data['parent_key'], str) or re.fullmatch('[0-9a-f]{64}', data['parent_key']) is None:
            raise ValueError('Stage2 requires a parent SHA256')
        frames = data['frames']
        if (not isinstance(frames, list) or not 1 <= len(frames) <= 4096
                or any(type(frame) is not int or frame < 0 for frame in frames)
                or len(set(frames)) != len(frames)):
            raise ValueError('Stage2 requires ordered unique nonnegative native frame indices')
        if data['policy'] not in ('full_rate', 'quick_anchor', 'quick_fallback'):
            raise ValueError('Unknown Stage2 call policy')
        execution = data['execution']
        if (not isinstance(execution, dict) or set(execution) != {'backend', 'device', 'autocast_dtype'}
                or execution['backend'] != 'pytorch' or execution['autocast_dtype'] not in DTYPE_NAMES
                or not isinstance(execution['device'], str)
                or re.fullmatch(r'cpu|cuda:\d+', execution['device']) is None):
            raise ValueError('Stage2 requires an explicit backend/device/autocast contract')
        inputs = data['inputs']
        if not isinstance(inputs, dict) or set(inputs) != {'rgb', 'baton'}:
            raise ValueError('Stage2 requires its complete model argument inventory')
        for item in inputs.values():
            if (not isinstance(item, dict) or set(item) != {'shape', 'dtype', 'sha256'}
                    or item['dtype'] not in DTYPE_NAMES or not isinstance(item['sha256'], str)
                    or re.fullmatch('[0-9a-f]{64}', item['sha256']) is None):
                raise ValueError('Stage2 inputs require exact floating dtype/shape/bit identities')
            shape = item['shape']
            if (not isinstance(shape, list) or len(shape) != 5
                    or any(type(size) is not int or size <= 0 for size in shape)
                    or shape[:2] != [1, len(frames)]):
                raise ValueError('Stage2 input batch geometry differs')
        rgb, baton = inputs['rgb']['shape'], inputs['baton']['shape']
        if (rgb[2] != 3 or any(size % 16 for size in rgb[-2:])
                or baton[-2:] != [rgb[-2] // 16, rgb[-1] // 16]):
            raise ValueError('Stage2 RGB/baton padded geometry differs')

    @property
    def key(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


def stage2_batch_identity(parent: ReplayIdentity | Stage2FinishingContext, *, frames, policy, execution, inputs):
    if not isinstance(parent, (ReplayIdentity, Stage2FinishingContext)):
        raise TypeError('Stage2 parent must be an explicit replay or finishing context instance')
    return Stage2BatchIdentity(canonical({'schema_version': 1, 'kind': 'stage2_prediction_batch',
        'parent_key': parent.key, 'frames': list(frames), 'policy': policy,
        'execution': execution, 'inputs': inputs}))
