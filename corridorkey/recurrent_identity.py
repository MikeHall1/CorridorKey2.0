"""Versioned exact-call identities for chronological RVM chunk checkpoints."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .recovery_identity import ReplayIdentity, UpstreamFinishingContext, canonical


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class RvmChunkIdentity:
    payload: str

    def __post_init__(self):
        data = json.loads(self.payload)
        fields = {'schema_version', 'kind', 'parent_key', 'shot_start', 'start', 'stop',
                  'proxy_sha256', 'input_shape', 'input_dtype', 'incoming_state_sha256', 'channels', 'grid_hw'}
        if (not isinstance(data, dict) or set(data) != fields or type(data['schema_version']) is not int or
                data['schema_version'] != 1 or data['kind'] != 'rvm_chronological_chunk' or
                self.payload != canonical(data)):
            raise ValueError('invalid canonical RVM chunk identity')
        if not _sha(data['parent_key']) or not _sha(data['proxy_sha256']):
            raise ValueError('RVM identity requires parent/proxy SHA256')
        if (any(type(data[name]) is not int for name in ('shot_start', 'start', 'stop', 'channels')) or
                not 0 <= data['shot_start'] <= data['start'] < data['stop'] or data['channels'] <= 0):
            raise ValueError('RVM chunk must have a positive ordered native range and channel width')
        state = data['incoming_state_sha256']
        if (data['start'] == data['shot_start'] and state is not None or
                data['start'] > data['shot_start'] and not _sha(state)):
            raise ValueError('RVM chunks require fresh state at shot start and exact incoming state afterwards')
        if data['input_dtype'] not in ('torch.float32', 'torch.bfloat16'):
            raise ValueError('RVM proxy precision must be explicit float32/bfloat16')
        for name, length in (('input_shape', 4), ('grid_hw', 2)):
            values = data[name]
            if (not isinstance(values, list) or len(values) != length or
                    any(type(value) is not int or value <= 0 for value in values)):
                raise ValueError('RVM input/grid geometry must be positive integers')
        if data['input_shape'][:2] != [data['stop'] - data['start'], 3]:
            raise ValueError('RVM proxy shape differs from exact chunk range/RGB contract')

    @property
    def key(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


def rvm_chunk_identity(parent: ReplayIdentity | UpstreamFinishingContext, *, shot_start, start, stop, proxy_sha256,
                       input_shape, input_dtype, incoming_state_sha256, channels, grid_hw):
    if not isinstance(parent, (ReplayIdentity, UpstreamFinishingContext)):
        raise TypeError('Parent must be an explicit replay or upstream finishing context instance')
    return RvmChunkIdentity(canonical({
        'schema_version': 1, 'kind': 'rvm_chronological_chunk', 'parent_key': parent.key,
        'shot_start': shot_start, 'start': start, 'stop': stop, 'proxy_sha256': proxy_sha256,
        'input_shape': list(input_shape), 'input_dtype': input_dtype,
        'incoming_state_sha256': incoming_state_sha256, 'channels': channels, 'grid_hw': list(grid_hw)}))
