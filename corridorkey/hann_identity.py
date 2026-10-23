"""Immutable identities for Stage1 events replayed through the Hann state machine."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .recovery_identity import ReplayIdentity, canonical


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class HannWindowIdentity:
    payload: str

    def __post_init__(self):
        data = json.loads(self.payload)
        fields = {'schema_version', 'kind', 'parent_key', 'schedule_sha256', 'incoming_replay_sha256',
                  'index', 'start', 'stop', 'next_start', 'segment', 'carry', 'd_16', 'inputs'}
        if (not isinstance(data, dict) or set(data) != fields or type(data['schema_version']) is not int or
                data['schema_version'] != 1 or data['kind'] != 'hann_stage1_event' or self.payload != canonical(data)):
            raise ValueError('invalid canonical Hann window identity')
        if any(not _sha(data[name]) for name in ('parent_key', 'schedule_sha256', 'incoming_replay_sha256')):
            raise ValueError('Hann window requires parent/schedule/incoming-state SHA256')
        if (any(type(data[name]) is not int for name in ('index', 'start', 'stop', 'next_start', 'd_16')) or
                data['index'] < 0 or not 0 <= data['start'] < data['stop'] or
                data['next_start'] < data['start'] or data['d_16'] <= 0 or type(data['carry']) is not bool):
            raise ValueError('Hann window bounds/cursor/width/carry contract differs')
        segment = data['segment']
        if (not isinstance(segment, list) or len(segment) != 2 or
                any(type(value) is not int for value in segment) or
                not 0 <= segment[0] <= data['start'] < data['stop'] <= segment[1]):
            raise ValueError('Hann window must be inside its selected-timeline shot')
        inputs = data['inputs']
        names = {'rgb', 'hint', 'hint_quality', 'moge', 'cradio', 'rvm'}
        if not isinstance(inputs, dict) or set(inputs) != names:
            raise ValueError('Hann window requires the complete Stage1 argument inventory')
        count = data['stop'] - data['start']
        for name, value in inputs.items():
            if value is None and name in {'cradio', 'rvm'}:
                continue
            if (not isinstance(value, dict) or set(value) != {'sha256', 'dtype', 'shape'} or
                    not _sha(value['sha256']) or value['dtype'] not in
                    ('torch.float32', 'torch.bfloat16', 'torch.float16')):
                raise ValueError('Hann input must have exact floating precision and bit identity')
            shape = value['shape']
            if (not isinstance(shape, list) or len(shape) != (2 if name == 'hint_quality' else 5) or
                    any(type(size) is not int or size <= 0 for size in shape) or shape[:2] != [1, count]):
                raise ValueError('Hann input geometry differs from exact native window')
        rgb, hint = inputs['rgb']['shape'], inputs['hint']['shape']
        if (rgb[2] != 3 or hint[2] != 1 or rgb[-2:] != hint[-2:] or
                any(size % 64 for size in rgb[-2:])):
            raise ValueError('Hann RGB/hint must share padded RGB geometry divisible by64')
        for name in ('moge', 'cradio', 'rvm'):
            value = inputs[name]
            if value is not None and value['shape'][-2:] != [rgb[-2] // 64, rgb[-1] // 64]:
                raise ValueError('Hann foundation grid differs from padded RGB geometry')

    @property
    def key(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


def hann_window_identity(parent: ReplayIdentity, *, schedule_sha256, incoming_replay_sha256,
                         index, start, stop, next_start, segment, carry, d_16, inputs):
    return HannWindowIdentity(canonical({
        'schema_version': 1, 'kind': 'hann_stage1_event', 'parent_key': parent.key,
        'schedule_sha256': schedule_sha256, 'incoming_replay_sha256': incoming_replay_sha256,
        'index': index, 'start': start, 'stop': stop, 'next_start': next_start,
        'segment': list(segment), 'carry': carry, 'd_16': d_16, 'inputs': inputs}))
