"""Versioned bounded local JSON frames; no model imports or executable payloads."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

from .diagnostics import finite_json_float, json_value
from .session import JobRequest

MAX_FRAME_BYTES = 4 * 1024 * 1024
EVENTS = {'ready', 'accepted', 'started', 'progress', 'result', 'failed', 'cancelled',
          'rejected', 'status', 'stopping', 'stopped'}
RECOVERY_FLAGS = ('resume', 'durable_foundations', 'durable_rvm', 'durable_hann', 'durable_stage2', 'durable_writer_prefix')


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}', value):
        raise ValueError('request identifiers require1..64 ASCII letters/digits/underscore/hyphen')
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON field')
        result[key] = value
    return result


def _constant(value) -> NoReturn:
    raise ValueError('worker JSON numbers must be finite')


def encode_frame(value):
    data = (json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError('worker frame exceeds byte limit')
    return data


def decode_frame(frame):
    if not isinstance(frame, bytes) or len(frame) > MAX_FRAME_BYTES or not frame.endswith(b'\n'):
        raise ValueError('worker frame requires a bounded complete UTF-8 JSON line')
    value = json.loads(frame, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=finite_json_float)
    if not isinstance(value, dict):
        raise TypeError('worker frame must be an object')
    return value


def _absolute(value):
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError('worker paths must be explicit absolute local paths')
    return value


def job_to_wire(job: JobRequest):
    config, writer = job.configurations()
    config = replace(config, temp_dir=Path(config.temp_dir).resolve())
    normalized = JobRequest.create(job.input_path, job.output_dir, config, writer,
                                   initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
                                   profile_stages=job.profile_stages)
    return json_value(asdict(normalized))


def job_from_wire(value):
    fields = {'input_path', 'output_dir', 'config_json', 'writer_json', 'initial_hint_path',
              'hint_video_path', 'profile_stages'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('worker job requires the complete shared JobRequest fields')
    for name in ('input_path', 'output_dir'):
        _absolute(value[name])
    for name in ('initial_hint_path', 'hint_video_path'):
        if value[name] is not None:
            _absolute(value[name])
    if type(value['profile_stages']) is not bool or any(not isinstance(value[name], str) for name in ('config_json', 'writer_json')):
        raise ValueError('worker job configuration/boolean types differ')
    normalized = dict(value)
    for field in ('config_json', 'writer_json'):
        parsed = json.loads(value[field], object_pairs_hook=_pairs, parse_constant=_constant, parse_float=finite_json_float)
        if not isinstance(parsed, dict):
            raise TypeError('worker settings must be JSON objects')
        normalized[field] = json.dumps(parsed, allow_nan=False)
    job = JobRequest(**normalized)
    try:
        config, writer = job.configurations()
    except (KeyError, TypeError) as exc:
        raise ValueError(f'invalid shared worker configuration: {exc}') from exc
    _absolute(str(config.temp_dir))
    return JobRequest.create(job.input_path, job.output_dir, config, writer,
                             initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
                             profile_stages=job.profile_stages)


@dataclass(frozen=True)
class Request:
    request_id: str
    action: str
    payload_json: str

    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)


def parse_request(value):
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'request_id', 'action', 'payload'} or
            type(value['schema_version']) is not int or value['schema_version'] != 1):
        raise ValueError('worker request requires exact schema_version1 envelope')
    request_id = identifier(value['request_id'])
    action, payload = value['action'], value['payload']
    if not isinstance(action, str) or not isinstance(payload, dict):
        raise TypeError('worker action/payload types differ')
    if action == 'render':
        if set(payload) != {'job'}:
            raise ValueError('render requires one shared job')
        payload = {'job': job_to_wire(job_from_wire(payload['job']))}
    elif action == 'recover':
        if set(payload) - {'batch', 'workspace', 'max_cache_bytes', *RECOVERY_FLAGS}:
            raise ValueError('unknown recovery field')
        for name in ('batch', 'workspace'):
            _absolute(payload.get(name))
        if type(payload.get('max_cache_bytes')) is not int or payload['max_cache_bytes'] < 0:
            raise ValueError('recovery byte budget must be nonnegative integer')
        payload = dict(payload)
        for flag in RECOVERY_FLAGS:
            if type(payload.get(flag, False)) is not bool:
                raise ValueError('recovery switches must be boolean')
            payload.setdefault(flag, False)
    elif action == 'cancel':
        if set(payload) - {'target', 'reason'} or not isinstance(payload.get('reason', ''), str):
            raise ValueError('cancel requires target and optional reason string')
        identifier(payload.get('target'))
        if len(payload.get('reason', '')) > 2000:
            raise ValueError('cancel reason exceeds limit')
    elif action == 'status':
        if payload:
            raise ValueError('status has no payload fields')
    elif action == 'shutdown':
        if set(payload) - {'drain'} or type(payload.get('drain', False)) is not bool:
            raise ValueError('shutdown requires optional drain boolean')
    else:
        raise ValueError('unsupported worker action')
    snapshot = json.dumps(json_value(payload), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return Request(request_id, action, snapshot)


def request_frame(request_id, action, payload):
    value = {'schema_version': 1, 'request_id': request_id, 'action': action, 'payload': payload}
    request = parse_request(value)
    return encode_frame({**value, 'payload': request.payload()})


def parse_event(frame):
    value = decode_frame(frame)
    fields = {'schema_version', 'generation', 'sequence', 'kind', 'request_id', 'payload'}
    if (set(value) != fields or type(value['schema_version']) is not int or value['schema_version'] != 1 or
            not isinstance(value['generation'], str) or not re.fullmatch('[0-9a-f]{32}', value['generation']) or
            type(value['sequence']) is not int or value['sequence'] < 0 or value['kind'] not in EVENTS or
            not isinstance(value['payload'], dict)):
        raise ValueError('invalid worker event envelope')
    if value['request_id'] is not None:
        identifier(value['request_id'])
    return value
