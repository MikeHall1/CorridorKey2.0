"""Atomic exact-call RVM features plus recurrent state; no Hann/output resume."""
from __future__ import annotations

import hashlib
import json
import math

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

from .durable_context import BoundModelCache
from .recovery_identity import canonical
from .recovery_store import CacheCapacity, CacheInvalid, read_payload_bounded
from .recurrent_identity import RvmChunkIdentity, rvm_chunk_identity

STATE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 65536
DTYPE_CODES = {'torch.float16': ('F16', 2), 'torch.bfloat16': ('BF16', 2), 'torch.float32': ('F32', 4)}


def _unique_pairs(items):
    result = {}
    for name, value in items:
        if name in result:
            raise CacheInvalid('duplicate durable tensor header field')
        result[name] = value
    return result


def validate_tensor_header(blob, descriptions):
    """Validate exact declared tensor extents before safetensors allocation."""
    if len(blob) < 8:
        raise CacheInvalid('truncated durable tensor header')
    length = int.from_bytes(blob[:8], 'little')
    if not 1 <= length <= MAX_HEADER_BYTES or 8 + length > len(blob):
        raise CacheInvalid('durable tensor header exceeds bounded extent')
    header = json.loads(blob[8:8 + length], object_pairs_hook=_unique_pairs)
    if not isinstance(header, dict) or set(header) != set(descriptions):
        raise CacheInvalid('durable tensor header inventory differs')
    intervals = []
    for name, expected in descriptions.items():
        item = header[name]
        dtype, size = DTYPE_CODES[expected['dtype']]
        if (not isinstance(item, dict) or set(item) != {'dtype', 'shape', 'data_offsets'}
                or item['dtype'] != dtype or item['shape'] != expected['shape']
                or not isinstance(item['shape'], list) or any(type(n) is not int for n in item['shape'])
                or not isinstance(item['data_offsets'], list) or len(item['data_offsets']) != 2
                or any(type(n) is not int for n in item['data_offsets'])):
            raise CacheInvalid('durable tensor header dtype/shape differs')
        first, last = item['data_offsets']
        if first < 0 or last - first != math.prod(expected['shape']) * size:
            raise CacheInvalid('durable tensor header offsets differ')
        intervals.append((first, last))
    end = 0
    for first, last in sorted(intervals):
        if first != end:
            raise CacheInvalid('durable tensor header has gaps/overlapping extents')
        end = last
    if 8 + length + end != len(blob):
        raise CacheInvalid('durable tensor payload length differs')


def tensor_digest(value):
    """Hash exact dtype/shape/bits, including bfloat16 without numeric conversion."""
    host = value.detach().to('cpu').contiguous()
    description = canonical({'shape': list(host.shape), 'dtype': str(host.dtype)}).encode()
    return hashlib.sha256(description + b'\0' + host.view(torch.uint8).numpy().tobytes()).hexdigest()


class DurableRvmChunks(BoundModelCache):
    def __init__(self, store, parent_identity, *, validate_context, check=None):
        super().__init__(store, parent_identity, validate_context=validate_context, check=check)
        self.hits = 0
        self.misses = 0
        self.frames_reused = 0
        self.stored = 0
        self.corrupt = 0
        self.budget_rejected = 0
        self.bytes_loaded = 0
        self.last_error = None

    def identity(self, start, shot_start, display, state, *, channels, grid_hw):
        self._require_bound()
        if (not isinstance(display, torch.Tensor) or display.ndim != 4 or display.shape[1] != 3 or
                display.dtype not in (torch.float32, torch.bfloat16) or
                not bool(torch.isfinite(display).all()) or bool(((display < 0) | (display > 1)).any())):
            raise ValueError('RVM proxy must be finite TCHW RGB float32/bfloat16 in [0,1]')
        if state is not None:
            self._state(state, channels, grid_hw)
        return rvm_chunk_identity(self.parent, shot_start=shot_start, start=start, stop=start + len(display),
                                  proxy_sha256=tensor_digest(display), input_shape=tuple(display.shape),
                                  input_dtype=str(display.dtype), channels=channels, grid_hw=grid_hw,
                                  incoming_state_sha256=tensor_digest(state) if state is not None else None)

    @staticmethod
    def _state(state, channels, grid_hw):
        if (not isinstance(state, torch.Tensor) or state.dtype not in STATE_DTYPES or
                tuple(state.shape) != (1, 1 + grid_hw[0] * grid_hw[1], channels) or
                not bool(torch.isfinite(state).all())):
            raise ValueError('RVM state must be finite exact floating (1, CLS+grid tokens, channels)')

    def _contract(self, identity):
        self._require_bound()
        if not isinstance(identity, RvmChunkIdentity) or identity.to_dict()['parent_key'] != self.parent.key:
            raise CacheInvalid('RVM checkpoint belongs to another typed job context')
        return identity.to_dict()

    def _validate(self, data, features, state):
        shape = (data['stop'] - data['start'], data['channels'], *data['grid_hw'])
        if (not isinstance(features, torch.Tensor) or features.dtype != torch.float16 or
                tuple(features.shape) != shape or not bool(torch.isfinite(features).all())):
            raise ValueError('RVM features must be finite exact normalized float16 TCHW')
        self._state(state, data['channels'], data['grid_hw'])

    def _shapes(self, data):
        height, width = data['grid_hw']
        return {'features': [data['stop'] - data['start'], data['channels'], height, width],
                'state': [1, 1 + height * width, data['channels']]}

    def _payload_bound(self, data):
        shapes = self._shapes(data)
        return min(self.store.max_bytes, 8 + MAX_HEADER_BYTES
                   + math.prod(shapes['features']) * 2 + math.prod(shapes['state']) * 4)

    def _record_schema(self, dtype, shape):
        return {'stage': 'rvm_exact_chronological_call', 'parent_identity': self.parent.to_dict(),
                'features_dtype': 'torch.float16', 'state_dtype': dtype,
                'state_shape': shape, 'hann_output_resume': False}

    def _record(self, state):
        return self._record_schema(str(state.dtype), list(state.shape))

    def _descriptions(self, data, record):
        shapes = self._shapes(data)
        dtype = record.get('state_dtype') if isinstance(record, dict) else None
        if (not isinstance(dtype, str) or dtype not in DTYPE_CODES
                or canonical(record) != canonical(self._record_schema(dtype, shapes['state']))):
            raise CacheInvalid('RVM provenance differs from exact producer contract')
        return {'features': {'dtype': 'torch.float16', 'shape': shapes['features']},
                'state': {'dtype': dtype, 'shape': shapes['state']}}

    def get(self, identity):
        data = self._contract(identity)
        try:
            manifest = self.store.lookup(identity, check=self.check, max_metadata_bytes=MAX_METADATA_BYTES,
                                         max_entry_files=3, max_payload_bytes=self._payload_bound(data))
            if manifest is None:
                self.misses += 1
                return None
            files = manifest['files']
            if len(files) != 1 or files[0]['name'] != 'rvm.safetensors':
                raise CacheInvalid('RVM checkpoint tensor inventory differs')
            file = self.store._entry(identity) / files[0]['blob']
            descriptions = self._descriptions(data, manifest.get('origin_record'))
            blob = read_payload_bounded(file, expected_size=files[0]['bytes'], expected_sha256=files[0]['sha256'],
                                        max_bytes=self._payload_bound(data), check=self.check)
            validate_tensor_header(blob, descriptions)
            if self.check is not None:
                self.check()
            tensors = load(blob)  # Immutable verified bytes; no pickle, live mmap or read/use race.
            if set(tensors) != {'features', 'state'}:
                raise CacheInvalid('RVM checkpoint requires features and outgoing state together')
            self._validate(data, tensors['features'], tensors['state'])
            if canonical(manifest['origin_record']) != canonical(self._record(tensors['state'])):
                raise CacheInvalid('RVM provenance differs from exact producer contract')
            self.hits += 1
            self.frames_reused += data['stop'] - data['start']
            self.bytes_loaded += len(blob)
            return tensors
        except (CacheInvalid, OSError, ValueError, SafetensorError, RecursionError) as exc:
            self.corrupt += 1
            self.misses += 1
            self.last_error = str(exc)[:512]
            self.store.discard(identity, max_metadata_bytes=MAX_METADATA_BYTES, max_entry_files=3)
            return None

    def put(self, identity, features, state):
        data = self._contract(identity)
        self._validate(data, features, state)
        minimum_bytes = sum(value.numel() * value.element_size() for value in (features, state))
        record = self._record(state)
        metadata_bytes = len(canonical(record).encode()) + len(identity.payload.encode()) + 4096
        if minimum_bytes > self.store.max_bytes or metadata_bytes > MAX_METADATA_BYTES:
            self.budget_rejected += 1
            return False
        # A lower bound on the actual sealed entry also avoids staging tensors
        # when even data plus required metadata cannot fit this byte budget.
        lower_manifest = {'schema_version': 1, 'key': identity.key, 'identity': identity.to_dict(),
                          'files': [{'name': 'rvm.safetensors', 'blob': 'payload-000000.bin',
                                     'bytes': minimum_bytes, 'sha256': '0' * 64}], 'origin_record': record}
        lower_owner = {'owner': self.store.owner, 'manifest_sha256': '0' * 64}
        if minimum_bytes + len(canonical(lower_manifest).encode()) + len(canonical(lower_owner).encode()) > self.store.max_bytes:
            self.budget_rejected += 1
            return False
        tensors = {name: value.detach().to(device='cpu', copy=True).contiguous()
                   for name, value in (('features', features), ('state', state))}
        blob = save(tensors)
        if self.check is not None:
            self.check()
        if len(blob) > self._payload_bound(data):
            self.budget_rejected += 1
            return False
        with self.store.private_stage(max_metadata_bytes=MAX_METADATA_BYTES, max_entry_files=3) as folder:
            file = folder / 'payload-000000.bin'
            file.write_bytes(blob)
            record = self._record(state)
            try:
                self.store.publish(identity, {'rvm.safetensors': file}, record, check=self.check,
                                   max_metadata_bytes=MAX_METADATA_BYTES, max_payload_bytes=self._payload_bound(data),
                                   target_max_entry_files=3)
            except CacheCapacity:
                self.budget_rejected += 1
                return False
        self.stored += 1
        return True

    def snapshot(self):
        return {'enabled': True, 'parent_key': self.parent.key, 'chunk_hits': self.hits, 'chunk_misses': self.misses,
                'frames_reused': self.frames_reused, 'stored_chunks': self.stored, 'corrupt_entries': self.corrupt,
                'budget_rejected_chunks': self.budget_rejected, 'bytes_loaded': self.bytes_loaded,
                'last_error': self.last_error, 'feature_dtype': 'normalized float16 (existing prepass precision)',
                'state_dtype': 'exact producer dtype; no conversion',
                'scope': 'chronological RVM prepass only; Hann/carried hints/output restart',
                'validation': 'CPU fixtures; pretrained GPU parity/performance NOT VERIFIED'}
