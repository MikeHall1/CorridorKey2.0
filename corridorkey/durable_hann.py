"""Verified Stage1 events replayed to reconstruct Hann state in timeline order."""
from __future__ import annotations

import hashlib
import math

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

from .durable_context import BoundModelCache
from .durable_rvm import (
    DTYPE_CODES,
    MAX_HEADER_BYTES,
    MAX_METADATA_BYTES,
    tensor_digest,
    validate_tensor_header,
)
from .hann_identity import HannWindowIdentity, hann_window_identity
from .recovery_identity import canonical
from .recovery_store import CacheCapacity, CacheInvalid, read_payload_bounded

DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def replay_state_digest(*, carry, accumulators, weights, pending_frames, previous_segment, check=None):
    """Bind every live value reconstructed by replay, without serializing aliases."""
    if set(accumulators) != set(weights):
        raise ValueError('Hann accumulator and weight inventories differ')
    records = {}
    for name, values in (('carry', carry), ('accumulators', accumulators), ('pending_frames', pending_frames)):
        if any(type(frame) is not int or frame < 0 for frame in values):
            raise ValueError('Hann replay frame keys must be nonnegative integers')
        items = []
        for frame, value in sorted(values.items()):
            if check is not None:
                check()
            tensor = torch.as_tensor(value)
            if tensor.dtype not in DTYPES or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Hann replay tensors must be finite floating values')
            items.append([frame, tensor_digest(tensor)])
        records[name] = items
    if any(type(value) not in (int, float) or value < 0 for value in weights.values()):
        raise ValueError('Hann replay weights must be finite nonnegative numbers')
    records['weights'] = [[frame, value] for frame, value in sorted(weights.items())]
    records['previous_segment'] = previous_segment
    return hashlib.sha256(canonical(records).encode()).hexdigest()


class DurableHannWindows(BoundModelCache):
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

    def identity(self, *, schedule, incoming_replay_sha256, index, start, stop, next_start,
                 segment, carry, d_16, inputs):
        self._require_bound()
        descriptions = {}
        for name, value in inputs.items():
            if self.check is not None:
                self.check()
            if value is None:
                descriptions[name] = None
                continue
            if (not isinstance(value, torch.Tensor) or value.dtype not in DTYPES or
                    not bool(torch.isfinite(value).all())):
                raise ValueError('Hann model inputs must be finite floating tensors')
            descriptions[name] = {'dtype': str(value.dtype), 'shape': list(value.shape), 'sha256': tensor_digest(value)}
        return hann_window_identity(
            self.parent, schedule_sha256=hashlib.sha256(canonical(schedule).encode()).hexdigest(),
            incoming_replay_sha256=incoming_replay_sha256, index=index, start=start, stop=stop,
            next_start=next_start, segment=segment, carry=carry, d_16=d_16, inputs=descriptions)

    def _contract(self, identity):
        self._require_bound()
        if not isinstance(identity, HannWindowIdentity) or identity.to_dict()['parent_key'] != self.parent.key:
            raise CacheInvalid('Hann event belongs to another typed job context')
        return identity.to_dict()

    def _validate(self, data, tensors):
        expected = {'baton', 'aux'} if data['carry'] else {'baton'}
        if set(tensors) != expected:
            raise ValueError('Hann event tensor inventory differs from carry policy')
        count = data['stop'] - data['start']
        height, width = data['inputs']['rgb']['shape'][-2:]
        shapes = {'baton': (1, count, data['d_16'], height // 16, width // 16),
                  'aux': (1, count, 1, height // 2, width // 2)}
        for name, value in tensors.items():
            if (not isinstance(value, torch.Tensor) or value.dtype not in DTYPES or
                    tuple(value.shape) != shapes[name] or not bool(torch.isfinite(value).all())):
                raise ValueError('Hann event must retain finite exact producer precision/geometry')

    def _shapes(self, data):
        count = data['stop'] - data['start']
        height, width = data['inputs']['rgb']['shape'][-2:]
        shapes = {'baton': [1, count, data['d_16'], height // 16, width // 16]}
        if data['carry']:
            shapes['aux'] = [1, count, 1, height // 2, width // 2]
        return shapes

    def _payload_bound(self, data):
        return min(self.store.max_bytes, 8 + MAX_HEADER_BYTES
                   + sum(math.prod(shape) * 4 for shape in self._shapes(data).values()))

    def _record_schema(self, descriptions):
        return {'stage': 'hann_stage1_raw_event', 'parent_identity': self.parent.to_dict(),
                'tensors': descriptions, 'replay_policy': 'chronological_prefix_reconstruction', 'writer_resume': False}

    def _record(self, tensors):
        return self._record_schema({name: {'dtype': str(value.dtype), 'shape': list(value.shape)}
                                    for name, value in tensors.items()})

    def _descriptions(self, data, record):
        descriptions = record.get('tensors') if isinstance(record, dict) else None
        shapes = self._shapes(data)
        if not isinstance(descriptions, dict) or set(descriptions) != set(shapes):
            raise CacheInvalid('Hann event tensor provenance differs')
        for name, shape in shapes.items():
            item = descriptions[name]
            if (not isinstance(item, dict) or set(item) != {'dtype', 'shape'}
                    or not isinstance(item['dtype'], str) or item['dtype'] not in DTYPE_CODES
                    or item['shape'] != shape or any(type(n) is not int for n in item['shape'])):
                raise CacheInvalid('Hann event tensor provenance shape/dtype differs')
        if canonical(record) != canonical(self._record_schema(descriptions)):
            raise CacheInvalid('Hann event provenance differs from exact producer contract')
        return descriptions

    def get(self, identity):
        data = self._contract(identity)
        try:
            manifest = self.store.lookup(identity, check=self.check, max_metadata_bytes=MAX_METADATA_BYTES,
                                         max_entry_files=3, max_payload_bytes=self._payload_bound(data))
            if manifest is None:
                self.misses += 1
                return None
            files = manifest['files']
            if len(files) != 1 or files[0]['name'] != 'hann.safetensors':
                raise CacheInvalid('Hann event payload inventory differs')
            descriptions = self._descriptions(data, manifest.get('origin_record'))
            blob = read_payload_bounded(self.store._entry(identity) / files[0]['blob'],
                                        expected_size=files[0]['bytes'], expected_sha256=files[0]['sha256'],
                                        max_bytes=self._payload_bound(data), check=self.check)
            validate_tensor_header(blob, descriptions)
            if self.check is not None:
                self.check()
            tensors = load(blob)
            self._validate(data, tensors)
            if canonical(manifest.get('origin_record')) != canonical(self._record(tensors)):
                raise CacheInvalid('Hann event provenance differs from exact producer contract')
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

    def put(self, identity, outputs):
        data = self._contract(identity)
        if outputs.get('recurrent_states') is not None:
            raise ValueError('Hann event reuse requires disabled Stage1 recurrent sites')
        tensors = {'baton': outputs['baton']}
        if data['carry']:
            tensors['aux'] = outputs['aux'][:, :, 0:1]
        self._validate(data, tensors)
        minimum_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
        record = self._record(tensors)
        metadata_bytes = len(canonical(record).encode()) + len(identity.payload.encode()) + 4096
        if minimum_bytes > self.store.max_bytes or metadata_bytes > MAX_METADATA_BYTES:
            self.budget_rejected += 1
            return False
        # A lower bound on the actual sealed entry also avoids staging tensors
        # when even data plus required metadata cannot fit this byte budget.
        lower_manifest = {'schema_version': 1, 'key': identity.key, 'identity': identity.to_dict(),
                          'files': [{'name': 'hann.safetensors', 'blob': 'payload-000000.bin',
                                     'bytes': minimum_bytes, 'sha256': '0' * 64}], 'origin_record': record}
        lower_owner = {'owner': self.store.owner, 'manifest_sha256': '0' * 64}
        if minimum_bytes + len(canonical(lower_manifest).encode()) + len(canonical(lower_owner).encode()) > self.store.max_bytes:
            self.budget_rejected += 1
            return False
        independent = {name: value.detach().to(device='cpu', copy=True).contiguous() for name, value in tensors.items()}
        blob = save(independent)
        if self.check is not None:
            self.check()
        if len(blob) > self._payload_bound(data):
            self.budget_rejected += 1
            return False
        with self.store.private_stage(max_metadata_bytes=MAX_METADATA_BYTES, max_entry_files=3) as folder:
            file = folder / 'payload-000000.bin'
            file.write_bytes(blob)
            try:
                self.store.publish(identity, {'hann.safetensors': file}, record, check=self.check,
                                   max_metadata_bytes=MAX_METADATA_BYTES, max_payload_bytes=self._payload_bound(data),
                                   target_max_entry_files=3)
            except CacheCapacity:
                self.budget_rejected += 1
                return False
        self.stored += 1
        return True

    def snapshot(self):
        return {'enabled': True, 'parent_key': self.parent.key, 'window_hits': self.hits, 'window_misses': self.misses,
                'stage1_frame_instances_reused': self.frames_reused, 'stored_windows': self.stored,
                'corrupt_entries': self.corrupt, 'budget_rejected_windows': self.budget_rejected,
                'bytes_loaded': self.bytes_loaded, 'last_error': self.last_error,
                'precision': 'exact producer dtype; raw baton and auxiliary alpha only',
                'scope': 'replay Hann windows in order to reconstruct carry/overlap; Stage2/writers restart',
                'validation': 'CPU fixtures; pretrained GPU parity/performance NOT VERIFIED'}
