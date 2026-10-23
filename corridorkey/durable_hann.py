"""Verified Stage1 events replayed to reconstruct Hann state in timeline order."""
from __future__ import annotations

import hashlib

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

from .durable_context import BoundModelCache
from .durable_rvm import tensor_digest
from .hann_identity import HannWindowIdentity, hann_window_identity
from .recovery_identity import canonical
from .recovery_store import CacheCapacity, CacheInvalid

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

    def _record(self, tensors):
        return {'stage': 'hann_stage1_raw_event', 'parent_identity': self.parent.to_dict(),
                'tensors': {name: {'dtype': str(value.dtype), 'shape': list(value.shape)}
                            for name, value in tensors.items()},
                'replay_policy': 'chronological_prefix_reconstruction', 'writer_resume': False}

    def get(self, identity):
        data = self._contract(identity)
        try:
            manifest = self.store.lookup(identity, check=self.check)
            if manifest is None:
                self.misses += 1
                return None
            files = manifest['files']
            if len(files) != 1 or files[0]['name'] != 'hann.safetensors':
                raise CacheInvalid('Hann event payload inventory differs')
            blob = (self.store._entry(identity) / files[0]['blob']).read_bytes()
            if len(blob) != files[0]['bytes'] or hashlib.sha256(blob).hexdigest() != files[0]['sha256']:
                raise CacheInvalid('Hann event changed while loading')
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
        except (CacheInvalid, OSError, ValueError, SafetensorError) as exc:
            self.corrupt += 1
            self.misses += 1
            self.last_error = str(exc)
            self.store.discard(identity)
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
        if minimum_bytes > self.store.max_bytes:
            self.budget_rejected += 1
            return False
        independent = {name: value.detach().to(device='cpu', copy=True).contiguous() for name, value in tensors.items()}
        blob = save(independent)
        if self.check is not None:
            self.check()
        with self.store.private_stage() as folder:
            file = folder / 'payload-000000.bin'
            file.write_bytes(blob)
            try:
                self.store.publish(identity, {'hann.safetensors': file}, self._record(tensors), check=self.check)
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
