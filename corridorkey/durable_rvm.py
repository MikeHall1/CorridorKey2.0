"""Atomic exact-call RVM features plus recurrent state; no Hann/output resume."""
from __future__ import annotations

import hashlib

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

from .durable_context import BoundModelCache
from .recovery_identity import canonical
from .recovery_store import CacheCapacity, CacheInvalid
from .recurrent_identity import RvmChunkIdentity, rvm_chunk_identity

STATE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


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

    def _record(self, state):
        return {'stage': 'rvm_exact_chronological_call', 'parent_identity': self.parent.to_dict(),
                'features_dtype': 'torch.float16', 'state_dtype': str(state.dtype),
                'state_shape': list(state.shape), 'hann_output_resume': False}

    def get(self, identity):
        data = self._contract(identity)
        try:
            manifest = self.store.lookup(identity, check=self.check)
            if manifest is None:
                self.misses += 1
                return None
            files = manifest['files']
            if len(files) != 1 or files[0]['name'] != 'rvm.safetensors':
                raise CacheInvalid('RVM checkpoint tensor inventory differs')
            file = self.store._entry(identity) / files[0]['blob']
            blob = file.read_bytes()
            if len(blob) != files[0]['bytes'] or hashlib.sha256(blob).hexdigest() != files[0]['sha256']:
                raise CacheInvalid('RVM checkpoint changed while loading')
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
        except (CacheInvalid, OSError, ValueError, SafetensorError) as exc:
            self.corrupt += 1
            self.misses += 1
            self.last_error = str(exc)
            self.store.discard(identity)
            return None

    def put(self, identity, features, state):
        data = self._contract(identity)
        self._validate(data, features, state)
        minimum_bytes = sum(value.numel() * value.element_size() for value in (features, state))
        if minimum_bytes > self.store.max_bytes:
            self.budget_rejected += 1
            return False
        tensors = {name: value.detach().to(device='cpu', copy=True).contiguous()
                   for name, value in (('features', features), ('state', state))}
        blob = save(tensors)
        if self.check is not None:
            self.check()
        with self.store.private_stage() as folder:
            file = folder / 'payload-000000.bin'
            file.write_bytes(blob)
            record = self._record(state)
            try:
                self.store.publish(identity, {'rvm.safetensors': file}, record, check=self.check)
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
