"""Explicit persistent normalized float32 MoGe/C-RADIO frames; never RVM state."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .durable_context import BoundModelCache
from .identities import file_sha256
from .recovery_identity import feature_identity
from .recovery_store import CacheCapacity, CacheInvalid
from .resources import close_mapping


class DurableFeatureFrames(BoundModelCache):
    def __init__(self, store, parent_identity, *, validate_context, check=None):
        super().__init__(store, parent_identity, validate_context=validate_context, check=check)
        self.hits = 0
        self.misses = 0
        self.corrupt = 0
        self.stored = 0
        self.budget_rejected = 0
        self.bytes_loaded = 0
        self.last_error = None

    def _identity(self, frame, proxy, channels, grid_hw):
        self._require_bound()
        if (not isinstance(proxy, torch.Tensor) or proxy.ndim != 3 or proxy.shape[0] != 3 or
                proxy.dtype not in (torch.float32, torch.bfloat16) or
                not bool(torch.isfinite(proxy).all()) or bool(((proxy < 0) | (proxy > 1)).any())):
            raise ValueError('IID proxy must be finite CHW RGB float32/bfloat16 in [0,1]')
        host = proxy.detach().to('cpu').contiguous()
        digest = hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()
        return feature_identity(self.parent, frame, digest, tuple(proxy.shape), channels, grid_hw, str(proxy.dtype))

    def get(self, frame, proxy, channels, grid_hw):
        identity = self._identity(frame, proxy, channels, grid_hw)
        manifest = None
        try:
            manifest = self.store.lookup(identity, check=self.check)
            if manifest is None:
                self.misses += 1
                return None
            expected = identity.to_dict()['channels']
            files = {item['name']: item for item in manifest['files']}
            if set(files) != {name + '.npy' for name in expected}:
                raise CacheInvalid('persistent feature channel inventory differs')
            result = {}
            loaded_bytes = 0
            for name, width in expected.items():
                if self.check is not None:
                    self.check()
                file = self.store.root / 'entries' / identity.key / files[name + '.npy']['blob']
                array = np.load(file, allow_pickle=False, mmap_mode='r')
                if not isinstance(array, np.memmap):
                    if hasattr(array, 'close'):
                        array.close()
                    raise CacheInvalid('persistent feature is not a single mmap-compatible array')
                try:
                    if array.dtype != np.float32 or tuple(array.shape) != (width, *grid_hw):
                        raise CacheInvalid('persistent feature dtype/shape differs from exact normalized float32 contract')
                    if not np.isfinite(array).all():
                        raise CacheInvalid('persistent features contain nonfinite values')
                    copied = np.array(array, copy=True)
                    result[name] = torch.from_numpy(copied)
                    loaded_bytes += copied.nbytes
                finally:
                    close_mapping(array)
                if file_sha256(file, check=self.check) != files[name + '.npy']['sha256']:
                    raise CacheInvalid('persistent feature changed while loading')
            self.hits += 1
            self.bytes_loaded += loaded_bytes
            return result
        except (CacheInvalid, OSError, ValueError, EOFError) as exc:
            self.corrupt += 1
            self.misses += 1
            self.last_error = str(exc)
            self.store.discard(identity)
            return None

    def put(self, frame, proxy, channels, grid_hw, features, *, producer_frames):
        identity = self._identity(frame, proxy, channels, grid_hw)
        expected = identity.to_dict()['channels']
        if set(features) != set(expected):
            raise ValueError('only exact declared IID feature names may be persisted')
        minimum_bytes = 0
        for name, width in expected.items():
            value = features[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                    tuple(value.shape) != (width, *grid_hw) or not bool(torch.isfinite(value).all())):
                raise ValueError('persistent feature producer must return finite normalized float32 CHW tensors')
            minimum_bytes += value.numel() * value.element_size()
        if minimum_bytes > self.store.max_bytes:
            self.budget_rejected += 1
            return False
        with self.store.private_stage() as folder:
            files = {}
            for index, (name, value) in enumerate(features.items()):
                if self.check is not None:
                    self.check()
                array = value.detach().to(device='cpu', copy=True).contiguous().numpy()
                file = folder / f'payload-{index:06d}.bin'
                with file.open('xb') as stream:
                    np.save(stream, array, allow_pickle=False)
                files[name + '.npy'] = file
            record = {'stage': 'independent_moge_cradio_normalized_float32', 'parent_identity': self.parent.to_dict(),
                      'producer_frames': list(producer_frames), 'dtype': 'float32', 'rvm_state': 'excluded'}
            try:
                self.store.publish(identity, files, record, check=self.check)
            except CacheCapacity:
                self.budget_rejected += 1
                return False
        self.stored += 1
        return True

    def snapshot(self):
        return {'enabled': True, 'parent_key': self.parent.key, 'hits': self.hits, 'misses': self.misses,
                'corrupt_entries': self.corrupt, 'stored_frames': self.stored, 'budget_rejected_frames': self.budget_rejected,
                'bytes_loaded': self.bytes_loaded, 'last_error': self.last_error, 'dtype': 'normalized float32',
                'scope': 'IID C-RADIO/MoGe only; no RVM/baton/recurrent resume',
                'numerical_validation': 'CPU fixtures only; real GPU producer batch-roundoff/parity unverified'}
