"""Bounded exact selected Stage2 tensors; chronological rendering still restarts."""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

from .durable_context import BoundModelCache
from .durable_rvm import tensor_digest
from .recovery_identity import canonical
from .recovery_store import CacheCapacity, CacheInvalid
from .stage2_identity import DTYPE_NAMES, Stage2BatchIdentity, stage2_batch_identity

MAX_TENSOR_BYTES = 256 * 1024 * 1024
MAX_HEADER_BYTES = 65536
MAX_METADATA_BYTES = 4 * 1024 * 1024
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
SIZES = {'torch.float32': 4, 'torch.bfloat16': 2, 'torch.float16': 2}
CODES = {'torch.float32': 'F32', 'torch.bfloat16': 'BF16', 'torch.float16': 'F16'}


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate prediction header key')
        result[name] = value
    return result


class DurableStage2Batches(BoundModelCache):
    def __init__(self, store, parent_identity, *, validate_context, check=None):
        super().__init__(store, parent_identity, validate_context=validate_context, check=check)
        self.hits = self.misses = self.frames_reused = self.stored = self.corrupt = 0
        self.budget_rejected = self.bytes_loaded = 0
        self.last_error = None
        self.hash_seconds = self.load_seconds = self.store_seconds = 0.0

    def identity(self, *, frames, policy, rgb, baton, autocast_dtype='torch.bfloat16'):
        self._require_bound()
        if (not isinstance(rgb, torch.Tensor) or not isinstance(baton, torch.Tensor)
                or rgb.dtype not in DTYPES or baton.dtype not in DTYPES
                or rgb.ndim != 5 or baton.ndim != 5 or rgb.device != baton.device):
            raise ValueError('Stage2 inputs require same-device floating BTCHW tensors')
        if not isinstance(frames, (tuple, list)) or not 1 <= len(frames) <= 4096:
            raise ValueError('Stage2 requires a bounded whole-batch frame sequence')
        execution = {'backend': 'pytorch', 'device': str(rgb.device), 'autocast_dtype': autocast_dtype}
        shapes = {name: {'dtype': str(tensor.dtype), 'shape': list(tensor.shape), 'sha256': '0' * 64}
                  for name, tensor in (('rgb', rgb), ('baton', baton))}
        # Validate schema/geometry before any identity-related host allocation.
        stage2_batch_identity(self.parent, frames=frames, policy=policy, execution=execution, inputs=shapes)
        # Bound GPU→host hashing copies as well as predicted output staging.
        input_bytes = rgb.numel() * rgb.element_size() + baton.numel() * baton.element_size()
        minimum_output = rgb.shape[0] * rgb.shape[1] * 4 * rgb.shape[-2] * rgb.shape[-1] * 2
        if input_bytes > MAX_TENSOR_BYTES or minimum_output > min(MAX_TENSOR_BYTES, self.store.max_bytes):
            self.budget_rejected += 1
            return None
        started = time.perf_counter()
        try:
            descriptions = {}
            for name, tensor in (('rgb', rgb), ('baton', baton)):
                if self.check is not None:
                    self.check()
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError('Stage2 model inputs must be finite')
                descriptions[name] = {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                                      'sha256': tensor_digest(tensor)}
            return stage2_batch_identity(self.parent, frames=frames, policy=policy,
                execution=execution,
                inputs=descriptions)
        finally:
            self.hash_seconds += time.perf_counter() - started

    def _contract(self, identity):
        self._require_bound()
        if not isinstance(identity, Stage2BatchIdentity) or identity.to_dict()['parent_key'] != self.parent.key:
            raise CacheInvalid('Stage2 batch belongs to another typed job context')
        return identity.to_dict()

    @staticmethod
    def _shape(data):
        return (1, len(data['frames']), 4, *data['inputs']['rgb']['shape'][-2:])

    def _validate(self, data, tensor):
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype not in DTYPES
                or tuple(tensor.shape) != self._shape(data) or not bool(torch.isfinite(tensor).all())):
            raise ValueError('Stage2 prediction requires finite exact producer dtype/geometry')

    def _record(self, tensor):
        return {'stage': 'stage2_selected_raw_prediction', 'parent_identity': self.parent.to_dict(),
                'tensor': {'dtype': str(tensor.dtype), 'shape': list(tensor.shape)},
                'selection': 'tuple_first_or_tensor', 'postprocessing': False,
                'replay_policy': 'whole_exact_batch_chronological_reconstruction', 'writer_resume': False}

    def _decode_header(self, blob, data, record):
        if len(blob) < 8:
            raise CacheInvalid('Truncated Stage2 prediction header')
        size = int.from_bytes(blob[:8], 'little')
        if not 1 <= size <= MAX_HEADER_BYTES or 8 + size > len(blob):
            raise CacheInvalid('Stage2 prediction header exceeds bound')
        header = json.loads(blob[8:8 + size], object_pairs_hook=_object)
        if not isinstance(header, dict) or set(header) != {'prediction'}:
            raise CacheInvalid('Stage2 prediction tensor inventory differs')
        tensor = header['prediction']
        description = record.get('tensor') if isinstance(record, dict) else None
        if (not isinstance(description, dict) or set(description) != {'dtype', 'shape'}
                or description['dtype'] not in DTYPE_NAMES):
            raise CacheInvalid('Stage2 producer dtype is missing')
        dtype = description['dtype']
        shape = self._shape(data)
        elements = 1
        for dimension in shape:
            elements *= dimension
        extent = elements * SIZES[dtype]
        if (extent > MAX_TENSOR_BYTES or not isinstance(tensor, dict)
                or set(tensor) != {'dtype', 'shape', 'data_offsets'}
                or tensor['dtype'] != CODES[dtype] or tensor['shape'] != list(shape)
                or any(type(n) is not int for n in tensor['shape'])
                or not isinstance(tensor['data_offsets'], list)
                or any(type(n) is not int for n in tensor['data_offsets'])
                or description['shape'] != list(shape) or tensor['data_offsets'] != [0, extent]
                or len(blob) != 8 + size + extent):
            raise CacheInvalid('Stage2 prediction encoded extents differ')

    def get(self, identity):
        data = self._contract(identity)
        started = time.perf_counter()
        entry = self.store._entry(identity)
        if entry.exists():
            # Unknown/oversize ownership metadata cannot authorize deletion.
            for name in ('owner.json', 'complete.json'):
                if (entry / name).stat().st_size > MAX_METADATA_BYTES:
                    raise CacheInvalid('Stage2 completion metadata exceeds safe ownership bound')
        try:
            if entry.exists():
                for index, path in enumerate(entry.iterdir()):
                    if self.check is not None:
                        self.check()
                    if index >= 3:
                        raise CacheInvalid('Unexpected Stage2 entry file count; manual inspection required')
                    if path.name.startswith('payload-') and path.stat().st_size > MAX_TENSOR_BYTES + MAX_HEADER_BYTES + 8:
                        raise CacheInvalid('Stage2 prediction file exceeds load bound')
            manifest = self.store.lookup(identity, check=self.check, max_metadata_bytes=MAX_METADATA_BYTES, max_entry_files=4096, max_payload_bytes=MAX_TENSOR_BYTES + MAX_HEADER_BYTES + 8)
            if manifest is None:
                self.misses += 1
                return None
            files = manifest['files']
            if (len(files) != 1 or files[0]['name'] != 'prediction.safetensors'
                    or not 8 < files[0]['bytes'] <= MAX_TENSOR_BYTES + MAX_HEADER_BYTES + 8):
                raise CacheInvalid('Stage2 prediction payload inventory or size differs')
            with (entry / files[0]['blob']).open('rb') as stream:
                before = os.fstat(stream.fileno())
                blob = stream.read(files[0]['bytes'] + 1)
                after = os.fstat(stream.fileno())
            if (len(blob) != files[0]['bytes'] or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                    or hashlib.sha256(blob).hexdigest() != files[0]['sha256']):
                raise CacheInvalid('Stage2 prediction changed while loading')
            self._decode_header(blob, data, manifest.get('origin_record'))
            if self.check is not None:
                self.check()
            tensors = load(blob)
            prediction = tensors['prediction']
            self._validate(data, prediction)
            if canonical(manifest.get('origin_record')) != canonical(self._record(prediction)):
                raise CacheInvalid('Stage2 prediction provenance differs')
            self.hits += 1
            self.frames_reused += len(data['frames'])
            self.bytes_loaded += len(blob)
            return prediction
        except (CacheInvalid, OSError, ValueError, SafetensorError, RecursionError) as exc:
            self.corrupt += 1
            self.misses += 1
            self.last_error = str(exc)[:512]
            self.store.discard(identity, max_metadata_bytes=MAX_METADATA_BYTES, max_entry_files=4096)
            return None
        finally:
            self.load_seconds += time.perf_counter() - started

    def put(self, identity, prediction):
        data = self._contract(identity)
        self._validate(data, prediction)
        size = prediction.numel() * prediction.element_size()
        record = self._record(prediction)
        metadata_size = len(canonical(record).encode()) + len(identity.payload.encode()) + 4096
        if size > min(MAX_TENSOR_BYTES, self.store.max_bytes) or metadata_size > MAX_METADATA_BYTES:
            self.budget_rejected += 1
            return False
        started = time.perf_counter()
        try:
            if self.check is not None:
                self.check()
            independent = prediction.detach().to(device='cpu', copy=True).contiguous()
            blob = save({'prediction': independent})
            if len(blob) > MAX_TENSOR_BYTES + MAX_HEADER_BYTES + 8:
                self.budget_rejected += 1
                return False
            if self.check is not None:
                self.check()
            with self.store.private_stage(max_metadata_bytes=MAX_METADATA_BYTES, max_entry_files=4096) as folder:
                path = folder / 'payload-000000.bin'
                path.write_bytes(blob)
                try:
                    self.store.publish(identity, {'prediction.safetensors': path}, record, check=self.check, max_metadata_bytes=MAX_METADATA_BYTES, max_entry_files=4096, max_payload_bytes=MAX_TENSOR_BYTES + MAX_HEADER_BYTES + 8)
                except CacheCapacity:
                    self.budget_rejected += 1
                    return False
            self.stored += 1
            return True
        finally:
            self.store_seconds += time.perf_counter() - started

    def snapshot(self):
        return {'enabled': True, 'parent_key': self.parent.key,
                'parent_kind': self.parent.to_dict()['kind'], 'batch_hits': self.hits, 'batch_misses': self.misses,
                'stage2_frame_instances_reused': self.frames_reused, 'stored_batches': self.stored,
                'corrupt_entries': self.corrupt, 'budget_rejected_batches': self.budget_rejected,
                'bytes_loaded': self.bytes_loaded, 'last_error': self.last_error,
                'hash_seconds': self.hash_seconds, 'load_seconds': self.load_seconds, 'store_seconds': self.store_seconds,
                'precision': 'exact selected producer dtype/bits before external refinement/corrections/render',
                'scope': 'whole exact Stage2 batch replay; upstream reconstruction, render and writers restart; no direct seek',
                'validation': 'CPU fixtures; pretrained GPU parity/performance NOT VERIFIED'}
