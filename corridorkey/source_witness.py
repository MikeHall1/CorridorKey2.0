"""Opt-in witnesses of native scene-linear pixels actually consumed by a job."""
from __future__ import annotations

import hashlib
import math
import os
import struct
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .control import JobControl

DOMAIN = b'CorridorKey/source/linear-rgb-f32/v1\0'
CONTRACT = 'scene_linear_rgb_f32_le_v1'
PREFIX = struct.Struct('<8sIIIqd')
MAX_FRAMES = 16384
MAX_PIXELS = 16 * 1024 * 1024


class SourceWitness:
    def __init__(self, meta: dict[str, Any], control: JobControl):
        height, width = meta['native_hw']
        count, start, fps = meta['num_frames'], meta['start_frame'], meta['native_fps']
        if (any(type(value) is not int for value in (width, height, count, start))
                or min(width, height, count) < 1 or count > MAX_FRAMES or width * height > MAX_PIXELS
                or not 0 <= start < 2**63 or start + count > 2**63 or not math.isfinite(fps) or fps <= 0):
            raise ValueError('Source witness exceeds native geometry/range/timing bounds (max16384 frames).')
        paths = meta['frame_paths']
        if len(paths) != count or len(set(paths)) != count:
            raise ValueError('Source witness requires one distinct selected path per frame.')
        self.paths = {str(path): index for index, path in enumerate(paths)}
        self.width, self.height, self.count, self.start, self.fps = width, height, count, start, float(fps)
        self.control = control
        self.lock = threading.Lock()
        self.digests: dict[int, bytes] = {}
        self.sealed = False

    def observe_path(self, path: str, rgb: np.ndarray):
        self.control.check()
        index = self.paths.get(str(path))
        if index is None or rgb.dtype != np.float32 or rgb.shape != (self.height, self.width, 3):
            raise ValueError('Witness requires selected native float32 RGB pixels.')
        if not np.isfinite(rgb).all():
            raise ValueError('Nonfinite source pixels cannot enter a source witness.')
        digest = hashlib.sha256(DOMAIN + struct.pack('<III', index, self.width, self.height))
        # Avoid a second complete frame byte copy while retaining canonical LE order.
        canonical = np.ascontiguousarray(rgb, dtype='<f4')
        digest.update(memoryview(canonical).cast('B'))
        value = digest.digest()
        with self.lock:
            if self.sealed:
                raise RuntimeError('Source witness was finalized before its readers stopped.')
            previous = self.digests.get(index)
            if previous is not None and previous != value:
                raise ValueError(f'Source frame {self.start + index} changed between processing reads.')
            self.digests[index] = value
        self.control.check()

    def finish(self, output: Path) -> dict[str, Any]:
        self.control.checkpoint('source_witness_publish', self.count, self.count)
        with self.lock:
            if self.sealed or len(self.digests) != self.count:
                raise ValueError('Source witness requires all selected frames and a single finalization.')
            raw = PREFIX.pack(b'CKSRC001', self.width, self.height, self.count, self.start, self.fps)
            raw += b''.join(self.digests[index] for index in range(self.count))
            self.sealed = True
        path = output / f'source-witness-{uuid.uuid4().hex}.bin'
        # A failed/partial file never gains completed-record metadata; no overwrite.
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return {'schema_version': 1, 'contract': CONTRACT, 'path': str(path.resolve()),
                'sha256': hashlib.sha256(raw).hexdigest(), 'width': self.width, 'height': self.height,
                'count': self.count, 'start_frame': self.start, 'fps': self.fps,
                'scope': 'All selected native RGB frames observed during processing; exact float32 bits, assumed scene-linear primaries. Not model/settings authentication.'}


def load_witness(record: dict[str, Any], *, width: int, height: int, count: int, start: int, fps: float) -> tuple[bytes, Path] | None:
    """Read a bounded producer sidecar; never manufacture provenance at seal time."""
    diagnostics = record['diagnostics']
    metadata = diagnostics.get('source_witness')
    settings = record.get('requested_settings')
    requested = isinstance(settings, dict) and settings.get('source_witness') is True
    if metadata is None and not requested:
        return None
    if (not isinstance(settings, dict) or not requested or not isinstance(metadata, dict) or type(metadata.get('schema_version')) is not int
            or metadata['schema_version'] != 1 or metadata.get('contract') != CONTRACT
            or settings.get('frame_dir_linear') is not True or settings.get('frame_dir_alpha') != 'ignore'
            or settings.get('source_colour_profile', 'legacy') != 'legacy'):
        raise ValueError('Bound caches require an explicit completed producer source witness.')
    if (count > MAX_FRAMES or start + count > 2**63 or any(type(metadata.get(name)) is not int or metadata[name] != expected
                                for name, expected in [('width', width), ('height', height), ('count', count), ('start_frame', start)])
            or type(metadata.get('fps')) not in (int, float) or metadata['fps'] != fps):
        raise ValueError('Producer witness geometry/range/timing differs from completed output.')
    name, output = metadata.get('path'), record.get('output_dir')
    if not isinstance(name, str) or not isinstance(output, str):
        raise TypeError('Producer witness requires an owned output path.')
    path = Path(name)
    if path.is_symlink() or path.resolve().parent != Path(output).resolve():
        raise ValueError('Producer witness must remain in its recorded output directory.')
    with path.open('rb') as stream:
        raw = stream.read(PREFIX.size + MAX_FRAMES * 32 + 1)
    prefix = PREFIX.pack(b'CKSRC001', width, height, count, start, fps)
    if (len(raw) != PREFIX.size + count * 32 or raw[:PREFIX.size] != prefix
            or hashlib.sha256(raw).hexdigest() != metadata.get('sha256')):
        raise ValueError('Producer source witness is incomplete, changed or incompatible.')
    return raw[PREFIX.size:], path
