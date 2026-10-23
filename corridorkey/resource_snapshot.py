"""Explicit sampled resources; never initialize CUDA or infer a job memory limit."""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


def _peak_rss() -> dict[str, Any]:
    try:
        import resource
        scale = {'darwin': 1, 'linux': 1024}.get(sys.platform)
        if scale is None:
            raise ValueError('Peak RSS units unsupported on this platform')
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
        if value < 0:
            raise ValueError('Negative peak RSS')
        return {'peak_rss_bytes': value, 'scope': 'Process lifetime peak RSS; not current RSS or a RAM limit.'}
    except (ImportError, OSError, ValueError) as exc:
        return {'peak_rss_bytes': None, 'scope': f'Peak RSS NOT VERIFIED: {type(exc).__name__}: {exc}'}


def _gpu() -> dict[str, Any]:
    # A status request must not import Torch or create a CUDA context. Normal
    # inference remains the only engine path that chooses/initializes a backend.
    module = sys.modules.get('torch')
    cuda = getattr(module, 'cuda', None)
    try:
        if cuda is None or not cuda.is_initialized():
            return {'state': 'not initialized', 'allocated_bytes': None, 'reserved_bytes': None,
                    'peak_allocated_bytes': None, 'scope': 'CUDA allocator NOT VERIFIED; no context initialized by this snapshot.'}
        device = cuda.current_device()
        return {'state': 'sampled', 'device': int(device), 'allocated_bytes': int(cuda.memory_allocated(device)),
                'reserved_bytes': int(cuda.memory_reserved(device)), 'peak_allocated_bytes': int(cuda.max_memory_allocated(device)),
                'scope': 'PyTorch allocator on the current device only; excludes other processes/driver allocations. Unsynchronized samples, not total GPU memory or a limit.'}
    except (RuntimeError, ValueError, OSError, AttributeError) as exc:
        return {'state': 'unavailable', 'allocated_bytes': None, 'reserved_bytes': None,
                'peak_allocated_bytes': None, 'scope': f'CUDA allocator NOT VERIFIED: {type(exc).__name__}: {exc}'}


def process_snapshot() -> dict[str, Any]:
    return {'sampled_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'pid': os.getpid(),
            'process': _peak_rss(), 'gpu': _gpu(),
            'scope': 'One worker process snapshot, not its descendants or whole-machine usage. No synchronization, model load or memory enforcement.'}


def disk_snapshot(requested: Path) -> dict[str, Any]:
    requested = requested.resolve()
    existing = requested
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    if existing.is_file():
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return {'sampled_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'requested': str(requested),
            'measured_at': str(existing), 'total_bytes': usage.total, 'used_bytes': usage.used, 'free_bytes': usage.free,
            'scope': 'Containing filesystem space at sampling time. No directory scan, file creation, job-size estimate, reservation or guarantee that a render fits.'}
