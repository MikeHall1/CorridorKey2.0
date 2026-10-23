"""Read-only local asset presence and isolated backend diagnostics."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_process import module_command
from .runtime_paths import RuntimePaths


@dataclass(frozen=True)
class AssetSelection:
    root: Path
    weights: Path | None = None


def inspect_assets(selection: AssetSelection) -> dict[str, Any]:
    paths = RuntimePaths.from_package(Path(__file__).resolve().parents[1], selection.weights, asset_root=selection.root)
    entries = []
    for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir'):
        path = getattr(paths, name)
        try:
            if name == 'cradio_dir':
                ready = path.is_dir() and next(path.iterdir(), None) is not None
                status = 'directory present; contents not validated' if ready else 'missing/wrong type/empty directory'
                size = None
            else:
                ready = path.is_file() and path.stat().st_size > 0
                status = 'nonempty file present' if ready else 'missing/wrong type/empty file'
                size = path.stat().st_size if ready else None
        except OSError as exc:
            ready, status, size = False, f'{type(exc).__name__}: {exc}', None
        entries.append({'asset': name, 'path': str(path), 'present': ready, 'bytes': size, 'status': status})
    return {'assets_present': all(entry['present'] for entry in entries), 'entries': entries,
            'scope': 'Presence/type/size only. Checkpoint format, tensor compatibility, contents, hashes, GPU inference and quality NOT VERIFIED. No model deserialized or download performed.'}


def inspect_backend() -> dict[str, Any]:
    # Probe in a bounded owned child, keeping CUDA initialization out of the GUI.
    try:
        result = subprocess.run(module_command('corridorkey.backend_probe'), cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=20, check=True)
        data = json.loads(result.stdout)
        if not isinstance(data, dict) or type(data.get('cuda_available')) is not bool:
            raise ValueError('Invalid backend diagnostic response')
        data['status'] = 'CUDA detected; inference NOT VERIFIED' if data['cuda_available'] else 'CUDA unavailable; production inference unsupported on this environment'
        data['memory'] = 'Live GPU memory NOT VERIFIED'
        return data
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'status': f'Backend NOT VERIFIED: {type(exc).__name__}: {exc}', 'cuda_available': None,
                'memory': 'Live GPU memory NOT VERIFIED'}
