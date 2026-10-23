"""Private canonical recovery-batch snapshots using the existing batch parser."""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .batch import load_jobs
from .diagnostics import write_json_atomic
from .local_json import read_object
from .session import JobRequest
from .worker_protocol import MAX_FRAME_BYTES, job_from_wire, job_to_wire


def separate(root: Path, paths: list[str | Path | None]) -> None:
    root = root.resolve()
    for value in paths:
        if value is None:
            continue
        path = Path(value).resolve()
        if path == root or path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError('Queue workspace must be separate from source/hints/corrections/outputs/cache/assets.')


def validate_job(root: Path, wire: dict[str, Any]) -> None:
    job = job_from_wire(wire)
    config, _ = job.configurations()
    separate(root, [job.input_path, job.output_dir, job.initial_hint_path, job.hint_video_path, config.corrections_path])


def _write(path: Path, data: dict[str, Any]) -> None:
    if len((json.dumps(data, indent=2, allow_nan=False) + '\n').encode()) > MAX_FRAME_BYTES:
        raise ValueError('Canonical batch exceeds the 4MiB snapshot limit.')
    write_json_atomic(path, data)


def capture_batch(root: Path, request_id: str, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    original = Path(payload['batch']).resolve()
    data, _ = read_object(original, max_bytes=MAX_FRAME_BYTES)
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1 or data.get('kind') != 'inference_batch':
        raise ValueError('Recovery batch requires schema_version integer1 and kind inference_batch.')
    if not isinstance(data.get('jobs'), list):
        raise TypeError('Recovery batch jobs must be a list.')
    def absolute(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Batch paths must be nonempty strings.')
        return str((original.parent / value).resolve())
    if data.get('weights') is not None:
        data['weights'] = absolute(data['weights'])
    for entry in data['jobs']:
        if not isinstance(entry, dict):
            raise TypeError('Batch job must be an object.')
        for key in ('input', 'output_dir', 'hint_first_frame', 'hint_video'):
            if entry.get(key) is not None:
                entry[key] = absolute(entry[key])
        settings = entry.get('settings', {})
        if not isinstance(settings, dict):
            raise TypeError('Batch settings must be an object.')
        for key in ('temp_dir', 'corrections_path'):
            if settings.get(key) is not None:
                settings[key] = absolute(settings[key])
    batches = root / 'batches'
    if batches.is_symlink():
        raise ValueError('Queue batch directory must not be a symlink.')
    batches.mkdir(exist_ok=True)
    stage = batches / f'.prepare-{uuid.uuid4().hex}.json'
    _write(stage, data)
    try:
        jobs, weights = load_jobs(stage)
    finally:
        stage.unlink(missing_ok=True)
    separate(root, [payload['workspace'], weights])
    for job in jobs:
        config, _ = job.configurations()
        if config.output_format != 'exr' or config.auto_vram:
            raise ValueError('Recovery requires EXR and explicit scheduling (auto_vram=false).')
        validate_job(root, {'input_path': job.input_path, 'output_dir': job.output_dir, 'config_json': job.config_json,
                            'writer_json': job.writer_json, 'initial_hint_path': job.initial_hint_path,
                            'hint_video_path': job.hint_video_path, 'profile_stages': job.profile_stages})
    snapshot = batches / f'{request_id}.json'
    if snapshot.exists():
        raise ValueError('Queue recovery snapshot already exists.')
    canonical = {'schema_version': 1, 'kind': 'inference_batch', 'weights': str(weights) if weights else None,
                 'jobs': [_batch_entry(job) for job in jobs]}
    _write(snapshot, canonical)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    return {**payload, 'batch': str(snapshot.resolve())}, digest


def _batch_entry(job: JobRequest) -> dict[str, Any]:
    job = job_from_wire(job_to_wire(job))
    return {'input': job.input_path, 'output_dir': job.output_dir,
            'settings': json.loads(job.config_json), 'writer': json.loads(job.writer_json),
            'hint_first_frame': job.initial_hint_path, 'hint_video': job.hint_video_path,
            'profile_stages': job.profile_stages}
