"""Validated desktop project snapshots; no Qt or model imports."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .diagnostics import write_json_atomic
from .session import JobRequest
from .worker_protocol import MAX_FRAME_BYTES, decode_frame, job_from_wire, job_to_wire

DESKTOP_MAX_PREVIEW_FRAME = 1_000_000_000


def settings_job(source: str, destination: str, config: str, writer: str, *,
                 first_hint: str = '', full_hint: str = '', profile: bool = False) -> JobRequest:
    if not source.strip() or not destination.strip():
        raise ValueError('Select an input and an output parent first.')
    # Reuse the strict wire codec, including embedded configuration validation.
    job = JobRequest(str(Path(source).resolve()), str(Path(destination).resolve()), config, writer,
                     str(Path(first_hint).resolve()) if first_hint.strip() else None,
                     str(Path(full_hint).resolve()) if full_hint.strip() else None, profile)
    normalized = job_from_wire(job_to_wire(job_from_wire(asdict(job))))
    settings, _ = normalized.configurations()
    normalized.validate_source(settings)
    return normalized


def save_project(path: Path, job: JobRequest, frame: int = 0) -> None:
    if type(frame) is not int or frame < 0:
        raise ValueError('Preview frame must be a nonnegative integer.')
    path = path.resolve()
    settings, _ = job.configurations()
    protected = [job.input_path, job.initial_hint_path, job.hint_video_path, settings.corrections_path]
    if settings.corrections_path is not None:
        from .corrections import MAX_CORRECTION_BYTES, CorrectionDocument
        from .local_json import read_object
        correction_path = Path(settings.corrections_path).resolve()
        data, _ = read_object(correction_path, max_bytes=MAX_CORRECTION_BYTES, regular_file=True)
        document = CorrectionDocument.parse(data, correction_path.parent)
        protected.extend(document.asset_paths())
    for item in protected:
        if item is None:
            continue
        source = Path(item).resolve()
        if path == source or (source.is_dir() and path.is_relative_to(source)):
            raise ValueError('A project must be saved separately from source, hints and corrections.')
        if path.exists() and source.exists() and path.samefile(source):
            raise ValueError('A project must be saved separately from source, hint and correction asset aliases.')
    record = {'schema_version': 1, 'type': 'corridorkey_desktop',
              'job': job_to_wire(job), 'preview_frame': frame}
    if len((json.dumps(record, indent=2, allow_nan=False) + '\n').encode('utf-8')) > MAX_FRAME_BYTES:
        raise ValueError('Desktop project exceeds the size limit.')
    write_json_atomic(path, record)


def load_project(path: Path) -> tuple[JobRequest, int]:
    with path.open('rb') as stream:
        # Bound untrusted project allocation before parsing.
        raw = stream.read(MAX_FRAME_BYTES + 1)
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError('Desktop project exceeds the size limit.')
    data: dict[str, Any] = decode_frame(raw.rstrip(b'\n') + b'\n')
    if set(data) != {'schema_version', 'type', 'job', 'preview_frame'}:
        raise ValueError('Invalid desktop project fields.')
    if type(data['schema_version']) is not int or data['schema_version'] != 1 or data['type'] != 'corridorkey_desktop':
        raise ValueError('Unsupported desktop project version/type.')
    frame = data['preview_frame']
    if type(frame) is not int or frame < 0:
        raise ValueError('Invalid preview frame.')
    return job_from_wire(data['job']), frame


def pretty_settings(job: JobRequest) -> tuple[str, str]:
    wire = job_to_wire(job)
    return (json.dumps(json.loads(wire['config_json']), indent=2),
            json.dumps(json.loads(wire['writer_json']), indent=2))
