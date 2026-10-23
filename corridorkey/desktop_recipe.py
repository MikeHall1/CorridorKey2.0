"""Immutable one-job recovery recipes, without copying or modifying media."""
from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path

from .diagnostics import write_json_atomic
from .queue_snapshot import _batch_entry, separate
from .session import JobRequest
from .worker_protocol import MAX_FRAME_BYTES, encode_frame, job_from_wire, job_to_wire


def prepare_recipe(parent: Path, job: JobRequest, *, weights: Path | None = None) -> Path:
    job = job_from_wire(job_to_wire(job))
    config, _ = job.configurations()
    job.validate_source(config)
    if config.output_format != 'exr' or config.auto_vram:
        raise ValueError('Select EXR and explicit scheduling (automatic scheduling off) before preparing recovery.')
    parent = parent.resolve()
    if not parent.is_dir():
        raise ValueError('Choose an existing directory for the recovery recipe.')
    token = uuid.uuid4().hex
    root = parent / f'corridorkey-recipe-{token}'
    # The workspace form names an output parent, never a reusable concrete output.
    job = replace(job, output_dir=str(Path(job.output_dir) / f'corridorkey-{token}'))
    separate(root, [job.input_path, job.output_dir, job.initial_hint_path, job.hint_video_path,
                    config.corrections_path, weights])
    data = {'schema_version': 1, 'kind': 'inference_batch',
            'weights': str(weights.resolve()) if weights is not None else None,
            'jobs': [_batch_entry(job)]}
    if len(encode_frame(data)) > MAX_FRAME_BYTES:
        raise ValueError('Recipe exceeds the worker frame limit.')
    if len((json.dumps(data, indent=2, allow_nan=False) + '\n').encode('utf-8')) > MAX_FRAME_BYTES:
        raise ValueError('Recipe exceeds the saved batch size limit.')
    root.mkdir(mode=0o700)
    write_json_atomic(root / 'recipe-owner.json', {'schema_version': 1, 'kind': 'corridorkey_desktop_recipe', 'id': token})
    path = root / 'job.json'
    write_json_atomic(path, data)
    return path
