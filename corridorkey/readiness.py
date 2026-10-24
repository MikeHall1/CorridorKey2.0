"""Read-only readiness observations and explicit bounded render preparation.

No model, Qt, network, decoder or GPU initialization is performed here. A PASS
describes only the named observation; it is never inference acceptance.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .asset_inventory import AssetSelection, inspect_assets
from .control import CancellationToken, CancelledError
from .output_paths import output_layout, validate_output_paths
from .resource_snapshot import disk_snapshot, process_snapshot
from .session import JobRequest
from .worker_protocol import job_to_wire

MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_RENDER_FRAMES = 120


def check_readiness(job: JobRequest | None, selection: AssetSelection | None, *,
                    cancellation: CancellationToken | None = None) -> dict[str, Any]:
    """Inspect local metadata only, with separately labelled deferred stages.

    Output permission checks are advisory and create no files. Shared engine
    admission repeats before model loading and writes. Cancellation returns the
    partial report, including the exact interrupted stage.
    """
    token = cancellation if cancellation is not None else CancellationToken()
    report: dict[str, Any] = {
        'schema_version': 1, 'type': 'corridorkey_readiness',
        'created_utc': datetime.now(UTC).isoformat(),
        'scope': 'Local metadata observations only. No models loaded, inference, decoding, downloads or installations.',
        'status': 'complete', 'stages': [],
        'environment': {'python': sys.version, 'executable': sys.executable,
                        'platform': platform.platform()},
        'requested_job': job_to_wire(job) if job is not None else None,
        'effective_settings': None,
        'protected_paths': [],
    }
    stages = report['stages']

    def stage(name, operation):
        token.check()
        try:
            status, details = operation()
        except CancelledError:
            stages.append({'stage': name, 'status': 'CANCELLED', 'details': 'Cancelled during this check.'})
            raise
        except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
            status, details = 'FAIL', f'{type(exc).__name__}: {exc}'[:4096]
        stages.append({'stage': name, 'status': status, 'details': details})
        token.check()

    def dependencies():
        # Installed metadata only, not proof that native extensions can import.
        from packaging.specifiers import SpecifierSet

        required = {'numpy': '>=1.26,<2', 'opencv-python': '>=4.11',
                    'torch': '==2.9.1', 'torchvision': '==0.24.1',
                    'safetensors': '>=0.6', 'transformers': '>=4.57,<5',
                    'timm': '>=1.0', 'huggingface-hub': '>=0.36',
                    'einops': '>=0.8', 'scipy': '>=1.16', 'pillow': '>=11',
                    'tqdm': '>=4.67', 'utils3d': ''}
        rows = []
        for name, specifier in required.items():
            token.check()
            try:
                version = importlib.metadata.version(name)
                compatible = version in SpecifierSet(specifier)
            except importlib.metadata.PackageNotFoundError:
                version, compatible = None, False
            rows.append({'package': name, 'required': specifier or 'installed',
                         'version': version, 'compatible_metadata': compatible})
        return ('PASS' if all(row['compatible_metadata'] for row in rows) else 'FAIL',
                {'packages': rows, 'limits': 'Installed versions only; native imports and utils3d Git identity NOT VERIFIED.'})

    def assets():
        chosen = selection or AssetSelection(Path(__file__).resolve().parents[1])
        report['protected_paths'].append(str(chosen.root.resolve()))
        if chosen.weights is not None:
            report['protected_paths'].append(str(chosen.weights.resolve()))
        result = inspect_assets(chosen)
        result['integrity'] = 'NOT VERIFIED. Use Local assets → Audit selected files for explicit hashes/container inspection.'
        return 'PASS' if result['assets_present'] else 'FAIL', result

    def source():
        if job is None:
            return 'NOT SELECTED', 'Choose local footage and an output parent in the workspace.'
        config, _ = job.configurations()
        job.validate_source(config)
        return 'PASS', 'Source/hint paths and settings checked; source decoding and content identity NOT VERIFIED.'

    def ffmpeg():
        if job is None:
            return 'NOT SELECTED', 'Choose a job to inspect its configured FFmpeg path.'
        config, writer = job.configurations()
        executable = shutil.which(writer.ffmpeg_bin)
        return ('NOT REQUIRED' if config.output_format == 'exr' else ('PASS' if executable else 'FAIL'),
                {'requested': writer.ffmpeg_bin, 'executable': executable,
                 'codec': writer.codec, 'capabilities': 'NOT VERIFIED. Executable lookup only; codecs/pixel formats are checked by the real encoder.',
                 'action': 'Set a working FFmpeg executable in Common job settings for video output.'})

    def destination():
        if job is None:
            return 'NOT SELECTED', 'Choose an output parent.'
        config, _ = job.configurations()
        validate_output_paths(job.input_path, job.output_dir, config,
                              initial_hint_path=job.initial_hint_path,
                              hint_video_path=job.hint_video_path, check=token.check)
        output = Path(job.output_dir)
        existing = output
        while not existing.exists() and existing != existing.parent:
            token.check()
            existing = existing.parent
        if not existing.is_dir() or not os.access(existing, os.W_OK | os.X_OK):
            raise ValueError('Output ancestor is not an accessible writable directory. Choose another output parent.')
        layout = output_layout(job.input_path, config)
        occupied = [str(output / name) for name in layout['metadata'] if (output / name).exists()]
        occupied += [str(output / item['path']) for item in layout['outputs'].values()
                     if (output / item['path']).exists()]
        return ('WARNING' if occupied else 'PASS',
                {'output_dir': str(output), 'existing_planned_paths': occupied,
                 'permissions': 'Advisory os.access check only; ACLs, quotas, races and actual writes NOT VERIFIED.',
                 'collision_policy': 'Ordinary jobs may replace unrelated exports. Guided renders use a fresh unique child.'})

    try:
        if job is not None:
            config, _ = job.configurations()
            report['protected_paths'].extend(str(Path(path).resolve()) for path in
                                             (job.input_path, job.initial_hint_path, job.hint_video_path)
                                             if path is not None)
            if config.corrections_path is not None:
                # Protect the document and its relative asset tree conservatively.
                report['protected_paths'].append(str(Path(config.corrections_path).resolve().parent))
        stage('application', lambda: ('PASS', 'Shared settings/readiness service imported; native dependencies NOT VERIFIED.'))
        stage('dependencies', dependencies)
        stage('assets', assets)
        stage('source', source)
        stage('ffmpeg', ffmpeg)
        stage('destination', destination)
        stage('resources', lambda: ('PASS', {'process': process_snapshot(),
                                            'disk': disk_snapshot(Path(job.output_dir)) if job else None}))
        for name, details in (
            ('backend', 'CUDA is the only production backend. Use Local assets → Check for an explicit isolated availability probe. No backend fallback.'),
            ('model_load', 'Requires an explicit render with selected local weights and CUDA; asset presence does not establish model compatibility.'),
            ('inference', 'Requires an explicit short render. CPU fixtures and readiness checks are not actual inference.'),
            ('output_decode', 'Requires completed exports and explicit preview inspection.'),
            ('display', 'Requires successful presentation in the desktop preview.')):
            stage(name, lambda details=details: ('NOT VERIFIED', details))
    except CancelledError as exc:
        report['status'], report['error'] = 'cancelled', str(exc)[:4096]
    return report


def prepare_short_render(job: JobRequest, start_frame: int, frame_count: int) -> JobRequest:
    """Prepare, never execute or create files. Preserve every quality setting."""
    if type(start_frame) is not int or not 0 <= start_frame <= 2**31 - 1:
        raise ValueError('Start frame must be an exact nonnegative integer up to 2^31-1.')
    if type(frame_count) is not int or not 1 <= frame_count <= MAX_RENDER_FRAMES:
        raise ValueError(f'Guided render length must be 1..{MAX_RENDER_FRAMES} frames.')
    config, writer = job.configurations()
    from .shot_profiles import reject_derived_profiles
    reject_derived_profiles(config, 'Guided short render')
    if config.start_mode != 'begin':
        raise ValueError('Choose an explicit source In with start_mode=begin before preparing a guided render.')
    if config.corrections_path is not None:
        raise ValueError('Guided subranges with corrections require deliberate rebasing. Use Diagnostic ranges preparation or a job without corrections.')
    if job.initial_hint_path is not None and start_frame != (config.start_frame or 0):
        raise ValueError('The first-frame hint belongs to the original source In. Choose that In or explicitly remove the hint in future settings.')
    offset = start_frame - (config.start_frame or 0)
    updated = replace(config, start_frame=start_frame, num_frames=frame_count,
                      quick_cut_frames=[index - offset for index in config.quick_cut_frames
                                        if offset <= index < offset + frame_count],
                      quick_reset_frames=[index - offset for index in config.quick_reset_frames
                                          if offset <= index < offset + frame_count])
    output = Path(job.output_dir) / f'readiness-{uuid.uuid4().hex}'
    if output.exists():
        raise FileExistsError(output)
    prepared = JobRequest.create(job.input_path, output, updated, writer,
                                 initial_hint_path=job.initial_hint_path,
                                 hint_video_path=job.hint_video_path,
                                 profile_stages=job.profile_stages)
    prepared.validate_source(updated)
    return prepared


def export_report(report: dict[str, Any], destination: Path) -> None:
    """Explicit fresh local export, never overwrite or write into inputs/assets."""
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError('Readiness report exceeds 4 MiB.')
    destination = destination.absolute()
    target = destination.resolve()
    for value in report.get('protected_paths', []):
        path = Path(value).resolve()
        if target == path or (path.is_dir() and target.is_relative_to(path)):
            raise ValueError('Save diagnostic reports separately from source, hints, corrections and assets.')
    # x mode refuses existing files, links and hard-link aliases; no mkdir.
    with destination.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
