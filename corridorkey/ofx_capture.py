"""Bounded, deliberate OFX upstream capture intake; no host calls or inference.

Version1 is opaque assumed-linear709 RGB float32 LE, top-down, with relative
source-witness digests. Host time stays separate from the local zero-based job.
Metadata/payload pins fence ordinary mutation, not hostile filesystem owners.
Materialization retains an owned snapshot; native codec/RSS allocations are not
hard-capped by the one-frame admission bounds. No model is loaded here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import unicodedata
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import InferenceConfig, WriterConfig
from .diagnostics import finite_json_float, json_value
from .identities import verify_file_observation
from .ofx_record import cache_spec
from .portable_paths import native_destination
from .project_collection import (
    FilePin,
    admit_file,
    assert_pin,
    check_now,
    cleanup_owned,
    fingerprint,
    fresh_directory,
    read_file,
    real_path,
)
from .session import JobRequest
from .worker_protocol import job_from_wire, job_to_wire

MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_PAYLOAD_BYTES = 512 * 1024**2
MAX_PIXELS = 16 * 1024**2
MAX_FRAMES = 120
DOMAIN = b'CorridorKey/source/linear-rgb-f32/v1\0'
PREFIX = struct.Struct('<8sIIIqd32s')
Check = Callable[[], None] | None


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate capture JSON field')
        result[key] = value
    return result


def _parse(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError('Capture metadata exceeds its immutable byte bound')
    try:
        data = json.loads(raw, object_pairs_hook=_pairs, parse_float=finite_json_float,
                          parse_constant=finite_json_float)
    except RecursionError as exc:
        raise ValueError('Capture metadata nesting exceeds parser limits') from exc
    if not isinstance(data, dict):
        raise TypeError('Capture metadata requires an object')
    return data


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _manifest(raw: bytes) -> dict[str, Any]:
    data = _parse(raw)
    fields = {'schema_version', 'type', 'capture_id', 'source_scope', 'host_start', 'frame_count',
              'width', 'height', 'fps', 'pixel_aspect', 'colour_contract', 'payload', 'source_digests'}
    if (set(data) != fields or type(data['schema_version']) is not int or data['schema_version'] != 1
            or data['type'] != 'corridorkey_ofx_capture' or data['source_scope'] != 'ofx_instance_upstream'
            or data['colour_contract'] != 'scene_linear_bt709_d65_opaque_rgb'
            or not isinstance(data['capture_id'], str) or not re.fullmatch('[0-9a-f]{32}', data['capture_id'])):
        raise ValueError('Unsupported capture schema/contract/fields')
    for name in ('host_start', 'frame_count', 'width', 'height'):
        if type(data[name]) is not int:
            raise TypeError('Capture time/count/geometry require exact integers')
    if (not -(2**31) <= data['host_start'] < 2**31 or not 1 <= data['frame_count'] <= MAX_FRAMES
            or data['host_start'] + data['frame_count'] >= 2**31
            or min(data['width'], data['height']) < 1 or max(data['width'], data['height']) > 32768
            or data['width'] * data['height'] > MAX_PIXELS):
        raise ValueError('Capture time/count/native geometry exceeds supported bounds')
    fps = data['fps']
    if type(fps) not in (int, float):
        raise ValueError('Capture FPS must be finite and positive')
    try:
        valid_fps = math.isfinite(float(fps)) and fps > 0
    except OverflowError:
        valid_fps = False
    if not valid_fps:
        raise ValueError('Capture FPS must be finite and positive')
    if type(data['pixel_aspect']) not in (int, float) or data['pixel_aspect'] != 1:
        raise ValueError('Capture requires square pixels')
    payload = data['payload']
    expected = data['width'] * data['height'] * data['frame_count'] * 12
    if (not isinstance(payload, dict) or set(payload) != {'path', 'bytes', 'sha256'}
            or payload['path'] != 'frames.rgb32' or type(payload['bytes']) is not int
            or payload['bytes'] != expected or expected > MAX_PAYLOAD_BYTES or not _sha(payload['sha256'])):
        raise ValueError('Capture payload name/length/hash exceeds its fixed contract or 512MiB limit')
    digests = data['source_digests']
    if not isinstance(digests, list) or len(digests) != data['frame_count'] or any(not _sha(item) for item in digests):
        raise ValueError('Capture requires one relative source-witness digest per frame')
    return data


def _pin(pin, path: Path, *, size=None, digest=None):
    if not isinstance(pin, FilePin):
        raise TypeError('Capture requires a FilePin instance')
    if not isinstance(pin.path, str) or native_destination(pin.path) != path:
        raise ValueError('Capture pin path association differs')
    sample = pin.fingerprint
    if (type(sample) is not tuple or len(sample) != 5 or any(type(value) is not int for value in sample)
            or sample[2] < 0 or not isinstance(pin.sha256, str) or (pin.sha256 and not _sha(pin.sha256))):
        raise ValueError('Capture pin fingerprint/hash is malformed')
    if (size is not None and pin.size != size) or (digest is not None and pin.sha256 != digest):
        raise ValueError('Capture pin size/hash differs')


def _raw(pin: FilePin, check: Check) -> bytes:
    if pin.size > MAX_MANIFEST_BYTES:
        raise ValueError('Capture metadata exceeds 4MiB')
    _, raw = read_file(pin, check=check, capture=True)
    return raw


@contextmanager
def _open(pin: FilePin):
    assert_pin(pin)
    path_before = Path(pin.path).lstat()
    if fingerprint(path_before) != pin.fingerprint:
        raise ValueError('Capture changed before descriptor admission')
    descriptor = os.open(pin.path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        fingerprint(before)
        verify_file_observation(pin.path, path_before, before, before)
        yield stream
        verify_file_observation(pin.path, path_before, before, os.fstat(stream.fileno()))


def _frames(pin: FilePin, data, check: Check):
    """One native frame in memory; hashes include exact signed-zero float bits."""
    whole = hashlib.sha256()
    size = data['width'] * data['height'] * 12
    with _open(pin) as stream:
        for index in range(data['frame_count']):
            check_now(check)
            raw = bytearray(size)
            view = memoryview(raw)
            for offset in range(0, size, 1024**2):
                check_now(check)
                chunk = stream.read(min(1024**2, size - offset))
                if len(chunk) != min(1024**2, size - offset):
                    raise ValueError('Capture payload is truncated')
                view[offset:offset + len(chunk)] = chunk
                whole.update(chunk)
            digest_state = hashlib.sha256(DOMAIN + struct.pack('<III', index, data['width'], data['height']))
            digest_state.update(raw)
            digest = digest_state.hexdigest()
            if digest != data['source_digests'][index]:
                raise ValueError('Captured source-witness digest differs from actual pixels')
            yield index, raw
        if stream.read(1):
            raise ValueError('Capture grew or changed during payload read')
    assert_pin(pin)
    if whole.hexdigest() != data['payload']['sha256']:
        raise ValueError('Capture payload content identity differs')
    check_now(check)


def _capture(path: Path, check: Check):
    manifest = admit_file(path)
    if Path(manifest.path).name != 'capture.json':
        raise ValueError('Choose the committed capture.json file')
    root = Path(manifest.path).parent
    marker = admit_file(root / 'COMPLETE')
    if marker.size != 0:
        raise ValueError('Capture COMPLETE marker must be empty')
    raw = _raw(manifest, check)
    data = _manifest(raw)
    payload = admit_file(root / 'frames.rgb32')
    if payload.size != data['payload']['bytes']:
        raise ValueError('Capture payload length differs from manifest')
    assert_pin(manifest)
    assert_pin(marker)
    return data, replace(manifest, sha256=hashlib.sha256(raw).hexdigest()), marker, payload


def _summary(data):
    return {'capture_id': data['capture_id'], 'source_scope': data['source_scope'],
            'host_start': data['host_start'], 'host_end_exclusive': data['host_start'] + data['frame_count'],
            'frame_count': data['frame_count'], 'width': data['width'], 'height': data['height'], 'fps': data['fps'],
            'payload_bytes': data['payload']['bytes'], 'read_bytes_required': data['payload']['bytes'],
            'colour_contract': data['colour_contract'], 'local_start': 0,
            'scope': 'Explicit effect-location upstream pixels; original camera/editor selection and actual Resolve acceptance NOT VERIFIED.'}


def inventory_capture(manifest_path: Path, *, check: Check = None) -> dict[str, Any]:
    data, _, _, _ = _capture(manifest_path, check)
    return dict(_summary(data), verified=False)


@dataclass(frozen=True)
class CapturePlan:
    manifest_json: bytes
    manifest_pin: FilePin
    marker_pin: FilePin
    payload_pin: FilePin

    def __post_init__(self):
        data = _manifest(self.manifest_json)
        if _json(data) != self.manifest_json:
            raise ValueError('CapturePlan requires canonical primitive metadata')
        if not isinstance(self.manifest_pin, FilePin):
            raise TypeError('CapturePlan requires manifest pin instance')
        root = native_destination(self.manifest_pin.path).parent
        _pin(self.manifest_pin, root / 'capture.json')
        _pin(self.marker_pin, root / 'COMPLETE', size=0)
        _pin(self.payload_pin, root / 'frames.rgb32', size=data['payload']['bytes'], digest=data['payload']['sha256'])
        if not _sha(self.manifest_pin.sha256):
            raise ValueError('Verified manifest requires its physical byte identity')

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.manifest_json + _json(asdict(self.manifest_pin)) + _json(asdict(self.payload_pin))).hexdigest()

    @property
    def summary(self) -> dict[str, Any]:
        return dict(_summary(_manifest(self.manifest_json)), verified=True, identity=self.identity)


def _assert_plan(plan: CapturePlan, check: Check):
    if type(plan) is not CapturePlan:
        raise TypeError('Expected a CapturePlan instance')
    plan.__post_init__()
    raw = _raw(plan.manifest_pin, check)
    if hashlib.sha256(raw).hexdigest() != plan.manifest_pin.sha256 or _json(_manifest(raw)) != plan.manifest_json:
        raise ValueError('Capture plan differs from pinned physical manifest')
    assert_pin(plan.marker_pin)
    assert_pin(plan.payload_pin)
    return _manifest(plan.manifest_json)


def verify_capture(manifest_path: Path, *, read_budget: int, check: Check = None) -> CapturePlan:
    if type(read_budget) is not int or not 0 <= read_budget < 2**63:
        raise ValueError('Capture verification requires an exact nonnegative read budget')
    data, manifest, marker, payload = _capture(manifest_path, check)
    if data['payload']['bytes'] > read_budget:
        raise ValueError(f"Explicit verification read budget insufficient: needs {data['payload']['bytes']} bytes")
    import numpy as np
    for _index, raw_pixels in _frames(payload, data, check):
        if not np.isfinite(np.frombuffer(raw_pixels, dtype='<f4')).all():
            raise ValueError('Captured native pixels must be finite')
    plan = CapturePlan(_json(data), manifest, marker, replace(payload, sha256=data['payload']['sha256']))
    _assert_plan(plan, check)
    return plan


def capture_job_config(plan: CapturePlan, config: InferenceConfig) -> InferenceConfig:
    """Pure explicit derived settings; never mutates the caller's requested job."""
    if type(plan) is not CapturePlan or type(config) is not InferenceConfig:
        raise TypeError('Expected capture plan and config instances')
    plan.__post_init__()
    config.validate()
    from .shot_profiles import reject_derived_profiles
    reject_derived_profiles(config, 'OFX capture rebinding')
    if (config.corrections_path is not None or config.source_colour_profile != 'legacy' or config.start_mode != 'begin'
            or config.quick_cut_frames or config.quick_reset_frames or config.frame_dir_alpha != 'ignore'
            or (config.linear_brightness, config.linear_contrast, config.linear_contrast_pivot) != (1., 1., .18)):
        raise ValueError('Capture input cannot preserve corrections, declared source profiles, nonbegin starts, explicit cuts/resets, alpha conversion or linear adjustments; create a deliberate capture-local job')
    values = json.loads(_json(json_value(config)))
    values['outputs'] = set(values['outputs']) | {'fg', 'alpha'}
    values['temp_dir'] = Path(values['temp_dir'])
    data = _manifest(plan.manifest_json)
    values.update(start_frame=0, num_frames=data['frame_count'], frame_dir_fps=data['fps'], frame_dir_linear=True,
                  source_witness=True, frame_dir_alpha='ignore', source_colour_profile='legacy', output_format='exr')
    result = InferenceConfig(**values)
    result.validate()
    return result


@dataclass(frozen=True)
class PreparedCapture:
    plan: CapturePlan
    workspace: Path
    job: JobRequest
    inputs: tuple[FilePin, ...]
    binding_pin: FilePin
    marker_pin: FilePin

    def __post_init__(self):
        if type(self.plan) is not CapturePlan or type(self.job) is not JobRequest or not isinstance(self.workspace, Path):
            raise TypeError('Prepared capture requires plan/job/native workspace instances')
        self.plan.__post_init__()
        root = native_destination(self.workspace)
        if type(self.inputs) is not tuple or len(self.inputs) != self.plan.summary['frame_count']:
            raise ValueError('Prepared capture requires one immutable input pin per frame')
        for index, pin in enumerate(self.inputs):
            _pin(pin, root / 'frames' / f'frame_{index:06d}.exr')
            if not _sha(pin.sha256):
                raise ValueError('Materialized input requires encoded content identity')
        _pin(self.binding_pin, root / 'capture-binding.json')
        _pin(self.marker_pin, root / 'COMPLETE', size=0)
        if self.job.input_path != str(root / 'frames') or _json(job_to_wire(job_from_wire(asdict(self.job)))) != _json(job_to_wire(self.job)):
            raise ValueError('Prepared job differs from captured native input')
        config, _ = self.job.configurations()
        if _json(json_value(capture_job_config(self.plan, config))) != _json(json_value(config)):
            raise ValueError('Prepared job changed required capture interpretation/range')

    @property
    def capture_identity(self) -> str:
        return self.plan.identity

    @property
    def host_start(self) -> int:
        return self.plan.summary['host_start']

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self.plan.summary, workspace=str(self.workspace), job=job_to_wire(self.job),
                    scope='Verified materialized float32 input; models/inference/host attachment NOT VERIFIED.')


def _binding(prepared: PreparedCapture):
    return {'capture_identity': prepared.plan.identity, 'job': job_to_wire(prepared.job),
            'inputs': [asdict(pin) for pin in prepared.inputs]}


def _assert_prepared(prepared: PreparedCapture, check: Check):
    if type(prepared) is not PreparedCapture:
        raise TypeError('Expected a PreparedCapture instance')
    prepared.__post_init__()
    data = _assert_plan(prepared.plan, check)
    raw = _raw(prepared.binding_pin, check)
    if hashlib.sha256(raw).hexdigest() != prepared.binding_pin.sha256 or _json(_parse(raw)) != _json(_binding(prepared)):
        raise ValueError('Prepared capture differs from persistent binding metadata')
    assert_pin(prepared.marker_pin)
    for pin in prepared.inputs:
        check_now(check)
        assert_pin(pin)
    return data


class CapturePreparationError(RuntimeError):
    def __init__(self, message: str, workspace: Path):
        super().__init__(message)
        self.workspace = workspace


def _overlap(first: Path, second: Path):
    a, b = tuple(unicodedata.normalize('NFC', part).casefold() for part in first.parts), tuple(unicodedata.normalize('NFC', part).casefold() for part in second.parts)
    return a[:len(b)] == b or b[:len(a)] == a


def materialize_capture(plan: CapturePlan, *, scratch_parent: Path, output_dir: Path,
                        config: InferenceConfig, writer: WriterConfig | None = None,
                        check: Check = None) -> PreparedCapture:
    candidate = capture_job_config(plan, config)
    writer = WriterConfig() if writer is None else writer
    if type(writer) is not WriterConfig:
        raise TypeError('Expected a writer config instance')
    writer.validate()
    data = _assert_plan(plan, check)
    parent = real_path(native_destination(scratch_parent), directory=True)
    output = real_path(native_destination(output_dir), missing=True)
    capture_root = Path(plan.manifest_pin.path).parent
    if _overlap(parent, capture_root) and len(parent.parts) >= len(capture_root.parts):
        raise ValueError('Scratch parent is inside original capture')
    if output.exists() or _overlap(output, capture_root) or _overlap(output, parent):
        raise ValueError('Choose a fresh output outside capture and scratch locations')
    root = fresh_directory(parent, 'ofx-capture-input-')
    try:
        # Only ready is a published workspace. A failed pending tree has a job
        # bound to the absent ready paths, even when cleanup is unavailable.
        pending, ready = root / 'pending', root / 'ready'
        pending.mkdir()
        frames = pending / 'frames'
        frames.mkdir()
        import cv2
        import numpy as np

        from .video_io import read_image_rgb
        pins = []
        for index, raw in _frames(plan.payload_pin, data, check):
            check_now(check)
            rgb = np.frombuffer(raw, dtype='<f4').reshape(data['height'], data['width'], 3)
            if not np.isfinite(rgb).all():
                raise ValueError('Captured native pixels must be finite')
            path = frames / f'frame_{index:06d}.exr'
            if not cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1]),
                               [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
                                cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]):
                raise RuntimeError('Native FLOAT EXR materialization failed')
            decoded = np.asarray(read_image_rgb(str(path), expected_channels=3), dtype='<f4')
            if decoded.shape != rgb.shape or not np.array_equal(decoded.view(np.uint32), rgb.view(np.uint32)):
                raise ValueError('FLOAT EXR decode differs from captured float32 bits; no conversion fallback')
            pin = admit_file(path)
            digest, _ = read_file(pin, check=check)
            pins.append(replace(pin, path=str(ready / 'frames' / path.name), sha256=digest))
        _assert_plan(plan, check)
        candidate = replace(candidate, temp_dir=ready / 'temp')
        job = JobRequest.create(ready / 'frames', output, candidate, writer)
        binding = {'capture_identity': plan.identity, 'job': job_to_wire(job), 'inputs': [asdict(pin) for pin in pins]}
        path = pending / 'capture-binding.json'
        with path.open('xb') as stream:
            stream.write(_json(binding))
            stream.flush()
            os.fsync(stream.fileno())
        binding_pin = replace(admit_file(path), path=str(ready / path.name), sha256=hashlib.sha256(_json(binding)).hexdigest())
        with (pending / 'COMPLETE').open('xb'):
            pass
        marker_pin = replace(admit_file(pending / 'COMPLETE'), path=str(ready / 'COMPLETE'))
        result = PreparedCapture(plan, ready, job, tuple(pins), binding_pin, marker_pin)
        check_now(check)
        # Moving the parent preserves the child file pins. No fallible work
        # follows this publication, including result construction or file close.
        pending.rename(ready)
        return result
    except BaseException as primary:
        try:
            cleanup_owned(root)
        except OSError as cleanup:
            raise CapturePreparationError(f'{type(primary).__name__}: {primary}; cleanup failed: {cleanup}', root) from primary
        raise


def verify_completed_capture(prepared: PreparedCapture, record: dict[str, Any], *, check: Check = None) -> dict[str, Any]:
    data = _assert_prepared(prepared, check)
    record = _parse(_json(record))
    spec = cache_spec(record)
    config, writer = prepared.job.configurations()
    if (record.get('input') != prepared.job.input_path or record.get('output_dir') != prepared.job.output_dir
            or _json(record.get('requested_settings')) != _json(json_value(config))
            or _json(record.get('writer_settings')) != _json(json_value(writer))
            or record.get('hints') != {'first_frame': None, 'full': None}
            or (spec.width, spec.height, spec.start, spec.count, spec.fps)
            != (data['width'], data['height'], 0, data['frame_count'], data['fps'])):
        raise ValueError('Completed record differs from exact captured job/range/settings')
    from .output_paths import output_layout
    from .source_witness import load_witness
    root = native_destination(prepared.job.output_dir)
    outputs = record['diagnostics']['outputs']
    expected_outputs = {role: str(root / row['path'])
                        for role, row in output_layout(prepared.job.input_path, config)['outputs'].items()}
    if _json(outputs) != _json(expected_outputs):
        raise ValueError('Completed output paths differ from exact captured job layout')
    for value in outputs.values():
        real_path(native_destination(value), directory=True)
    metadata = record['diagnostics'].get('source_witness')
    if not isinstance(metadata, dict) or not isinstance(metadata.get('path'), str):
        raise TypeError('Completed capture requires native consumed-source witness metadata')
    witness_path = native_destination(metadata['path'])
    if witness_path.parent != root:
        raise ValueError('Completed witness must remain in the exact job output directory')
    real_path(witness_path, directory=False)
    effective = record['diagnostics'].get('effective_settings')
    requested = _parse(_json(json_value(config)))
    if not isinstance(effective, dict) or set(effective) != set(requested):
        raise ValueError('Completed capture requires actual effective settings')
    allowed = {'hann_chunk', 'hann_stride', 'low_vram'} if config.auto_vram else set()
    if config.auto_vram and effective.get('low_vram') is True and effective.get('foundation_cache_mb') == 0:
        allowed.add('foundation_cache_mb')
    if _json({key: value for key, value in effective.items() if key not in allowed}) != _json({key: value for key, value in requested.items() if key not in allowed}):
        raise ValueError('Effective capture quality/source settings differ from requested settings')
    effective_wire = job_to_wire(prepared.job)
    effective_wire['config_json'] = _json(effective).decode()
    job_from_wire(effective_wire)  # Validate actual scheduling values, without model/source decode.
    witness = load_witness(record, width=spec.width, height=spec.height, count=spec.count, start=0, fps=spec.fps)
    expected = b''.join(bytes.fromhex(value) for value in data['source_digests'])
    if witness is None or witness[0] != expected:
        raise ValueError('Actual consumed-source witness differs from captured upstream pixels')
    execution = record['diagnostics'].get('execution')
    observed = {}
    if isinstance(execution, dict):
        for key in ('neural_inference', 'scope', 'backend', 'requested_device', 'gpu', 'device_index',
                    'cuda_version', 'stage2_calls', 'frame_evaluations', 'completed_prediction_frames'):
            value = execution.get(key)
            if type(value) in (bool, int, float, str):
                observed[key] = value[:1024] if isinstance(value, str) else value
    return dict(_summary(data), capture_identity=prepared.capture_identity, completed_job_verified=True, execution=observed,
                scope='Engine-reported settings and exact consumed-source witness; model authenticity/quality and Resolve acceptance NOT VERIFIED.')


def validate_capture_destination(prepared: PreparedCapture, parent: Path, *, check: Check = None) -> Path:
    """Admit an existing parent before any sealer/receipt workspace is created.

    A shared ancestor is safe: publication always creates an exclusive child.
    Parents inside captured inputs or engine exports are refused.
    """
    _assert_prepared(prepared, check)
    parent = real_path(native_destination(parent), directory=True)
    protected = (prepared.workspace, Path(prepared.plan.manifest_pin.path).parent,
                 native_destination(prepared.job.output_dir))
    for root in protected:
        if _overlap(parent, root) and len(parent.parts) >= len(root.parts):
            raise ValueError('Cache/receipt parent is inside protected capture/input/export data')
    check_now(check)
    return parent


def attachment_receipt(prepared: PreparedCapture, record: dict[str, Any], cache: Path,
                       destination_parent: Path, *, check: Check = None) -> Path:
    """Validate actual CKOFX002 bytes, then publish an explicit handoff receipt."""
    summary = verify_completed_capture(prepared, record, check=check)
    pin = admit_file(cache)
    record_pin = admit_file(Path(pin.path).parent / 'record.json')
    raw_record = _raw(record_pin, check)
    if _json(_parse(raw_record)) != _json(record):
        raise ValueError('Sealed cache record differs from completed captured job')
    record_hash = hashlib.sha256(raw_record).digest()
    data = _manifest(prepared.plan.manifest_json)
    count, width, height = data['frame_count'], data['width'], data['height']
    frame_bytes = width * height * 16
    if pin.size != PREFIX.size + 32 + count * 64 + frame_bytes * count:
        raise ValueError('Sealed cache file length differs from captured range')
    whole = hashlib.sha256()
    with _open(pin) as stream:
        prefix = stream.read(PREFIX.size)
        expected = PREFIX.pack(b'CKOFX002', width, height, count, 0, float(data['fps']), record_hash)
        header, table = stream.read(32), stream.read(count * 64)
        if prefix != expected or header != hashlib.sha256(prefix + table).digest():
            raise ValueError('Sealed cache header/record identity differs')
        whole.update(prefix + header + table)
        import numpy as np
        for index in range(count):
            if table[index * 64 + 32:index * 64 + 64] != bytes.fromhex(data['source_digests'][index]):
                raise ValueError('Sealed cache source binding differs from capture')
            digest = hashlib.sha256(record_hash + struct.pack('<I', index))
            for offset in range(0, frame_bytes, 1024**2):
                check_now(check)
                block = stream.read(min(1024**2, frame_bytes - offset))
                if len(block) != min(1024**2, frame_bytes - offset):
                    raise ValueError('Sealed cache truncated during verification')
                rgba = np.frombuffer(block, dtype='<f4').reshape(-1, 4)
                if not np.isfinite(rgba).all() or np.any((rgba[:, 3] < 0) | (rgba[:, 3] > 1)):
                    raise ValueError('Sealed cache contains invalid RGB/coverage')
                whole.update(block)
                digest.update(block)
            if digest.digest() != table[index * 64:index * 64 + 32]:
                raise ValueError('Sealed cache output frame content identity differs')
        if stream.read(1):
            raise ValueError('Sealed cache changed during verification')
    assert_pin(pin)
    assert_pin(record_pin)
    _assert_prepared(prepared, check)
    parent = validate_capture_destination(prepared, destination_parent, check=check)
    root = fresh_directory(parent, 'ofx-attachment-')
    try:
        pending, ready = root / 'pending', root / 'ready'
        pending.mkdir()
        receipt = dict(summary, schema_version=1, type='corridorkey_ofx_attachment', cache_file=pin.path,
                       cache_sha256=whole.hexdigest(), record_sha256=record_hash.hex(), cache_start=prepared.host_start,
                       local_start=0, instruction='Explicitly choose CacheFile and CacheStart in a permitted host action. Existing host parameters were not changed.')
        path = pending / 'attachment.json'
        with path.open('xb') as stream:
            stream.write(_json(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        with (pending / 'COMPLETE').open('xb'):
            pass
        result = ready / 'attachment.json'
        check_now(check)
        assert_pin(pin)
        pending.rename(ready)
        return result
    except BaseException as primary:
        try:
            cleanup_owned(root)
        except OSError as cleanup:
            raise CapturePreparationError(f'{type(primary).__name__}: {primary}; cleanup failed: {cleanup}', root) from primary
        raise
