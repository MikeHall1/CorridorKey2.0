"""Bounded local diagnostics to explicit standalone range jobs; never submits work.

Current dependency observations are not authentication of the original render's
source bytes. A range job starts fresh temporal state, not a full-clip patch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .control import ProgressEvent
from .diagnostics import finite_json_float, json_value
from .local_json import _depth, _pairs
from .session import JobRequest

MAX_FRAMES = 100_000
MAX_TRACE_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_DEPENDENCIES = 100_000
MAX_DEPENDENCY_BYTES = 16 * 1024 * 1024


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _check(check):
    if check is not None:
        check()


def _integer(value, minimum=0, maximum=2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Diagnostic frame/count must be an exact bounded integer.')
    return value


def _number(value, minimum=0.0, maximum=float('inf')) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('Diagnostic number must be finite and in its declared range.')
    return float(value)


def _safe(path: Path):
    """Reject lexical symlink/reparse parents before canonicalizing."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Diagnostic paths cannot use symbolic links/reparse aliases.')
    return path


def _stat(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _observation(path, check):
    path = _safe(Path(path))
    info = path.stat()
    if stat.S_ISREG(info.st_mode):
        return {'path': str(path), 'kind': 'file', 'identity': _stat(info)}
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError('Dependency must be a regular file or frame directory.')
    rows, total = [], 0
    with os.scandir(path) as entries:
        for entry in entries:
            _check(check)
            if len(rows) >= MAX_DEPENDENCIES:
                raise ValueError('Dependency directory entry bound exceeded.')
            child = entry.stat(follow_symlinks=False)
            if not (stat.S_ISREG(child.st_mode) or stat.S_ISDIR(child.st_mode)):
                raise ValueError('Frame dependency directory contains a nonregular entry.')
            row = [entry.name, _stat(child)]
            total += len(_json(row).encode())
            if total > MAX_DEPENDENCY_BYTES:
                raise ValueError('Dependency metadata byte bound exceeded.')
            rows.append(row)
    if _stat(info) != _stat(path.stat()):
        raise ValueError('Dependency directory changed during inspection.')
    rows.sort(key=lambda row: row[0])
    return {'path': str(path), 'kind': 'directory', 'identity': _stat(info),
            'entries': len(rows), 'entries_token': _hash(rows)}


def _read_lines(path, count, check) -> Iterator[dict[str, Any]]:
    path = _safe(path)
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    digest, size, previous = hashlib.sha256(), 0, -1
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_TRACE_BYTES:
            raise ValueError('Trace requires a bounded regular single-link file.')
        while True:
            _check(check)
            raw = stream.readline(min(MAX_LINE_BYTES + 1, MAX_TRACE_BYTES - size + 1))
            if not raw:
                break
            size += len(raw)
            if size > MAX_TRACE_BYTES or len(raw) > MAX_LINE_BYTES or not raw.endswith(b'\n'):
                raise ValueError('Trace line/byte bound or complete-line requirement failed.')
            digest.update(raw)
            _depth(raw, 16)
            row = json.loads(raw, object_pairs_hook=_pairs, parse_float=finite_json_float,
                             parse_constant=finite_json_float)
            if not isinstance(row, dict):
                raise TypeError('Trace row must be an object.')
            if '_receipt' in row:
                raise ValueError('Reserved diagnostic row field.')
            frame = _integer(row.get('frame'), 0, count - 1)
            if frame <= previous:
                raise ValueError('Trace frames must be unique and chronological.')
            previous = frame
            yield row
        if _stat(before) != _stat(os.fstat(stream.fileno())) or _stat(before) != _stat(path.stat()):
            raise ValueError('Trace changed during inspection.')
    yield {'_receipt': {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': size}}


@dataclass(frozen=True)
class DiagnosticSnapshot:
    payload_json: str

    def to_dict(self):
        if not isinstance(self.payload_json, str) or len(self.payload_json.encode()) > MAX_SNAPSHOT_BYTES:
            raise ValueError('Diagnostic snapshot exceeds its byte bound.')
        _depth(self.payload_json.encode(), 32)
        data = json.loads(self.payload_json, object_pairs_hook=_pairs,
                          parse_float=finite_json_float, parse_constant=finite_json_float)
        fields = {'schema_version', 'token', 'job', 'record_identity', 'terminal_status', 'source', 'traces',
                  'events', 'dependencies', 'coverage', 'warnings', 'historical_source_identity_verified', 'scope'}
        if (not isinstance(data, dict) or set(data) != fields or type(data.get('schema_version')) is not int or
                data['schema_version'] != 1):
            raise ValueError('Unknown diagnostic snapshot schema.')
        unsigned = dict(data)
        token = unsigned.pop('token', None)
        if token != _hash(unsigned):
            raise ValueError('Diagnostic snapshot token mismatch.')
        job = JobRequest(**data['job'])
        job.configurations()
        source = data['source']
        if (not isinstance(source, dict) or source.get('path') != job.input_path or
                type(source.get('scene_linear')) is not bool or data['historical_source_identity_verified'] is not False):
            raise ValueError('Invalid diagnostic snapshot source binding.')
        start = _integer(source.get('start_frame'))
        count = _integer(source.get('num_frames'), 1, MAX_FRAMES)
        if data['terminal_status'] not in ('complete', 'failed', 'cancelled'):
            raise ValueError('Diagnostic snapshot requires a terminal job.')
        if not isinstance(data['traces'], list) or len(data['traces']) > 2:
            raise ValueError('Diagnostic snapshot trace bound exceeded.')
        for trace in data['traces']:
            if (not isinstance(trace, dict) or trace.get('kind') not in ('quick', 'shot') or
                    not Path(trace['path']).is_absolute() or not Path(trace['path']).is_relative_to(job.output_dir)):
                raise ValueError('Diagnostic snapshot trace ownership mismatch.')
            _integer(trace['bytes'], 0, MAX_TRACE_BYTES)
            observed = _integer(trace['observed_frames'], 0, count)
            if trace['missing_frames'] != count - observed:
                raise ValueError('Diagnostic snapshot coverage mismatch.')
        if not isinstance(data['events'], list) or len(data['events']) > 8192:
            raise ValueError('Diagnostic snapshot event bound exceeded.')
        previous = -1
        for event in data['events']:
            relative = _integer(event['relative_frame'], 0, count - 1)
            if (relative <= previous or event['frame'] != start + relative or type(event['observed']) is not bool or
                    not isinstance(event['reasons'], list) or not event['reasons'] or len(event['reasons']) > 16 or
                    any(not isinstance(value, str) or len(value) > 64 for value in event['reasons']) or
                    not isinstance(event['sources'], list) or not set(event['sources']) <= {'quick', 'shot', 'summary'}):
                raise ValueError('Diagnostic snapshot event schema mismatch.')
            previous = relative
        if not isinstance(data['dependencies'], list) or len(data['dependencies']) > MAX_DEPENDENCIES:
            raise ValueError('Diagnostic snapshot dependency bound exceeded.')
        required = {job.input_path, job.initial_hint_path, job.hint_video_path, job.configurations()[0].corrections_path} - {None}
        if not required <= {entry['path'] for entry in data['dependencies']}:
            raise ValueError('Diagnostic snapshot lacks selected job dependencies.')
        return data


@dataclass(frozen=True)
class RangeCandidate:
    id: str
    start_frame: int
    end_frame: int
    reasons: tuple[str, ...]
    observed_frames: int

    def to_dict(self):
        return asdict(self)


def inspect_diagnostics(job: JobRequest, record: dict, *, check=None, progress=None) -> DiagnosticSnapshot:
    _check(check)
    if not isinstance(job, JobRequest) or not isinstance(record, dict):
        raise TypeError('Select a matching immutable job and terminal record.')
    raw_record = _json(json_value(record))
    if len(raw_record.encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError('Selected terminal record exceeds inspection bound.')
    cfg, writer = job.configurations()
    if (record.get('status') not in ('complete', 'failed', 'cancelled') or
            record.get('input') != job.input_path or record.get('output_dir') != job.output_dir or
            record.get('requested_settings') != json_value(cfg) or record.get('writer_settings') != json_value(writer) or
            record.get('hints') != {'first_frame': job.initial_hint_path, 'full': job.hint_video_path}):
        raise ValueError('Terminal record does not match the selected job settings/paths/hints.')
    diagnostics = record.get('diagnostics')
    source = diagnostics.get('source') if isinstance(diagnostics, dict) else None
    if not isinstance(diagnostics, dict) or not isinstance(source, dict):
        raise TypeError('Terminal record lacks observed source selection.')
    start, count = _integer(source.get('start_frame')), _integer(source.get('num_frames'), 1, MAX_FRAMES)
    expected_start = cfg.start_frame if cfg.start_frame is not None else (0 if cfg.start_mode == 'begin' else None)
    if (expected_start is not None and start != expected_start) or (cfg.num_frames > 0 and count > cfg.num_frames):
        raise ValueError('Observed terminal source range conflicts with requested selection.')
    hw, fps = source.get('native_hw'), _number(source.get('native_fps'), 1e-300)
    if (not isinstance(hw, list) or len(hw) != 2 or any(type(x) is not int or x < 1 for x in hw)):
        raise ValueError('Invalid observed source geometry/FPS.')
    output = _safe(Path(job.output_dir))
    if not output.is_dir():
        raise ValueError('Job output directory is unavailable.')
    deps = [job.input_path, job.initial_hint_path, job.hint_video_path, cfg.corrections_path]
    if cfg.corrections_path:
        from .corrections import CorrectionDocument
        from .local_json import read_object
        document_path = Path(cfg.corrections_path)
        data, _ = read_object(document_path, max_bytes=MAX_SNAPSHOT_BYTES, regular_file=True)
        document = CorrectionDocument.parse(data, document_path.parent)
        deps.extend(str(path) for path in document.asset_paths())
    dependencies = [_observation(path, check) for path in dict.fromkeys(str(p) for p in deps if p is not None)]
    quick = diagnostics.get('quick_key') or {}
    if not isinstance(quick, dict):
        raise TypeError('Quick Key diagnostics must be an object.')
    analysis = quick.get('analysis') or {}
    if not isinstance(analysis, dict):
        raise TypeError('Shot diagnostics must be an object.')
    paths = [('quick', quick.get('frame_diagnostics_path')), ('shot', analysis.get('trace_path'))]
    events: dict[int, dict[str, Any]] = {}
    def event(frame, reason, origin, observed):
        frame = _integer(frame, 0, count - 1)
        value = events.setdefault(frame, {'frame': start + frame, 'relative_frame': frame,
                                         'reasons': [], 'sources': [], 'observed': False})
        if reason not in value['reasons']:
            value['reasons'].append(reason)
        if origin not in value['sources']:
            value['sources'].append(origin)
        value['observed'] |= observed
        if len(events) > 8192:
            raise ValueError('Diagnostic event count exceeds bounded inspection capacity.')
    traces, observed_counts = [], {'quick': 0, 'shot': 0}
    for kind, name in paths:
        if name is None:
            continue
        if not isinstance(name, str):
            raise TypeError('Trace path must be a string.')
        path = _safe(Path(name))
        if not path.is_relative_to(output):
            raise ValueError('Trace must belong to selected job output directory.')
        for row in _read_lines(path, count, check):
            if '_receipt' in row:
                traces.append({**row['_receipt'], 'kind': kind,
                               'observed_frames': observed_counts[kind],
                               'missing_frames': count - observed_counts[kind],
                               'coverage': 'complete' if observed_counts[kind] == count else 'partial'})
                continue
            observed_counts[kind] += 1
            if kind == 'shot':
                if (type(row.get('cut')) is not bool or row.get('calibrated') is not False or
                        any(_number(row.get(k)) < 0 for k in ('score', 'mean_rgb_change', 'histogram_distance'))):
                    raise ValueError('Invalid native shot diagnostic row.')
                if row['cut']:
                    event(row['frame'], 'heuristic_cut', kind, True)
            else:
                if (row.get('event') != 'rendered_before_writer_submission' or row.get('source_rgb_draft') is not True or
                        row.get('fallback') not in (None, 'stop', 'neural', 'approximate')):
                    raise ValueError('Invalid native Quick Key diagnostic row.')
                for key in ('uncertain_fraction', 'motion_fraction', 'mean_reliability'):
                    if key in row:
                        _number(row[key], 0, float('inf') if key == 'motion_fraction' else 1)
                if _number(row.get('uncertain_fraction', 0), 0, 1) > 0:
                    event(row['frame'], 'uncertain_propagation', kind, True)
                fallback = row.get('fallback')
                if fallback:
                    event(row['frame'], 'fallback_' + str(fallback), kind, True)
                if row.get('promoted_anchor') is True:
                    event(row['frame'], 'promoted_anchor', kind, True)
    for key, reason in (('fallback_frames', 'fallback_summary'), ('extra_neural_frames', 'extra_neural_summary'),
                        ('promoted_anchors', 'promoted_anchor_summary')):
        values = quick.get(key, [])
        if not isinstance(values, list) or len(values) > count:
            raise ValueError('Invalid bounded Quick Key summary frame list.')
        for frame in values:
            event(frame, reason, 'summary', False)
    cuts = analysis.get('detected_cuts', [])
    if not isinstance(cuts, list) or len(cuts) > count:
        raise ValueError('Invalid bounded shot summary frame list.')
    for frame in cuts:
        event(frame, 'heuristic_cut', 'summary', False)
    rows = sorted(events.values(), key=lambda row: row['frame'])
    for row in rows:
        row['reasons'].sort()
        row['sources'].sort()
    result = {'schema_version': 1, 'job': asdict(job), 'record_identity': hashlib.sha256(raw_record.encode()).hexdigest(),
              'terminal_status': record['status'],
              'source': {'path': job.input_path, 'start_frame': start, 'num_frames': count, 'native_hw': hw, 'native_fps': fps,
                         'scene_linear': source.get('exr_linear') is True, 'source_colour_profile': cfg.source_colour_profile,
                         'alpha_mode': cfg.frame_dir_alpha},
              'traces': traces, 'events': rows, 'dependencies': dependencies,
              'coverage': {'expected_frames': count, 'quick_observed_frames': observed_counts['quick'],
                           'shot_observed_frames': observed_counts['shot'],
                           'summary_only_events': sum(not row['observed'] for row in rows)},
              'warnings': ['Reliability and cut events are uncalibrated diagnostics, not quality confidence.',
                           'Standalone range jobs restart temporal state; not full-clip-equivalent patches.'],
              'historical_source_identity_verified': False,
              'scope': 'Current local dependency observations; original rendered source bytes are not authenticated.'}
    result['token'] = _hash(result)
    snapshot = DiagnosticSnapshot(_json(result))
    snapshot.to_dict()
    _revalidate(snapshot, check)
    if progress is not None:
        progress(ProgressEvent('diagnostic_ranges', count, count))
    return snapshot


def _revalidate(snapshot, check):
    data = snapshot.to_dict()
    for old in data['dependencies']:
        _check(check)
        if _observation(old['path'], check) != old:
            raise ValueError('A source/hint/correction dependency changed since inspection.')
    for old in data['traces']:
        receipt = None
        for row in _read_lines(Path(old['path']), data['source']['num_frames'], check):
            if '_receipt' in row:
                receipt = row['_receipt']
        if receipt != {key: old[key] for key in ('path', 'bytes', 'sha256')}:
            raise ValueError('Diagnostic trace changed since inspection.')
    _check(check)
    return data


def candidates(snapshot: DiagnosticSnapshot, *, padding_frames=0) -> tuple[RangeCandidate, ...]:
    _integer(padding_frames, 0, MAX_FRAMES)
    data = snapshot.to_dict()
    start, count = data['source']['start_frame'], data['source']['num_frames']
    groups = []
    for row in data['events']:
        lo, hi = max(start, row['frame'] - padding_frames), min(start + count, row['frame'] + padding_frames + 1)
        if groups and lo <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], hi)
            groups[-1][2].update(row['reasons'])
            groups[-1][3] += int(row['observed'])
        else:
            groups.append([lo, hi, set(row['reasons']), int(row['observed'])])
    return tuple(RangeCandidate(_hash([data['token'], lo, hi])[:16], lo, hi, tuple(sorted(reasons)), observed)
                 for lo, hi, reasons, observed in groups)


def prepare_range(snapshot: DiagnosticSnapshot, start_frame, end_frame, existing_parent, *,
                  correction_policy='reject', initial_hint_policy='reject', full_quality=True,
                  protected_paths=(), check=None) -> JobRequest:
    if correction_policy not in ('reject', 'drop') or initial_hint_policy not in ('reject', 'drop') or type(full_quality) is not bool:
        raise ValueError('Choose explicit correction/initial-hint and full-quality policies.')
    data = _revalidate(snapshot, check)
    source = data['source']
    lo, hi = _integer(start_frame), _integer(end_frame, 1)
    if not source['start_frame'] <= lo < hi <= source['start_frame'] + source['num_frames']:
        raise ValueError('Standalone range must lie within observed original source selection.')
    job = JobRequest(**data['job'])
    cfg, writer = job.configurations()
    from .shot_profiles import reject_derived_profiles
    reject_derived_profiles(cfg, 'Standalone diagnostic range')
    changed = (lo, hi) != (source['start_frame'], source['start_frame'] + source['num_frames'])
    if changed and cfg.corrections_path and correction_policy != 'drop':
        raise ValueError('Correction document is bound to original range; explicitly drop or cancel.')
    if lo != source['start_frame'] and job.initial_hint_path and initial_hint_policy != 'drop':
        raise ValueError('First-frame hint belongs to original start; explicitly drop or cancel.')
    parent = _safe(Path(existing_parent))
    if not parent.is_dir():
        raise ValueError('Choose an existing output parent directory.')
    protected = [job.input_path, job.output_dir, job.initial_hint_path, job.hint_video_path, cfg.corrections_path,
                 *(old['path'] for old in data['dependencies']), *protected_paths]
    for name in protected:
        if name is None:
            continue
        target = Path(name).resolve()
        if parent == target or parent.is_relative_to(target):
            raise ValueError('Output parent is inside a protected source/assets/hints/original output path.')
    root = parent / ('range-' + uuid.uuid4().hex)
    offset = lo - source['start_frame']
    cfg = replace(cfg, start_mode='begin', start_frame=lo, num_frames=hi - lo, temp_dir=root / 'temp',
                  corrections_path=None if correction_policy == 'drop' else cfg.corrections_path,
                  quick_cut_frames=[v - offset for v in cfg.quick_cut_frames if offset <= v < offset + hi - lo],
                  quick_reset_frames=[v - offset for v in cfg.quick_reset_frames if offset <= v < offset + hi - lo])
    if full_quality:
        cfg = replace(cfg, model_scale=1.0, quick_key=False, quick_approach='offline', quick_auto_cuts=False,
                      quick_adaptive=False, quick_max_gap=None, quick_cut_frames=[], quick_reset_frames=[])
    prepared = JobRequest.create(job.input_path, root / 'outputs', cfg, writer,
                                 initial_hint_path=None if initial_hint_policy == 'drop' else job.initial_hint_path,
                                 hint_video_path=job.hint_video_path, profile_stages=job.profile_stages)
    _check(check)
    _revalidate(snapshot, check)
    root.mkdir()  # Exclusive final reservation. Children are created by the ordinary runtime.
    # No fallible observations or cancellation checks after completed preparation.
    return prepared


def revalidate(snapshot: DiagnosticSnapshot, *, check=None, progress=None) -> None:
    """Recheck current dependencies/trace bytes without executing or preparing jobs."""
    _revalidate(snapshot, check)
    if progress is not None:
        progress(ProgressEvent('diagnostic_ranges_verify', 1, 1))
