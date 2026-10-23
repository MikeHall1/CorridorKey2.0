"""CPU-only decoded-frame reuse investigation, never a production cache option.

Generated bounded fixtures pass through the existing sequence readers and colour
preprocessing. A serialized process-local hook is restored after every arm.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
import statistics
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .control import CancelledError

_MAX_PIXELS = 2**20
_FIELDS = {'schema_version', 'kind', 'description', 'artifact_parent', 'repetitions', 'cache_bytes',
           'frame_count', 'shape_hw', 'chunk', 'stride', 'quick_interval'}
_GUARD = threading.Lock()
_RUN_GUARD = threading.Lock()
_COUNTERS = ('decode_calls', 'preprocess_calls', 'read_requests', 'cache_hits', 'cache_misses',
             'evictions', 'oversize', 'retained_bytes', 'peak_retained_bytes', 'copy_bytes')
_TIMES = ('elapsed_seconds', 'decode_seconds', 'preprocess_seconds')
_SCHEDULES = ('overlap', 'sequential_prepasses', 'quick_current_anchor')
_CASES = ('uint16_png', 'signed_hdr_exr', 'premultiplied_bt2020_exr')


def _integer(value, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f'Expected exact integer in [{lo}, {hi}].')
    return value


def validate_manifest(manifest) -> dict:
    """Normalize research settings without filesystem/model/heavy-package access."""
    if not isinstance(manifest, dict) or set(manifest) - _FIELDS:
        raise ValueError('Unknown decoded-reuse manifest field.')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or manifest.get('kind') != 'decode_reuse':
        raise ValueError('Expected schema_version=1, kind=decode_reuse.')
    parent = manifest.get('artifact_parent')
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError('Decoded-reuse requires an explicit existing artifact_parent.')
    description = manifest.get('description', '')
    if not isinstance(description, str) or len(description) > 4096:
        raise ValueError('Research description must be bounded text.')
    shape = manifest.get('shape_hw', [64, 64])
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError('shape_hw requires two integer dimensions.')
    shape = [_integer(v, 16, 512) for v in shape]
    frames = _integer(manifest.get('frame_count', 12), 8, 128)
    if frames * shape[0] * shape[1] > _MAX_PIXELS:
        raise ValueError('Aggregate generated fixture pixel bound exceeded.')
    chunk = _integer(manifest.get('chunk', 8), 1, 128)
    return {'schema_version': 1, 'kind': 'decode_reuse', 'description': description, 'artifact_parent': parent,
            'repetitions': _integer(manifest.get('repetitions', 3), 3, 20),
            'cache_bytes': _integer(manifest.get('cache_bytes', 262144), 0, 256 * 1024 * 1024),
            'frame_count': frames, 'shape_hw': shape, 'chunk': chunk,
            'stride': _integer(manifest.get('stride', 4), 1, chunk),
            'quick_interval': _integer(manifest.get('quick_interval', 3), 2, frames)}


def _check(check):
    if check is not None:
        check()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _fingerprint(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _identity(root, check, frame_count):
    rows = []
    with os.scandir(root) as entries:
        names = []
        for entry in entries:
            if len(names) == len(_CASES) or entry.name not in _CASES or not entry.is_dir(follow_symlinks=False):
                raise ValueError('Generated fixture root inventory changed.')
            names.append(entry.name)
    if set(names) != set(_CASES):
        raise ValueError('Generated fixture cases are missing.')
    for case in _CASES:
        extension = 'png' if case == 'uint16_png' else 'exr'
        expected = {f'{frame:04}.{extension}' for frame in range(frame_count)}
        found = set()
        with os.scandir(root / case) as entries:
            for entry in entries:
                _check(check)
                if entry.name not in expected or entry.name in found:
                    raise ValueError('Generated fixture file inventory changed.')
                found.add(entry.name)
        if found != expected:
            raise ValueError('Generated fixture frames are missing.')
        for name in sorted(found):
            _check(check)
            path = root / case / name
            flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
            digest = hashlib.sha256()
            with os.fdopen(os.open(path, flags), 'rb') as stream:
                before = os.fstat(stream.fileno())
                if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 8 * 1024 * 1024 or
                        getattr(before, 'st_file_attributes', 0) & 0x400):
                    raise ValueError('Generated fixture must remain a bounded owned regular file.')
                remaining = before.st_size
                while remaining:
                    _check(check)
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError('Generated source truncated during identity capture.')
                    remaining -= len(block)
                    digest.update(block)
                if stream.read(1):
                    raise ValueError('Generated source grew during identity capture.')
                after = os.fstat(stream.fileno())
            identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            if identity(before) != identity(after) or identity(before) != identity(path.lstat()):
                raise ValueError('Generated source changed during identity capture.')
            rows.append({'path': path.relative_to(root).as_posix(), 'bytes': before.st_size, 'sha256': digest.hexdigest()})
    return rows


def _fixtures(root, settings, check):
    import cv2
    import numpy as np

    height, width = settings['shape_hw']
    yy, xx = np.mgrid[:height, :width]
    for case in _CASES:
        (root / case).mkdir()
    for frame in range(settings['frame_count']):
        _check(check)
        channels = [(xx * 4099 + yy * 769 + frame * 97 + offset) % 65536 for offset in (0, 11939, 33631)]
        png = np.stack(channels, axis=-1).astype(np.uint16)
        linear = png.astype(np.float32) / np.float32(65535) * np.float32(8) - np.float32(2)
        linear[0, 0, 0] = np.float32(-0.0)
        alpha = (((xx + yy + frame) % 9) / 8).astype(np.float32)[..., None]
        premult = np.concatenate((linear * alpha, alpha), axis=-1)
        for case, image, extension in ((_CASES[0], png, 'png'), (_CASES[1], linear, 'exr'), (_CASES[2], premult, 'exr')):
            path = root / case / f'{frame:04}.{extension}'
            params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT] if extension == 'exr' else []
            if not cv2.imwrite(str(path), image, params):
                raise OSError('Could not write bounded generated decode fixture.')


class _Digests:
    def __init__(self):
        self.values: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.hash_seconds = 0.0

    def observe(self, name, value):
        import numpy as np
        import torch

        if value is None:
            return
        started = time.perf_counter()
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
            dtype, shape = str(value.dtype), list(value.shape)
        else:
            array = np.ascontiguousarray(value)
            dtype, shape = str(array.dtype), list(array.shape)
        digest = self.values.setdefault(name, hashlib.sha256())
        digest.update(_encoded([dtype, shape]))
        digest.update(memoryview(array))
        self.counts[name] = self.counts.get(name, 0) + 1
        self.hash_seconds += time.perf_counter() - started

    def snapshot(self):
        return {'digests': {name: value.hexdigest() for name, value in self.values.items()},
                'observations': dict(self.counts), 'hash_seconds': self.hash_seconds}


class _ReadHook:
    """Serial research scope; callers must not run inference in this process."""
    def __init__(self, video_io, budget, candidate, check, observer=None, *, paths=()):
        self.video_io, self.budget, self.candidate = video_io, budget, candidate
        self.paths = frozenset(os.path.abspath(path) for path in paths)
        self.owner_thread = None
        self.check, self.observer = check, observer
        self.cache = OrderedDict()
        self.stats: dict[str, Any] = dict.fromkeys(_COUNTERS, 0)
        self.stats.update(dict.fromkeys(_TIMES, 0.0))
        self.depth = 0
        self.original = None

    def __enter__(self):
        if not _GUARD.acquire(blocking=False):
            raise RuntimeError('Decoded-reuse research hook is already owned; concurrent/reentrant runs refuse.')
        self.owner_thread = threading.get_ident()
        self.original = (self.video_io.read_image_rgb, self.video_io.preprocess_frame_dir_path,
                         self.video_io._read_frame_dir_views)
        self.video_io.read_image_rgb = self.read
        self.video_io.preprocess_frame_dir_path = self.preprocess
        self.video_io._read_frame_dir_views = self.views
        return self

    def __exit__(self, *unused):
        if self.original is not None:
            (self.video_io.read_image_rgb, self.video_io.preprocess_frame_dir_path,
             self.video_io._read_frame_dir_views) = self.original
        self.cache.clear()
        self.stats['retained_bytes'] = 0
        _GUARD.release()

    def owns(self, path):
        if threading.get_ident() != self.owner_thread:
            return False
        return os.path.abspath(path) in self.paths

    def _process(self, function, *args, **kwargs):
        _check(self.check)
        outer = self.depth == 0
        self.depth += 1
        started = time.perf_counter()
        if outer:
            self.stats['preprocess_calls'] += 1
        try:
            return function(*args, **kwargs)
        finally:
            self.depth -= 1
            if outer:
                self.stats['preprocess_seconds'] += time.perf_counter() - started

    def preprocess(self, *args, **kwargs):
        assert self.original is not None
        if not self.owns(args[0] if args else kwargs['path']):
            return self.original[1](*args, **kwargs)
        return self._process(self.original[1], *args, **kwargs)

    def views(self, *args, **kwargs):
        assert self.original is not None
        if not self.owns(args[0] if args else kwargs['path']):
            return self.original[2](*args, **kwargs)
        return self._process(self.original[2], *args, **kwargs)

    def read(self, path, *, info=None, alpha_mode='ignore', expected_channels=None):
        assert self.original is not None
        if not self.owns(path):
            return self.original[0](path, info=info, alpha_mode=alpha_mode, expected_channels=expected_channels)
        _check(self.check)
        self.stats['read_requests'] += 1
        path = Path(path).resolve()
        before = _fingerprint(path)
        key = (str(path), before, alpha_mode, expected_channels)
        cached = self.cache.pop(key, None) if self.candidate and self.budget else None
        if cached is not None:
            self.cache[key] = cached
            retained, stored_info = cached
            result = retained.copy()
            self.stats['cache_hits'] += 1
            self.stats['copy_bytes'] += result.nbytes
            if info is not None:
                info.update(stored_info)
        else:
            self.stats['cache_misses'] += 1
            self.stats['decode_calls'] += 1
            decoded_info = {}
            started = time.perf_counter()
            try:
                result = self.original[0](path, info=decoded_info, alpha_mode=alpha_mode, expected_channels=expected_channels)
            finally:
                self.stats['decode_seconds'] += time.perf_counter() - started
            if info is not None:
                info.update(decoded_info)
            if before != _fingerprint(path):
                raise ValueError('Source changed across native decoding.')
            if self.candidate and self.budget:
                if result.nbytes > self.budget:
                    self.stats['oversize'] += 1
                else:
                    while self.stats['retained_bytes'] + result.nbytes > self.budget:
                        _, (old, _) = self.cache.popitem(last=False)
                        self.stats['retained_bytes'] -= old.nbytes
                        self.stats['evictions'] += 1
                    retained = result.copy()
                    retained.flags.writeable = False
                    self.cache[key] = (retained, decoded_info)
                    self.stats['copy_bytes'] += retained.nbytes
                    self.stats['retained_bytes'] += retained.nbytes
                    self.stats['peak_retained_bytes'] = max(self.stats['peak_retained_bytes'], self.stats['retained_bytes'])
        if before != _fingerprint(path):
            raise ValueError('Source changed across native decoded-frame access.')
        if self.observer is not None:
            self.observer.observe('native_rgb', result)
        return result


def _schedule(video_io, meta, schedule, settings, observer, check):
    import numpy as np

    from .draft import offline_frames
    from .motion import MotionSettings
    from .quick_key import plan_anchors
    from .scheduling import window_starts

    source = str(Path(meta['frame_paths'][0]).parent)
    def observe(name, value):
        if observer is not None:
            observer.observe(name, value)
    def windows():
        reader = video_io.SequentialFrameDirWindowReader(source, meta)
        try:
            for start in window_starts(meta['num_frames'], settings['chunk'], settings['stride']):
                _check(check)
                model, foundation, review = reader.window(start, min(start + settings['chunk'], meta['num_frames']))
                observe('model', model)
                observe('foundation', foundation)
                observe('review', review)
        finally:
            reader.close()
    if schedule == 'overlap':
        windows()
    elif schedule == 'sequential_prepasses':
        for _ in range(2):
            generator = video_io.iter_frame_dir_chunks(source, meta, chunk_size=settings['chunk'], need_model=False)
            try:
                for _start, model, foundation in generator:
                    _check(check)
                    observe('model', model)
                    observe('foundation', foundation)
            finally:
                generator.close()
        windows()
    else:
        current = video_io.DraftSourceReader(source, meta, is_dir=True)
        anchor = video_io.DraftSourceReader(source, meta, is_dir=True)
        plan = plan_anchors(meta['num_frames'], settings['quick_interval'])
        frames = offline_frames(plan, current.read, anchor.read,
                                lambda _frame: np.full(meta['output_hw'], 0.5, dtype=np.float32),
                                settings=MotionSettings(max_side=64), fallback='approximate')
        try:
            for frame in frames:
                _check(check)
                observe('rgb', frame.rgb)
                # The real proxy/linear views are observed at their shared read boundary.
        finally:
            frames.close()
            current.close()
            anchor.close()


def _arm(video_io, meta, schedule, settings, candidate, check, *, validate=False) -> dict[str, Any]:
    observer = _Digests() if validate else None
    hook = _ReadHook(video_io, settings['cache_bytes'], candidate, check, observer, paths=meta['frame_paths'])
    with hook:
        # Hook the real frame-dir view result only for untimed proxy/linear verification.
        if observer is not None:
            base_views = video_io._read_frame_dir_views
            def observed_views(*args, **kwargs):
                display, linear = base_views(*args, **kwargs)
                if not hook.owns(args[0] if args else kwargs['path']):
                    return display, linear
                observer.observe('proxy', display)
                observer.observe('linear', linear)
                return display, linear
            video_io._read_frame_dir_views = observed_views
        started = time.perf_counter()
        _schedule(video_io, meta, schedule, settings, observer, check)
        hook.stats['elapsed_seconds'] = time.perf_counter() - started
        stats = dict(hook.stats)
    if validate:
        return observer.snapshot() if observer is not None else {}
    return stats


def run_decode_reuse(*, artifact_parent, repetitions=3, cache_bytes=262144, frame_count=12,
                     shape_hw=(64, 64), chunk=8, stride=4, quick_interval=3, check=None, progress=None) -> dict:
    settings = validate_manifest({'schema_version': 1, 'kind': 'decode_reuse', 'artifact_parent': str(artifact_parent),
                                  'repetitions': repetitions, 'cache_bytes': cache_bytes, 'frame_count': frame_count,
                                  'shape_hw': shape_hw, 'chunk': chunk, 'stride': stride, 'quick_interval': quick_interval})
    report = {'schema_version': 1, 'kind': 'decode_reuse', 'status': 'failed',
              'summary': {'success': False, 'completed_results': 0, 'expected_results': 9, 'exact_equal': False},
              'scope': {'device': 'cpu', 'model': False, 'os_cache': 'uncontrolled; no cache drops or cold-storage claim',
                        'cache_boundary': 'read_image_rgb float32 RGB after normalization/association; before transfer/primaries',
                        'measurement_scope': 'reader/preprocessing CPU workload incl stacks and real QuickKey Farneback; not inference',
                        'parity_scope': 'separate untimed streaming arms per measured pair; fresh caches each arm',
                        'budget_scope': 'retained cache payload only, excludes returned copies/tensors/process RSS',
                        'timing_scopes': 'preprocess_seconds includes decode_seconds; do not sum',
                        'hook_scope': 'serialized research arms; hook delegates foreign paths and nonowner threads uninstrumented'},
              'results': [], 'warnings': ['No GPU, model, quality, production-cache or end-to-end benefit claim.'],
              'artifact_dir': None, 'fixture_identity': {'recipe': 'decode-reuse-v1', 'files': [], 'preserved': False},
              'settings': settings, 'inference': {'status': 'not_measured', 'fps': None, 'quality': None},
              'error': None, 'cleanup': {'hook_restored': True, 'retained_generated_artifacts': True}}
    owned = False
    video_io = None
    before_hooks = None
    try:
        owned = _RUN_GUARD.acquire(blocking=False)
        if not owned:
            raise RuntimeError('Decoded-reuse investigation already active in this process.')
        _check(check)
        from . import video_io
        before_hooks = (video_io.read_image_rgb, video_io.preprocess_frame_dir_path, video_io._read_frame_dir_views)
        parent = Path(artifact_parent).resolve(strict=True)
        if not parent.is_dir():
            raise ValueError('Research artifact parent must be an existing directory.')
        root = parent / ('decode-reuse-' + uuid.uuid4().hex)
        root.mkdir()
        report['artifact_dir'] = str(root)
        _fixtures(root, settings, check)
        initial = _identity(root, check, settings['frame_count'])
        report['fixture_identity']['files'] = initial
        for case in _CASES:
            meta = video_io.probe_frame_dir_meta(str(root / case), num_frames=frame_count, shape_quantum=16, fps=24,
                                                 exr_linear=case != 'uint16_png', model_scale=1.0,
                                                 alpha_mode='premultiplied' if case == _CASES[2] else 'ignore',
                                                 source_colour_profile='linear_bt2020_d65' if case == _CASES[2] else 'legacy')
            for schedule in _SCHEDULES:
                _check(check)
                row = {'case_id': case, 'schedule': schedule, 'settings': {'source_colour_contract': meta['source_colour_contract'],
                        'source_alpha_mode': meta['source_alpha_mode'], 'shape_hw': list(meta['shape_hw'])},
                       'setup': {'probe_decodes': 1, 'measured': False, 'decoded_dtypes': meta['decoded_source_dtypes_observed']},
                       'warmup': {'pairs': 1, 'measured': False}, 'pairs': [], 'exact_equal': False, 'summary': None}
                report['results'].append(row)
                for candidate in (False, True):
                    _arm(video_io, meta, schedule, settings, candidate, check)
                for index in range(repetitions):
                    _check(check)
                    order = ['reference', 'candidate'] if index % 2 == 0 else ['candidate', 'reference']
                    pair = {'index': index, 'order': order}
                    for name in order:
                        pair[name] = _arm(video_io, meta, schedule, settings, name == 'candidate', check)
                    validation: dict[str, Any] = {name: _arm(video_io, meta, schedule, settings, name == 'candidate', check, validate=True)
                                  for name in order}
                    validation['exact_equal'] = all(validation['reference'][key] == validation['candidate'][key] for key in ('digests', 'observations'))
                    pair['validation'] = validation
                    row['pairs'].append(pair)
                    if not validation['exact_equal']:
                        raise ValueError('Reference/candidate decoded representations differ bitwise.')
                row['exact_equal'] = True
                med_ref = statistics.median(pair['reference']['elapsed_seconds'] for pair in row['pairs'])
                med_cand = statistics.median(pair['candidate']['elapsed_seconds'] for pair in row['pairs'])
                row['summary'] = {'median_reference_seconds': med_ref, 'median_candidate_seconds': med_cand,
                                  'median_ratio_candidate_over_reference': statistics.median(pair['candidate']['elapsed_seconds'] / pair['reference']['elapsed_seconds'] for pair in row['pairs']) if all(pair['reference']['elapsed_seconds'] > 0 for pair in row['pairs']) else None,
                                  'useful_decode_reuse': any(pair['candidate']['decode_calls'] < pair['reference']['decode_calls'] for pair in row['pairs'])}
                report['summary']['completed_results'] += 1
                if progress is not None:
                    progress('decode_reuse', report['summary']['completed_results'], 9)
        report['fixture_identity']['preserved'] = _identity(root, check, settings['frame_count']) == initial
        if not report['fixture_identity']['preserved']:
            raise ValueError('Generated source bytes changed during the investigation.')
        report['status'] = 'complete'
        report['summary'].update(success=True, exact_equal=True)
    except (CancelledError, KeyboardInterrupt):
        report['status'], report['error'] = 'cancelled', 'Decoded-reuse investigation cancelled; generated partial artifacts retained.'
    except Exception as exc:
        logging.getLogger(__name__).exception('Decoded-reuse investigation failed')
        report['error'] = f'{type(exc).__name__}: {str(exc)[:1024]}'
    finally:
        if owned:
            if before_hooks is not None and video_io is not None:
                report['cleanup']['hook_restored'] = before_hooks == (video_io.read_image_rgb, video_io.preprocess_frame_dir_path, video_io._read_frame_dir_views)
            _RUN_GUARD.release()
    return report


def _expected_observations(settings, schedule):
    from .quick_key import plan_anchors
    from .scheduling import window_starts

    count, chunk = settings['frame_count'], settings['chunk']
    if schedule == 'quick_current_anchor':
        plan = plan_anchors(count, settings['quick_interval'])
        retained, reads = set(), count
        for frame in range(count):
            left, right = plan.bracket(frame)
            retained.intersection_update((left, right))
            if left != right:
                for anchor in (left, right):
                    if anchor not in retained:
                        reads += 1
                        retained.add(anchor)
        return {'native_rgb': reads, 'proxy': reads, 'linear': reads, 'rgb': count}, reads
    windows = len(window_starts(count, chunk, settings['stride']))
    extra = 2 * ((count + chunk - 1) // chunk) if schedule == 'sequential_prepasses' else 0
    reads = count * (3 if schedule == 'sequential_prepasses' else 1)
    return {'native_rgb': reads, 'proxy': reads, 'linear': reads,
            'model': windows, 'foundation': windows + extra, 'review': windows}, reads


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(letter in '0123456789abcdef' for letter in value)


def _finite_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid decoded-reuse CPU time.')


def _validate_report(report) -> None:
    allowed = {'schema_version', 'kind', 'status', 'summary', 'scope', 'results', 'warnings', 'artifact_dir',
               'fixture_identity', 'settings', 'inference', 'error', 'cleanup',
               'manifest', 'manifest_sha256', 'repository', 'environment'}
    if not isinstance(report, dict) or set(report) - allowed:
        raise ValueError('Unknown decoded-reuse report fields.')
    if len(_encoded(report)) > 16 * 1024 * 1024:
        raise ValueError('Decoded-reuse report exceeds byte limit.')
    if type(report.get('schema_version')) is not int or report['schema_version'] != 1 or report.get('kind') != 'decode_reuse':
        raise ValueError('Invalid decoded-reuse report schema/kind.')
    if report.get('status') not in ('complete', 'failed', 'cancelled'):
        raise ValueError('Invalid decoded-reuse report status.')
    settings = validate_manifest(report['settings'])
    if report['inference'] != {'status': 'not_measured', 'fps': None, 'quality': None}:
        raise ValueError('Decoded-reuse reports cannot claim inference/quality results.')
    rows = report['results']
    if not isinstance(rows, list) or len(rows) > 9:
        raise ValueError('Decoded-reuse result count exceeds bound.')
    seen = set()
    for row in rows:
        key = (row['case_id'], row['schedule'])
        if key in seen or key[0] not in _CASES or key[1] not in _SCHEDULES:
            raise ValueError('Unknown/duplicate decoded-reuse case/schedule.')
        seen.add(key)
        expected_observations, expected_reads = _expected_observations(settings, row['schedule'])
        if type(row['exact_equal']) is not bool:
            raise ValueError('Decoded-reuse row exact_equal must be boolean.')
        if not isinstance(row['pairs'], list) or len(row['pairs']) > settings['repetitions']:
            raise ValueError('Decoded-reuse pair count exceeds declared repetitions.')
        for index, pair in enumerate(row['pairs']):
            expected = ['reference', 'candidate'] if index % 2 == 0 else ['candidate', 'reference']
            if pair['index'] != index or pair['order'] != expected:
                raise ValueError('Decoded-reuse pairs must preserve alternating execution order.')
            for side in ('reference', 'candidate'):
                stats = pair[side]
                if set(stats) != set(_COUNTERS + _TIMES):
                    raise ValueError('Unknown/missing decoded-reuse measurement fields.')
                for key in _COUNTERS:
                    _integer(stats[key], 0, 2**63 - 1)
                for key in _TIMES:
                    _finite_seconds(stats[key])
                if stats['peak_retained_bytes'] > settings['cache_bytes'] or stats['read_requests'] != stats['decode_calls'] + stats['cache_hits']:
                    raise ValueError('Decoded-reuse cache/count accounting mismatch.')
                if stats['read_requests'] != expected_reads or stats['preprocess_calls'] != expected_reads:
                    raise ValueError('Decoded-reuse measured workload population mismatch.')
                if side == 'reference' and (stats['decode_calls'] != expected_reads or stats['cache_hits'] != 0 or
                                            stats['peak_retained_bytes'] != 0 or stats['copy_bytes'] != 0):
                    raise ValueError('Reference must execute the actual uncached native decode population.')
                if stats['cache_misses'] != stats['decode_calls'] or stats['retained_bytes'] > stats['peak_retained_bytes']:
                    raise ValueError('Decoded-reuse miss/retained accounting mismatch.')
            validation = pair['validation']
            for side in ('reference', 'candidate'):
                observed = validation[side]
                if set(observed) != {'digests', 'observations', 'hash_seconds'}:
                    raise ValueError('Missing decoded-reuse streaming validation fields.')
                if set(observed['digests']) != set(expected_observations) or observed['observations'] != expected_observations:
                    raise ValueError('Decoded-reuse validation stream population mismatch.')
                if any(type(value) is not int or value <= 0 for value in observed['observations'].values()):
                    raise ValueError('Decoded-reuse validation observations require exact positive integers.')
                if not all(_hex(value) for value in observed['digests'].values()):
                    raise ValueError('Decoded-reuse validation requires real SHA256-shaped digests.')
                _finite_seconds(observed['hash_seconds'])
            if type(validation['exact_equal']) is not bool or validation['exact_equal'] != all(validation['reference'][key] == validation['candidate'][key] for key in ('digests', 'observations')):
                raise ValueError('Decoded-reuse parity claim does not match streaming digests.')
        if report['status'] == 'complete' and (len(row['pairs']) != settings['repetitions'] or not row['exact_equal'] or
                                              not all(pair['validation']['exact_equal'] for pair in row['pairs'])):
            raise ValueError('Complete decoded-reuse result lacks exact paired coverage.')
        if row['exact_equal']:
            if len(row['pairs']) != settings['repetitions'] or not all(pair['validation']['exact_equal'] for pair in row['pairs']):
                raise ValueError('Decoded-reuse assessed row lacks complete exact paired coverage.')
            expected_summary = {
                'median_reference_seconds': statistics.median(pair['reference']['elapsed_seconds'] for pair in row['pairs']),
                'median_candidate_seconds': statistics.median(pair['candidate']['elapsed_seconds'] for pair in row['pairs']),
                'median_ratio_candidate_over_reference': statistics.median(pair['candidate']['elapsed_seconds'] / pair['reference']['elapsed_seconds'] for pair in row['pairs']) if all(pair['reference']['elapsed_seconds'] > 0 for pair in row['pairs']) else None,
                'useful_decode_reuse': any(pair['candidate']['decode_calls'] < pair['reference']['decode_calls'] for pair in row['pairs'])}
            if row['summary'] != expected_summary or type(row['summary']['useful_decode_reuse']) is not bool:
                raise ValueError('Decoded-reuse derived summary conflicts with raw pairs.')
        elif row['summary'] is not None:
            raise ValueError('Incomplete decoded-reuse row must not claim a summary.')
    summary = report['summary']
    if (type(summary['exact_equal']) is not bool or summary['exact_equal'] != summary['success'] or
            type(summary['completed_results']) is not int or summary['completed_results'] != sum(row['exact_equal'] for row in rows) or
            type(summary['expected_results']) is not int or summary['expected_results'] != 9 or
            type(report['fixture_identity']['preserved']) is not bool):
        raise ValueError('Decoded-reuse summary coverage/type mismatch.')
    files = report['fixture_identity']['files']
    expected_files = {f'{case}/{frame:04}.{"png" if case == "uint16_png" else "exr"}'
                      for case in _CASES for frame in range(settings['frame_count'])}
    if not isinstance(files, list) or len(files) > len(expected_files):
        raise ValueError('Decoded-reuse fixture inventory exceeds declared bound.')
    paths = set()
    for entry in files:
        if entry['path'] not in expected_files or entry['path'] in paths or not _hex(entry['sha256']):
            raise ValueError('Decoded-reuse fixture identity is invalid.')
        _integer(entry['bytes'], 1, 8 * 1024 * 1024)
        paths.add(entry['path'])
    if report['status'] == 'complete' and paths != expected_files:
        raise ValueError('Complete decoded-reuse report lacks fixture identities.')
    if type(summary['success']) is not bool or summary['success'] != (report['status'] == 'complete'):
        raise ValueError('Decoded-reuse terminal success mismatch.')
    if report['status'] == 'complete' and (len(seen) != 9 or summary['completed_results'] != 9 or
                                          not report['fixture_identity']['preserved'] or not summary['exact_equal']):
        raise ValueError('Complete decoded-reuse report lacks source/parity coverage.')


def validate_report(report) -> None:
    """Strict finite measurement admission; malformed reports always raise ValueError."""
    try:
        _validate_report(report)
    except (TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed decoded-reuse report.') from exc
