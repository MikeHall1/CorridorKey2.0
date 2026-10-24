"""Explicit bounded full-context selected-frame preview recipes for the shared inference engine.

Preparation imports no model or Qt and does not render. Native admission reads
bounded encoded bytes without decoding. Runtime still reads source paths: the
before/after fences detect ordinary drift, not hostile concurrent mutation.
Scratch admission/cleanup budgets are not an allocation or GPU-memory hard cap.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .diagnostics import finite_json_float, json_value
from .native_image_admission import MAX_ENCODED_BYTES, MAX_PIXELS, admit_image
from .preview_sequence import SequenceIndex, scan_sequence
from .progressive_preview import PreviewContext, PreviewStage
from .project_collection import FilePin, admit_file, read_file
from .runtime_paths import RuntimePaths
from .session import JobRequest
from .worker_protocol import MAX_FRAME_BYTES, job_from_wire

Check = Callable[[], None]
MAX_CLEANUP_FILES = 256
MAX_CLEANUP_BYTES = 512 * 1024**2
WARNING = ('Every selected preview processes the entire captured timeline with original temporal context and Stage 2 groups; '
           'only the selected EXR is retained. This has full-timeline inference and temporary-disk cost, not a latency guarantee. '
           'Scratch budgets are admission/cleanup bounds, not allocation hard limits; the original temp-map budget still applies. '
           'Model references are metadata only; compatibility and actual inference are unverified.')


def _json(value: Any, limit: int = MAX_FRAME_BYTES) -> bytes:
    raw = json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > limit:
        raise ValueError('Preview provenance exceeds its byte bound.')
    return raw


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_id(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('Preview paths must not contain aliases/reparse points.')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _plain_parents(path: Path) -> None:
    for item in (path, *path.parents):
        _file_id(item)


def _positive(value: int, maximum: int, label: str) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{label} must be an integer in 1..{maximum}.')


def _model_paths(refs: dict[str, Any]) -> tuple[str, ...]:
    paths = RuntimePaths.from_package(Path(__file__).resolve().parents[1], refs.get('weights'),
                                      asset_root=refs.get('asset_root'))
    return tuple(str(getattr(paths, name)) for name in
                 ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir'))


def _model_stat(path: Path) -> tuple[int, ...] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('Model metadata pins refuse aliases/reparse points.')
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError('Model metadata pins require regular files or directories.')
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _pin_models(refs: dict[str, Any], check: Check) -> tuple[tuple[str, tuple[int, ...] | None], ...]:
    roots = _model_paths(refs)
    rows: list[tuple[str, tuple[int, ...] | None]] = []
    pending = [Path(path) for path in roots]
    path_bytes = 0
    while pending:
        check()
        path = pending.pop()
        path_bytes += len(os.fsencode(path))
        if len(rows) >= 2048 or path_bytes > 512 * 1024:
            raise ValueError('Model metadata tree exceeds preview pin count/path bounds.')
        identity = _model_stat(path)
        rows.append((str(path), identity))
        if identity is not None:
            _plain_parents(path)
            if stat.S_ISDIR(identity[2]):
                with os.scandir(path) as entries:
                    for entry in entries:
                        check()
                        if len(rows) + len(pending) >= 2048:
                            raise ValueError('Model metadata tree exceeds preview pin count bound.')
                        pending.append(path / entry.name)
    rows.sort()
    for path, identity in rows:
        check()
        if _model_stat(Path(path)) != identity:
            raise ValueError('Model metadata changed while capturing pins.')
    return tuple(rows)


@dataclass(frozen=True)
class PreviewSource:
    original_job: JobRequest
    index: SequenceIndex
    source_start: int
    source_end: int
    requested_settings_json: bytes
    provenance_json: bytes
    source_identity: str
    max_pixels: int
    max_encoded_bytes: int
    _directory_identity: tuple[int, int, int, int, int] = field(repr=False)
    _source_ctimes: tuple[int, ...] = field(repr=False)
    _model_references_json: bytes = field(repr=False)
    _model_paths: tuple[str, ...] = field(repr=False)
    _model_pins: tuple[tuple[str, tuple[int, ...] | None], ...] = field(repr=False)
    _dependency_pins: tuple[FilePin, ...] = field(repr=False)
    _dependency_directories: tuple[tuple[str, tuple[int, ...]], ...] = field(repr=False)

    def check(self, frame: int, check: Check | None = None) -> None:
        check = check or (lambda: None)
        check()
        if type(frame) is not int or not self.source_start <= frame < self.source_end:
            raise ValueError('Preview frame lies outside the captured requested source range.')
        root = Path(self.index.source)
        _plain_parents(root)
        if _file_id(root) != self._directory_identity:
            raise ValueError('Source sequence membership changed; prepare previews again.')
        for path, identity in self._dependency_directories:
            if _file_id(Path(path)) != identity:
                raise ValueError('Preview context dependency directory changed; capture again.')
        for pin in self._dependency_pins:
            check()
            read_file(pin, check=check)
        if _model_paths(json.loads(self._model_references_json)) != self._model_paths:
            raise ValueError('Selected model paths changed; capture a new preview context.')
        for path, identity in self._model_pins:
            check()
            if _model_stat(Path(path)) != identity:
                raise ValueError('Selected model metadata changed; capture a new preview context.')
        check()


@dataclass
class PreviewWorkspace:
    root: Path
    max_cleanup_files: int
    max_cleanup_bytes: int
    _root_id: tuple[int, int] = field(repr=False)
    _marker: bytes = field(repr=False)
    closed: bool = False
    cleanup_error: str | None = None
    _marker_retired: bool = field(default=False, repr=False)

    def _entries(self, check: Check) -> list[tuple[Path, os.stat_result]]:
        check()
        if self.closed:
            raise ValueError('Preview workspace has already been released.')
        _plain_parents(self.root)
        if _file_id(self.root)[:2] != self._root_id:
            raise ValueError('Owned preview workspace was replaced.')
        marker = self.root / '.preview-owner'
        if self._marker_retired:
            # We removed the marker ourselves only after all admitted contents.
            # A retry may remove this exact empty inode; never adopt new contents.
            with os.scandir(self.root) as entries:
                if next(entries, None) is not None:
                    raise ValueError('Retired preview workspace acquired new contents; retained for review.')
            check()
            return []
        info = marker.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != len(self._marker):
            raise ValueError('Preview workspace ownership marker is invalid.')
        with marker.open('rb') as stream:
            if stream.read(len(self._marker) + 1) != self._marker:
                raise ValueError('Preview workspace ownership marker changed.')
        rows: list[tuple[Path, os.stat_result]] = []
        pending = [(self.root, 0)]
        total = 0
        while pending:
            parent, depth = pending.pop()
            if depth > 16:
                raise ValueError('Preview workspace nesting exceeds cleanup bound.')
            with os.scandir(parent) as entries:
                for entry in entries:
                    check()
                    if len(rows) >= self.max_cleanup_files:
                        raise ValueError('Preview workspace entry count exceeds cleanup bound; retained for review.')
                    path = parent / entry.name
                    item = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(item.st_mode) or getattr(item, 'st_file_attributes', 0) & 0x400:
                        raise ValueError('Preview workspace contains an alias; retained for review.')
                    if stat.S_ISDIR(item.st_mode):
                        pending.append((path, depth + 1))
                    elif stat.S_ISREG(item.st_mode) and item.st_nlink == 1:
                        total += item.st_size
                        if total > self.max_cleanup_bytes:
                            raise ValueError('Preview workspace bytes exceed cleanup bound; retained for review.')
                    else:
                        raise ValueError('Preview workspace contains an unsupported/hard-linked entry.')
                    rows.append((path, item))
        check()
        return rows

    def check_budget(self, check: Check | None = None) -> dict[str, int]:
        rows = self._entries(check or (lambda: None))
        return {'entries': len(rows), 'bytes': sum(info.st_size for _, info in rows if stat.S_ISREG(info.st_mode))}

    def cleanup(self, check: Check | None = None) -> None:
        """Release only this owned tree after bounded admission; failure retains truth."""
        if self.closed:
            return
        check = check or (lambda: None)
        try:
            rows = self._entries(check)
            # Admit everything before the first deletion. Retain the marker until last.
            rows.sort(key=lambda row: (row[0] == self.root / '.preview-owner', -len(row[0].parts)))
            for path, info in rows:
                check()
                current = path.lstat()
                if (current.st_dev, current.st_ino, current.st_mode) != (info.st_dev, info.st_ino, info.st_mode):
                    raise ValueError('Preview workspace changed during cleanup.')
                if stat.S_ISDIR(info.st_mode):
                    path.rmdir()
                else:
                    if (current.st_size, current.st_mtime_ns, current.st_ctime_ns, current.st_nlink) != (info.st_size, info.st_mtime_ns, info.st_ctime_ns, 1):
                        raise ValueError('Preview file changed during cleanup.')
                    path.unlink()
                    if path == self.root / '.preview-owner':
                        self._marker_retired = True
            if _file_id(self.root)[:2] != self._root_id:
                raise ValueError('Owned preview workspace was replaced before final removal.')
            self.root.rmdir()
        except BaseException as exc:
            self.cleanup_error = f'{type(exc).__name__}: {exc}'[:4096]
            raise
        self.closed, self.cleanup_error = True, None


class PreviewPreparationError(RuntimeError):
    """A failed preparation whose owned scratch could not be released."""

    def __init__(self, workspace: PreviewWorkspace, primary_error: BaseException,
                 cleanup_failure: BaseException):
        self.workspace = workspace
        self.primary_error = primary_error
        self.cleanup_failure = cleanup_failure
        super().__init__(f'Preview preparation failed: {type(primary_error).__name__}: {str(primary_error)[:1024]}. '
                         f'Owned scratch retained at {workspace.root}; cleanup failed: '
                         f'{type(cleanup_failure).__name__}: {str(cleanup_failure)[:1024]}')
        primary_error.add_note(f'Owned preview scratch retained at {workspace.root}; cleanup: {cleanup_failure}')


def create_preview_workspace(parent: Path, *, max_cleanup_files: int = MAX_CLEANUP_FILES,
                             max_cleanup_bytes: int = MAX_CLEANUP_BYTES,
                             check: Check | None = None) -> PreviewWorkspace:
    check = check or (lambda: None)
    check()
    _positive(max_cleanup_files, MAX_CLEANUP_FILES, 'Cleanup entries')
    _positive(max_cleanup_bytes, MAX_CLEANUP_BYTES, 'Cleanup bytes')
    if max_cleanup_bytes < 32:
        raise ValueError('Cleanup byte budget cannot hold the mandatory 32-byte ownership marker.')
    parent = Path(parent).resolve(strict=True)
    _plain_parents(parent)
    if not parent.is_dir():
        raise ValueError('Choose an existing preview scratch parent.')
    if shutil.disk_usage(parent).free < max_cleanup_bytes:
        raise ValueError('Preview scratch free-space admission failed.')
    root = parent / f'preview-{uuid.uuid4().hex}'
    root.mkdir(mode=0o700, exist_ok=False)
    marker = uuid.uuid4().hex.encode('ascii')
    workspace = PreviewWorkspace(root, max_cleanup_files, max_cleanup_bytes, _file_id(root)[:2], marker)
    try:
        with (root / '.preview-owner').open('xb') as stream:
            stream.write(marker)
        return workspace
    except BaseException as primary:
        # Only paths created by this function exist at this point.
        try:
            (root / '.preview-owner').unlink(missing_ok=True)
            workspace._marker_retired = True
            if _file_id(root)[:2] != workspace._root_id:
                raise ValueError('Owned preview workspace was replaced before initialization cleanup.')
            root.rmdir()
        except (OSError, ValueError, RuntimeError, MemoryError, KeyboardInterrupt, SystemExit) as cleanup:
            workspace.cleanup_error = f'{type(cleanup).__name__}: {cleanup}'[:4096]
            raise PreviewPreparationError(workspace, primary, cleanup) from primary
        raise


@dataclass(frozen=True)
class PreparedPreview:
    job: JobRequest
    workspace: PreviewWorkspace
    identity: str
    requested_settings_json: bytes
    effective_settings_json: bytes
    provenance_json: bytes
    frame: int
    stage: str
    source_sha256: str
    source: PreviewSource = field(repr=False)

    def check_source(self, check: Check | None = None) -> None:
        """Re-admit the exact frame content before dispatch and after execution."""
        self.source.check(self.frame, check)
        admitted = admit_image(self.source.index.path_at(self.frame), expected_hw=None,
                               max_pixels=self.source.max_pixels, max_bytes=self.source.max_encoded_bytes, check=check)
        if admitted.sha256 != self.source_sha256:
            raise ValueError('Preview source content changed after job preparation.')
        self.source.check(self.frame, check)


@dataclass(frozen=True)
class PreviewRecipe:
    source: PreviewSource
    scratch_parent: Path
    draft_scale: float | None
    max_cleanup_files: int
    max_cleanup_bytes: int

    @property
    def context(self) -> PreviewContext:
        return PreviewContext(self.source.source_identity, self.source.requested_settings_json,
                              self.source.provenance_json, self.source.source_start, self.source.source_end)

    @property
    def stages(self) -> tuple[PreviewStage, ...]:
        config, writer = self.source.original_job.configurations()
        stages = []
        for name, scale in ((('draft', self.draft_scale), ('final', config.model_scale))
                            if self.draft_scale is not None else (('final', config.model_scale),)):
            effective = json.loads(_json(config))
            effective.update(model_scale=scale, output_format='exr', temp_dir='owned_per_request_scratch')
            stages.append(PreviewStage.capture(name, {'inference_settings': effective,
                          'writer_settings': json_value(writer), 'output_dir': 'owned_per_request_exports',
                          'temporal_context': 'full original timeline and groups; selected output only',
                          'reservation_scope': 'Native pixel float-buffer estimate only; no model/GPU/RAM hard cap.'},
                          estimated_work_bytes=self.source.max_pixels * 16))
        return tuple(stages)

    def summary(self) -> dict[str, Any]:
        return {'source': self.source.original_job.input_path, 'source_start': self.source.source_start,
                'source_end': self.source.source_end, 'source_identity': self.source.source_identity,
                'draft_scale': self.draft_scale, 'final_scale': self.source.original_job.configurations()[0].model_scale,
                'scratch_parent': str(self.scratch_parent), 'max_cleanup_entries': self.max_cleanup_files,
                'max_cleanup_bytes': self.max_cleanup_bytes, 'warning': WARNING}

    def job_for(self, frame: int, stage: Any, check: Check | None = None) -> PreparedPreview:
        check = check or (lambda: None)
        self.source.check(frame, check)
        if not isinstance(stage, (str, PreviewStage)):
            raise TypeError('Preview stage must be a configured name or PreviewStage instance.')
        name = stage if isinstance(stage, str) else stage.name
        if isinstance(stage, PreviewStage) and stage not in self.stages:
            raise ValueError('Preview stage does not match the captured recipe.')
        if name not in ('draft', 'final') or (name == 'draft' and self.draft_scale is None):
            raise ValueError('Choose one of this recipe\'s explicitly configured preview stages.')
        config, writer = self.source.original_job.configurations()
        admitted = admit_image(self.source.index.path_at(frame), expected_hw=None, max_pixels=self.source.max_pixels,
                               max_bytes=self.source.max_encoded_bytes, check=check)
        self.source.check(frame, check)
        # Conservative uncompressed output admission; runtime scratch/model allocations
        # remain unbounded by this estimate and are reported as such.
        output_estimate = admitted.shape[0] * admitted.shape[1] * 16 * len(config.outputs) + 1024**2
        if output_estimate > self.max_cleanup_bytes:
            raise ValueError('Estimated native EXR outputs exceed preview scratch admission budget.')
        workspace = create_preview_workspace(self.scratch_parent, max_cleanup_files=self.max_cleanup_files,
                                             max_cleanup_bytes=self.max_cleanup_bytes, check=check)
        try:
            effective = replace(config, output_format='exr', temp_dir=workspace.root / 'scratch',
                                model_scale=self.draft_scale if name == 'draft' else config.model_scale)
            job = JobRequest.create(self.source.original_job.input_path, workspace.root / 'exports', effective, writer,
                                    initial_hint_path=self.source.original_job.initial_hint_path,
                                    hint_video_path=self.source.original_job.hint_video_path,
                                    profile_stages=self.source.original_job.profile_stages)
            effective_json = _json(asdict(job))
            provenance = json.loads(self.source.provenance_json)
            provenance.update(stage=name, source_frame=frame, source_sha256=admitted.sha256,
                              native_hw=list(admitted.shape), execution_range=[self.source.source_start, self.source.source_end], selected_source_frame=frame,
                              effective_settings_identity=_digest(effective_json))
            provenance_json = _json(provenance)
            # Identity omits ephemeral scratch/output paths so compatible jobs can reuse thumbnails.
            identity = _digest(_json({'context': self.source.source_identity, 'requested': _digest(self.source.requested_settings_json),
                                      'provenance': _digest(self.source.provenance_json), 'frame': frame, 'stage': name,
                                      'scale': effective.model_scale, 'sha256': admitted.sha256}))
            check()
            return PreparedPreview(job, workspace, identity, self.source.requested_settings_json,
                                   effective_json, provenance_json, frame, name, admitted.sha256, self.source)
        except BaseException as primary:
            try:
                workspace.cleanup()
            except (OSError, ValueError, RuntimeError, MemoryError, KeyboardInterrupt, SystemExit) as cleanup:
                raise PreviewPreparationError(workspace, primary, cleanup) from primary
            raise


def prepare_preview(job: JobRequest, *, scratch_parent: Path, declared_fps: float,
                    draft_scale: float | None = None, model_references: dict[str, Any] | None = None,
                    max_entries: int = 100000, max_frames: int = 100000,
                    max_pixels: int = MAX_PIXELS, max_encoded_bytes: int = MAX_ENCODED_BYTES,
                    max_cleanup_files: int = MAX_CLEANUP_FILES, max_cleanup_bytes: int = MAX_CLEANUP_BYTES,
                    check: Check | None = None) -> PreviewRecipe:
    """Capture a recipe without creating scratch, decoding pixels or loading models."""
    check = check or (lambda: None)
    check()
    if not isinstance(job, JobRequest):
        raise TypeError('Preview preparation requires a shared JobRequest instance.')
    job_from_wire(asdict(job))  # Validate strict original JSON before any normalizing serialization.
    config, _ = job.configurations()
    from .shot_profiles import reject_derived_profiles
    reject_derived_profiles(config, 'Full-context progressive preview')
    if not Path(job.input_path).is_dir():
        raise ValueError('Progressive preview currently supports image sequences only; video/VFR is unsupported.')
    if config.start_mode != 'begin' or config.quick_key:
        raise ValueError('Full-context preview requires explicit begin/In and no Quick Key; context cannot be silently discarded.')
    if isinstance(declared_fps, bool) or not isinstance(declared_fps, (int, float)) or not math.isfinite(declared_fps) or declared_fps <= 0 or declared_fps != config.frame_dir_fps:
        raise ValueError('Explicit declared sequence FPS must equal the captured job FPS.')
    for value, maximum, label in ((max_entries, 100000, 'Sequence entries'), (max_frames, 100000, 'Sequence frames'),
                                  (max_pixels, MAX_PIXELS, 'Native pixels'), (max_encoded_bytes, MAX_ENCODED_BYTES, 'Encoded bytes'),
                                  (max_cleanup_files, MAX_CLEANUP_FILES, 'Cleanup entries'), (max_cleanup_bytes, MAX_CLEANUP_BYTES, 'Cleanup bytes')):
        _positive(value, maximum, label)
    if draft_scale is not None and (isinstance(draft_scale, bool) or not isinstance(draft_scale, (int, float)) or not math.isfinite(draft_scale) or not 0 < draft_scale < config.model_scale):
        raise ValueError('Draft scale must be explicitly finite, positive and below the original requested scale.')
    parent = Path(scratch_parent).resolve(strict=True)
    _plain_parents(parent)
    if not parent.is_dir():
        raise ValueError('Choose an existing preview scratch parent.')
    protected = [Path(job.input_path).resolve(), Path(job.output_dir).resolve()]
    refs = json.loads(_json(model_references or {}, 16 * 1024))
    if not isinstance(refs, dict):
        raise TypeError('Model references must be a JSON object.')
    for name in ('asset_root', 'weights'):
        if refs.get(name) is not None and not isinstance(refs[name], str):
            raise TypeError('Model root/weights references must be path strings or null.')
        if isinstance(refs.get(name), str):
            protected.append(Path(refs[name]).resolve())
    if any(parent == path or parent.is_relative_to(path) for path in protected):
        raise ValueError('Preview scratch must be separate from source, export destination and model references.')
    model_paths = _model_paths(refs)
    model_pins = _pin_models(refs, check)
    root = Path(job.input_path)
    _plain_parents(root)
    before = _file_id(root)
    index = scan_sequence(root, check=check, max_entries=max_entries, max_frames=max_frames)
    source_ctimes = []
    for path, expected in zip(index.paths, index.fingerprints, strict=True):
        check()
        actual = _file_id(Path(path))
        if actual[:4] != expected:
            raise ValueError('Source frame changed during preparation.')
        source_ctimes.append(actual[-1])
    if _file_id(root) != before:
        raise ValueError('Source sequence changed during preparation.')
    start = config.start_frame or 0
    end = min(len(index.paths), start + config.num_frames) if config.num_frames > 0 else len(index.paths)
    if not 0 <= start < end:
        raise ValueError('Requested preview source range is empty or outside the sequence.')
    # Reuse pinned-file admission for every context/dependency, not only the displayed frame.
    # Read/hash budget is per capture/check; it does not bound inference allocations.
    paths = list(index.paths)
    directories = [(str(root), before)]
    if job.initial_hint_path is not None:
        paths.append(job.initial_hint_path)
    if job.hint_video_path is not None:
        hint = Path(job.hint_video_path)
        _plain_parents(hint)
        if hint.is_dir():
            protected.append(hint)
            hint_identity = _file_id(hint)
            hint_index = scan_sequence(hint, check=check, max_entries=max_entries, max_frames=max_frames)
            paths.extend(hint_index.paths)
            directories.append((str(hint), hint_identity))
        else:
            paths.append(str(hint))
    document = None
    correction_pin = None
    if config.corrections_path is not None:
        from .corrections import MAX_CORRECTION_BYTES, CorrectionDocument, _unique_pairs
        pin = admit_file(config.corrections_path)
        if pin.size > MAX_CORRECTION_BYTES:
            raise ValueError('Preview correction document exceeds its metadata byte bound')
        digest, raw = read_file(pin, check=check, capture=True)
        correction_pin = replace(pin, sha256=digest)
        document = CorrectionDocument.parse(json.loads(raw, object_pairs_hook=_unique_pairs,
            parse_float=finite_json_float, parse_constant=finite_json_float), Path(config.corrections_path).parent)
        paths.extend((str(config.corrections_path), *map(str, document.asset_paths())))
    pins = []
    total = 0
    for path in dict.fromkeys(paths):
        check()
        pin = correction_pin if correction_pin is not None and path == correction_pin.path else admit_file(path)
        total += pin.size
        if len(pins) >= max_entries or pin.size > max_encoded_bytes or total > MAX_CLEANUP_BYTES:
            raise ValueError('Preview full-context dependencies exceed count/encoded/read byte bounds (512 MiB total)')
        digest, _ = read_file(pin, check=check)
        pins.append(replace(pin, sha256=digest))
        protected.append(Path(path))
    if any(parent == path or parent.is_relative_to(path) for path in protected):
        raise ValueError('Preview scratch must be separate from all context/guidance dependencies')
    if document is not None:
        from .correction_tracking_context import tracking_cuts, tracking_decode
        from .identities import sequence_identity
        hashes = {pin.path: pin.sha256 for pin in pins}
        actual_source = sequence_identity(job.input_path, [
            {'name': Path(path).name, 'sha256': hashes[path]} for path in index.paths])
        if actual_source['sha256'] != document.to_dict()['source']['sha256']:
            raise ValueError('Preview correction source content hash mismatch')
        document.verify_assets(check=check)
        document.validate_track_context(cuts=tracking_cuts(config), decode=tracking_decode(config, job.input_path))
        document.validate_guidance_context(decode=tracking_decode(config, job.input_path))
    requested = _json(asdict(job), 64 * 1024)
    dependency_identity = _digest(_json([{'path': pin.path, 'sha256': pin.sha256,
                                       'fingerprint': pin.fingerprint} for pin in pins], 32 * 1024**2))
    source_metadata_identity = _digest(_json({'sequence': index.identity, 'change_times': source_ctimes,
                                            'dependencies': dependency_identity}))
    provenance = _json({'schema_version': 1, 'kind': 'full_context_selected_progressive_preview',
                        'source_identity': source_metadata_identity,
                        'source_identity_scope': 'All source/context and hint/correction/asset content hashes with pinned metadata; rechecked before/after execution.',
                        'dependency_identity': dependency_identity, 'dependency_bytes': total,
                        'dependency_read_limit_bytes': MAX_CLEANUP_BYTES,
                        'requested_job_identity': _digest(requested), 'requested_range': [start, end],
                        'requested_frame_count': config.num_frames, 'declared_fps': declared_fps,
                        'model_references': refs, 'model_compatibility': 'NOT VERIFIED',
                        'model_metadata_identity': _digest(_json(model_pins, 1024**2)),
                        'model_metadata_pins': len(model_pins), 'model_paths': model_paths,
                        'model_identity_scope': 'Bounded path/stat pins, including nested files; no content hashes/authenticity guarantee.',
                        'restarted_temporal_context': False, 'warning': WARNING})
    source = PreviewSource(job, index, start, end, requested, provenance, source_metadata_identity, max_pixels,
                           max_encoded_bytes, before, tuple(source_ctimes), _json(refs), model_paths, model_pins,
                           tuple(pins), tuple(directories))
    source.check(start, check)
    check()
    recipe = PreviewRecipe(source, parent, draft_scale, max_cleanup_files, max_cleanup_bytes)
    _ = recipe.context, recipe.stages
    return recipe


def selected_preview_view(prepared: PreparedPreview, record: dict[str, Any]):
    """Admit sparse preview success without weakening ordinary completed export admission."""
    from .desktop_comparison import RenderView
    config, writer = prepared.job.configurations()
    diagnostics = record.get('diagnostics', {})
    if not isinstance(diagnostics, dict):
        raise TypeError('Selected preview diagnostics must be an object')
    source = diagnostics.get('source')
    selection = diagnostics.get('preview_selection')
    if (record.get('status') != 'preview_complete' or record.get('input') != prepared.job.input_path
            or record.get('output_dir') != prepared.job.output_dir
            or record.get('hints') != {'first_frame': prepared.job.initial_hint_path, 'full': prepared.job.hint_video_path}
            or record.get('requested_settings') != json_value(config) or record.get('writer_settings') != json_value(writer)
            or not isinstance(source, dict) or not isinstance(selection, dict)
            or source.get('start_frame') != prepared.source.source_start
            or source.get('num_frames') != prepared.source.source_end - prepared.source.source_start
            or selection != {'source_frame': prepared.frame, 'relative_frame': prepared.frame - prepared.source.source_start,
                'context_frames_processed': source['num_frames'], 'captured_frames': 1, 'full_context_processed': True}
            or selection.get('full_context_processed') is not True
            or type(diagnostics.get('frames_submitted')) is not int or diagnostics['frames_submitted'] != 1
            or any(type(selection.get(key)) is not int for key in
                   ('source_frame', 'relative_frame', 'context_frames_processed', 'captured_frames'))
            or type(source.get('start_frame')) is not int or type(source.get('num_frames')) is not int):
        raise ValueError('Preview requires its matching successful full-context selected capture')
    hw, fps = source.get('native_hw'), source.get('native_fps')
    outputs = diagnostics.get('outputs')
    if (not isinstance(hw, (list, tuple)) or len(hw) != 2 or any(type(v) is not int or v < 1 for v in hw)
            or isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
            or not isinstance(outputs, dict) or set(outputs) != config.outputs
            or any(not isinstance(path, str) or not Path(path).resolve().is_relative_to(Path(prepared.job.output_dir).resolve()) for path in outputs.values())):
        raise ValueError('Selected preview has invalid geometry/timing/output paths')
    for path in outputs.values():
        _plain_parents(Path(path))
    if list(hw) != json.loads(prepared.provenance_json)['native_hw'][:2] or fps != config.frame_dir_fps:
        raise ValueError('Selected preview geometry/timing differs from captured source')
    return RenderView(prepared.job.input_path, prepared.frame, 1, hw[0], hw[1], float(fps), tuple(outputs.items()),
                      True, config.frame_dir_linear, config.frame_dir_alpha, config.source_colour_profile)
