"""Strict portable projects, explicit selected collection and exact relinking.

Legacy desktop schema1 is untouched. Portable schema2 preserves the original
wire job, source basenames/order/offsets and correction semantic identity.
Inventory reads only bounded correction/project metadata (4MiB/32MiB). Verification reads all
required render bytes under an explicit per-pass payload budget; publish/Apply
repeat authentication of that admitted extent. read_bytes reports admitted bytes
charged, including a failed dependency's full charge. Model presence is NOT model
compatibility. No model, sidecar, output, cache or environment is collected.

ProjectInspection.apply requires explicit output AND scratch locations. It
revalidates content and materializes a fresh path-only correction snapshot; it
never edits the package, original document, queued job or asset selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .corrections import MAX_CORRECTION_BYTES, CorrectionDocument
from .diagnostics import finite_json_float
from .identities import verify_file_observation
from .portable_paths import (
    archive_job,
    archived_path,
    infer_flavour,
    native_destination,
)
from .project_collection import (
    MAX_FILES,
    FilePin,
    admit_file,
    assert_pin,
    check_now,
    cleanup_owned,
    fingerprint,
    fresh_directory,
    read_file,
    real_path,
    safe_name,
)
from .runtime_paths import RuntimePaths
from .scheduling import frame_path_key
from .session import JobRequest
from .worker_protocol import job_from_wire, job_to_wire

MAX_PROJECT_BYTES = 32 * 1024 * 1024
MAX_PATH_BYTES = 32 * 1024 * 1024
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.exr'}
COMMIT_MARKER = 'COMPLETE'
Check = Callable[[], None]


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate portable JSON field')
        result[key] = value
    return result


def _parse(raw: bytes) -> dict[str, Any]:
    def invalid(_value):
        raise ValueError('Nonfinite portable JSON number')
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid, parse_float=finite_json_float)
    except RecursionError as exc:
        raise ValueError('Portable JSON nesting exceeds decoder limits') from exc
    if not isinstance(value, dict):
        raise TypeError('Portable metadata must be an object')
    return value


def _metadata(pin: FilePin, maximum: int, check: Check | None) -> bytes:
    if pin.size > maximum:
        raise ValueError('Metadata byte limit exceeded')
    assert_pin(pin)
    path_before = Path(pin.path).lstat()
    if fingerprint(path_before) != pin.fingerprint:
        raise ValueError('Metadata changed before read')
    fd = os.open(pin.path, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    try:
        before = os.fstat(fd)
        fingerprint(before)
        verify_file_observation(pin.path, path_before, before, before)
        blocks, remaining = [], pin.size
        while remaining:
            check_now(check)
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                raise ValueError('Metadata shrank during read')
            blocks.append(block)
            remaining -= len(block)
        if os.read(fd, 1):
            raise ValueError('Metadata changed during read')
        verify_file_observation(pin.path, path_before, before, os.fstat(fd))
        assert_pin(pin)
        return b''.join(blocks)
    finally:
        os.close(fd)


def _budget(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f'{name} must be an exact nonnegative byte count below 2**63')


def _frame(frame: int) -> None:
    if type(frame) is not int or not 0 <= frame < 2**63:
        raise ValueError('Preview frame must be an exact nonnegative integer')


def _names(files: list[dict[str, Any]]) -> None:
    seen = set()
    for row in files:
        name = safe_name(row['relative_path'])
        key = unicodedata.normalize('NFC', name).casefold()
        if key in seen:
            raise ValueError('Casefold/Unicode-normalization filename collision')
        seen.add(key)


def _scan(path: Path, check: Check | None) -> tuple[str, tuple[FilePin, ...]]:
    root = real_path(path)
    if root.is_dir():
        pins = []
        entries = 0
        with os.scandir(root) as directory:
            for entry in directory:
                check_now(check)
                entries += 1
                if entries > MAX_FILES:
                    raise ValueError('Dependency directory entry limit exceeded')
                # Flat supported image frames only: never collect sidecars or recurse.
                if Path(entry.name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    continue
                safe_name(entry.name)
                pins.append(admit_file(root / entry.name))
        if not pins:
            raise ValueError('No supported image files in dependency directory')
        pins.sort(key=lambda pin: frame_path_key(Path(pin.path)))
        return 'sequence', tuple(pins)
    return 'file', (admit_file(root),)


def _assets(data: dict[str, Any]):
    for entry in data['entries']:
        if 'asset' in entry:
            yield entry['asset']
    for track in data.get('tracks', []):
        for row in track['frames']:
            yield row['asset']
    for origin in data.get('guidance', []):
        for resource in origin['resources']:
            yield resource['asset']


def _model_refs(asset_root: Path | None, weights: Path | None) -> dict[str, Any]:
    if asset_root is None and weights is None:
        return {'asset_root': None, 'weights': None, 'entries': [], 'scope': 'Model selection unspecified; runtime compatibility and inference NOT VERIFIED.'}
    root = Path(asset_root) if asset_root is not None else Path(__file__).resolve().parents[1]
    paths = RuntimePaths.from_package(Path(__file__).resolve().parents[1], weights, asset_root=root)
    entries = []
    for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir'):
        path = getattr(paths, name)
        try:
            info = path.lstat()
            status = 'present; contents NOT VERIFIED' if path.is_file() or path.is_dir() else 'wrong type'
            size = info.st_size if path.is_file() else None
        except OSError:
            status, size = 'missing', None
        entries.append({'id': name, 'path': str(path), 'status': status, 'bytes': size})
    return {'asset_root': str(root.absolute()), 'weights': str(Path(weights).absolute()) if weights is not None else None,
            'entries': entries, 'scope': 'Presence metadata only; weights/tensor compatibility, hashes, loading and inference NOT VERIFIED. Never collected.'}


def _inventory(job: JobRequest, preview_frame: int, asset_root, weights, check):
    if not isinstance(job, JobRequest):
        raise TypeError('Expected a JobRequest instance, not a dataclass class')
    _frame(preview_frame)
    # Validate original embedded JSON before any codec can normalize duplicate keys.
    job = job_from_wire(asdict(job))
    wire = job_to_wire(job)
    config, _ = job.configurations()
    from .shot_profiles import reject_derived_profiles
    reject_derived_profiles(config, 'Portable source/scratch relocation')
    deps: list[dict[str, Any]] = []
    pins: dict[str, tuple[FilePin, ...]] = {}
    total_files = total_paths = 0

    def add(identifier: str, role: str, path: Path, *, expected_hash: str | None = None):
        nonlocal total_files, total_paths
        check_now(check)
        lexical = Path(os.path.abspath(path))
        row = {'id': identifier, 'role': role, 'path': str(lexical), 'kind': 'file',
               'status': 'available; content NOT VERIFIED', 'required': True, 'depends_on': [],
               'files': [], 'file_count': 0, 'bytes': 0, 'collected': False}
        try:
            kind, admitted = _scan(lexical, check)
            if role in ('corrections', 'correction_asset') and kind != 'file':
                raise ValueError('Correction dependency must be a single file')
            rows = [{'relative_path': Path(pin.path).name, 'bytes': pin.size} for pin in admitted]
            _names(rows)
            total_files += len(rows)
            total_paths += sum(len(pin.path.encode('utf-8')) for pin in admitted)
            if total_files > MAX_FILES or total_paths > MAX_PATH_BYTES:
                raise ValueError('Project inventory file/path budget exceeded')
            row.update(kind=kind, files=rows, file_count=len(rows), bytes=sum(pin.size for pin in admitted))
            pins[identifier] = admitted
            if expected_hash is not None:
                row['files'][0]['sha256'] = expected_hash
        except (OSError, ValueError) as exc:
            row['status'] = f'unavailable: {type(exc).__name__}: {exc}'
        deps.append(row)
        return row

    source = add('source', 'source', Path(job.input_path))
    if job.initial_hint_path is not None:
        add('initial_hint', 'initial_hint', Path(job.initial_hint_path))
    if job.hint_video_path is not None:
        add('full_hint', 'full_hint', Path(job.hint_video_path))
    correction = None
    if config.corrections_path is not None:
        row = add('corrections', 'corrections', Path(config.corrections_path))
        if 'corrections' in pins:
            try:
                raw = _metadata(pins['corrections'][0], MAX_CORRECTION_BYTES, check)
                doc = CorrectionDocument.parse(_parse(raw), Path(row['path']).parent)
                ids, by_path = [], {}
                for asset in _assets(doc.to_dict()):
                    path = Path(os.path.abspath(doc.base_dir / asset['path']))
                    key = str(path)
                    if key in by_path:
                        identifier, expected = by_path[key]
                        if asset['sha256'] != expected:
                            raise ValueError('One correction path has conflicting hashes')
                    else:
                        identifier = f'correction_asset_{len(by_path):06d}'
                        by_path[key] = identifier, asset['sha256']
                        add(identifier, 'correction_asset', path, expected_hash=asset['sha256'])
                    ids.append(identifier)
                row['depends_on'] = list(dict.fromkeys(ids))
                correction = {'document': doc.to_dict(), 'identity': doc.identity, 'asset_ids': ids}
            except (OSError, ValueError, TypeError) as exc:
                row['status'] = f'unavailable: {type(exc).__name__}: {exc}'
    if source['kind'] == 'sequence' and source['file_count']:
        start = config.start_frame or 0
        if config.start_mode != 'begin' or start >= source['file_count']:
            source['status'] = 'unavailable: source range/start mode outside frozen sequence'
    report = {'dependencies': deps, 'model_references': _model_refs(asset_root, weights),
              'preview_frame': preview_frame, 'ready': False, 'read_bytes': 0, 'copy_bytes': 0,
              'collect_ids': [], 'requested_job': wire, 'correction_identity': None if correction is None else correction['identity']}
    return wire, report, pins, correction


def inventory_project(job: JobRequest, preview_frame: int, *, asset_root: Path | None = None,
                      weights: Path | None = None, check: Check | None = None) -> dict[str, Any]:
    """Metadata-only bounded inventory, including precise missing dependencies."""
    return _inventory(job, preview_frame, asset_root, weights, check)[1]


@dataclass(frozen=True)
class ProjectPlan:
    manifest_json: bytes
    pins: tuple[tuple[str, tuple[FilePin, ...]], ...]
    summary_json: bytes

    @property
    def summary(self) -> dict[str, Any]:
        return _parse(self.summary_json)


def _source_hash(dep: dict[str, Any]) -> str:
    if dep['kind'] == 'file':
        return dep['files'][0]['sha256']
    rows = [{'name': row['relative_path'], 'sha256': row['sha256']} for row in dep['files']]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def _correction_timing(wire, source, correction):
    if source['kind'] != 'sequence':
        return  # Native video timing requires the shared decoder at render time.
    config = archive_job(wire).config
    start = config.start_frame or 0
    count = source['file_count'] - start
    if config.num_frames > 0:
        count = min(count, config.num_frames)
    stored = correction['document']['source']
    if (stored['start_frame'], stored['frame_count'], stored['fps']) != (start, count, config.frame_dir_fps):
        raise ValueError('Correction selected sequence offsets/count/FPS differ from requested job')


def plan_project(job: JobRequest, preview_frame: int, *, collect_ids: Iterable[str] = (),
                 read_budget: int, copy_budget: int, asset_root: Path | None = None,
                 weights: Path | None = None, check: Check | None = None) -> ProjectPlan:
    _budget(read_budget, 'read_budget')
    _budget(copy_budget, 'copy_budget')
    if isinstance(collect_ids, (str, bytes, type)):
        raise TypeError('collect_ids requires a bounded iterable of dependency IDs')
    selected = set()
    for selection_count, identifier in enumerate(collect_ids, 1):
        if not isinstance(identifier, str) or selection_count > MAX_FILES:
            raise ValueError('Invalid/beyond-bound collection IDs')
        selected.add(identifier)
    wire, report, pins, correction = _inventory(job, preview_frame, asset_root, weights, check)
    deps = report['dependencies']
    if selected - {dep['id'] for dep in deps}:
        raise ValueError('Unknown collection dependency ID (models/caches are not collection dependencies)')
    if 'corrections' in selected:
        selected.update(next(dep for dep in deps if dep['id'] == 'corrections')['depends_on'])
    if any(dep['id'] not in pins or dep['status'].startswith('unavailable') for dep in deps):
        raise ValueError('All render dependencies must be available before creating a verified plan; inspect inventory for missing/invalid resources')
    total = sum(dep['bytes'] for dep in deps)
    copied = sum(dep['bytes'] for dep in deps if dep['id'] in selected)
    if total > read_budget or copied > copy_budget:
        raise ValueError(f'Explicit byte budget exceeded: verification needs {total}, collection needs {copied}')
    verified = {}
    for dep in deps:
        results = []
        for row, pin in zip(dep['files'], pins[dep['id']], strict=True):
            digest, _ = read_file(pin, check=check)
            if row.get('sha256', digest) != digest:
                raise ValueError('Correction asset content hash mismatch')
            row['sha256'] = digest
            results.append(replace(pin, sha256=digest))
        verified[dep['id']] = tuple(results)
        dep['status'] = 'verified'
        dep['collected'] = dep['id'] in selected
    if correction is not None:
        if correction['document']['source']['sha256'] != _source_hash(deps[0]):
            raise ValueError('Correction source content identity differs from full frozen source')
        _correction_timing(wire, deps[0], correction)
    _fence(deps, verified, check)
    report.update(ready=True, read_bytes=total, copy_bytes=copied, collect_ids=sorted(selected))
    manifest = {'schema_version': 2, 'type': 'corridorkey_portable_project', 'original_job': wire,
                'preview_frame': preview_frame, 'dependencies': deps,
                'model_references': report['model_references'], 'corrections': correction}
    raw = _json(manifest)
    if len(raw) > MAX_PROJECT_BYTES:
        raise ValueError('Portable manifest exceeds 32MiB')
    return ProjectPlan(raw, tuple(verified.items()), _json(report))


def _validate_pin(pin, expected_path: Path, *, size=None, sha256=None) -> None:
    """Validate public Python records lexically, before any filesystem operation."""
    if not isinstance(pin, FilePin):
        raise TypeError('Expected a FilePin instance')
    if not isinstance(pin.path, str) or native_destination(pin.path) != expected_path:
        raise ValueError('Pin path differs from its native dependency association')
    fingerprint = pin.fingerprint
    if (not isinstance(fingerprint, tuple) or len(fingerprint) != 5
            or any(type(value) is not int for value in fingerprint) or fingerprint[2] < 0):
        raise ValueError('Invalid pin fingerprint')
    if not isinstance(pin.sha256, str) or (pin.sha256 != '' and not re.fullmatch(r'[0-9a-f]{64}', pin.sha256)):
        raise ValueError('Invalid pin content identity')
    if (size is not None and pin.size != size) or (sha256 is not None and pin.sha256 != sha256):
        raise ValueError('Pin size/content identity differs from frozen metadata')


def _validate_pin_association(frozen_deps, resolved_deps, raw_pins):
    if not isinstance(resolved_deps, list) or len(resolved_deps) != len(frozen_deps):
        raise ValueError('Resolved dependency inventory differs')
    if not isinstance(raw_pins, tuple) or len(raw_pins) != len(frozen_deps):
        raise TypeError('Expected complete immutable dependency pins')
    pins = {}
    for frozen, resolved, pair in zip(frozen_deps, resolved_deps, raw_pins, strict=True):
        if (not isinstance(resolved, dict) or resolved.get('status') != 'verified'
                or _json({key: value for key, value in resolved.items() if key not in ('path', 'status')})
                != _json({key: value for key, value in frozen.items() if key not in ('path', 'status')})):
            raise ValueError('Resolved dependency metadata differs from frozen manifest')
        if not isinstance(resolved.get('path'), str):
            raise TypeError('Expected a native dependency path string')
        root = native_destination(resolved['path'])
        if (not isinstance(pair, tuple) or len(pair) != 2 or pair[0] != frozen['id']
                or not isinstance(pair[1], tuple) or len(pair[1]) != len(frozen['files'])):
            raise ValueError('Pin dependency IDs/count/type differ from frozen manifest')
        for row, pin in zip(frozen['files'], pair[1], strict=True):
            path = root / row['relative_path'] if frozen['kind'] == 'sequence' else root
            if path.name != row['relative_path']:
                raise ValueError('Pin basename differs from frozen file')
            _validate_pin(pin, path, size=row['bytes'], sha256=row['sha256'])
        pins[frozen['id']] = pair[1]
    return pins


def _fence(deps, pins, check):
    for dep in deps:
        check_now(check)
        for pin in pins[dep['id']]:
            assert_pin(pin)
        if dep['kind'] == 'sequence':
            _, live = _scan(Path(dep['path']), check)
            if tuple(pin.path for pin in live) != tuple(pin.path for pin in pins[dep['id']]):
                raise ValueError('Frozen sequence membership changed')


def _protect(destination: Path, manifest: dict[str, Any]) -> None:
    candidate = real_path(destination, missing=True)
    roots = [dep['path'] for dep in manifest['dependencies']]
    refs = manifest['model_references']
    if refs.get('asset_root'):
        roots.append(refs['asset_root'])
    roots.extend(row['path'] for row in refs.get('entries', []))
    for reference in roots:
        if infer_flavour(reference) != ('windows' if os.name == 'nt' else 'posix'):
            continue  # Archival foreign references have no native filesystem meaning.
        root = Path(reference).resolve()
        if _locations_overlap(candidate, root):
            raise ValueError('Write location overlaps a protected dependency/model reference')


def _collection_dir(dep: dict[str, Any], flavour: str | None = None) -> str:
    parts = ['assets', dep['id']]
    if dep['kind'] == 'sequence':
        path = archived_path(dep['path'], flavour or infer_flavour(dep['path']))
        if path.stem.lower() in ('frames', 'rgb', 'input'):
            parts.append(safe_name(path.parent.name))
        parts.append(safe_name(path.name))
    return '/'.join(parts)


def publish_project(plan: ProjectPlan, destination_parent: Path, check: Check | None = None) -> Path:
    """Fresh package; exclusive zero-byte COMPLETE creation is the commit point.

    Metadata flush/fsync and cancellation finish before commit. Directory-entry
    survival across power loss is not claimed. No failure is reported afterward.
    """
    if not isinstance(plan, ProjectPlan):
        raise TypeError('Expected immutable ProjectPlan instance')
    manifest = _validate_manifest(plan.manifest_json)
    if archive_job(manifest['original_job']).flavour != ('windows' if os.name == 'nt' else 'posix'):
        raise ValueError('Foreign archival plans cannot be published directly. Apply with explicit native destinations, then create and review a new native plan.')
    pins = _validate_pin_association(manifest['dependencies'], manifest['dependencies'], plan.pins)
    parent = real_path(destination_parent, directory=True)
    _protect(parent / 'prospective-package', manifest)
    _fence(manifest['dependencies'], pins, check)
    root = fresh_directory(parent, 'corridorkey-project-')
    try:
        for dep in manifest['dependencies']:
            for row, pin in zip(dep['files'], pins[dep['id']], strict=True):
                destination = None
                if dep['collected']:
                    directory = root / _collection_dir(dep)
                    directory.mkdir(parents=True, exist_ok=True)
                    destination = directory / row['relative_path']
                digest, _ = read_file(pin, check=check, destination=destination)
                if digest != row['sha256']:
                    raise ValueError('Plan pins differ from frozen content identity')
            if dep['collected']:
                dep['package_path'] = _collection_dir(dep)
        _fence(manifest['dependencies'], pins, check)
        check_now(check)
        raw = _json(manifest)
        if len(raw) > MAX_PROJECT_BYTES:
            raise ValueError('Portable manifest byte budget exceeded')
        with (root / 'project.json').open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        check_now(check)
        marker_fd = os.open(root / COMMIT_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        # Successful exclusive creation is committed truth. Descriptor retirement
        # cannot turn a committed package into a reported failed publication.
        try:
            os.close(marker_fd)
        except OSError:
            pass
        return root
    except BaseException as exc:
        try:
            cleanup_owned(root)
        except OSError as cleanup_error:
            exc.add_note(f'Owned incomplete package cleanup failed: {cleanup_error}; path={root}')
        raise


def _validate_manifest(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or len(raw) > MAX_PROJECT_BYTES:
        raise ValueError('Portable manifest exceeds byte limit')
    data = _parse(raw)
    if set(data) != {'schema_version', 'type', 'original_job', 'preview_frame', 'dependencies', 'model_references', 'corrections'} or type(data['schema_version']) is not int or data['schema_version'] != 2 or data['type'] != 'corridorkey_portable_project':
        raise ValueError('Unsupported portable schema/type/fields')
    archive = archive_job(data['original_job'])
    wire = archive.original_wire
    _frame(data['preview_frame'])
    deps = data['dependencies']
    if not isinstance(deps, list) or not 1 <= len(deps) <= MAX_FILES:
        raise ValueError('Invalid portable dependencies')
    ids = set()
    count = paths = 0
    for dep in deps:
        if not isinstance(dep, dict):
            raise TypeError('Portable dependency must be an object')
        fields = {'id', 'role', 'path', 'kind', 'status', 'required', 'depends_on', 'files', 'file_count', 'bytes', 'collected'}
        if dep.get('collected') and 'package_path' in dep:
            fields.add('package_path')
        if not isinstance(dep, dict) or set(dep) != fields:
            raise ValueError('Unknown/missing dependency fields')
        identifier = dep['id']
        if not isinstance(identifier, str) or not re.fullmatch(r'(source|initial_hint|full_hint|corrections|correction_asset_[0-9]{6})', identifier) or identifier in ids:
            raise ValueError('Invalid/duplicate portable dependency ID')
        ids.add(identifier)
        if dep['role'] in ('corrections', 'correction_asset') and dep['kind'] != 'file':
            raise ValueError('Correction dependencies must be single files')
        if dep['role'] != ('correction_asset' if identifier.startswith('correction_asset_') else identifier) or dep['required'] is not True or type(dep['collected']) is not bool or dep['kind'] not in ('file', 'sequence') or dep['status'] != 'verified':
            raise ValueError('Invalid portable role/kind/state')
        archived_path(dep['path'], archive.flavour)
        if 'package_path' in dep and dep['package_path'] != _collection_dir(dep, archive.flavour):
            raise ValueError('Unsafe collection package path')
        if not isinstance(dep['depends_on'], list) or any(not isinstance(item, str) for item in dep['depends_on']):
            raise ValueError('Invalid dependency closure')
        files = dep['files']
        if not isinstance(files, list) or not files or len(files) > MAX_FILES or (dep['kind'] == 'file' and len(files) != 1):
            raise ValueError('Invalid frozen dependency file list')
        if any(not isinstance(row, dict) or set(row) != {'relative_path', 'bytes', 'sha256'} for row in files):
            raise ValueError('Frozen file rows have invalid fields/types')
        _names(files)
        for row in files:
            if set(row) != {'relative_path', 'bytes', 'sha256'} or type(row['bytes']) is not int or not 0 <= row['bytes'] < 2**63 or not isinstance(row['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
                raise ValueError('Invalid frozen file metadata')
            paths += len(row['relative_path'].encode('utf-8'))
        count += len(files)
        if count > MAX_FILES or paths > MAX_PATH_BYTES or type(dep['file_count']) is not int or dep['file_count'] != len(files) or type(dep['bytes']) is not int or dep['bytes'] != sum(row['bytes'] for row in files):
            raise ValueError('Frozen list/count/size exceeds budget or differs')
        if dep['role'] == 'corrections' and dep['bytes'] > MAX_CORRECTION_BYTES:
            raise ValueError('Correction metadata exceeds 4MiB')
        if dep['kind'] == 'sequence' and [row['relative_path'] for row in files] != sorted((row['relative_path'] for row in files), key=lambda name: frame_path_key(Path(name))):
            raise ValueError('Frozen frame order differs from source scheduler')
    expected = {'source': wire['input_path']}
    if wire['initial_hint_path'] is not None:
        expected['initial_hint'] = wire['initial_hint_path']
    if wire['hint_video_path'] is not None:
        expected['full_hint'] = wire['hint_video_path']
    config = archive.config
    correction = data['corrections']
    if config.corrections_path is not None:
        expected['corrections'] = str(config.corrections_path)
        if not isinstance(correction, dict) or set(correction) != {'document', 'identity', 'asset_ids'}:
            raise ValueError('Missing/invalid correction relocation descriptor')
        doc = CorrectionDocument.parse(correction['document'])
        if not isinstance(correction['asset_ids'], list) or any(not isinstance(item, str) for item in correction['asset_ids']):
            raise ValueError('Correction asset IDs must be an ordered list')
        if doc.identity != correction['identity'] or len(correction['asset_ids']) != len(doc.asset_bindings()):
            raise ValueError('Correction identity/assets differ')
        mapping = {dep['id']: dep for dep in deps}
        for asset, identifier in zip(doc.asset_bindings(), correction['asset_ids'], strict=True):
            if not isinstance(identifier, str) or identifier not in mapping or mapping[identifier]['role'] != 'correction_asset' or mapping[identifier]['files'][0]['sha256'] != asset['sha256']:
                raise ValueError('Correction asset closure/hash differs')
        if set(correction['asset_ids']) != {dep['id'] for dep in deps if dep['role'] == 'correction_asset'}:
            raise ValueError('Unexpected/unbound correction resource')
        correction_dep = mapping.get('corrections')
        if correction_dep is None or correction_dep['depends_on'] != list(dict.fromkeys(correction['asset_ids'])):
            raise ValueError('Correction collection closure differs')
        _correction_timing(data['original_job'], mapping['source'], correction)
        if doc.to_dict()['source']['sha256'] != _source_hash(mapping['source']):
            raise ValueError('Correction source identity differs')
    elif correction is not None or any(dep['role'] == 'correction_asset' for dep in deps):
        raise ValueError('Unexpected correction relocation data')
    if {dep['id'] for dep in deps if dep['role'] != 'correction_asset'} != set(expected):
        raise ValueError('Wire job/dependency inventory differs')
    for dep in deps:
        if dep['role'] != 'correction_asset' and dep['path'] != expected[dep['id']]:
            raise ValueError('Wire job/reference path differs')
        if dep['role'] != 'corrections' and dep['depends_on']:
            raise ValueError('Unexpected dependency edge')
    refs = data['model_references']
    if not isinstance(refs, dict) or set(refs) != {'asset_root', 'weights', 'entries', 'scope'}:
        raise ValueError('Invalid model-reference metadata')
    for key in ('asset_root', 'weights'):
        if refs[key] is not None:
            archived_path(refs[key], archive.flavour)
    if not isinstance(refs['entries'], list) or len(refs['entries']) > 5 or not isinstance(refs['scope'], str):
        raise ValueError('Invalid model-reference inventory')
    for row in refs['entries']:
        if not isinstance(row, dict) or set(row) != {'id', 'path', 'status', 'bytes'} or not isinstance(row['path'], str) or row['id'] not in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir') or not isinstance(row['status'], str) or (row['bytes'] is not None and (type(row['bytes']) is not int or row['bytes'] < 0)):
            raise ValueError('Invalid model-reference row')
        archived_path(row['path'], archive.flavour)
    if archive.flavour != ('windows' if os.name == 'nt' else 'posix') and any(not dep['collected'] for dep in deps):
        raise ValueError('Foreign path-flavour packages require every render dependency to be collected; partial/reference-only foreign packages are unsupported.')
    return data


def _package(package: Path, check):
    root = real_path(package, directory=True)
    marker_pin = admit_file(root / COMMIT_MARKER)
    if marker_pin.size != 0:
        raise ValueError('Portable completion marker must be an empty regular single-link file')
    manifest_pin = admit_file(root / 'project.json')
    data = _validate_manifest(_metadata(manifest_pin, MAX_PROJECT_BYTES, check))
    assert_pin(marker_pin)
    return root, manifest_pin, marker_pin, data


def _resolve(root, data, relinks, check):
    if not isinstance(relinks, dict) or any(not isinstance(key, str) or not isinstance(value, (str, Path)) for key, value in relinks.items()):
        raise TypeError('Relinks require dependency ID -> explicit local path')
    if set(relinks) - {dep['id'] for dep in data['dependencies']}:
        raise ValueError('Unknown relink dependency ID')
    archive = archive_job(data['original_job'])
    foreign = archive.flavour != ('windows' if os.name == 'nt' else 'posix')
    report = {'archive_path_flavour': archive.flavour, 'foreign_paths': foreign,
              'portability_scope': ('Foreign all-collected dependencies only; external model/executable references remain unverified.' if foreign else 'Native path flavour; collected or explicit exact-identity references.'),
              'dependencies': [], 'model_references': data['model_references'], 'preview_frame': data['preview_frame'],
              'ready': False, 'read_bytes': 0, 'copy_bytes': 0, 'collect_ids': [],
              'requested_job': data['original_job'], 'correction_identity': None if data['corrections'] is None else data['corrections']['identity']}
    pins = {}
    for original in data['dependencies']:
        dep = json.loads(_json(original))
        identifier = dep['id']
        if identifier in relinks:
            path = native_destination(relinks[identifier]) if foreign else Path(relinks[identifier])
        elif dep['collected']:
            if 'package_path' not in dep:
                raise ValueError('Incomplete collected dependency descriptor')
            path = root / dep['package_path']
            if dep['kind'] == 'file':
                path /= dep['files'][0]['relative_path']
        else:
            path = Path(dep['path'])
        dep['path'] = str(Path(os.path.abspath(path)))
        try:
            kind, admitted = _scan(path, check)
            actual = [{'relative_path': Path(pin.path).name, 'bytes': pin.size} for pin in admitted]
            expected = [{key: row[key] for key in ('relative_path', 'bytes')} for row in dep['files']]
            if kind != dep['kind'] or actual != expected:
                raise ValueError('Frozen file names/order/sizes differ; relink requires exact identity')
            dep['status'] = 'available; content NOT VERIFIED'
            pins[identifier] = admitted
        except (OSError, ValueError) as exc:
            dep['status'] = f'unavailable: {type(exc).__name__}: {exc}'
        report['dependencies'].append(dep)
    return report, pins


def inventory_package(package: Path, *, relinks: dict[str, Path] | None = None,
                      check: Check | None = None) -> dict[str, Any]:
    """Read-only metadata preview. Never hashes footage/models or returns ready."""
    root, manifest_pin, marker_pin, data = _package(package, check)
    report = _resolve(root, data, relinks or {}, check)[0]
    assert_pin(manifest_pin)
    assert_pin(marker_pin)
    return report


@dataclass(frozen=True)
class ProjectInspection:
    manifest_json: bytes
    report_json: bytes
    pins: tuple[tuple[str, tuple[FilePin, ...]], ...]
    package_path: str
    manifest_pin: FilePin
    marker_pin: FilePin

    @property
    def report(self) -> dict[str, Any]:
        return _parse(self.report_json)

    @property
    def ready(self) -> bool:
        return self.report['ready']

    @property
    def preview_frame(self) -> int:
        return self.report['preview_frame']

    @property
    def model_references(self) -> dict[str, Any]:
        return self.report['model_references']

    def apply(self, *, output_dir: Path, temp_dir: Path, check: Check | None = None) -> JobRequest:
        return _apply(self, output_dir=output_dir, temp_dir=temp_dir, check=check)


def inspect_project(package: Path, *, relinks: dict[str, Path] | None = None,
                    read_budget: int, check: Check | None = None) -> ProjectInspection:
    _budget(read_budget, 'read_budget')
    root, manifest_pin, marker_pin, data = _package(package, check)
    report, pins = _resolve(root, data, relinks or {}, check)
    verified = {}
    remaining = read_budget
    for dep in report['dependencies']:
        if dep['id'] not in pins:
            continue
        if dep['bytes'] > remaining:
            dep['status'] = 'not verified: explicit read budget insufficient'
            continue
        try:
            results = []
            for row, pin in zip(dep['files'], pins[dep['id']], strict=True):
                digest, raw = read_file(pin, check=check, capture=dep['role'] == 'corrections')
                if digest != row['sha256']:
                    raise ValueError('Frozen content identity mismatch')
                if dep['role'] == 'corrections' and CorrectionDocument.parse(_parse(raw)).identity != data['corrections']['identity']:
                    raise ValueError('Correction descriptor/document semantic identity mismatch')
                results.append(replace(pin, sha256=digest))
            verified[dep['id']] = tuple(results)
            dep['status'] = 'verified'
        except (OSError, ValueError, TypeError) as exc:
            dep['status'] = f'unavailable: {type(exc).__name__}: {exc}'
        remaining -= dep['bytes']
        report['read_bytes'] += dep['bytes']
    assert_pin(manifest_pin)
    assert_pin(marker_pin)
    report['ready'] = len(verified) == len(report['dependencies'])
    if report['ready']:
        _fence(report['dependencies'], verified, check)
    return ProjectInspection(_json(data), _json(report), tuple(verified.items()), str(root), manifest_pin, marker_pin)


def _locations_overlap(first: Path, second: Path) -> bool:
    """Conservative portable case/NFC overlap plus native existing-inode checks."""
    def key(path):
        return tuple(unicodedata.normalize('NFC', part).casefold() for part in path.parts)
    left, right = key(first), key(second)
    if left[:len(right)] == right or right[:len(left)] == left:
        return True
    # Also fence native aliases (including case-insensitive filesystems) without
    # assuming that lexical path spelling identifies an existing directory.
    for path, ancestors in ((first, (second, *second.parents)), (second, (first, *first.parents))):
        if path.exists():
            for ancestor in ancestors:
                if ancestor.exists() and os.path.samefile(path, ancestor):
                    return True
    return False


def _apply(inspection, *, output_dir, temp_dir, check):
    if not isinstance(inspection, ProjectInspection):
        raise TypeError('Expected a ProjectInspection instance, not a dataclass class')
    report = inspection.report
    if report.get('ready') is not True:
        raise ValueError('A completely verified ProjectInspection instance is required before Apply')
    data = _validate_manifest(inspection.manifest_json)
    deps = report.get('dependencies')
    if not isinstance(deps, list):
        raise TypeError('Expected resolved dependency list')
    pins = _validate_pin_association(data['dependencies'], deps, inspection.pins)
    actual_provenance = {key: report.get(key) for key in
                         ('requested_job', 'model_references', 'preview_frame', 'correction_identity')}
    expected_provenance = {'requested_job': data['original_job'], 'model_references': data['model_references'],
                           'preview_frame': data['preview_frame'],
                           'correction_identity': None if data['corrections'] is None else data['corrections']['identity']}
    if _json(actual_provenance) != _json(expected_provenance):
        raise ValueError('Inspection provenance differs from frozen manifest')
    if not isinstance(inspection.package_path, str):
        raise TypeError('Expected native package path string')
    package = native_destination(inspection.package_path)
    _validate_pin(inspection.manifest_pin, package / 'project.json')
    _validate_pin(inspection.marker_pin, package / COMMIT_MARKER, size=0)
    assert_pin(inspection.manifest_pin)
    assert_pin(inspection.marker_pin)
    physical = _parse(_metadata(inspection.manifest_pin, MAX_PROJECT_BYTES, check))
    if _json(physical) != _json(data):
        raise ValueError('Inspection manifest differs from the pinned package manifest')
    effective = dict(data, dependencies=deps)
    archive = archive_job(data['original_job'])
    from .shot_profiles import reject_derived_profiles
    reject_derived_profiles(archive.config, 'Portable source/scratch relocation')
    foreign = archive.flavour != ('windows' if os.name == 'nt' else 'posix')
    if (foreign and archive.config.output_format == 'video' and archive.writer_executable_absolute
            and archive.writer_executable_flavour != ('windows' if os.name == 'nt' else 'posix')):
        raise ValueError('The archived video job refers to a foreign absolute FFmpeg executable. Choose an explicitly compatible encoder in the originating job and republish, or use an EXR package; the encoder reference is never silently rewritten.')
    output_dir = real_path(native_destination(output_dir) if foreign else output_dir, missing=True)
    temp_dir = real_path(native_destination(temp_dir) if foreign else temp_dir, missing=True)
    for path in (output_dir, temp_dir):
        package = Path(inspection.package_path)
        if _locations_overlap(path, package):
            raise ValueError('Output/scratch location overlaps portable package')
        _protect(path, effective)
        _protect(path, data)
    if _locations_overlap(output_dir, temp_dir):
        raise ValueError('Choose separate output and scratch locations')
    _fence(deps, pins, check)
    for dep in deps:
        for row, pin in zip(dep['files'], pins[dep['id']], strict=True):
            digest, _ = read_file(pin, check=check)
            if digest != row['sha256']:
                raise ValueError('Inspection pins differ from frozen content identity')
    _fence(deps, pins, check)
    original = data['original_job']
    wire = dict(original)
    mapping = {dep['id']: dep for dep in deps}
    wire['input_path'] = mapping['source']['path']
    wire['output_dir'] = str(output_dir)
    for field, identifier in (('initial_hint_path', 'initial_hint'), ('hint_video_path', 'full_hint')):
        wire[field] = mapping[identifier]['path'] if identifier in mapping else None
    config = json.loads(wire['config_json'])
    config['temp_dir'] = str(temp_dir)
    workspace = None
    pending = None
    path = None
    try:
        if data['corrections'] is not None:
            scratch = real_path(temp_dir, directory=True)
            workspace = fresh_directory(scratch, 'portable-corrections-')
            correction = data['corrections']
            document = json.loads(_json(correction['document']))
            for asset, identifier in zip(_assets(document), correction['asset_ids'], strict=True):
                asset['path'] = mapping[identifier]['path']
            relocated = CorrectionDocument.parse(document, workspace)
            if relocated.identity != correction['identity']:
                raise ValueError('Relocation changed correction semantic identity')
            path = workspace / 'corrections.json'
            pending = workspace / '.corrections.pending'
            with pending.open('xb') as stream:
                stream.write(relocated.payload.encode())
                stream.flush()
                os.fsync(stream.fileno())
            config['corrections_path'] = str(path)
        wire['config_json'] = json.dumps(config, sort_keys=True, allow_nan=False)
        result = job_from_wire(wire)
        _fence(deps, pins, check)
        assert_pin(inspection.manifest_pin)
        assert_pin(inspection.marker_pin)
        check_now(check)
        if pending is not None and path is not None:
            # This private 0700 workspace has one owner. A conflict is never
            # overwritten; no callback/native work runs between this test and
            # rename. Hostile concurrent writers are outside the contract.
            if os.path.lexists(path):
                raise FileExistsError('Fresh correction destination already exists')
            pending.rename(path)
        # Return is committed truth; no fallible callback/I/O follows rename.
        return result
    except BaseException as exc:
        if workspace is not None:
            try:
                cleanup_owned(workspace)
            except OSError as error:
                exc.add_note(f'Owned correction relocation cleanup failed: {error}; path={workspace}')
        raise
