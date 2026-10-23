"""Checked local transfer of owned recovery workspaces, without external media.

Backups are flat numbered blobs with a bounded manifest, never tar extraction.
Existing absolute dependency paths remain historical; copying is not inference or
proof of cross-machine replay compatibility. Partial destinations are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from . import cache_maintenance as maintenance
from .control import CancelledError

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
SCOPE = ('Copies owned recovery entries, staging, journals and reports only. External source/export/model paths '
         'remain unchanged historical references. Only recorded native absolute paths can be protected as destination parents. '
         'Transfer verification is not inference or cross-machine replay acceptance.')
Check = maintenance.Check


def _sync(path: Path) -> None:
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _parent(value: str | Path, source: Path) -> Path:
    path = Path(value).absolute()
    maintenance._stat(path, directory=True)
    path = path.resolve(strict=True)
    if path == source or path.is_relative_to(source):
        raise ValueError('Destination parent must be outside the source workspace/backup')
    if len(str(path)) > 4000:
        raise ValueError('Destination path exceeds its bounded length')
    return path


def _hash(path: Path, check: Check, *, expected_bytes: int) -> str:
    before = maintenance._stat(path)
    if before['bytes'] != expected_bytes:
        raise ValueError('File size changed before verification')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        remaining = expected_bytes
        while remaining:
            maintenance._check(check)
            block = stream.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError('File truncated during verification')
            remaining -= len(block)
            digest.update(block)
        if stream.read(1):
            raise ValueError('File grew beyond its admitted byte size')
    if maintenance._stat(path) != before:
        raise ValueError('File changed during verification')
    return digest.hexdigest()


def _copy(source: Path, destination: Path, expected: dict[str, Any], result: dict[str, Any], check: Check) -> str:
    before = maintenance._stat(source)
    if before['bytes'] != expected['bytes']:
        raise ValueError('Source size changed before copy')
    digest = hashlib.sha256()
    with source.open('rb') as reader, destination.open('xb') as writer:
        remaining = expected['bytes']
        while remaining:
            maintenance._check(check)
            block = reader.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError('Source truncated during copy')
            remaining -= len(block)
            written = writer.write(block)
            result['copied_bytes'] += written
            if written != len(block):
                raise OSError('Short file write during backup transfer')
            digest.update(block)
        if reader.read(1):
            raise ValueError('Source grew beyond its admitted byte size')
        writer.flush()
        os.fsync(writer.fileno())
    if maintenance._stat(source) != before or destination.stat().st_size != expected['bytes']:
        raise ValueError('Source changed during copy')
    actual = digest.hexdigest()
    if expected.get('sha256', actual) != actual:
        raise ValueError('Payload checksum mismatch during copy')
    os.utime(destination, ns=(expected['mtime_ns'], expected['mtime_ns']))
    result['copied_files'] += 1
    return actual


def _seal(path: Path, raw: bytes) -> None:
    # Prepared bytes become visible atomically; even a valid JSON prefix cannot
    # admit an unfinished restore. Rename consumes the stage without an alias.
    # Fresh UUID destination ownership excludes cooperating competitors. This
    # is not an atomic no-replace guarantee against hostile external editors.
    stage = path.with_name('.seal-' + uuid.uuid4().hex)
    with stage.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() or path.is_symlink():
        raise FileExistsError('Completion marker already exists')
    stage.rename(path)
    _sync(path.parent)


def _committed(path: Path | None, raw: bytes | None) -> bool:
    if path is None or raw is None:
        return False
    try:
        info = maintenance._stat(path)
        if info['bytes'] != len(raw):
            return False
        with path.open('rb') as stream:
            return stream.read(len(raw) + 1) == raw
    except (OSError, ValueError):
        return False


def _result() -> dict[str, Any]:
    return {'status': 'failed', 'destination': None, 'usable': False, 'copied_bytes': 0, 'copied_files': 0,
            'entry_count': None, 'journal_count': None, 'report_count': None, 'error': None, 'warning': None, 'scope': SCOPE}


def _failure(result: dict[str, Any], exc: BaseException, marker: Path | None, raw: bytes | None) -> dict[str, Any]:
    # Account for an interrupt after an OS write but before Python increments its
    # counters. The private destination is bounded, flat or shallow managed data.
    if result['destination'] is not None:
        try:
            pending = [Path(result['destination'])]
            total = count = visited = 0
            while pending:
                directory = pending.pop()
                maintenance._stat(directory, directory=True)
                for name in maintenance._names(directory, maintenance.MAX_FILES + 1, None):
                    path = directory / name
                    visited += 1
                    if visited > maintenance.MAX_FILES + 2 * maintenance.MAX_ENTRIES + 10:
                        raise ValueError('Partial destination accounting exceeds its bounds')
                    if path.is_dir() and not path.is_symlink():
                        maintenance._stat(path, directory=True)
                        pending.append(path)
                    elif path.name not in {'lock.sqlite', 'backup.json'} and not path.name.startswith('.seal-'):
                        total += maintenance._stat(path)['bytes']
                        count += 1
            result.update(copied_bytes=total, copied_files=count)
        except (OSError, ValueError) as accounting_error:
            result.update(copied_bytes=None, copied_files=None, warning=f'Partial accounting unavailable: {accounting_error}'[:4096])
    if _committed(marker, raw):
        try:
            if marker is None:
                raise ValueError('Missing completion marker')
            if marker.name == 'backup.json':
                _read_backup(marker.parent, None)
            else:
                _, snapshot = maintenance._scan(marker.parent, None)
                budget = [maintenance.MAX_METADATA_BYTES]
                for name in snapshot['files']:
                    if name.startswith('entries/') and name.endswith('/complete.json'):
                        manifest = maintenance._object(marker.parent / name, budget)[0]
                        for item in manifest['files']:
                            if _hash(marker.parent / Path(name).parent / item['blob'], None,
                                     expected_bytes=item['bytes']) != item['sha256']:
                                raise ValueError('Restored payload checksum mismatch after publication')
        except BaseException as verification_error:
            logging.getLogger(__name__).exception('Committed transfer did not pass final verification')
            result.update(status='failed', usable=False,
                          error=f'Completion exists but final verification failed: {verification_error}'[:4096])
        else:
            warning = f'Committed transfer; finalization reported {type(exc).__name__}: {exc}'
            result.update(status='complete', usable=True, warning=((result['warning'] or '') + ' ' + warning).strip()[:4096])
    else:
        result.update(status='cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed',
                      usable=False, error=f'{type(exc).__name__}: {exc}'[:4096])
    return result


def _summarize(result: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ('entry_count', 'journal_count', 'report_count'):
        result[key] = current[key]


def _protect_parent(parent: Path, metadata: dict[str, Path], check: Check) -> str | None:
    """Reject writes inside known external dependencies/exports without copying them."""
    budget = [maintenance.MAX_METADATA_BYTES]
    unknown = False
    nodes = 0
    for name, path in sorted(metadata.items()):
        if not (name == 'workspace.json' or (name.startswith('entries/') and name.endswith('/complete.json')) or
                name.startswith(('journals/', 'reports/'))):
            continue
        maintenance._check(check)
        try:
            value, _ = maintenance._object(path, budget)
        except (OSError, ValueError, TypeError, RecursionError):
            if not name.startswith(('journals/', 'reports/')):
                raise
            unknown = True
            continue
        pending = [value]
        while pending:
            maintenance._check(check)
            item = pending.pop()
            nodes += 1
            if nodes > 262144:
                raise ValueError('Recorded dependency traversal exceeds its bounded node limit')
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, str) and Path(item).is_absolute():
                if len(item) > 4096:
                    unknown = True
                    continue
                protected = Path(item).resolve(strict=False)
                if parent == protected or parent.is_relative_to(protected):
                    raise ValueError('Destination parent overlaps a recorded source, hint, model or render/export path')
    return ('Some journal/report provenance could not be decoded; unrecorded or undecodable dependency paths cannot be protected.'
            if unknown else None)


def backup_workspace(source: str | Path, existing_parent: str | Path, *, check: Check = None) -> dict[str, Any]:
    """Create a new immutable-by-convention backup; never prune or change source."""
    result = _result()
    marker = raw = None
    try:
        with maintenance._locked(source, check) as root:
            current, snapshot = maintenance._scan(root, check)
            parent = _parent(existing_parent, root)
            result['warning'] = _protect_parent(parent, {name: root / name for name in snapshot['files']}, check)
            _summarize(result, current)
            destination = Path(tempfile.mkdtemp(prefix='recovery-backup-', dir=parent))
            result['destination'] = str(destination)
            _sync(parent)
            directories = [{'name': name, 'mtime_ns': info['mtime_ns']}
                           for name, info in sorted(snapshot['directories'].items())]
            files = []
            for name, info in sorted(snapshot['files'].items()):
                if name == 'lock.sqlite':
                    continue
                maintenance._check(check)
                item = {'name': name, 'blob': f'blob-{len(files):06d}.bin', 'bytes': info['bytes'], 'mtime_ns': info['mtime_ns']}
                item['sha256'] = _copy(root / name, destination / item['blob'], item, result, check)
                files.append(item)
            after, _ = maintenance._scan(root, check)
            if after['token'] != current['token']:
                raise ValueError('Workspace snapshot changed during backup')
            payload = {'schema_version': 1, 'kind': 'corridorkey_recovery_backup', 'owner': current['owner'],
                       'source_workspace': str(root), 'source_token': current['token'],
                       'directories': directories, 'files': files}
            raw = maintenance._canonical({'payload': payload, 'sha256': maintenance._digest(payload)})
            if len(raw) > MAX_MANIFEST_BYTES:
                raise ValueError('Backup manifest exceeds its bounded byte limit')
            # Verify the same format/semantic envelope as restore before sealing.
            _validate(destination, payload, check, sealed=False)
            maintenance._check(check)
            marker = destination / 'backup.json'
            _seal(marker, raw)
            result.update(status='complete', usable=True)
    except BaseException as exc:
        logging.getLogger(__name__).exception('Recovery workspace backup stopped')
        return _failure(result, exc, marker, raw)
    return result


def _path(name: Any, owner: str, *, directory: bool = False) -> str:
    if not isinstance(name, str) or len(name) > 256:
        raise ValueError('Invalid bounded backup-relative path')
    if directory and name == '.':
        return name
    parts = name.split('/')
    if any(part in ('', '.', '..') for part in parts) or PurePosixPath(name).is_absolute() or '\\' in name or ':' in name:
        raise ValueError('Invalid backup-relative path')
    if directory:
        valid = (name in {'entries', 'staging', 'journals', 'reports'} or
                 re.fullmatch(r'entries/[0-9a-f]{64}', name) or
                 re.fullmatch('staging/entry-' + owner + r'-[A-Za-z0-9_]+', name))
    else:
        valid = (name == 'workspace.json' or
                 re.fullmatch(r'entries/[0-9a-f]{64}/(owner\.json|complete\.json|payload-\d{6,}\.bin)', name) or
                 re.fullmatch('staging/entry-' + owner + r'-[A-Za-z0-9_]+/(owner\.json|complete\.json|payload-\d{6,}\.bin|\.record-[A-Za-z0-9_]+)', name) or
                 re.fullmatch(r'journals/[0-9a-f]{64}\.json', name) or
                 re.fullmatch(r'reports/run-[0-9a-f]{32}\.json', name))
    if not valid:
        raise ValueError('Unrecognized backup-relative managed path')
    return name


def _validate(root: Path, payload: dict[str, Any], check: Check, *, sealed: bool = True) -> dict[str, Any]:
    if (set(payload) != {'schema_version', 'kind', 'owner', 'source_workspace', 'source_token', 'directories', 'files'} or
            type(payload.get('schema_version')) is not int or payload['schema_version'] != 1 or
            payload.get('kind') != 'corridorkey_recovery_backup' or not isinstance(payload.get('owner'), str) or
            not re.fullmatch('[0-9a-f]{32}', payload['owner']) or not isinstance(payload.get('source_workspace'), str) or
            len(payload['source_workspace']) > 4096 or not isinstance(payload.get('source_token'), str) or
            not re.fullmatch('[0-9a-f]{64}', payload['source_token']) or
            not isinstance(payload.get('files'), list) or not 1 <= len(payload['files']) < maintenance.MAX_FILES or
            not isinstance(payload.get('directories'), list) or len(payload['directories']) > 2 * maintenance.MAX_ENTRIES + 5):
        raise ValueError('Invalid backup manifest schema or bounds')
    owner = payload['owner']
    directories = {}
    for item in payload['directories']:
        if (not isinstance(item, dict) or set(item) != {'name', 'mtime_ns'} or
                type(item['mtime_ns']) is not int or not -(2**63) <= item['mtime_ns'] < 2**63):
            raise ValueError('Invalid backup directory record')
        name = _path(item['name'], owner, directory=True)
        if name in directories:
            raise ValueError('Duplicate backup directory')
        directories[name] = item
    if not {'.', 'entries', 'staging'} <= set(directories):
        raise ValueError('Incomplete backup workspace directories')
    if sum(name.startswith('staging/') for name in directories) > maintenance.MAX_ENTRIES:
        raise ValueError('Too many backup staging directories')
    files: dict[str, dict[str, Any]] = {}
    blobs = {'backup.json'} if sealed else set()
    for item in payload['files']:
        maintenance._check(check)
        if (not isinstance(item, dict) or set(item) != {'name', 'blob', 'bytes', 'mtime_ns', 'sha256'} or
                not isinstance(item.get('blob'), str) or not re.fullmatch(r'blob-\d{6}\.bin', item['blob']) or
                item['blob'] in blobs or type(item.get('bytes')) is not int or not 0 <= item['bytes'] < 2**63 or
                type(item.get('mtime_ns')) is not int or not -(2**63) <= item['mtime_ns'] < 2**63 or
                not isinstance(item.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
            raise ValueError('Invalid backup file record')
        name = _path(item['name'], owner)
        if name in files or str(PurePosixPath(name).parent) not in directories:
            raise ValueError('Duplicate backup file or missing parent')
        info = maintenance._stat(root / item['blob'])
        if info['bytes'] != item['bytes'] or _hash(root / item['blob'], check, expected_bytes=item['bytes']) != item['sha256']:
            raise ValueError('Backup blob size/checksum mismatch')
        files[name] = item
        blobs.add(item['blob'])
    if set(maintenance._names(root, maintenance.MAX_FILES + 1, check)) != blobs:
        raise ValueError('Unexpected or missing backup container content')
    budget = [maintenance.MAX_METADATA_BYTES]

    def record(name: str) -> dict[str, Any]:
        if name not in files:
            raise ValueError('Incomplete backup ownership/completion metadata')
        return maintenance._object(root / files[name]['blob'], budget)[0]

    workspace = record('workspace.json')
    if workspace != {'schema_version': 1, 'kind': 'corridorkey_recovery', 'owner': owner} or type(workspace['schema_version']) is not int:
        raise ValueError('Invalid backup workspace ownership marker')
    entries = [name for name in directories if name.startswith('entries/')]
    if len(entries) > maintenance.MAX_ENTRIES:
        raise ValueError('Too many backup cache entries')
    for directory in entries:
        manifest = record(directory + '/complete.json')
        ownership = record(directory + '/owner.json')
        key = directory.split('/')[1]
        if (ownership != {'owner': owner, 'manifest_sha256': maintenance._digest(manifest)} or
                set(manifest) != {'schema_version', 'key', 'identity', 'files', 'origin_record'} or
                type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or manifest.get('key') != key or
                not isinstance(manifest.get('identity'), dict) or maintenance._digest(manifest['identity']) != key or
                not isinstance(manifest.get('files'), list) or not manifest['files']):
            raise ValueError('Invalid backup sealed entry metadata')
        expected = {directory + '/owner.json', directory + '/complete.json'}
        logical = set()
        for item in manifest['files']:
            if (not isinstance(item, dict) or set(item) != {'name', 'blob', 'bytes', 'sha256'} or
                    not isinstance(item.get('blob'), str) or not re.fullmatch(r'payload-\d{6,}\.bin', item['blob']) or
                    not isinstance(item.get('name'), str) or not item['name'] or item['name'] in logical or
                    type(item.get('bytes')) is not int or item['bytes'] < 0 or
                    not isinstance(item.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
                raise ValueError('Invalid backup sealed payload inventory')
            name = directory + '/' + item['blob']
            if name in expected or name not in files or files[name]['bytes'] != item['bytes'] or files[name]['sha256'] != item['sha256']:
                raise ValueError('Backup sealed payload does not match its original completion record')
            expected.add(name)
            logical.add(item['name'])
        if {name for name in files if name.startswith(directory + '/')} != expected:
            raise ValueError('Unexpected backup sealed entry files')
    for name in directories:
        if name != '.' and str(PurePosixPath(name).parent) not in directories:
            raise ValueError('Backup directory has no declared parent')
    return {'entry_count': len(entries), 'journal_count': sum(name.startswith('journals/') for name in files),
            'report_count': sum(name.startswith('reports/') for name in files)}


def _read_backup(root: Path, check: Check) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    maintenance._stat(root, directory=True)
    if len(str(root)) > 4096:
        raise ValueError('Backup path exceeds its bounded length')
    snapshot = {'files': {name: maintenance._stat(root / name) for name in maintenance._names(root, maintenance.MAX_FILES + 1, check)},
                'root': maintenance._stat(root, directory=True)}
    info = maintenance._stat(root / 'backup.json')
    if info['bytes'] > MAX_MANIFEST_BYTES:
        raise ValueError('Backup manifest exceeds its bounded byte limit')
    # _object normally limits individual workspace metadata to4MiB; this flat
    # transfer manifest admits16MiB without changing those production limits.
    with (root / 'backup.json').open('rb') as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) != info['bytes'] or maintenance._stat(root / 'backup.json') != info:
        raise ValueError('Backup manifest changed during admission')

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate backup manifest key')
            value[key] = item
        return value

    envelope = json.loads(raw, object_pairs_hook=pairs)
    if (not isinstance(envelope, dict) or set(envelope) != {'payload', 'sha256'} or
            not isinstance(envelope.get('payload'), dict) or envelope.get('sha256') != maintenance._digest(envelope['payload'])):
        raise ValueError('Invalid backup completion checksum')
    summary = _validate(root, envelope['payload'], check)
    after = {'files': {name: maintenance._stat(root / name) for name in maintenance._names(root, maintenance.MAX_FILES + 1, check)},
             'root': maintenance._stat(root, directory=True)}
    if after != snapshot:
        raise ValueError('Backup changed during verification')
    return envelope['payload'], summary, snapshot


def inspect_backup(backup: str | Path, *, check: Check = None) -> dict[str, Any]:
    result = _result()
    try:
        root = Path(backup).absolute()
        payload, summary, _ = _read_backup(root, check)
        result.update(summary, status='complete', usable=True, destination=str(root),
                      file_count=len(payload['files']), bytes=sum(item['bytes'] for item in payload['files']))
    except BaseException as exc:
        logging.getLogger(__name__).exception('Recovery backup inspection stopped')
        return _failure(result, exc, None, None)
    return result


def restore_backup(backup: str | Path, existing_parent: str | Path, *, check: Check = None) -> dict[str, Any]:
    """Restore into a new root; original workspace.json is the final commit marker."""
    result = _result()
    marker = raw = None
    connection = None
    try:
        root = Path(backup).absolute()
        payload, summary, snapshot = _read_backup(root, check)
        parent = _parent(existing_parent, root.resolve(strict=True))
        result['warning'] = _protect_parent(parent, {item['name']: root / item['blob'] for item in payload['files']}, check)
        _summarize(result, summary)
        destination = Path(tempfile.mkdtemp(prefix='recovery-restored-', dir=parent))
        result['destination'] = str(destination)
        _sync(parent)
        for item in sorted(payload['directories'], key=lambda value: len(PurePosixPath(value['name']).parts)):
            if item['name'] != '.':
                (destination / item['name']).mkdir()
        # Never transfer historical SQLite journal/connection state.
        connection = sqlite3.connect(destination / 'lock.sqlite', timeout=0)
        connection.execute('CREATE TABLE owner (version INTEGER)')
        connection.commit()
        connection.execute('BEGIN EXCLUSIVE')
        workspace = None
        for item in payload['files']:
            if item['name'] == 'workspace.json':
                workspace = item
                continue
            _copy(root / item['blob'], destination / item['name'], item, result, check)
        if workspace is None:
            raise ValueError('Backup has no workspace ownership marker')
        # A final immutable backup snapshot check detects changes after admission.
        now = {'files': {name: maintenance._stat(root / name) for name in maintenance._names(root, maintenance.MAX_FILES + 1, check)},
               'root': maintenance._stat(root, directory=True)}
        if now != snapshot:
            raise ValueError('Backup changed during restore')
        for item in sorted(payload['directories'], key=lambda value: len(PurePosixPath(value['name']).parts), reverse=True):
            directory = destination / item['name']
            os.utime(directory, ns=(item['mtime_ns'], item['mtime_ns']))
            _sync(directory)
        maintenance._check(check)
        with (root / workspace['blob']).open('rb') as stream:
            raw = stream.read(workspace['bytes'] + 1)
        if len(raw) != workspace['bytes'] or hashlib.sha256(raw).hexdigest() != workspace['sha256']:
            raise ValueError('Backup ownership marker changed before publication')
        marker = destination / 'workspace.json'
        _seal(marker, raw)
        result['copied_bytes'] += len(raw)
        result['copied_files'] += 1
        os.utime(marker, ns=(workspace['mtime_ns'], workspace['mtime_ns']))
        root_info = next(item for item in payload['directories'] if item['name'] == '.')
        os.utime(destination, ns=(root_info['mtime_ns'], root_info['mtime_ns']))
        maintenance._scan(destination, None)
        result.update(status='complete', usable=True)
    except BaseException as exc:
        logging.getLogger(__name__).exception('Recovery workspace restore stopped')
        return _failure(result, exc, marker, raw)
    finally:
        if connection is not None:
            cleanup_error = None
            try:
                connection.rollback()
            except sqlite3.DatabaseError as exc:
                cleanup_error = exc
            try:
                connection.close()
            except sqlite3.DatabaseError as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                _failure(result, cleanup_error, marker, raw)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    backup = commands.add_parser('backup')
    backup.add_argument('--workspace', required=True)
    backup.add_argument('--output-parent', required=True)
    restore = commands.add_parser('restore')
    restore.add_argument('--backup', required=True)
    restore.add_argument('--output-parent', required=True)
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--backup', required=True)
    args = parser.parse_args(argv)
    if args.command == 'backup':
        result = backup_workspace(args.workspace, args.output_parent)
    elif args.command == 'restore':
        result = restore_backup(args.backup, args.output_parent)
    else:
        result = inspect_backup(args.backup)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
