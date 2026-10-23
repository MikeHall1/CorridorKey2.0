"""Explicit portable sealed queue history; never activation or automatic replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from . import cache_backup as copying
from . import cache_maintenance as maintenance
from .control import CancelledError
from .local_json import read_object
from .queue_store import MAX_REQUESTS, _archive, _marker, _relocations
from .worker_protocol import (
    MAX_FRAME_BYTES,
    decode_frame,
    encode_frame,
    identifier,
    parse_request,
)

MAX_DB_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_FILES = MAX_REQUESTS + 5
FLAVOUR = 'windows' if os.name == 'nt' else 'posix'
SCOPE = ('Archived observations only; original requests/events and historical dependency paths are unchanged. '
         'Imported history stays read-only. Only the owned recovery batch snapshot is relocated for explicit loading; '
         'no media/assets/outputs are copied or validated, no jobs are submitted. Native path flavour must match.')
Check = maintenance.Check


def _sqlite_file(path: Path, *, empty: bool = False) -> None:
    info = maintenance._stat(path)
    if empty and info['bytes'] == 0:
        return
    if info['bytes'] > MAX_DB_BYTES:
        raise ValueError('Queue SQLite exceeds the256MiB transfer bound')
    with path.open('rb') as stream:
        raw = stream.read(100)
    if len(raw) != 100 or raw[:16] != b'SQLite format 3\0' or raw[18:20] != b'\1\1':
        raise ValueError('Queue transfer requires rollback SQLite (WAL refused)')


@contextmanager
def _locked(root: Path, check: Check):
    maintenance._check(check)
    maintenance._stat(root, directory=True)
    _sqlite_file(root / 'owner-lock.sqlite', empty=True)
    before = maintenance._stat(root / 'owner-lock.sqlite')
    with closing(sqlite3.connect((root / 'owner-lock.sqlite').as_uri() + '?mode=ro', uri=True, timeout=0)) as lock:
        try:
            lock.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError as exc:
            raise RuntimeError('Queue owner is busy or unavailable') from exc
        try:
            if maintenance._stat(root / 'owner-lock.sqlite') != before:
                raise ValueError('Queue lock changed during acquisition')
            yield
        finally:
            lock.rollback()


def _relocation_payload(queue_id: str, recoveries: list[dict[str, Any]]) -> dict[str, Any]:
    return {'schema_version': 1, 'kind': 'corridorkey_queue_relocation', 'queue_id': queue_id,
            'path_flavour': FLAVOUR, 'recoveries': recoveries}


def _verify_db(path: Path, queue_id: str, batches: dict[str, Path], check: Check,
               declared: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[Any]]:
    _sqlite_file(path)
    recoveries = []
    provenance: set[str] = set()
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0)) as db:
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_FRAME_BYTES + 8192)
        db.row_factory = sqlite3.Row
        rows = db.execute('SELECT queue_id,schema_version FROM metadata LIMIT 2').fetchall()
        if len(rows) != 1 or rows[0]['queue_id'] != queue_id or rows[0]['schema_version'] != 1:
            raise ValueError('Queue database ownership/version mismatch')
        count = db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
        if count > MAX_REQUESTS:
            raise ValueError('Queue transfer exceeds4096requests')
        if db.execute('SELECT COUNT(*) FROM requests r LEFT JOIN sessions s ON r.session=s.id WHERE s.id IS NULL').fetchone()[0]:
            raise ValueError('Queue request has no original session')
        if db.execute('''SELECT 1 FROM requests WHERE typeof(id)!='text' OR length(id)>64 OR
                         length(CAST(request AS BLOB))>? OR length(CAST(event AS BLOB))>? LIMIT 1''',
                      (MAX_FRAME_BYTES, MAX_FRAME_BYTES)).fetchone():
            raise ValueError('Queue observation exceeds bounded size')
        for row in db.execute('SELECT * FROM requests ORDER BY rowid'):
            maintenance._check(check)
            request_id = identifier(row['id'])
            decoded = {}
            for name in ('request', 'event'):
                text = row[name]
                if text is not None:
                    if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != row[name + '_hash']:
                        raise ValueError('Queue original observation checksum mismatch')
                    decoded[name] = decode_frame(text.encode() + b'\n')
            request = parse_request(decoded['request'])
            if request.request_id != request_id or request.action != row['action'] or request.action not in {'render', 'recover'}:
                raise ValueError('Queue original request metadata mismatch')
            event = decoded.get('event')
            if event is not None and (event.get('request_id') != request_id or event.get('kind') != row['status']):
                raise ValueError('Queue original event metadata mismatch')
            if event is None and row['status'] not in {'prepared', 'dispatching', 'sent'}:
                raise ValueError('Queue observation has no saved event')
            _references(list(decoded.values()), provenance, check)
            if request.action == 'render':
                for name in ('config_json', 'writer_json'):
                    _references([json.loads(request.payload()['job'][name])], provenance, check)
            if request.action == 'recover':
                digest = row['batch_hash']
                historical = request.payload()['batch']
                name = f'batches/{request_id}.json'
                if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or
                        not isinstance(historical, str) or len(historical) > 4096 or not Path(historical).is_absolute() or name not in batches):
                    raise ValueError('Portable recovery history requires a native absolute snapshotted batch and checksum')
                batch = batches[name]
                info = maintenance._stat(batch)
                if info['bytes'] > MAX_FRAME_BYTES or copying._hash(batch, check, expected_bytes=info['bytes']) != digest:
                    raise ValueError('Queue owned batch snapshot checksum mismatch')
                batch_data, _ = read_object(batch, max_bytes=MAX_FRAME_BYTES)
                _references([batch_data], provenance, check)
                recoveries.append({'request_id': request_id, 'historical_batch': historical,
                                   'local_batch': name, 'batch_sha256': digest})
    if set(batches) != {item['local_batch'] for item in recoveries}:
        raise ValueError('Queue contains orphan or missing batch snapshots')
    if declared is not None and declared != recoveries:
        raise ValueError('Queue relocation map differs from original checksummed history')
    return recoveries, sorted(provenance)


def _references(values: list[Any], result: set[str], check: Check) -> None:
    pending = list(values)
    nodes = 0
    while pending:
        maintenance._check(check)
        value = pending.pop()
        nodes += 1
        if nodes > 1048576:
            raise ValueError('Queue dependency traversal exceeds its bounded node count')
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and Path(value).is_absolute():
            if len(value) > 4096:
                raise ValueError('Queue historical path exceeds bounded length')
            result.add(value)
            if len(result) > 32768 or sum(len(item) for item in result) > 16 * 1024 * 1024:
                raise ValueError('Queue recorded path inventory exceeds its bounded size')


def _guard(parent: Path, values: list[Any], check: Check) -> None:
    pending = list(values)
    count = 0
    while pending:
        maintenance._check(check)
        value = pending.pop()
        count += 1
        if count > 1048576:
            raise ValueError('Queue dependency traversal exceeds its bounded node count')
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and Path(value).is_absolute():
            if len(value) > 4096:
                raise ValueError('Queue historical path exceeds bounded length')
            path = Path(value).resolve(strict=False)
            if parent == path or parent.is_relative_to(path):
                raise ValueError('Destination overlaps recorded media, model, workspace or export path')


def _scan(root: Path, check: Check) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    maintenance._stat(root, directory=True)
    if len(str(root)) > 4096:
        raise ValueError('Queue path exceeds its bounded length')
    marker = _marker(root)
    archive = _archive(root, marker['queue_id'])
    if archive is None:
        raise ValueError('Archive stopped queue history explicitly before exporting')
    relocations = _relocations(root, marker['queue_id'], archive)
    allowed = {'queue-owner.json', 'queue-archive.json', 'queue.sqlite', 'owner-lock.sqlite', 'batches', 'queue-transfer.json'}
    names = maintenance._names(root, 6, check)
    if set(names) - allowed:
        raise ValueError('Unexpected queue files; no automatic cleanup or export')
    files = {}
    directories = {'.': maintenance._stat(root, directory=True)}
    for name in names:
        if name != 'batches':
            files[name] = maintenance._stat(root / name)
    batches = {}
    if 'batches' in names:
        directories['batches'] = maintenance._stat(root / 'batches', directory=True)
        for name in maintenance._names(root / 'batches', MAX_REQUESTS, check):
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}\.json', name):
                raise ValueError('Unexpected queue batch snapshot file')
            relative = 'batches/' + name
            files[relative] = maintenance._stat(root / relative)
            batches[relative] = root / relative
    recoveries, provenance = _verify_db(root / 'queue.sqlite', marker['queue_id'], batches, check)
    if relocations is not None and list(relocations.values()) != recoveries:
        raise ValueError('Imported queue relocation inventory differs from original observations')
    if relocations is None:
        for item in recoveries:
            if Path(item['historical_batch']).resolve() != (root / item['local_batch']).resolve():
                raise ValueError('Queue original snapshot is outside its owned batches')
    state = {'files': files, 'directories': directories}
    return {'queue_id': marker['queue_id'], 'recoveries': recoveries,
            'request_count': _count(root / 'queue.sqlite')}, state, provenance


def _count(path: Path) -> int:
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0)) as db:
        return db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]


def _result() -> dict[str, Any]:
    return {'status': 'failed', 'destination': None, 'usable': False, 'copied_bytes': 0, 'copied_files': 0,
            'request_count': None, 'error': None, 'warning': None, 'scope': SCOPE}


def _failed(result: dict[str, Any], exc: BaseException, marker: Path | None, raw: bytes | None) -> dict[str, Any]:
    if result['destination'] is not None:
        try:
            root = Path(result['destination'])
            paths = []
            for name in maintenance._names(root, MAX_FILES + 4, None):
                if name == 'batches':
                    maintenance._stat(root / name, directory=True)
                    paths.extend(root / name / child for child in maintenance._names(root / name, MAX_REQUESTS, None))
                elif name not in {'owner-lock.sqlite', 'queue-export.json', 'queue-transfer.json'} and not name.startswith('.seal-'):
                    paths.append(root / name)
            result.update(copied_bytes=sum(maintenance._stat(path)['bytes'] for path in paths), copied_files=len(paths))
        except (OSError, ValueError) as error:
            result.update(copied_bytes=None, copied_files=None, warning=f'Partial copy accounting unavailable: {error}'[:4096])
    if copying._committed(marker, raw) and marker is not None:
        try:
            if marker.name == 'queue-export.json':
                _read_export(marker.parent, None)
            else:
                _scan(marker.parent, None)
        except BaseException as failure:
            logging.getLogger(__name__).exception('Published queue transfer failed verification')
            result.update(status='failed', usable=False, error=f'Published transfer verification failed: {failure}'[:4096])
        else:
            result.update(status='complete', usable=True, warning=f'Committed transfer; finalization reported {exc}'[:4096])
    else:
        result.update(status='cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed',
                      usable=False, error=f'{type(exc).__name__}: {exc}'[:4096])
    return result


def export_history(root: str | Path, parent: str | Path, *, check: Check = None) -> dict[str, Any]:
    result = _result()
    final = raw = None
    try:
        source = Path(root).absolute()
        with _locked(source, check):
            current, before, provenance = _scan(source, check)
            parent_path = copying._parent(parent, source.resolve())
            _guard(parent_path, provenance, check)
            result['request_count'] = current['request_count']
            destination = Path(tempfile.mkdtemp(prefix='queue-export-', dir=parent_path))
            result['destination'] = str(destination)
            copying._sync(parent_path)
            files = []
            for name, info in sorted(before['files'].items()):
                if name in {'owner-lock.sqlite', 'queue-transfer.json'}:
                    continue
                item = {'name': name, 'blob': f'blob-{len(files):06d}.bin', 'bytes': info['bytes'], 'mtime_ns': info['mtime_ns']}
                item['sha256'] = copying._copy(source / name, destination / item['blob'], item, result, check)
                files.append(item)
            if _scan(source, check)[1] != before:
                raise ValueError('Queue history changed during export')
            payload = {'schema_version': 1, 'kind': 'corridorkey_queue_export', 'path_flavour': FLAVOUR,
                       'queue_id': current['queue_id'], 'request_count': current['request_count'], 'files': files,
                       'batches_present': 'batches' in before['directories'], 'recoveries': current['recoveries']}
            raw = maintenance._canonical({'payload': payload, 'sha256': maintenance._digest(payload)})
            if len(raw) > MAX_MANIFEST_BYTES:
                raise ValueError('Queue export manifest exceeds8MiB')
            _validate_export(destination, payload, check, sealed=False)
            maintenance._check(check)
            final = destination / 'queue-export.json'
            copying._seal(final, raw)
            result.update(status='complete', usable=True)
    except BaseException as exc:
        logging.getLogger(__name__).exception('Queue history export stopped')
        return _failed(result, exc, final, raw)
    return result


def _validate_export(root: Path, payload: dict[str, Any], check: Check, *, sealed: bool = True) -> tuple[dict[str, Any], list[Any]]:
    if (set(payload) != {'schema_version', 'kind', 'path_flavour', 'queue_id', 'request_count', 'files', 'batches_present', 'recoveries'} or
            type(payload.get('schema_version')) is not int or payload['schema_version'] != 1 or
            payload.get('kind') != 'corridorkey_queue_export' or payload.get('path_flavour') != FLAVOUR or
            type(payload.get('batches_present')) is not bool or type(payload.get('request_count')) is not int or
            not 0 <= payload['request_count'] <= MAX_REQUESTS or not isinstance(payload.get('files'), list) or
            not 3 <= len(payload['files']) <= MAX_FILES or not isinstance(payload.get('recoveries'), list)):
        raise ValueError('Invalid/incompatible queue export manifest')
    identifier(payload['queue_id'])
    files = {}
    blobs = {'queue-export.json'} if sealed else set()
    for item in payload['files']:
        maintenance._check(check)
        if (not isinstance(item, dict) or set(item) != {'name', 'blob', 'bytes', 'mtime_ns', 'sha256'} or
                not isinstance(item.get('name'), str) or not isinstance(item.get('blob'), str) or
                not re.fullmatch(r'blob-\d{6}\.bin', item['blob']) or item['blob'] in blobs or item['name'] in files or
                type(item.get('bytes')) is not int or item['bytes'] < 0 or
                type(item.get('mtime_ns')) is not int or not -(2**63) <= item['mtime_ns'] < 2**63 or
                not isinstance(item.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
            raise ValueError('Invalid queue export file record')
        name = item['name']
        if name not in {'queue-owner.json', 'queue-archive.json', 'queue.sqlite'} and not re.fullmatch(r'batches/[A-Za-z0-9_-]{1,64}\.json', name):
            raise ValueError('Invalid queue export managed relative path')
        maximum = MAX_DB_BYTES if name == 'queue.sqlite' else MAX_FRAME_BYTES
        if item['bytes'] > maximum or copying._hash(root / item['blob'], check, expected_bytes=item['bytes']) != item['sha256']:
            raise ValueError('Queue export blob checksum/size mismatch')
        files[name] = item
        blobs.add(item['blob'])
    if not {'queue-owner.json', 'queue-archive.json', 'queue.sqlite'} <= set(files):
        raise ValueError('Incomplete queue export')
    if set(maintenance._names(root, MAX_FILES + 1, check)) != blobs:
        raise ValueError('Unexpected or missing queue export file')
    owner, _ = read_object(root / files['queue-owner.json']['blob'], max_bytes=MAX_FRAME_BYTES)
    archive, _ = read_object(root / files['queue-archive.json']['blob'], max_bytes=4096)
    if owner != {'schema_version': 1, 'kind': 'corridorkey_desktop_queue', 'queue_id': payload['queue_id']} or type(owner['schema_version']) is not int:
        raise ValueError('Queue export owner mismatch')
    if (set(archive) != {'schema_version', 'kind', 'queue_id', 'created_utc'} or type(archive['schema_version']) is not int or
            archive['schema_version'] != 1 or archive['kind'] != 'corridorkey_queue_archive' or archive['queue_id'] != payload['queue_id'] or
            not isinstance(archive['created_utc'], str) or not 0 < len(archive['created_utc']) <= 80):
        raise ValueError('Queue export archive mismatch')
    batches = {name: root / item['blob'] for name, item in files.items() if name.startswith('batches/')}
    if batches and not payload['batches_present']:
        raise ValueError('Queue export batch directory inventory mismatch')
    recoveries, provenance = _verify_db(root / files['queue.sqlite']['blob'], payload['queue_id'], batches, check, payload['recoveries'])
    if _count(root / files['queue.sqlite']['blob']) != payload['request_count']:
        raise ValueError('Queue export request count mismatch')
    # Admission also proves the bounded relocation envelope can be imported.
    encode_frame({'payload': _relocation_payload(payload['queue_id'], recoveries),
                  'sha256': maintenance._digest(_relocation_payload(payload['queue_id'], recoveries))})
    return files, provenance


def _snapshot(root: Path, check: Check) -> dict[str, Any]:
    maintenance._stat(root, directory=True)
    return {'root': maintenance._stat(root, directory=True),
            'files': {name: maintenance._stat(root / name) for name in maintenance._names(root, MAX_FILES + 1, check)}}


def _read_export(root: Path, check: Check) -> tuple[dict[str, Any], dict[str, Any], list[Any], dict[str, Any]]:
    before = _snapshot(root, check)
    envelope, _ = read_object(root / 'queue-export.json', max_bytes=MAX_MANIFEST_BYTES, max_depth=128)
    if (set(envelope) != {'payload', 'sha256'} or not isinstance(envelope.get('payload'), dict) or
            envelope['sha256'] != maintenance._digest(envelope['payload'])):
        raise ValueError('Invalid queue export completion checksum')
    payload = envelope['payload']
    files, provenance = _validate_export(root, payload, check)
    if _snapshot(root, check) != before:
        raise ValueError('Queue export changed during verification')
    return payload, files, provenance, before


def inspect_export(root: str | Path, *, check: Check = None) -> dict[str, Any]:
    result = _result()
    try:
        path = Path(root).absolute()
        payload, _, _, _ = _read_export(path, check)
        result.update(status='complete', usable=True, destination=str(path), request_count=payload['request_count'])
    except BaseException as exc:
        logging.getLogger(__name__).exception('Queue export inspection stopped')
        return _failed(result, exc, None, None)
    return result


def import_history(archive: str | Path, parent: str | Path, *, check: Check = None) -> dict[str, Any]:
    result = _result()
    final = raw = None
    lock = None
    try:
        source = Path(archive).absolute()
        payload, files, provenance, before = _read_export(source, check)
        parent_path = copying._parent(parent, source.resolve())
        _guard(parent_path, provenance, check)
        destination = Path(tempfile.mkdtemp(prefix='queue-imported-', dir=parent_path))
        result.update(destination=str(destination), request_count=payload['request_count'])
        copying._sync(parent_path)
        if payload['batches_present']:
            (destination / 'batches').mkdir()
        (destination / 'owner-lock.sqlite').touch(exist_ok=False)
        lock = sqlite3.connect((destination / 'owner-lock.sqlite').as_uri() + '?mode=ro', uri=True, timeout=0)
        lock.execute('BEGIN EXCLUSIVE')
        for name, item in files.items():
            if name != 'queue-owner.json':
                copying._copy(source / item['blob'], destination / name, item, result, check)
        relocation = _relocation_payload(payload['queue_id'], payload['recoveries'])
        copying._seal(destination / 'queue-transfer.json', encode_frame({'payload': relocation, 'sha256': maintenance._digest(relocation)}))
        if payload['batches_present']:
            copying._sync(destination / 'batches')
        if _snapshot(source, check) != before:
            raise ValueError('Queue export changed during import')
        maintenance._check(check)
        owner = files['queue-owner.json']
        with (source / owner['blob']).open('rb') as stream:
            raw = stream.read(owner['bytes'] + 1)
        if len(raw) != owner['bytes'] or hashlib.sha256(raw).hexdigest() != owner['sha256']:
            raise ValueError('Queue export owner changed before commit')
        final = destination / 'queue-owner.json'
        copying._seal(final, raw)
        result['copied_bytes'] += len(raw)
        result['copied_files'] += 1
        _scan(destination, None)
        result.update(status='complete', usable=True)
    except BaseException as exc:
        logging.getLogger(__name__).exception('Queue history import stopped')
        return _failed(result, exc, final, raw)
    finally:
        if lock is not None:
            cleanup_error = None
            try:
                lock.rollback()
            except sqlite3.DatabaseError as exc:
                cleanup_error = exc
            try:
                lock.close()
            except sqlite3.DatabaseError as exc:
                cleanup_error = exc
            if cleanup_error is not None:
                _failed(result, cleanup_error, final, raw)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--workspace', required=True)
    export.add_argument('--output-parent', required=True)
    importing = commands.add_parser('import')
    importing.add_argument('--archive', required=True)
    importing.add_argument('--output-parent', required=True)
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--archive', required=True)
    args = parser.parse_args(argv)
    if args.command == 'export':
        result = export_history(args.workspace, args.output_parent)
    elif args.command == 'import':
        result = import_history(args.archive, args.output_parent)
    else:
        result = inspect_export(args.archive)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
