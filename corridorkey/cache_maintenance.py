"""Read-only inspection and explicit retention of existing owned recovery bundles.

The SQLite lock excludes cooperating recovery writers, not arbitrary filesystem
editors. No constructor/pruning path is used. File lengths are apparent bytes;
payload integrity, allocated blocks and space actually reclaimed are not measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .control import CancelledError

MAX_ENTRIES = 4096
MAX_FILES = 32768
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 4 * 1024 * 1024
Check = Callable[[], None] | None


def _check(check: Check) -> None:
    if check is not None:
        check()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stat(path: Path, *, directory: bool = False) -> dict[str, int]:
    value = path.lstat()
    expected = stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    if not expected or (not directory and value.st_nlink != 1):
        raise ValueError(f'Expected owned regular {"directory" if directory else "single-link file"}: {path.name}')
    # Windows junctions/reparse points must never authorize a recursive deletion.
    if getattr(value, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
        raise ValueError(f'Reparse points are unsupported: {path.name}')
    return {'device': value.st_dev, 'inode': value.st_ino, 'bytes': value.st_size,
            'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns, 'mode': value.st_mode}


def _names(path: Path, limit: int, check: Check) -> list[str]:
    names = []
    with os.scandir(path) as iterator:
        for item in iterator:
            _check(check)
            names.append(item.name)
            if len(names) > limit:
                raise ValueError('Workspace inventory exceeds the bounded entry/file limit')
    return sorted(names)


def _object(path: Path, budget: list[int]) -> tuple[dict[str, Any], str]:
    info = _stat(path)
    size = info['bytes']
    if size > MAX_RECORD_BYTES or size > budget[0]:
        raise ValueError('Workspace metadata exceeds its bounded byte limit')
    with path.open('rb') as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or _stat(path) != info:
        raise ValueError('Workspace metadata changed during inspection')
    budget[0] -= size

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate workspace metadata key')
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f'Nonfinite workspace metadata: {value}')

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError('Invalid or excessively nested workspace metadata') from exc
    if not isinstance(value, dict):
        raise TypeError('Workspace metadata must be an object')
    return value, hashlib.sha256(raw).hexdigest()


@contextmanager
def _locked(root: str | Path, check: Check, cleanup_result: dict[str, Any] | None = None) -> Iterator[Path]:
    _check(check)
    path = Path(root).absolute()
    _stat(path, directory=True)
    path = path.resolve(strict=True)
    if len(str(path)) > 4096:
        raise ValueError('Workspace path exceeds the bounded path limit')
    lock = path / 'lock.sqlite'
    before = _stat(lock)
    # WAL opening can create shared-memory files even with mode=ro. Reject it
    # from the header before connecting; managed stores use rollback journals.
    with lock.open('rb') as stream:
        header = stream.read(100)
    if len(header) != 100 or header[:16] != b'SQLite format 3\x00' or header[18:20] != b'\x01\x01':
        raise ValueError('An existing rollback-journal SQLite workspace lock is required (WAL unsupported)')
    connection = sqlite3.connect(lock.as_uri() + '?mode=ro', uri=True, timeout=0)
    try:
        try:
            connection.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError as exc:
            raise RuntimeError('Recovery workspace is busy or its lock is unavailable') from exc
        if _stat(lock) != before:
            raise ValueError('Workspace lock changed during ownership acquisition')
        _check(check)
        yield path
    finally:
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
            if cleanup_result:
                cleanup_result.update(status='failed', error=f'Lock cleanup failed: {cleanup_error}'[:4096])
            else:
                raise cleanup_error


def _scan(root: Path, check: Check) -> tuple[dict[str, Any], dict[str, Any]]:
    budget = [MAX_METADATA_BYTES]
    count = 0
    snapshot: dict[str, Any] = {'directories': {}, 'files': {}}
    directories: dict[str, Any] = snapshot['directories']
    files: dict[str, Any] = snapshot['files']

    def directory(path: Path) -> list[str]:
        _check(check)
        directories[str(path.relative_to(root))] = _stat(path, directory=True)
        return _names(path, MAX_FILES, check)

    def file(path: Path) -> dict[str, int]:
        nonlocal count
        _check(check)
        count += 1
        if count > MAX_FILES:
            raise ValueError('Workspace inventory exceeds the bounded file limit')
        value = _stat(path)
        files[str(path.relative_to(root))] = value
        return value

    top = directory(root)
    if not {'workspace.json', 'lock.sqlite', 'entries', 'staging'} <= set(top) or set(top) - {
            'workspace.json', 'lock.sqlite', 'entries', 'staging', 'journals', 'reports'}:
        raise ValueError('Unexpected or missing workspace content; no automatic deletion')
    file(root / 'workspace.json')
    file(root / 'lock.sqlite')
    marker, marker_hash = _object(root / 'workspace.json', budget)
    owner = marker.get('owner')
    if (set(marker) != {'schema_version', 'kind', 'owner'} or type(marker.get('schema_version')) is not int or
            marker['schema_version'] != 1 or marker.get('kind') != 'corridorkey_recovery' or
            not isinstance(owner, str) or not re.fullmatch('[0-9a-f]{32}', owner)):
        raise ValueError('Invalid recovery workspace ownership marker')
    snapshot['owner'] = owner
    snapshot['marker_sha256'] = marker_hash
    entries = []
    keys = directory(root / 'entries')
    if len(keys) > MAX_ENTRIES:
        raise ValueError('Workspace exceeds the bounded sealed-entry limit')
    for key in keys:
        if not re.fullmatch('[0-9a-f]{64}', key):
            raise ValueError('Unrecognized sealed entry name')
        entry = root / 'entries' / key
        names = directory(entry)
        for name in names:
            if name not in {'owner.json', 'complete.json'} and not re.fullmatch(r'payload-\d{6,}\.bin', name):
                raise ValueError('Unexpected/unowned cache entry content; no automatic deletion')
            file(entry / name)
        record, record_hash = _object(entry / 'complete.json', budget)
        ownership, owner_hash = _object(entry / 'owner.json', budget)
        if (set(ownership) != {'owner', 'manifest_sha256'} or ownership['owner'] != owner or
                ownership['manifest_sha256'] != _digest(record) or
                set(record) != {'schema_version', 'key', 'identity', 'files', 'origin_record'} or
                type(record.get('schema_version')) is not int or record['schema_version'] != 1 or
                record.get('key') != key or not isinstance(record.get('identity'), dict) or
                _digest(record['identity']) != key or not isinstance(record.get('files'), list) or not record['files']):
            raise ValueError('Invalid owned sealed entry completion metadata')
        expected = {'owner.json', 'complete.json'}
        logical_names = set()
        for item in record['files']:
            if not isinstance(item, dict) or set(item) != {'name', 'blob', 'bytes', 'sha256'}:
                raise ValueError('Invalid sealed entry inventory record')
            blob, name = item['blob'], item['name']
            if (not isinstance(blob, str) or not re.fullmatch(r'payload-\d{6,}\.bin', blob) or blob in expected or
                    not isinstance(name, str) or not name or name in logical_names or
                    type(item['bytes']) is not int or item['bytes'] < 0 or
                    not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256']) or
                    blob not in names or files[f'entries/{key}/{blob}']['bytes'] != item['bytes']):
                raise ValueError('Invalid sealed entry inventory or payload size')
            expected.add(blob)
            logical_names.add(name)
        if expected != set(names):
            raise ValueError('Incomplete or unexpected sealed entry inventory')
        snapshot[f'entry_metadata:{key}'] = [owner_hash, record_hash]
        entries.append({'key': key, 'bytes': sum(files[f'entries/{key}/{name}']['bytes'] for name in names),
                        'mtime_ns': directories[f'entries/{key}']['mtime_ns']})
    staging_bytes = 0
    stages = directory(root / 'staging')
    if len(stages) > MAX_ENTRIES:
        raise ValueError('Workspace exceeds the bounded staging-entry limit')
    for name in stages:
        if not re.fullmatch('entry-' + owner + r'-[A-Za-z0-9_]+', name):
            raise ValueError('Unrecognized staging ownership')
        stage = root / 'staging' / name
        for name in directory(stage):
            if (name not in {'owner.json', 'complete.json'} and
                    not re.fullmatch(r'(payload-\d{6,}\.bin|\.record-[A-Za-z0-9_]+)', name)):
                raise ValueError('Unexpected/unowned staging content')
            staging_bytes += file(stage / name)['bytes']
    journal_bytes = journal_count = 0
    if 'journals' in top:
        for name in directory(root / 'journals'):
            if not re.fullmatch(r'[0-9a-f]{64}\.json', name):
                raise ValueError('Unexpected journal content; preserved without deletion')
            journal_bytes += file(root / 'journals' / name)['bytes']
            journal_count += 1
    report_bytes = report_count = 0
    if 'reports' in top:
        for name in directory(root / 'reports'):
            if not re.fullmatch(r'run-[0-9a-f]{32}\.json', name):
                raise ValueError('Unexpected recovery report content; preserved without deletion')
            report_bytes += file(root / 'reports' / name)['bytes']
            report_count += 1
    _check(check)
    result = {'schema_version': 1, 'kind': 'recovery_cache_inventory', 'workspace': str(root), 'owner': owner,
              'sealed_bytes': sum(item['bytes'] for item in entries), 'entry_count': len(entries), 'entries': entries,
              'staging_bytes': staging_bytes, 'staging_entries': len(stages),
              'journal_bytes': journal_bytes, 'journal_count': journal_count,
              'report_bytes': report_bytes, 'report_count': report_count,
              'allocation': 'Apparent file lengths; filesystem allocation and reclaimed free space are not measured.',
              'scope': 'Owned sealed entries only are removable. Staging, journals, reports, source exports and render copies are preserved. '
                       'Ownership/metadata envelope and sizes verified; restore validity and payload checksums are not verified.'}
    result['token'] = _digest({'workspace': str(root), 'snapshot': snapshot})
    return result, snapshot


def inventory(root: str | Path, *, check: Check = None) -> dict[str, Any]:
    """Inspect an existing idle workspace without creating, pruning or touching it."""
    with _locked(root, check) as path:
        return _scan(path, check)[0]


def _plan(current: dict[str, Any], target_bytes: int) -> dict[str, Any]:
    if type(target_bytes) is not int or target_bytes < 0:
        raise ValueError('Target must be a nonnegative integer byte count')
    remaining = current['sealed_bytes']
    remove = []
    for item in sorted(current['entries'], key=lambda entry: (entry['mtime_ns'], entry['key'])):
        if remaining <= target_bytes:
            break
        remove.append(dict(item))
        remaining -= item['bytes']
    value = {'schema_version': 1, 'kind': 'recovery_cache_trim', 'workspace': current['workspace'],
             'owner': current['owner'], 'inventory_token': current['token'], 'target_bytes': target_bytes,
             'before_bytes': current['sealed_bytes'], 'after_bytes': remaining,
             'remove_bytes': current['sealed_bytes'] - remaining, 'remove_count': len(remove), 'remove': remove}
    value['token'] = _digest(value)
    return value


def plan_trim(root: str | Path, target_bytes: int, *, check: Check = None) -> dict[str, Any]:
    """Preview oldest-accessed completed entries to remove; no mutation occurs."""
    return _plan(inventory(root, check=check), target_bytes)


def apply_trim(root: str | Path, plan: dict[str, Any], *, check: Check = None) -> dict[str, Any]:
    """Revalidate an explicit preview and delete whole owned entries under its lock.

    Cancellation is observed between entries. Once an entry starts, its bounded
    files finish without cancellation. An I/O failure can leave a partial entry;
    the terminal report counts each successful unlink and never implies rollback.
    """
    # Freeze caller-owned JSON before acquiring ownership, including size bound.
    raw = _canonical(plan)
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError('Trim preview exceeds its bounded byte limit')
    frozen = json.loads(raw)
    if not isinstance(frozen, dict):
        raise TypeError('Trim preview must be an object')
    result: dict[str, Any] = {}
    with _locked(root, check, result) as path:
        current, snapshot = _scan(path, check)
        target = frozen.get('target_bytes')
        if type(target) is not int or target < 0:
            raise ValueError('Trim preview target must be a nonnegative integer byte count')
        expected = _plan(current, target)
        if expected != frozen:
            raise ValueError('Trim preview is stale or changed; inspect and preview again')
        result.update(status='complete', workspace=str(path), removed=[], removed_bytes=0,
                      remaining_sealed_bytes=current['sealed_bytes'], remaining_entry_bytes=current['sealed_bytes'],
                      partial_entry=None, error=None)
        for item in expected['remove']:
            key = item['key']
            entry = path / 'entries' / key
            try:
                _check(check)
            except CancelledError as exc:
                result.update(status='cancelled', error=str(exc))
                break
            except BaseException as exc:
                logging.getLogger(__name__).exception('Cache trim check failed; prior removals remain committed')
                result.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                              error=f'{type(exc).__name__}: {exc}'[:4096])
                break
            before_entry_removed = result['removed_bytes']
            try:
                # Cooperating writers are excluded throughout. Recheck directory
                # and each file against the approved snapshot before committing.
                if _stat(entry, directory=True) != snapshot['directories'][f'entries/{key}']:
                    raise ValueError('Entry changed after trim preflight')
                names = _names(entry, MAX_FILES, None)
                approved_names = {name.rsplit('/', 1)[-1] for name in snapshot['files']
                                  if name.startswith(f'entries/{key}/')}
                if set(names) != approved_names:
                    raise ValueError('Entry inventory changed after trim preflight')
                for name in names:
                    if _stat(entry / name) != snapshot['files'].get(f'entries/{key}/{name}'):
                        raise ValueError('Entry file changed after trim preflight')
                for name in sorted(names, key=lambda value: (value in {'owner.json', 'complete.json'}, value == 'owner.json', value)):
                    (entry / name).unlink()
                    result['removed_bytes'] += snapshot['files'][f'entries/{key}/{name}']['bytes']
                    result['partial_entry'] = key
                entry.rmdir()
                result['removed'].append(key)
                result['partial_entry'] = None
                if os.name != 'nt':
                    descriptor = os.open(entry.parent, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            except BaseException as exc:
                logging.getLogger(__name__).exception('Cache trim stopped; prior removals remain committed')
                result.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
                              error=f'{type(exc).__name__}: {exc}'[:4096])
                # An interrupt can arrive after the OS unlink/rmdir succeeded
                # but before Python increments its counter. Reconcile this
                # bounded current entry before returning authoritative evidence.
                try:
                    try:
                        _stat(entry, directory=True)
                    except FileNotFoundError:
                        if key not in result['removed']:
                            result['removed'].append(key)
                        result['removed_bytes'] = before_entry_removed + item['bytes']
                        result['partial_entry'] = None
                    else:
                        remaining_names = _names(entry, MAX_FILES, None)
                        approved = {name.rsplit('/', 1)[-1] for name in snapshot['files']
                                    if name.startswith(f'entries/{key}/')}
                        if set(remaining_names) - approved:
                            raise ValueError('Entry changed outside the managed writer lock')
                        retained = sum(_stat(entry / name)['bytes'] for name in remaining_names)
                        if retained > item['bytes']:
                            raise ValueError('Entry grew outside the managed writer lock')
                        result['removed_bytes'] = before_entry_removed + item['bytes'] - retained
                        result['partial_entry'] = key if set(remaining_names) != approved else None
                except (OSError, ValueError) as reconciliation_error:
                    result['removed_bytes'] = None
                    result['partial_entry'] = key
                    result['error'] = (result['error'] + f'; retained byte accounting unavailable: {reconciliation_error}')[:4096]
                break
        remaining = None if result['removed_bytes'] is None else current['sealed_bytes'] - result['removed_bytes']
        result['remaining_entry_bytes'] = remaining
        result['remaining_sealed_bytes'] = None if result['partial_entry'] is not None else remaining
        result['remaining_entry_count'] = current['entry_count'] - len(result['removed'])
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'plan', 'apply'):
        command = commands.add_parser(name)
        command.add_argument('--workspace', required=True)
        if name == 'plan':
            command.add_argument('--target-bytes', type=int, required=True)
        elif name == 'apply':
            command.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            result = inventory(args.workspace)
        elif args.command == 'plan':
            result = plan_trim(args.workspace, args.target_bytes)
        else:
            plan, _ = _object(args.plan, [MAX_RECORD_BYTES])
            result = apply_trim(args.workspace, plan)
    except (OSError, ValueError, TypeError, RuntimeError, sqlite3.DatabaseError) as exc:
        print(json.dumps({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'[:4096]}), file=sys.stdout)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result.get('status', 'complete') == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
