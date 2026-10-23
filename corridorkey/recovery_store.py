"""Single-owner bounded immutable file bundles with atomic checked completion records."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from .identities import file_sha256
from .lifecycle import cleanup_all
from .recovery_identity import canonical


class CacheInvalid(ValueError):
    pass


class CacheCapacity(ValueError):
    pass


MAX_OWNER_RECORD_BYTES = 4 * 1024 * 1024


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _open_regular(path):
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise CacheInvalid('cache read requires a regular single-link file')
        yield stream, info


def _read_extent(path, *, max_bytes, expected_size=None, expected_sha256=None, check=None):
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError('cache read bound must be a nonnegative integer')
    if expected_size is not None and (type(expected_size) is not int or expected_size < 0):
        raise ValueError('expected payload size must be a nonnegative integer')
    if check is not None:
        check()
    with _open_regular(path) as (stream, before):
        size = before.st_size if expected_size is None else expected_size
        if size > max_bytes or before.st_size != size:
            raise CacheInvalid('cache file exceeds admitted read extent')
        remaining, chunks = size, []
        digest = hashlib.sha256()
        while remaining:
            if check is not None:
                check()
            block = stream.read(min(1024 * 1024, remaining + 1))
            if not block or len(block) > remaining:
                raise CacheInvalid('cache file changed during bounded read')
            chunks.append(block)
            digest.update(block)
            remaining -= len(block)
        if stream.read(1):
            raise CacheInvalid('cache file grew beyond admitted read extent')
        after = os.fstat(stream.fileno())
        current = Path(path).stat(follow_symlinks=False)
        if _file_identity(before) != _file_identity(after) or _file_identity(after) != _file_identity(current):
            raise CacheInvalid('cache file changed during bounded read')
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise CacheInvalid('cache payload checksum changed during bounded read')
    if check is not None:
        check()
    return b''.join(chunks)


def read_payload_bounded(path, *, expected_size, expected_sha256, max_bytes, check=None):
    """Admit exact bytes on one descriptor before returning an immutable blob."""
    return _read_extent(path, max_bytes=max_bytes, expected_size=expected_size,
                        expected_sha256=expected_sha256, check=check)


def _owner_limit(explicit):
    return MAX_OWNER_RECORD_BYTES if explicit is None else min(MAX_OWNER_RECORD_BYTES, explicit)


def _read_record_bounded(path, max_bytes=None):
    """Optional metadata bound; internal owner/manifest callers provide limits."""
    if max_bytes is None:
        return json.loads(Path(path).read_text())
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('metadata read bound must be a positive integer')
    blob = _read_extent(path, max_bytes=max_bytes)
    try:
        return json.loads(blob)
    except RecursionError as exc:
        raise CacheInvalid('cache metadata nesting exceeds decoder bound') from exc


def _sync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_record(path, value):
    """Atomic JSON replacement with file/directory sync on this filesystem."""
    path = Path(path)
    payload = canonical(value).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.record-', delete=False) as stream:
            name = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def relative_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
            any(part in ('', '.', '..') for part in name.split('/')) or PurePosixPath(name).is_absolute() or
            any(part.endswith(('.', ' ')) for part in name.split('/')) or
            name.casefold() == '.replay-complete.json'):
        raise CacheInvalid('invalid bundle-relative artifact path')
    return name


def validate_inventory_names(names):
    portable = [unicodedata.normalize('NFC', relative_name(name)).casefold() for name in names]
    unique = set(portable)
    if len(unique) != len(portable) or any(str(parent) in unique for name in portable for parent in PurePosixPath(name).parents):
        raise CacheInvalid('artifact paths conflict by name/case/Unicode normalization or file-parent relationship')


def _copy_checked(source, target, expected, *, expected_size=None, check=None):
    digest = hashlib.sha256()
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as reader, Path(target).open('xb') as writer:
        while True:
            if check is not None:
                check()
            amount = 1024 * 1024 if expected_size is None else min(1024 * 1024, expected_size - written + 1)
            block = reader.read(amount)
            if not block:
                break
            if expected_size is not None and len(block) > expected_size - written:
                raise CacheInvalid('artifact grew beyond declared copy extent')
            writer.write(block)
            digest.update(block)
            written += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if digest.hexdigest() != expected or (expected_size is not None and written != expected_size):
        raise CacheInvalid('artifact changed/corrupted during copy')
    if check is not None:
        check()


def _file_sha256_bounded(path, expected_size, *, check=None):
    """Hash no more than the admitted file extent plus one growth sentinel."""
    digest = hashlib.sha256()
    seen = 0
    with _open_regular(path) as (stream, before):
        if before.st_size != expected_size:
            raise CacheInvalid('artifact size changed before bounded hash')
        while True:
            if check is not None:
                check()
            block = stream.read(min(1024 * 1024, expected_size - seen + 1))
            if not block:
                break
            if len(block) > expected_size - seen:
                raise CacheInvalid('artifact grew beyond declared hash extent')
            digest.update(block)
            seen += len(block)
        after = os.fstat(stream.fileno())
        current = Path(path).stat(follow_symlinks=False)
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if seen != expected_size or identity(before) != identity(after) or identity(after) != identity(current):
        raise CacheInvalid('artifact changed during bounded hash')
    return digest.hexdigest()


class BundleStore:
    def __init__(self, root, max_bytes, *, structural_admission=False):
        if type(structural_admission) is not bool:
            raise ValueError('structural_admission must be a boolean')
        self.structural_admission = structural_admission
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('cache budget must be a nonnegative integer byte count')
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.connection = None
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / 'workspace.json'
        if not marker.exists():
            if any(self.root.iterdir()):
                raise ValueError('recovery workspace must be empty or already owned by CorridorKey')
            with marker.open('x') as stream:
                stream.write(canonical({'schema_version': 1, 'kind': 'corridorkey_recovery', 'owner': uuid.uuid4().hex}))
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(self.root)
        data = _read_record_bounded(marker, MAX_OWNER_RECORD_BYTES)
        if (marker.is_symlink() or not isinstance(data, dict) or type(data.get('schema_version')) is not int or
                data['schema_version'] != 1 or data.get('kind') != 'corridorkey_recovery' or
                not re.fullmatch('[0-9a-f]{32}', data.get('owner', ''))):
            raise ValueError('invalid recovery workspace ownership marker')
        self.owner = data['owner']
        for name in ('entries', 'staging'):
            directory = self.root / name
            if directory.is_symlink():
                raise ValueError('recovery workspace directories cannot be symlinks')
            directory.mkdir(exist_ok=True)

    def __enter__(self):
        if (self.root / 'lock.sqlite').is_symlink():
            raise ValueError('workspace lock cannot be a symlink')
        connection = sqlite3.connect(self.root / 'lock.sqlite', timeout=0)
        try:
            connection.execute('CREATE TABLE IF NOT EXISTS owner (version INTEGER)')
            connection.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError as exc:
            connection.close()
            raise RuntimeError('recovery workspace is busy or its lock is unavailable') from exc
        self.connection = connection
        try:
            if self.structural_admission:
                self._structural_scan(None)
            for path in (self.root / 'staging').iterdir():
                self._discard(path)
            self._make_room(0)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, kind, error, traceback):
        if self.connection is not None:
            self.connection.rollback()
            self.connection.close()
            self.connection = None

    def _require_owner(self):
        if self.connection is None:
            raise RuntimeError('bundle operations require the exclusive workspace context')

    def _entry(self, identity):
        key = identity.key
        if (not isinstance(key, str) or not re.fullmatch('[0-9a-f]{64}', key) or
                hashlib.sha256(canonical(identity.to_dict()).encode()).hexdigest() != key):
            raise CacheInvalid('invalid content identity/key')
        return self.root / 'entries' / key

    @contextmanager
    def private_stage(self, *, max_metadata_bytes=None, max_entry_files=None):
        self._require_owner()
        stage = Path(tempfile.mkdtemp(prefix=f'entry-{self.owner}-', dir=self.root / 'staging'))
        try:
            yield stage
        finally:
            if stage.exists():
                cleanup_all([lambda: self._discard(stage, max_metadata_bytes=max_metadata_bytes, max_entry_files=max_entry_files)], primary_error=sys.exception())

    def _owned(self, path, *, max_metadata_bytes=None, max_entry_files=None):
        if path.is_symlink() or not path.is_dir():
            raise CacheInvalid('cache entry must be an owned directory')
        stage = path.parent == self.root / 'staging'
        if stage and not re.fullmatch('entry-' + self.owner + r'-[A-Za-z0-9_]+', path.name):
            raise CacheInvalid('unrecognized staging ownership')
        if max_entry_files is None:
            files = list(path.iterdir())
        else:
            if type(max_entry_files) is not int or max_entry_files <= 0:
                raise ValueError('entry file count bound must be a positive integer')
            files = []
            for file in path.iterdir():
                if len(files) >= max_entry_files:
                    raise CacheInvalid('cache entry file count exceeds bound')
                files.append(file)
        for file in files:
            allowed = (file.name in ('owner.json', 'complete.json') or
                       re.fullmatch(r'payload-\d{6,}\.bin', file.name) or
                       (stage and re.fullmatch(r'\.record-[A-Za-z0-9_]+', file.name)))
            if not allowed or file.is_symlink() or not file.is_file():
                raise CacheInvalid('unexpected/unowned cache entry content; no automatic deletion')
        if not stage and files:
            owner = path / 'owner.json'
            if not owner.is_file():
                raise CacheInvalid('cache entry ownership cannot be established')
            data = _read_record_bounded(owner, _owner_limit(max_metadata_bytes))
            if not isinstance(data, dict) or data.get('owner') != self.owner:
                raise CacheInvalid('cache entry ownership cannot be established')
        return files

    def _discard(self, path, *, max_metadata_bytes=None, max_entry_files=None):
        for file in self._owned(path, max_metadata_bytes=max_metadata_bytes, max_entry_files=max_entry_files):
            file.unlink()
        path.rmdir()
        _sync_directory(path.parent)

    def _make_room(self, needed, *, max_metadata_bytes=None, max_entry_files=None):
        entries = []
        for path in (self.root / 'entries').iterdir():
            if not re.fullmatch('[0-9a-f]{64}', path.name):
                raise CacheInvalid('unrecognized cache entry name')
            size = sum(file.stat().st_size for file in self._owned(path, max_metadata_bytes=max_metadata_bytes, max_entry_files=max_entry_files))
            entries.append((path.stat().st_mtime_ns, path.name, size, path))
        total = sum(item[2] for item in entries)
        for _, _, size, path in sorted(entries):
            if total + needed <= self.max_bytes:
                break
            self._discard(path, max_metadata_bytes=max_metadata_bytes, max_entry_files=max_entry_files)
            total -= size
        if total + needed > self.max_bytes:
            raise CacheCapacity('entry exceeds the configured cache byte budget')

    def lookup(self, identity, *, check=None, max_metadata_bytes=None, max_entry_files=None, max_payload_bytes=None):
        self._require_owner()
        entry = self._entry(identity)
        if not entry.exists():
            return None
        try:
            owned = self._owned(entry, max_metadata_bytes=max_metadata_bytes, max_entry_files=max_entry_files)
            manifest = _read_record_bounded(entry / 'complete.json',
                                            self.max_bytes if max_metadata_bytes is None else min(self.max_bytes, max_metadata_bytes))
            owner = _read_record_bounded(entry / 'owner.json', _owner_limit(max_metadata_bytes))
            if owner.get('manifest_sha256') != hashlib.sha256(canonical(manifest).encode()).hexdigest():
                raise CacheInvalid('completion metadata checksum mismatch')
            if (type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or manifest['key'] != identity.key or
                    canonical(manifest['identity']) != canonical(identity.to_dict())):
                raise CacheInvalid('cache identity/completion record mismatch')
            expected = {'owner.json', 'complete.json'}
            names = set()
            for item in manifest['files']:
                name = relative_name(item['name'])
                if name in names or not re.fullmatch(r'payload-\d{6,}\.bin', item['blob']):
                    raise CacheInvalid('duplicate/invalid cache artifact record')
                names.add(name)
                if item['blob'] in expected:
                    raise CacheInvalid('duplicate cache payload')
                expected.add(item['blob'])
                payload = entry / item['blob']
                if (type(item['bytes']) is not int or item['bytes'] < 0
                        or (max_payload_bytes is not None and item['bytes'] > max_payload_bytes)
                        or payload.stat().st_size != item['bytes']):
                    raise CacheInvalid('cache payload size/checksum mismatch')
                digest = (file_sha256(payload, check=check) if max_payload_bytes is None else
                          _file_sha256_bounded(payload, item['bytes'], check=check))
                if digest != item['sha256']:
                    raise CacheInvalid('cache payload size/checksum mismatch')
            if not names or {file.name for file in owned} != expected:
                raise CacheInvalid('incomplete or unexpected cache inventory')
            validate_inventory_names(names)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise CacheInvalid(f'invalid sealed cache entry: {exc}') from exc
        os.utime(entry, None)
        return manifest

    def _structural_scan(self, check):
        # Reuse the bounded maintenance ownership contract under our existing
        # exclusive SQLite owner. Do not reacquire its public read-only lock.
        from . import cache_maintenance as maintenance

        bounded = {
            'Workspace inventory exceeds the bounded entry/file limit',
            'Workspace metadata exceeds its bounded byte limit',
            'Workspace exceeds the bounded sealed-entry limit',
            'Workspace exceeds the bounded staging-entry limit',
            'Workspace inventory exceeds the bounded file limit',
        }
        try:
            return maintenance._scan(self.root, check)
        except (ValueError, TypeError) as exc:
            if str(exc) in bounded:
                raise CacheCapacity('managed workspace structural capacity exhausted') from exc
            raise CacheInvalid('managed workspace structural validation failed') from exc

    def _structural_headroom(self, manifest, owner, payload_count, check):
        from . import cache_maintenance as maintenance

        current, snapshot = self._structural_scan(check)
        manifest_bytes, owner_bytes = len(canonical(manifest).encode()), len(canonical(owner).encode())
        metadata_bytes = sum(info['bytes'] for name, info in snapshot['files'].items()
                             if name == 'workspace.json'
                             or (name.startswith('entries/') and name.rsplit('/', 1)[-1] in ('owner.json', 'complete.json'))
                             or (name.startswith('staging/') and
                                 (name.rsplit('/', 1)[-1] in ('owner.json', 'complete.json')
                                  or name.rsplit('/', 1)[-1].startswith('.record-'))))
        if (current['entry_count'] + 1 > maintenance.MAX_ENTRIES
                or current['staging_entries'] + 1 > maintenance.MAX_ENTRIES
                or len(snapshot['files']) + payload_count + 4 > maintenance.MAX_FILES
                or max(manifest_bytes, owner_bytes) > maintenance.MAX_RECORD_BYTES
                or metadata_bytes + manifest_bytes + owner_bytes + max(manifest_bytes, owner_bytes) > maintenance.MAX_METADATA_BYTES):
            raise CacheCapacity('managed workspace structural capacity lacks publication/transient headroom')

    def publish(self, identity, files, record, *, check=None, max_metadata_bytes=None, max_entry_files=None, max_payload_bytes=None,
                expected_inventory=None, target_max_entry_files=None):
        self._require_owner()
        if self.structural_admission:
            from . import cache_maintenance as maintenance
            if len(files) + 3 > maintenance.MAX_FILES:
                raise CacheCapacity('artifact inventory cannot fit managed workspace structural limits')
        if expected_inventory is not None:
            if not isinstance(expected_inventory, dict) or set(expected_inventory) != set(files):
                raise CacheInvalid('receipt inventory must exactly match artifact names')
            for item in expected_inventory.values():
                if (not isinstance(item, dict) or set(item) != {'bytes', 'sha256'}
                        or type(item['bytes']) is not int or item['bytes'] < 0
                        or not isinstance(item['sha256'], str) or re.fullmatch('[0-9a-f]{64}', item['sha256']) is None):
                    raise CacheInvalid('invalid artifact receipt size/hash')
        existing = self.lookup(identity, check=check, max_metadata_bytes=max_metadata_bytes,
                               max_entry_files=max_entry_files if target_max_entry_files is None else target_max_entry_files,
                               max_payload_bytes=max_payload_bytes)
        if existing is not None:
            if expected_inventory is not None and {item['name']: {'bytes': item['bytes'], 'sha256': item['sha256']}
                                                    for item in existing['files']} != expected_inventory:
                raise CacheInvalid('artifact receipt differs from existing sealed bytes')
            return existing
        inventory = []
        for index, (name, source) in enumerate(sorted(files.items())):
            relative_name(name)
            source = Path(source)
            if source.is_symlink() or not source.is_file():
                raise CacheInvalid('cache captures only regular artifact files')
            source_size = source.stat().st_size
            if max_payload_bytes is not None and source_size > max_payload_bytes:
                raise CacheCapacity('artifact exceeds bounded payload admission')
            if expected_inventory is not None and source_size != expected_inventory[name]['bytes']:
                raise CacheInvalid('artifact size differs from fsynced receipt')
            inventory.append({'name': name, 'blob': f'payload-{index:06d}.bin', 'bytes': source_size,
                              'sha256': (expected_inventory[name]['sha256'] if expected_inventory is not None else
                                         file_sha256(source, check=check) if max_payload_bytes is None else
                                         _file_sha256_bounded(source, source_size, check=check))})
        if not inventory:
            raise CacheInvalid('cannot seal an empty artifact inventory')
        validate_inventory_names([item['name'] for item in inventory])
        manifest = {'schema_version': 1, 'key': identity.key, 'identity': identity.to_dict(),
                    'files': inventory, 'origin_record': record}
        owner = {'owner': self.owner, 'manifest_sha256': hashlib.sha256(canonical(manifest).encode()).hexdigest()}
        needed = sum(item['bytes'] for item in inventory) + len(canonical(manifest).encode()) + len(canonical(owner).encode())
        if needed > self.max_bytes:
            raise CacheCapacity('entry exceeds the configured cache byte budget')
        if self.structural_admission:
            from . import cache_maintenance as maintenance
            manifest_bytes, owner_bytes = len(canonical(manifest).encode()), len(canonical(owner).encode())
            if (max(manifest_bytes, owner_bytes) > maintenance.MAX_RECORD_BYTES
                    or manifest_bytes + owner_bytes + max(manifest_bytes, owner_bytes) > maintenance.MAX_METADATA_BYTES):
                raise CacheCapacity('artifact metadata cannot fit managed workspace structural limits')
            self._structural_scan(check)
            self._make_room(needed, max_metadata_bytes=maintenance.MAX_RECORD_BYTES, max_entry_files=maintenance.MAX_FILES)
            self._structural_headroom(manifest, owner, len(inventory), check)
        else:
            self._make_room(needed, max_metadata_bytes=max_metadata_bytes, max_entry_files=max_entry_files)
        stage = Path(tempfile.mkdtemp(prefix=f'entry-{self.owner}-', dir=self.root / 'staging'))
        try:
            write_record(stage / 'owner.json', owner)
            for item in inventory:
                _copy_checked(files[item['name']], stage / item['blob'], item['sha256'], expected_size=item['bytes'], check=check)
            if check is not None:
                check()
            write_record(stage / 'complete.json', manifest)
            os.rename(stage, self._entry(identity))
            _sync_directory(self.root / 'entries')
        finally:
            if stage.exists():
                cleanup_all([lambda: self._discard(stage, max_metadata_bytes=max_metadata_bytes,
                                                    max_entry_files=max_entry_files if target_max_entry_files is None else target_max_entry_files)],
                            primary_error=sys.exception())
        return manifest

    def discard(self, identity, *, max_metadata_bytes=None, max_entry_files=None):
        self._require_owner()
        entry = self._entry(identity)
        if entry.exists():
            self._discard(entry, max_metadata_bytes=max_metadata_bytes, max_entry_files=max_entry_files)

    def restore(self, identity, destination, *, check=None):
        self._require_owner()
        manifest = self.lookup(identity, check=check)
        if manifest is None:
            return None
        destination = Path(destination).resolve()
        if destination.is_relative_to(self.root):
            raise ValueError('render destinations must be outside the recovery workspace')
        destination.mkdir(parents=True, exist_ok=False)
        entry = self._entry(identity)
        for item in manifest['files']:
            target = destination / relative_name(item['name'])
            _copy_checked(entry / item['blob'], target, item['sha256'], expected_size=item['bytes'], check=check)
        write_record(destination / '.replay-complete.json', {'key': identity.key, 'files': manifest['files']})
        _sync_directory(destination)
        return manifest
