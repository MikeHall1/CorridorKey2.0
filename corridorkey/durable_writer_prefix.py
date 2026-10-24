"""Explicit checked EXR frame-prefix reuse; no decoder, neural or state seeking."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .durable_context import BoundModelCache
from .identities import verify_file_observation
from .recovery_identity import ReplayIdentity, canonical
from .recovery_store import (
    CacheCapacity,
    CacheInvalid,
    _copy_checked,
    _sync_directory,
    relative_name,
    validate_inventory_names,
)

MAX_FRAMES = 100000
MAX_IDENTITY_BYTES = 65536
MAX_METADATA_BYTES = 262144
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_ENTRY_FILES = 6
MAX_RESTORED_FILES = 32768
MAX_RESTORED_INVENTORY_BYTES = 16 * 1024 * 1024
ROLES = {'alpha': 1, 'fg': 3, 'cutout': 4, 'checker': 3}
BOUNDS = {'max_metadata_bytes': MAX_METADATA_BYTES, 'max_entry_files': MAX_ENTRY_FILES,
          'max_payload_bytes': MAX_PAYLOAD_BYTES}


def _regular(path, *, directory=False):
    info = Path(path).stat(follow_symlinks=False)
    if (getattr(info, 'st_file_attributes', 0) & 0x400 or
            not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)) or
            (not directory and info.st_nlink != 1)):
        raise CacheInvalid('writer prefix requires owned regular paths without links/reparse points')
    return info


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _relative(name):
    if not isinstance(name, str) or len(name.encode()) > 1024 or len(name.split('/')) > 32:
        raise CacheInvalid('writer prefix relative name exceeds bounds')
    return relative_name(name)


def _layout(outputs):
    if not isinstance(outputs, dict) or not outputs or set(outputs) - set(ROLES):
        raise CacheInvalid('writer prefix requires the exact selected EXR output roles')
    raw = canonical(outputs)
    if len(raw.encode()) > MAX_IDENTITY_BYTES // 2:
        raise CacheInvalid('writer output contract exceeds metadata bounds')
    outputs = json.loads(raw)
    directories = []
    for role, item in outputs.items():
        if not isinstance(item, dict) or set(item) != {'directory', 'frame_name', 'contract'}:
            raise CacheInvalid('invalid writer prefix output layout')
        directory = _relative(item['directory'])
        name = _relative(item['frame_name'])
        if '/' in name or len(name.encode()) > 200:
            raise CacheInvalid('writer frame name must be a bounded basename')
        directories.append(directory)
        contract = item['contract']
        if (not isinstance(contract, dict) or contract.get('format') != 'exr'
                or type(contract.get('channels')) is not int or contract['channels'] != ROLES[role]
                or contract.get('dtype') not in ('half', 'float')
                or (role != 'alpha' and contract.get('rgb_representation') != 'scene_linear')):
            raise CacheInvalid('writer prefix format/channel/dtype/RGB contract differs')
    validate_inventory_names(directories)
    return outputs


def _description(frame_count, native_hw, fps, outputs):
    if type(frame_count) is not int or not 1 <= frame_count <= MAX_FRAMES:
        raise CacheInvalid('writer prefix frame count must be 1..100000')
    if (not isinstance(native_hw, (tuple, list)) or len(native_hw) != 2
            or any(type(size) is not int or not 1 <= size <= 2 ** 31 - 1 for size in native_hw)):
        raise CacheInvalid('writer prefix requires positive native height/width')
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise CacheInvalid('writer prefix requires finite positive FPS')
    return {'frame_count': frame_count, 'native_hw': list(native_hw), 'fps': float(fps), 'outputs': _layout(outputs)}


@dataclass(frozen=True)
class WriterFrameIdentity:
    payload: str

    def __post_init__(self):
        if not isinstance(self.payload, str) or len(self.payload.encode()) > MAX_IDENTITY_BYTES:
            raise CacheInvalid('writer frame identity exceeds metadata bounds')
        try:
            data = json.loads(self.payload)
            expected = {'schema_version', 'kind', 'parent_key', 'frame', 'frame_count', 'native_hw', 'fps', 'outputs'}
            if (not isinstance(data, dict) or set(data) != expected or type(data['schema_version']) is not int
                    or data['schema_version'] != 1 or data['kind'] != 'writer_exr_frame'
                    or not isinstance(data['parent_key'], str) or re.fullmatch('[0-9a-f]{64}', data['parent_key']) is None
                    or type(data['frame']) is not int or not 0 <= data['frame'] < data['frame_count']
                    or self.payload != canonical(data)):
                raise CacheInvalid('invalid canonical writer frame identity')
            _description(data['frame_count'], data['native_hw'], data['fps'], data['outputs'])
        except (KeyError, TypeError, ValueError, RecursionError) as exc:
            raise CacheInvalid('invalid bounded writer frame identity') from exc

    @property
    def key(self):
        return hashlib.sha256(self.payload.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.payload)


class DurableWriterPrefix(BoundModelCache):
    def __init__(self, store, parent_identity, *, validate_context, check=None, resume=False):
        if type(resume) is not bool or not isinstance(parent_identity, ReplayIdentity):
            raise ValueError('writer prefix requires a typed parent and boolean resume policy')
        super().__init__(store, parent_identity, validate_context=validate_context, check=check)
        self.resume_requested = resume
        self.description: dict[str, Any] | None = None
        self.destination: Path | None = None
        self.root_stamp: tuple[int, int] | None = None
        self.next_frame = 0
        self.disabled = False
        self.restored_inventory: dict[str, dict[str, Any]] = {}
        self.restored_inventory_bytes = 0
        self.stats: dict[str, Any] = {'enabled': True, 'resume_requested': resume, 'restored_prefix_frames': 0,
                      'restored_files': 0, 'restored_bytes': 0, 'stored_frames': 0, 'stored_files': 0,
                      'stored_bytes': 0, 'receipt_bytes_hashed': 0, 'corrupt_entries': 0,
                      'budget_rejected_frames': 0, 'checkpoint_disabled': False, 'last_error': None,
                      'partial_restore': None, 'restore_inventory_limited': False,
                      'restored_prefix_verified': None, 'restored_verification_bytes': 0,
                      'chronological_replay': True, 'direct_seek': False, 'native_write_events_reused': False,
                      'retention': 'global LRU; ever-stored counts do not promise a currently retained contiguous prefix',
                      'durability': 'file/directory fsync calls and checked sealed copies; power-loss/target filesystem behaviour not verified'}

    def _check(self):
        self._require_bound()
        if self.destination is not None:
            info = _regular(self.destination, directory=True)
            if (info.st_dev, info.st_ino) != self.root_stamp:
                raise CacheInvalid('owned writer destination changed identity')

    def _identity(self, frame):
        if self.description is None:
            raise CacheInvalid('writer prefix must be prepared before use')
        return WriterFrameIdentity(canonical({'schema_version': 1, 'kind': 'writer_exr_frame',
                                              'parent_key': self.parent.key, 'frame': frame, **self.description}))

    def _names(self, frame):
        if self.description is None:
            raise CacheInvalid('writer prefix must be prepared before use')
        return {role: _relative(f"{item['directory']}/{item['frame_name']}_{frame:06d}.exr")
                for role, item in self.description['outputs'].items()}

    def _path(self, relative, *, create=False):
        if self.destination is None:
            raise CacheInvalid('writer prefix must be prepared before use')
        path = self.destination
        parts = _relative(relative).split('/')
        for part in parts[:-1]:
            path = path / part
            if create and not path.exists():
                path.mkdir()
            _regular(path, directory=True)
        return path / parts[-1]

    def prepare(self, destination, *, frame_count, native_hw, fps, outputs, resume):
        self._check()
        if self.destination is not None or type(resume) is not bool or resume != self.resume_requested:
            raise CacheInvalid('writer prefix prepare requires its captured resume policy and one fresh destination')
        description = _description(frame_count, native_hw, fps, outputs)
        parent = self.parent.to_dict()
        settings = parent.get('settings')
        if isinstance(settings, dict) and (set(settings.get('outputs', [])) != set(outputs)
                                          or settings.get('output_format') != 'exr'
                                          or any(item['contract']['dtype'] != settings.get('exr_dtype') for item in outputs.values())):
            raise CacheInvalid('writer prefix differs from parent job settings')
        destination = Path(destination)
        info = _regular(destination, directory=True)
        if destination.resolve().is_relative_to(self.store.root) or self.store.root.is_relative_to(destination.resolve()):
            raise CacheInvalid('writer output and recovery workspace must be separate')
        if any(destination.iterdir()):
            raise CacheInvalid('writer prefix destination must be fresh and empty')
        self.validate_context()
        self.destination = destination.resolve()
        self.root_stamp = (info.st_dev, info.st_ino)
        self.description = description
        if resume and self.store.max_bytes > 0:
            for frame in range(frame_count):
                self._check()
                identity = self._identity(frame)
                try:
                    manifest = self.store.lookup(identity, check=self.check, **BOUNDS)
                    if manifest is None:
                        break
                    names = self._names(frame)
                    files = {item['name']: item for item in manifest['files']}
                    if (set(files) != set(names.values()) or manifest.get('origin_record') != self._record(identity)):
                        raise CacheInvalid('writer prefix sealed output/receipt contract differs')
                except CacheInvalid as exc:
                    self.stats['corrupt_entries'] += 1
                    self.stats['last_error'] = str(exc)[:512]
                    self.store.discard(identity, max_metadata_bytes=MAX_METADATA_BYTES, max_entry_files=MAX_ENTRY_FILES)
                    break
                receipt_sizes = {name: len(canonical({name: {'bytes': item['bytes'], 'sha256': item['sha256']}}).encode())
                                 for name, item in files.items()}
                if (len(self.restored_inventory) + len(files) > MAX_RESTORED_FILES
                        or self.restored_inventory_bytes + sum(receipt_sizes.values()) > MAX_RESTORED_INVENTORY_BYTES):
                    self.stats['restore_inventory_limited'] = True
                    self.stats['last_error'] = 'restored-prefix receipt inventory capacity reached; suffix will render normally'
                    break
                for name, item in files.items():
                    target = self._path(name, create=True)
                    source = self.store.root / 'entries' / identity.key / item['blob']
                    _regular(source.parent, directory=True)
                    _regular(source)
                    try:
                        _copy_checked(source, target,
                                      item['sha256'], expected_size=item['bytes'], check=self.check)
                    except BaseException:
                        try:
                            if target.exists():
                                self.stats['partial_restore'] = {'path': name, 'observed_bytes': _regular(target).st_size,
                                                                 'verified': False}
                        except (OSError, CacheInvalid):
                            logging.getLogger(__name__).debug('Could not inspect partial writer-prefix copy', exc_info=True)
                            self.stats['partial_restore'] = {'path': name, 'observed_bytes': None, 'verified': False}
                        raise
                    self.restored_inventory[name] = {'bytes': item['bytes'], 'sha256': item['sha256']}
                    self.restored_inventory_bytes += receipt_sizes[name]
                    self.stats['restored_prefix_verified'] = False
                    self.stats['restored_files'] += 1
                    self.stats['restored_bytes'] += item['bytes']
                    _sync_directory(target.parent)
                self.next_frame = frame + 1
                self.stats['restored_prefix_frames'] = self.next_frame
            self.validate_context()
            if self.next_frame:
                _sync_directory(self.destination)
        return self.next_frame

    def _record(self, identity):
        return {'stage': 'durable_exr_writer_frame', 'identity_key': identity.key,
                'receipt': 'all_selected_output_files_fsynced_and_sha256_verified', 'complete_job': False}

    def _receipt(self, path, *, expected=None, sync=True):
        before = _regular(path)
        if expected is not None and before.st_size != expected['bytes']:
            raise CacheInvalid('restored writer prefix size changed')
        if before.st_size <= 0:
            raise CacheInvalid('completed writer EXR is empty')
        limit = min(MAX_PAYLOAD_BYTES, self.store.max_bytes) if sync else MAX_PAYLOAD_BYTES
        if before.st_size > limit:
            raise CacheCapacity('writer EXR exceeds bounded prefix payload admission')
        flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise CacheInvalid('writer EXR changed before durable receipt')
            try:
                verify_file_observation(path, before, opened, opened)
            except ValueError as exc:
                raise CacheInvalid('writer EXR changed before durable receipt') from exc
            if sync:
                os.fsync(descriptor)
            value = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                self._check()
                block = os.read(descriptor, min(1024 * 1024, remaining))
                if not block:
                    raise CacheInvalid('writer EXR truncated during receipt')
                value.update(block)
                remaining -= len(block)
                self.stats['receipt_bytes_hashed' if sync else 'restored_verification_bytes'] += len(block)
            if os.read(descriptor, 1) or _stamp(os.fstat(descriptor)) != _stamp(opened):
                raise CacheInvalid('writer EXR changed during receipt')
            try:
                verify_file_observation(path, before, opened, os.fstat(descriptor))
            except ValueError as exc:
                raise CacheInvalid('writer EXR changed during receipt') from exc
            _regular(path)
        finally:
            os.close(descriptor)
        return {'bytes': before.st_size, 'sha256': value.hexdigest()}

    def verify_restored_prefix(self) -> dict[str, dict[str, Any]]:
        """Revalidate copied bytes after replay and immediately before final sealing.

        This reads the retained bounded expected inventory; it performs no fsync,
        publication, tensor work, or diagnostic native-write events.
        """
        self.stats['restored_prefix_verified'] = False
        try:
            self._check()
            if self.destination is None:
                raise CacheInvalid('writer prefix must be prepared before verification')
            self.validate_context()
            for name, expected in self.restored_inventory.items():
                self._check()
                actual = self._receipt(self._path(name), expected=expected, sync=False)
                if actual != expected:
                    raise CacheInvalid('restored writer prefix bytes changed after copying')
            self.validate_context()
            self._check()
            self.stats['restored_prefix_verified'] = True
            return {name: dict(item) for name, item in self.restored_inventory.items()}
        except OSError as exc:
            self.stats['last_error'] = f'{type(exc).__name__}: {exc}'[:512]
            raise CacheInvalid('restored writer prefix file is unavailable or changed') from exc
        except BaseException as exc:
            self.stats['last_error'] = f'{type(exc).__name__}: {exc}'[:512]
            raise

    def checkpoint(self, frame_ids, *, completed):
        self._check()
        if self.description is None:
            raise CacheInvalid('writer prefix was not prepared')
        if self.disabled:
            return False
        if (not isinstance(frame_ids, (list, tuple)) or not frame_ids or len(frame_ids) > MAX_FRAMES
                or any(type(frame) is not int for frame in frame_ids)
                or frame_ids[0] != self.next_frame
                or list(frame_ids) != list(range(self.next_frame, self.next_frame + len(frame_ids)))
                or frame_ids[-1] >= self.description['frame_count']):
            raise CacheInvalid('writer checkpoint must advance contiguous native output frames')
        if (not isinstance(completed, dict) or set(completed) != set(self.description['outputs'])
                or any(type(cursor) is not int or not frame_ids[-1] < cursor <= self.description['frame_count']
                       for cursor in completed.values())):
            raise CacheInvalid('all output native flush watermarks must cover the complete checkpoint')
        self.validate_context()
        for frame in frame_ids:
            self._check()
            identity = self._identity(frame)
            if self.store.max_bytes == 0:
                self._quota('zero cache quota')
                return False
            files = {name: self._path(name) for name in self._names(frame).values()}
            receipts = {}
            try:
                if sum(_regular(path).st_size for path in files.values()) > self.store.max_bytes:
                    raise CacheCapacity('all-output frame exceeds cache quota')
                receipts = {name: self._receipt(path) for name, path in files.items()}
                for directory in {path.parent for path in files.values()}:
                    _sync_directory(directory)
                self.validate_context()
                published = self.store.publish(identity, files, self._record(identity), check=self.check,
                                               expected_inventory=receipts, **BOUNDS)
                if published.get('origin_record') != self._record(identity):
                    raise CacheInvalid('sealed writer receipt record differs from producer contract')
            except CacheCapacity as exc:
                self._quota(str(exc))
                return False
            except BaseException as exc:
                # Rename may already have committed valid bytes before a directory
                # fsync/cleanup failure. Preserve that fact while retaining the error.
                try:
                    committed = self.store.lookup(identity, check=None, **BOUNDS)
                    expected = {name: {'bytes': item['bytes'], 'sha256': item['sha256']}
                                for name, item in receipts.items()}
                    if (committed is not None and committed.get('origin_record') == self._record(identity)
                            and {item['name']: {'bytes': item['bytes'], 'sha256': item['sha256']}
                                 for item in committed['files']} == expected):
                        self._stored(frame, receipts)
                except BaseException:
                    logging.getLogger(__name__).debug('Could not establish late writer-prefix publication', exc_info=True)
                self.stats['last_error'] = f'{type(exc).__name__}: {exc}'[:512]
                raise
            self._stored(frame, receipts)
        return True

    def _stored(self, frame, receipts):
        self.stats['stored_frames'] += 1
        self.stats['stored_files'] += len(receipts)
        self.stats['stored_bytes'] += sum(item['bytes'] for item in receipts.values())
        self.next_frame = frame + 1

    def _quota(self, reason):
        self.disabled = True
        self.stats['budget_rejected_frames'] += 1
        self.stats['checkpoint_disabled'] = True
        self.stats['last_error'] = reason[:512]

    def snapshot(self):
        return json.loads(canonical(self.stats))
