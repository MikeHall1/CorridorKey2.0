"""Bounded local-file admission and owned fresh-directory collection primitives.

No recursive directory collection: callers provide every admitted file explicitly.
File fingerprints fence ordinary mutation, not hostile concurrent filesystem owners.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .identities import verify_file_observation

Check = Callable[[], None]
MAX_FILES = 100000
CHUNK_BYTES = 1024 * 1024


def check_now(check: Check | None) -> None:
    if check is not None:
        check()


def real_path(path: str | Path, *, directory: bool | None = None, missing: bool = False) -> Path:
    """Absolute lexical path; refuse symlink/reparse components before resolving."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError(f'Alias/symlink/reparse path refused: {part}')
    if not missing:
        info = path.lstat()
        if directory is True and not stat.S_ISDIR(info.st_mode):
            raise ValueError('Expected a real directory')
        if directory is False and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise ValueError('Expected a regular single-link local file')
    return path


def fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('Asset must remain a regular single-link file')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@dataclass(frozen=True)
class FilePin:
    path: str
    fingerprint: tuple[int, int, int, int, int]
    sha256: str = ''

    @property
    def size(self) -> int:
        return self.fingerprint[2]


def admit_file(path: str | Path) -> FilePin:
    path = real_path(path, directory=False)
    return FilePin(str(path), fingerprint(path.lstat()))


def assert_pin(pin: FilePin) -> None:
    path = real_path(pin.path, directory=False)
    if fingerprint(path.lstat()) != pin.fingerprint:
        raise ValueError(f'Admitted asset changed: {pin.path}')


def read_file(pin: FilePin, *, check: Check | None = None, capture: bool = False,
              destination: Path | None = None) -> tuple[str, bytes]:
    """Read exactly admitted bytes through one pinned FD; optional exclusive copy."""
    check_now(check)
    assert_pin(pin)
    path_before = Path(pin.path).lstat()
    if fingerprint(path_before) != pin.fingerprint:
        raise ValueError('Asset changed before pinned read')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(pin.path, flags)
    output = None
    blocks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(fd)
        fingerprint(before)
        verify_file_observation(pin.path, path_before, before, before)
        if destination is not None:
            output = destination.open('xb')
        remaining = pin.size
        while remaining:
            check_now(check)
            block = os.read(fd, min(CHUNK_BYTES, remaining))
            if not block:
                raise ValueError('Asset shrank during read')
            remaining -= len(block)
            digest.update(block)
            if capture:
                blocks.append(block)
            if output is not None:
                output.write(block)
        if os.read(fd, 1):
            raise ValueError('Asset grew/changed during read')
        verify_file_observation(pin.path, path_before, before, os.fstat(fd))
        assert_pin(pin)
        check_now(check)
        result = digest.hexdigest()
        if pin.sha256 and result != pin.sha256:
            raise ValueError('Asset content identity changed')
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
        return result, b''.join(blocks)
    finally:
        os.close(fd)
        if output is not None:
            output.close()


def fresh_directory(parent: Path, prefix: str) -> Path:
    parent = real_path(parent, directory=True)
    root = parent / (prefix + uuid.uuid4().hex)
    root.mkdir(mode=0o700)
    return root


def cleanup_owned(root: Path) -> None:
    # Only called for a freshly created exclusive root owned by this operation.
    shutil.rmtree(root)


def safe_name(name: object) -> str:
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or any(character in name for character in '/\\:*?"<>|\x00')
            or any(ord(character) < 32 for character in name) or name.rstrip(' .') != name or len(name.encode('utf-8')) > 255):
        raise ValueError('Unsafe/nonportable collection basename')
    if name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('Reserved collection basename')
    return name
