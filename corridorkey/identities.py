"""Local content identities shared by correction preflight and benchmarks."""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from .scheduling import frame_paths


def _stat_identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def verify_file_observation(path, path_before, fd_before, fd_after):
    """Keep path and descriptor mutation fences separate; return path metadata."""
    current = Path(path).stat(follow_symlinks=False)
    if (_stat_identity(fd_before) != _stat_identity(fd_after)
            or _stat_identity(path_before) != _stat_identity(current)
            or fd_before.st_nlink != fd_after.st_nlink
            or path_before.st_nlink != current.st_nlink):
        raise ValueError('File changed during descriptor-bound observation')
    if sys.platform != 'win32':
        if _stat_identity(current) != _stat_identity(fd_after):
            raise ValueError('File path and descriptor identities differ')
        return current
    # Windows path ctime may mean birthtime while fstat ctime means ChangeTime.
    if (not stat.S_ISREG(current.st_mode) or current.st_nlink != fd_after.st_nlink
            or getattr(current, 'st_file_attributes', 0) & 0x400
            or _stat_identity(current)[:4] != _stat_identity(fd_after)[:4]):
        raise ValueError('File path is not the admitted regular file')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    descriptor = os.open(path, flags)
    try:
        reopened = os.fstat(descriptor)
        if (not stat.S_ISREG(reopened.st_mode) or reopened.st_nlink != fd_after.st_nlink
                or getattr(reopened, 'st_file_attributes', 0) & 0x400
                or _stat_identity(reopened) != _stat_identity(fd_after)
                or _stat_identity(Path(path).stat(follow_symlinks=False)) != _stat_identity(current)):
            raise ValueError('File changed while rebinding its descriptor')
    finally:
        os.close(descriptor)
    return current


def file_sha256(path, *, check=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if check is not None:
                check()
            digest.update(block)
    if check is not None:
        check()
    return digest.hexdigest()


def sequence_identity(path, files):
    """Serialize already verified ordered frame hashes using the existing wire identity."""
    files = [dict(item) for item in files]
    return {"path": str(Path(path)), "files": files,
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def source_identity(path, *, check=None):
    path = Path(path)
    if path.is_file():
        return {"path": str(path), "sha256": file_sha256(path, check=check)}
    files = [{"name": Path(file).name, "sha256": file_sha256(file, check=check)}
             for file in frame_paths(path)]
    return sequence_identity(path, files)
