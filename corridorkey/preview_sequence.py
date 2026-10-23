"""Bounded frozen local sequence membership for preview and guidance readers.

The manifest identity covers paths/stat metadata, not source content. Native
image reads retain the shared decoder's memory requirements; no pixel cache or
process-wide RAM guarantee is introduced. Files added after scanning are ignored
until explicit re-preparation. Changed/missing indexed files refuse to decode.
"""
from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .desktop_comparison import RenderView
from .scheduling import frame_path_key
from .session import JobRequest

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

MAX_SEQUENCE_ENTRIES = 100000
MAX_SEQUENCE_FRAMES = 100000
MAX_SEQUENCE_PATH_BYTES = 32 * 1024**2


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Indexed source must remain a regular local image file.')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


@dataclass(frozen=True)
class SequenceIndex:
    source: str
    paths: tuple[str, ...]
    fingerprints: tuple[tuple[int, int, int, int], ...]
    identity: str
    entries_inspected: int
    path_bytes: int

    def check_path(self, index: int) -> None:
        if type(index) is not int or not 0 <= index < len(self.paths):
            raise ValueError('Source frame lies outside the prepared sequence index.')
        if _fingerprint(Path(self.paths[index])) != self.fingerprints[index]:
            raise ValueError('Indexed source file changed. Prepare source playback again explicitly.')

    def path_at(self, index: int) -> str:
        self.check_path(index)
        return self.paths[index]

    def read_rgb(self, index: int, *, alpha_mode: str = 'ignore') -> NDArray[np.float32]:
        import numpy as np

        from .video_io import read_image_rgb
        pixels = np.asarray(read_image_rgb(self.path_at(index), alpha_mode=alpha_mode), dtype=np.float32)
        self.check_path(index)
        return pixels


def scan_sequence(source: str | Path, *, check: Callable[[], None] | None = None,
                  max_entries: int = MAX_SEQUENCE_ENTRIES,
                  max_frames: int = MAX_SEQUENCE_FRAMES,
                  max_path_bytes: int = MAX_SEQUENCE_PATH_BYTES) -> SequenceIndex:
    """Stream directory admission before allocating a bounded natural sort."""
    for value in (max_entries, max_frames, max_path_bytes):
        if type(value) is not int or value < 1:
            raise ValueError('Sequence index limits must be exact positive integers.')
    check = check or (lambda: None)
    check()
    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError('Source playback requires an image-sequence directory; video/VFR timing is unsupported.')
    rows = []
    entries, path_bytes = 0, 0
    with os.scandir(root) as directory:
        for entry in directory:
            check()
            entries += 1
            if entries > max_entries:
                raise ValueError('Source directory exceeds the sequence entry limit; choose a smaller local sequence directory.')
            path = root / entry.name
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.exr'} or not entry.is_file():
                continue
            if len(rows) >= max_frames:
                raise ValueError('Source sequence exceeds the indexed-frame limit.')
            path_bytes += len(os.fsencode(path))
            if path_bytes > max_path_bytes:
                raise ValueError('Source sequence exceeds the indexed-path byte limit.')
            rows.append((path, _fingerprint(path)))
    check()
    if not rows:
        raise ValueError('No supported local images found in the source directory.')
    rows.sort(key=lambda row: frame_path_key(row[0]))
    digest = hashlib.sha256(os.fsencode(root))
    for path, fingerprint in rows:
        check()
        digest.update(b'\0' + os.fsencode(path) + b'\0' + repr(fingerprint).encode('ascii'))
    return SequenceIndex(str(root), tuple(str(row[0]) for row in rows), tuple(row[1] for row in rows),
                         digest.hexdigest(), entries, path_bytes)


@dataclass(frozen=True)
class PreparedSequence:
    job: JobRequest
    index: SequenceIndex
    view: RenderView


def prepare_sequence(job: JobRequest, *, check: Callable[[], None] | None = None) -> PreparedSequence:
    """Explicit sequence inspection; call off the GUI thread, never infer."""
    check = check or (lambda: None)
    check()
    config, _ = job.configurations()
    if config.start_mode != 'begin':
        raise ValueError('Source-sequence playback requires explicit start_mode=begin and source In.')
    index = scan_sequence(job.input_path, check=check)
    start = config.start_frame or 0
    if not 0 <= start < len(index.paths):
        raise ValueError('Requested source In lies outside the prepared image sequence.')
    count = min(config.num_frames, len(index.paths) - start) if config.num_frames > 0 else len(index.paths) - start
    check()
    pixels = index.read_rgb(start, alpha_mode=config.frame_dir_alpha)
    check()
    height, width = pixels.shape[:2]
    view = RenderView(index.source, start, count, height, width, float(config.frame_dir_fps), (), False,
                      config.frame_dir_linear, config.frame_dir_alpha, config.source_colour_profile)
    return PreparedSequence(job, index, view)
