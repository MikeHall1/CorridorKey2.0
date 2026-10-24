"""Bound encoded bytes and declared native layout before optional image decoding.

This is conservative header admission, not a codec validity/authenticity proof.
No decoder runs here. Callers must decode encoded_bytes, then check actual shape
and channel/dtype contracts. EXR is restricted to zero-origin single-part scanline
Y/RGB/RGBA HALF/FLOAT with identical display/data windows and unit sampling.
"""
from __future__ import annotations

import hashlib
import os
import stat
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from .identities import verify_file_observation

MAX_ENCODED_BYTES = 64 * 1024**2
MAX_HEADER_BYTES = 1024**2
MAX_PIXELS = 4 * 1024**2
# Larger callers must opt in; existing workflow defaults remain unchanged.
HARD_MAX_PIXELS = 16 * 1024**2
HARD_MAX_ENCODED_BYTES = 256 * 1024**2
MAX_EXTENT = 32768


@dataclass(frozen=True)
class AdmittedImage:
    path: Path
    encoded_bytes: bytes = field(repr=False)
    sha256: str
    shape: tuple[int, int]
    channels: int
    bit_depth: int
    format: str


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _lexical(path):
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or getattr(value, 'st_file_attributes', 0) & 0x400:
        raise ValueError('Image must not be a symlink/reparse alias.')
    return value


def _read(path, max_bytes, check):
    # Pin the file descriptor; parent aliases are refused at admission/final fence.
    for parent in path.parents:
        _lexical(parent)
    initial = _lexical(path)
    if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
        raise ValueError('Image must be a regular single-link file without aliases.')
    if not 0 < initial.st_size <= max_bytes:
        raise ValueError('Image encoded byte bound exceeded or file empty.')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError('Image changed before pinned read.')
        verify_file_observation(path, initial, before, before)
        blocks, remaining = [], before.st_size
        digest = hashlib.sha256()
        while remaining:
            check()
            block = os.read(fd, min(1024**2, remaining))
            if not block:
                raise ValueError('Image changed/truncated during pinned read.')
            blocks.append(block)
            digest.update(block)
            remaining -= len(block)
        check()
        if os.read(fd, 1) or _identity(os.fstat(fd)) != _identity(before):
            raise ValueError('Image changed/grew during pinned read.')
        final = verify_file_observation(path, initial, before, os.fstat(fd))
        if _identity(_lexical(path)) != _identity(final):
            raise ValueError('Image path replaced during pinned read.')
        for parent in path.parents:
            _lexical(parent)
        return b''.join(blocks), digest.hexdigest(), _identity(final)
    finally:
        os.close(fd)


def _png(raw, check):
    if len(raw) < 33 or raw[8:16] != b'\0\0\0\rIHDR':
        raise ValueError('Invalid/truncated PNG IHDR.')
    width, height, depth, colour, compression, filtering, interlace = struct.unpack('>IIBBBBB', raw[16:29])
    if colour not in (0, 2, 6) or depth not in (8, 16) or compression or filtering or interlace not in (0, 1):
        raise ValueError('PNG admission supports only 8/16-bit grayscale/RGB/RGBA.')
    pos, count, has_data, ended = 8, 0, False, False
    while pos < len(raw):
        check()
        if pos + 12 > len(raw) or count >= 4096:
            raise ValueError('PNG chunk bounds/count exceeded or truncated.')
        size = struct.unpack_from('>I', raw, pos)[0]
        end = pos + 12 + size
        if end > len(raw):
            raise ValueError('Truncated PNG chunk.')
        kind = raw[pos + 4:pos + 8]
        if zlib.crc32(raw[pos + 4:pos + 8 + size]) != struct.unpack_from('>I', raw, pos + 8 + size)[0]:
            raise ValueError('Invalid PNG chunk checksum.')
        if count and kind == b'IHDR':
            raise ValueError('Duplicate PNG IHDR.')
        if kind in (b'acTL', b'fcTL', b'fdAT', b'tRNS'):
            raise ValueError('Animated PNG/auxiliary transparency is not admitted; use explicit static gray/RGB/RGBA.')
        if kind == b'IDAT':
            has_data = True
        if kind == b'IEND':
            if size or end != len(raw):
                raise ValueError('Invalid PNG end/trailing data.')
            ended = True
            break
        count += 1
        pos = end
    if not ended or not has_data:
        raise ValueError('PNG missing image data/end.')
    return (height, width), {0: 1, 2: 3, 6: 4}[colour], depth, 'png'


def _jpeg(raw, check):
    pos, count, header_bytes, layout, in_scan, scanned = 2, 0, 0, None, False, False
    while pos < len(raw):
        check()
        if count >= 65536:
            raise ValueError('JPEG marker count exceeds bound.')
        if in_scan:
            pos = raw.find(b'\xff', pos)
            if pos < 0:
                break
        elif raw[pos] != 255:
            raise ValueError('Unsupported JPEG marker layout.')
        while pos < len(raw) and raw[pos] == 255:
            pos += 1
        if pos >= len(raw):
            break
        marker = raw[pos]
        pos += 1
        count += 1
        if in_scan and (marker == 0 or 0xD0 <= marker <= 0xD7):
            continue
        in_scan = False
        if marker == 0xD9:
            if layout is None or not scanned or pos != len(raw):
                raise ValueError('JPEG end/layout/trailing bytes unsupported.')
            return layout
        if marker in (0, 0xD8) or 0xD0 <= marker <= 0xD7:
            raise ValueError('Unsupported JPEG header marker.')
        if pos + 2 > len(raw):
            break
        size = struct.unpack_from('>H', raw, pos)[0]
        header_bytes += size
        if size < 2 or pos + size > len(raw) or header_bytes > MAX_HEADER_BYTES:
            raise ValueError('JPEG segment truncated or header bound exceeded.')
        if marker in set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}:
            if marker not in (0xC0, 0xC2) or layout is not None or size < 8:
                raise ValueError('Only one 8-bit baseline/progressive JPEG frame is admitted.')
            depth, height, width, channels = struct.unpack_from('>BHHB', raw, pos + 2)
            if depth != 8 or channels not in (1, 3) or size != 8 + 3 * channels:
                raise ValueError('JPEG must be 8-bit grayscale/RGB without CMYK.')
            layout = ((height, width), channels, depth, 'jpeg')
        if marker == 0xDA:
            if layout is None:
                raise ValueError('JPEG scan has no admitted frame header.')
            in_scan, scanned = True, True
        pos += size
    raise ValueError('Missing/truncated/bounded JPEG frame and end marker.')


def _exr(raw, check, max_pixels):
    if len(raw) < 8:
        raise ValueError('Truncated EXR version.')
    version = struct.unpack_from('<I', raw, 4)[0]
    if version & ~0x4FF or version & 255 != 2:
        raise ValueError('Tiled/deep/multipart EXR is not admitted.')
    limit, pos, attrs = min(len(raw), MAX_HEADER_BYTES), 8, {}
    name_limit = 255 if version & 0x400 else 31

    def text():
        nonlocal pos
        end = raw.find(b'\0', pos, min(limit, pos + name_limit + 1))
        if end < 0:
            raise ValueError('EXR name exceeds bounded header.')
        value, pos = raw[pos:end], end + 1
        return value

    while True:
        check()
        name = text()
        if not name:
            break
        kind = text()
        if len(attrs) >= 64 or name in attrs or pos + 4 > limit:
            raise ValueError('EXR duplicate/attribute-count/header bounds.')
        size = struct.unpack_from('<i', raw, pos)[0]
        pos += 4
        if size < 0 or pos + size > limit:
            raise ValueError('EXR attribute exceeds bounded header.')
        attrs[name] = (kind, raw[pos:pos + size])
        pos += size
    if b'tiles' in attrs or (b'type' in attrs and attrs[b'type'] != (b'string', b'scanlineimage')):
        raise ValueError('Only single-part scanline EXR is admitted.')
    window = attrs.get(b'dataWindow')
    if window is None or window[0] != b'box2i' or len(window[1]) != 16 or attrs.get(b'displayWindow') != window:
        raise ValueError('EXR requires identical bounded data/display windows.')
    x0, y0, x1, y1 = struct.unpack('<iiii', window[1])
    if x0 or y0 or x1 < 0 or y1 < 0:
        raise ValueError('Only zero-origin EXR windows are admitted.')
    if max(x1 + 1, y1 + 1) > MAX_EXTENT or (x1 + 1) * (y1 + 1) > max_pixels:
        raise ValueError('EXR native pixel/extent bound exceeded.')
    channel_attr = attrs.get(b'channels')
    if channel_attr is None or channel_attr[0] != b'chlist':
        raise ValueError('Missing EXR channels.')
    channels, types, ch, cp = [], set(), channel_attr[1], 0
    while cp < len(ch) and ch[cp]:
        end = ch.find(b'\0', cp, min(len(ch), cp + name_limit + 1))
        if end < 0 or end + 17 > len(ch) or len(channels) >= 4:
            raise ValueError('Invalid/bounded EXR channels.')
        name = ch[cp:end]
        pixel_type, linear, xs, ys = struct.unpack_from('<iB3xii', ch, end + 1)
        if name in channels or pixel_type not in (1, 2) or linear not in (0, 1) or xs != 1 or ys != 1:
            raise ValueError('EXR requires unique HALF/FLOAT channels with unit sampling.')
        channels.append(name)
        types.add(pixel_type)
        cp = end + 17
    if cp != len(ch) - 1 or ch[-1:] != b'\0' or set(channels) not in ({b'Y'}, {b'R', b'G', b'B'}, {b'R', b'G', b'B', b'A'}) or len(types) != 1:
        raise ValueError('Only homogeneous HALF/FLOAT Y/RGB/RGBA EXR is admitted.')
    compression = attrs.get(b'compression')
    if compression is None or compression[0] != b'compression' or compression[1] not in tuple(bytes([x]) for x in range(10)):
        raise ValueError('Unsupported EXR compression declaration.')
    if attrs.get(b'lineOrder') not in ((b'lineOrder', b'\0'), (b'lineOrder', b'\1')):
        raise ValueError('Unsupported EXR scanline ordering.')
    block_lines = (1, 1, 1, 16, 32, 16, 32, 32, 32, 256)[compression[1][0]]
    chunks = (y1 + block_lines) // block_lines
    table_end = pos + chunks * 8
    if table_end > len(raw):
        raise ValueError('Truncated EXR scanline offset table.')
    intervals, rows = [], set()
    for index in range(chunks):
        check()
        offset = struct.unpack_from('<Q', raw, pos + index * 8)[0]
        if offset < table_end or offset + 8 > len(raw):
            raise ValueError('EXR scanline offset is outside captured bytes.')
        row, size = struct.unpack_from('<ii', raw, offset)
        if row < 0 or row > y1 or row % block_lines or row in rows or size <= 0 or offset + 8 + size > len(raw):
            raise ValueError('Truncated/duplicate/invalid EXR scanline chunk.')
        rows.add(row)
        intervals.append((offset, offset + 8 + size))
    intervals.sort()
    if any(right[0] < left[1] for left, right in pairwise(intervals)):
        raise ValueError('Overlapping EXR scanline chunks.')
    return (y1 + 1, x1 + 1), len(channels), {1: 16, 2: 32}[next(iter(types))], 'exr'


def admit_image(path: str | os.PathLike, *, expected_hw: tuple[int, int] | None,
                max_pixels: int, max_bytes: int = MAX_ENCODED_BYTES,
                check: Callable[[], None] | None = None) -> AdmittedImage:
    """Capture one bounded regular file and admit its native header before decode."""
    check = check or (lambda: None)
    check()
    if type(max_pixels) is not int or not 1 <= max_pixels <= HARD_MAX_PIXELS:
        raise ValueError('Native pixel bound must be an integer within the hard cap.')
    if type(max_bytes) is not int or not 1 <= max_bytes <= HARD_MAX_ENCODED_BYTES:
        raise ValueError('Encoded byte bound must be an integer within the hard cap.')
    if expected_hw is not None and (type(expected_hw) is not tuple or len(expected_hw) != 2 or any(type(v) is not int or v <= 0 for v in expected_hw) or expected_hw[0] * expected_hw[1] > max_pixels):
        raise ValueError('Expected native geometry is invalid or exceeds pixel bound.')
    lexical_path = Path(os.path.abspath(path))
    raw, digest, identity = _read(lexical_path, max_bytes, check)
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        layout = _png(raw, check)
    elif raw.startswith(b'\xff\xd8'):
        layout = _jpeg(raw, check)
    elif raw.startswith(struct.pack('<I', 20000630)):
        layout = _exr(raw, check, max_pixels)
    else:
        raise ValueError('Only admitted PNG/JPEG/scanline HALF/FLOAT EXR headers are supported.')
    hw, channels, depth, format_name = layout
    if min(hw) <= 0 or max(hw) > MAX_EXTENT or hw[0] * hw[1] > max_pixels or (expected_hw is not None and hw != expected_hw):
        raise ValueError('Image header native geometry mismatches expected shape or exceeds pixel bound.')
    check()
    if _identity(_lexical(lexical_path)) != identity:
        raise ValueError('Image path changed while admitting its header.')
    for parent in lexical_path.parents:
        _lexical(parent)
    return AdmittedImage(lexical_path, raw, digest, hw, channels, depth, format_name)
