"""Strict native three-class assets and bounded, final-only joint refinement.

Classes are coverage constraints, never RGB or additional neural channels.
The fixed local filter is a diagnostic/editor operation, not a learned matte.
"""
from __future__ import annotations

import hashlib
import os
import stat
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

MAX_ENCODED_BYTES = 256 * 1024**2
MAX_NATIVE_PIXELS = 64 * 1024**2
MAX_EXR_HEADER_BYTES = 1024**2


@dataclass(frozen=True)
class TrimapAsset:
    labels: Any
    encoded_bytes: bytes
    sha256: str
    format: str


def refinement_contract():
    return {"version": "native-trimap-joint-v1", "radius": 1, "sigma": .1, "spatial_sigma": .5,
            "classes": {"background": 0, "unknown": 128, "foreground": 255},
            "domain": "final_native_only", "neural_conditioning": "skipped",
            "order": "authored_entries", "strength": "blend_current_to_target_once"}


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read(path, check):
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        lexical = os.lstat(path)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or stat.S_ISLNK(lexical.st_mode) or _identity(before) != _identity(lexical)
                or getattr(lexical, 'st_file_attributes', 0) & 0x400):
            raise ValueError('trimap must be a regular single-link file without aliases')
        if not 0 < before.st_size <= MAX_ENCODED_BYTES:
            raise ValueError('trimap encoded byte limit exceeded or file empty')
        chunks, remaining = [], before.st_size
        digest = hashlib.sha256()
        while remaining:
            check()
            block = os.read(fd, min(1024**2, remaining))
            if not block:
                raise ValueError('trimap changed/truncated during read')
            digest.update(block)
            chunks.append(block)
            remaining -= len(block)
        if os.read(fd, 1) or _identity(os.fstat(fd)) != _identity(before):
            raise ValueError('trimap changed during read')
        if _identity(os.lstat(path)) != _identity(before):
            raise ValueError('trimap path changed during read')
        return b''.join(chunks), digest.hexdigest()
    finally:
        os.close(fd)


def _admit_header(raw, hw):
    h, w = hw
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        if len(raw) < 33 or raw[8:16] != b'\0\0\0\rIHDR':
            raise ValueError('invalid PNG IHDR')
        width, height, depth, colour, compression, filtering, interlace = struct.unpack('>IIBBBBB', raw[16:29])
        if zlib.crc32(raw[12:29]) != struct.unpack('>I', raw[29:33])[0]:
            raise ValueError('invalid PNG IHDR checksum')
        if (height, width) != hw or colour != 0 or depth not in (8, 16):
            raise ValueError('trimap PNG must be native-size single-channel 8/16 bit')
        if compression or filtering or interlace not in (0, 1):
            raise ValueError('unsupported PNG layout')
        return 'png'
    if len(raw) < 8 or struct.unpack('<I', raw[:4])[0] != 20000630:
        raise ValueError('trimap must be a PNG or scanline floating EXR')
    version = struct.unpack('<I', raw[4:8])[0]
    if version & ~0x4ff or version & 255 != 2:
        raise ValueError('unsupported tiled/deep/multipart EXR layout')
    name_limit = 255 if version & 0x400 else 31
    pos, attrs = 8, {}
    limit = min(len(raw), MAX_EXR_HEADER_BYTES)

    def text():
        nonlocal pos
        end = raw.find(b'\0', pos, min(limit, pos + name_limit + 1))
        if end < 0:
            raise ValueError('invalid/bounded EXR attribute name')
        value = raw[pos:end]
        pos = end + 1
        return value

    while True:
        name = text()
        if not name:
            break
        kind = text()
        if len(attrs) >= 64 or name in attrs or pos + 4 > limit:
            raise ValueError('EXR header bounds/duplicate attribute')
        size = struct.unpack('<i', raw[pos:pos + 4])[0]
        pos += 4
        if size < 0 or pos + size > limit:
            raise ValueError('EXR attribute exceeds header bound')
        attrs[name] = (kind, raw[pos:pos + size])
        pos += size
    if b'tiles' in attrs or (b'type' in attrs and attrs[b'type'] != (b'string', b'scanlineimage')):
        raise ValueError('unsupported EXR image type')
    expected_window = struct.pack('<iiii', 0, 0, w - 1, h - 1)
    for name in (b'dataWindow', b'displayWindow'):
        if attrs.get(name) != (b'box2i', expected_window):
            raise ValueError('trimap EXR must have native-size zero-origin windows')
    channels = attrs.get(b'channels')
    if channels is None or channels[0] != b'chlist':
        raise ValueError('missing EXR channels')
    ch = channels[1]
    end = ch.find(b'\0')
    if end < 1 or len(ch) != end + 18 or ch[-1:] != b'\0':
        raise ValueError('trimap EXR must have exactly one channel')
    pixel_type, _linear, xs, ys = struct.unpack('<iB3xii', ch[end + 1:-1])
    if ch[:end] != b'Y' or pixel_type not in (1, 2) or xs != 1 or ys != 1:
        raise ValueError('trimap EXR requires one floating Y channel with unit sampling')
    compression = attrs.get(b'compression')
    if compression is None or compression[0] != b'compression' or compression[1] not in tuple(bytes([x]) for x in range(5)):
        raise ValueError('unsupported/lossy EXR compression for categorical trimap')
    return 'exr'


def load_trimap_asset(path, *, expected_hw, expected_sha256=None, check=None):
    """Read once, admit encoded geometry before decoding, retain exact class bits."""
    import cv2
    import numpy as np

    check = check or (lambda: None)
    hw = tuple(expected_hw)
    if (len(hw) != 2 or any(type(x) is not int or x <= 0 for x in hw)
            or hw[0] * hw[1] > MAX_NATIVE_PIXELS):
        raise ValueError('trimap native geometry exceeds bounds')
    check()
    raw, digest = _read(Path(path), check)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('trimap asset hash changed')
    format_name = _admit_header(raw, hw)
    check()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != hw:
        raise ValueError('trimap decoder must return exactly native single-channel pixels')
    if image.dtype == np.uint8:
        values = (0, 128, 255)
    elif image.dtype == np.uint16:
        values = (0, 32768, 65535)
    elif np.issubdtype(image.dtype, np.floating):
        values = (0, .5, 1)
    else:
        raise ValueError('unsupported trimap precision')
    valid = (image == values[0]) | (image == values[1]) | (image == values[2])
    if not valid.all():
        raise ValueError('trimap contains values outside exact background/unknown/foreground classes')
    labels = np.zeros(hw, dtype=np.uint8)
    labels[image == values[1]] = 128
    labels[image == values[2]] = 255
    labels.setflags(write=False)
    check()
    return TrimapAsset(labels, raw, digest, format_name)


def refine_trimap(alpha, labels, guidance, *, strength=1.0, check=None):
    """Seed known neighbours, filter unknown centres, blend once; no endpoint lock."""
    import math

    import numpy as np

    check = check or (lambda: None)
    alpha = np.asarray(alpha, dtype=np.float32)
    labels = np.asarray(labels)
    guidance = np.asarray(guidance, dtype=np.float32)
    if (alpha.ndim != 2 or alpha.size > MAX_NATIVE_PIXELS or labels.shape != alpha.shape
            or guidance.shape != (*alpha.shape, 3) or not np.isfinite(alpha).all()
            or not np.isfinite(guidance).all() or np.any((alpha < 0) | (alpha > 1))
            or np.any((guidance < 0) | (guidance > 1))
            or not np.isin(labels, (0, 128, 255)).all()):
        raise ValueError('trimap refinement requires finite native coverage/classes/RGB proxy')
    if isinstance(strength, bool) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('trimap strength must be finite in [0,1]')
    check()
    result = alpha.copy()
    if strength == 0:
        return result
    h, w = alpha.shape
    # Only tile-local float scratch. Border neighbours replicate native edges.
    for top in range(0, h, 128):
        for left in range(0, w, 128):
            check()
            bottom, right = min(top + 128, h), min(left + 128, w)
            yy = np.clip(np.arange(top - 1, bottom + 1), 0, h - 1)
            xx = np.clip(np.arange(left - 1, right + 1), 0, w - 1)
            seed = alpha[np.ix_(yy, xx)].copy()
            classes = labels[np.ix_(yy, xx)]
            seed[classes == 0], seed[classes == 255] = 0, 1
            guide = guidance[np.ix_(yy, xx)]
            centre = guide[1:-1, 1:-1]
            total, weights = np.zeros_like(seed[1:-1, 1:-1]), np.zeros_like(seed[1:-1, 1:-1])
            th, tw = bottom - top, right - left
            for dy in range(3):
                for dx in range(3):
                    check()
                    delta = guide[dy:dy + th, dx:dx + tw] - centre
                    weight = np.exp(-np.sum(delta * delta, axis=2) / (.02)
                                    - ((dy - 1)**2 + (dx - 1)**2) / .5)
                    total += weight * seed[dy:dy + th, dx:dx + tw]
                    weights += weight
            target = total / weights
            local_labels = labels[top:bottom, left:right]
            target[local_labels == 0], target[local_labels == 255] = 0, 1
            original = alpha[top:bottom, left:right]
            result[top:bottom, left:right] = original * (1 - strength) + target * strength
    return np.clip(result, 0, 1).astype(np.float32, copy=False)
