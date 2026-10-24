"""Bounded finite local JSON inspection shared by desktop report/snapshot tools."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from .diagnostics import finite_json_float
from .identities import verify_file_observation


def _depth(raw: bytes, maximum: int):
    depth, quoted, escaped = 0, False, False
    for value in raw:
        if quoted:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == 34:
                quoted = False
        elif value == 34:
            quoted = True
        elif value in (91, 123):
            depth += 1
            if depth > maximum:
                raise ValueError('Local JSON nesting exceeds inspection limit.')
        elif value in (93, 125):
            depth -= 1


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate local JSON field.')
        result[key] = value
    return result


def read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Bound a same-descriptor regular-file read; refuse pipes before reading."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('Local byte limit must be a positive integer.')
    path_before = path.stat(follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError('Local JSON requires a regular file within its byte limit.')
        verify_file_observation(path, path_before, before, before)
        raw = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
        if len(raw) > max_bytes:
            raise ValueError('Local JSON grew or changed during its bounded read.')
        verify_file_observation(path, path_before, before, after)
    return raw


def read_object(path: Path, *, max_bytes: int, max_depth: int = 128,
                regular_file: bool = False) -> tuple[dict[str, Any], bytes]:
    if type(max_bytes) is not int or max_bytes < 1 or type(max_depth) is not int or max_depth < 1:
        raise ValueError('Local JSON byte/depth limits must be positive integers.')
    if regular_file:
        raw = read_regular_bytes(path, max_bytes=max_bytes)
    else:
        with path.open('rb') as stream:
            raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError('Local JSON exceeds inspection byte limit.')
    _depth(raw, max_depth)
    try:
        data = json.loads(raw, object_pairs_hook=_pairs, parse_float=finite_json_float, parse_constant=finite_json_float)
    except RecursionError as exc:
        raise ValueError('Local JSON nesting exceeds decoder limit.') from exc
    if not isinstance(data, dict):
        raise TypeError('Local JSON must be an object.')
    return data, raw
