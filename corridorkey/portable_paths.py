"""Lexical archived job validation, independent of the reading host filesystem.

Archived strings are provenance, never runnable native paths. Rebind every
render dependency and explicit destination before constructing a JobRequest.
An absolute archived executable is not implicitly replaced or made executable.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

from .config import InferenceConfig, WriterConfig
from .diagnostics import finite_json_float

MAX_WIRE_BYTES = 4 * 1024**2
MAX_PATH_BYTES = 32768
_FIELDS = {'input_path','output_dir','config_json','writer_json','initial_hint_path','hint_video_path','profile_stages'}
_DEVICE = re.compile(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', re.IGNORECASE)


def _constant(value: str) -> NoReturn:
    raise ValueError('Archived JSON numbers must be finite')


def _pairs(items):
    result = {}
    for key,value in items:
        if key in result:
            raise ValueError('Duplicate archived JSON field')
        result[key] = value
    return result


def _json(raw: str | bytes) -> dict[str,Any]:
    if not isinstance(raw,(str,bytes)) or len(raw.encode() if isinstance(raw,str) else raw)>MAX_WIRE_BYTES:
        raise ValueError('Archived JSON requires bounded text or immutable bytes')
    value = json.loads(raw,object_pairs_hook=_pairs,parse_constant=_constant,parse_float=finite_json_float)
    if not isinstance(value,dict):
        raise TypeError('Archived JSON must be an object')
    pending: list[tuple[Any,int]] = [(value,0)]
    count = 0
    while pending:
        item,depth = pending.pop()
        count += 1
        if depth>64 or count>100000:
            raise ValueError('Archived JSON structural bound exceeded')
        if isinstance(item,dict):
            pending.extend((child,depth+1) for child in item.values())
        elif isinstance(item,list):
            pending.extend((child,depth+1) for child in item)
        elif isinstance(item,float) and not math.isfinite(item):
            raise ValueError('Archived JSON numbers must be finite')
    return value


def _text(value: str) -> None:
    if not isinstance(value,str) or not value or len(value.encode())>MAX_PATH_BYTES or any(ord(char)<32 for char in value):
        raise ValueError('Archived path requires bounded nonempty text without control characters')


def infer_flavour(value: str) -> str:
    """Infer only explicit POSIX, drive-absolute Windows, or ordinary UNC."""
    _text(value)
    if value.startswith(('\\\\?\\','\\\\.\\')):
        raise ValueError('Windows device/extended namespaces are unsupported')
    if re.match(r'^[A-Za-z]:[\\/]',value) or value.startswith('\\\\'):
        return 'windows'
    if value.startswith('/') and not value.startswith('//'):
        return 'posix'
    raise ValueError('Archived path requires unambiguous absolute origin flavour')


def archived_path(value: str, flavour: str) -> PurePosixPath | PureWindowsPath:
    """Strict absolute path parsing without resolve, stat, cwd, or native I/O."""
    if flavour not in ('posix','windows') or infer_flavour(value)!=flavour:
        raise ValueError('Archived absolute paths must share one origin flavour')
    if flavour=='posix':
        # A leading single slash unambiguously selects POSIX. Backslashes and
        # colons inside components are literal native names, not Windows drives.
        if '//' in value:
            raise ValueError('Ambiguous archived POSIX separators')
        components = value.split('/')[1:]
        if any(part in ('.','..') for part in components):
            raise ValueError('Dot segments in archived path')
        return PurePosixPath(value)
    if '/' in value and '\\' in value:
        raise ValueError('Mixed archived Windows separators')
    separator = '\\' if '\\' in value else '/'
    if value.startswith('\\\\'):
        components = value[2:].split('\\')
        if len(components)<2 or not components[0] or not components[1]:
            raise ValueError('UNC paths require explicit server and share')
    else:
        components = value[3:].split(separator)
    if components==['']:
        components = []  # Drive root is absolute, but has no collection basename.
    for part in components:
        if (not part or part in ('.','..') or part.endswith((' ','.'))
                or any(char in part for char in '<>:"|?*') or _DEVICE.fullmatch(part)):
            raise ValueError('Unsafe/ambiguous archived Windows path component')
    result = PureWindowsPath(value)
    if not result.is_absolute():
        raise ValueError('Drive-relative/root-relative archived Windows paths are unsupported')
    return result


def native_destination(value: str | Path) -> Path:
    """Validate native lexical absolute spelling; caller must admit its filesystem."""
    if not isinstance(value,(str,Path)):
        raise TypeError('Native destination requires a path instance or text')
    text = str(value)
    native = 'windows' if os.name=='nt' else 'posix'
    try:
        archived_path(text,native)
    except ValueError as exc:
        raise ValueError(f'Native destination requires an explicit absolute {native} path without ambiguous or unsafe components') from exc
    return Path(text)


def _validate(wire: dict[str,Any]) -> tuple[str,InferenceConfig,WriterConfig]:
    if set(wire)!=_FIELDS or type(wire['profile_stages']) is not bool:
        raise ValueError('Archived job requires exact shared job fields and boolean types')
    flavour = infer_flavour(wire['input_path'])
    for field in ('input_path','output_dir'):
        archived_path(wire[field],flavour)
    for field in ('initial_hint_path','hint_video_path'):
        if wire[field] is not None:
            archived_path(wire[field],flavour)
    if not isinstance(wire['config_json'],str) or not isinstance(wire['writer_json'],str):
        raise TypeError('Archived embedded settings require JSON strings')
    settings,writer_settings = _json(wire['config_json']),_json(wire['writer_json'])
    archived_path(settings.get('temp_dir',''),flavour)
    if settings.get('corrections_path') is not None:
        archived_path(settings['corrections_path'],flavour)
    outputs = settings.get('outputs')
    if not isinstance(outputs,list) or any(not isinstance(item,str) for item in outputs) or len(outputs)!=len(set(outputs)):
        raise ValueError('Archived outputs require a list of unique output names')
    settings['outputs'] = set(outputs)
    try:
        config,writer = InferenceConfig(**settings),WriterConfig(**writer_settings)
        config.validate()
        writer.validate()
    except TypeError as exc:
        raise ValueError(f'Invalid archived shared settings: {exc}') from exc
    executable = writer.ffmpeg_bin
    _text(executable)
    if re.match(r'^[A-Za-z]:',executable) or executable.startswith(('/','\\')):
        archived_path(executable,infer_flavour(executable))
    elif re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',executable):
        raise ValueError('Archived executable cannot be a URI')
    return flavour,config,writer


@dataclass(frozen=True)
class ArchivalJob:
    flavour: str
    wire_json: bytes

    def __post_init__(self):
        if not isinstance(self.wire_json,bytes):
            raise TypeError('Archived wire requires immutable bytes')
        inferred,_,_ = _validate(_json(self.wire_json))
        if self.flavour!=inferred:
            raise ValueError('Archived declared/inferred origin flavours differ')

    @property
    def original_wire(self) -> dict[str,Any]:
        return _json(self.wire_json)

    @property
    def config(self) -> InferenceConfig:
        return _validate(self.original_wire)[1]

    @property
    def writer(self) -> WriterConfig:
        return _validate(self.original_wire)[2]

    @property
    def writer_executable_flavour(self) -> str | None:
        return infer_flavour(self.writer.ffmpeg_bin) if self.writer_executable_absolute else None

    @property
    def writer_executable_absolute(self) -> bool:
        value = self.writer.ffmpeg_bin
        return bool(re.match(r'^[A-Za-z]:',value) or value.startswith(('/','\\')))


def archive_job(wire: dict[str,Any] | bytes) -> ArchivalJob:
    if isinstance(wire,bytes):
        raw = wire
    elif isinstance(wire,dict):
        # Reject dataclass instances/classes and sets; never stringify foreign paths.
        raw = json.dumps(wire,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    else:
        raise TypeError('Archived job requires a wire object or immutable JSON bytes')
    data = _json(raw)
    flavour,_,_ = _validate(data)
    return ArchivalJob(flavour,raw)
