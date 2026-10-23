"""Frozen metadata indexes for completed sequence playback, not manual stills.

No pixels are cached or decoded during preparation. Existing reader/native image
memory requirements, read-only symlink/hardlink behavior and native geometry are
preserved. Identities cover ordered paths and sampled target stat metadata, not
content authentication or hostile concurrent filesystem writers. Additions are
ignored until deliberate preparation/Refresh; ordinary changes refuse reads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .desktop_comparison import RenderView, common_range
from .preview_sequence import SequenceIndex, scan_sequence

_MODES = {'Source','Alpha','Foreground','Checker composite','Solid composite','Photo composite'}
_ROW_BYTES = 768
_ROOT_BYTES = 1024
_BASE_BYTES = 8192
_MAX_REQUEST_BYTES = 64*1024
Check = Callable[[],None] | None


def _check(check: Check):
    if check is not None:
        check()


def _integer(value,low,high,name):
    if type(value) is not int or not low<=value<=high:
        raise ValueError(f'{name} requires an exact integer in [{low},{high}]')


def _path(value):
    if (not isinstance(value,str) or not value or len(os.fsencode(value))>32768
            or any(ord(char)<32 for char in value) or not Path(value).is_absolute()
            or '..' in Path(value).parts):
        raise ValueError('Playback index paths require bounded absolute native text without traversal/control characters')


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


@dataclass(frozen=True)
class ExportIndexLimits:
    max_directories: int = 6
    max_entries: int = 100000
    max_frames: int = 100000
    max_path_bytes: int = 32*1024**2
    max_metadata_bytes: int = 64*1024**2

    def __post_init__(self):
        for name,maximum in (('max_directories',6),('max_entries',100000),('max_frames',100000),
                             ('max_path_bytes',32*1024**2),('max_metadata_bytes',64*1024**2)):
            _integer(getattr(self,name),1,maximum,name)


def _view(view):
    if type(view) is not RenderView:
        raise TypeError('Export indexing requires immutable RenderView instances')
    _path(view.source)
    for name in ('start','count','height','width'):
        _integer(getattr(view,name),0 if name=='start' else 1,2**31-1,name)
    if type(view.fps) not in (int,float) or not math.isfinite(view.fps) or view.fps<=0:
        raise ValueError('Recorded playback frame rate must be finite and positive')
    if type(view.linear) is not bool or type(view.source_linear) is not bool:
        raise ValueError('Recorded playback transfer flags require booleans')
    if view.source_alpha not in ('ignore','premultiplied') or view.source_colour_profile not in ('legacy','linear_bt709_d65','linear_bt2020_d65'):
        raise ValueError('Unsupported recorded source interpretation')
    if type(view.outputs) is not tuple or len(view.outputs)>16:
        raise ValueError('Recorded playback outputs require a bounded immutable tuple')
    names = set()
    for pair in view.outputs:
        if type(pair) is not tuple or len(pair)!=2 or not isinstance(pair[0],str) or not 1<=len(pair[0])<=64 or pair[0] in names:
            raise ValueError('Recorded playback output entries require unique immutable name/path pairs')
        names.add(pair[0])
        _path(pair[1])


def _required(view,mode):
    outputs = dict(view.outputs)
    if mode=='Source':
        return ((view.source,True,view.end),)
    if not view.linear:
        raise ValueError('Indexed export playback requires completed EXR sequences; video/VFR is unsupported')
    kinds = ('alpha',) if mode=='Alpha' else ('fg','alpha') if 'composite' in mode else ('fg',)
    if any(kind not in outputs for kind in kinds):
        raise ValueError('The selected playback view requires missing completed exports')
    return tuple((outputs[kind],False,view.count) for kind in kinds)


@dataclass(frozen=True)
class ExportRequest:
    view: RenderView
    mode: str
    reference: RenderView | None = None
    reference_mode: str | None = None
    limits: ExportIndexLimits = field(default_factory=ExportIndexLimits)
    identity: str = field(init=False)

    def __post_init__(self):
        _view(self.view)
        if type(self.limits) is not ExportIndexLimits:
            raise TypeError('Playback index limits require an immutable limits instance')
        if not isinstance(self.mode,str) or self.mode not in _MODES:
            raise ValueError('Unknown indexed playback view mode')
        _required(self.view,self.mode)
        if self.reference is not None:
            _view(self.reference)
            common_range(self.view,self.reference)
        reference_mode = self.reference_mode if self.reference_mode is not None else self.mode
        if not isinstance(reference_mode,str) or reference_mode not in _MODES:
            raise ValueError('Unknown indexed reference playback mode')
        object.__setattr__(self,'reference_mode',reference_mode)
        if self.reference is not None:
            _required(self.reference,reference_mode)
        raw = _canonical({'view':asdict(self.view),'mode':self.mode,
                          'reference':None if self.reference is None else asdict(self.reference),
                          'reference_mode':reference_mode,'limits':asdict(self.limits)})
        if len(raw)>_MAX_REQUEST_BYTES:
            raise ValueError('Captured playback metadata exceeds the small request byte bound')
        object.__setattr__(self,'identity',hashlib.sha256(raw).hexdigest())


def capture_export_request(view: RenderView,*,mode: str,reference: RenderView | None=None,
                           reference_mode: str | None=None,limits: ExportIndexLimits | None=None) -> ExportRequest:
    """Pure capture: no resolve, filesystem metadata, pixels or Qt calls."""
    return ExportRequest(view,mode,reference,reference_mode,limits if limits is not None else ExportIndexLimits())


def _sample(path):
    info = Path(path).stat()  # Preserve the existing read-only alias semantics.
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Indexed export must remain a regular readable image file')
    return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns


def _index_identity(source,paths,samples,root_identity):
    digest = hashlib.sha256(_canonical({'source':source,'root':root_identity,'version':'export-stat-ctime-v1'}))
    for path,sample in zip(paths,samples,strict=True):
        digest.update(b'\0'+os.fsencode(path)+b'\0'+_canonical(sample))
    return digest.hexdigest()


@dataclass(frozen=True)
class ExportSequenceIndex(SequenceIndex):
    change_fingerprints: tuple[tuple[int,int,int,int,int],...]
    root_identity: tuple[int,int]

    def __post_init__(self):
        _path(self.source)
        if (type(self.paths) is not tuple or not self.paths or len(self.paths)>100000
                or type(self.fingerprints) is not tuple or type(self.change_fingerprints) is not tuple
                or len(self.paths)!=len(self.fingerprints) or len(self.paths)!=len(self.change_fingerprints)
                or type(self.root_identity) is not tuple or len(self.root_identity)!=2
                or any(type(value) is not int or value<0 for value in self.root_identity)):
            raise ValueError('Export index requires immutable bounded paths/stat/root metadata')
        names = set()
        for path,old,current in zip(self.paths,self.fingerprints,self.change_fingerprints,strict=True):
            _path(path)
            if Path(path).parent!=Path(self.source) or path in names:
                raise ValueError('Export index contains a duplicate or escaped direct-child path')
            names.add(path)
            if (type(old) is not tuple or type(current) is not tuple or len(old)!=4 or len(current)!=5
                    or any(type(value) is not int for value in current) or old!=current[:4]):
                raise ValueError('Export index stat metadata differs from its sequence snapshot')
        _integer(self.entries_inspected,len(self.paths),100000,'Inspected entry count')
        if type(self.path_bytes) is not int or self.path_bytes!=sum(len(os.fsencode(path)) for path in self.paths):
            raise ValueError('Export index path accounting differs')
        if self.identity!=_index_identity(self.source,self.paths,self.change_fingerprints,self.root_identity):
            raise ValueError('Export index identity differs from its frozen metadata')

    def check_path(self,index: int) -> None:
        if type(index) is not int or not 0<=index<len(self.paths):
            raise ValueError('Frame lies outside the prepared export sequence')
        root = Path(self.source).stat()
        if not stat.S_ISDIR(root.st_mode) or (root.st_dev,root.st_ino)!=self.root_identity:
            raise ValueError('Indexed playback directory changed; Refresh indexes explicitly')
        if _sample(self.paths[index])!=self.change_fingerprints[index]:
            raise ValueError('Indexed playback file changed; Refresh indexes explicitly')


def _requirements(request):
    values = list(_required(request.view,request.mode))
    if request.reference is not None:
        values.extend(_required(request.reference,request.reference_mode))
    return tuple(values)


def _bundle_identity(request,streams,bindings):
    return hashlib.sha256(_canonical({'request':request.identity,'indexes':[stream.identity for stream in streams],'bindings':bindings})).hexdigest()


@dataclass(frozen=True)
class PreparedExportIndexes:
    request: ExportRequest
    streams: tuple[ExportSequenceIndex,...]
    bindings: tuple[tuple[str,int],...]
    identity: str
    metadata_bytes: int

    def __post_init__(self):
        if type(self.request) is not ExportRequest or type(self.streams) is not tuple or type(self.bindings) is not tuple:
            raise TypeError('Prepared export indexes require immutable request/stream/binding instances')
        limits = self.request.limits
        if not 1<=len(self.streams)<=limits.max_directories or any(type(stream) is not ExportSequenceIndex for stream in self.streams):
            raise ValueError('Prepared export stream count/types differ from declared bounds')
        expected = {path for path,_,_ in _requirements(self.request)}
        seen = set()
        for pair in self.bindings:
            if type(pair) is not tuple or len(pair)!=2:
                raise ValueError('Prepared export bindings require immutable path/index pairs')
            path,index = pair
            _path(path)
            if path in seen or type(index) is not int or not 0<=index<len(self.streams):
                raise ValueError('Prepared export binding is duplicate or outside its stream table')
            seen.add(path)
        if seen!=expected or {index for _,index in self.bindings}!=set(range(len(self.streams))):
            raise ValueError('Prepared export bindings omit or add captured playback roots')
        entries = sum(stream.entries_inspected for stream in self.streams)
        frames = sum(len(stream.paths) for stream in self.streams)
        paths = sum(stream.path_bytes for stream in self.streams)
        metadata = _BASE_BYTES+len(self.streams)*_ROOT_BYTES+frames*_ROW_BYTES+paths*4
        if entries>limits.max_entries or frames>limits.max_frames or paths>limits.max_path_bytes or metadata>limits.max_metadata_bytes or self.metadata_bytes!=metadata:
            raise ValueError('Prepared export aggregate metadata budget/accounting differs')
        for path,is_source,count in _requirements(self.request):
            stream = self.streams[dict(self.bindings)[path]]
            if (is_source and len(stream.paths)<count) or (not is_source and len(stream.paths)!=count):
                raise ValueError('Frozen source-global/export-local frame coverage differs from the completed view')
        if self.identity!=_bundle_identity(self.request,self.streams,self.bindings):
            raise ValueError('Prepared export bundle identity differs from its captured metadata')

    def options_for(self,source_frame: int,*,background=None,background_image=None) -> dict[str,Any]:
        table = dict(self.bindings)
        def options(view,mode):
            result = view.options(mode,source_frame,background,background_image=background_image)
            result['sequence'] = self.streams[table[result['path']]]
            if 'alpha_path' in result:
                result['alpha_sequence'] = self.streams[table[result['alpha_path']]]
            return result
        current = options(self.request.view,self.request.mode)
        if self.request.reference is None:
            return current
        return {'pair':{'a':options(self.request.reference,self.request.reference_mode),'b':current}}


def prepare_export_indexes(request: ExportRequest,*,check: Check=None) -> PreparedExportIndexes:
    """Index each unique resolved directory once, with aggregate admission limits."""
    if type(request) is not ExportRequest:
        raise TypeError('Export preparation requires a captured ExportRequest instance')
    _check(check)
    limits = request.limits
    streams = []
    bindings = []
    roots = {}
    entries = frames = paths = 0
    metadata = _BASE_BYTES
    for original,_,_ in _requirements(request):
        if any(path==original for path,_ in bindings):
            continue
        _check(check)
        root = Path(original).resolve()
        key = os.path.normcase(str(root))
        if key not in roots:
            if len(streams)>=limits.max_directories:
                raise ValueError('Playback aggregate directory budget exceeded')
            remaining = limits.max_metadata_bytes-metadata-_ROOT_BYTES
            # Split remaining admission between fixed row objects and path text;
            # the exact combined accounting is checked before retaining a stream.
            maximum_frames = min(limits.max_frames-frames,(remaining//2)//_ROW_BYTES)
            maximum_paths = min(limits.max_path_bytes-paths,(remaining//2)//4)
            maximum_entries = limits.max_entries-entries
            if min(maximum_frames,maximum_paths,maximum_entries)<1:
                raise ValueError('Playback aggregate index metadata/entry/frame/path budget exhausted')
            index = scan_sequence(root,check=check,max_entries=maximum_entries,max_frames=maximum_frames,max_path_bytes=maximum_paths)
            samples = []
            for path,old in zip(index.paths,index.fingerprints,strict=True):
                _check(check)
                sample = _sample(path)
                if sample[:4]!=old:
                    raise ValueError('Playback file changed while preparing indexes')
                samples.append(sample)
            info = root.stat()
            root_identity = info.st_dev,info.st_ino
            fingerprint = tuple(samples)
            indexed = ExportSequenceIndex(index.source,index.paths,index.fingerprints,
                                          _index_identity(index.source,index.paths,fingerprint,root_identity),
                                          index.entries_inspected,index.path_bytes,fingerprint,root_identity)
            roots[key] = len(streams)
            streams.append(indexed)
            entries += index.entries_inspected
            frames += len(index.paths)
            paths += index.path_bytes
            metadata += _ROOT_BYTES+len(index.paths)*_ROW_BYTES+index.path_bytes*4
        bindings.append((original,roots[key]))
    _check(check)
    immutable_streams,immutable_bindings = tuple(streams),tuple(bindings)
    return PreparedExportIndexes(request,immutable_streams,immutable_bindings,
                                 _bundle_identity(request,immutable_streams,immutable_bindings),metadata)
