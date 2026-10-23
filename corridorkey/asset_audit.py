"""Nonexecuting local asset structure and streamed identity audit; never a loader."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import logging
import math
import os
import re
import stat
import struct
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .asset_inventory import AssetSelection
from .control import CancelledError, ProgressEvent
from .runtime_paths import RuntimePaths

MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_TREE_FILES = 4096
MAX_TENSORS = 65536
MAX_NPY_HEADER = 65536
MAX_ZIP_EXPANDED = 16 * 1024**3
STATS = ('moge_mean', 'moge_std', 'rvm_mean', 'rvm_std', 'cradio_mean', 'cradio_std')
DTYPES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
          'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8, 'F8_E4M3': 1, 'F8_E5M2': 1}
SCOPE = ('Read-only local structure and SHA256 identity, not trusted publisher verification, tensor-value validation, '
         'model compatibility, inference, GPU/backend support or quality. No tensors, pickle, model code or downloads executed. '
         'Symlink aliases and resolved targets are recorded; normal filesystem reads may update access times.')


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _check(check) -> None:
    if check is not None:
        check()


def _progress(progress, stage: str, completed: int, total: int | None) -> None:
    if progress is not None:
        progress(ProgressEvent(stage, completed, total))


def _json(raw: bytes | str) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value

    def constant(value):
        raise ValueError(f'Nonfinite JSON value: {value}')

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise TypeError('Asset JSON header must be an object')
    pending: list[tuple[Any, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 128:
            raise ValueError('Asset JSON exceeds depth128')
        if isinstance(item, dict):
            pending.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            pending.extend((nested, depth + 1) for nested in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Nonfinite JSON number')
    return value


def _stat(value: os.stat_result) -> dict[str, int]:
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns,
            'device': value.st_dev, 'inode': value.st_ino, 'mode': value.st_mode}


def _identity(path: Path, *, directory: bool = False) -> dict[str, Any]:
    lexical = path.absolute()
    resolved = lexical.resolve(strict=True)
    if len(str(lexical)) > 4096 or len(str(resolved)) > 4096:
        raise ValueError('Asset path exceeds4096characters')
    target = resolved.stat()
    valid = stat.S_ISDIR(target.st_mode) if directory else stat.S_ISREG(target.st_mode)
    if not valid:
        raise ValueError('Asset must resolve to a regular file/directory')
    return {'path': str(lexical), 'resolved_path': str(resolved), 'lexical_stat': _stat(lexical.lstat()), 'target_stat': _stat(target)}


def _hash(path: Path, identity: dict[str, Any], check, progress) -> str:
    size = identity['target_stat']['bytes']
    digest = hashlib.sha256()
    completed = 0
    with path.open('rb') as stream:
        while completed < size:
            _check(check)
            block = stream.read(min(size - completed, 1024 * 1024))
            if not block:
                raise ValueError('Asset truncated during hashing')
            digest.update(block)
            completed += len(block)
            _progress(progress, 'asset_hash', completed, size)
        if stream.read(1):
            raise ValueError('Asset grew beyond its admitted byte size')
    if _identity(path) != identity:
        raise ValueError('Asset alias or contents changed during hashing')
    return digest.hexdigest()


def _safetensors(path: Path, size: int, *, role: str, check) -> tuple[str, dict[str, Any]]:
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= MAX_HEADER_BYTES or length + 8 > size:
            raise ValueError('Safetensors header exceeds bounds or file extent')
        header = _json(stream.read(length))
    if len(header) > MAX_TENSORS + 1:
        raise ValueError('Safetensors tensor count exceeds65536')
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError('Safetensors metadata must map strings to strings')
    intervals = []
    tensors = {}
    for name, item in header.items():
        _check(check)
        if (not name or len(name) > 1024 or not isinstance(item, dict) or set(item) != {'dtype', 'shape', 'data_offsets'} or
                item.get('dtype') not in DTYPES or not isinstance(item.get('shape'), list) or len(item['shape']) > 32 or
                any(type(value) is not int or not 0 <= value < 2**63 for value in item['shape']) or
                not isinstance(item.get('data_offsets'), list) or len(item['data_offsets']) != 2 or
                any(type(value) is not int for value in item['data_offsets'])):
            raise ValueError('Invalid safetensors tensor descriptor')
        begin, end = item['data_offsets']
        extent = math.prod(item['shape']) * DTYPES[item['dtype']]
        if not 0 <= begin <= end <= size - 8 - length or end - begin != extent:
            raise ValueError('Safetensors dtype/shape/offset extent mismatch')
        intervals.append((begin, end))
        tensors[name] = item
    position = 0
    for begin, end in sorted(intervals):
        if begin != position:
            raise ValueError('Safetensors payload has gaps or overlapping tensors')
        position = end
    if position != size - 8 - length:
        raise ValueError('Safetensors payload extent differs from header')
    details: dict[str, Any] = {'tensor_count': len(tensors), 'tensor_names_sample': sorted(tensors)[:32],
                               'tensor_values': 'NOT VERIFIED'}
    status = 'PASS'
    if role == 'weights':
        if metadata.get('format') != 'corridorkey_v2_runtime_checkpoint':
            raise ValueError('Runtime checkpoint format metadata mismatch')
        config = _json(metadata.get('model_config', '{}'))
        _json(metadata.get('metadata', '{}'))
        details['runtime_format'] = metadata['format']
        details['model_config_keys'] = sorted(config)[:128]
        details['model_config_defaulted'] = 'model_config' not in metadata
        if 'model_config' not in metadata:
            status = 'UNVERIFIED'
    if role == 'foundation_stats':
        details['vectors'] = {name: {'shape': tensors[name]['shape'], 'dtype': tensors[name]['dtype']}
                              for name in STATS if name in tensors}
        details['missing_vectors'] = [name for name in STATS if name not in tensors]
        details['normalization_values'] = 'NOT VERIFIED; no tensor values materialized'
        if details['missing_vectors'] or any(len(tensors[name]['shape']) != 1 for name in STATS if name in tensors):
            status = 'UNVERIFIED'
        for stem in ('moge', 'rvm', 'cradio'):
            if (stem + '_mean' in tensors and stem + '_std' in tensors and
                    tensors[stem + '_mean']['shape'] != tensors[stem + '_std']['shape']):
                status = 'UNVERIFIED'
        details['loader_contract'] = 'Six named vectors; runtime skips missing/mismatched buffers. Model-dependent channels NOT VERIFIED.'
    return status, details


def _npy(stream, member_size: int) -> dict[str, Any]:
    prefix = stream.read(8)
    if len(prefix) != 8 or prefix[:6] != b'\x93NUMPY' or prefix[6:8] not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
        raise ValueError('Invalid NPY header/version')
    count = 2 if prefix[6] == 1 else 4
    length_bytes = stream.read(count)
    if len(length_bytes) != count:
        raise ValueError('Truncated NPY header length')
    length = int.from_bytes(length_bytes, 'little')
    if not 1 <= length <= MAX_NPY_HEADER:
        raise ValueError('NPY header exceeds64KiB')
    raw = stream.read(length)
    if len(raw) != length:
        raise ValueError('Truncated NPY header')
    value = ast.literal_eval(raw.decode('utf-8' if prefix[6] == 3 else 'latin1').strip())
    if (not isinstance(value, dict) or set(value) != {'descr', 'fortran_order', 'shape'} or
            type(value.get('fortran_order')) is not bool or not isinstance(value.get('shape'), tuple) or len(value['shape']) > 32 or
            any(type(dimension) is not int or not 0 <= dimension < 2**63 for dimension in value['shape']) or
            not isinstance(value.get('descr'), str) or not re.fullmatch(r'[<>=|][biufc][124816]{1,2}', value['descr'])):
        raise ValueError('Unsupported/object/structured NPY dtype or shape')
    itemsize = int(value['descr'][2:])
    allowed = {'b': (1,), 'i': (1, 2, 4, 8), 'u': (1, 2, 4, 8), 'f': (2, 4, 8, 16), 'c': (8, 16)}
    if itemsize not in allowed[value['descr'][1]] or 8 + count + length + math.prod(value['shape']) * itemsize != member_size:
        raise ValueError('NPY shape/dtype extent mismatch')
    return {'dtype': value['descr'], 'shape': list(value['shape'])}


class _BoundedAssetFile(io.BufferedReader):
    """Cap parser allocation/read range even if archive metadata changes in place."""
    def __init__(self, raw, size: int, check):
        super().__init__(raw)
        self.limit = size
        self.check = check

    def read(self, size: int | None = -1) -> bytes:
        _check(self.check)
        remaining = self.limit - self.tell()
        requested = remaining if size is None or size < 0 else size
        if requested > MAX_HEADER_BYTES:
            raise ValueError('NPZ parser read exceeds bounded header/directory allocation')
        return super().read(min(requested, remaining))

    def seek(self, offset: int, whence: int = 0) -> int:
        _check(self.check)
        position = offset if whence == 0 else (self.tell() + offset if whence == 1 else self.limit + offset)
        if whence not in (0, 1, 2) or not 0 <= position <= self.limit:
            raise ValueError('NPZ parser seek exceeds admitted file range')
        return super().seek(position)


def _npz(path: Path, size: int, check) -> tuple[str, dict[str, Any]]:
    with path.open('rb', buffering=0) as raw:
        if os.fstat(raw.fileno()).st_size != size:
            raise ValueError('NPZ file size changed before parser admission')
        with _BoundedAssetFile(raw, size, check) as stream:
            return _npz_stream(stream, size, check)


def _npz_stream(stream: _BoundedAssetFile, size: int, check) -> tuple[str, dict[str, Any]]:
    stream.seek(max(0, size - 65557))
    tail = stream.read(min(size, 65557))
    index = tail.rfind(b'PK\x05\x06')
    if index < 0 or len(tail) - index < 22:
        raise ValueError('Missing ZIP end-of-directory record')
    absolute_end = max(0, size - 65557) + index
    if absolute_end >= 20:
        # A maximum-length comment places the locator outside our EOCD tail.
        # ZipFile consults it independently and may override32-bit bounds.
        stream.seek(absolute_end - 20)
        locator = stream.read(20)
        if locator[:4] == b'PK\x06\x07':
            raise ValueError('ZIP64 central directories are outside this structural audit; local ZIP64 headers are supported')
    end = struct.unpack('<4s4H2IH', tail[index:index + 22])
    _, disk, central_disk, disk_count, total, central_size, central_offset, comment = end
    if (disk or central_disk or disk_count != total or not 1 <= total <= MAX_TREE_FILES or
            central_size > MAX_HEADER_BYTES or central_offset + central_size > size or index + 22 + comment != len(tail)):
        raise ValueError('Unsupported ZIP64/multidisk or excessive NPZ directory')
    names = set()
    intervals = []
    expanded = 0
    sample = []
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        if len(infos) != total:
            raise ValueError('ZIP directory count mismatch')
        for item in infos:
            _check(check)
            name = item.filename
            parts = name.split('/')
            if (len(name) > 1024 or '\\' in name or ':' in name or any(part in ('', '.', '..') for part in parts) or
                    PurePosixPath(name).is_absolute() or not name.endswith('.npy') or name[:-4] in names or
                    item.flag_bits & 1 or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) or
                    stat.S_ISLNK(item.external_attr >> 16) or item.file_size > MAX_ZIP_EXPANDED or
                    item.file_size > max(1, item.compress_size) * 1000):
                raise ValueError('Unsafe/duplicate/encrypted/oversized NPZ member')
            logical = name[:-4]
            if any(logical == other or logical.startswith(other + '/') or other.startswith(logical + '/') for other in names):
                raise ValueError('Conflicting NPZ logical tensor paths')
            names.add(logical)
            expanded += item.file_size
            if expanded > MAX_ZIP_EXPANDED:
                raise ValueError('NPZ expanded size exceeds16GiB')
            stream.seek(item.header_offset)
            local = stream.read(30)
            if len(local) != 30 or local[:4] != b'PK\x03\x04':
                raise ValueError('Invalid ZIP local header')
            name_size, extra_size = struct.unpack('<HH', local[26:30])
            finish = item.header_offset + 30 + name_size + extra_size + item.compress_size
            if item.header_offset < 0 or finish > central_offset:
                raise ValueError('ZIP member overlaps central directory')
            intervals.append((item.header_offset, finish))
            with archive.open(item) as member:
                descriptor = _npy(member, item.file_size)
            if len(sample) < 16:
                sample.append({'name': logical, **descriptor})
    finish = 0
    for begin, end in sorted(intervals):
        if begin < finish:
            raise ValueError('Overlapping NPZ compressed members')
        finish = end
    return 'PASS', {'tensor_count': len(names), 'uncompressed_bytes': expanded, 'tensor_sample': sample,
                    'rvm_mapping': 'NOT VERIFIED; slash-separated logical names supported, no arrays loaded',
                    'payload_crc_and_values': 'NOT VERIFIED; only compressed NPY headers inspected'}


def _structure(path: Path, size: int, role: str, check) -> tuple[str, str, dict[str, Any]]:
    if path.suffix == '.safetensors':
        status, details = _safetensors(path, size, role=role, check=check)
        return status, 'safetensors', details
    if path.suffix == '.npz':
        status, details = _npz(path, size, check)
        return status, 'npz/npy', details
    if path.suffix in {'.pt', '.pth', '.bin'}:
        return 'UNVERIFIED', 'legacy checkpoint', {'reason': 'Opaque legacy checkpoint retained; no torch.load/pickle execution performed.'}
    return 'UNVERIFIED', 'unrecognized', {'reason': 'Hashed only; no supported nonexecuting structural parser.'}


def _file(path: Path, role: str, check, progress) -> dict[str, Any]:
    identity = _identity(path)
    size = identity['target_stat']['bytes']
    digest = _hash(path, identity, check, progress)
    try:
        if size == 0:
            raise ValueError('Asset file is empty')
        status, format_name, details = _structure(path, size, role, check)
    except (OSError, ValueError, TypeError, RecursionError, zipfile.BadZipFile, struct.error, SyntaxError) as exc:
        status, format_name, details = 'FAIL', path.suffix, {'error': f'{type(exc).__name__}: {exc}'[:2048]}
    if _identity(path) != identity:
        raise ValueError('Asset changed during structural inspection')
    return {'path': str(path.absolute()), 'resolved_path': identity['resolved_path'], 'present': True, 'bytes': size,
            'format_status': status, 'format': format_name, 'sha256': digest, 'identity': identity, 'details': details}


def _tree(path: Path, check, progress, *, observed: list | None = None) -> dict[str, Any]:
    identity = _identity(path, directory=True)
    pending = [(path, frozenset())]
    files = []
    directories = []
    bytes_total = 0
    metadata_bytes = 0
    while pending:
        directory, ancestors = pending.pop()
        _check(check)
        info = _identity(directory, directory=True)
        resolved = info['resolved_path']
        if resolved in ancestors:
            raise ValueError('C-RADIO symlink directory cycle')
        directories.append(info)
        if len(directories) > MAX_TREE_FILES:
            raise ValueError('C-RADIO directory count exceeds4096')
        with os.scandir(directory) as iterator:
            names = []
            for item in iterator:
                _check(check)
                names.append(item.name)
                if len(names) + len(files) > MAX_TREE_FILES:
                    raise ValueError('C-RADIO tree exceeds4096files')
        for name in sorted(names):
            child = directory / name
            if child.is_dir():
                if len(directories) + len(pending) + 1 > MAX_TREE_FILES:
                    raise ValueError('C-RADIO discovered directory frontier exceeds4096')
                pending.append((child, ancestors | {resolved}))
                continue
            file_identity = _identity(child)
            digest = _hash(child, file_identity, check, progress)
            record = {'name': child.relative_to(path).as_posix(), 'identity': file_identity, 'sha256': digest}
            metadata_bytes += len(_canonical(record))
            if metadata_bytes > 16 * 1024 * 1024:
                raise ValueError('C-RADIO identity inventory exceeds16MiB')
            files.append(record)
            if len(files) > MAX_TREE_FILES:
                raise ValueError('C-RADIO tree exceeds4096files')
            bytes_total += file_identity['target_stat']['bytes']
    if not files:
        raise ValueError('C-RADIO directory is empty')
    indexed = {record['name']: record for record in files}
    try:
        status, details = _tree_structure(path, indexed, check)
    except (OSError, ValueError, TypeError, RecursionError, struct.error) as error:
        # Hashes already cover every readable file. A malformed config/header
        # remains an exportable diagnostic, not an unreadable source identity.
        status, details = 'FAIL', dict[str, Any](error=f'{type(error).__name__}: {error}'[:2048])
    details.update(file_count=len(files), local_code='NOT EXECUTED', class_compatibility='NOT VERIFIED')
    for record in files:
        if _identity(path / record['name']) != record['identity']:
            raise ValueError('C-RADIO file/alias changed during audit')
    for record in directories:
        if _identity(Path(record['path']), directory=True) != record:
            raise ValueError('C-RADIO directory/alias changed during audit')
    if observed is not None:
        observed.extend((Path(record['identity']['path']), False, record['identity']) for record in files)
        observed.extend((Path(record['path']), True, record) for record in directories)
    return {'path': str(path.absolute()), 'resolved_path': identity['resolved_path'], 'present': True, 'bytes': bytes_total,
            'format_status': status, 'format': 'C-RADIO local tree', 'sha256': _digest(sorted(files, key=lambda item: item['name'])),
            'identity': {'root': identity, 'tree_sha256': _digest({'files': files, 'directories': directories})}, 'details': details}


def _tree_structure(path: Path, indexed: dict[str, Any], check) -> tuple[str, dict[str, Any]]:
    status = 'PASS'
    details: dict[str, Any] = {}
    if 'hf_model.py' not in indexed or 'config.json' not in indexed:
        status = 'FAIL'
        details['reason'] = 'Actual local loader requires hf_model.py and config.json'
    else:
        config_size = indexed['config.json']['identity']['target_stat']['bytes']
        if config_size > MAX_HEADER_BYTES:
            raise ValueError('C-RADIO config exceeds8MiB')
        with (path / 'config.json').open('rb') as stream:
            config = _json(stream.read(config_size + 1))
        details['config_keys'] = sorted(config)[:128]
    safe_files = [name for name in indexed if name.endswith('.safetensors')]
    if not safe_files:
        status = 'UNVERIFIED' if status == 'PASS' else status
        details['weights'] = 'No safetensors found; legacy/alternate loader formats NOT VERIFIED'
    for name in safe_files:
        _safetensors(path / name, indexed[name]['identity']['target_stat']['bytes'], role='cradio', check=check)
    index_name = 'model.safetensors.index.json'
    if index_name in indexed:
        length = indexed[index_name]['identity']['target_stat']['bytes']
        if length > MAX_HEADER_BYTES:
            raise ValueError('C-RADIO shard index exceeds8MiB')
        with (path / index_name).open('rb') as stream:
            index = _json(stream.read(length + 1))
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map or len(weight_map) > MAX_TENSORS:
            raise ValueError('Invalid C-RADIO shard weight map')
        for name in weight_map.values():
            if not isinstance(name, str) or '/' in name or '\\' in name or name not in safe_files:
                raise ValueError('Missing/unsafe C-RADIO weight shard')
        details['shard_count'] = len(set(weight_map.values()))
    return status, details


def _paths(selection: AssetSelection) -> dict[str, Path]:
    root = Path(selection.root).absolute()
    weights = selection.weights
    if weights is None:
        checkpoint = root / RuntimePaths._prefer_safetensors(root, 'weights/corridorkey_v2')
    else:
        checkpoint = Path(weights)
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint) if (Path.cwd() / checkpoint).exists() else root / checkpoint
    return {'weights': checkpoint.absolute(), 'foundation_stats': root / RuntimePaths._prefer_safetensors(root, 'weights/foundation_stats'),
            'rvm_ckpt': root / 'vfms/rvm/pretrain_rvm_small16_256_204031069.npz',
            'moge_checkpoint': root / 'vfms/moge/model.pt', 'cradio_dir': root / 'vfms/cradio/C-RADIOv4-SO400M'}


def _snapshot_check(selection: AssetSelection, paths: dict[str, Path], observed: list, missing: list[Path], check) -> None:
    for path, directory, identity in observed:
        _check(check)
        if _identity(path, directory=directory) != identity:
            raise ValueError('Selected asset snapshot changed before publication')
    if any(path.exists() for path in missing) or _paths(selection) != paths:
        raise ValueError('Missing assets appeared or runtime asset preference changed')


def _audit_assets(selection: AssetSelection, *, check=None, progress=None, observed_sink: list | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {'schema_version': 1, 'kind': 'corridorkey_asset_audit', 'status': 'complete', 'assets_present': False,
                              'entries': [], 'model_compatibility': 'NOT VERIFIED', 'publisher_verification': 'NOT VERIFIED',
                              'selection': {'root': str(Path(selection.root).absolute()),
                                            'weights': str(selection.weights) if selection.weights is not None else None},
                              'error': None, 'scope': SCOPE}
    try:
        paths = _paths(selection)
        observed = [] if observed_sink is None else observed_sink
        for index, (name, path) in enumerate(paths.items()):
            _check(check)
            _progress(progress, 'asset_structure', index, len(paths))
            try:
                entry = _tree(path, check, progress, observed=observed) if name == 'cradio_dir' else _file(path, name, check, progress)
                if name != 'cradio_dir':
                    observed.append((path, False, entry['identity']))
            except (OSError, ValueError, TypeError, RecursionError, zipfile.BadZipFile, struct.error, SyntaxError) as exc:
                entry = {'path': str(path), 'resolved_path': None, 'present': path.exists(), 'bytes': None,
                         'format_status': 'FAIL', 'format': 'unverified', 'sha256': None, 'identity': None,
                         'details': {'error': f'{type(exc).__name__}: {exc}'[:2048]}}
            report['entries'].append({'asset': name, **entry})
        report['assets_present'] = all(entry['present'] for entry in report['entries'])
        _check(check)
        _progress(progress, 'asset_structure', len(paths), len(paths))
        _snapshot_check(selection, paths, observed, [Path(entry['path']) for entry in report['entries'] if not entry['present']], check)
    except BaseException as exc:
        logging.getLogger(__name__).exception('Asset audit stopped')
        report.update(status='cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed',
                      error=f'{type(exc).__name__}: {exc}'[:2048])
    report['token'] = _digest(report)
    if len(_canonical(report)) > MAX_REPORT_BYTES:
        raise ValueError('Asset audit report exceeds4MiB')
    return report


def audit_assets(selection: AssetSelection, *, check=None, progress=None) -> dict[str, Any]:
    return _audit_assets(selection, check=check, progress=progress)


def export_audit(report: dict[str, Any], output_path: str | Path, *, check=None) -> dict[str, Any]:
    result: dict[str, Any] = {'status': 'failed', 'destination': None, 'usable': False, 'bytes': 0, 'error': None,
                              'warning': None, 'partial_path': None}
    stage = None
    raw = None
    linked = False
    output = Path(output_path).absolute()
    try:
        raw = _canonical(report)
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError('Asset audit report exceeds4MiB')
        frozen = _json(raw)
        token = frozen.pop('token', None)
        if frozen.get('status') != 'complete' or _digest(frozen) != token:
            raise ValueError('Choose an intact completed audit report')
        if not output.parent.is_dir() or output.exists() or output.is_symlink():
            raise ValueError('Choose a new report file in an existing folder')
        selection = frozen['selection']
        selected = AssetSelection(Path(selection['root']), Path(selection['weights']) if selection['weights'] is not None else None)
        observed = []
        selected_paths = _paths(selected)
        current = _audit_assets(selected, check=check, observed_sink=observed)
        if current['status'] == 'cancelled':
            raise CancelledError(current['error'])
        if current['token'] != token:
            raise ValueError('Asset audit is stale; audit the current selection again')
        if any(entry['present'] and entry['identity'] is None for entry in current['entries']):
            raise ValueError('Present unreadable assets lack a complete identity; resolve audit errors before exporting')
        for entry in current['entries']:
            if entry['asset'] == 'cradio_dir' and entry['present'] and entry['resolved_path'] is not None:
                protected = Path(entry['resolved_path'])
                if output.parent.resolve() == protected or output.parent.resolve().is_relative_to(protected):
                    raise ValueError('Report destination must be outside the audited C-RADIO tree')
        if not output.parent.is_dir() or output.exists() or output.is_symlink():
            raise ValueError('Choose a new report file in an existing folder')
        _check(check)
        descriptor, name = tempfile.mkstemp(prefix='.asset-audit-', dir=output.parent)
        stage = Path(name)
        result['partial_path'] = str(stage)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _snapshot_check(selected, selected_paths, observed,
                        [Path(entry['path']) for entry in current['entries'] if not entry['present']], check)
        _check(check)
        os.link(stage, output)
        linked = True
        result.update(status='complete', destination=str(output), usable=True, bytes=len(raw))
        stage.unlink()
        result['partial_path'] = None
        if os.name != 'nt':
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException as exc:
        logging.getLogger(__name__).exception('Asset audit export stopped')
        committed = False
        if raw is not None and stage is not None:
            try:
                own = linked or (output.lstat().st_ino == stage.lstat().st_ino and output.lstat().st_dev == stage.lstat().st_dev)
                if own and not output.is_symlink():
                    with output.open('rb') as stream:
                        committed = stream.read(len(raw) + 1) == raw
            except OSError:
                pass
        if committed:
            result.update(status='complete', destination=str(output), usable=True, bytes=len(raw or b''), warning=str(exc)[:2048])
        else:
            result.update(status='cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed',
                          error=f'{type(exc).__name__}: {exc}'[:2048])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit')
    audit.add_argument('--asset-root', required=True, type=Path)
    audit.add_argument('--weights', type=Path)
    audit.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = audit_assets(AssetSelection(args.asset_root, args.weights))
    if args.output is not None:
        exported = export_audit(report, args.output)
        if exported['status'] != 'complete':
            print(json.dumps(exported, sort_keys=True, allow_nan=False))
            return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
