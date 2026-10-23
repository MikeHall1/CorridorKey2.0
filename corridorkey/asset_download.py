"""Explicit anonymous publisher metadata and verified fresh local asset downloads.

No authentication, global cache, extraction, retry, resume or model execution.
Production requests use HTTPS; tests inject an owned local transport explicitly.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import logging
import os
import re
import ssl
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import cache_backup as publication
from . import cache_maintenance as maintenance
from .asset_audit import _json
from .control import CancelledError, ProgressEvent

MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_PLAN_BYTES = 4 * 1024 * 1024
MAX_FILES = 4096
MAX_DOWNLOAD_BYTES = 1024**4
MAX_DIRECTORIES = 16384
RVM_NAME = 'pretrain_rvm_small16_256_204031069.npz'
RVM_URL = 'https://storage.googleapis.com/representations4d/checkpoints/' + RVM_NAME
RVM_MD5 = base64.b64decode('ZJb6kj8+tasiJbVeplXu7Q==').hex()
SCOPE = ('Publisher metadata/checksum verified byte transfer into a fresh asset root. RVM MD5 is a legacy checksum; '
         'observed SHA256 is local identity, not a trusted publisher SHA256. No model code, tensors, compatibility, GPU or quality verified. '
         'No assets selected/activated automatically. No authentication, retry, resume, extraction or global cache.')


def catalog() -> dict[str, Any]:
    """Repository-owned provenance only; no network or invented file sizes/hashes."""
    return {'schema_version': 1, 'kind': 'corridorkey_asset_catalog', 'publishers': [
        {'asset': 'corridorkey', 'publisher': 'Corridor Digital', 'repository': 'nikopueringer/CorridorKey2.0',
         'requested_revision': 'HEAD', 'licence': 'Corridor Key Licence', 'licence_url': 'https://huggingface.co/nikopueringer/CorridorKey2.0',
         'source': 'https://huggingface.co/nikopueringer/CorridorKey2.0', 'bytes': None},
        {'asset': 'cradio', 'publisher': 'NVIDIA', 'repository': 'nvidia/C-RADIOv4-SO400M',
         'requested_revision': 'c0457f5dc26ca145f954cd4fc5bb6114e5705ad8', 'licence': 'NVIDIA Open Model License',
         'licence_url': 'https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf',
         'source': 'https://huggingface.co/nvidia/C-RADIOv4-SO400M', 'bytes': None},
        {'asset': 'moge', 'publisher': 'Microsoft', 'repository': 'Ruicheng/moge-2-vitb-normal',
         'requested_revision': 'ca5f0e07ff01d3e5a364c1d954ed12ee1814b368', 'licence': 'MIT (code and weights)',
         'licence_url': 'https://github.com/microsoft/MoGe/blob/main/LICENSE',
         'source': 'https://huggingface.co/Ruicheng/moge-2-vitb-normal', 'bytes': None},
        {'asset': 'rvm', 'publisher': 'Google DeepMind', 'repository': None, 'requested_revision': None,
         'licence': 'Creative Commons Attribution4.0 (weights); Apache2.0 (code)',
         'licence_url': 'https://github.com/google-deepmind/representations4d', 'source': RVM_URL, 'bytes': None}],
        'scope': 'Offline catalog from scripts/download_weights.py and THIRD_PARTY_NOTICES.md. Sizes unknown until explicit metadata resolution.'}


def _check(check, deadline: float | None = None) -> None:
    if check is not None:
        check()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Asset transfer exceeded its overall deadline')


def _progress(progress, stage: str, completed: int, total: int | None) -> None:
    if progress is not None:
        progress(ProgressEvent(stage, completed, total))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _DeadlineRaw(io.RawIOBase):
    def __init__(self, source, sock, deadline, check, timeout):
        super().__init__()
        self.source, self.sock = source, sock
        self.deadline, self.check, self.timeout = deadline, check, timeout
        self.header_mode = True
        self.header_bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        _check(self.check, self.deadline)
        self.sock.settimeout(min(self.timeout, max(0.001, self.deadline - time.monotonic())))
        block = self.source.read1(len(buffer))
        _check(self.check, self.deadline)
        if self.header_mode:
            self.header_bytes += len(block)
            if self.header_bytes > 65536:
                raise ValueError('Publisher response headers exceed64KiB')
        buffer[:len(block)] = block
        return len(block)

    def close(self) -> None:
        try:
            self.source.close()
        finally:
            super().close()


def _response_class(deadline, check, timeout) -> type[http.client.HTTPResponse]:
    class DeadlineResponse(http.client.HTTPResponse):
        def __init__(self, sock, *args, **kwargs):
            super().__init__(sock, *args, **kwargs)
            self.checked = _DeadlineRaw(self.fp, sock, deadline, check, timeout)
            self.fp = io.BufferedReader(self.checked)

        def begin(self) -> None:
            try:
                super().begin()
            finally:
                self.checked.header_mode = False
    return DeadlineResponse


class _DeadlineHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, response_class: type[http.client.HTTPResponse]):
        super().__init__()
        self.response_type = response_class

    def http_open(self, req):
        def connection(host, **kwargs):
            client = http.client.HTTPConnection(host, **kwargs)
            client.response_class = self.response_type
            return client
        return self.do_open(connection, req)


class _DeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, response_class: type[http.client.HTTPResponse], context: ssl.SSLContext):
        super().__init__(context=context)
        self.context = context
        self.response_type = response_class

    def https_open(self, req):
        def connection(host, **kwargs):
            client = http.client.HTTPSConnection(host, **kwargs)
            client.response_class = self.response_type
            return client
        return self.do_open(connection, req, context=self.context)


class _PublicResponse:
    def __init__(self, response):
        self.response = response
        self.headers, self.status = response.headers, response.status

    def read1(self, size: int) -> bytes:
        try:
            return self.response.read1(size)
        except (http.client.HTTPException, OSError, urllib.error.URLError):
            raise RuntimeError('Publisher stream failed or timed out; no retry performed') from None


class HttpTransport:
    """Owned anonymous stdlib transport; no credential/proxy/global HF state."""
    def __init__(self, *, socket_timeout: float = 10, metadata_seconds: float = 60, file_seconds: float = 3600):
        for value in (socket_timeout, metadata_seconds, file_seconds):
            if not isinstance(value, (int, float)) or not 0 < value < 86401:
                raise ValueError('Transport deadlines must be positive finite seconds up to86400')
        self.socket_timeout = socket_timeout
        self.metadata_seconds = metadata_seconds
        self.file_seconds = file_seconds
        self.context = ssl.create_default_context()

    def validate_url(self, url: str) -> None:
        if not isinstance(url, str) or len(url) > 16384 or any(ord(character) <= 32 or ord(character) == 127 for character in url):
            raise ValueError('Asset transport requires a bounded anonymous HTTPS URL')
        try:
            parsed = urllib.parse.urlsplit(url)
        except (ValueError, UnicodeError):
            raise ValueError('Publisher URL is malformed; sensitive URL omitted') from None
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.fragment):
            raise ValueError('Asset transport requires a bounded anonymous HTTPS URL')

    @contextmanager
    def open(self, url: str, *, method: str = 'GET', check=None, deadline: float):
        current = url
        for redirect in range(6):
            _check(check, deadline)
            self.validate_url(current)
            response = None
            try:
                request = urllib.request.Request(current, method=method, headers={
                    'Accept-Encoding': 'identity', 'User-Agent': 'CorridorKey-local-assets/1'})
                response_type = _response_class(deadline, check, self.socket_timeout)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect(),
                                                     _DeadlineHTTPHandler(response_type), _DeadlineHTTPSHandler(response_type, self.context))
                response = opener.open(request, timeout=min(self.socket_timeout, max(0.001, deadline - time.monotonic())))
            except urllib.error.HTTPError as exc:
                response = exc
            except (OSError, urllib.error.URLError, http.client.HTTPException, UnicodeError):
                raise RuntimeError('Publisher connection failed or timed out; no retry performed') from None
            try:
                status = response.status
                if status in {301, 302, 303, 307, 308}:
                    location = response.headers.get('Location')
                    if not location or redirect == 5:
                        raise RuntimeError('Publisher redirect is missing or exceeds five redirects')
                    try:
                        current = urllib.parse.urljoin(current, location)
                    except (ValueError, UnicodeError):
                        raise ValueError('Publisher redirect URL is malformed; sensitive URL omitted') from None
                    self.validate_url(current)
                    continue
                if status in {401, 403}:
                    raise RuntimeError('Publisher requires access or authentication; anonymous download refused')
                if status != 200:
                    raise RuntimeError(f'Publisher returned HTTP{status}; no retry performed')
                if response.headers.get('Content-Encoding', 'identity').strip().casefold() not in {'', 'identity'}:
                    raise ValueError('Publisher content encoding must be identity')
                _check(check, deadline)
                yield _PublicResponse(response)
                return
            finally:
                try:
                    response.close()
                except (OSError, http.client.HTTPException):
                    raise RuntimeError('Publisher connection cleanup failed') from None
        raise RuntimeError('Publisher redirect limit exceeded')


def _length(response) -> int | None:
    values = response.headers.get_all('Content-Length', [])
    if not values:
        return None
    if len(values) != 1 or not re.fullmatch('[0-9]{1,16}', values[0].strip()):
        raise ValueError('Invalid or ambiguous publisher Content-Length')
    return int(values[0])


def _read(response, limit: int, check, deadline: float) -> bytes:
    declared = _length(response)
    if declared is not None and declared > limit:
        raise ValueError('Publisher metadata exceeds its admitted byte limit')
    data = bytearray()
    while len(data) <= limit:
        _check(check, deadline)
        block = response.read1(min(65536, limit + 1 - len(data)))
        _check(check, deadline)
        if not block:
            break
        data.extend(block)
    if len(data) > limit or (declared is not None and len(data) != declared):
        raise ValueError('Publisher metadata size differs from bounded response')
    return bytes(data)


def _path(value: Any) -> str:
    if (not isinstance(value, str) or not value or len(value.encode('utf-8')) > 1024 or len(value.split('/')) > 32 or
            '\\' in value or PurePosixPath(value).is_absolute()):
        raise ValueError('Publisher file path must be bounded and relative')
    for part in value.split('/'):
        if (not part or len(part.encode('utf-8')) > 255 or part in {'.', '..'} or part.endswith(('.', ' ')) or
                any(ord(character) < 32 or character in '<>:"|?*' for character in part) or
                re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part, re.IGNORECASE)):
            raise ValueError('Publisher path is not portable or safe')
    return value


def _file_url(repo: str, revision: str, name: str) -> str:
    return f'https://huggingface.co/{repo}/resolve/{revision}/' + urllib.parse.quote(name, safe='/')


def _destination(asset: str, name: str) -> str:
    prefix = {'corridorkey': 'weights/', 'cradio': 'vfms/cradio/C-RADIOv4-SO400M/', 'moge': 'vfms/moge/', 'rvm': 'vfms/rvm/'}[asset]
    return _path(prefix + _path(name))


def _checksum(sibling: dict[str, Any], size: int) -> dict[str, str]:
    lfs = sibling.get('lfs')
    if lfs is not None:
        if not isinstance(lfs, dict) or lfs.get('size') != size or not isinstance(lfs.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', lfs['sha256']):
            raise ValueError('Publisher LFS size/SHA256 metadata is missing or invalid')
        return {'algorithm': 'sha256', 'value': lfs['sha256']}
    value = sibling.get('blobId')
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('Publisher Git blob identity is missing or invalid')
    return {'algorithm': 'git-sha1', 'value': value}


def resolve_plan(*, check=None, progress=None, transport=None) -> dict[str, Any]:
    result: dict[str, Any] = {'status': 'failed', 'plan': None, 'error': None, 'warning': None}
    client = HttpTransport() if transport is None else transport
    try:
        files = []
        publishers = []
        for index, publisher in enumerate(catalog()['publishers']):
            _check(check)
            _progress(progress, 'download_metadata', index, 4)
            asset = publisher['asset']
            if asset == 'rvm':
                deadline = time.monotonic() + client.metadata_seconds
                with client.open(RVM_URL, method='HEAD', check=check, deadline=deadline) as response:
                    size = _length(response)
                if type(size) is not int or not 0 < size <= MAX_DOWNLOAD_BYTES:
                    raise ValueError('RVM publisher does not provide an exact positive file size')
                files.append({'asset': asset, 'source_path': RVM_NAME, 'path': _destination(asset, RVM_NAME),
                              'url': RVM_URL, 'bytes': size, 'checksum': {'algorithm': 'md5', 'value': RVM_MD5},
                              'repository': None, 'revision': None})
                publishers.append({**publisher, 'resolved_revision': None})
                continue
            repo = publisher['repository']
            requested = publisher['requested_revision']
            suffix = '' if requested == 'HEAD' else '/revision/' + requested
            url = f'https://huggingface.co/api/models/{repo}{suffix}?blobs=true'
            deadline = time.monotonic() + client.metadata_seconds
            with client.open(url, check=check, deadline=deadline) as response:
                metadata = _json(_read(response, MAX_METADATA_BYTES, check, deadline))
            revision = metadata.get('sha')
            siblings = metadata.get('siblings')
            if (not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision) or
                    (requested != 'HEAD' and revision != requested) or not isinstance(siblings, list) or len(siblings) > MAX_FILES):
                raise ValueError('Publisher revision/file inventory is unpinned or outside bounds')
            chosen = set()
            for sibling in siblings:
                _check(check)
                if not isinstance(sibling, dict):
                    raise TypeError('Invalid publisher file metadata')
                name = _path(sibling.get('rfilename'))
                include = (name in {'corridorkey_v2.safetensors', 'foundation_stats.safetensors'} if asset == 'corridorkey' else
                           name == 'model.pt' if asset == 'moge' else not name.endswith('.pth.tar'))
                if not include:
                    continue
                size = sibling.get('size')
                if type(size) is not int or not 0 <= size <= MAX_DOWNLOAD_BYTES or name in chosen:
                    raise ValueError('Publisher size is unknown/invalid or file is duplicated')
                chosen.add(name)
                files.append({'asset': asset, 'source_path': name, 'path': _destination(asset, name), 'url': _file_url(repo, revision, name),
                              'bytes': size, 'checksum': _checksum(sibling, size), 'repository': repo, 'revision': revision})
            required = {'corridorkey_v2.safetensors', 'foundation_stats.safetensors'} if asset == 'corridorkey' else {'model.pt'} if asset == 'moge' else {'hf_model.py', 'config.json', 'model.safetensors'}
            if not required <= chosen:
                raise ValueError('Publisher snapshot lacks required runtime files')
            publishers.append({**publisher, 'resolved_revision': revision})
        plan = {'schema_version': 1, 'kind': 'corridorkey_asset_download_plan', 'publishers': publishers,
                'files': sorted(files, key=lambda item: item['path']), 'total_bytes': sum(item['bytes'] for item in files)}
        plan['token'] = maintenance._digest(plan)
        validate_plan(plan)
        _check(check)
        _progress(progress, 'download_metadata', 4, 4)
        result.update(status='complete', plan=plan, warning='RVM uses the repository-published legacy MD5 checksum; model compatibility remains NOT VERIFIED.')
    except BaseException as exc:
        logging.getLogger(__name__).exception('Asset metadata resolution stopped')
        result.update(status='cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed', error=f'{type(exc).__name__}: {exc}'[:2048])
    return result


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    raw = maintenance._canonical(plan)
    if len(raw) > MAX_PLAN_BYTES:
        raise ValueError('Asset download plan exceeds4MiB')
    frozen = _json(raw)
    token = frozen.pop('token', None)
    if (set(frozen) != {'schema_version', 'kind', 'publishers', 'files', 'total_bytes'} or
            type(frozen.get('schema_version')) is not int or frozen['schema_version'] != 1 or
            frozen.get('kind') != 'corridorkey_asset_download_plan' or maintenance._digest(frozen) != token or
            not isinstance(frozen.get('publishers'), list) or len(frozen['publishers']) != 4 or
            not isinstance(frozen.get('files'), list) or not 1 <= len(frozen['files']) <= MAX_FILES or
            type(frozen.get('total_bytes')) is not int or not 0 < frozen['total_bytes'] <= MAX_DOWNLOAD_BYTES):
        raise ValueError('Invalid/changed asset download plan')
    sources = {}
    for item, known in zip(frozen['publishers'], catalog()['publishers'], strict=True):
        if not isinstance(item, dict) or set(item) != set(known) | {'resolved_revision'} or any(item[key] != value for key, value in known.items()):
            raise ValueError('Asset plan publisher/catalogue provenance differs from repository policy')
        revision = item['resolved_revision']
        if known['asset'] == 'rvm':
            if revision is not None:
                raise ValueError('Unexpected RVM revision')
        elif not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision) or (known['requested_revision'] != 'HEAD' and revision != known['requested_revision']):
            raise ValueError('Asset plan contains unpinned revision')
        sources[known['asset']] = item
    normalized = set()
    directories = set()
    found = {name: set() for name in sources}
    for item in frozen['files']:
        if (not isinstance(item, dict) or set(item) != {'asset', 'source_path', 'path', 'url', 'bytes', 'checksum', 'repository', 'revision'} or
                item.get('asset') not in sources or type(item.get('bytes')) is not int or not 0 <= item['bytes'] <= MAX_DOWNLOAD_BYTES):
            raise ValueError('Invalid download file record')
        source = sources[item['asset']]
        name = _path(item['source_path'])
        destination = _destination(item['asset'], name)
        if item['asset'] == 'corridorkey' and name not in {'corridorkey_v2.safetensors', 'foundation_stats.safetensors'}:
            raise ValueError('Unexpected CorridorKey file')
        if item['asset'] == 'moge' and name != 'model.pt':
            raise ValueError('Unexpected MoGe file')
        if item['asset'] == 'cradio' and name.endswith('.pth.tar'):
            raise ValueError('Excluded duplicate C-RADIO archive')
        expected_url = RVM_URL if item['asset'] == 'rvm' else _file_url(source['repository'], source['resolved_revision'], name)
        if (item['path'] != destination or item['url'] != expected_url or item['repository'] != source['repository'] or
                item['revision'] != source['resolved_revision']):
            raise ValueError('Plan file origin/destination differs from pinned publisher')
        checksum = item['checksum']
        if not isinstance(checksum, dict) or set(checksum) != {'algorithm', 'value'}:
            raise ValueError('Missing download checksum')
        algorithm = checksum['algorithm']
        length = {'sha256': 64, 'git-sha1': 40, 'md5': 32}.get(algorithm)
        if length is None or not isinstance(checksum['value'], str) or not re.fullmatch('[0-9a-f]{' + str(length) + '}', checksum['value']):
            raise ValueError('Unknown/invalid publisher checksum')
        if item['asset'] == 'rvm':
            if name != RVM_NAME or checksum != {'algorithm': 'md5', 'value': RVM_MD5} or item['bytes'] <= 0:
                raise ValueError('RVM plan differs from repository-published checksum/source')
        elif algorithm == 'md5':
            raise ValueError('HF assets require LFS SHA256 or Git blob SHA1')
        folded = unicodedata.normalize('NFC', destination).casefold()
        if folded in normalized or any(str(parent) in normalized for parent in PurePosixPath(folded).parents) or any(other.startswith(folded + '/') for other in normalized):
            raise ValueError('Download paths collide by case/Unicode/file-parent relationship')
        normalized.add(folded)
        # Case-sensitive filesystems create each raw spelling independently.
        # Portable file-collision checks above remain Unicode/case normalized.
        directories.update(str(parent) for parent in PurePosixPath(destination).parents if str(parent) != '.')
        if len(directories) > MAX_DIRECTORIES:
            raise ValueError('Download directory inventory exceeds16384')
        found[item['asset']].add(name)
    if (found['corridorkey'] != {'corridorkey_v2.safetensors', 'foundation_stats.safetensors'} or found['moge'] != {'model.pt'} or
            found['rvm'] != {RVM_NAME} or not {'hf_model.py', 'config.json', 'model.safetensors'} <= found['cradio'] or
            sum(item['bytes'] for item in frozen['files']) != frozen['total_bytes']):
        raise ValueError('Download plan inventory/byte total is incomplete')
    frozen['token'] = token
    return frozen


def _digester(item: dict[str, Any]):
    checksum = item['checksum']
    algorithm = checksum['algorithm']
    digest = hashlib.sha256() if algorithm == 'sha256' else hashlib.md5() if algorithm == 'md5' else hashlib.sha1()
    if algorithm == 'git-sha1':
        digest.update(b'blob ' + str(item['bytes']).encode() + b'\0')
    return digest


def _parent(parent: str | Path, protected_paths) -> Path:
    lexical = Path(parent).absolute()
    maintenance._stat(lexical, directory=True)
    resolved = lexical.resolve(strict=True)
    if len(str(resolved)) > 3500:
        raise ValueError('Download destination path is too long')
    for value in protected_paths:
        path = Path(value).resolve(strict=False)
        if resolved == path or resolved.is_relative_to(path):
            raise ValueError('Download destination overlaps a selected asset/source/output path')
    return resolved


def _copy_file(item: dict[str, Any], target: Path, client, result, check, progress) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name('.download-' + uuid.uuid4().hex)
    expected = _digester(item)
    observed = hashlib.sha256()
    count = 0
    deadline = time.monotonic() + client.file_seconds
    with stage.open('xb') as stream:
        with client.open(item['url'], check=check, deadline=deadline) as response:
            length = _length(response)
            if length is not None and length != item['bytes']:
                raise ValueError('Publisher Content-Length differs from reviewed plan')
            while count < item['bytes']:
                _check(check, deadline)
                block = response.read1(min(1024 * 1024, item['bytes'] - count))
                _check(check, deadline)
                if not block:
                    raise ValueError('Publisher stream truncated before reviewed byte count')
                written = stream.write(block)
                result['copied_bytes'] += written
                if written != len(block):
                    raise OSError('Short download file write')
                count += written
                expected.update(block)
                observed.update(block)
                _progress(progress, 'asset_download_bytes', result['copied_bytes'], result['total_bytes'])
            _check(check, deadline)
            if response.read1(1):
                raise ValueError('Publisher stream exceeds reviewed byte count')
        if expected.hexdigest() != item['checksum']['value']:
            raise ValueError('Downloaded bytes fail the reviewed publisher checksum')
        stream.flush()
        os.fsync(stream.fileno())
        descriptor = os.fstat(stream.fileno())
        stat_before = {'bytes': descriptor.st_size, 'mtime_ns': descriptor.st_mtime_ns,
                       'device': descriptor.st_dev, 'inode': descriptor.st_ino}
        path_info = maintenance._stat(stage)
        if descriptor.st_size != count or any(path_info[key] != value for key, value in stat_before.items()):
            raise ValueError('Download staging pathname differs from its hashed descriptor')
    _check(check, deadline)
    if any(maintenance._stat(stage)[key] != value for key, value in stat_before.items()):
        raise ValueError('Download staging pathname changed before publication')
    if target.exists() or target.is_symlink():
        raise FileExistsError('Download target unexpectedly exists; no overwrite')
    stage.rename(target)
    publication._sync(target.parent)
    stat_after = maintenance._stat(target)
    if any(stat_after[key] != stat_before[key] for key in ('bytes', 'mtime_ns', 'device', 'inode')):
        raise ValueError('Downloaded file changed during publication')
    result['copied_files'] += 1
    return {'path': item['path'], 'bytes': count, 'sha256': observed.hexdigest(), 'identity': stat_after,
            'publisher_checksum': item['checksum']}


def download_plan(plan: dict[str, Any], existing_parent: str | Path, max_bytes: int, *, protected_paths=(), check=None,
                  progress=None, transport=None) -> dict[str, Any]:
    result: dict[str, Any] = {'status': 'failed', 'destination': None, 'usable': False, 'copied_bytes': 0, 'copied_files': 0,
                              'total_bytes': None, 'partial_files': [], 'error': None, 'warning': None, 'scope': SCOPE,
                              'weights_relative': 'weights/corridorkey_v2.safetensors'}
    final = raw = None
    records = []
    expected_paths: set[str] = set()
    try:
        frozen = validate_plan(plan)
        expected_paths = {item['path'] for item in frozen['files']}
        if type(max_bytes) is not int or not 0 <= max_bytes < 2**63 or frozen['total_bytes'] > max_bytes:
            raise ValueError('Reviewed download exceeds the explicit maximum byte budget')
        _check(check)
        parent = _parent(existing_parent, protected_paths)
        client = HttpTransport() if transport is None else transport
        root = Path(tempfile.mkdtemp(prefix='corridorkey-assets-', dir=parent))
        result.update(destination=str(root), total_bytes=frozen['total_bytes'])
        publication._sync(parent)
        for index, item in enumerate(frozen['files']):
            _check(check)
            records.append(_copy_file(item, root / item['path'], client, result, check, progress))
            _progress(progress, 'asset_download_files', index + 1, len(frozen['files']))
        for record in records:
            if maintenance._stat(root / record['path']) != record['identity']:
                raise ValueError('Downloaded asset changed before completion')
        payload = {'schema_version': 1, 'kind': 'corridorkey_asset_download', 'plan': frozen, 'files': records, 'scope': SCOPE,
                   'compatibility': 'NOT VERIFIED', 'signed_redirects': 'Not persisted; stable publisher URLs only'}
        raw = maintenance._canonical({'payload': payload, 'sha256': maintenance._digest(payload)})
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError('Asset download provenance exceeds8MiB')
        _check(check)
        final = root / 'asset-download.json'
        publication._seal(final, raw)
        result.update(status='complete', usable=True)
    except BaseException as exc:
        logging.getLogger(__name__).exception('Asset download stopped')
        if publication._committed(final, raw) and result['destination'] is not None:
            try:
                for record in records:
                    if maintenance._stat(Path(result['destination']) / record['path']) != record['identity']:
                        raise ValueError('Published downloaded asset changed')
            except (OSError, ValueError) as failure:
                result.update(status='failed', usable=False, error=f'Published assets failed final identity validation: {failure}'[:2048])
            else:
                result.update(status='complete', usable=True, warning=f'Committed download; finalization reported {type(exc).__name__}'[:2048])
        else:
            result.update(status='cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed',
                          error=f'{type(exc).__name__}: {exc}'[:2048])
        if result['destination'] is not None:
            try:
                root = Path(result['destination'])
                count = total = visited = 0
                pending = [root]
                partial = []
                while pending:
                    directory = pending.pop()
                    maintenance._stat(directory, directory=True)
                    for name in maintenance._names(directory, MAX_FILES + 1, None):
                        path = directory / name
                        visited += 1
                        if visited > MAX_FILES * 8:
                            raise ValueError('Partial download inventory exceeds bound')
                        if path.is_dir() and not path.is_symlink():
                            maintenance._stat(path, directory=True)
                            pending.append(path)
                        else:
                            relative = path.relative_to(root).as_posix()
                            if relative in expected_paths:
                                total += maintenance._stat(path)['bytes']
                                count += 1
                            elif relative != 'asset-download.json' and not (path.parent == root and name.startswith('.seal-')):
                                total += maintenance._stat(path)['bytes']
                                partial.append(relative)
                result.update(copied_bytes=total, copied_files=count, partial_files=partial)
            except (OSError, ValueError) as failure:
                result.update(copied_bytes=None, copied_files=None, warning=f'Partial inventory unavailable: {failure}'[:2048])
    return result


def persist_plan(plan: dict[str, Any], output_path: str | Path, *, check=None) -> dict[str, Any]:
    result: dict[str, Any] = {'status': 'failed', 'destination': None, 'usable': False, 'partial_path': None, 'error': None, 'warning': None}
    stage = None
    published = False
    output = Path(output_path).absolute()
    raw = None
    try:
        raw = maintenance._canonical(validate_plan(plan))
        if not output.parent.is_dir() or output.exists() or output.is_symlink():
            raise ValueError('Choose a new plan file in an existing folder')
        _check(check)
        descriptor, name = tempfile.mkstemp(prefix='.asset-plan-', dir=output.parent)
        stage = Path(name)
        result['partial_path'] = str(stage)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _check(check)
        os.link(stage, output)
        published = True
        result.update(status='complete', destination=str(output), usable=True)
        stage.unlink()
        result['partial_path'] = None
        publication._sync(output.parent)
    except BaseException as exc:
        logging.getLogger(__name__).exception('Asset plan persistence stopped')
        committed = published
        if not committed and stage is not None:
            try:
                committed = output.lstat().st_ino == stage.lstat().st_ino and output.lstat().st_dev == stage.lstat().st_dev
            except OSError:
                pass
        if committed and publication._committed(output, raw):
            result.update(status='complete', destination=str(output), usable=True, warning=f'Plan committed; finalization reported {type(exc).__name__}')
        else:
            result.update(status='cancelled' if isinstance(exc, (CancelledError, KeyboardInterrupt)) else 'failed', error=f'{type(exc).__name__}: {exc}'[:2048])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('catalog')
    plan = commands.add_parser('plan', aliases=['resolve'])
    plan.add_argument('--output', type=Path)
    download = commands.add_parser('download')
    download.add_argument('--plan', required=True, type=Path)
    download.add_argument('--output-parent', required=True, type=Path)
    download.add_argument('--max-bytes', required=True, type=int)
    download.add_argument('--protect', action='append', default=[], type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'catalog':
            result = catalog()
        elif args.command in {'plan', 'resolve'}:
            result = resolve_plan()
            if args.output is not None and result['status'] == 'complete':
                persisted = persist_plan(result['plan'], args.output)
                if not persisted['usable']:
                    result = persisted
        else:
            with args.plan.open('rb') as stream:
                raw = stream.read(MAX_PLAN_BYTES + 1)
            if len(raw) > MAX_PLAN_BYTES:
                raise ValueError('Asset plan exceeds4MiB')
            result = download_plan(_json(raw), args.output_parent, args.max_bytes, protected_paths=args.protect)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'[:2048]}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result.get('status', 'complete') == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
