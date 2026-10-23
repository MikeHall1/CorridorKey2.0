"""Actual owned loopback HTTP only; no remote publisher/model data is requested."""
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import urllib.parse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from corridorkey import asset_download as download
from corridorkey.control import CancelledError
from tests.test_asset_audit import valid_assets


class LocalTransport(download.HttpTransport):
    """Private injection maps stable public origins onto this owned HTTP server."""
    def __init__(self, origin, **kwargs):
        super().__init__(**kwargs)
        self.origin = origin

    def validate_url(self, url):
        if not url.startswith(self.origin + '/'):
            raise ValueError('Local fixture refuses requests outside its owned origin')

    @contextmanager
    def open(self, url, *, method='GET', check=None, deadline):
        mapped = self.origin + '/route?url=' + urllib.parse.quote(url, safe='')
        with super().open(mapped, method=method, check=check, deadline=deadline) as response:
            yield response


class PublisherFixture:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        valid_assets(self.root / 'tiny-assets')
        root = self.root / 'tiny-assets'
        self.routes = {}
        self.requests = []
        self.mode: str | None = None
        self.affected: str | None = None
        self.rvm = (root / 'vfms/rvm' / download.RVM_NAME).read_bytes()
        self.md5_patch = patch.object(download, 'RVM_MD5', hashlib.md5(self.rvm).hexdigest())
        for publisher in download.catalog()['publishers']:
            asset = publisher['asset']
            if asset == 'rvm':
                self.routes[download.RVM_URL] = self.rvm
                continue
            revision = 'a' * 40 if asset == 'corridorkey' else publisher['requested_revision']
            if asset == 'corridorkey':
                files = {name: (root / 'weights' / name).read_bytes() for name in ('corridorkey_v2.safetensors', 'foundation_stats.safetensors')}
                files['ignored.txt'] = b'not selected'
            elif asset == 'moge':
                files = {'model.pt': (root / 'vfms/moge/model.pt').read_bytes()}
            else:
                tree = root / 'vfms/cradio/C-RADIOv4-SO400M'
                files = {path.name: path.read_bytes() for path in tree.iterdir()}
                files.update({'README.md': b'local fixture licence and documentation', 'duplicate.pth.tar': b'excluded duplicate'})
            siblings = []
            for name, raw in files.items():
                item = {'rfilename': name, 'size': len(raw), 'blobId': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}
                if name.endswith('.safetensors'):
                    item['lfs'] = {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'pointerSize': 128}
                siblings.append(item)
                self.routes[download._file_url(publisher['repository'], revision, name)] = raw
            suffix = '' if publisher['requested_revision'] == 'HEAD' else '/revision/' + revision
            url = f'https://huggingface.co/api/models/{publisher["repository"]}{suffix}?blobs=true'
            self.routes[url] = json.dumps({'sha': revision, 'siblings': siblings}).encode()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_HEAD(self):
                self.respond(head=True)

            def do_GET(self):
                self.respond(head=False)

            def respond(self, *, head):
                route = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get('url', [None])[0]
                fixture.requests.append((route, dict(self.headers)))
                raw = fixture.routes.get(route)
                if fixture.mode == 'header-slow' and route == fixture.affected:
                    try:
                        for byte in b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nx':
                            self.connection.sendall(bytes([byte]))
                            time.sleep(0.02)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if fixture.mode == 'malformed-redirect' and route == fixture.affected:
                    self.send_response(302)
                    self.send_header('Location', '/private?signature=SIGNED-SECRET SPACE')
                    self.end_headers()
                    return
                if self.path == '/loop' or fixture.mode == 'redirect' and route == fixture.affected:
                    self.send_response(302)
                    self.send_header('Location', '/loop?secret=SIGNED-URL-MUST-NOT-PERSIST')
                    self.end_headers()
                    return
                if self.path.startswith('/loop'):
                    self.send_response(302)
                    self.send_header('Location', '/loop?secret=SIGNED-URL-MUST-NOT-PERSIST')
                    self.end_headers()
                    return
                if raw is None:
                    self.send_error(404)
                    return
                if fixture.mode == 'auth' and route == fixture.affected:
                    self.send_error(403)
                    return
                self.send_response(200)
                size = len(raw)
                if fixture.mode == 'length' and route == fixture.affected:
                    size += 1
                if fixture.mode == 'oversize' and route == fixture.affected:
                    size = download.MAX_METADATA_BYTES + 1
                self.send_header('Content-Length', str(size))
                if fixture.mode == 'encoding' and route == fixture.affected:
                    self.send_header('Content-Encoding', 'gzip')
                self.end_headers()
                if head:
                    return
                if fixture.mode == 'corrupt' and route == fixture.affected:
                    raw = b'x' * len(raw)
                if fixture.mode == 'truncate' and route == fixture.affected:
                    raw = raw[:-1]
                try:
                    if fixture.mode == 'slow' and route == fixture.affected:
                        for byte in raw:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(0.03)
                    else:
                        self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.md5_patch.start()
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        self.thread.start()
        self.transport = LocalTransport(f'http://127.0.0.1:{self.server.server_address[1]}', socket_timeout=0.15,
                                        metadata_seconds=2, file_seconds=2)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.md5_patch.stop()
        self.directory.cleanup()

    def plan(self):
        result = download.resolve_plan(transport=self.transport)
        if result['status'] != 'complete':
            raise AssertionError(result)
        return result['plan']


class AssetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PublisherFixture()
        self.addCleanup(self.fixture.close)
        self.parent = self.fixture.root / 'downloads'
        self.parent.mkdir()

    def test_real_http_metadata_and_download_exact_layout_and_publisher_hashes(self):
        plan = self.fixture.plan()
        self.assertEqual(plan['publishers'][0]['resolved_revision'], 'a' * 40)
        self.assertEqual({item['checksum']['algorithm'] for item in plan['files']}, {'sha256', 'git-sha1', 'md5'})
        self.assertFalse(any(item['source_path'].endswith('.pth.tar') for item in plan['files']))
        progress = []
        result = download.download_plan(plan, self.parent, plan['total_bytes'], transport=self.fixture.transport, progress=progress.append)
        self.assertEqual(result['status'], 'complete', result)
        self.assertTrue(result['usable'])
        root = Path(result['destination'])
        for item in plan['files']:
            self.assertEqual((root / item['path']).read_bytes(), self.fixture.routes[item['url']])
        provenance = json.loads((root / 'asset-download.json').read_text())
        self.assertEqual(provenance['payload']['plan'], plan)
        self.assertEqual(result['copied_bytes'], plan['total_bytes'])
        self.assertEqual(result['copied_files'], len(plan['files']))
        self.assertTrue(progress)
        self.assertTrue(all(headers.get('Accept-Encoding') == 'identity' and 'Authorization' not in headers
                            for _, headers in self.fixture.requests))

    def test_catalog_offline_and_cli_import_no_models_no_network(self):
        with patch.object(download.HttpTransport, 'open', side_effect=AssertionError('Offline catalog')):
            data = download.catalog()
        self.assertTrue(all(item['bytes'] is None for item in data['publishers']))
        result = subprocess.run([sys.executable, '-m', 'corridorkey.asset_download', 'catalog'],
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(json.loads(result.stdout), data)
        command = "import sys; import corridorkey.asset_download; print('torch' in sys.modules,'numpy' in sys.modules)"
        imported = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(imported.stdout.strip(), 'False False')

    def test_plan_tamper_rehashed_unknown_origin_unpinned_paths_and_budget_refuse(self):
        plan = self.fixture.plan()
        for kind in ('origin', 'revision', 'path', 'unknown', 'collision'):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(plan)
                if kind == 'origin':
                    bad['files'][0]['url'] = 'https://example.invalid/arbitrary-code'
                elif kind == 'revision':
                    bad['files'][0]['revision'] = 'HEAD'
                elif kind == 'path':
                    bad['files'][0]['source_path'] = '../outside'
                elif kind == 'unknown':
                    bad['files'][0]['bytes'] = None
                else:
                    bad['files'].append(copy.deepcopy(bad['files'][0]))
                bad.pop('token')
                bad['token'] = download.maintenance._digest(bad)
                result = download.download_plan(bad, self.parent, plan['total_bytes'], transport=self.fixture.transport)
                self.assertFalse(result['usable'])
                self.assertIsNone(result['destination'])
        self.assertIsNone(download.download_plan(plan, self.parent, plan['total_bytes'] - 1, transport=self.fixture.transport)['destination'])
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_missing_cradio_weights_and_directory_bounds_refuse_before_execution(self):
        plan = self.fixture.plan()
        bad = copy.deepcopy(plan)
        bad['files'] = [item for item in bad['files'] if not (item['asset'] == 'cradio' and item['source_path'] == 'model.safetensors')]
        bad['total_bytes'] = sum(item['bytes'] for item in bad['files'])
        bad.pop('token')
        bad['token'] = download.maintenance._digest(bad)
        self.assertIsNone(download.download_plan(bad, self.parent, plan['total_bytes'], transport=self.fixture.transport)['destination'])
        metadata_url = next(url for url in self.fixture.routes if '/api/models/nvidia/' in url)
        metadata = json.loads(self.fixture.routes[metadata_url])
        metadata['siblings'] = [item for item in metadata['siblings'] if item['rfilename'] != 'model.safetensors']
        self.fixture.routes[metadata_url] = json.dumps(metadata).encode()
        self.assertIsNone(download.resolve_plan(transport=self.fixture.transport)['plan'])
        for path in ('a/' * 33 + 'x', 'é' * 128, 'CON.txt', 'a/../file', 'a./file'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                download._path(path)
        with patch.object(download, 'MAX_DIRECTORIES', 5):
            rejected = download.download_plan(plan, self.parent, plan['total_bytes'], transport=self.fixture.transport)
        self.assertIsNone(rejected['destination'])
        aliases = copy.deepcopy(plan)
        publisher = aliases['publishers'][1]
        for name in ('Alias/one.txt', 'alias/two.txt'):
            aliases['files'].append({'asset': 'cradio', 'source_path': name, 'path': download._destination('cradio', name),
                                     'url': download._file_url(publisher['repository'], publisher['resolved_revision'], name),
                                     'bytes': 1, 'checksum': {'algorithm': 'git-sha1', 'value': hashlib.sha1(b'blob 1\0x').hexdigest()},
                                     'repository': publisher['repository'], 'revision': publisher['resolved_revision']})
        aliases['total_bytes'] += 2
        aliases.pop('token')
        aliases['token'] = download.maintenance._digest(aliases)
        original_dirs = {str(parent) for item in plan['files'] for parent in PurePosixPath(item['path']).parents if str(parent) != '.'}
        requests_before = len(self.fixture.requests)
        with patch.object(download, 'MAX_DIRECTORIES', len(original_dirs) + 1):
            rejected = download.download_plan(aliases, self.parent, aliases['total_bytes'], transport=self.fixture.transport)
        self.assertIsNone(rejected['destination'])
        self.assertEqual(len(self.fixture.requests), requests_before)
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_large_budget_is_independent_of_plan_payload_cap(self):
        plan = self.fixture.plan()
        result = download.download_plan(plan, self.parent, 2**63 - 1, transport=self.fixture.transport)
        self.assertTrue(result['usable'], result)

    def test_unknown_metadata_sizes_oversized_encoding_auth_and_redirect_refuse(self):
        route = next(name for name in self.fixture.routes if '/api/models/' in name)
        for mode in ('oversize', 'encoding', 'auth', 'redirect'):
            with self.subTest(mode=mode):
                self.fixture.mode, self.fixture.affected = mode, route
                result = download.resolve_plan(transport=self.fixture.transport)
                self.assertEqual(result['status'], 'failed')
                self.assertIsNone(result['plan'])
                self.assertNotIn('SIGNED-URL', json.dumps(result))
        self.fixture.mode = None
        metadata = json.loads(self.fixture.routes[route])
        metadata['siblings'][0]['size'] = None
        self.fixture.routes[route] = json.dumps(metadata).encode()
        self.assertIsNone(download.resolve_plan(transport=self.fixture.transport)['plan'])

    def test_payload_mismatch_length_truncation_and_cancel_keep_unsealed_partial(self):
        plan = self.fixture.plan()
        route = plan['files'][0]['url']
        for mode in ('corrupt', 'length', 'truncate'):
            with self.subTest(mode=mode):
                self.fixture.mode, self.fixture.affected = mode, route
                result = download.download_plan(plan, self.parent, plan['total_bytes'], transport=self.fixture.transport)
                self.assertFalse(result['usable'])
                root = Path(result['destination'])
                self.assertFalse((root / 'asset-download.json').exists())
                self.assertTrue(result['partial_files'])
                self.assertEqual(result['copied_bytes'], sum(path.stat().st_size for path in root.rglob('*') if path.is_file()))
        self.fixture.mode = None

        def progress(event):
            if event.stage == 'asset_download_bytes' and event.completed:
                raise CancelledError('fixture cancelled after first payload bytes')

        cancelled = download.download_plan(plan, self.parent, plan['total_bytes'], transport=self.fixture.transport, progress=progress)
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertFalse(cancelled['usable'])
        self.assertFalse((Path(cancelled['destination']) / 'asset-download.json').exists())

    def test_slow_stream_overall_deadline_is_terminal(self):
        plan = self.fixture.plan()
        self.fixture.mode, self.fixture.affected = 'slow', plan['files'][0]['url']
        self.fixture.transport.file_seconds = 0.1
        started = time.monotonic()
        result = download.download_plan(plan, self.parent, plan['total_bytes'], transport=self.fixture.transport)
        self.assertFalse(result['usable'])
        self.assertLess(time.monotonic() - started, 1)

    def test_header_trickle_and_header_cancellation_are_bounded(self):
        route = next(url for url in self.fixture.routes if '/api/models/' in url)
        self.fixture.mode, self.fixture.affected = 'header-slow', route
        self.fixture.transport.metadata_seconds = 0.1
        started = time.monotonic()
        result = download.resolve_plan(transport=self.fixture.transport)
        self.assertEqual(result['status'], 'failed')
        self.assertLess(time.monotonic() - started, 0.8)
        self.fixture.transport.metadata_seconds = 2
        started = time.monotonic()

        def check():
            if time.monotonic() - started > 0.06:
                raise CancelledError('fixture cancellation during response headers')

        result = download.resolve_plan(transport=self.fixture.transport, check=check)
        self.assertEqual(result['status'], 'cancelled')
        self.assertLess(time.monotonic() - started, 0.8)

    def test_signed_redirect_errors_are_sanitized_in_results_and_logs(self):
        route = next(url for url in self.fixture.routes if '/api/models/' in url)
        self.fixture.mode, self.fixture.affected = 'malformed-redirect', route
        original = self.fixture.transport.validate_url

        def validate(url):
            if any(ord(character) <= 32 or ord(character) == 127 for character in url):
                raise ValueError('Malformed anonymous URL')
            original(url)

        with patch.object(self.fixture.transport, 'validate_url', validate), self.assertLogs('corridorkey.asset_download', level='ERROR') as logs:
            result = download.resolve_plan(transport=self.fixture.transport)
        self.assertFalse(result['plan'])
        self.assertNotIn('SIGNED-SECRET', json.dumps(result) + '\n'.join(logs.output))
        malformed = 'https://user:SIGNED-SECRET@bad\uff0fhost/file?signature=SIGNED-SECRET'
        try:
            download.HttpTransport().validate_url(malformed)
        except ValueError as error:
            rendered = ''.join(traceback.format_exception(error))
            self.assertNotIn('SIGNED-SECRET', rendered)
        else:
            self.fail('Malformed sensitive URL must refuse before request construction')
        original_join = download.urllib.parse.urljoin

        def malformed_join(base, location):
            return original_join(base, malformed)

        with (patch.object(download.urllib.parse, 'urljoin', malformed_join),
              self.assertLogs('corridorkey.asset_download', level='ERROR') as logs):
            result = download.resolve_plan(transport=self.fixture.transport)
        self.assertFalse(result['plan'])
        self.assertNotIn('SIGNED-SECRET', json.dumps(result) + '\n'.join(logs.output))

    def test_protected_assets_preserved_and_two_downloads_are_isolated(self):
        plan = self.fixture.plan()
        old = self.fixture.root / 'tiny-assets'
        before = {str(path): path.read_bytes() for path in old.rglob('*') if path.is_file()}
        rejected = download.download_plan(plan, old, plan['total_bytes'], protected_paths=(old,), transport=self.fixture.transport)
        self.assertIsNone(rejected['destination'])
        results = []
        threads = [threading.Thread(target=lambda: results.append(download.download_plan(
            plan, self.parent, plan['total_bytes'], transport=self.fixture.transport))) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertTrue(all(result['usable'] for result in results), results)
        self.assertEqual(len({result['destination'] for result in results}), 2)
        self.assertEqual({str(path): path.read_bytes() for path in old.rglob('*') if path.is_file()}, before)

    def test_disk_failure_and_late_commit_report_truthfully(self):
        metadata_url = next(url for url in self.fixture.routes if '/api/models/nvidia/' in url)
        metadata = json.loads(self.fixture.routes[metadata_url])
        publisher = download.catalog()['publishers'][1]
        for name, body in {'asset-download.json': b'legitimate nested publisher JSON', '.download-publisher': b'legitimate publisher dotfile'}.items():
            metadata['siblings'].append({'rfilename': name, 'size': len(body),
                                         'blobId': hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()})
            self.fixture.routes[download._file_url(publisher['repository'], publisher['requested_revision'], name)] = body
        self.fixture.routes[metadata_url] = json.dumps(metadata).encode()
        plan = self.fixture.plan()
        original = os.fsync

        def denied(descriptor):
            if stat.S_ISREG(os.fstat(descriptor).st_mode) and os.fstat(descriptor).st_size > 0:
                raise OSError('fixture disk full')
            original(descriptor)

        with patch.object(download.os, 'fsync', denied):
            failed = download.download_plan(plan, self.parent, plan['total_bytes'], transport=self.fixture.transport)
        self.assertFalse(failed['usable'])
        self.assertFalse((Path(failed['destination']) / 'asset-download.json').exists())
        seal = download.publication._seal

        def sealed(path, raw):
            seal(path, raw)
            raise CancelledError('fixture late cancellation after atomic complete marker')

        with patch.object(download.publication, '_seal', sealed):
            result = download.download_plan(plan, self.parent, plan['total_bytes'], transport=self.fixture.transport)
        self.assertTrue(result['usable'], result)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['copied_bytes'], plan['total_bytes'])
        self.assertEqual(result['copied_files'], len(plan['files']))
        self.assertEqual(result['partial_files'], [])

    def test_staging_path_replacement_cannot_publish_wrong_hashed_inode(self):
        plan = self.fixture.plan()
        original = os.fsync
        changed = []

        def replacing(descriptor):
            if stat.S_ISREG(os.fstat(descriptor).st_mode) and not changed:
                stage = next(self.parent.rglob('.download-*'))
                size = stage.stat().st_size
                stage.rename(stage.with_name(stage.name + '.original'))
                stage.write_bytes(b'x' * size)
                changed.append(True)
            original(descriptor)

        with patch.object(download.os, 'fsync', replacing):
            result = download.download_plan(plan, self.parent, plan['total_bytes'], transport=self.fixture.transport)
        self.assertFalse(result['usable'])
        self.assertIn('descriptor', result['error'])
        self.assertFalse((Path(result['destination']) / 'asset-download.json').exists())

    def test_plan_persistence_is_exact_fresh_and_production_https_policy(self):
        plan = self.fixture.plan()
        path = self.parent / 'reviewed-plan.json'
        self.assertTrue(download.persist_plan(plan, path)['usable'])
        self.assertEqual(json.loads(path.read_bytes()), plan)
        self.assertFalse(download.persist_plan(plan, path)['usable'])
        transport = download.HttpTransport()
        for url in ('http://example.com/file', 'https://user:password@example.com/file', 'https://example.com/file#fragment'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                transport.validate_url(url)


if __name__ == '__main__':
    unittest.main()
