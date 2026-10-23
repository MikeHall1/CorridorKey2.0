"""Owned durable/cache read admission; tiny CPU fixtures, no learned inference."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recovery_store import BundleStore, CacheInvalid, read_payload_bounded

STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'safetensors'))


class StoreReadBoundsTests(unittest.TestCase):
    def test_owner_default_bound_covers_startup_and_workspace_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.bin'
            source.write_bytes(b'owned payload')
            identity = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'owner'}))
            cache = root / 'cache'
            with BundleStore(cache, 100000) as store:
                store.publish(identity, {'payload.bin': source}, {})
            owner = cache / 'entries' / identity.key / 'owner.json'
            owner.write_bytes(owner.read_bytes() + b' ' * (4 * 1024**2))
            before = {path: path.read_bytes() for path in cache.rglob('*') if path.is_file()}
            with self.assertRaises(CacheInvalid), BundleStore(cache, 0):
                self.fail('oversized owner must fail before eviction')
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            marker = cache / 'workspace.json'
            marker.write_bytes(marker.read_bytes() + b' ' * (4 * 1024**2))
            marker_before = marker.read_bytes()
            with self.assertRaises(CacheInvalid):
                BundleStore(cache, 100000)
            self.assertEqual(marker.read_bytes(), marker_before)
            self.assertEqual(owner.read_bytes(), before[owner])

    def test_payload_growth_is_capped_to_admitted_plus_one_on_same_descriptor(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'payload.bin'
            path.write_bytes(b'abc')
            original = os.fdopen
            requested, returned = [], []
            class GrowingReader:
                def __init__(self, stream):
                    self.stream = stream
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.stream.close()
                def fileno(self):
                    return self.stream.fileno()
                def read(self, count):
                    requested.append(count)
                    with path.open('ab') as writer:
                        writer.write(b'x' * 1024**2)
                    block = self.stream.read(count)
                    returned.append(len(block))
                    return block
            def opened(descriptor, *args, **kwargs):
                return GrowingReader(original(descriptor, *args, **kwargs))
            with patch('corridorkey.recovery_store.os.fdopen', opened), self.assertRaises(CacheInvalid):
                read_payload_bounded(path, expected_size=3, expected_sha256=hashlib.sha256(b'abc').hexdigest(), max_bytes=3)
            self.assertEqual(requested, [4])
            self.assertEqual(returned, [4])

    def test_same_descriptor_replacement_and_cancellation_never_return_unverified_bytes(self):
        from corridorkey.control import CancelledError

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'payload.bin'
            path.write_bytes(b'abc')
            replacement = path.with_name('replacement.bin')
            replacement.write_bytes(b'abc')
            calls = 0
            def replace_after_open():
                nonlocal calls
                calls += 1
                if calls == 2:
                    replacement.replace(path)
            with self.assertRaisesRegex(CacheInvalid, 'changed'):
                read_payload_bounded(path, expected_size=3, expected_sha256=hashlib.sha256(b'abc').hexdigest(),
                                     max_bytes=3, check=replace_after_open)
            self.assertEqual(path.read_bytes(), b'abc')
            def cancel():
                raise CancelledError('stop before payload allocation')
            with self.assertRaises(CancelledError):
                read_payload_bounded(path, expected_size=3, expected_sha256=hashlib.sha256(b'abc').hexdigest(),
                                     max_bytes=3, check=cancel)
            path.rename(path.with_name('closed.bin'))

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_workspace_fifo_is_refused_without_blocking_owned_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            os.mkfifo(root / 'workspace.json')
            code = '''import sys
from corridorkey.recovery_store import BundleStore,CacheInvalid
try:
    BundleStore(sys.argv[1],100000)
except CacheInvalid:
    raise SystemExit(0)
raise SystemExit(1)
'''
            result = subprocess.run([sys.executable, '-c', code, str(root)], check=False,
                                    capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual({path.name for path in root.iterdir()}, {'workspace.json'})


@unittest.skipUnless(STACK, 'requires CPU tensor adapters; no model execution')
class DurableReadBoundsTests(unittest.TestCase):
    def fixtures(self):
        from tests.test_durable_hann import DurableHannTests
        from tests.test_durable_rvm import DurableRvmTests

        for name, cls in (('rvm', DurableRvmTests), ('hann', DurableHannTests)):
            fixture = cls()
            fixture.setUp()
            self.addCleanup(fixture.folder.cleanup)
            yield name, fixture

    def put(self, name, fixture, adapter, identity):
        return (adapter.put(identity, fixture.features, fixture.state) if name == 'rvm'
                else adapter.put(identity, fixture.outputs))

    def test_actual_post_lookup_growth_never_deserializes_or_reads_unbounded(self):
        for name, fixture in self.fixtures():
            with self.subTest(adapter=name), BundleStore(fixture.root / 'growth', 100000) as store:
                adapter = fixture.adapter(store)
                identity = fixture.identity(adapter)
                self.assertTrue(self.put(name, fixture, adapter, identity))
                payload = store._entry(identity) / 'payload-000000.bin'
                original_lookup = store.lookup
                def changed(*args, _lookup=original_lookup, _payload=payload, **kwargs):
                    result = _lookup(*args, **kwargs)
                    with _payload.open('ab') as writer:
                        writer.write(b'x' * 1024**2)
                    return result
                with patch.object(store, 'lookup', changed), patch(f'corridorkey.durable_{name}.load', side_effect=AssertionError('must not allocate tensors')), patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded payload read')):
                    self.assertIsNone(adapter.get(identity))
                self.assertEqual(adapter.corrupt, 1)
                self.assertFalse(store._entry(identity).exists())

    def test_bad_header_is_rejected_before_safetensors_load(self):
        from tests.test_durable_hann import DurableHannTests

        for name, fixture in self.fixtures():
            with self.subTest(adapter=name), BundleStore(fixture.root / 'header', 100000) as store:
                adapter = fixture.adapter(store)
                identity = fixture.identity(adapter)
                if name == 'rvm':
                    record = adapter._record(fixture.state)
                    header = {'features': {'dtype': 'F16', 'shape': [2, 1, 1000000000, 1], 'data_offsets': [0, 4]},
                              'state': {'dtype': 'BF16', 'shape': [1, 2, 1], 'data_offsets': [4, 8]}}
                else:
                    assert isinstance(fixture, DurableHannTests)
                    record = adapter._record({'baton': fixture.outputs['baton'], 'aux': fixture.outputs['aux'][:, :, :1]})
                    header = {'baton': {'dtype': 'BF16', 'shape': [1, 2, 1, 1000000000, 4], 'data_offsets': [0, 4]},
                              'aux': {'dtype': 'BF16', 'shape': [1, 2, 1, 32, 32], 'data_offsets': [4, 8]}}
                encoded = json.dumps(header).encode()
                bad = fixture.root / 'bad.bin'
                bad.write_bytes(len(encoded).to_bytes(8, 'little') + encoded + b'\0' * 8)
                store.publish(identity, {f'{name}.safetensors': bad}, record)
                with patch(f'corridorkey.durable_{name}.load', side_effect=AssertionError('must not allocate tensors')):
                    self.assertIsNone(adapter.get(identity))
                self.assertEqual(adapter.corrupt, 1)

    def test_typed_owner_oversize_fails_closed_and_small_budget_skips_before_host_copy(self):
        import torch

        for name, fixture in self.fixtures():
            with self.subTest(adapter=name), BundleStore(fixture.root / 'owner', 100000) as store:
                adapter = fixture.adapter(store)
                identity = fixture.identity(adapter)
                self.assertTrue(self.put(name, fixture, adapter, identity))
                owner = store._entry(identity) / 'owner.json'
                owner.write_bytes(owner.read_bytes() + b' ' * 2048)
                before = owner.read_bytes()
                with patch('corridorkey.recovery_store.MAX_OWNER_RECORD_BYTES', 1024), self.assertRaises(CacheInvalid):
                    adapter.get(identity)
                self.assertEqual(owner.read_bytes(), before)
                self.assertTrue(store._entry(identity).exists())
            with BundleStore(fixture.root / 'quota', 256) as store:
                adapter = fixture.adapter(store)
                identity = fixture.identity(adapter)
                with patch.object(torch.Tensor, 'to', side_effect=AssertionError('quota rejection must precede host copy')):
                    self.assertFalse(self.put(name, fixture, adapter, identity))
                self.assertEqual(adapter.budget_rejected, 1)
                self.assertEqual(list((store.root / 'staging').iterdir()), [])

    def test_mixed_four_file_entry_and_large_manifest_remain_usable(self):
        from tests.test_durable_rvm import DurableRvmTests

        fixture = DurableRvmTests()
        fixture.setUp()
        self.addCleanup(fixture.folder.cleanup)
        with BundleStore(fixture.root / 'mixed', 8 * 1024**2) as store:
            source = fixture.root / 'source.bin'
            source.write_bytes(b'preserve mixed cache')
            generic = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 'large'}))
            manifest = store.publish(generic, {'one.bin': source, 'two.bin': source}, {'padding': 'x' * (4 * 1024**2 + 128)})
            self.assertGreater((store._entry(generic) / 'complete.json').stat().st_size, 4 * 1024**2)
            self.assertEqual(len(list(store._entry(generic).iterdir())), 4)
            adapter = fixture.adapter(store)
            identity = fixture.identity(adapter)
            self.assertTrue(adapter.put(identity, fixture.features, fixture.state))
            self.assertIsNotNone(adapter.get(identity))
            self.assertEqual(store.lookup(generic), manifest)
