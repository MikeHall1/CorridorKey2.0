"""Receipt transport tests use opaque producer files; runtime tests verify real EXR pixels."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corridorkey.control import CancelledError
from corridorkey.durable_writer_prefix import DurableWriterPrefix, WriterFrameIdentity
from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recovery_store import BundleStore, CacheInvalid


class DurableWriterPrefixTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
            'assets': {}, 'environment': {'runtime': {'synthetic': True}},
            'settings': {'outputs': ['alpha', 'fg'], 'output_format': 'exr', 'exr_dtype': 'float'}}))
        self.outputs = {role: {'directory': role, 'frame_name': role,
                              'contract': {'format': 'exr', 'channels': channels, 'dtype': 'float',
                                           'rgb_representation': None if role == 'alpha' else 'scene_linear'}}
                        for role, channels in (('alpha', 1), ('fg', 3))}
        self.valid_context = True
        self.original = self.root / 'original'
        self.original.mkdir()

    def context(self):
        if not self.valid_context:
            raise CacheInvalid('source/config/model context changed')

    def adapter(self, store, path, *, resume=False, parent=None, check=None):
        adapter = DurableWriterPrefix(store, parent or self.parent, validate_context=self.context, check=check, resume=resume)
        adapter.bind(None)
        path.mkdir(exist_ok=True)
        count = adapter.prepare(path, frame_count=3, native_hw=(2, 3), fps=24, outputs=self.outputs, resume=resume)
        return adapter, count

    def write(self, path, frame):
        result = {}
        for role in self.outputs:
            directory = path / role
            directory.mkdir(exist_ok=True)
            target = directory / f'{role}_{frame:06d}.exr'
            data = f'explicit opaque {role} producer frame {frame}\n'.encode() * 10
            target.write_bytes(data)
            result[target.relative_to(path).as_posix()] = data
        return result

    def test_all_output_roundtrip_only_explicit_resume_and_fresh_copies(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, count = self.adapter(store, self.original)
            self.assertEqual(count, 0)
            original = {**self.write(self.original, 0), **self.write(self.original, 1)}
            self.assertTrue(adapter.checkpoint((0, 1), completed={'alpha': 2, 'fg': 2}))
            fresh = self.root / 'fresh'
            resumed, count = self.adapter(store, fresh, resume=True)
            self.assertEqual(count, 2)
            self.assertEqual(resumed.snapshot()['restored_files'], 4)
            self.assertFalse((fresh / '.replay-complete.json').exists())
            for name, data in original.items():
                self.assertEqual((fresh / name).read_bytes(), data)
                self.assertEqual((self.original / name).read_bytes(), data)
                self.assertEqual((fresh / name).stat().st_nlink, 1)
                self.assertNotEqual((fresh / name).stat().st_ino, (self.original / name).stat().st_ino)
            _, count = self.adapter(store, self.root / 'not-resume')
            self.assertEqual(count, 0)
            self.assertEqual(list((self.root / 'not-resume').iterdir()), [])

    def test_missing_frame_or_corrupt_output_stops_common_prefix(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            for frame in range(3):
                self.write(self.original, frame)
            adapter.checkpoint((0, 1, 2), completed={'alpha': 3, 'fg': 3})
            store.discard(adapter._identity(1))
            _, count = self.adapter(store, self.root / 'gap', resume=True)
            self.assertEqual(count, 1)
            path = store.root / 'entries' / adapter._identity(0).key / 'payload-000000.bin'
            path.write_bytes(b'corrupt')
            resumed, count = self.adapter(store, self.root / 'corrupt', resume=True)
            self.assertEqual(count, 0)
            self.assertEqual(resumed.snapshot()['corrupt_entries'], 1)
            self.assertTrue((self.original / 'alpha/alpha_000000.exr').exists())

    def test_changed_parent_or_render_contract_never_reuses(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            data = self.parent.to_dict()
            data['corrections'] = 'new revision identity'
            changed = ReplayIdentity(canonical(data))
            _, count = self.adapter(store, self.root / 'changed-parent', resume=True, parent=changed)
            self.assertEqual(count, 0)
            self.outputs['fg']['contract']['foreground_mode'] = 'changed render policy'
            _, count = self.adapter(store, self.root / 'changed-render', resume=True)
            self.assertEqual(count, 0)

    def test_zero_quota_skips_without_hash_or_failure_for_later_frames(self):
        with BundleStore(self.root / 'cache', 0) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            with patch.object(adapter, '_receipt', side_effect=AssertionError('must not hash')):
                self.assertFalse(adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1}))
                self.assertFalse(adapter.checkpoint((1,), completed={'alpha': 2, 'fg': 2}))
            self.assertEqual(adapter.snapshot()['receipt_bytes_hashed'], 0)
            self.assertTrue(adapter.snapshot()['checkpoint_disabled'])
            self.assertEqual(list((store.root / 'entries').iterdir()), [])

    def test_bad_watermarks_or_source_context_refuse_before_receipt(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            with patch.object(adapter, '_receipt', side_effect=AssertionError('must not hash')):
                for completed in ({'alpha': 1}, {'alpha': 1, 'fg': 0}, {'alpha': True, 'fg': 1}):
                    with self.assertRaises(CacheInvalid):
                        adapter.checkpoint((0,), completed=completed)
                self.valid_context = False
                with self.assertRaisesRegex(CacheInvalid, 'context changed'):
                    adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})

    def test_evicted_first_entry_is_not_reported_as_retained_prefix(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            used = sum(p.stat().st_size for p in (store.root / 'entries').rglob('*') if p.is_file())
            store.max_bytes = used + 10
            self.write(self.original, 1)
            adapter.checkpoint((1,), completed={'alpha': 2, 'fg': 2})
            self.assertEqual(adapter.snapshot()['stored_frames'], 2)
            self.assertIsNone(store.lookup(adapter._identity(0)))
            _, count = self.adapter(store, self.root / 'evicted', resume=True)
            self.assertEqual(count, 0)

    def test_receipt_growth_and_publication_mutation_cannot_seal(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            source = self.original / 'alpha/alpha_000000.exr'
            original_read = os.read
            mutated = False

            def growing(descriptor, amount):
                nonlocal mutated
                data = original_read(descriptor, amount)
                if not mutated:
                    mutated = True
                    with source.open('ab') as stream:
                        stream.write(b'extra')
                return data
            with patch('corridorkey.durable_writer_prefix.os.read', side_effect=growing), self.assertRaises(CacheInvalid):
                adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
            self.write(self.original, 0)
            publish = store.publish

            def changed(*args, **kwargs):
                source.write_bytes(b'x' * source.stat().st_size)
                return publish(*args, **kwargs)
            with patch.object(store, 'publish', side_effect=changed), self.assertRaises(CacheInvalid):
                adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
            self.assertEqual(list((store.root / 'staging').iterdir()), [])

    @unittest.skipUnless(hasattr(os, 'link') and hasattr(os, 'symlink'), 'requires local links')
    def test_links_and_unowned_destination_are_refused(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            source = self.original / 'alpha/alpha_000000.exr'
            outside = self.root / 'outside'
            outside.write_bytes(b'preserve external bytes')
            for link in (os.link, os.symlink):
                source.unlink()
                link(outside, source)
                with self.assertRaises(CacheInvalid):
                    adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
                self.assertEqual(outside.read_bytes(), b'preserve external bytes')
            with self.assertRaisesRegex(CacheInvalid, 'fresh and empty'):
                self.adapter(store, self.original, resume=True)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_fifo_output_refuses_without_opening_or_blocking(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            fifo = self.original / 'alpha/alpha_000000.exr'
            fifo.unlink()
            os.mkfifo(fifo)
            with patch('corridorkey.durable_writer_prefix.os.open', side_effect=AssertionError('must not open FIFO')), \
                    self.assertRaises(CacheInvalid):
                adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})

    def test_empty_completed_native_output_is_not_hidden_as_quota_skip(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            (self.original / 'alpha/alpha_000000.exr').write_bytes(b'')
            with self.assertRaisesRegex(CacheInvalid, 'empty'):
                adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            self.assertEqual(adapter.snapshot()['budget_rejected_frames'], 0)
            self.assertEqual(list((store.root / 'entries').iterdir()), [])

    def test_cancelled_restore_reports_partial_file_without_rollback_claim(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            fresh = self.root / 'cancelled'
            fresh.mkdir()
            resumed = DurableWriterPrefix(store, self.parent, validate_context=self.context, resume=True)
            resumed.bind(None)

            def cancel():
                if any(path.is_file() for path in fresh.rglob('*')):
                    raise CancelledError('stop during restore copy')
            resumed.check = cancel
            with self.assertRaises(CancelledError):
                resumed.prepare(fresh, frame_count=3, native_hw=(2, 3), fps=24, outputs=self.outputs, resume=True)
            self.assertEqual(resumed.snapshot()['restored_prefix_frames'], 0)
            self.assertIsNotNone(resumed.snapshot()['partial_restore'])
            self.assertTrue(list(fresh.rglob('*.exr')))

    def test_late_directory_sync_failure_retains_committed_publication_count(self):
        from corridorkey import recovery_store
        with BundleStore(self.root / 'cache', 100000) as store:
            adapter, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            sync = recovery_store._sync_directory

            def late(path):
                if Path(path) == store.root / 'entries':
                    raise OSError('injected post-rename directory sync failure')
                return sync(path)
            with patch.object(recovery_store, '_sync_directory', side_effect=late), self.assertRaises(OSError):
                adapter.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            self.assertEqual(adapter.snapshot()['stored_frames'], 1)
            self.assertIsNotNone(store.lookup(adapter._identity(0)))

    def test_restored_prefix_fence_detects_same_length_edit_missing_and_replacement(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            producer, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            producer.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            original_bytes = (self.original / 'alpha/alpha_000000.exr').read_bytes()
            for mutation in ('same-length', 'missing', 'replacement'):
                fresh = self.root / mutation
                resumed, count = self.adapter(store, fresh, resume=True)
                self.assertEqual(count, 1)
                with patch('corridorkey.durable_writer_prefix.os.fsync', side_effect=AssertionError('verification is read-only')):
                    expected = resumed.verify_restored_prefix()
                self.assertEqual(expected['alpha/alpha_000000.exr']['sha256'], hashlib.sha256(original_bytes).hexdigest())
                expected['alpha/alpha_000000.exr']['sha256'] = '0' * 64
                self.assertNotEqual(resumed.restored_inventory['alpha/alpha_000000.exr']['sha256'], '0' * 64)
                self.assertTrue(resumed.snapshot()['restored_prefix_verified'])
                target = fresh / 'alpha/alpha_000000.exr'
                if mutation == 'same-length':
                    target.write_bytes(b'x' * len(original_bytes))
                elif mutation == 'missing':
                    target.unlink()
                else:
                    target.rename(target.with_suffix('.old'))
                    target.write_bytes(b'y' * len(original_bytes))
                with self.assertRaises(CacheInvalid):
                    resumed.verify_restored_prefix()
                self.assertFalse(resumed.snapshot()['restored_prefix_verified'])
                self.assertEqual(resumed.snapshot()['restored_prefix_frames'], 1)
                self.assertEqual(resumed.snapshot()['restored_files'], 2)
                self.assertEqual((self.original / 'alpha/alpha_000000.exr').read_bytes(), original_bytes)

    def test_restored_prefix_fence_cancel_and_path_identity_race(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            producer, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            producer.checkpoint((0,), completed={'alpha': 1, 'fg': 1})
            fresh = self.root / 'verify'
            resumed, _ = self.adapter(store, fresh, resume=True)
            def cancel():
                raise CancelledError('cancel prefix verification')
            resumed.check = cancel
            with self.assertRaises(CancelledError):
                resumed.verify_restored_prefix()
            self.assertFalse(resumed.snapshot()['restored_prefix_verified'])
            resumed.check = None
            target = fresh / 'alpha/alpha_000000.exr'
            read = os.read
            changed = False
            def replace_during_read(descriptor, amount):
                nonlocal changed
                data = read(descriptor, amount)
                if not changed:
                    changed = True
                    target.rename(target.with_suffix('.old'))
                    target.write_bytes(target.with_suffix('.old').read_bytes())
                return data
            with patch('corridorkey.durable_writer_prefix.os.read', side_effect=replace_during_read), \
                    self.assertRaisesRegex(CacheInvalid, 'changed during receipt'):
                resumed.verify_restored_prefix()

    def test_restore_inventory_cap_stops_before_next_complete_frame_copy(self):
        with BundleStore(self.root / 'cache', 100000) as store:
            producer, _ = self.adapter(store, self.original)
            self.write(self.original, 0)
            self.write(self.original, 1)
            producer.checkpoint((0, 1), completed={'alpha': 2, 'fg': 2})
            with patch('corridorkey.durable_writer_prefix.MAX_RESTORED_FILES', 3):
                resumed, count = self.adapter(store, self.root / 'bounded', resume=True)
            self.assertEqual(count, 1)
            self.assertEqual(resumed.snapshot()['restored_files'], 2)
            self.assertTrue(resumed.snapshot()['restore_inventory_limited'])
            self.assertFalse((self.root / 'bounded/alpha/alpha_000001.exr').exists())
            resumed.verify_restored_prefix()

    def test_identity_layout_and_resume_policy_validation(self):
        with self.assertRaises(CacheInvalid):
            WriterFrameIdentity('{}')
        with BundleStore(self.root / 'cache', 100000) as store:
            with self.assertRaises(ValueError):
                DurableWriterPrefix(store, self.parent, validate_context=self.context, **json.loads('{"resume":1}'))
            adapter = DurableWriterPrefix(store, self.parent, validate_context=self.context)
            adapter.bind(None)
            with self.assertRaises(CacheInvalid):
                adapter.prepare(self.original, frame_count=3, native_hw=(2, 3), fps=24, outputs=self.outputs, resume=True)
            self.outputs['fg']['directory'] = '../outside'
            with self.assertRaises(CacheInvalid):
                adapter.prepare(self.original, frame_count=3, native_hw=(2, 3), fps=24, outputs=self.outputs, resume=False)


if __name__ == '__main__':
    unittest.main()
