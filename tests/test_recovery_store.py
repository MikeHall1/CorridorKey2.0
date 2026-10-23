import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from corridorkey.control import CancelledError
from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recovery_store import BundleStore, CacheCapacity, CacheInvalid


def identity(name):
    return ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': name}))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'known artifact bytes' * 50)
        self.cache = self.root / 'cache'

    def test_verified_roundtrip_and_fresh_destination_only(self):
        key = identity('one')
        with BundleStore(self.cache, 10000) as store:
            manifest = store.publish(key, {'alpha/frame.exr': self.source}, {'status': 'complete'})
            restored = self.root / 'restored'
            self.assertEqual(store.restore(key, restored), manifest)
            self.assertEqual((restored / 'alpha/frame.exr').read_bytes(), self.source.read_bytes())
            self.assertTrue((restored / '.replay-complete.json').is_file())
            with self.assertRaises(FileExistsError):
                store.restore(key, restored)
            (self.cache / 'entries' / key.key / 'payload-000000.bin').write_bytes(b'corrupt')
            with self.assertRaisesRegex(CacheInvalid, 'checksum|size'):
                store.restore(key, self.root / 'not-created')
            self.assertFalse((self.root / 'not-created').exists())

    def test_budget_evicts_owned_entries_before_new_allocation(self):
        first, second = identity('one'), identity('two')
        with BundleStore(self.cache, 2500) as store:
            store.publish(first, {'alpha.exr': self.source}, {})
            store.publish(second, {'alpha.exr': self.source}, {})
            self.assertIsNone(store.lookup(first))
            self.assertIsNotNone(store.lookup(second))
            used = sum(p.stat().st_size for p in (self.cache / 'entries').rglob('*') if p.is_file())
            self.assertLessEqual(used, 2500)
        with BundleStore(self.cache, 0) as store:
            self.assertIsNone(store.lookup(second))
            with self.assertRaises(CacheCapacity):
                store.publish(first, {'alpha.exr': self.source}, {})

    def test_cancellation_never_publishes_partial_stage(self):
        key = identity('cancel')
        with BundleStore(self.cache, 10000) as store:
            def check():
                if any(path.name.startswith('payload-') for path in (self.cache / 'staging').rglob('*')):
                    raise CancelledError('cancel during copy')

            with self.assertRaisesRegex(CancelledError, 'cancel during copy'):
                store.publish(key, {'alpha.exr': self.source}, {}, check=check)
            self.assertIsNone(store.lookup(key))
            self.assertEqual(list((self.cache / 'staging').iterdir()), [])
        self.assertTrue(self.source.exists())

    def test_expected_receipt_hash_binds_publication_and_existing_entry(self):
        import hashlib
        from unittest.mock import patch
        key = identity('receipt')
        expected = {'alpha.exr': {'bytes': self.source.stat().st_size,
                                  'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}}
        with BundleStore(self.cache, 10000) as store:
            self.source.write_bytes(b'x' * self.source.stat().st_size)
            with self.assertRaisesRegex(CacheInvalid, 'during copy'):
                store.publish(key, {'alpha.exr': self.source}, {}, expected_inventory=expected)
            self.assertIsNone(store.lookup(key))
            expected['alpha.exr']['sha256'] = hashlib.sha256(self.source.read_bytes()).hexdigest()
            with patch('corridorkey.recovery_store.file_sha256', side_effect=AssertionError('receipt must supply hash')):
                store.publish(key, {'alpha.exr': self.source}, {}, expected_inventory=expected)
            changed = {'alpha.exr': {**expected['alpha.exr'], 'sha256': '0' * 64}}
            with self.assertRaisesRegex(CacheInvalid, 'existing sealed bytes'):
                store.publish(key, {'alpha.exr': self.source}, {}, expected_inventory=changed)
            with self.assertRaises(CacheInvalid):
                store.publish(key, {'alpha.exr': self.source}, {}, expected_inventory={'other': expected['alpha.exr']})

    def test_optional_structural_policy_counts_mixed_entries_and_transient_files(self):
        from unittest.mock import patch

        from corridorkey import cache_maintenance
        with BundleStore(self.cache, 100000, structural_admission=True) as store:
            store.publish(identity('ordinary-adapter'), {'one.bin': self.source}, {})
            with patch.object(cache_maintenance, 'MAX_ENTRIES', 1), self.assertRaises(CacheCapacity):
                store.publish(identity('prefix-adapter'), {'alpha.exr': self.source}, {})
            self.assertIsNotNone(store.lookup(identity('ordinary-adapter')))
            with patch.object(cache_maintenance, 'MAX_FILES', 8), self.assertRaises(CacheCapacity):
                store.publish(identity('files'), {'alpha.exr': self.source}, {})
            store.publish(identity('prefix-adapter'), {'alpha.exr': self.source}, {})
            self.assertEqual(cache_maintenance._scan(store.root, None)[0]['entry_count'], 2)
        with BundleStore(self.root / 'legacy', 100000) as store, patch.object(cache_maintenance, 'MAX_ENTRIES', 1):
            store.publish(identity('one'), {'one.bin': self.source}, {})
            store.publish(identity('two'), {'two.bin': self.source}, {})
            self.assertEqual(len(list((store.root / 'entries').iterdir())), 2)

    def test_optional_structural_metadata_guard_counts_existing_json_and_private_stage(self):
        from unittest.mock import patch

        from corridorkey import cache_maintenance
        with BundleStore(self.cache, 100000, structural_admission=True) as store:
            store.publish(identity('first'), {'first.bin': self.source}, {'details': 'x' * 500})
            with store.private_stage() as folder:
                staged = folder / 'payload-000000.bin'
                staged.write_bytes(b'staged tensor bytes')
                current, _ = cache_maintenance._scan(store.root, None)
                self.assertEqual(current['staging_entries'], 1)
                with patch.object(cache_maintenance, 'MAX_METADATA_BYTES', 1800), self.assertRaises(CacheCapacity):
                    store.publish(identity('second'), {'second.bin': staged}, {'details': 'x' * 1000})
            self.assertEqual(list((store.root / 'staging').iterdir()), [])
            (store.root / 'entries' / identity('first').key / 'unowned.txt').write_text('preserve')
            with self.assertRaises(CacheInvalid):
                store.publish(identity('blocked'), {'blocked.bin': self.source}, {})
            self.assertTrue((store.root / 'entries' / identity('first').key / 'unowned.txt').exists())

    def test_structural_admission_reserves_future_report_and_atomic_update_slots(self):
        from unittest.mock import patch

        from corridorkey import cache_maintenance
        with BundleStore(self.cache, 100000, structural_admission=True) as store:
            # Two existing root files + three new entry files + two spare slots.
            with patch.object(cache_maintenance, 'MAX_FILES', 6), self.assertRaises(CacheCapacity):
                store.publish(identity('reserve'), {'alpha.exr': self.source}, {})
            self.assertEqual(list((store.root / 'entries').iterdir()), [])
            with patch.object(cache_maintenance, 'MAX_FILES', 7):
                store.publish(identity('reserve'), {'alpha.exr': self.source}, {})
                _, snapshot = store._structural_scan(None)
                self.assertEqual(len(snapshot['files']), 5)
                self.assertEqual(cache_maintenance.MAX_FILES - len(snapshot['files']), 2)

    def test_optional_startup_scan_fails_before_pruning_then_admits_owned_crash_stage(self):
        from unittest.mock import patch

        from corridorkey import cache_maintenance
        with BundleStore(self.cache, 100000) as store:
            stage = store.root / 'staging' / f'entry-{store.owner}-fixture'
            stage.mkdir()
            (stage / 'payload-000000.bin').write_bytes(b'incomplete owned tensor')
            (stage / '.record-fixture').write_bytes(b'{')
        guarded = BundleStore(self.cache, 100000, structural_admission=True)
        with patch.object(cache_maintenance, 'MAX_FILES', 1), self.assertRaises(CacheCapacity), guarded:
            self.fail('bounded startup must refuse before pruning')
        self.assertTrue(stage.exists())
        self.assertIsNone(guarded.connection)
        with BundleStore(self.cache, 100000, structural_admission=True):
            self.assertFalse(stage.exists())

    def test_process_crash_releases_owner_and_partial_metadata_is_recovered(self):
        with BundleStore(self.cache, 10000):
            pass
        code = '''import os,sys
from pathlib import Path
from corridorkey.recovery_store import BundleStore
with BundleStore(sys.argv[1],10000) as store:
    stage=store.root/'staging'/('entry-'+store.owner+'-crashfixture')
    stage.mkdir()
    (stage/'.record-partial').write_bytes(b'{"owner":')
    os._exit(73)
'''
        child = subprocess.run([sys.executable, '-c', code, str(self.cache)], check=False,
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 73, child.stderr)
        with BundleStore(self.cache, 10000):
            self.assertEqual(list((self.cache / 'staging').iterdir()), [])

    def test_exclusive_owner_and_unowned_directory_protection(self):
        other = self.root / 'footage'
        other.mkdir()
        (other / 'important.bin').write_bytes(b'user footage')
        with self.assertRaisesRegex(ValueError, 'empty or already owned'):
            BundleStore(other, 10000)
        self.assertEqual((other / 'important.bin').read_bytes(), b'user footage')
        with BundleStore(self.cache, 10000), self.assertRaisesRegex(RuntimeError, 'busy'), \
                BundleStore(self.cache, 10000):
            self.fail('second workspace owner must not enter')

    def test_path_escape_and_unowned_content_are_never_overwritten_or_deleted(self):
        key = identity('paths')
        with BundleStore(self.cache, 10000) as store:
            for name in ('../outside.exr', '/absolute.exr', 'alpha/../outside.exr', 'a\\b', '.replay-complete.json'):
                with self.subTest(name=name), self.assertRaises(CacheInvalid):
                    store.publish(key, {name: self.source}, {})
            store.publish(key, {'alpha.exr': self.source}, {})
            entry = self.cache / 'entries' / key.key
            (entry / 'important-user-file.bin').write_bytes(b'preserve me')
            with self.assertRaisesRegex(CacheInvalid, 'unowned'):
                store.discard(key)
            self.assertEqual((entry / 'important-user-file.bin').read_bytes(), b'preserve me')
            (entry / 'important-user-file.bin').unlink()
            record = json.loads((entry / 'complete.json').read_text())
            record['files'][0]['name'] = '../outside.exr'
            (entry / 'complete.json').write_text(json.dumps(record))
            with self.assertRaises(CacheInvalid):
                store.restore(key, self.root / 'rejected')
            self.assertFalse((self.root / 'outside.exr').exists())

    def test_completion_metadata_and_portable_inventory_are_checked(self):
        key = identity('metadata')
        with BundleStore(self.cache, 10000) as store:
            store.publish(key, {'alpha.exr': self.source}, {'status': 'complete', 'note': 'original'})
            path = self.cache / 'entries' / key.key / 'complete.json'
            data = json.loads(path.read_text())
            data['origin_record']['note'] = 'changed'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(CacheInvalid, 'metadata checksum'):
                store.lookup(key)
            store.discard(key)
            for names in (['a', 'a/b'], ['A.exr', 'a.exr'], ['.REPLAY-COMPLETE.JSON'], ['a.'], ['a ']):
                with self.subTest(names=names), self.assertRaises(CacheInvalid):
                    store.publish(key, {name: self.source for name in names}, {})
            self.assertIsNone(store.lookup(key))

    def test_size_race_and_disk_full_never_publish_complete_entry(self):
        from unittest.mock import patch

        from corridorkey import recovery_store

        key = identity('faults')
        original_hash = recovery_store.file_sha256
        original_write = recovery_store.write_record

        def changed_size(path, **kwargs):
            self.source.write_bytes(b'changed after inventory stat')
            return original_hash(path, **kwargs)

        def disk_full(path, value):
            if Path(path).name == 'complete.json':
                raise OSError('fixture disk full')
            return original_write(path, value)

        with BundleStore(self.cache, 10000) as store:
            with patch.object(recovery_store, 'file_sha256', side_effect=changed_size), \
                    self.assertRaisesRegex(CacheInvalid, 'during copy'):
                store.publish(key, {'alpha.exr': self.source}, {})
            self.assertIsNone(store.lookup(key))
            with patch.object(recovery_store, 'write_record', side_effect=disk_full), \
                    self.assertRaisesRegex(OSError, 'fixture disk full'):
                store.publish(key, {'alpha.exr': self.source}, {})
            self.assertIsNone(store.lookup(key))
            self.assertEqual(list((self.cache / 'staging').iterdir()), [])
