import tempfile
import unittest
from pathlib import Path

from corridorkey.recovery_identity import ReplayIdentity, canonical, feature_identity
from corridorkey.recovery_store import BundleStore, CacheInvalid


class FeatureIdentityTests(unittest.TestCase):
    def parent(self, value='one'):
        return ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': value}))

    def frame(self, **changes):
        args = {'parent': self.parent(), 'frame': 0, 'proxy_sha256': 'a' * 64, 'input_shape': (3, 64, 64),
                'channels': {'moge': 2, 'cradio': 0}, 'grid_hw': (1, 1)}
        return feature_identity(**{**args, **changes})

    def test_frame_schema_pixel_parent_geometry_and_channels_invalidate(self):
        first = self.frame()
        self.assertEqual(first, self.frame(channels={'moge': 2}))
        for change in ({'frame': 1}, {'proxy_sha256': 'b' * 64}, {'parent': self.parent('other')},
                       {'input_shape': (3, 128, 64)}, {'grid_hw': (2, 1)}, {'channels': {'moge': 3}}):
            with self.subTest(change=change):
                self.assertNotEqual(first.key, self.frame(**change).key)
        data = first.to_dict()
        data['channels']['moge'] = 9
        self.assertEqual(first.to_dict()['channels'], {'moge': 2})

    def test_invalid_frame_stage_dtype_geometry_reject(self):
        for change in ({'frame': True}, {'proxy_sha256': 'wrong'}, {'channels': {'rvm': 2, 'moge': 1}},
                       {'channels': {'moge': False}}, {'channels': {'moge': 0}}, {'grid_hw': (0, 1)},
                       {'input_shape': (4, 64, 64)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.frame(**change)

    def test_generic_store_retains_distinct_typed_content_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'fixture.npy'
            source.write_bytes(b'opaque generic-store fixture')
            with BundleStore(root / 'cache', 10000) as store:
                identity = self.frame()
                record = store.publish(identity, {'moge.npy': source}, {'stage': 'iid'})
                self.assertEqual(store.lookup(identity), record)
                self.assertEqual(record['identity']['kind'], 'iid_foundation_frame')
                self.assertIsNone(store.lookup(self.frame(frame=1)))

    def test_store_rejects_forged_escaping_or_mismatched_identity(self):
        class BadIdentity:
            key = '../outside'

            def to_dict(self):
                return {}

        with tempfile.TemporaryDirectory() as folder, BundleStore(Path(folder) / 'cache', 10000) as store:
            with self.assertRaises(CacheInvalid):
                store.lookup(BadIdentity())
            with self.assertRaises(CacheInvalid):
                store.discard(BadIdentity())
