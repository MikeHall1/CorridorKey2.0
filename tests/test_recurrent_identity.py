import json
import unittest

from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recurrent_identity import RvmChunkIdentity, rvm_chunk_identity


class RecurrentIdentityTests(unittest.TestCase):
    def setUp(self):
        self.parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 1}))
        self.values = {'shot_start': 0, 'start': 0, 'stop': 2, 'proxy_sha256': 'a' * 64,
                       'input_shape': (2, 3, 128, 128), 'input_dtype': 'torch.bfloat16',
                       'incoming_state_sha256': None, 'channels': 4, 'grid_hw': (2, 2)}

    def test_exact_call_state_and_parent_invalidate(self):
        first = rvm_chunk_identity(self.parent, **self.values)
        values = {**self.values, 'start': 2, 'stop': 4, 'incoming_state_sha256': 'b' * 64}
        second = rvm_chunk_identity(self.parent, **values)
        self.assertNotEqual(first.key, second.key)
        for changes in ({'incoming_state_sha256': 'c' * 64}, {'proxy_sha256': 'd' * 64},
                        {'shot_start': 1}, {'grid_hw': (1, 4)}, {'channels': 8},
                        {'input_dtype': 'torch.float32'}):
            self.assertNotEqual(second.key, rvm_chunk_identity(self.parent, **{**values, **changes}).key)
        other = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay', 'fixture': 2}))
        self.assertNotEqual(first.key, rvm_chunk_identity(other, **self.values).key)
        snapshot = first.to_dict()
        snapshot['input_shape'][0] = 9
        self.assertEqual(first.to_dict()['input_shape'][0], 2)

    def test_range_reset_dtype_schema_and_hash_are_strict(self):
        for changes in ({'start': True}, {'stop': 0}, {'shot_start': -1}, {'channels': False},
                        {'input_shape': (1, 3, 128, 128)}, {'grid_hw': (0, 2)},
                        {'incoming_state_sha256': 'b' * 64}, {'input_dtype': 'torch.float16'},
                        {'start': 1}, {'proxy_sha256': '../escape'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                rvm_chunk_identity(self.parent, **{**self.values, **changes})
        data = rvm_chunk_identity(self.parent, **self.values).to_dict()
        for changes in ({'schema_version': True}, {'kind': 'iid_foundation_frame'}, {'unknown': 0}):
            with self.assertRaises(ValueError):
                RvmChunkIdentity(canonical({**data, **changes}))
        with self.assertRaises(ValueError):
            RvmChunkIdentity(json.dumps(data))

    def test_reset_begins_a_new_chain(self):
        values = {**self.values, 'shot_start': 10, 'start': 10, 'stop': 12}
        identity = rvm_chunk_identity(self.parent, **values)
        self.assertIsNone(identity.to_dict()['incoming_state_sha256'])
        with self.assertRaises(ValueError):
            rvm_chunk_identity(self.parent, **{**values, 'incoming_state_sha256': 'b' * 64})
