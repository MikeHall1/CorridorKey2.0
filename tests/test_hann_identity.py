import unittest

from corridorkey.hann_identity import HannWindowIdentity, hann_window_identity
from corridorkey.recovery_identity import ReplayIdentity, canonical


class HannIdentityTests(unittest.TestCase):
    def setUp(self):
        self.parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay'}))
        def tensor(shape):
            return {'shape': shape, 'dtype': 'torch.bfloat16', 'sha256': 'a' * 64}
        self.inputs = {'rgb': tensor([1, 2, 3, 128, 128]), 'hint': tensor([1, 2, 1, 128, 128]),
                       'hint_quality': tensor([1, 2]), 'moge': tensor([1, 2, 4, 2, 2]),
                       'cradio': None, 'rvm': tensor([1, 2, 4, 2, 2])}
        self.values = {'schedule_sha256': 'b' * 64, 'incoming_replay_sha256': 'c' * 64,
                       'index': 0, 'start': 0, 'stop': 2, 'next_start': 1,
                       'segment': (0, 8), 'carry': True, 'd_16': 4, 'inputs': self.inputs}

    def test_schedule_cursor_state_inputs_and_parent_invalidate(self):
        first = hann_window_identity(self.parent, **self.values)
        for changes in ({'schedule_sha256': 'd' * 64}, {'incoming_replay_sha256': 'e' * 64},
                        {'index': 1}, {'next_start': 2}, {'segment': (0, 9)}, {'carry': False}, {'d_16': 8},
                        {'inputs': {**self.inputs, 'rvm': None}},
                        {'inputs': {**self.inputs, 'hint': {**self.inputs['hint'], 'sha256': 'f' * 64}}}):
            self.assertNotEqual(first.key, hann_window_identity(self.parent, **{**self.values, **changes}).key)
        other = ReplayIdentity(canonical({**self.parent.to_dict(), 'different': True}))
        self.assertNotEqual(first.key, hann_window_identity(other, **self.values).key)
        snapshot = first.to_dict()
        snapshot['inputs']['rgb']['shape'][1] = 9
        self.assertEqual(first.to_dict()['inputs']['rgb']['shape'][1], 2)

    def test_types_inventory_and_shapes_are_strict(self):
        for changes in ({'index': True}, {'next_start': -1}, {'stop': 0}, {'carry': 1}, {'d_16': False},
                        {'segment': (1, 8)}, {'incoming_replay_sha256': '../escape'}, {'inputs': {}},
                        {'inputs': {**self.inputs, 'hint': None}},
                        {'inputs': {**self.inputs, 'moge': {**self.inputs['moge'], 'shape': [1, 2, 4, 1, 1]}}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                hann_window_identity(self.parent, **{**self.values, **changes})
        data = hann_window_identity(self.parent, **self.values).to_dict()
        for changes in ({'schema_version': True}, {'kind': 'rvm_chronological_chunk'}, {'unknown': 0}):
            with self.assertRaises(ValueError):
                HannWindowIdentity(canonical({**data, **changes}))
