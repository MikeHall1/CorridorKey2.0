"""Upstream reuse has its own typed, conservative finishing-edit domain."""
import copy
import unittest
from typing import Any
from unittest.mock import patch

from tests import test_stage2_finishing_identity as fixtures

DESPILL_EDITS = ({'despill_strength': .2}, {'despill_screen': 'blue'}, {'despill_space': 'legacy'},
                 {'despill_preserve_luma': True}, {'despill_edges_only': True},
                 {'despill_strength': .2, 'despill_screen': 'blue', 'despill_space': 'linear',
                  'despill_preserve_luma': True, 'despill_edges_only': True})


class UpstreamFinishingIdentityTests(unittest.TestCase):
    def test_only_seven_final_controls_are_excluded_from_complete_context(self):
        from corridorkey.recovery_identity import (
            UpstreamFinishingContext,
            upstream_finishing_context,
        )
        fixture = fixtures.Stage2FinishingIdentityTests()
        fixture.setUp()
        base = fixture.identity(fixture.data)
        context = upstream_finishing_context(base)
        self.assertIsInstance(context, UpstreamFinishingContext)
        self.assertNotEqual(context.key, base.key)
        self.assertNotIn(base.key, context.payload)
        for name in fixture.data['settings']:
            changed = copy.deepcopy(fixture.data)
            changed['settings'][name] = {'changed': True}
            candidate = upstream_finishing_context(fixture.identity(changed))
            self.assertEqual(candidate == context, name in fixtures.FINAL_FIELDS, name)
        for name in fixture.data.keys() - {'kind', 'schema_version', 'settings'}:
            changed = copy.deepcopy(fixture.data)
            value = changed[name]
            changed[name] = ({**value, 'changed': True} if isinstance(value, dict) else
                             not value if isinstance(value, bool) else
                             value + 'changed' if isinstance(value, str) else {'changed': True})
            if name == 'corrections':
                with self.assertRaises(ValueError):
                    upstream_finishing_context(fixture.identity(changed))
            else:
                self.assertNotEqual(upstream_finishing_context(fixture.identity(changed)), context, name)
        from corridorkey.stage2_identity import stage2_finishing_context
        self.assertNotEqual(context.key, stage2_finishing_context(base).key)
        for name in context.to_dict():
            data = context.to_dict()
            del data[name]
            from corridorkey.recovery_identity import canonical
            with self.subTest(missing=name), self.assertRaises(ValueError):
                UpstreamFinishingContext(canonical(data))
        with self.assertRaises(TypeError):
            bad: Any = base.to_dict()
            upstream_finishing_context(bad)

    def test_public_admission_rejects_before_io(self):
        from corridorkey.recovery import run_recoverable
        from corridorkey.worker_client import run_worker_batch
        invalid_values: tuple[Any, ...] = (True, 1, None, 'true')
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_recoverable([object()], workspace='/unused', max_cache_bytes=0,
                    assets={}, environment={}, reuse_upstream_finishing_edits=value)
            with (patch('corridorkey.batch.load_jobs', side_effect=AssertionError('must not read batch')),
                  self.assertRaises(ValueError)):
                run_worker_batch('/unused', '/report', reuse_upstream_finishing_edits=value)

    def test_typed_constructor_domains_and_exact_call_dependencies(self):
        from corridorkey.hann_identity import hann_window_identity
        from corridorkey.recovery_identity import (
            feature_identity,
            upstream_finishing_context,
        )
        from corridorkey.recurrent_identity import rvm_chunk_identity
        fixture = fixtures.Stage2FinishingIdentityTests()
        fixture.setUp()
        parent = upstream_finishing_context(fixture.identity(fixture.data))
        iid = {'frame': 0, 'proxy_sha256': 'a' * 64, 'input_shape': (3, 64, 64),
               'channels': {'moge': 1}, 'grid_hw': (1, 1)}
        rvm = {'shot_start': 0, 'start': 2, 'stop': 4, 'proxy_sha256': 'a' * 64,
               'input_shape': (2, 3, 64, 64), 'input_dtype': 'torch.float32',
               'incoming_state_sha256': 'b' * 64, 'channels': 1, 'grid_hw': (1, 1)}
        inputs: dict[str, Any] = {name: {'sha256': 'a' * 64, 'dtype': 'torch.float32', 'shape': shape}
                  for name, shape in {'rgb': [1, 2, 3, 64, 64], 'hint': [1, 2, 1, 64, 64],
                      'hint_quality': [1, 2], 'moge': [1, 2, 1, 1, 1]}.items()}
        inputs.update(cradio=None, rvm=None)
        hann = {'schedule_sha256': 'a' * 64, 'incoming_replay_sha256': 'b' * 64,
                'index': 0, 'start': 0, 'stop': 2, 'next_start': 2, 'segment': (0, 2),
                'carry': True, 'd_16': 1, 'inputs': inputs}
        cases: Any = (
                (feature_identity, iid, ({'frame': 1}, {'proxy_sha256': 'c' * 64},
                     {'input_dtype': 'torch.bfloat16'}, {'input_shape': (3, 64, 128)})),
                (rvm_chunk_identity, rvm, ({'proxy_sha256': 'c' * 64},
                     {'incoming_state_sha256': 'c' * 64}, {'input_dtype': 'torch.bfloat16'}, {'shot_start': 1})),
                (hann_window_identity, hann, ({'schedule_sha256': 'c' * 64},
                     {'incoming_replay_sha256': 'c' * 64}, {'carry': False})))
        for constructor, arguments, mutations in cases:
            original = constructor(parent, **arguments)
            for change in mutations:
                self.assertNotEqual(constructor(parent, **{**arguments, **change}).key, original.key)
            for invalid in (parent.to_dict(), type(parent), type('Duck', (), {'key': parent.key})()):
                with self.assertRaises(TypeError):
                    constructor(invalid, **arguments)
