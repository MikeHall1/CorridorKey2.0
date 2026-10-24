"""Conservative raw-prediction reuse excludes exactly seven final coverage/composite controls."""
import copy
import json
import unittest
from typing import Any

from corridorkey.config import InferenceConfig
from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.stage2_identity import Stage2FinishingContext, stage2_finishing_context

FINAL_FIELDS = frozenset({'alpha_expand_radius', 'alpha_feather_radius', 'despill_strength',
                          'despill_screen', 'despill_space', 'despill_preserve_luma', 'despill_edges_only'})


class Stage2FinishingIdentityTests(unittest.TestCase):
    def setUp(self):
        self.data = {'schema_version': 1, 'kind': 'whole_job_exr_replay',
                     'settings': json.loads(canonical(InferenceConfig())),
                     'source': {'sha256': 'a'}, 'hints': {'first': None, 'full': None}, 'corrections': None,
                     'assets': {}, 'environment': {'runtime': {'synthetic': True}},
                     'implementation': {'sha256': 'b'}, 'writer': {'dtype': 'float'},
                     'profile_stages': False, 'scope': 'whole job'}

    def identity(self, data):
        return ReplayIdentity(canonical(data))

    def test_only_finishing_settings_are_excluded(self):
        original = self.identity(self.data)
        context = stage2_finishing_context(original)
        for field in FINAL_FIELDS:
            changed = copy.deepcopy(self.data)
            changed['settings'][field] = 3
            full = self.identity(changed)
            self.assertNotEqual(full.key, original.key)
            self.assertEqual(stage2_finishing_context(full), context)
        self.assertNotEqual(context.key, original.key)
        self.assertNotIn(original.key, context.payload)
        self.assertEqual(context.to_dict()['kind'], 'stage2_finishing_context')
        self.assertEqual(context.to_dict()['schema_version'], 5)

    def test_every_other_admitted_dependency_remains(self):
        original = stage2_finishing_context(self.identity(self.data))
        for field in self.data.keys() - {'kind', 'schema_version', 'settings'}:
            changed = copy.deepcopy(self.data)
            value = changed[field]
            changed[field] = ({**value, 'changed': True} if isinstance(value, dict) else
                              not value if isinstance(value, bool) else
                              value + 'changed' if isinstance(value, str) else {'changed': True})
            if field == 'corrections':
                with self.assertRaises(ValueError):
                    stage2_finishing_context(self.identity(changed))
            else:
                self.assertNotEqual(stage2_finishing_context(self.identity(changed)), original, field)
        for field in self.data['settings'].keys() - FINAL_FIELDS:
            changed = copy.deepcopy(self.data)
            changed['settings'][field] = {'changed': True}
            if field == 'corrections_path':
                with self.assertRaises(ValueError):stage2_finishing_context(self.identity(changed))
            else:
                self.assertNotEqual(stage2_finishing_context(self.identity(changed)), original, field)

    def test_context_rejects_incomplete_and_wrong_domain_records(self):
        context = stage2_finishing_context(self.identity(self.data)).to_dict()
        for field in context:
            data = dict(context)
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                Stage2FinishingContext(canonical(data))
        for change in ({'kind': 'whole_job_exr_replay'}, {'settings': []}, {'hints': {}},
                       {'profile_stages': 1}, {'scope': None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Stage2FinishingContext(canonical({**context, **change}))
        with self.assertRaises(ValueError):
            Stage2FinishingContext(canonical({**context, 'settings': {'alpha_expand_radius': 0}}))

    def test_explicit_parent_instance_and_complete_finishing_inventory(self):
        with self.assertRaises(TypeError):
            invalid: Any = ReplayIdentity
            stage2_finishing_context(invalid)
        for field in FINAL_FIELDS:
            changed = copy.deepcopy(self.data)
            del changed['settings'][field]
            with self.assertRaises(ValueError):
                stage2_finishing_context(self.identity(changed))


def legacy_stage2_context(parent):
    """Read the preserved v2-v4 Stage2 contract via unchanged upstream policy."""
    from corridorkey.recovery_identity import upstream_finishing_context
    data = upstream_finishing_context(parent).to_dict()
    data['kind'] = 'stage2_finishing_context'
    del data['reuse_scope']
    return Stage2FinishingContext(canonical(data))
