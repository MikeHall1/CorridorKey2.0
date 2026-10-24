"""Versioned private final-only policies retain old keys and all other inputs."""
import copy
import hashlib
import unittest
from dataclasses import replace
from typing import Any

from corridorkey.recovery_identity import (
    UpstreamFinishingContext,
    canonical,
    feature_identity,
    upstream_finishing_context,
)
from corridorkey.stage2_identity import (
    Stage2FinishingContext,
    stage2_batch_identity,
    stage2_finishing_context,
)
from tests import test_stage2_finishing_identity as fixtures
from tests.test_stage2_finishing_identity import FINAL_FIELDS


class DespillFinishingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Stage2FinishingIdentityTests()
        self.fixture.setUp()
        self.parent = self.fixture.identity(self.fixture.data)

    def test_legacy_v1_payload_keys_unchanged_and_v2_policy_explicit(self):
        for factory, constructor, kind, expected_key in (
                (stage2_finishing_context, Stage2FinishingContext, 'stage2_finishing_context',
                 '0d7f7ed9a005b2d0b4b2ecbb458a5b0a888ea5038245c56e3b4e8393bbb6b740'),
                (upstream_finishing_context, UpstreamFinishingContext, 'upstream_finishing_context',
                 'c185801f0a99fa2f3975864c22a72069482b527df92b5610859f165d1c39f72e')):
            legacy = self.parent.to_dict()
            legacy['kind'] = kind
            if constructor is UpstreamFinishingContext:
                legacy['reuse_scope'] = 'iid_rvm_hann_raw_inputs'
            legacy['settings'] = {key:value for key,value in legacy['settings'].items()
                                  if key not in {'alpha_expand_radius','alpha_feather_radius'}}
            payload = canonical(legacy)
            old = constructor(payload)
            self.assertEqual(old.key,expected_key)
            self.assertEqual(old.payload,payload)
            self.assertEqual(old.key,hashlib.sha256(payload.encode()).hexdigest())
            current = factory(self.parent)
            self.assertEqual(current.to_dict()['schema_version'],5 if constructor is Stage2FinishingContext else 2)
            self.assertNotEqual(current.key,old.key)
            self.assertEqual(set(self.parent.to_dict()['settings']) - set(current.to_dict()['settings']),FINAL_FIELDS | ({'corrections_path'} if constructor is Stage2FinishingContext else set()))
            for name in FINAL_FIELDS:
                wrong = current.to_dict()
                wrong['settings'][name] = self.fixture.data['settings'][name]
                with self.subTest(kind=kind,field=name), self.assertRaises(ValueError):
                    constructor(canonical(wrong))
            for name in FINAL_FIELDS - {'alpha_expand_radius','alpha_feather_radius'}:
                changed = copy.deepcopy(legacy)
                changed['settings'][name] = {'changed':True}
                self.assertNotEqual(constructor(canonical(changed)).key,old.key)

    def test_factories_need_all_seven_and_wrong_domain_class_unknown_policy_refused(self):
        for factory, constructor in ((stage2_finishing_context,Stage2FinishingContext),
                                     (upstream_finishing_context,UpstreamFinishingContext)):
            current = factory(self.parent)
            for name in FINAL_FIELDS:
                data = self.parent.to_dict()
                del data['settings'][name]
                with self.assertRaises(ValueError):
                    factory(self.fixture.identity(data))
            invalids: tuple[Any, ...] = (type(self.parent),self.parent.to_dict(),current,type(current))
            for invalid in invalids:
                with self.assertRaises(TypeError):
                    factory(invalid)
            unknown = self.parent.to_dict()
            unknown['settings']['future_neural_input'] = {'value': 1}
            self.assertNotEqual(factory(self.fixture.identity(unknown)).key,current.key)
            for version in (0,3,True,'2'):
                data = current.to_dict()
                data['schema_version'] = version
                with self.assertRaises(ValueError):
                    constructor(canonical(data))
            for payload in ('[]','{}','not JSON'):
                with self.assertRaises(ValueError):
                    constructor(payload)
        stage2 = stage2_finishing_context(self.parent)
        upstream = upstream_finishing_context(self.parent)
        with self.assertRaises(ValueError):
            Stage2FinishingContext(upstream.payload)
        with self.assertRaises(ValueError):
            UpstreamFinishingContext(stage2.payload)
        bad: Any = upstream
        with self.assertRaises(TypeError):
            stage2_batch_identity(bad,frames=[],policy='',execution={},inputs={})
        bad = stage2
        with self.assertRaises(TypeError):
            feature_identity(bad,0,'',(),{},())

    def test_shot_profile_recipe_binding_and_payload_stay_conservative(self):
        from corridorkey.session import JobRequest
        from corridorkey.shot_profiles import ProfilePlan, base_recipe_key
        from tests.test_shot_profiles import ShotProfileCodecTests
        fixture = ShotProfileCodecTests()
        fixture.setUp()
        config,writer = fixture.job.configurations()
        config = replace(config,despill_space='linear')
        base = JobRequest.create(fixture.job.input_path,fixture.job.output_dir,config,writer)
        profile = {**fixture.data,'recipe_sha256':base_recipe_key(base)}
        bound = JobRequest.create(base.input_path,base.output_dir,replace(config,shot_profiles=profile),writer)
        ProfilePlan.parse(profile).validate_recipe(bound)
        data = self.parent.to_dict()
        data['settings'] = {**data['settings'],'shot_profiles':profile}
        for field,value in [('despill_strength',.8),('despill_screen','blue'),('despill_space','legacy'),
                             ('despill_preserve_luma',True),('despill_edges_only',True)]:
            edited = JobRequest.create(base.input_path,base.output_dir,replace(config,shot_profiles=profile,**{field:value}),writer)
            with self.assertRaisesRegex(ValueError,'base recipe changed'):
                ProfilePlan.parse(profile).validate_recipe(edited)
            renewed = {**profile,'recipe_sha256':base_recipe_key(edited)}
            changed = copy.deepcopy(data)
            changed['settings'].update({field:value,'shot_profiles':renewed})
            for factory in (stage2_finishing_context,upstream_finishing_context):
                self.assertNotEqual(factory(self.fixture.identity(data)),factory(self.fixture.identity(changed)))
