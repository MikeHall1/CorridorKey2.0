"""Temporal-only scope cannot enter IID and preserves complete child contracts."""
import copy
import dataclasses
import unittest
from typing import Any
from unittest.mock import patch

from corridorkey.durable_foundations import DurableFeatureFrames
from corridorkey.recovery_identity import (
    ReplayIdentity,
    UpstreamFinishingContext,
    canonical,
    feature_identity,
    temporal_finishing_context,
    upstream_finishing_context,
)
from corridorkey.stage2_identity import Stage2FinishingContext, stage2_finishing_context
from tests import test_purpose_finishing_identity as pure
from tests import test_stage2_finishing_identity as fixtures
from tests import test_stage2_incremental_identity as stage2


class TemporalIncrementalIdentityTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.Stage2FinishingIdentityTests();fixture.setUp()
        self.parent = fixture.identity(fixture.data)

    def test_paired_scope_and_constant_time_accessor_without_dataclass_fields(self):
        context = temporal_finishing_context(self.parent)
        self.assertEqual(context.to_dict()['schema_version'],5)
        self.assertEqual(context.reuse_scope,'rvm_hann_exact_inputs')
        self.assertEqual([field.name for field in dataclasses.fields(context)],['payload'])
        self.assertEqual(dataclasses.asdict(context),{'payload':context.payload})
        self.assertEqual(context,UpstreamFinishingContext(context.payload))
        with patch('corridorkey.recovery_identity.json.loads',side_effect=AssertionError('scope must not parse per-frame JSON')):
            for _ in range(100):self.assertEqual(context.reuse_scope,'rvm_hann_exact_inputs')
        for version,scope in ((1,'rvm_hann_exact_inputs'),(4,'rvm_hann_exact_inputs'),
                              (5,'iid_rvm_hann_raw_inputs'),(5,'future'),(6,'rvm_hann_exact_inputs')):
            data = context.to_dict();data.update(schema_version=version,reuse_scope=scope)
            with self.subTest(version=version,scope=scope),self.assertRaises(ValueError):UpstreamFinishingContext(canonical(data))

    def test_both_IID_admission_routes_reject_temporal_before_use(self):
        context = temporal_finishing_context(self.parent)
        with self.assertRaisesRegex(ValueError,'Temporal-only'):
            DurableFeatureFrames(None,context,validate_context=lambda:None)
        with self.assertRaisesRegex(ValueError,'Temporal-only'):
            feature_identity(context,0,'a'*64,(3,16,16),{'moge':1},(1,1))
        legacy = upstream_finishing_context(self.parent)
        self.assertEqual(legacy.reuse_scope,'iid_rvm_hann_raw_inputs')
        feature_identity(legacy,0,'a'*64,(3,16,16),{'moge':1},(1,1))
        DurableFeatureFrames(None,legacy,validate_context=lambda:None)

    def test_closed_projection_cross_stage_domains_and_original_only(self):
        temporal = temporal_finishing_context(self.parent).to_dict()
        stage = stage2_finishing_context(self.parent).to_dict()
        with self.assertRaises(ValueError):Stage2FinishingContext(canonical({**temporal,'kind':'stage2_finishing_context'}))
        with self.assertRaises(ValueError):
            UpstreamFinishingContext(canonical({**stage,'kind':'upstream_finishing_context','reuse_scope':'rvm_hann_exact_inputs'}))
        for slot in ('corrections','first','full'):
            for changes in ({'schema_version':True},{'kind':'other'},{'unknown':True}):
                data = copy.deepcopy(temporal);record = data['corrections'] if slot=='corrections' else data['hints'][slot]
                record.update(changes)
                with self.assertRaises(ValueError):UpstreamFinishingContext(canonical(data))
        data = self.parent.to_dict();data['corrections'] = temporal['corrections']
        with self.assertRaises(ValueError):temporal_finishing_context(ReplayIdentity(canonical(data)))
        invalids: tuple[Any, ...] = (ReplayIdentity,self.parent.to_dict(),upstream_finishing_context(self.parent))
        for invalid in invalids:
            with self.assertRaises(TypeError):temporal_finishing_context(invalid)

    def test_shared_original_hint_and_correction_validation_all_current_guards(self):
        # Run identical structural-negative inventory against the new temporal factory.
        fixture: Any = stage2.Stage2IncrementalIdentityTests();fixture.setUp()
        fixture.context = lambda data:temporal_finishing_context(ReplayIdentity(canonical(data)))
        fixture.test_hint_records_strict_structure_and_order_aggregate()
        fixture.test_original_corrections_not_projected_forged_or_bad_binding()
        original = temporal_finishing_context(self.parent)
        for section in ('source','assets','environment','implementation','writer','settings','hints'):
            data = self.parent.to_dict();data[section]['future_dependency'] = {'keep':True}
            self.assertNotEqual(temporal_finishing_context(ReplayIdentity(canonical(data))),original)
        for field in fixtures.FINAL_FIELDS:
            data = self.parent.to_dict();del data['settings'][field]
            with self.assertRaises(ValueError):temporal_finishing_context(ReplayIdentity(canonical(data)))
        # Known authored edits may qualify temporally; the legacy IID parent does not.
        authored = pure.parent(kind='matte');data = authored.to_dict();document = data['corrections']['document']
        document['entries'][0]['strength'] = .8;changed = pure.wrap(data,document)
        self.assertEqual(temporal_finishing_context(authored),temporal_finishing_context(changed))
        self.assertNotEqual(upstream_finishing_context(authored),upstream_finishing_context(changed))
