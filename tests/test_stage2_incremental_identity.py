"""Stage2 v5 narrows authored context only behind complete exact-call keys."""
import copy
import unittest

from corridorkey.identities import sequence_identity
from corridorkey.recovery_identity import (
    ReplayIdentity,
    canonical,
    upstream_finishing_context,
)
from corridorkey.stage2_identity import (
    Stage2FinishingContext,
    stage2_batch_identity,
    stage2_finishing_context,
)
from tests import test_purpose_finishing_identity as pure
from tests import test_stage2_finishing_identity as fixtures


class Stage2IncrementalIdentityTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.Stage2FinishingIdentityTests();f.setUp()
        self.data = f.data

    def context(self, data):
        return stage2_finishing_context(ReplayIdentity(canonical(data)))

    def test_current_v5_admits_originals_and_narrows_only_known_slots(self):
        original = self.context(self.data)
        self.assertEqual(original.to_dict()['schema_version'],5)
        variants = [pure.parent(kind='matte'),pure.parent('refinement'),pure.parent(),pure.parent(entries=False)]
        for parent in variants:
            data = parent.to_dict()
            # Keep all other source dependencies identical to isolate authored inputs.
            self.data['source'] = data['source']
            base = self.context(self.data)
            self.assertEqual(base,self.context(data))
            self.assertNotEqual(upstream_finishing_context(ReplayIdentity(canonical(self.data))),upstream_finishing_context(parent))
        for slot in ('first','full'):
            for hint in ({'path':'/local/hint.png','sha256':'a'*64},
                         sequence_identity('/local/hints',[{'name':'1.png','sha256':'a'*64},{'name':'10.png','sha256':'b'*64}])):
                data = copy.deepcopy(self.data);data['hints'][slot] = hint
                self.assertEqual(self.context(data),self.context(self.data))
        for section in ('source','assets','environment','implementation','writer','settings','hints'):
            data = copy.deepcopy(self.data);data[section]['unknown'] = {'must':'remain'}
            self.assertNotEqual(self.context(data),self.context(self.data),section)

    def test_hint_records_strict_structure_and_order_aggregate(self):
        valid = sequence_identity('hints',[{'name':'1.png','sha256':'a'*64},{'name':'2.png','sha256':'b'*64}])
        bad = [{},[],{'path':'x','sha256':'x'},{'path':'x','sha256':'a'*64,'unknown':True},
               {**valid,'sha256':'c'*64},{**valid,'files':list(reversed(valid['files']))},
               sequence_identity('hints',[{'name':'../1.png','sha256':'a'*64}]),
               sequence_identity('hints',[{'name':'1.png','sha256':'a'*64}]*2),
               sequence_identity('hints',[{'name':'1.png','sha256':'a'*64,'extra':True}])]
        for slot in ('first','full'):
            for hint in bad:
                data = copy.deepcopy(self.data);data['hints'][slot] = hint
                with self.subTest(slot=slot,hint=hint),self.assertRaises(ValueError):self.context(data)
        # Existing structural empty-sequence identity is not reinterpreted here.
        data = copy.deepcopy(self.data);data['hints']['full'] = sequence_identity('empty',[])
        self.assertEqual(self.context(data),self.context(self.data))

    def test_original_corrections_not_projected_forged_or_bad_binding(self):
        parent = pure.parent();data = parent.to_dict()
        cases = []
        bad = copy.deepcopy(data);bad['corrections']['identity'] = 'f'*64;cases.append(bad)
        bad = copy.deepcopy(data);bad['corrections']['document']['source']['sha256'] = 'd'*64;cases.append(bad)
        bad = copy.deepcopy(data);bad['corrections']['unknown'] = True;cases.append(bad)
        bad = copy.deepcopy(data);bad['settings']['corrections_path'] = None;cases.append(bad)
        bad = copy.deepcopy(self.data);bad['settings']['corrections_path'] = 'missing.json';cases.append(bad)
        projected = self.context(data).to_dict()
        bad = copy.deepcopy(data);bad['corrections'] = projected['corrections'];cases.append(bad)
        for field in ('corrections','hints'):
            bad = copy.deepcopy(data);del bad[field];cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.context(bad)

    def test_closed_v5_domain_and_call_tensor_group_execution_guards(self):
        context = self.context(self.data);data = context.to_dict()
        for section in ('corrections','first','full'):
            bad = copy.deepcopy(data)
            record = bad['corrections'] if section=='corrections' else bad['hints'][section]
            for changes in ({'schema_version':True},{'kind':'wrong'},{'extra':1}):
                changed = copy.deepcopy(bad)
                target = changed['corrections'] if section=='corrections' else changed['hints'][section]
                target.update(changes)
                with self.assertRaises(ValueError):Stage2FinishingContext(canonical(changed))
            self.assertIsInstance(record,dict)
        bad = copy.deepcopy(data);bad['settings']['corrections_path'] = None
        with self.assertRaises(ValueError):Stage2FinishingContext(canonical(bad))
        with self.assertRaises(ValueError):
            from corridorkey.recovery_identity import UpstreamFinishingContext
            UpstreamFinishingContext(canonical({**data,'kind':'upstream_finishing_context','reuse_scope':'iid_rvm_hann_raw_inputs'}))
        execution = {'backend':'pytorch','device':'cpu','autocast_dtype':'torch.float32'}
        inputs = {'rgb':{'shape':[1,2,3,64,64],'dtype':'torch.float32','sha256':'a'*64},
                  'baton':{'shape':[1,2,1,4,4],'dtype':'torch.float32','sha256':'b'*64}}
        def batch(**overrides):return stage2_batch_identity(context,**{'frames':[0,1],'policy':'full_rate','execution':execution,'inputs':inputs,**overrides})
        original = batch()
        for slot in ('rgb','baton'):
            changed = copy.deepcopy(inputs);changed[slot]['sha256'] = 'c'*64
            self.assertNotEqual(batch(inputs=changed),original)
            changed[slot]['dtype'] = 'torch.float16';self.assertNotEqual(batch(inputs=changed),original)
        self.assertNotEqual(batch(frames=[1,0]),original)
        self.assertNotEqual(batch(policy='quick_anchor'),original)
        self.assertNotEqual(batch(execution={**execution,'device':'cuda:0'}),original)

    def test_direct_legacy_literal_keys_unchanged(self):
        # Literal records/keys captured before v5; independent of current factories.
        for payload, key in LEGACY_GOLDENS:
            context = Stage2FinishingContext(payload)
            self.assertEqual(context.key,key)
            self.assertEqual(context.payload,payload)


LEGACY_GOLDENS = (('{"assets":{},"corrections":null,"environment":{"runtime":{"synthetic":true}},"hints":{"first":null,"full":null},"implementation":{"sha256":"b"},"kind":"stage2_finishing_context","profile_stages":false,"schema_version":1,"scope":"whole job","settings":{},"source":{"sha256":"a"},"writer":{"dtype":"float"}}', 'ff89c53daf81758c07cdb1213d2851be3ff8c287bfe6e5bcff88eb9da156e29b'), ('{"assets":{},"corrections":null,"environment":{"runtime":{"synthetic":true}},"hints":{"first":null,"full":null},"implementation":{"sha256":"b"},"kind":"stage2_finishing_context","profile_stages":false,"schema_version":2,"scope":"whole job","settings":{},"source":{"sha256":"a"},"writer":{"dtype":"float"}}', 'af45aeaafefe47acb3a45b8bb226426ee1ee8c6d93ccd2f3eabe783acf1edc37'), ('{"assets":{},"corrections":{"document_schema_version":5,"kind":"pure_render_region_schedule","present":true,"schema_version":1,"shots":[[0,3]],"source":{"fps":24.0,"frame_count":3,"height":8,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","start_frame":0,"width":8}},"environment":{"runtime":{"synthetic":true}},"hints":{"first":null,"full":null},"implementation":{"sha256":"b"},"kind":"stage2_finishing_context","profile_stages":false,"schema_version":3,"scope":"whole job","settings":{},"source":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"writer":{"dtype":"float"}}', '471831df11f21f25a43e2e4f76cc2dab83be7a59b8899791bae0844ce6cce806'), ('{"assets":{},"corrections":{"document":{"entries":[{"asset":{"path":"mask.png","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"direction":"both","frame":0,"id":"authored","kind":"matte","range":[0,3],"revision":1,"strength":0.5}],"guidance":[],"revision":0,"schema_version":5,"shots":[[0,3]],"source":{"fps":24.0,"frame_count":3,"height":8,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","start_frame":0,"width":8},"tracks":[]},"kind":"mixed_finishing_dependencies","present":true,"schema_version":1},"environment":{"runtime":{"synthetic":true}},"hints":{"first":null,"full":null},"implementation":{"sha256":"b"},"kind":"stage2_finishing_context","profile_stages":false,"schema_version":4,"scope":"whole job","settings":{},"source":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"writer":{"dtype":"float"}}', 'd7ceb8c19923ca4a884dfa3d1c2344cc3f63a8d02f8a8c0b0491f2105a97db54'))
