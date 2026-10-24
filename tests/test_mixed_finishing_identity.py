"""Mixed raw dependencies retain every nonfinal field, not derived final masks."""
import copy
import unittest

from corridorkey.recovery_identity import (
    ReplayIdentity,
    UpstreamFinishingContext,
    canonical,
    upstream_finishing_context,
)
from corridorkey.stage2_identity import Stage2FinishingContext
from tests import test_purpose_finishing_identity as pure

FACTORIES = (pure.fixtures.legacy_stage2_context,upstream_finishing_context)


def mixed(*, final=True):
    data = pure.parent(kind='matte').to_dict()
    document = data['corrections']['document']
    document['entries'][0]['id'] = 'authored'
    if final:
        region = pure.parent().to_dict()['corrections']['document']['entries'][0]
        document['entries'].append(region)
    return pure.wrap(data,document)


class MixedFinishingIdentityTests(unittest.TestCase):
    def test_first_add_last_remove_and_outer_revision_only_do_not_rekey_raw_inputs(self):
        base = mixed(final=False)
        edited = mixed()
        data = edited.to_dict();document = data['corrections']['document']
        document['revision'] += 6;document['entries'][-1]['revision'] += 6
        document['entries'][-1]['asset']['sha256'] = 'c'*64
        data['settings']['corrections_path'] = '/elsewhere/final-change.json'
        changed = pure.wrap(data,document)
        for factory in FACTORIES:
            original = factory(base)
            self.assertEqual(original.to_dict()['schema_version'],4)
            self.assertEqual(original,factory(edited))
            self.assertEqual(original,factory(changed))
            projected = original.to_dict()['corrections']
            self.assertIs(projected['present'],True)
            retained = projected['document']
            self.assertEqual(retained['entries'],base.to_dict()['corrections']['document']['entries'])
            self.assertEqual(retained['source'],document['source']);self.assertEqual(retained['shots'],document['shots'])
            self.assertEqual(retained['guidance'],document['guidance']);self.assertEqual(retained['tracks'],[])
            self.assertEqual(retained['revision'],0)
            self.assertNotIn('corrections_path',original.to_dict()['settings'])
            self.assertNotEqual(base.key,edited.key)

    def test_all_retained_fields_order_disabled_and_unknown_dependencies_still_bind(self):
        original = mixed()
        for key,value in [('id','renamed'),('revision',3),('frame',1),('strength',0),
                          ('direction','forward'),('range',[0,2]),('kind','remove'),('asset',{'path':'new.png','sha256':'b'*64})]:
            data = original.to_dict();document = data['corrections']['document']
            document['entries'][0][key] = value
            for factory in FACTORIES:
                self.assertNotEqual(factory(original),factory(pure.wrap(data,document)),key)
        data = original.to_dict();document = data['corrections']['document']
        other = {**document['entries'][0],'id':'second','kind':'keep'}
        document['entries'].append(other)
        before = pure.wrap(data,document)
        document['entries'][0],document['entries'][-1] = document['entries'][-1],document['entries'][0]
        after = pure.wrap(data,document)
        for factory in FACTORIES:self.assertNotEqual(factory(before),factory(after))
        for mutation in ('source','shots','settings'):
            data = original.to_dict();document = data['corrections']['document']
            if mutation=='source':document['source']['start_frame'] += 1
            elif mutation=='shots':
                document['shots'] = [[0,1],[1,3]]
                for entry in document['entries']:entry['range'] = [0,1]
            else:data['settings']['future_neural_dependency'] = True
            for factory in FACTORIES:self.assertNotEqual(factory(original),factory(pure.wrap(data,document)))

    def test_only_known_final_pairs_omitted_and_none_empty_older_pure_versions_stay(self):
        original = mixed()
        for kind,purpose,target in [('region','refinement','edge_finishing'),('trimap',None,None),('reset',None,None)]:
            data = original.to_dict();document = data['corrections']['document']
            entry = document['entries'][-1]
            if kind=='reset':document['entries'][-1] = {'id':'reset','revision':1,'frame':1,'kind':'reset'}
            else:
                entry['kind'] = kind
                if purpose:entry.update(purpose=purpose,target=target)
                else:entry.pop('purpose');entry.pop('target')
            for factory in FACTORIES:
                retained = factory(pure.wrap(data,document)).to_dict()['corrections']['document']['entries']
                self.assertEqual(retained,document['entries'])
        for factory in FACTORIES:
            self.assertEqual(factory(pure.parent()).to_dict()['schema_version'],3)
            self.assertEqual(factory(pure.parent(entries=False)).to_dict()['schema_version'],2)
            data = mixed(final=False).to_dict();document = data['corrections']['document']
            document['schema_version'] = 4
            self.assertEqual(factory(pure.wrap(data,document)).to_dict()['schema_version'],2)

    def test_closed_v4_shape_rejects_final_entries_tracks_revision_and_preprojection(self):
        for factory,reader in ((pure.fixtures.legacy_stage2_context,Stage2FinishingContext),
                               (upstream_finishing_context,UpstreamFinishingContext)):
            context = factory(mixed()).to_dict()
            self.assertEqual(context['schema_version'],4)
            for mutation in ('path','extra','revision','tracks','empty','final','source','present'):
                data = copy.deepcopy(context);projection = data['corrections'];document = projection['document']
                if mutation=='path':data['settings']['corrections_path'] = None
                elif mutation=='extra':projection['future_dependency'] = True
                elif mutation=='revision':document['revision'] = 1
                elif mutation=='tracks':document['tracks'] = [{}]
                elif mutation=='empty':document['entries'] = []
                elif mutation=='final':document['entries'] = pure.parent().to_dict()['corrections']['document']['entries']
                elif mutation=='source':document['source']['sha256'] = 'd'*64
                else:projection['present'] = False
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):reader(canonical(data))
            data = mixed().to_dict();data['corrections'] = context['corrections']
            with self.assertRaises(ValueError):factory(ReplayIdentity(canonical(data)))

    def test_original_track_bindings_validate_before_all_derived_tracks_are_removed(self):
        import hashlib

        from corridorkey.corrections import track_binding
        from tests.test_tracked_refinement_regions import RegionTrackingCoreTests
        fixture = RegionTrackingCoreTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        document,_ = fixture.build('both',both_anchors=True)
        for entry in document['entries']:
            entry['kind'] = 'matte';entry.pop('purpose');entry.pop('target')
        track = document['tracks'][0];old = track['binding']
        track['binding'] = track_binding(document,track['anchor_ids'],cuts=old['cuts'],decode=old['decode'],tracking=old['tracking'])
        track['binding_sha256'] = hashlib.sha256(canonical(track['binding']).encode()).hexdigest()
        wrapper = mixed().to_dict()
        original = pure.wrap(wrapper,document)
        without = copy.deepcopy(document);without['tracks'] = [];without['revision'] += 2
        # Inserting a final region may retire a formerly contiguous multi-anchor
        # track; do not fabricate a binding for the privately filtered document.
        without['entries'].insert(1,pure.parent().to_dict()['corrections']['document']['entries'][0])
        for factory in FACTORIES:
            self.assertEqual(factory(original),factory(pure.wrap(wrapper,without)))
            retained = factory(original).to_dict()['corrections']['document']
            self.assertEqual(retained['entries'],document['entries'])
            for mutation in ('binding','anchor','partial'):
                data = original.to_dict();doc = data['corrections']['document']
                if mutation=='binding':doc['tracks'][0]['binding']['entry_order'] = []
                elif mutation=='anchor':doc['entries'][0]['revision'] += 1
                else:doc['tracks'][0]['frames'].pop()
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):factory(ReplayIdentity(canonical(data)))

    def test_full_guidance_binding_resources_and_provenance_remain_dependencies(self):
        from tests.test_guidance_provenance import GuidanceProvenanceTests
        fixture = GuidanceProvenanceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        document = fixture.attach().to_dict();document['schema_version'] = 5
        final = pure.parent().to_dict()['corrections']['document']['entries'][0]
        document['entries'].append(final)
        data = mixed().to_dict()
        original = pure.wrap(data,document)
        changed = copy.deepcopy(document)
        changed['guidance'][0]['parameters']['settings']['threshold'] = .2
        renewed = fixture.rebind(changed).to_dict()
        for factory in FACTORIES:
            retained = factory(original).to_dict()['corrections']['document']
            self.assertEqual(retained['guidance'],document['guidance'])
            self.assertEqual(retained['entries'],[document['entries'][0]])
            self.assertNotEqual(factory(original),factory(pure.wrap(data,renewed)))
            stale = original.to_dict()
            stale['corrections']['document']['guidance'][0]['parameters']['settings']['threshold'] = .3
            with self.assertRaises(ValueError):factory(ReplayIdentity(canonical(stale)))
            relocated = copy.deepcopy(document)
            relocated['guidance'][0]['resources'][0]['asset']['path'] = '../reference.bin'
            self.assertNotEqual(factory(original),factory(pure.wrap(data,relocated)))

    def test_v4_exact_entry_state_hint_precision_and_grouping_guards_remain(self):
        from corridorkey.hann_identity import hann_window_identity
        from corridorkey.recurrent_identity import rvm_chunk_identity
        from corridorkey.stage2_identity import stage2_batch_identity
        upstream = upstream_finishing_context(mixed())
        arguments = {'shot_start':0,'start':1,'stop':2,'proxy_sha256':'a'*64,'input_shape':[1,3,64,64],
            'input_dtype':'torch.float32','incoming_state_sha256':'b'*64,'channels':1,'grid_hw':[1,1]}
        original = rvm_chunk_identity(upstream,**arguments)
        for change in ({'incoming_state_sha256':'c'*64},{'input_dtype':'torch.bfloat16'},
                       {'shot_start':1,'incoming_state_sha256':None}):
            self.assertNotEqual(original,rvm_chunk_identity(upstream,**{**arguments,**change}))
        inputs: dict = {name:{'sha256':'a'*64,'dtype':'torch.float32','shape':shape} for name,shape in
                  {'rgb':[1,2,3,64,64],'hint':[1,2,1,64,64],'hint_quality':[1,2],'moge':[1,2,1,1,1]}.items()}
        inputs.update(cradio=None,rvm=None)
        arguments = {'schedule_sha256':'a'*64,'incoming_replay_sha256':'b'*64,'index':0,'start':0,'stop':2,
                     'next_start':1,'segment':[0,3],'carry':True,'d_16':1,'inputs':inputs}
        original = hann_window_identity(upstream,**arguments)
        for change in ({'incoming_replay_sha256':'c'*64},{'schedule_sha256':'c'*64},{'carry':False}):
            self.assertNotEqual(original,hann_window_identity(upstream,**{**arguments,**change}))
        for name in ('hint','hint_quality','rgb'):
            altered = copy.deepcopy(inputs);altered[name]['sha256'] = 'c'*64
            self.assertNotEqual(original,hann_window_identity(upstream,**{**arguments,'inputs':altered}))
        stage2 = pure.fixtures.legacy_stage2_context(mixed())
        args = {'frames':[0,1],'policy':'full_rate','execution':{'backend':'pytorch','device':'cpu','autocast_dtype':'torch.float32'},
                'inputs':{name:{'sha256':'a'*64,'dtype':'torch.float32','shape':shape}
                          for name,shape in [('rgb',[1,2,3,64,64]),('baton',[1,2,1,4,4])]}}
        original = stage2_batch_identity(stage2,**args)
        self.assertNotEqual(original,stage2_batch_identity(stage2,**{**args,'frames':[1,0]}))
        altered = copy.deepcopy(args);altered['inputs']['baton']['sha256'] = 'c'*64
        self.assertNotEqual(original,stage2_batch_identity(stage2,**altered))
