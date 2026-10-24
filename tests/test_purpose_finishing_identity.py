"""Legacy Stage2 v3 and unchanged upstream pure-region projection contracts."""
import copy
import hashlib
import unittest

from corridorkey.corrections import CorrectionDocument
from corridorkey.recovery_identity import (
    ReplayIdentity,
    UpstreamFinishingContext,
    canonical,
    upstream_finishing_context,
)
from corridorkey.stage2_identity import Stage2FinishingContext
from tests import test_stage2_finishing_identity as fixtures

FACTORIES = (fixtures.legacy_stage2_context, upstream_finishing_context)


def parent(purpose='composite', *, kind='region', entries=True):
    fixture = fixtures.Stage2FinishingIdentityTests()
    fixture.setUp()
    data = fixture.data
    data['source']['sha256'] = 'a' * 64
    data['settings']['corrections_path'] = '/local/regions.json'
    entry = {'id':'region','revision':1,'frame':0,'kind':kind,'strength':.5,'direction':'both','range':[0,3],
             'asset':{'path':'mask.png','sha256':'b'*64}}
    if kind == 'region':
        entry.update(purpose=purpose,target={'composite':'despill','diagnostic':'tracking_overlay','refinement':'edge_finishing'}[purpose])
    doc = {'schema_version':5,'revision':1,'source':{'sha256':'a'*64,'start_frame':0,'frame_count':3,
            'fps':24.,'width':8,'height':8},'shots':[[0,3]],'entries':[entry] if entries else [],'tracks':[],'guidance':[]}
    return wrap(data,doc)


def wrap(data, document):
    data = copy.deepcopy(data)
    parsed = CorrectionDocument.parse(document)
    data['corrections'] = {'manifest_sha256':hashlib.sha256(canonical(document).encode()).hexdigest(),
                           'identity':parsed.identity,'document':document}
    return ReplayIdentity(canonical(data))


class PurposeFinishingIdentityTests(unittest.TestCase):
    def test_pure_edits_and_relocation_share_only_v3_raw_context(self):
        for purpose in ('composite','diagnostic'):
            original = parent(purpose)
            data = original.to_dict()
            doc = copy.deepcopy(data['corrections']['document'])
            doc['revision'] += 1
            doc['entries'][0].update(revision=2,strength=.8,range=[0,2])
            doc['entries'][0]['asset'].update(path='../new/mask.png',sha256='c'*64)
            data['settings']['corrections_path'] = '/relocated/regions2.json'
            edited = wrap(data,doc)
            self.assertNotEqual(original.key,edited.key)
            for factory in FACTORIES:
                context = factory(original)
                self.assertEqual(context.to_dict()['schema_version'],3)
                self.assertEqual(context,factory(edited))
                self.assertNotIn('corrections_path',context.to_dict()['settings'])
                self.assertEqual(context.to_dict()['corrections']['source'],doc['source'])
                self.assertEqual(context.to_dict()['corrections']['shots'],doc['shots'])
        mixed = parent().to_dict()
        doc = mixed['corrections']['document']
        doc['entries'].append({**doc['entries'][0],'id':'diagnostic','purpose':'diagnostic','target':'tracking_overlay'})
        self.assertEqual(fixtures.legacy_stage2_context(wrap(mixed,doc)).to_dict()['schema_version'],3)

    def test_empty_none_remain_v2_and_schema5_nonfinal_fields_are_retained_in_v4(self):
        cases = [parent(entries=False)]
        fixture = fixtures.Stage2FinishingIdentityTests();fixture.setUp()
        cases.append(fixture.identity(fixture.data))
        for original in cases:
            for factory in FACTORIES:
                result = factory(original)
                self.assertEqual(result.to_dict()['schema_version'],2)
                data = result.to_dict()
                self.assertEqual(data['corrections'],original.to_dict()['corrections'])
                self.assertEqual(data['settings']['corrections_path'],original.to_dict()['settings']['corrections_path'])
        for original in (parent('refinement'),parent(kind='matte'),parent(kind='keep'),parent(kind='remove')):
            for factory in FACTORIES:
                result = factory(original).to_dict()
                self.assertEqual(result['schema_version'],4)
                retained = result['corrections']['document']
                source = original.to_dict()['corrections']['document']
                for field in ('entries','source','shots','guidance'):
                    self.assertEqual(retained[field],source[field])
                self.assertEqual(retained['tracks'],[])
                self.assertNotIn('corrections_path',result['settings'])
        data = parent().to_dict();doc = data['corrections']['document']
        for extra in ({'id':'reset','revision':1,'frame':1,'kind':'reset'},
                      {**doc['entries'][0],'id':'ordinary','kind':'keep'}):
            extra.pop('purpose',None);extra.pop('target',None)
            altered = {**doc,'entries':[doc['entries'][0],extra]}
            result = fixtures.legacy_stage2_context(wrap(data,altered)).to_dict()
            self.assertEqual(result['schema_version'],4)
            for field in ('source','shots','guidance'):
                self.assertEqual(result['corrections']['document'][field],altered[field])
            self.assertEqual(result['corrections']['document']['entries'],[extra])

    def test_source_shots_presence_and_unknown_dependencies_still_bind(self):
        original = parent()
        for mutation in ('shots','source','unknown'):
            data = original.to_dict();doc = data['corrections']['document']
            if mutation == 'shots':
                doc['shots'] = [[0,1],[1,3]];doc['entries'][0]['range'] = [0,1]
            elif mutation == 'source':
                doc['source']['start_frame'] = 2
            else:
                data['settings']['future_neural_setting'] = True
            for factory in FACTORIES:
                self.assertNotEqual(factory(original),factory(wrap(data,doc)))
        for factory in FACTORIES:
            self.assertNotEqual(factory(original),factory(parent(entries=False)))

    def test_malformed_wrapper_identity_binding_and_preprojection_refused(self):
        for factory in FACTORIES:
            original = parent()
            for mutation in ('extra','identity','manifest','document','path','source','preprojected'):
                data = original.to_dict()
                if mutation == 'extra':data['corrections']['future_dependency'] = True
                elif mutation == 'identity':data['corrections']['identity'] = 'c'*64
                elif mutation == 'manifest':data['corrections']['manifest_sha256'] = 'bad'
                elif mutation == 'document':data['corrections']['document']['entries'][0]['purpose'] = 'neural'
                elif mutation == 'path':data['settings']['corrections_path'] = None
                elif mutation == 'source':data['source']['sha256'] = 'd'*64
                else:data['corrections'] = factory(original).to_dict()['corrections']
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    factory(ReplayIdentity(canonical(data)))

    def test_v3_projection_closed_fields_path_and_domain(self):
        for factory,constructor in ((fixtures.legacy_stage2_context,Stage2FinishingContext),
                                    (upstream_finishing_context,UpstreamFinishingContext)):
            current = factory(parent()).to_dict()
            self.assertEqual(current['schema_version'],3)
            for mutation in ('path','extra','presence','source','shots','kind','missing'):
                data = copy.deepcopy(current)
                if mutation == 'path':data['settings']['corrections_path'] = None
                elif mutation == 'extra':data['corrections']['future_dependency'] = True
                elif mutation == 'presence':data['corrections']['present'] = False
                elif mutation == 'source':data['corrections']['source']['width'] = 0
                elif mutation == 'shots':data['corrections']['shots'] = []
                elif mutation == 'kind':data['corrections']['kind'] = 'unknown'
                else:del data['corrections']['source']
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    constructor(canonical(data))

    def test_accepted_tracks_validate_complete_anchor_binding_before_projection(self):
        from corridorkey.corrections import track_binding
        from tests.test_tracked_refinement_regions import RegionTrackingCoreTests
        fixture = RegionTrackingCoreTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        document,_ = fixture.build()
        for entry in document['entries']:entry.update(purpose='diagnostic',target='tracking_overlay')
        track = document['tracks'][0];old = track['binding']
        track['binding'] = track_binding(document,track['anchor_ids'],cuts=old['cuts'],decode=old['decode'],tracking=old['tracking'])
        track['binding_sha256'] = hashlib.sha256(canonical(track['binding']).encode()).hexdigest()
        data = parent().to_dict()
        valid = wrap(data,document)
        for factory in FACTORIES:
            self.assertEqual(factory(valid).to_dict()['schema_version'],3)
            for mutation in ('anchor','binding','partial'):
                broken = valid.to_dict()
                if mutation == 'anchor':broken['corrections']['document']['entries'][0]['strength'] = .9
                elif mutation == 'binding':broken['corrections']['document']['tracks'][0]['binding']['entry_order'] = []
                else:broken['corrections']['document']['tracks'][0]['frames'].pop()
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    factory(ReplayIdentity(canonical(broken)))
