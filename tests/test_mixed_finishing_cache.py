"""Cached mixed-document B must reproduce current clean B, never old tracked A."""
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.corrections import load_corrections
from corridorkey.identities import file_sha256
from corridorkey.recovery import run_recoverable
from corridorkey.session import InferenceSession
from tests import test_mixed_finishing_recovery as proof
from tests.test_durable_stage2 import cpu_runtime_context


class MixedFinishingCacheTests(unittest.TestCase):
    root: Path
    source: Path
    config: InferenceConfig
    assets: dict
    environment: dict
    implementation: Path
    originals: dict
    scene = proof.MixedFinishingPreproofTests.scene
    request = proof.MixedFinishingPreproofTests.request
    edits = proof.MixedFinishingPreproofTests.edits
    model = proof.MixedFinishingPreproofTests.model
    save = proof.MixedFinishingPreproofTests.save

    def setUp(self: Any):
        proof.MixedFinishingPreproofTests.setUp(self)
        self.scene()
        self.seen = []
        self.rows = []

    def factory(self, **kwargs):
        def load():
            trace = {name:[] for name in ('iid','rvm','stage1','stage2')};held = []
            model = self.model(trace,held)
            self.seen.append((trace,held))
            return model
        return InferenceSession(loader=load,disposer=lambda _:None,**kwargs)

    def run_job(self, path, config, workspace, policies=(True,True), *, budget=8_000_000, progress=None, cancellation=None):
        request = self.request(replace(config,corrections_path=path),self.root/'output')
        result = run_recoverable([request],workspace=workspace,max_cache_bytes=budget,
            assets=self.assets,environment=self.environment,implementation_root=self.implementation,
            session_factory=self.factory,durable_foundations=True,durable_rvm=True,durable_hann=True,
            durable_stage2=True,durable_writer_prefix=True,reuse_stage2_finishing_edits=policies[0],
            reuse_upstream_finishing_edits=policies[1],progress=progress,cancellation=cancellation)
        self.assertEqual(result['status'],'complete',result.get('error'))
        for value,before in self.seen[-1][1]:self.assertEqual(proof.frozen(value),before)
        return result

    @staticmethod
    def hashes(record):
        return {f'{role}/{path.name}':file_sha256(path) for role,directory in record['outputs'].items()
                for path in Path(directory).glob('*.exr')}

    def assert_calls(self, policies):
        trace = self.seen[-1][0]
        self.assertEqual(bool(trace['stage2']),not policies[0])
        for name in ('iid','rvm','stage1'):self.assertEqual(bool(trace[name]),not policies[1],name)
        return {name:len(trace[name]) for name in trace}

    def compare(self, purpose, mode, paths, config, policies, *, alpha_changed=True):
        a,b = paths
        workspace = self.root/(purpose+'-'+mode+'-'+str(len(self.rows)))
        left = self.run_job(a,config,workspace,policies);old = left['jobs'][0];preserved = self.hashes(old)
        right = self.run_job(b,config,workspace,policies);new = right['jobs'][0];calls = self.assert_calls(policies)
        self.assertEqual(new['mode'],'render');self.assertTrue(new['new_session_run'])
        self.assertNotEqual(left['journal'],right['journal']);self.assertNotEqual(old['key'],new['key'])
        self.assertNotEqual(old['output_dir'],new['output_dir'])
        stats = new['origin_record']['diagnostics']
        self.assertEqual(stats['durable_stage2']['batch_hits']>0,policies[0])
        for name,hit in (('durable_foundations','hits'),('durable_rvm','chunk_hits'),('durable_hann','window_hits')):
            if policies[1]:self.assertGreater(stats[name][hit],0)
        self.assertEqual(stats['durable_writer_prefix']['restored_prefix_frames'],0)
        clean = self.run_job(b,config,self.root/('clean-'+str(len(self.rows))),policies=(False,False),budget=0)
        self.assertEqual(self.hashes(new),self.hashes(clean['jobs'][0]))
        self.assertTrue(self.seen[-1][0]['stage2'])
        changes = []
        for role,directory in new['outputs'].items():
            for current,reference,previous in zip(sorted(Path(directory).glob('*.exr')),
                    sorted(Path(clean['jobs'][0]['outputs'][role]).glob('*.exr')),
                    sorted(Path(old['outputs'][role]).glob('*.exr'))):
                value = cv2.imread(str(current),-1)
                np.testing.assert_array_equal(value,cv2.imread(str(reference),-1))
                if role=='alpha':changes.append(int(np.count_nonzero(value!=cv2.imread(str(previous),-1))))
                if role=='fg':np.testing.assert_array_equal(value,cv2.imread(str(previous),-1))
        if alpha_changed:self.assertGreater(sum(changes),0,'cached B must not restore tracked A alpha')
        else:self.assertEqual(sum(changes),0)
        self.assertEqual(self.hashes(old),preserved)
        self.assertEqual({p.name:p.read_bytes() for p in self.source.iterdir()},self.originals)
        self.rows.append({'purpose':purpose,'mode':mode,'policies':policies,'cached_calls':calls,
            'stage2_hits':stats['durable_stage2']['batch_hits'],'prefix_restored':0,
            'alpha_different_pixels':changes,'clean_B_exact':True,'A_preserved':True})

    def test_retired_ordinary_tracks_current_B_cached_parity_independent_policies_modes(self):
        from corridorkey import draft
        from corridorkey.motion import Propagation
        originals = draft.offline_frames,draft.past_frames
        def motion(*args,**kwargs):
            alpha = next(value for value in args if isinstance(value,np.ndarray) and value.ndim==2).copy()
            return Propagation(alpha,np.zeros_like(alpha),np.ones_like(alpha,dtype=bool),
                {'flow_evaluations':4,'motion_seconds':0.,'uncertain_fraction':1.,'motion_fraction':1.})
        def offline(*args,**kwargs):return originals[0](*args,interpolate=motion,**kwargs)
        def past(*args,**kwargs):return originals[1](*args,propagate=motion,**kwargs)
        with cpu_runtime_context(),patch.object(draft,'offline_frames',side_effect=offline),patch.object(draft,'past_frames',side_effect=past):
            for purpose in ('composite','diagnostic'):
                a,b,_ = self.edits(purpose)
                for policies in ((False,False),(True,False),(False,True),(True,True)):
                    with self.subTest(purpose=purpose,policies=policies):self.compare(purpose,'full',(a,b),self.config,policies)
            a,b,_ = self.edits('composite')
            for mode,overrides in [('low',{'low_vram':True}),('spatial',{'model_scale':.5}),
                    ('hold',{'quick_key':True,'quick_approach':'hold'}),('offline',{'quick_key':True}),
                    ('causal',{'quick_key':True,'quick_approach':'causal'}),
                    ('offline_adaptive',{'quick_key':True,'quick_adaptive':True}),
                    ('causal_adaptive',{'quick_key':True,'quick_approach':'causal','quick_adaptive':True})]:
                with self.subTest(mode=mode):self.compare('composite',mode,(a,b),replace(self.config,**overrides),(True,True))
        self.export_rows('retirement')

    def test_first_add_last_remove_and_accepted_final_tracks(self):
        with cpu_runtime_context():
            for purpose in ('composite','diagnostic'):
                a,b,_ = self.edits(purpose)
                editor = CorrectionEditor(self.request(self.config,self.root/'editor-extra'),self.root,existing=b)
                region_id = next(entry['id'] for entry in editor.document.to_dict()['entries'] if entry['kind']=='region')
                editor.remove(region_id);without = editor.save()
                mask = np.zeros((64,64),np.uint8);mask[:,32:] = 255
                editor.commit_brush(mask,kind='region',purpose=purpose,frame=0,strength=.7,direction='both',span=[0,6])
                added = editor.save()
                region_id = editor.document.to_dict()['entries'][-1]['id'];editor.remove(region_id);removed = editor.save()
                self.compare(purpose,'first-add',(without,added),self.config,(True,True),alpha_changed=False)
                self.compare(purpose,'last-remove',(added,removed),self.config,(True,True),alpha_changed=False)
                tracked = CorrectionEditor(self.request(self.config,self.root/'editor-tracked'),self.root,existing=a)
                region_id = next(entry['id'] for entry in tracked.document.to_dict()['entries'] if entry['kind']=='region')
                tracked.prepare_tracking(anchor_ids=[region_id],span=[0,6],direction='forward',unreliable_policy='retain_approximation')
                tracked.tracking_view(2);tracked.accept_tracking()
                with_both_tracks = tracked.save()
                self.assertEqual(len(load_corrections(with_both_tracks).to_dict()['tracks']),2)
                tracked.edit(region_id,{'strength':.9})
                retired_both = tracked.save()
                self.compare(purpose,'ordinary-plus-final-tracks',(with_both_tracks,retired_both),self.config,(True,True))
        self.export_rows('add-remove-tracks')

    def test_nonfinal_strength_and_current_assets_binding_late_publication_stay_guarded(self):
        from contextlib import ExitStack

        from corridorkey.durable_hann import DurableHannWindows
        from corridorkey.durable_rvm import DurableRvmChunks
        from corridorkey.durable_stage2 import DurableStage2Batches
        from corridorkey.recovery_identity import (
            job_identity,
            upstream_finishing_context,
        )
        from corridorkey.recovery_store import CacheInvalid
        policy = {name:True for name in ('durable_foundations','durable_rvm','durable_hann','durable_stage2',
                   'durable_writer_prefix','reuse_stage2_finishing_edits','reuse_upstream_finishing_edits')}
        identity_environment = {**self.environment,'recovery_policy':policy}
        with cpu_runtime_context():
            a,b,control = self.edits('composite')
            workspace = self.root/'negatives'
            packets = []
            original_get = DurableStage2Batches.get
            def observe(owner,key):
                result = original_get(owner,key)
                if key is not None:packets.append((key.key,result is not None))
                return result
            temporal_packets = {'rvm':[],'hann':[]}
            def observe_temporal(adapter,name):
                original = adapter.get
                def get(owner,key):
                    result = original(owner,key);temporal_packets[name].append((key.key,result is not None))
                    return result
                return get
            with ExitStack() as stack:
                stack.enter_context(patch.object(DurableStage2Batches,'get',observe))
                stack.enter_context(patch.object(DurableRvmChunks,'get',observe_temporal(DurableRvmChunks,'rvm')))
                stack.enter_context(patch.object(DurableHannWindows,'get',observe_temporal(DurableHannWindows,'hann')))
                first = self.run_job(a,self.config,workspace);preserved = self.hashes(first['jobs'][0])
                old_packet_keys = {key for key,_ in packets};packets.clear()
                old_temporal_keys = {name:{key for key,_ in values} for name,values in temporal_packets.items()}
                for values in temporal_packets.values():values.clear()
                changed = self.run_job(control,self.config,workspace)
            self.assertTrue(self.seen[-1][0]['iid'])
            for name,values in temporal_packets.items():
                self.assertEqual([hit for _,hit in values],[key in old_temporal_keys[name] for key,_ in values])
            identity = job_identity(self.request(replace(self.config,corrections_path=control),self.root/'output'),
                assets=self.assets,environment=identity_environment,implementation_root=self.implementation)
            self.assertEqual(changed['jobs'][0]['origin_record']['diagnostics']['durable_foundations']['parent_key'],
                             upstream_finishing_context(identity).key)
            self.assertEqual([hit for _,hit in packets],[key in old_packet_keys for key,_ in packets])
            self.assertTrue(any(not hit for _,hit in packets),'changed raw input must miss')
            self.assertEqual(changed['jobs'][0]['origin_record']['diagnostics']['durable_stage2']['batch_hits'],
                             sum(hit for _,hit in packets))
            clean = self.run_job(control,self.config,self.root/'clean-strength',policies=(False,False),budget=0)
            self.assertEqual(self.hashes(changed['jobs'][0]),self.hashes(clean['jobs'][0]))
            document = load_corrections(a);data = document.to_dict()
            corrupt_binding = json.loads(a.read_text());corrupt_binding['tracks'][0]['binding']['entry_order'] = []
            broken = self.root/'bad-binding.json';broken.write_text(json.dumps(corrupt_binding))
            with self.assertRaises(ValueError):self.run_job(broken,self.config,workspace)
            for item in (data['entries'][0]['asset'],data['tracks'][0]['frames'][2]['asset']):
                asset = a.parent/item['path'];original = asset.read_bytes();asset.write_bytes(original+b'corrupt')
                models_before = len(self.seen)
                with self.assertRaises((ValueError,CacheInvalid)):self.run_job(a,self.config,workspace)
                self.assertEqual(len(self.seen),models_before,'invalid current assets must refuse before model/cache replay')
                asset.write_bytes(original)
            late = b.parent/'late.json';late.write_bytes(b.read_bytes())
            mutated = False
            def progress(event):
                nonlocal mutated
                if event.stage=='output_frames' and not mutated:
                    mutated = True
                    content = json.loads(late.read_text());content['revision'] += 1;late.write_text(json.dumps(content))
            with self.assertRaisesRegex(CacheInvalid,'dependencies changed'):
                self.run_job(late,self.config,workspace,progress=progress)
            self.assertTrue(mutated)
            self.assertEqual(self.hashes(first['jobs'][0]),preserved)

    def test_v4_zero_quota_and_cancelled_cached_final_publication(self):
        from corridorkey.control import CancellationToken, CancelledError
        with cpu_runtime_context():
            a,b,_ = self.edits('diagnostic')
            zero = self.run_job(b,self.config,self.root/'zero-quota',budget=0)
            self.assertTrue(all(self.seen[-1][0][name] for name in ('iid','rvm','stage1','stage2')))
            self.assertEqual(zero['jobs'][0]['origin_record']['diagnostics']['durable_stage2']['batch_hits'],0)
            self.assertEqual(list((self.root/'zero-quota'/'entries').iterdir()),[])
            workspace = self.root/'cancelled-cache'
            first = self.run_job(a,self.config,workspace);preserved = self.hashes(first['jobs'][0])
            token = CancellationToken()
            def progress(event):
                if event.stage=='recovery_publish':token.cancel('cancel current mixed final publication')
            with self.assertRaisesRegex(CancelledError,'cancel current mixed'):
                self.run_job(b,self.config,workspace,progress=progress,cancellation=token)
            self.assertTrue(all(not values for values in self.seen[-1][0].values()))
            reports = [json.loads(path.read_text()) for path in (workspace/'reports').glob('*.json')]
            cancelled = [report for report in reports if report['status']=='cancelled']
            self.assertEqual(len(cancelled),1);self.assertEqual(cancelled[0]['jobs'],[])
            self.assertEqual(self.hashes(first['jobs'][0]),preserved)

    def test_guidance_provenance_survives_retirement_and_current_resource_integrity_is_required(self):
        from corridorkey.corrections import attach_guidance
        from corridorkey.video_io import _read_frame_dir_views
        from tests.test_guidance_provenance import GuidanceProvenanceTests
        fixture = GuidanceProvenanceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        editor = CorrectionEditor(self.request(self.config,self.root/'guided-editor'),self.root)
        mask = np.zeros((64,64),np.uint8);mask[16:45,12:29] = 255
        editor.commit_brush(mask,kind='matte',frame=0,strength=.6,direction='both',span=[0,6])
        owner = editor.document.to_dict()['entries'][0]
        reference = self.root/'reference.png';reference.write_bytes((self.source/'1.png').read_bytes())
        proxy,_ = _read_frame_dir_views(str(reference),exr_linear=False)
        import hashlib
        rgb_hash = hashlib.sha256(proxy.astype('<f4',copy=False).tobytes()).hexdigest()
        resources = [{'id':'plate','role':'reference','asset':{'path':str(reference),'sha256':file_sha256(reference)}}]
        observation = {'source_samples':[{'frame':0,'source_frame':1,'rgb_sha256':rgb_hash}],
                       'reference_samples':[{'resource_id':'plate','frame':0,'rgb_sha256':rgb_hash}],
                       'result_asset_sha256':owner['asset']['sha256'],
                       'diagnostics':{'quality_verified':False,'evidence':'cache-provenance fixture only'}}
        guided = attach_guidance(editor.document,owner['id'],guidance_id='cache-guidance',kind='clean_plate',
            resources=resources,parameters=fixture.parameters,decode=fixture.decode,observation=observation)
        seed = self.save(guided,'guided-seed.json')
        editor = CorrectionEditor(self.request(self.config,self.root/'guided-editor-2'),self.root,existing=seed)
        region = np.zeros((64,64),np.uint8);region[:,:32] = 255
        editor.commit_brush(region,kind='region',purpose='composite',frame=0,strength=.35,direction='both',span=[0,6])
        final_id = editor.document.to_dict()['entries'][-1]['id']
        editor.prepare_tracking(anchor_ids=[owner['id']],span=[0,6],direction='forward',unreliable_policy='retain_approximation')
        editor.tracking_view(2);editor.accept_tracking();a = editor.save()
        before = editor.document.to_dict()['guidance']
        editor.edit(final_id,{'strength':.9});b = editor.save()
        self.assertEqual(editor.document.to_dict()['guidance'],before)
        with cpu_runtime_context():
            self.compare('composite','guided-retirement',(a,b),self.config,(True,True))
            old = reference.read_bytes();reference.write_bytes(old+b'corrupt')
            with self.assertRaisesRegex(ValueError,'guidance resource'):
                self.run_job(b,self.config,self.root/'guided-corruption')
            reference.write_bytes(old)
        self.export_rows('guided')

    def export_rows(self, suffix):
        destination = os.environ.get('CORRIDORKEY_MIXED_CACHE_REPORT')
        if destination:
            root = Path(destination);root.mkdir(parents=True,exist_ok=True)
            (root/(suffix+'-matrix.json')).write_text(json.dumps(self.rows,indent=2))
