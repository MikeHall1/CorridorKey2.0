"""Uncached ordinary-track retirement proof at actual neural/state seams."""
import hashlib
import json
import os
import unittest
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.nn import functional

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.durable_foundations import DurableFeatureFrames
from corridorkey.durable_hann import DurableHannWindows
from corridorkey.durable_rvm import DurableRvmChunks
from corridorkey.durable_stage2 import DurableStage2Batches
from corridorkey.recovery_identity import ReplayIdentity, canonical
from corridorkey.recovery_store import BundleStore
from corridorkey.session import InferenceSession, JobRequest
from tests import test_purpose_finishing_recovery as fixtures
from tests.test_durable_stage2 import cpu_runtime_context, make_cpu_model


def frozen(value):
    """Immediate immutable bytes, never a view of subsequently mutable storage."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {'dtype':str(tensor.dtype),'shape':list(tensor.shape),
                'sha256':hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'nonzero':int(torch.count_nonzero(tensor)),
                'min':float(tensor.float().min()),'max':float(tensor.float().max())}
    if isinstance(value, np.ndarray):
        return {'dtype':str(value.dtype),'shape':list(value.shape),
                'sha256':hashlib.sha256(value.tobytes()).hexdigest(),
                'nonzero':int(np.count_nonzero(value)),'min':float(value.min()),'max':float(value.max())}
    if isinstance(value, dict):return {str(key):frozen(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):return [frozen(item) for item in value]
    return value


class MixedFinishingPreproofTests(unittest.TestCase):
    root: Path
    source: Path
    config: InferenceConfig

    setUp = fixtures.PurposeFinishingRecoveryTests.setUp
    save = fixtures.PurposeFinishingRecoveryTests.save

    def scene(self):
        texture = np.random.default_rng(271).integers(35,205,(64,64,3),dtype=np.uint8)
        texture = cv2.GaussianBlur(texture,(3,3),.7)
        for frame in range(8):
            self.assertTrue(cv2.imwrite(str(self.source/f'{frame}.png'),np.roll(texture,2*frame,axis=1)))
        self.originals = {p.name:p.read_bytes() for p in self.source.iterdir()}
        self.hints = self.root/'hints';self.hints.mkdir()
        x,y = np.meshgrid(np.arange(64),np.arange(64))
        for frame in range(8):
            self.assertTrue(cv2.imwrite(str(self.hints/f'{frame}.png'),((x+y+frame*3)%64*1024).astype(np.uint16)))
        self.initial = self.root/'initial.png'
        self.assertTrue(cv2.imwrite(str(self.initial),((x*5+y*3)%64*1024).astype(np.uint16)))
        self.config = replace(self.config,num_frames=6,hann_chunk=3,hann_stride=2,quick_interval=3,
                              hint_quality=.8,carry_hint_quality=.65,low_vram_foundation_chunk=2)

    def request(self, config, output):
        return JobRequest.create(self.source,output,config,initial_hint_path=self.initial,hint_video_path=self.hints)

    def edits(self, purpose):
        editor = CorrectionEditor(self.request(self.config,self.root/'editor-output'),self.root)
        mask = np.zeros((64,64),np.uint8);mask[16:45,12:29] = 255
        editor.commit_brush(mask,kind='matte',frame=0,strength=.6,direction='both',span=[0,6])
        ordinary = editor.document.to_dict()['entries'][0]
        region = np.zeros((64,64),np.uint8);region[:,:32] = 255
        editor.commit_brush(region,kind='region',purpose=purpose,frame=0,strength=.35,direction='both',span=[0,6])
        final_id = editor.document.to_dict()['entries'][-1]['id']
        editor.prepare_tracking(anchor_ids=[ordinary['id']],span=[0,6],direction='forward',
                                unreliable_policy='retain_approximation')
        editor.tracking_view(2);editor.accept_tracking()
        a = editor.save();a_document = editor.document
        self.assertTrue(a_document.to_dict()['tracks'])
        editor.edit(final_id,{'strength':.9})
        b = editor.save();b_document = editor.document
        self.assertEqual(b_document.to_dict()['tracks'],[])
        self.assertEqual(a_document.to_dict()['entries'][0],b_document.to_dict()['entries'][0])
        self.assertEqual(a_document.to_dict()['source'],b_document.to_dict()['source'])
        self.assertEqual(a_document.to_dict()['shots'],b_document.to_dict()['shots'])
        self.assertNotEqual(a_document.to_dict()['revision'],b_document.to_dict()['revision'])
        # A control changes a genuinely neural authored dependency, not just counts.
        editor.edit(ordinary['id'],{'strength':.2})
        control = editor.save()
        return a,b,control

    def model(self, trace, held):
        from corridorkey.color_utils import linear_to_asinh
        from tests.test_rvm_prepass import RvmPrepassTests
        model = make_cpu_model()
        helper = RvmPrepassTests();helper.setUp();self.addCleanup(helper.doCleanups)
        delattr(type(model.foundations),'rvm')
        model.foundations.rvm = helper.model.foundations.rvm
        model.foundations.rvm_mean = helper.model.foundations.rvm_mean
        model.foundations.rvm_std = helper.model.foundations.rvm_std

        def retain(value):
            if isinstance(value,(torch.Tensor,np.ndarray)):held.append((value,frozen(value)))
            elif isinstance(value,dict):
                for item in value.values():retain(item)
            elif isinstance(value,(tuple,list)):
                for item in value:retain(item)

        def observed(name, forward):
            def call(*args,**kwargs):
                before = frozen({'args':args,'kwargs':kwargs});retain(args);retain(kwargs)
                result = forward(*args,**kwargs)
                trace[name].append({'inputs':before,'outputs':frozen(result)})
                retain(result)
                return result
            return call

        def iid(*,rgb_cradio,rgb_moge):
            # Nonzero foundation features depend on the actual source RGB.
            return {'moge':rgb_moge.float().mean(dim=(1,2,3)).reshape(-1,1,1,1)}

        def stage1(rgb,hint,moge_feat,*,cradio_feat,rvm_feat,t_stride,hint_quality):
            count = rgb.shape[1]
            spatial = functional.adaptive_avg_pool2d(rgb[0].float().mean(dim=1,keepdim=True),(4,4)).unsqueeze(0)
            guidance = functional.adaptive_avg_pool2d(hint[0].float(),(4,4)).unsqueeze(0)
            q = hint_quality.float().reshape(1,count,1,1,1)
            recurrent = torch.zeros_like(moge_feat) if rvm_feat is None else rvm_feat.float()
            baton = torch.sigmoid(spatial*.2+guidance*.4+q*.03+moge_feat.float()*.1+recurrent*.08)
            aux_alpha = functional.interpolate(baton[0],size=(32,32),mode='bilinear',align_corners=False).unsqueeze(0)
            return {'baton':baton,'aux':aux_alpha.expand(-1,-1,4,-1,-1).clone()}

        def stage2(rgb,baton):
            interpolated = functional.interpolate(baton[0].float(),size=(64,64),mode='bilinear',align_corners=False)
            alpha = .2+.6*torch.sigmoid(interpolated*.7+rgb[0].float().mean(dim=1,keepdim=True)*.1)
            colour = linear_to_asinh(torch.tensor([-.2,2.4,.35])).view(1,3,1,1)+rgb[0].float()*.03
            return torch.cat((alpha,colour),dim=1).unsqueeze(0)

        model.foundations.forward = observed('iid',iid)
        model.foundations.rvm.forward = observed('rvm',model.foundations.rvm.forward)
        model.stage1.forward = observed('stage1',stage1)
        model.stage2.forward = observed('stage2',stage2)
        return model

    def run_observed(self, config, label, *, force_motion=True):
        from corridorkey import draft, runtime
        from corridorkey.motion import Propagation, interpolate_alpha, propagate_alpha
        trace: dict[str, Any] = {name:[] for name in ('iid','rvm','stage1','stage2','entries','carry','plans','draft','anchor_storage','motion')}
        held = []
        parent = ReplayIdentity(canonical({'schema_version':1,'kind':'whole_job_exr_replay','assets':{},
            'environment':{'runtime':{'synthetic':True}}}))
        store_root = self.root/('observe-'+label)
        with BundleStore(store_root,8_000_000) as store,ExitStack() as stack:
            # Real identity builders observe exact seams; all lookups are forced
            # misses and puts only snapshot, so this run cannot reuse neural work.
            adapters = {'iid':DurableFeatureFrames(store,parent,validate_context=lambda:None),
                        'rvm':DurableRvmChunks(store,parent,validate_context=lambda:None),
                        'hann':DurableHannWindows(store,parent,validate_context=lambda:None),
                        'stage2':DurableStage2Batches(store,parent,validate_context=lambda:None)}
            for name,adapter in adapters.items():
                if name == 'iid':
                    def iid_get(frame,proxy,channels,grid_hw,owner=adapter):
                        trace['entries'].append({'stage':'iid','input':frozen(proxy),
                            'identity':owner._identity(frame,proxy,channels,grid_hw).to_dict()})
                    lookup = iid_get
                else:
                    original = adapter.identity
                    def identity(*args,_name=name,_original=original,**kwargs):
                        value = _original(*args,**kwargs)
                        trace['entries'].append({'stage':_name,'arguments':frozen({'args':args,'kwargs':kwargs}),
                                                  'identity':value.to_dict() if value is not None else None})
                        return value
                    stack.enter_context(patch.object(adapter,'identity',side_effect=identity))
                    def miss(*args,**kwargs):
                        pass
                    lookup = miss
                def put(*args,_name=name,**kwargs):
                    # Tensor outputs also captured in the model seam above.
                    return False
                stack.enter_context(patch.object(adapter,'get',side_effect=lookup))
                stack.enter_context(patch.object(adapter,'put',side_effect=put))
            original_state = runtime.replay_state_digest
            def state(**kwargs):
                trace['carry'].append(frozen({key:value for key,value in kwargs.items() if key!='check'}))
                return original_state(**kwargs)
            stack.enter_context(patch.object(runtime,'replay_state_digest',side_effect=state))
            original_plan = runtime.plan_anchors
            def plan(*args,**kwargs):
                value = original_plan(*args,**kwargs);trace['plans'].append(asdict(value));return value
            stack.enter_context(patch.object(runtime,'plan_anchors',side_effect=plan))
            originals = {'offline':draft.offline_frames,'past':draft.past_frames}
            def forced(*args,**kwargs):
                # Depend on actual incoming alpha; force only uncertainty/motion
                # to exercise real neural-fallback and promotion control paths.
                candidates = [value for value in args if isinstance(value,np.ndarray) and value.ndim==2]
                alpha = candidates[0].copy() if candidates else np.full((64,64),.4,np.float32)
                return Propagation(alpha,np.zeros_like(alpha),np.ones_like(alpha,dtype=bool),
                    {'flow_evaluations':4,'motion_seconds':0.,'uncertain_fraction':1.,'motion_fraction':1.})
            def generator(kind):
                def frames(*args,**kwargs):
                    def motion(*args,**kwargs):
                        trace['motion'].append({'kind':kind,'args':frozen(args),
                                                'settings':asdict(kwargs['settings'])})
                        if force_motion:
                            return forced(*args,**kwargs)
                        return (interpolate_alpha if kind=='offline' else propagate_alpha)(*args,**kwargs)
                    kwargs['interpolate' if kind=='offline' else 'propagate'] = motion
                    stream = originals[kind](*args,**kwargs)
                    for frame in stream:
                        local = stream.gi_frame.f_locals
                        retained = ([value[1] for value in local.get('cached',{}).values()]
                                    if kind=='offline' else [local['anchor_alpha']])
                        snapshots = [(array,frozen(array)) for array in retained if array is not None]
                        self.assertTrue(all(not np.shares_memory(frame.alpha,array) for array,_ in snapshots))
                        trace['anchor_storage'].append({'frame':frame.index,'retained':[value for _,value in snapshots]})
                        trace['draft'].append({'frame':frame.index,'preconstraint_alpha':frozen(frame.alpha),
                                               'events':frozen({key:value for key,value in frame.diagnostics.items()
                                                                if key!='motion_seconds'})})
                        yield frame
                        for array,before in snapshots:self.assertEqual(frozen(array),before,'output edit mutated cached anchor storage')
                return frames
            stack.enter_context(patch.object(draft,'offline_frames',side_effect=generator('offline')))
            stack.enter_context(patch.object(draft,'past_frames',side_effect=generator('past')))
            session = InferenceSession(loader=lambda:self.model(trace,held),disposer=lambda _:None)
            try:
                result = session.run(self.request(config,self.root/('output-'+label)),
                    durable_foundations=adapters['iid'],durable_rvm=adapters['rvm'],
                    durable_hann=adapters['hann'],durable_stage2=adapters['stage2']).to_dict()
            finally:session.close()
            self.assertEqual(list((store_root/'entries').iterdir()),[],'preproof must be uncached')
        for value,before in held:self.assertEqual(frozen(value),before,'final constraints mutated neural input/output/state storage')
        quick = result['diagnostics'].get('quick_key')
        if quick:
            trace['quick'] = {key:quick.get(key) for key in ('anchors','extra_neural_frames','promoted_anchors',
                'promotion_reasons','fallback_frames','effective_cuts','approach','native_context_policy')}
        pixels = {role:[cv2.imread(str(path),-1) for path in sorted(Path(directory).glob('*.exr'))]
                  for role,directory in result['diagnostics']['outputs'].items()}
        return trace,pixels

    def test_actual_editor_track_retirement_preserves_neural_and_temporal_seams(self):
        self.scene()
        rows = []
        cases = [('full',{}),('low',{'low_vram':True}),('spatial',{'model_scale':.5}),
                 ('hold',{'quick_key':True,'quick_approach':'hold'}),
                 ('offline',{'quick_key':True}),('causal',{'quick_key':True,'quick_approach':'causal'}),
                 ('offline_real',{'quick_key':True}),('causal_real',{'quick_key':True,'quick_approach':'causal'}),
                 ('offline_adaptive',{'quick_key':True,'quick_adaptive':True}),
                 ('causal_adaptive',{'quick_key':True,'quick_approach':'causal','quick_adaptive':True})]
        with cpu_runtime_context():
            for purpose in ('composite','diagnostic'):
                a,b,control = self.edits(purpose)
                for mode,overrides in cases:
                    with self.subTest(purpose=purpose,mode=mode):
                        config = replace(self.config,**overrides)
                        left,left_pixels = self.run_observed(replace(config,corrections_path=a),purpose+'-'+mode+'-a',force_motion=not mode.endswith('_real'))
                        right,right_pixels = self.run_observed(replace(config,corrections_path=b),purpose+'-'+mode+'-b',force_motion=not mode.endswith('_real'))
                        self.assertEqual(left,right,'ordinary-track retirement changed neural/schedule/state input')
                        for x,y in zip(left_pixels['fg'],right_pixels['fg']):
                            np.testing.assert_array_equal(x,y)
                        alpha_diffs = [int(np.count_nonzero(x!=y)) for x,y in zip(left_pixels['alpha'],right_pixels['alpha'])]
                        self.assertGreater(sum(alpha_diffs),0,'retirement witness must change final alpha')
                        self.assertTrue(all(left[name] for name in ('iid','rvm','stage1','stage2','entries','carry')))
                        self.assertTrue(any(event['inputs']['kwargs']['state'] is not None for event in left['rvm']))
                        self.assertTrue(any(event['inputs']['args'][1]['nonzero']>0 for event in left['stage1']))
                        if mode in ('full','low','spatial'):
                            self.assertTrue(any(state['carry'] for state in left['carry']),'exercise nonempty carried state')
                        if mode in ('offline','causal','offline_adaptive','causal_adaptive'):
                            self.assertTrue(left['quick']['extra_neural_frames'])
                        if mode in ('causal','causal_adaptive','offline_adaptive'):
                            self.assertTrue(left['quick']['promoted_anchors'])
                        rows.append({'purpose':purpose,'mode':mode,'identical_all_seams':True,
                            'alpha_different_pixels':alpha_diffs,'final_outputs_a':frozen(left_pixels),
                            'final_outputs_b':frozen(right_pixels),'raw_foreground_equal':True,'trace_sha256':hashlib.sha256(canonical(left).encode()).hexdigest(),
                            'incidental_event_fields_excluded':['motion_seconds'],
                            'motion_forced':not mode.endswith('_real'),'source_start':config.start_frame,'num_frames':config.num_frames,
                            'calls':{name:len(left[name]) for name in ('iid','rvm','stage1','stage2')},
                            'quick':left.get('quick'),'trace':left})
                changed,_ = self.run_observed(replace(self.config,corrections_path=control),purpose+'-sensitivity')
                reference = next(row for row in rows if row['purpose']==purpose and row['mode']=='full')['trace']
                self.assertNotEqual(reference['stage1'],changed['stage1'],'fixture must react to authored hint strength')
                self.assertNotEqual(reference['stage2'],changed['stage2'],'fixture outputs must depend on neural inputs/state')
                output_controls = {}
                for stage in ('stage1','stage2'):
                    old_outputs = [event['outputs'] for event in reference[stage]]
                    new_outputs = [event['outputs'] for event in changed[stage]]
                    self.assertNotEqual(old_outputs,new_outputs,'raw model outputs must be input-sensitive')
                    output_controls[stage] = {'reference':hashlib.sha256(canonical(old_outputs).encode()).hexdigest(),
                                              'changed':hashlib.sha256(canonical(new_outputs).encode()).hexdigest()}
                for row in rows:
                    if row['purpose']==purpose:
                        row['sensitivity_control'] = {'stage1_changed':True,'stage2_changed':True,
                            'raw_output_digests':output_controls,
                            'trace_sha256':hashlib.sha256(canonical(changed).encode()).hexdigest()}
        self.assertEqual({p.name:p.read_bytes() for p in self.source.iterdir()},self.originals)
        destination = os.environ.get('CORRIDORKEY_MIXED_PREPROOF_DIR')
        if destination:
            root = Path(destination);root.mkdir(parents=True,exist_ok=True)
            (root/'matrix.json').write_text(json.dumps(rows,indent=2))
