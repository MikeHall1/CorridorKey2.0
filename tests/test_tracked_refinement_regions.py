"""Moving final-finishing weights; CPU fixtures are not neural/quality evidence."""
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.correction_rasters import CorrectionRasters
from corridorkey.correction_tracking import (
    TrackingAnchor,
    TrackingFrame,
    TrackingRequest,
    track_corrections,
)
from corridorkey.correction_tracking_context import tracking_decode
from corridorkey.corrections import CorrectionDocument, canonical, track_binding


class RegionTrackingCoreTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root=Path(folder.name).resolve()
        self.shape=(16,20)
        self.meta={'native_hw':self.shape,'start_frame':100,'num_frames':6,'native_fps':24.}
        self.source={'sha256':'a'*64,'start_frame':100,'frame_count':6,'fps':24.,'height':16,'width':20}
        self.mask=np.zeros(self.shape,np.float32);self.mask[4:11,4:9]=.73
        self.reads=[]

    @staticmethod
    def flow(previous,following):
        flow=np.zeros((*previous.shape,2),np.float32)
        flow[...,0]=float(following[0,0])-float(previous[0,0])
        return flow

    def read(self,absolute):
        self.reads.append(absolute)
        return TrackingFrame('a'*64,absolute,np.full((*self.shape,3),(absolute-100)/255,np.float32))

    def build(self,direction='forward', *, both_anchors=False, overlap=False, cuts=()):
        selected=[(1,self.mask)] if direction!='backward' else [(4,np.roll(self.mask,3,axis=1))]
        if both_anchors:selected=[(1,self.mask),(4,np.roll(self.mask,3,axis=1))]
        entries=[];anchors=[]
        for number,(frame,mask) in enumerate(selected):
            path=self.root/f'anchor{number}.exr'
            self.assertTrue(cv2.imwrite(str(path),mask))
            entries.append({'id':f'anchor{number}','revision':1,'frame':frame,'kind':'region',
                'purpose':'refinement','target':'edge_finishing','direction':'both','strength':.4,
                'range':[0,6],'asset':{'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}})
            anchors.append(TrackingAnchor.create(f'anchor{number}',1,frame,'region',.4,mask))
        if overlap:
            path=self.root/'static.exr';self.assertTrue(cv2.imwrite(str(path),np.full(self.shape,.2,np.float32)))
            entries.append({**entries[0],'id':'static','strength':.6,
                'asset':{'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}})
        data={'schema_version':5,'revision':1,'source':self.source,'shots':[[0,6]],'entries':entries,'tracks':[],'guidance':[]}
        request=TrackingRequest('a'*64,1,5,source_start=100,direction=direction,cuts=cuts)
        candidates=list(track_corrections(request,tuple(anchors),self.read,estimator=self.flow,estimator_identity='fixture-flow-v1'))
        settings={k:v for k,v in asdict(request).items() if k in {'direction','unreliable_policy','max_uncertain_fraction','motion','max_frames','max_pixels','max_anchor_bytes','max_anchors'}}
        settings.update(kernel_version='correction-tracking-v1',estimator_identity='fixture-flow-v1')
        binding=track_binding(data,[e['id'] for e in entries[:len(anchors)]],cuts=list(cuts),
            decode=tracking_decode(InferenceConfig(),self.root),tracking=settings)
        rows=[];observations=[]
        for candidate in sorted(candidates,key=lambda c:c.local_frame):
            mask=candidate.mask()
            if mask is None:continue
            path=self.root/f'derived{candidate.local_frame}.exr';self.assertTrue(cv2.imwrite(str(path),mask))
            rows.append({'frame':candidate.local_frame,'asset':{'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}})
            provenance=json.loads(candidate.provenance_json)
            observations.append({'frame':candidate.local_frame,'source_frame':candidate.source_frame,'status':candidate.status,
                'candidate_identity':candidate.identity,'source_proxy_sha256':provenance['source_proxy_sha256'],
                'derived_mask_sha256':provenance['derived_mask_sha256'],'diagnostics':json.loads(candidate.diagnostics_json)})
        data['tracks']=[{'id':'track','owner_id':entries[0]['id'],'anchor_ids':[a.identifier for a in anchors],
            'range':[1,5],'direction':direction,'binding':binding,'binding_sha256':hashlib.sha256(canonical(binding).encode()).hexdigest(),
            'frames':rows,'observations':observations}]
        return data,candidates

    def test_soft_region_tracks_forward_backward_both_only_gate_final_weights_once(self):
        for direction in ('forward','backward','both'):
            with self.subTest(direction=direction):
                data,candidates=self.build(direction,both_anchors=direction=='both',overlap=True)
                document=CorrectionDocument.parse(data,self.root)
                rasters=CorrectionRasters(document,self.meta)
                self.addCleanup(rasters.close)
                for candidate in candidates:
                    mask=candidate.mask();assert mask is not None
                    weight=rasters.refinement_region(candidate.local_frame)
                    assert weight is not None
                    np.testing.assert_array_equal(weight,np.maximum(mask*np.float32(.4),np.float32(.2)*np.float32(.6)))
                    alpha=np.full(self.shape,.3,np.float32)
                    np.testing.assert_array_equal(rasters.apply(alpha,candidate.local_frame),alpha)
                    np.testing.assert_array_equal(rasters.apply(alpha,candidate.local_frame,temporal=True),alpha)
                    self.assertFalse(rasters.has_effect(candidate.local_frame))
                    self.assertFalse(rasters.has_condition(candidate.local_frame))
                    self.assertEqual(rasters.condition(candidate.local_frame),(None,None))
                self.assertEqual(document.frames,())
                outside=np.maximum(self.mask*np.float32(.4),np.float32(.12))
                if direction=='backward':outside=np.maximum(np.roll(self.mask,3,axis=1)*np.float32(.4),np.float32(.12))
                if direction=='both':outside=np.maximum(outside,np.roll(self.mask,3,axis=1)*np.float32(.4))
                weight=rasters.refinement_region(0);assert weight is not None
                np.testing.assert_allclose(weight,outside,rtol=0,atol=1e-8)
        self.assertTrue(all(101<=frame<=104 for frame in self.reads))

    def test_region_purpose_target_strength_and_track_assets_remain_bound(self):
        data,_=self.build('both',both_anchors=True)
        for field,value in (('purpose','neural'),('target','alpha_refine'),('strength',.2)):
            altered=copy.deepcopy(data);altered['entries'][1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):CorrectionDocument.parse(altered,self.root)
        derived=self.root/data['tracks'][0]['frames'][1]['asset']['path']
        derived.write_bytes(derived.read_bytes()+b'corrupt')
        with self.assertRaises(ValueError):CorrectionRasters(CorrectionDocument.parse(data,self.root),self.meta)

    def test_cut_stops_and_partial_track_cannot_be_accepted(self):
        data,candidates=self.build(cuts=(3,))
        self.assertTrue(any(c.mask() is None for c in candidates))
        with self.assertRaises(ValueError):CorrectionDocument.parse(data,self.root)

    def test_occlusion_and_cancel_are_visible_without_relabeling_coverage(self):
        from corridorkey.control import CancelledError, JobControl
        anchor=TrackingAnchor.create('region',0,1,'region',.4,self.mask)
        request=TrackingRequest('a'*64,1,5,source_start=100,direction='forward')
        def occluded(absolute):
            if absolute>=103:
                return TrackingFrame('a'*64,absolute,np.ones((*self.shape,3),np.float32))
            return self.read(absolute)
        rows=list(track_corrections(request,(anchor,),occluded,estimator=self.flow,estimator_identity='fixture-flow-v1'))
        self.assertTrue(any(r.mask() is None and r.status!='authored_anchor' for r in rows))
        def cancel():raise CancelledError('stop region tracking')
        with self.assertRaises(CancelledError):
            list(track_corrections(request,(anchor,),self.read,control=JobControl(check_hook=cancel)))
        np.testing.assert_array_equal(anchor.mask(),self.mask)


class RegionEditorTests(unittest.TestCase):
    def setUp(self):
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.session import JobRequest
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.root=Path(folder.name).resolve();self.source=self.root/'source';self.source.mkdir()
        self.shape=(24,32)
        for index in range(6):
            self.assertTrue(cv2.imwrite(str(self.source/f'frame{index}.png'),np.full((*self.shape,3),100,np.uint8)))
        self.originals={p.name:p.read_bytes() for p in self.source.iterdir()}
        self.job=JobRequest.create(self.source,self.root/'output',InferenceConfig(start_frame=1,num_frames=4,output_format='exr'))
        self.editor=CorrectionEditor(self.job,self.root)
        self.mask=np.zeros(self.shape,np.float32);self.mask[6:18,8:24]=.73
        path=self.root/'soft.exr';self.assertTrue(cv2.imwrite(str(path),self.mask))
        self.editor.import_mask(path,kind='region',frame=1,strength=.4,direction='both',span=[0,4])
        self.entry=self.editor.document.to_dict()['entries'][0]

    def prepare(self, direction='forward', span=None):
        self.editor.prepare_tracking(anchor_ids=[self.entry['id']],span=span or [1,4],direction=direction)

    def accept(self):
        self.prepare();self.editor.tracking_view(2);self.editor.accept_tracking()

    def test_soft_region_inspect_accept_overlay_save_reopen_undo_and_edit_invalidation(self):
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.corrections import load_corrections, save_corrections
        original=self.editor.document.identity
        self.prepare()
        preview=self.editor.tracking_preview;assert preview is not None
        anchor=preview['track']['binding']['anchors'][0]
        self.assertEqual((anchor['kind'],anchor['purpose'],anchor['target'],anchor['strength']),('region','refinement','edge_finishing',.4))
        self.assertEqual(self.editor.document.identity,original)
        with self.assertRaises(ValueError):self.editor.accept_tracking()
        self.editor.tracking_view(2);self.editor.accept_tracking()
        accepted=self.editor.document.identity
        _,overlay=self.editor.view(2,identifier=self.entry['id'])
        assert overlay is not None
        np.testing.assert_array_equal(overlay,np.rint(self.mask*255).astype(np.uint8))
        rasters=CorrectionRasters(self.editor.document,self.editor.meta);self.addCleanup(rasters.close)
        for frame in (0,2):
            weight=rasters.refinement_region(frame);assert weight is not None
            np.testing.assert_array_equal(weight,self.mask*np.float32(.4))
        path=self.editor.save();before=path.read_bytes()
        reopened=CorrectionEditor(self.job,self.root,existing=path)
        self.assertEqual(reopened.document.identity,accepted)
        relocated=self.root/'moved'/'regions.json';save_corrections(relocated,self.editor.document)
        self.assertEqual(load_corrections(relocated,source=self.source).identity,accepted)
        self.editor.undo();self.assertEqual(self.editor.document.identity,original)
        self.editor.redo();self.assertEqual(self.editor.document.identity,accepted)
        self.editor.edit(self.entry['id'],{'strength':.7})
        self.assertEqual(self.editor.document.to_dict()['tracks'],[])
        self.assertEqual(self.editor.invalidated,[(0,4)])
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual({p.name:p.read_bytes() for p in self.source.iterdir()},self.originals)

    def test_accepted_track_portable_collection_and_tampered_overlay_refusal(self):
        from corridorkey.corrections import load_corrections
        from corridorkey.portable_project import (
            inspect_project,
            plan_project,
            publish_project,
        )
        from corridorkey.session import JobRequest
        self.accept();path=self.editor.save()
        cfg,writer=self.job.configurations()
        job=JobRequest.create(self.source,self.root/'output',replace(cfg,corrections_path=path),writer)
        packages=self.root/'packages';packages.mkdir()
        plan=plan_project(job,0,collect_ids=('source','corrections'),read_budget=10**7,copy_budget=10**7)
        package=publish_project(plan,packages)
        (self.root/'destination').mkdir();(self.root/'scratch').mkdir()
        moved=inspect_project(package,read_budget=10**7).apply(output_dir=self.root/'destination',temp_dir=self.root/'scratch')
        cfg,_=moved.configurations();assert cfg.corrections_path is not None
        self.assertEqual(load_corrections(cfg.corrections_path,source=moved.input_path).identity,self.editor.document.identity)
        track=self.editor.document.to_dict()['tracks'][0]
        derived=self.editor.document.base_dir/track['frames'][1]['asset']['path']
        derived.write_bytes(derived.read_bytes()+b'corrupt')
        with self.assertRaises(ValueError):self.editor.view(2,identifier=self.entry['id'])

    def test_partial_cancelled_cut_or_stale_source_cannot_be_accepted(self):
        from unittest.mock import patch

        from corridorkey.control import CancelledError
        before=self.editor.document.identity
        with patch('corridorkey.correction_tracking_editor.track_corrections',return_value=iter(())):
            self.prepare()
        with self.assertRaises(ValueError):self.editor.accept_tracking()
        self.assertEqual(self.editor.document.identity,before)
        self.editor.discard_tracking()
        with patch('corridorkey.correction_tracking_editor.track_corrections',side_effect=CancelledError('fixture cancellation')),self.assertRaises(CancelledError):self.prepare()
        self.assertIsNone(self.editor.tracking_preview)
        self.prepare();self.editor.tracking_view(2)
        source=self.source/'frame3.png';source.write_bytes(source.read_bytes()+b'drift')
        with self.assertRaisesRegex(ValueError,'Source changed'):self.editor.accept_tracking()
        self.assertEqual(self.editor.document.identity,before)

    def test_backward_and_both_region_tracking_keep_authored_direction_and_stop_at_reset(self):
        for direction,span in (('backward',[0,2]),('both',[0,4])):
            with self.subTest(direction=direction):
                self.prepare(direction,span)
                preview=self.editor.tracking_preview;assert preview is not None
                self.assertTrue(preview['complete'])
                self.assertEqual(preview['track']['direction'],direction)
                self.assertEqual(self.editor.document.to_dict()['entries'][0]['direction'],'both')
                self.editor.discard_tracking()
        self.editor.reset(2)
        with self.assertRaisesRegex(ValueError,'boundary'):self.prepare()

    def test_accepted_overlay_rejects_changed_authored_anchor(self):
        self.accept()
        anchor=self.editor.document.asset_path(self.entry)
        anchor.write_bytes(anchor.read_bytes()+b'changed authored bytes')
        with self.assertRaises(ValueError):self.editor.view(2,identifier=self.entry['id'])

    def test_accepted_overlay_rejects_changed_source_bytes(self):
        self.accept()
        source=self.source/'frame3.png'
        source.write_bytes(source.read_bytes()+b'changed source bytes')
        with self.assertRaises(ValueError):self.editor.view(2,identifier=self.entry['id'])

    def _late_source_publication_race(self, *, add_frame):
        from unittest.mock import patch

        from corridorkey import (
            correction_editor,
            correction_tracking_editor,
            identities,
        )
        self.accept()
        display,overlay=self.editor.view(2,identifier=self.entry['id'])
        self.assertIsNotNone(overlay)
        np.testing.assert_array_equal(display.pixels[0,0], [100,100,100])
        original_hash,original_preview=identities.file_sha256,correction_editor.preview
        decoded=False
        changed=False
        def read_preview(*args,**kwargs):
            nonlocal decoded
            result=original_preview(*args,**kwargs)
            decoded=True
            return result
        def hash_then_change(path,**kwargs):
            nonlocal changed
            result=original_hash(path,**kwargs)
            if decoded and not changed and Path(path).name=='frame4.png':
                changed=True
                destination=self.source/('frame99.png' if add_frame else 'frame3.png')
                self.assertTrue(cv2.imwrite(str(destination),np.full((*self.shape,3),150,np.uint8)))
            return result
        with patch.object(correction_editor,'preview',side_effect=read_preview), \
             patch.object(identities,'file_sha256',side_effect=hash_then_change), \
             patch.object(correction_tracking_editor,'file_sha256',side_effect=hash_then_change), \
             self.assertRaises(ValueError):
            self.editor.view(2,identifier=self.entry['id'])
        self.assertTrue(changed,'Real hash-forwarding race must fire after actual preview, not be bypassed by a changed helper')

    def test_accepted_overlay_rejects_displayed_source_change_during_later_real_hash(self):
        self._late_source_publication_race(add_frame=False)

    def test_accepted_overlay_rejects_supported_membership_addition_during_real_hash(self):
        self._late_source_publication_race(add_frame=True)

    def test_accepted_overlay_rejects_displayed_path_swap_after_real_decode(self):
        from unittest.mock import patch

        from corridorkey import correction_editor
        self.accept()
        original=correction_editor.preview
        changed=False
        def decode_then_replace(*args,**kwargs):
            nonlocal changed
            result=original(*args,**kwargs)
            replacement=self.root/'replacement.png'
            self.assertTrue(cv2.imwrite(str(replacement),np.full((*self.shape,3),150,np.uint8)))
            replacement.replace(self.source/'frame3.png')
            changed=True
            return result
        with patch.object(correction_editor,'preview',side_effect=decode_then_replace),self.assertRaisesRegex(ValueError,'changed'):
            self.editor.view(2,identifier=self.entry['id'])
        self.assertTrue(changed)

    def test_accepted_overlay_preserves_shape_and_cancellation_errors(self):
        from unittest.mock import patch

        from corridorkey import correction_editor
        from corridorkey.control import CancelledError
        self.accept()
        original=correction_editor.preview
        decoded=False
        def decode(*args,**kwargs):
            nonlocal decoded
            result=original(*args,**kwargs);decoded=True
            return result
        def cancel():
            if decoded:raise CancelledError('cancel accepted region inspection')
        self.editor.check=cancel
        with patch.object(correction_editor,'preview',side_effect=decode),self.assertRaises(CancelledError):
            self.editor.view(2,identifier=self.entry['id'])
        self.editor.check=lambda:None
        self.assertTrue(cv2.imwrite(str(self.source/'frame3.png'),np.full((8,12,3),150,np.uint8)))
        with self.assertRaisesRegex(ValueError,'geometry changed'):
            self.editor.view(2,identifier=self.entry['id'])


class RegionTrackingRuntimeTests(unittest.TestCase):
    def test_actual_native_exports_all_paths_are_final_only_with_hints_foreground_unchanged(self):
        from unittest.mock import patch

        import torch

        from corridorkey import draft
        from corridorkey.correction_editor import CorrectionEditor
        from corridorkey.motion import Propagation
        from corridorkey.refinement import finish_alpha
        from corridorkey.session import JobRequest
        from tests import test_spatial_draft as fixture
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        root=Path(folder.name).resolve()
        helper=fixture.SpatialDraftTests();helper.root=root
        source=helper.source(exr=True)
        original={p.name:p.read_bytes() for p in source.iterdir()}
        job=JobRequest.create(source,root/'editor-output',InferenceConfig(start_frame=1,num_frames=4,frame_dir_linear=True,output_format='exr'))
        editor=CorrectionEditor(job,root)
        mask=np.zeros((65,97),np.float32);mask[:,15:60]=.73
        authored=root/'soft.exr';self.assertTrue(cv2.imwrite(str(authored),mask))
        editor.import_mask(authored,kind='region',frame=0,strength=.5,direction='both',span=[0,1])
        identifier=editor.document.to_dict()['entries'][0]['id']
        editor.prepare_tracking(anchor_ids=[identifier],span=[0,4],direction='forward',unreliable_policy='retain_approximation')
        editor.tracking_view(2);editor.accept_tracking();manifest=editor.save()
        rasters=CorrectionRasters(editor.document,editor.meta);self.addCleanup(rasters.close)
        factory,frames=fixture.make_spatial_cpu_model,draft.offline_frames
        def patterned(**kwargs):
            model=factory(**kwargs);forward=model.stage2.forward
            def render(rgb,baton):
                logits=forward(rgb,baton)
                values=torch.arange(logits.shape[-1]).remainder(13).float()/12
                logits[:,:,0]=values.view(1,1,1,-1)
                return logits
            model.stage2.forward=render
            return model
        def uncertain(_first,_last,_target,alpha,_right,_weight,**kwargs):
            return Propagation(alpha,np.zeros_like(alpha),np.ones_like(alpha,dtype=bool),
                {'flow_evaluations':4,'motion_seconds':0.,'uncertain_fraction':1.,'motion_fraction':0.})
        def forced(*args,**kwargs):return frames(*args,interpolate=uncertain,**kwargs)
        with patch.object(fixture,'make_spatial_cpu_model',patterned),patch.object(draft,'offline_frames',forced):
            for scale,low,quick in ((1.,False,False),(1.,True,False),(.5,True,False),(1.,False,True),(.5,True,True)):
                with self.subTest(scale=scale,low=low,quick=quick):
                    overrides={'start_frame':1,'num_frames':4,'cutout_alpha_mode':'premultiplied'}
                    kwargs={'scale':scale,'low':low,'quick':quick,'refine':True,'outputs':{'alpha','fg','cutout'}}
                    baseline_model,baseline,_=helper.run_job(source,overrides=overrides,**kwargs)
                    model,actual,_=helper.run_job(source,correction=manifest,
                        overrides={**overrides,'alpha_expand_radius':1,'alpha_feather_radius':2},**kwargs)
                    self.assertEqual(model.stage2.shapes,baseline_model.stage2.shapes)
                    self.assertEqual(len(model.stage1.seen_hints),len(baseline_model.stage1.seen_hints))
                    for a,b in zip(model.stage1.seen_hints,baseline_model.stage1.seen_hints,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
                    for index,(a,b) in enumerate(zip(sorted(Path(baseline.outputs['alpha']).glob('*.exr')),sorted(Path(actual.outputs['alpha']).glob('*.exr')),strict=True)):
                        raw=cv2.imread(str(a),cv2.IMREAD_UNCHANGED);result=cv2.imread(str(b),cv2.IMREAD_UNCHANGED)
                        weight=rasters.refinement_region(index)
                        np.testing.assert_array_equal(result,finish_alpha(raw,expansion=1,feather=2,region=weight))
                    for a,b,c in zip(sorted(Path(baseline.outputs['fg']).glob('*.exr')),sorted(Path(actual.outputs['fg']).glob('*.exr')),sorted(Path(actual.outputs['cutout']).glob('*.exr')),strict=True):
                        self.assertEqual(a.read_bytes(),b.read_bytes())
                        fg=cv2.imread(str(b),cv2.IMREAD_UNCHANGED);cutout=cv2.imread(str(c),cv2.IMREAD_UNCHANGED)
                        np.testing.assert_allclose(cutout[...,:3],fg*cutout[...,3:4],rtol=0,atol=1e-6)
                    assert actual.corrections is not None
                    self.assertEqual(actual.corrections['hint_frame_evaluations_conditioned'],0)
                    self.assertEqual(actual.corrections['output_frames_refined'],0)
                    if not quick and not low:
                        _,disabled,_=helper.run_job(source,correction=manifest,overrides=overrides,**kwargs)
                        for kind in ('alpha','fg','cutout'):
                            for a,b in zip(sorted(Path(baseline.outputs[kind]).glob('*.exr')),sorted(Path(disabled.outputs[kind]).glob('*.exr')),strict=True):self.assertEqual(a.read_bytes(),b.read_bytes())
        self.assertEqual({p.name:p.read_bytes() for p in source.iterdir()},original)


class RegionTrackingPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import test_desktop_tracking as fixture
        fixture.DesktopTrackingTests.setUpClass()

    def setUp(self):
        from tests import test_desktop_tracking as fixture
        self.fixture=fixture.DesktopTrackingTests();self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def test_region_tracking_direction_is_distinct_and_accepted_overlay_is_visible(self):
        f=self.fixture;dialog=f.dialog
        dialog.request('brush',mask=np.full((24,32),255,np.uint8),kind='region',frame=0,
                       strength=.4,direction='both',span=[0,1])
        f.until(lambda:len(dialog.entries)==2 and dialog.pending is None)
        dialog.table.setCurrentCell(1,0);dialog.select()
        self.assertEqual(dialog.direction.currentText(),'both')
        self.assertFalse(dialog.direction.isEnabled())
        self.assertTrue(dialog.region_tracking_direction.isEnabled())
        dialog.region_tracking_direction.setCurrentText('forward')
        dialog.add_tracking_anchor();dialog.start.setValue(0);dialog.end.setValue(4)
        dialog.preview_tracking();f.until(lambda:dialog.pending is None)
        assert dialog.snapshot is not None
        self.assertEqual(dialog.snapshot['tracking_preview']['request']['direction'],'forward')
        self.assertEqual(dialog.entries[1]['direction'],'both')
        dialog.frame.setValue(2);dialog.inspect_tracking();f.until(lambda:dialog.pending is None)
        dialog.request('track_accept');f.until(lambda:dialog.pending is None)
        dialog.frame.setValue(2);dialog.load_frame();f.until(lambda:dialog.pending is None and dialog.displayed_frame==2)
        assert dialog.canvas.saved_overlay is not None
        self.assertTrue(np.all(dialog.canvas.saved_overlay==255))
        self.assertFalse(dialog.service.done.is_set())
