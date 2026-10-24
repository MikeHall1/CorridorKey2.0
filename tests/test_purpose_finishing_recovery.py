"""Pure-region A/editB/cleanB CPU exports through the actual recovery runner."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import torch

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.corrections import (
    CorrectionDocument,
    load_corrections,
    save_corrections,
)
from corridorkey.identities import file_sha256
from corridorkey.recovery import run_recoverable
from corridorkey.session import InferenceSession, JobRequest

# Reuse the established exact-entry adversarial fixtures in the v3 namespace.
from tests import test_durable_stage2 as stage2_fixtures
from tests import test_upstream_finishing_adapters as adapter_fixtures
from tests.test_durable_stage2 import cpu_runtime_context, make_cpu_model
from tests.test_purpose_finishing_identity import parent


class PurposeFinishingRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.source = self.root/'source';self.source.mkdir()
        for frame in range(5):
            image = np.empty((64,64,3),np.uint8);image[:] = (50,180,80)
            self.assertTrue(cv2.imwrite(str(self.source/f'{frame}.png'),image))
        self.originals = {p.name:p.read_bytes() for p in self.source.iterdir()}
        self.implementation = self.root/'implementation'
        (self.implementation/'corridorkey').mkdir(parents=True)
        (self.implementation/'corridorkey/runtime.py').write_text('# synthetic CPU fixture pin')
        self.assets = {}
        for name in ('weights','foundation_stats','rvm_ckpt','moge_checkpoint'):
            self.assets[name] = self.root/name;self.assets[name].write_bytes(b'local CPU fixture, not pretrained')
        self.assets['cradio_dir'] = self.root/'cradio';self.assets['cradio_dir'].mkdir()
        (self.assets['cradio_dir']/'config.json').write_text('{}')
        self.environment = {'backend':'cpu-test-fixture','precision':'fixture-float32',
                            'dependencies':{'fixture':'1'},'runtime':{'synthetic':True}}
        self.models = []
        self.config = InferenceConfig(hann_chunk=3,hann_stride=2,shape_quantum=64,stage2_batch=2,
            start_frame=1,num_frames=3,output_format='exr',exr_dtype='float',
            outputs={'alpha','fg','cutout','checker'},cutout_alpha_mode='premultiplied',
            despill_strength=.65,despill_space='linear',temp_dir=self.root/'temp',
            quick_interval=2,quick_context_radius=1,quick_fallback='neural')

    def factory(self, **kwargs):
        def load():
            from corridorkey.color_utils import linear_to_asinh
            from tests.test_rvm_prepass import RvmPrepassTests
            model = make_cpu_model()
            helper = RvmPrepassTests();helper.setUp();self.addCleanup(helper.doCleanups)
            delattr(type(model.foundations),'rvm')
            model.foundations.rvm = helper.model.foundations.rvm
            model.foundations.rvm_mean = helper.model.foundations.rvm_mean
            model.foundations.rvm_std = helper.model.foundations.rvm_std
            model.fixture_rvm_calls = helper.calls
            model.fixture_iid_calls = [];model.fixture_stage1_calls = []
            iid,stage1,stage2 = model.foundations.forward,model.stage1.forward,model.stage2.forward
            def foundations(**inputs):
                model.fixture_iid_calls.append(len(inputs['rgb_moge']))
                return iid(**inputs)
            def first(*args,**inputs):
                model.fixture_stage1_calls.append(args[0].shape[1]);return stage1(*args,**inputs)
            def second(rgb,baton):
                result = stage2(rgb,baton)
                result[:,:,0] = (torch.arange(64)%13/12).view(1,1,1,64)
                result[:,:,1:4] = linear_to_asinh(torch.tensor([-.2,2.4,.35]).view(1,1,3,1,1))
                return result
            model.foundations.forward = foundations;model.stage1.forward = first;model.stage2.forward = second
            self.models.append(model)
            return model
        return InferenceSession(loader=load,disposer=lambda _:None,**kwargs)

    def job(self, config=None):
        return JobRequest.create(self.source,self.root/'output',config or self.config)

    def document(self, purpose, *, changed=False, tracked=False, config=None):
        editor = CorrectionEditor(self.job(config),self.root)
        mask = np.zeros((64,64),np.uint8)
        mask[:,32:] = 255 if changed else 0
        if not changed:mask[:,:32] = 255
        editor.commit_brush(mask,kind='region',purpose=purpose,frame=0,strength=.7 if changed else .5,
                            direction='both',span=[0,3])
        if tracked:
            entry = editor.document.to_dict()['entries'][0]
            editor.prepare_tracking(anchor_ids=[entry['id']],span=[0,3],direction='both' if changed else 'forward')
            editor.tracking_view(1);editor.accept_tracking()
        return editor.document

    def save(self, document, name):
        path = self.root/name
        save_corrections(path,document)
        return path

    def run_job(self, request, workspace, *, policies=(True,True), budget=8_000_000, progress=None):
        stage2,upstream = policies
        return run_recoverable([request],workspace=workspace,max_cache_bytes=budget,
            assets=self.assets,environment=self.environment,implementation_root=self.implementation,
            session_factory=self.factory,durable_foundations=True,durable_rvm=True,durable_hann=True,
            durable_stage2=True,durable_writer_prefix=True,reuse_stage2_finishing_edits=stage2,
            reuse_upstream_finishing_edits=upstream,progress=progress)

    @staticmethod
    def hashes(record):
        return {f'{role}/{p.name}':file_sha256(p) for role,directory in record['outputs'].items()
                for p in Path(directory).glob('*.exr')}

    def compare(self, purpose, variant, policies, *, config=None):
        config = config or self.config
        first_doc = self.document(purpose,tracked=variant!='static',config=config)
        if variant == 'provenance':
            data = first_doc.to_dict();data['revision'] += 1;data['tracks'][0]['id'] += '-reinspected'
            data['tracks'][0]['observations'][1]['diagnostics']['artist_review'] = 'reinspected, no new measurement'
            second_doc = CorrectionDocument.parse(data,first_doc.base_dir)
        else:
            second_doc = self.document(purpose,changed=True,tracked=variant=='tracked',config=config)
        a_path = self.save(first_doc,f'a-{len(self.models)}.json')
        b_path = self.save(second_doc,f'relocated/b-{len(self.models)}.json')
        job = self.job(replace(config,corrections_path=a_path))
        edited = self.job(replace(config,corrections_path=b_path))
        workspace = self.root/('cache-'+str(len(self.models)))
        a = self.run_job(job,workspace,policies=policies);self.assertEqual(a['status'],'complete',a.get('error'))
        old = a['jobs'][0];preserved = self.hashes(old)
        b = self.run_job(edited,workspace,policies=policies);self.assertEqual(b['status'],'complete',b.get('error'))
        current = b['jobs'][0];model = self.models[-1]
        self.assertEqual(bool(model.stage2.calls),not policies[0])
        for calls in (model.fixture_iid_calls,model.fixture_rvm_calls,model.fixture_stage1_calls):
            self.assertEqual(bool(calls),not policies[1])
        stats = current['origin_record']['diagnostics']
        self.assertEqual(stats['durable_stage2']['batch_hits']>0,policies[0])
        for name,hit in [('durable_foundations','hits'),('durable_rvm','chunk_hits'),('durable_hann','window_hits')]:
            if policies[1]:self.assertGreater(stats[name][hit],0)
        self.assertEqual(stats['durable_writer_prefix']['restored_prefix_frames'],0)
        self.assertEqual(current['mode'],'render');self.assertTrue(current['new_session_run'])
        self.assertNotEqual(a['journal'],b['journal']);self.assertNotEqual(old['key'],current['key'])
        self.assertNotEqual(old['output_dir'],current['output_dir'])
        clean = self.run_job(replace(edited,output_dir=str(self.root/'clean')),self.root/('clean-'+str(len(self.models))),
                             policies=(False,False),budget=0)
        self.assertEqual(clean['status'],'complete',clean.get('error'))
        self.assertGreater(len(self.models[-1].stage2.calls),0)
        self.assertEqual(self.hashes(current),self.hashes(clean['jobs'][0]))
        self.assertEqual(self.hashes(old),preserved)
        if purpose == 'diagnostic' or variant == 'provenance':self.assertEqual(self.hashes(current),preserved)
        else:self.assertNotEqual(self.hashes(current),preserved)
        for role,directory in current['outputs'].items():
            for left,right in zip(sorted(Path(directory).glob('*.exr')),sorted(Path(clean['jobs'][0]['outputs'][role]).glob('*.exr'))):
                decoded = cv2.imread(str(left),-1)
                np.testing.assert_array_equal(decoded,cv2.imread(str(right),-1))
                if role in ('alpha','fg'):
                    np.testing.assert_array_equal(decoded,cv2.imread(str(Path(old['outputs'][role])/left.name),-1))
                if role == 'fg' and not config.quick_key:
                    self.assertLess(decoded.min(),0);self.assertGreater(decoded.max(),1)
        self.assertEqual({p.name:p.read_bytes() for p in self.source.iterdir()},self.originals)

    def test_static_tracked_provenance_relocation_all_independent_policies(self):
        with cpu_runtime_context():
            for purpose in ('composite','diagnostic'):
                for variant in ('static','tracked','provenance'):
                    for policies in ((False,False),(True,False),(False,True),(True,True)):
                        with self.subTest(purpose=purpose,variant=variant,policies=policies):
                            self.compare(purpose,variant,policies)

    def test_low_spatial_quick_fallback_raw_inputs_and_clean_parity(self):
        from corridorkey import draft
        from corridorkey.motion import Propagation
        original = draft.offline_frames
        def uncertain(*args,**kwargs):
            alpha = np.full((64,64),.4,np.float32)
            return Propagation(alpha,np.zeros_like(alpha),np.ones_like(alpha,dtype=bool),
                {'flow_evaluations':4,'motion_seconds':0.,'uncertain_fraction':1.,'motion_fraction':0.})
        def forced(*args,**kwargs):return original(*args,interpolate=uncertain,**kwargs)
        with cpu_runtime_context(),patch.object(draft,'offline_frames',forced):
            for purpose in ('composite','diagnostic'):
                for config in (replace(self.config,low_vram=True),replace(self.config,model_scale=.5),
                               replace(self.config,quick_key=True)):
                    with self.subTest(purpose=purpose,low=config.low_vram,scale=config.model_scale,quick=config.quick_key):
                        self.compare(purpose,'static',(True,True),config=config)

    def test_current_corrupt_asset_and_late_manifest_changes_refuse_publication(self):
        from corridorkey.recovery_store import CacheInvalid
        document = self.document('composite',tracked=True)
        path = self.save(document,'bound.json')
        job = self.job(replace(self.config,corrections_path=path))
        workspace = self.root/'integrity-cache'
        with cpu_runtime_context():
            first = self.run_job(job,workspace);self.assertEqual(first['status'],'complete')
            preserved = self.hashes(first['jobs'][0])
            data = load_corrections(path).to_dict()
            derived = path.parent/data['tracks'][0]['frames'][1]['asset']['path']
            original = derived.read_bytes();derived.write_bytes(original+b'corrupt')
            with self.assertRaises((ValueError,CacheInvalid)):
                self.run_job(job,workspace)
            derived.write_bytes(original)
            late = self.save(document,'late.json')
            edited = self.job(replace(self.config,corrections_path=late))
            # Independent full-current checks still see the entire manifest, not its projection.
            changed = False
            def progress(event):
                nonlocal changed
                if event.stage == 'output_frames' and not changed:
                    changed = True
                    content = json.loads(late.read_text());content['revision'] += 1
                    late.write_text(json.dumps(content))
            with self.assertRaisesRegex(CacheInvalid,'dependencies changed'):
                self.run_job(edited,workspace,progress=progress)
            self.assertTrue(changed)
            self.assertEqual(self.hashes(first['jobs'][0]),preserved)

    def test_mixed_corrections_cuts_and_reset_remain_conservative(self):
        from contextlib import ExitStack

        from corridorkey.durable_hann import DurableHannWindows
        from corridorkey.durable_rvm import DurableRvmChunks
        from corridorkey.durable_stage2 import DurableStage2Batches
        from corridorkey.recovery_identity import (
            job_identity,
            upstream_finishing_context,
        )
        from corridorkey.stage2_identity import stage2_finishing_context
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
        policy = {name:True for name in ('durable_foundations','durable_rvm','durable_hann','durable_stage2',
                   'durable_writer_prefix','reuse_stage2_finishing_edits','reuse_upstream_finishing_edits')}
        identity_environment = {**self.environment,'recovery_policy':policy}
        with cpu_runtime_context(),ExitStack() as stack:
            stack.enter_context(patch.object(DurableStage2Batches,'get',observe))
            stack.enter_context(patch.object(DurableRvmChunks,'get',observe_temporal(DurableRvmChunks,'rvm')))
            stack.enter_context(patch.object(DurableHannWindows,'get',observe_temporal(DurableHannWindows,'hann')))
            for change in ('keep','shots','reset'):
                config = replace(self.config,quick_key=True)
                document = self.document('composite',config=config)
                a_path = self.save(document,f'{change}-a.json')
                a_job = self.job(replace(config,corrections_path=a_path))
                workspace = self.root/('negative-'+change)
                packets.clear()
                for values in temporal_packets.values():values.clear()
                first = self.run_job(a_job,workspace)
                old_packet_keys = {key for key,_ in packets}
                old_temporal_keys = {name:{key for key,_ in values} for name,values in temporal_packets.items()}
                self.assertEqual(first['status'],'complete')
                data = document.to_dict()
                if change == 'keep':
                    entry = {**data['entries'][0],'id':'keep','kind':'keep','frame':1}
                    entry.pop('purpose');entry.pop('target');data['entries'].append(entry)
                elif change == 'shots':
                    data['shots'] = [[0,1],[1,3]];data['entries'][0]['range'] = [0,1]
                else:
                    data['entries'][0]['range'] = [0,1]
                    data['entries'].append({'id':'reset','revision':1,'kind':'reset','frame':1})
                b_path = self.save(CorrectionDocument.parse(data,document.base_dir),f'{change}-b.json')
                b_job = self.job(replace(config,corrections_path=b_path))
                from tests.test_stage2_finishing_identity import legacy_stage2_context
                context = legacy_stage2_context(job_identity(b_job,assets=self.assets,environment=self.environment,
                                                                implementation_root=self.implementation))
                self.assertEqual(context.to_dict()['schema_version'],3 if change == 'shots' else 4)
                if change != 'shots':
                    retained = context.to_dict()['corrections']['document']
                    admitted = load_corrections(b_path).to_dict()
                    self.assertEqual(retained['entries'],[entry for entry in admitted['entries'] if entry['kind'] != 'region'])
                    for field in ('source','shots','guidance'):
                        self.assertEqual(retained[field],admitted[field])
                current = stage2_finishing_context(job_identity(b_job,assets=self.assets,environment=self.environment,
                                                                 implementation_root=self.implementation))
                self.assertEqual(current.to_dict()['schema_version'],5)
                packets.clear()
                for values in temporal_packets.values():values.clear()
                second = self.run_job(b_job,workspace)
                self.assertEqual(second['status'],'complete',second.get('error'))
                model = self.models[-1]
                for calls in (model.stage2.calls,model.fixture_iid_calls):
                    self.assertGreater(len(calls),0)
                for name,values in temporal_packets.items():
                    self.assertEqual([hit for _,hit in values],[key in old_temporal_keys[name] for key,_ in values])
                stats = second['jobs'][0]['origin_record']['diagnostics']
                self.assertEqual(stats['durable_foundations']['parent_key'],upstream_finishing_context(
                        job_identity(b_job,assets=self.assets,environment=identity_environment,implementation_root=self.implementation)).key)
                self.assertEqual([hit for _,hit in packets],[key in old_packet_keys for key,_ in packets])
                self.assertEqual(stats['durable_stage2']['batch_hits'],sum(hit for _,hit in packets))
                clean = self.run_job(b_job,self.root/('clean-negative-'+change),policies=(False,False),budget=0)
                self.assertEqual(self.hashes(second['jobs'][0]),self.hashes(clean['jobs'][0]))
                self.assertEqual(stats['durable_writer_prefix']['restored_prefix_frames'],0)
                if change == 'shots':self.assertEqual(stats['quick_key']['effective_cuts'],[1])

    def test_cancelled_projected_attempt_never_publishes_finished_output(self):
        from corridorkey.control import CancellationToken, CancelledError
        document = self.document('diagnostic')
        path = self.save(document,'cancel.json')
        job = self.job(replace(self.config,corrections_path=path))
        token = CancellationToken()
        def progress(event):
            if event.stage == 'recovery_publish':token.cancel('cancel pure-region final publication')
        with cpu_runtime_context(),self.assertRaisesRegex(CancelledError,'cancel pure-region'):
            run_recoverable([job],workspace=self.root/'cancel-cache',max_cache_bytes=8_000_000,
                assets=self.assets,environment=self.environment,implementation_root=self.implementation,
                session_factory=self.factory,durable_stage2=True,reuse_stage2_finishing_edits=True,
                cancellation=token,progress=progress)
        reports = list((self.root/'cancel-cache'/'reports').glob('*.json'))
        self.assertTrue(reports)
        report = json.loads(reports[-1].read_text())
        self.assertNotEqual(report['status'],'complete')




class PureRegionIidTests(adapter_fixtures.UpstreamIidTests):
    def setUp(self):
        super().setUp()
        from corridorkey.recovery_identity import upstream_finishing_context
        self.parent = upstream_finishing_context(parent())


class PureRegionRvmTests(adapter_fixtures.UpstreamRvmTests):
    def setUp(self):
        super().setUp()
        from corridorkey.recovery_identity import upstream_finishing_context
        self.parent = upstream_finishing_context(parent())


class PureRegionHannTests(adapter_fixtures.UpstreamHannTests):
    def setUp(self):
        super().setUp()
        from corridorkey.recovery_identity import upstream_finishing_context
        self.parent = upstream_finishing_context(parent())


class PureRegionPrepassTests(adapter_fixtures.UpstreamPrepassTests):
    def setUp(self):
        super().setUp()
        from corridorkey.recovery_identity import upstream_finishing_context
        self.parent = upstream_finishing_context(parent())


class PureRegionStage2Tests(unittest.TestCase):
    def setUp(self: Any):
        stage2_fixtures.DurableStage2Tests.setUp(self)
        from corridorkey.stage2_identity import stage2_finishing_context
        self.parent = stage2_finishing_context(parent())

    adapter = stage2_fixtures.DurableStage2Tests.adapter
    identity = stage2_fixtures.DurableStage2Tests.identity
    test_exact_dtypes_signed_zero_and_independent_loaded_values = stage2_fixtures.DurableStage2Tests.test_exact_dtypes_signed_zero_and_independent_loaded_values
    test_identity_binds_whole_actual_call_and_parent = stage2_fixtures.DurableStage2Tests.test_identity_binds_whole_actual_call_and_parent
    test_oversize_and_zero_budget_skip_before_hash_or_host_copy = stage2_fixtures.DurableStage2Tests.test_oversize_and_zero_budget_skip_before_hash_or_host_copy
    test_corrupt_wrong_provenance_and_header_never_materialize = stage2_fixtures.DurableStage2Tests.test_corrupt_wrong_provenance_and_header_never_materialize
    test_disk_failure_cancellation_and_sealed_prefix_preservation = stage2_fixtures.DurableStage2Tests.test_disk_failure_cancellation_and_sealed_prefix_preservation
