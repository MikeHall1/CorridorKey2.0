"""Measured-plan admission plumbing; no real GPU measurements."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from corridorkey.session import JobRequest
from corridorkey.shot_profiles import ProfilePlan
from tests import test_shot_profiles as fixtures


class ExecutionCloneTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ShotProfileCodecTests()
        fixture.setUp()
        config, writer = fixture.job.configurations()
        self.job = JobRequest.create(fixture.job.input_path, fixture.job.output_dir,
            replace(config, shot_profiles=fixture.data, quick_key=True, quick_interval=3, quick_cut_frames=[2], quick_reset_frames=[3],
                    corrections_path='/tmp/authored.json'), writer,
            initial_hint_path='/tmp/first.png', hint_video_path='/tmp/hints')
        # Bind exactly the conditioned recipe, not the fixture's earlier recipe.
        from corridorkey.shot_profiles import base_recipe_key
        data = fixture.data
        data['recipe_sha256'] = base_recipe_key(self.job)
        self.job = JobRequest.create(self.job.input_path, self.job.output_dir,
            replace(self.job.configurations()[0], shot_profiles=data), writer,
            initial_hint_path=self.job.initial_hint_path, hint_video_path=self.job.hint_video_path)

    def test_clone_rebinds_only_owned_execution_fields_and_keeps_original_strict(self):
        from corridorkey.shot_profiles import execution_clone
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            clone = execution_clone(self.job, owned_root=root, temp_dir=root/'scratch',
                                    output_dir=root/'outputs', profile_stages=True)
            before, after = (j.configurations()[0] for j in (self.job, clone))
            old = ProfilePlan.parse(before.shot_profiles)
            new = ProfilePlan.parse(after.shot_profiles)
            new.validate_recipe(clone)
            with self.assertRaises(ValueError):
                old.validate_recipe(clone)
            a, b = old.to_dict(), new.to_dict()
            del a['recipe_sha256']; del b['recipe_sha256']
            self.assertEqual(a, b)
            self.assertEqual(replace(after, temp_dir=before.temp_dir, shot_profiles=before.shot_profiles), before)
            self.assertEqual(clone.initial_hint_path, self.job.initial_hint_path)
            self.assertEqual(clone.hint_video_path, self.job.hint_video_path)

    def test_clone_rejects_stale_recipe_and_unowned_or_nonbool_execution(self):
        from corridorkey.shot_profiles import execution_clone
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            import json
            for temp, output, flag in ((root.parent/'foreign', root/'outputs', True),
                    (root/'scratch', root, True), (root/'scratch', root/'outputs', json.loads('1'))):
                with self.subTest(temp=temp, output=output, flag=flag), self.assertRaises(ValueError):
                    execution_clone(self.job, owned_root=root, temp_dir=temp, output_dir=output, profile_stages=flag)
            cfg, writer = self.job.configurations()
            stale = JobRequest.create(self.job.input_path, self.job.output_dir, replace(cfg, start_frame=11), writer)
            with self.assertRaises(ValueError):
                execution_clone(stale, owned_root=root, temp_dir=root/'scratch', output_dir=root/'outputs', profile_stages=True)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        import copy
        import json

        from corridorkey.shot_profiles import analyze_job
        from tests import test_shot_profiles
        fixture = test_shot_profiles.ShotProfileAnalysisTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.job = fixture.root, fixture.job
        plan = analyze_job(self.job).to_dict()
        left, right = copy.deepcopy(plan), copy.deepcopy(plan)
        for row in left['rows']: row['stage2_batch'] = 3
        for row in right['rows']: row['stage2_batch'] = 1
        config, writer = self.job.configurations()
        jobs = [JobRequest.create(self.job.input_path, self.job.output_dir,
            replace(config, shot_profiles=data), writer) for data in (left, right)]
        self.report = {'provenance': {side: {'normalized_batch': {'schema_version':1,
            'kind':'inference_batch', 'jobs':[{'input':j.input_path, 'output_dir':j.output_dir,
            'settings':json.loads(j.config_json), 'writer':json.loads(j.writer_json),
            'hint_first_frame':None, 'hint_video':None}]}} for side,j in zip(('baseline','candidate'),jobs)}}

    def test_only_complete_batch_cap_combination_can_be_recommended(self):
        import copy

        from corridorkey.shot_profile_recommendations import candidate_plan
        plan = candidate_plan(self.job, self.report)
        self.assertTrue(all(row['stage2_batch'] == 1 for row in plan.to_dict()['rows']))
        for field,value in (('alpha_feather_radius',2),('enabled',False),('range',[1,2])):
            data=copy.deepcopy(self.report)
            data['provenance']['candidate']['normalized_batch']['jobs'][0]['settings']['shot_profiles']['rows'][0][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                candidate_plan(self.job,data)

    def test_request_rejects_absent_receipts_and_never_infers_claims(self):
        from corridorkey.shot_profile_recommendations import recommend
        report=self.root/'pair.json'
        report.write_text('{}')
        with self.assertRaises((TypeError,ValueError)):
            recommend(self.job, report_path=report, policy_path=None, policy_sha256=None,
                numerical_evidence_path=None, numerical_evidence_sha256=None,
                resource_evidence_path=None, resource_evidence_sha256=None)


class BoundGateTests(unittest.TestCase):
    """Actual gate/native EXR/receipt flow with labelled structural GPU-record fixtures."""
    def setUp(self):
        import copy
        import hashlib
        import json

        import cv2
        import numpy as np

        from corridorkey import regression_gate as gate
        from corridorkey.batch import jobs_from_data
        from corridorkey.shot_profiles import analyze_job
        from tests.test_regression_resources import ResourcePhaseTests
        fixture = ResourcePhaseTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        baseline = fixture.report['provenance']['baseline']['normalized_batch']
        self.job = jobs_from_data(baseline, fixture.report_path)[0][0]
        cfg, writer = self.job.configurations()
        from corridorkey import benchmark
        from corridorkey.corrections import CorrectionDocument, save_corrections
        from corridorkey.identities import file_sha256, source_identity
        mask=self.root/'authored.png'
        self.assertTrue(cv2.imwrite(str(mask),np.full((8,12),255,np.uint8)))
        correction=self.root/'authored.json'
        document=CorrectionDocument.parse({'schema_version':1,'revision':0,
            'source':{'sha256':source_identity(self.job.input_path)['sha256'],'start_frame':1,
                      'frame_count':2,'fps':cfg.frame_dir_fps,'width':12,'height':8},
            'shots':[[0,2]], 'entries':[{'id':'authored','revision':0,'frame':0,'kind':'matte',
                'asset':{'path':mask.name,'sha256':file_sha256(mask)},'strength':1,
                'direction':'both','range':[0,1]}]},self.root)
        save_corrections(correction,document)
        hint=next(Path(self.job.input_path).glob('*.png'))
        self.job = JobRequest.create(self.job.input_path, self.job.output_dir,
            replace(cfg, stage2_batch=3, corrections_path=correction, quick_key=True,
                    quick_interval=3, quick_cut_frames=[1], quick_reset_frames=[1]), writer,
            initial_hint_path=hint, hint_video_path=self.job.input_path)
        plan = analyze_job(self.job).to_dict()
        for side, cap in (('baseline',2), ('candidate',1)):
            frozen = fixture.report['provenance'][side]['normalized_batch']['jobs'][0]
            frozen['settings'] = json.loads(self.job.config_json)
            data = copy.deepcopy(plan)
            for row in data['rows']: row['stage2_batch'] = cap
            frozen['settings']['shot_profiles'] = data
            frozen.update(hint_first_frame=self.job.initial_hint_path,hint_video=self.job.hint_video_path)
            for pair in fixture.report['pairs']:
                sample = pair['samples'][side]
                sample['corrections']=[benchmark.correction_identity(str(correction),self.job.input_path)]
                sample['assets'][self.job.initial_hint_path]=benchmark.asset_identity(self.job.initial_hint_path)
                for record in [sample['runs'][0]['jobs'][0], *sample['warmup']]:
                    record['hints']={'first_frame':self.job.initial_hint_path,'full':self.job.hint_video_path}
                    record['requested_settings'] = copy.deepcopy(frozen['settings'])
                    d = record['diagnostics']
                    d['effective_settings'] = copy.deepcopy(frozen['settings'])
                    d['execution']['foundation_cache_dtype'] = None
                    d['execution']['temporal_strategy']='sparse_native_context_with_full_rate_reset_aware_rvm'
                    d['output_contracts']['alpha']['alpha_representation'] = 'coverage (no transfer function)'
                record = sample['runs'][0]['jobs'][0]
                directory = Path(record['diagnostics']['outputs']['alpha'])
                directory.mkdir(parents=True)
                for index in range(2):
                    self.assertTrue(cv2.imwrite(str(directory/f'alpha_{index:06d}.exr'),np.full((8,12),.5,np.float32), [cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_HALF]))
        fixture.policy.update(required_evidence=['observed_memory','numerical_parity'], numerical={
            'constraints':[{'job':0,'metric':'alpha.mae','units':'alpha/pixel','maximum':0.001}],
            'budgets':{'max_files':128,'max_bytes':64*1024**2,'max_pixels':96}})
        fixture.policy['context_sha256'] = gate.digest(gate.context(fixture.report))
        fixture.report_path.write_text(json.dumps(fixture.report))
        fixture.policy_path.write_text(json.dumps(fixture.policy))
        fixture.policy_sha = hashlib.sha256(fixture.policy_path.read_bytes()).hexdigest()
        receipt = gate.capture_numerical(fixture.report_path, fixture.policy_path, fixture.policy_sha)
        self.numerical = self.root/'numerical.json'
        self.numerical.write_text(json.dumps(receipt))
        Client = fixture.Client
        class ProfileClient(Client):
            def send(self, action, payload):
                from corridorkey.worker_protocol import job_from_wire
                job = None
                if action != 'cancel':
                    job = job_from_wire(payload['job'])
                    ProfilePlan.parse(job.configurations()[0].shot_profiles).validate_recipe(job)
                result = super().send(action,payload)
                if action != 'cancel':
                    assert job is not None
                    self.event['payload']['record']['hints']={'first_frame':job.initial_hint_path,'full':job.hint_video_path}
                    d = self.event['payload']['record']['diagnostics']
                    d['execution']['foundation_cache_dtype'] = None
                    d['execution']['temporal_strategy']='sparse_native_context_with_full_rate_reset_aware_rvm'
                    d['output_contracts']['alpha']['alpha_representation'] = 'coverage (no transfer function)'
                return result
        self.Client = ProfileClient

    def resource(self, *, injected=False):
        import hashlib
        from unittest.mock import patch

        from corridorkey.regression_resources import run_phase
        f = self.fixture
        with f.temp_only_provenance(), patch('corridorkey.regression_resources.WorkerClient', self.Client):
            attempt = run_phase(f.report_path, f.policy_path, f.policy_sha, f.parent,
                                worker_factory=self.Client if injected else None)
        self.assertEqual(attempt.to_dict()['status'],'complete',attempt.to_dict().get('error'))
        return attempt.report_path,hashlib.sha256(attempt.report_path.read_bytes()).hexdigest()

    def request(self, resource):
        import hashlib
        f=self.fixture
        return {'report_path': f.report_path, 'policy_path': f.policy_path, 'policy_sha256': f.policy_sha,
            'numerical_evidence_path': self.numerical,
            'numerical_evidence_sha256': hashlib.sha256(self.numerical.read_bytes()).hexdigest(),
            'resource_evidence_path': resource[0], 'resource_evidence_sha256': resource[1]}

    def test_actual_gate_receipts_native_pixels_and_resource_clone_then_whole_candidate(self):
        from corridorkey.shot_profile_recommendations import recommend
        resource=self.resource()
        with self.fixture.temp_only_provenance():
            plan,summary=recommend(self.job,**self.request(resource))
        self.assertEqual(plan.to_dict()['source_range'],[1,3])
        self.assertEqual(plan.to_dict()['analysis']['resource_fit'],'unavailable')
        self.assertTrue(all(r['stage2_batch']==1 for r in plan.to_dict()['rows']))
        self.assertEqual(summary['observed_memory'][0]['observed_max_bytes'],100)
        self.assertEqual(summary['candidate_over_baseline'],[1.,1.,1.])
        self.assertIsNone(self.job.configurations()[0].shot_profiles)

    def test_fixture_provider_cannot_qualify_and_receipt_or_source_drift_rejected(self):
        from corridorkey.shot_profile_recommendations import recommend
        resource=self.resource(injected=True)
        with self.fixture.temp_only_provenance(), self.assertRaisesRegex(ValueError,'passing bound'):
            recommend(self.job,**self.request(resource))
        resource=self.resource()
        request=self.request(resource)
        request['numerical_evidence_sha256']='0'*64
        with self.fixture.temp_only_provenance(), self.assertRaises(ValueError):
            recommend(self.job,**request)

    def test_noisy_matched_data_and_late_policy_replacement_cannot_publish(self):
        import hashlib
        import json
        from unittest.mock import patch

        from corridorkey import regression_gate as gate
        from corridorkey.shot_profile_recommendations import recommend
        f=self.fixture
        resource=self.resource()
        original=gate.evaluate
        def late(*args,**kwargs):
            result=original(*args,**kwargs)
            self.assertEqual(result['status'],'pass')
            f.policy_path.write_bytes(f.policy_path.read_bytes())
            return result
        with f.temp_only_provenance(), patch.object(gate,'evaluate',side_effect=late), self.assertRaises(ValueError):
            recommend(self.job,**self.request(resource))
        f.report['pairs'][1]['samples']['baseline']['runs'][0]['elapsed_seconds']=4.
        f.report_path.write_text(json.dumps(f.report))
        self.numerical.write_text(json.dumps(gate.capture_numerical(f.report_path,f.policy_path,f.policy_sha)))
        resource=self.resource()
        with f.temp_only_provenance(), self.assertRaisesRegex(ValueError,'inconclusive'):
            recommend(self.job,**self.request(resource))
        self.assertEqual(hashlib.sha256(f.policy_path.read_bytes()).hexdigest(),f.policy_sha)

    def test_cpu_source_drift_and_cancellation_remain_visible(self):
        from corridorkey.control import CancelledError
        from corridorkey.shot_profile_recommendations import recommend
        f=self.fixture
        request=self.request(self.resource())
        def cancelled(): raise CancelledError('user cancelled inspection')
        with self.assertRaises(CancelledError): recommend(self.job,check=cancelled,**request)
        source=next(Path(self.job.input_path).glob('*.png'))
        source.write_bytes(source.read_bytes()+b'drift')
        with f.temp_only_provenance(), self.assertRaises(ValueError): recommend(self.job,**request)
    def test_cpu_original_is_rejected_before_worker_creation(self):
        import hashlib
        import json
        from unittest.mock import patch

        from corridorkey import regression_gate as gate
        from corridorkey.regression_resources import run_phase
        f=self.fixture
        for pair in f.report['pairs']:
            for sample in pair['samples'].values():
                sample['runs'][0]['jobs'][0]['diagnostics']['execution']['backend']='cpu'
        f.policy['context_sha256']=gate.digest(gate.context(f.report))
        f.report_path.write_text(json.dumps(f.report))
        f.policy_path.write_text(json.dumps(f.policy))
        f.policy_sha=hashlib.sha256(f.policy_path.read_bytes()).hexdigest()
        with f.temp_only_provenance(), patch('corridorkey.regression_resources.WorkerClient') as worker, self.assertRaisesRegex(ValueError,'CPU/unknown'):
            run_phase(f.report_path,f.policy_path,f.policy_sha,f.parent)
        worker.assert_not_called()

    def test_resource_record_cannot_change_caps_while_rebinding_recipe(self):
        import json

        from corridorkey.regression_resources import _record
        from corridorkey.worker_protocol import job_from_wire
        path,_=self.resource()
        receipt=json.loads(path.read_text())
        row=next(r for r in receipt['trials'] if r['side']=='candidate')
        job=job_from_wire(row['job_request'])
        cfg,writer=job.configurations()
        assert cfg.shot_profiles is not None
        cfg.shot_profiles['rows'][0]['stage2_batch']=2
        altered=JobRequest.create(job.input_path,job.output_dir,cfg,writer,profile_stages=True,
            initial_hint_path=job.initial_hint_path,hint_video_path=job.hint_video_path)
        record=json.loads(Path(row['record']['path']).read_text())
        record['requested_settings']=json.loads(altered.config_json)
        record['diagnostics']['effective_settings']=json.loads(altered.config_json)
        original=self.fixture.report['pairs'][row['pair']]['samples']['candidate']['runs'][0]['jobs'][0]
        with self.assertRaises(ValueError): _record(record,altered,original,row.get('model_reused', True))

    def test_actual_v4_late_visual_declaration_change_cannot_publish(self):
        import hashlib
        import json
        from unittest.mock import patch

        from corridorkey import regression_gate as gate
        from corridorkey.regression_visual import capture
        from corridorkey.shot_profile_recommendations import recommend

        f = self.fixture
        f.policy.update(schema_version=4, visual_review_required=True,
            visual={'mode':'bound_declaration_required',
                    'selections':[{'pair':0,'job':0,'source_frames':[1]}]})
        f.policy_path.write_text(json.dumps(f.policy))
        f.policy_sha = hashlib.sha256(f.policy_path.read_bytes()).hexdigest()
        self.numerical.write_text(json.dumps(gate.capture_numerical(f.report_path,f.policy_path,f.policy_sha)))
        numerical_sha = hashlib.sha256(self.numerical.read_bytes()).hexdigest()
        declaration = {'schema_version':1,'kind':'regression_visual_declaration',
            'reviewer':'Structural fixture, NOT authenticated human inspection','outcome':'accept',
            'policy_sha256':f.policy_sha,'numerical_evidence_sha256':numerical_sha,
            'selections':f.policy['visual']['selections'],'accepted_tradeoffs':f.policy['accepted_tradeoffs'],
            'observations':'Generated declaration solely for the final-fence race regression.'}
        declaration_path, visual_path = self.root/'declaration.json', self.root/'visual.json'
        declaration_path.write_text(json.dumps(declaration))
        visual = capture(f.report_path,f.policy_path,f.policy_sha,self.numerical,numerical_sha,
            declaration_path,hashlib.sha256(declaration_path.read_bytes()).hexdigest())
        visual_path.write_text(json.dumps(visual))
        visual_bytes = visual_path.read_bytes()
        request = self.request(self.resource())
        request.update(visual_evidence_path=visual_path,
            visual_evidence_sha256=hashlib.sha256(visual_bytes).hexdigest())
        with f.temp_only_provenance():
            plan,_ = recommend(self.job,**request)
            self.assertEqual(plan.to_dict()['source_range'],[1,3])
        actual_evaluate = gate.evaluate
        def late_change(*args,**kwargs):
            result = actual_evaluate(*args,**kwargs)
            self.assertEqual(result['status'],'pass',result)
            declaration['outcome']='reject'
            declaration_path.write_text(json.dumps(declaration))
            return result
        with f.temp_only_provenance(), patch.object(gate,'evaluate',side_effect=late_change), self.assertRaisesRegex(ValueError,'changed'):
            recommend(self.job,**request)
        self.assertEqual(visual_path.read_bytes(),visual_bytes,'Top receipt remains unchanged; authored nested input must be fenced')


class PairRestagingTests(unittest.TestCase):
    root: Path
    job: JobRequest
    report: dict
    setUp = CandidateTests.setUp
    def test_actual_pair_freeze_relocates_only_outputs_and_retains_bound_plan(self):
        import json

        from corridorkey import benchmark_pair
        from corridorkey.batch import jobs_from_data
        from corridorkey.shot_profiles import validate_live
        manifests={}
        for side in ('baseline','candidate'):
            batch=self.report['provenance'][side]['normalized_batch']
            path=self.root/f'{side}-batch.json'
            path.write_text(json.dumps(batch))
            manifest={'batch_manifest':str(path),'model_state':'warm_session'}
            manifest_path=self.root/f'{side}.json'
            manifest_path.write_text(json.dumps(manifest))
            manifests[side]=(manifest_path,manifest)
        staging=self.root/'restaged';staging.mkdir()
        frozen,provenance,batches,watched=benchmark_pair._freeze(manifests,staging)
        benchmark_pair._match(frozen,batches,['shot_profiles'])
        for side in ('baseline','candidate'):
            job=jobs_from_data(provenance[side]['normalized_batch'],staging/'manifest.json')[0][0]
            self.assertNotEqual(job.output_dir,self.job.output_dir)
            validate_live(ProfilePlan.parse(job.configurations()[0].shot_profiles),job)
            self.assertEqual(job.configurations()[0].start_frame,1)
            self.assertEqual(job.configurations()[0].num_frames,4)
        self.assertIn(self.job.input_path,watched)


class PanelBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import test_desktop_shot_profiles as desktop
        desktop.DesktopShotProfileTests.setUpClass()

    def setUp(self):
        from tests import test_desktop_shot_profiles as desktop
        self.fixture=desktop.DesktopShotProfileTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)

    def test_offthread_import_then_deliberate_apply_preserves_queued_snapshot(self):
        import threading
        from unittest.mock import patch

        from corridorkey.shot_profiles import analyze_job
        f=self.fixture
        queued=f.workspace.current
        plan=analyze_job(queued)
        threads=[]
        def inspect(*args,**kwargs):
            threads.append(threading.current_thread().name)
            return plan,{'scope':'labelled bridge fixture; not real GPU evidence'}
        with patch('corridorkey.shot_profile_recommendations.recommend',side_effect=inspect):
            f.panel.recommend_request({})
            f.until_idle()
        self.assertEqual(threads,['shot-profile-io'])
        self.assertEqual(f.workspace.current,queued)
        self.assertIn('labelled bridge fixture',f.panel.status.text())
        f.panel.apply();f.until_idle()
        self.assertIsNone(queued.configurations()[0].shot_profiles)
        self.assertEqual(f.workspace.current.configurations()[0].shot_profiles,plan.to_dict())

    def test_cancelled_import_and_strict_request_text_cannot_replace_editor(self):
        import threading
        from unittest.mock import patch

        from corridorkey.shot_profiles import analyze_job
        f=self.fixture
        plan=analyze_job(f.workspace.current)
        ready,release=threading.Event(),threading.Event()
        def inspect(*args,check,**kwargs):
            ready.set();release.wait(5);check()
            return plan,{}
        before=f.panel.document.toPlainText()
        with patch('corridorkey.shot_profile_recommendations.recommend',side_effect=inspect):
            f.panel.recommend_request({})
            self.assertTrue(ready.wait(3))
            f.panel.cancel();release.set();f.until_idle()
        self.assertEqual(f.panel.document.toPlainText(),before)
        with patch('corridorkey.desktop_shot_profiles.QInputDialog.getMultiLineText',return_value=('{"report_path":0,"report_path":1}',True)):
            f.panel.choose_recommendation()
        self.assertIn('Duplicate',f.panel.status.text())
