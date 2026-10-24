"""Owned actual CPU fixture child plus deterministic lifecycle fault providers."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, ClassVar

from corridorkey.calibration import (
    CalibrationLimits,
    CalibrationProfile,
    plan_calibration,
)
from corridorkey.calibration_task import (
    ArtifactBudgetError,
    CalibrationTask,
    _metrics,
    _size,
    run_calibration,
)
from corridorkey.control import CancellationToken
from corridorkey.worker_client import WorkerClient
from tests.test_calibration import calibration_fixture

FIXTURE_CODE = '''
import faulthandler
import json
faulthandler.dump_traceback_later(15)
from tests.worker_fixture import FixtureEngine
from corridorkey.worker_transport import serve_stdio
original=FixtureEngine.render
def measured(self,model,job,config,writer,diagnostics):
    original(self,model,job,config,writer,diagnostics)
    diagnostics.effective=json.loads(job.config_json)
FixtureEngine.render=measured
try:
    serve_stdio(capacity=1,engine=FixtureEngine())
finally:
    faulthandler.cancel_dump_traceback_later()
'''


def calibration_worker_factory(**options):
    """Actual owned child for UI tests, explicit fixture not production models."""
    return WorkerClient(command=[sys.executable,'-u','-c',FIXTURE_CODE],**options)


class FakeClient:
    instances: ClassVar[list["FakeClient"]] = []
    live = 0
    def __init__(self,**options):
        self.options = options
        self.jobs = 0
        self.closed = False
        self.cleanup_error = None
        self.event: dict[str,Any] = {}
        FakeClient.live += 1
        if FakeClient.live != 1:
            raise AssertionError('Concurrent worker owner')
        self.instances.append(self)
    def send(self,action,payload):
        if action=='cancel':
            return 'cancel'
        self.jobs += 1
        wire = payload['job']
        settings = json.loads(wire['config_json'])
        record = {'status':'complete','input':wire['input_path'],'output_dir':wire['output_dir'],'writer_settings':json.loads(wire['writer_json']),'requested_settings':settings,'model_reused':self.jobs>1,
                  'processing_seconds':1.,'elapsed_seconds':1.,'model_load_seconds':0.,'model_initialization_seconds':0.,
                  'diagnostics':{'effective_settings':settings,'frames_submitted':settings['num_frames'],
                                 'source':{'start_frame':settings['start_frame'],'num_frames':settings['num_frames'],'native_fps':settings['frame_dir_fps'],'native_hw':[8,12]},
                                 'outputs':{kind:str(Path(wire['output_dir'])/kind) for kind in settings['outputs']},
                                 'output_contracts':{kind:{'format':'exr','channels':1 if kind=='alpha' else 4 if kind=='cutout' else 3} for kind in settings['outputs']},
                                 'execution':{'neural_inference':False},'resources':None,'first_rendered_batch_seconds':.1}}
        self.event = {'kind':'result','request_id':f'job-{self.jobs}','payload':{'record':record}}
        return f'job-{self.jobs}'
    def next_event(self,timeout):
        return self.event
    def close(self):
        if not self.closed:
            FakeClient.live -= 1
            self.closed = True
        return self.cleanup_error is None


class CalibrationTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.job,self.options = calibration_fixture(self.root)
        self.plan = plan_calibration(self.job,**self.options)
        FakeClient.instances = []
        FakeClient.live = 0

    def tearDown(self):
        self.temp.cleanup()

    def test_sequential_alternating_trials_warmup_actual_job_counts_and_no_fixture_recommendations(self):
        profiles = (CalibrationProfile.capture('baseline',{}),CalibrationProfile.capture('draft',{'model_scale':.5},'preview'))
        plan = plan_calibration(self.job,**dict(self.options,profiles=profiles))
        report = run_calibration(plan,worker_factory=FakeClient).to_dict()
        self.assertEqual(report['status'],'complete')
        self.assertEqual(len(FakeClient.instances),12)
        self.assertEqual(sum(owner.jobs for owner in FakeClient.instances),18)
        self.assertTrue(all(owner.closed for owner in FakeClient.instances))
        self.assertEqual([row['profile'] for row in report['trials']],['baseline','baseline','draft','draft','draft','draft','baseline','baseline','baseline','baseline','draft','draft'])
        self.assertEqual([owner.jobs for owner in FakeClient.instances],[1,2]*6)
        self.assertFalse(report['summary']['recommendations'])
        self.assertTrue(all(row['measurement']['neural_gpu_verified'] is False for row in report['trials']))
        self.assertFalse(Path(self.job.output_dir).exists())
        self.assertTrue(all(row['record']['sha256'] for row in report['trials']))
        self.assertTrue(all(owner.options['asset_root']==str(self.options['asset_selection'].root) for owner in FakeClient.instances))

    def test_actual_worker_fixture_task_persists_full_records_and_no_gpu_claim(self):
        originals = {path:path.read_bytes() for path in Path(self.job.input_path).iterdir()}
        task = CalibrationTask(self.plan,worker_factory=calibration_worker_factory)
        try:
            if not task.done.wait(30):
                client = task.client
                worker = None
                if client is not None:
                    with client.condition:
                        worker = {
                            'log_tail':client.log_tail,'failure':client.failure,'eof':client.eof,
                            'returncode':client.process.returncode,'outstanding':sorted(client.outstanding),
                            'queued_requests':client.outgoing.qsize(),'buffered_events':len(client.events),
                            'buffered_progress':len(client.progress),
                            'threads':{owner.name:owner.is_alive() for owner in (client.writer,client.reader,client.log_reader)}}
                self.fail({'task':task.snapshot(),'worker':worker})
        finally:
            task.cancel()
            task.thread.join(15)
            self.assertFalse(task.thread.is_alive(),'Calibration task must retire before fixture directory cleanup')
        state = task.snapshot()
        self.assertEqual(state['status'],'complete',state)
        self.assertIsNone(task.cleanup_error)
        self.assertIsNone(task.client)
        self.assertIsNotNone(task.report_path)
        assert task.report_path is not None
        data = json.loads(task.report_path.read_bytes())
        self.assertEqual(len(data['trials']),6)
        self.assertFalse(data['summary']['recommendations'])
        self.assertEqual({path:path.read_bytes() for path in originals},originals)
        self.assertFalse(Path(self.job.output_dir).exists())
        self.assertLessEqual(len(json.dumps(state).encode()),64*1024)
        self.assertEqual(len(list(task.report_path.parent.glob('record-*.json'))),9)

    def test_cleanup_uncertainty_stops_next_owner_and_stays_visible(self):
        class Denied(FakeClient):
            def close(self):
                self.cleanup_error = 'fixture process containment NOT VERIFIED'
                return False
        task = CalibrationTask(self.plan,worker_factory=Denied)
        self.assertTrue(task.done.wait(10))
        self.assertEqual(len(FakeClient.instances),1)
        self.assertIsNotNone(task.client)
        self.assertIn('containment',task.cleanup_error or '')
        self.assertEqual(task.snapshot()['status'],'failed')
        self.assertIsNone(task.artifact_error)
        FakeClient.live = 0

    def test_cancel_native_wait_retires_worker_before_done(self):
        entered = threading.Event()
        class Blocking(FakeClient):
            def next_event(self,timeout):
                entered.set()
                entered.wait(.01)
                raise TimeoutError('fixture running')
        task = CalibrationTask(self.plan,worker_factory=Blocking)
        self.assertTrue(entered.wait(3))
        task.cancel()
        self.assertTrue(task.done.wait(10))
        self.assertEqual(task.snapshot()['status'],'cancelled')
        self.assertTrue(FakeClient.instances[0].closed)
        self.assertIsNone(task.client)
        self.assertIsNone(task.cleanup_error)

    def test_failed_trial_excluded_and_later_trials_continue_only_after_verified_close(self):
        class Failed(FakeClient):
            def send(self,action,payload):
                identifier = super().send(action,payload)
                if action=='preview' and len(FakeClient.instances)==1:
                    self.event['kind'] = 'failed'
                    self.event['payload']['error'] = 'fixture failed'
                    self.event['payload']['record']['status'] = 'failed'
                return identifier
        report = run_calibration(self.plan,worker_factory=Failed).to_dict()
        self.assertEqual(report['trials'][0]['status'],'failed')
        self.assertFalse(report['trials'][0]['eligible'])
        self.assertEqual(len(FakeClient.instances),6)
        self.assertTrue(all(owner.closed for owner in FakeClient.instances))
        self.assertTrue(any(row['status']=='complete' for row in report['trials']))

    def test_zero_budget_wall_and_pre_cancel_start_no_new_owner(self):
        token = CancellationToken()
        token.cancel('before start')
        with self.assertRaises(RuntimeError):
            run_calibration(self.plan,worker_factory=FakeClient,cancellation=token)
        self.assertEqual(FakeClient.instances,[])
        tick = iter(range(10000))
        limits = CalibrationLimits(max_wall_seconds=.001,read_budget=64*1024**2)
        plan = plan_calibration(self.job,**dict(self.options,limits=limits))
        report = run_calibration(plan,worker_factory=FakeClient,clock=lambda:next(tick)).to_dict()
        self.assertEqual(report['status'],'failed')
        self.assertIn('wall-time',report['error'])
        self.assertEqual(FakeClient.instances,[])

    def test_source_drift_before_execution_and_effective_mismatch_do_not_recommend(self):
        path = Path(self.job.input_path)/'1.png'
        raw = path.read_bytes()
        path.write_bytes(raw+b'changed')
        with self.assertRaises(ValueError):
            run_calibration(self.plan,worker_factory=FakeClient)
        self.assertEqual(FakeClient.instances,[])
        path.write_bytes(raw)
        plan = plan_calibration(self.job,**self.options)
        class Mismatch(FakeClient):
            def send(self,action,payload):
                identifier = super().send(action,payload)
                self.event['payload']['record']['diagnostics']['effective_settings'] = {'wrong':True}
                return identifier
        report = run_calibration(plan,worker_factory=Mismatch).to_dict()
        self.assertEqual(report['status'],'failed')
        self.assertFalse(report['summary']['recommendations'])
        self.assertTrue(all(owner.closed for owner in FakeClient.instances))

    def test_warmup_coverage_mismatch_cannot_become_measured(self):
        class BadWarmup(FakeClient):
            def send(self,action,payload):
                identifier = super().send(action,payload)
                if action=='preview' and len(FakeClient.instances)%2==0 and self.jobs==1:
                    self.event['payload']['record']['diagnostics']['frames_submitted'] = 0
                return identifier
        report = run_calibration(self.plan,worker_factory=BadWarmup).to_dict()
        self.assertEqual([owner.jobs for owner in FakeClient.instances],[1,1]*3)
        self.assertTrue(all(row['status']=='failed' for row in report['trials'] if row['state']=='warm_session'))
        self.assertFalse(report['summary']['recommendations'])

    def test_unknown_close_and_report_fsync_failure_keep_process_uncertainty(self):
        from unittest.mock import patch
        class Unknown(FakeClient):
            def close(self):
                return False
        with patch('corridorkey.calibration_task.os.fsync',side_effect=OSError('report fsync fault')):
            task = CalibrationTask(self.plan,worker_factory=Unknown)
            self.assertTrue(task.done.wait(10))
        self.assertEqual(len(FakeClient.instances),1)
        self.assertIsNotNone(task.client)
        self.assertIn('not been verified',task.cleanup_error or '')
        self.assertIn('fsync fault',task.artifact_error or '')
        FakeClient.live = 0

    def test_record_binding_rejects_contradictions_and_accepts_sparse_neural_counters(self):
        # Synthetic records validate admission only, never actual GPU evidence.
        from copy import deepcopy
        from dataclasses import replace
        wire = self.plan.to_dict()['profiles'][0]['job']
        from corridorkey.worker_protocol import job_from_wire
        owner = FakeClient()
        owner.send('preview',{'job':wire})
        record = owner.event['payload']['record']
        owner.close()
        _metrics(record,job_from_wire(wire),2,'cold_process',[8,12])
        mutations = [lambda value:value.update(input='/wrong'),lambda value:value.update(output_dir='/wrong'),
                     lambda value:value['writer_settings'].update(crf=999),
                     lambda value:value['diagnostics']['source'].update(start_frame=999),
                     lambda value:value['diagnostics']['source'].update(native_fps=1),
                     lambda value:value['diagnostics']['source'].update(native_hw=[True,12]),
                     lambda value:value['diagnostics']['source'].update(num_frames=1),
                     lambda value:value['diagnostics']['outputs'].update(alpha='/outside')]
        for change in mutations:
            bad = deepcopy(record)
            change(bad)
            with self.assertRaises(ValueError):
                _metrics(bad,job_from_wire(wire),2,'cold_process',[8,12])
        config,writer = job_from_wire(wire).configurations()
        config.quick_key = True
        config.num_frames = 120
        quickjob = type(self.job).create(self.job.input_path,self.job.output_dir,config,writer)
        quickwire = dict(wire,input_path=quickjob.input_path,output_dir=quickjob.output_dir,config_json=quickjob.config_json)
        owner = FakeClient()
        owner.send('preview',{'job':quickwire})
        sparse = owner.event['payload']['record']
        owner.close()
        sparse['diagnostics']['execution'] = {'backend':'pytorch','requested_device':'cuda','gpu':'SYNTHETIC counter fixture',
            'device_index':0,'autocast_dtype':'bfloat16','temporal_strategy':'sparse_native_context_with_full_rate_reset_aware_rvm',
            'completed_prediction_frames':10,'stage2_frame_evaluations':10,'stage2_calls':2}
        self.assertTrue(_metrics(sparse,quickjob,120,'cold_process',[8,12])['neural_gpu_verified'])
        for key,badvalue in [('completed_prediction_frames',0),('completed_prediction_frames',121),('stage2_frame_evaluations',True),('stage2_calls','2'),('device_index',False)]:
            bad = deepcopy(sparse)
            bad['diagnostics']['execution'][key] = badvalue
            with self.assertRaises(ValueError):
                _metrics(bad,quickjob,120,'cold_process',[8,12])
        sparse['diagnostics']['execution']['neural_inference'] = False
        self.assertFalse(_metrics(sparse,quickjob,120,'cold_process',[8,12])['neural_gpu_verified'])
        config.quick_key = False
        fulljob = replace(quickjob,config_json=json.dumps(dict(json.loads(quickjob.config_json),quick_key=False)))
        fullrecord = deepcopy(sparse)
        fullrecord['requested_settings']['quick_key'] = False
        fullrecord['diagnostics']['effective_settings']['quick_key'] = False
        fullrecord['diagnostics']['execution'].pop('neural_inference')
        fullrecord['diagnostics']['execution']['temporal_strategy'] = 'full_rate_with_chronological_rvm_prepass'
        with self.assertRaises(ValueError):
            _metrics(fullrecord,fulljob,120,'cold_process',[8,12])

    def test_second_artifact_allocation_failure_retains_first_path_without_worker(self):
        from unittest.mock import patch

        from corridorkey.project_collection import fresh_directory
        created = []
        def allocate(parent,prefix):
            if created:
                raise OSError('second artifact allocation fault')
            root = fresh_directory(parent,prefix)
            created.append(root)
            return root
        with patch('corridorkey.calibration_task.fresh_directory',side_effect=allocate):
            task = CalibrationTask(self.plan,worker_factory=FakeClient)
            self.assertTrue(task.done.wait(5))
        self.assertEqual(FakeClient.instances,[])
        self.assertEqual(task.snapshot()['retained_paths']['report_root'],str(created[0]))
        self.assertIsNone(task.snapshot()['retained_paths']['scratch_root'])
        self.assertTrue(created[0].is_dir())
        self.assertIn('second artifact allocation fault',task.snapshot()['error'])
        self.assertIsNone(task.client)
        self.assertIsNone(task.cleanup_error)

    def test_global_model_unavailable_stops_after_first_retired_owner_and_keeps_failed_record(self):
        class MissingModel(FakeClient):
            def send(self,action,payload):
                identifier = super().send(action,payload)
                self.event['kind'] = 'failed'
                self.event['payload']['error'] = 'Explicit fixture missing model, no GPU probe'
                record = self.event['payload']['record']
                record['status'] = 'failed'
                record['diagnostics']['last_checkpoint'] = {'stage':'model_load','completed':0,'total':1}
                record['diagnostics']['frames_submitted'] = 0
                return identifier
        report = run_calibration(self.plan,worker_factory=MissingModel).to_dict()
        self.assertEqual(len(FakeClient.instances),1)
        self.assertTrue(FakeClient.instances[0].closed)
        self.assertEqual(report['status'],'failed')
        self.assertTrue(Path(report['trials'][0]['failed_record']['path']).is_file())
        self.assertEqual(sum(row['status']=='not_run' for row in report['trials']),5)
        self.assertTrue(all(row.get('exclusion_reason') for row in report['trials'][1:]))
        self.assertFalse(report['summary']['recommendations'])

    def test_verified_startup_unavailable_attempted_once_and_not_retried(self):
        calls = []
        def unavailable(**options):
            calls.append(options)
            raise RuntimeError('fixture startup unavailable; no child was launched')
        report = run_calibration(self.plan,worker_factory=unavailable).to_dict()
        self.assertEqual(len(calls),1)
        self.assertEqual(report['status'],'failed')
        self.assertIn('Worker startup unavailable',report['error'])
        self.assertIsNone(report['cleanup_error'])
        self.assertEqual(sum(row['status']=='not_run' for row in report['trials']),5)
        self.assertEqual(FakeClient.instances,[])

    def test_sampled_artifact_scan_tolerates_vanished_descendants_only(self):
        import os
        from contextlib import nullcontext
        from unittest.mock import patch
        root = self.root/'live-artifacts'
        root.mkdir()
        child = root/'native-temporary.bin'
        child.write_bytes(b'temporary')
        original_scandir = os.scandir
        def snapshot_then_retire(path):
            with original_scandir(path) as stream:
                entries = list(stream)
            child.unlink()
            return nullcontext(iter(entries))
        with patch('corridorkey.calibration_task.os.scandir',side_effect=snapshot_then_retire):
            self.assertEqual(_size(root,100),0)
        nested = root/'native-temporary-dir'
        nested.mkdir()
        def directory_retired_before_descent(path):
            if Path(path)==nested:
                nested.rmdir()
            return original_scandir(path)
        with patch('corridorkey.calibration_task.os.scandir',side_effect=directory_retired_before_descent):
            self.assertEqual(_size(root,100),0)
        root.rmdir()
        with self.assertRaises(FileNotFoundError):
            _size(root,100)
        root.mkdir()
        child.write_bytes(b'123')
        with self.assertRaises(ArtifactBudgetError):
            _size(root,2)
        child.unlink()
        child.symlink_to(self.root/'missing-target')
        with self.assertRaises(ValueError):
            _size(root,100)
        child.unlink()
        other = self.root/'regular-owned-elsewhere'
        other.write_bytes(b'1')
        os.link(other,child)
        with self.assertRaises(ArtifactBudgetError):
            _size(root,100)
        child.unlink()
        child.write_bytes(b'1')
        with original_scandir(root) as stream:
            entry = next(stream)
        with (patch('corridorkey.calibration_task.os.scandir',side_effect=lambda path:nullcontext(iter([entry]*100001))),
              self.assertRaisesRegex(ArtifactBudgetError,'file-count')):
            _size(root,200000)

    def test_sampled_artifact_scan_does_not_hide_root_loss_during_churn(self):
        import os
        from contextlib import nullcontext
        from unittest.mock import patch
        root = self.root/'lost-artifacts'
        root.mkdir()
        child = root/'temporary'
        child.write_bytes(b'1')
        original_scandir = os.scandir
        def root_retired(path):
            with original_scandir(path) as stream:
                entries = list(stream)
            child.unlink()
            root.rmdir()
            return nullcontext(iter(entries))
        with (patch('corridorkey.calibration_task.os.scandir',side_effect=root_retired),
              self.assertRaises(FileNotFoundError)):
            _size(root,100)
