"""Tiny local assets are hash fixtures, never model compatibility evidence."""
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.asset_inventory import AssetSelection
from corridorkey.calibration import (
    CalibrationLimits,
    CalibrationPlan,
    CalibrationProfile,
    calibration_validity,
    inventory_calibration,
    plan_calibration,
    revalidate_plan,
    summarize_trials,
)
from corridorkey.config import InferenceConfig
from corridorkey.session import JobRequest


def calibration_fixture(root:Path,*,profiles=None,limits=None,frames=2) -> tuple[JobRequest,dict[str,Any]]:
    """Reusable actual verified plan fixture for parent Qt lifecycle tests.

    root must be a fresh existing directory. Assets contain tiny arbitrary bytes,
    never deserialized; use only an explicitly injected test worker factory.
    """
    root = root.resolve()
    source = root/'source'
    source.mkdir()
    for index in range(4):
        if not cv2.imwrite(str(source/f'{index}.png'),np.full((8,12,3),index*30,np.uint8)):
            raise AssertionError('PNG fixture write failed')
    (source/'.worker-control.json').write_text('{}')
    models = root/'models'
    for relative in ('weights/corridorkey_v2.safetensors','weights/foundation_stats.safetensors',
                     'vfms/rvm/pretrain_rvm_small16_256_204031069.npz','vfms/moge/model.pt',
                     'vfms/cradio/C-RADIOv4-SO400M/config.json'):
        path = models/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'Explicit test-only local model hash fixture')
    scratch,reports = root/'scratch',root/'reports'
    scratch.mkdir(); reports.mkdir()
    job = JobRequest.create(source,root/'original-output',InferenceConfig(output_format='exr',outputs={'alpha'},temp_dir=root/'original-temp',start_frame=1,num_frames=3))
    values: dict[str,Any] = {'sample_start':1,'frame_count':frames,'profiles':profiles or (CalibrationProfile.capture('baseline',{}),),
              'asset_selection':AssetSelection(models),'scratch_parent':scratch,'report_parent':reports,
              'limits':limits or CalibrationLimits(read_budget=64*1024**2)}
    return job,values


class CalibrationPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.job,self.options = calibration_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_metadata_then_budgeted_verified_plan_preserves_settings_sources(self):
        before = self.job
        with patch('corridorkey.calibration.read_file',side_effect=AssertionError('metadata only')):
            inventory = inventory_calibration(self.job,**self.options)
        self.assertEqual((inventory['total_jobs'],inventory['warmup_jobs'],inventory['total_frames']),(9,3,18))
        self.assertGreater(inventory['read_bytes_required'],0)
        plan = plan_calibration(self.job,**self.options)
        self.assertEqual(plan.summary['sample_start'],1)
        self.assertEqual(plan.summary['frame_count'],2)
        self.assertEqual(self.job,before)
        self.assertEqual(list(self.options['scratch_parent'].iterdir()),[])
        self.assertEqual(list(self.options['report_parent'].iterdir()),[])
        self.assertTrue(all(pin.sha256 for pin in plan.pins))
        self.assertEqual(len(plan.to_dict()['identities']['models']),5)
        detached = plan.summary
        detached['profiles'].clear()
        self.assertEqual(len(plan.summary['profiles']),1)
        revalidate_plan(plan)

    def test_budget_missing_model_and_matrix_limits_before_hash(self):
        with patch('corridorkey.calibration.read_file',side_effect=AssertionError('admit first')):
            with self.assertRaisesRegex(ValueError,'read budget'):
                plan_calibration(self.job,**dict(self.options,limits=CalibrationLimits(read_budget=0)))
            with self.assertRaisesRegex(ValueError,'workload'):
                plan_calibration(self.job,**dict(self.options,limits=CalibrationLimits(max_jobs=8)))
        models = self.options['asset_selection'].root
        (models/'vfms/moge/model.pt').unlink()
        self.assertTrue(inventory_calibration(self.job,**self.options)['missing'])
        with self.assertRaisesRegex(ValueError,'unavailable'):
            plan_calibration(self.job,**self.options)

    def test_strict_original_json_classes_unsupported_conditioning_and_purpose(self):
        bad = replace(self.job,config_json='{"hann_chunk":70,'+self.job.config_json[1:])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            plan_calibration(bad,**self.options)
        invalid:Any = JobRequest
        with self.assertRaises(TypeError):
            plan_calibration(invalid,**self.options)
        with self.assertRaises(ValueError):
            CalibrationProfile.capture('bad',{'outputs':['fg']})
        with self.assertRaises(ValueError):
            plan_calibration(self.job,**dict(self.options,profiles=(CalibrationProfile.capture('draft',{'model_scale':.5},'final'),)))
        config,writer = self.job.configurations()
        config.corrections_path = self.root/'missing-corrections.json'
        changed = JobRequest.create(self.job.input_path,self.job.output_dir,config,writer)
        with self.assertRaisesRegex(ValueError,'corrections'):
            plan_calibration(changed,**self.options)
        with self.assertRaises(TypeError):
            CalibrationProfile.capture('bad',{'model_scale':{1,'heterogeneous'}})

    def test_source_model_membership_and_same_size_mutation_invalidates(self):
        plan = plan_calibration(self.job,**self.options)
        path = Path(self.job.input_path)/'1.png'
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0]^1])+raw[1:])
        with self.assertRaises(ValueError):
            revalidate_plan(plan)
        path.write_bytes(raw)
        plan = plan_calibration(self.job,**self.options)
        (self.options['asset_selection'].root/'vfms/cradio/C-RADIOv4-SO400M/new.json').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError,'membership'):
            revalidate_plan(plan)

    def test_write_paths_aliases_profile_recipe_and_expiry(self):
        with self.assertRaises(ValueError):
            plan_calibration(self.job,**dict(self.options,scratch_parent=Path(self.job.input_path)))
        profiles = (CalibrationProfile.capture('baseline',{}),CalibrationProfile.capture('draft',{'model_scale':.5},'preview'))
        plan = plan_calibration(self.job,**dict(self.options,profiles=profiles))
        self.assertEqual(plan.summary['total_jobs'],18)
        self.assertEqual(json.loads(plan.to_dict()['profiles'][1]['job']['config_json'])['model_scale'],.5)
        now = datetime.now(UTC)
        report = {'created_utc':now.isoformat(),'binding_identity':'current'}
        self.assertTrue(calibration_validity(report,'current',now=now)['valid'])
        self.assertFalse(calibration_validity(report,'changed',now=now)['valid'])
        self.assertFalse(calibration_validity(report,'current',now=now+timedelta(days=8))['valid'])
        invalid:Any = CalibrationPlan
        with self.assertRaises(TypeError):
            revalidate_plan(invalid)

    def test_hints_refused_without_stripping_and_serialization_contracts(self):
        hinted = replace(self.job,initial_hint_path=str(self.root/'hint.png'))
        with self.assertRaisesRegex(ValueError,'hints are unsupported'):
            plan_calibration(hinted,**self.options)
        from corridorkey.calibration import canonical
        for value in (CalibrationLimits, CalibrationLimits(), {1,'mixed'}):
            with self.assertRaises(TypeError):
                canonical(value)

    def test_synthetic_aggregation_duration_clipping_rates_and_unverified_fixtures(self):
        # Pure fabricated numbers test arithmetic only, not execution evidence.
        plan = plan_calibration(self.job,**self.options).to_dict()
        data = {'plan':plan,'status':'complete','trials':[]}
        for seconds in (1.,2.,4.):
            data['trials'].append({'profile':'baseline','state':'warm_session','eligible':True,
                                   'status':'complete','elapsed_seconds':seconds+1,
                                   'measurement':{'neural_gpu_verified':True,'processing_seconds':seconds,'frames':2}})
        result = summarize_trials(data)
        estimate = result['groups'][1]['estimated_render_duration']
        self.assertEqual(estimate['frames'],3)  # original In1, three available/requested
        self.assertEqual((estimate['min_seconds'],estimate['central_seconds'],estimate['max_seconds']),(1.5,3.,6.))
        self.assertIn('not a confidence interval',estimate['scope'])
        self.assertEqual(result['recommendations'][0]['estimated_render_duration'],estimate)
        settings = json.loads(plan['requested_job']['config_json'])
        settings['num_frames'] = 200
        plan['requested_job']['config_json'] = json.dumps(settings)
        self.assertEqual(summarize_trials(data)['groups'][1]['estimated_render_duration']['frames'],3)
        settings['num_frames'] = -1
        plan['requested_job']['config_json'] = json.dumps(settings)
        self.assertEqual(summarize_trials(data)['groups'][1]['estimated_render_duration']['frames'],3)
        data['trials'][0]['measurement']['neural_gpu_verified'] = False
        self.assertEqual(summarize_trials(data)['groups'][1]['estimated_render_duration']['status'],'NOT VERIFIED')
        data['trials'][0]['measurement']['neural_gpu_verified'] = True
        del data['trials'][0]['measurement']['processing_seconds']
        self.assertIsNone(summarize_trials(data)['groups'][1]['estimated_render_duration']['central_seconds'])
        data['trials'].pop()
        self.assertEqual(summarize_trials(data)['groups'][1]['estimated_render_duration']['status'],'NOT VERIFIED')

    def test_original_range_admission_and_aggregation_sample_count_guard(self):
        config,writer = self.job.configurations()
        config.start_frame = 10
        changed = JobRequest.create(self.job.input_path,self.job.output_dir,config,writer)
        with self.assertRaisesRegex(ValueError,'Original requested source In'):
            plan_calibration(changed,**self.options)
        data = {'plan':{'summary':{'frame_count':0},'limits':{'repetitions':3}}}
        with self.assertRaisesRegex(ValueError,'sample frame count'):
            summarize_trials(data)
