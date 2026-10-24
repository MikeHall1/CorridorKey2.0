"""Stage2 reuse is qualified by actual complete call inputs, not edit locality."""
import json
import os
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.durable_stage2 import DurableStage2Batches
from corridorkey.recovery_identity import canonical
from corridorkey.session import JobRequest
from tests import test_mixed_finishing_cache as cached
from tests import test_mixed_finishing_recovery as proof
from tests.test_durable_stage2 import cpu_runtime_context


class Stage2IncrementalRecoveryTests(unittest.TestCase):
    root: Path
    source: Path
    hints: Path
    initial: Path
    config: InferenceConfig
    seen: list
    rows: list
    setUp_base = cached.MixedFinishingCacheTests.setUp
    factory = cached.MixedFinishingCacheTests.factory
    hashes = staticmethod(cached.MixedFinishingCacheTests.hashes)
    run_job_base = cached.MixedFinishingCacheTests.run_job

    def setUp(self: Any):
        self.setUp_base()
        texture = cv2.imread(str(self.source/'0.png'))
        for frame in range(8,14):
            self.assertTrue(cv2.imwrite(str(self.source/f'{frame}.png'),np.roll(texture,2*frame,axis=1)))
            self.assertTrue(cv2.imwrite(str(self.hints/f'{frame}.png'),np.full((64,64),12000+frame*900,np.uint16)))
        self.originals = {p.name:p.read_bytes() for p in self.source.iterdir()}
        self.config = replace(self.config,num_frames=12)
        self.current_hints = self.hints
        self.current_initial = self.initial
        self.calls = []

    def scene(self: Any):
        proof.MixedFinishingPreproofTests.scene(self)

    def request(self, config, output):
        return JobRequest.create(self.source,output,config,initial_hint_path=self.current_initial,
                                 hint_video_path=self.current_hints)

    def model(self: Any, trace, held):
        model = proof.MixedFinishingPreproofTests.model(self,trace,held)
        original = model.stage1.forward
        def temporal(*args,**kwargs):
            result = original(*args,**kwargs)
            # Nontrivial window-wide mixing makes genuine Hann lookahead matter.
            influence = args[1].float().mean()*.07
            result = {name:value+influence for name,value in result.items()}
            trace['stage1'][-1]['temporal_outputs'] = proof.frozen(result)
            for value in result.values():held.append((value,proof.frozen(value)))
            return result
        model.stage1.forward = temporal
        return model

    def run_job(self, path, config, workspace, policies=(True,True), **kwargs):
        packets = []
        identity = DurableStage2Batches.identity
        get = DurableStage2Batches.get
        def observed_identity(owner,**inputs):
            result = identity(owner,**inputs)
            if result is not None:
                value = result.to_dict()
                packets.append({'identity':value,'signature':canonical({k:v for k,v in value.items() if k!='parent_key'})})
            return result
        def observed_get(owner,key):
            result = get(owner,key)
            if key is not None:packets[-1]['hit'] = result is not None
            return result
        with patch.object(DurableStage2Batches,'identity',observed_identity),patch.object(DurableStage2Batches,'get',observed_get):
            result = self.run_job_base(path,config,workspace,policies,**kwargs)
        self.calls.append(packets)
        return result

    def edits(self, kind):
        if kind == 'correction':
            editor = CorrectionEditor(self.request(self.config,self.root/'editor'),self.root)
            mask = np.zeros((64,64),np.uint8);mask[8:43,9:34] = 255
            editor.commit_brush(mask,kind='matte',frame=9,strength=.25,direction='both',span=[9,12])
            a = editor.save();entry = editor.document.to_dict()['entries'][0]
            editor.edit(entry['id'],{'strength':.85});b = editor.save()
            return a,b
        return None,None

    def compare(self, kind, config, *, policies=(True,True), second_config=None):
        a,b = self.edits(kind)
        workspace = self.root/f'{kind}-{len(self.rows)}'
        first = self.run_job(a,config,workspace,policies)
        packets_a = self.calls[-1]
        old = first['jobs'][0];preserved = self.hashes(old)
        if kind in ('full_hint','initial_hint'):
            if kind=='full_hint':
                other = self.root/f'hints-B-{len(self.rows)}';shutil.copytree(self.hints,other)
                self.assertTrue(cv2.imwrite(str(other/'10.png'),np.full((64,64),61000,np.uint16)))
                self.current_hints = other
            else:
                other = self.root/f'initial-B-{len(self.rows)}.png'
                self.assertTrue(cv2.imwrite(str(other),np.full((64,64),61000,np.uint16)))
                self.current_initial = other
        second = self.run_job(b,second_config or config,workspace,policies)
        packets_b = self.calls[-1];trace_b = self.seen[-1][0]
        new = second['jobs'][0]
        clean = self.run_job(b,second_config or config,self.root/f'clean-{len(self.rows)}',(False,False),budget=0)
        self.assertEqual(self.hashes(new),self.hashes(clean['jobs'][0]))
        differences = 0
        for current,previous in zip(sorted(Path(new['outputs']['alpha']).glob('*.exr')),
                                    sorted(Path(old['outputs']['alpha']).glob('*.exr'))):
            differences += int(np.count_nonzero(cv2.imread(str(current),-1)!=cv2.imread(str(previous),-1)))
        if second_config is None:self.assertGreater(differences,0,'fixture must change final alpha')
        self.assertNotEqual(old['key'],new['key'])
        self.assertNotEqual(first['journal'],second['journal'])
        self.assertNotEqual(old['output_dir'],new['output_dir'])
        self.assertEqual(new['origin_record']['diagnostics']['durable_writer_prefix']['restored_prefix_frames'],0)
        self.assertEqual(self.hashes(old),preserved)
        self.assertEqual({p.name:p.read_bytes() for p in self.source.iterdir()},self.originals)
        prior = {packet['signature'] for packet in packets_a}
        matches = [packet['signature'] in prior for packet in packets_b]
        expected = [match and policies[0] and second_config is None for match in matches]
        actual = [packet['hit'] for packet in packets_b]
        row = {'kind':kind,'settings':{'low_vram':config.low_vram,'model_scale':config.model_scale,
               'quick':config.quick_key,'approach':config.quick_approach},'policies':policies,
               'A':packets_a,'B':packets_b,'same_actual_calls':matches,'expected_hits':expected,
               'actual_hits':actual,'B_calls':{name:len(values) for name,values in trace_b.items()},
               'B_stage1':trace_b['stage1'],'B_stage2':trace_b['stage2'],'clean_B_exact':True,'alpha_changed_pixels':differences,
               'fresh_full_key_journal_outputs':True,'old_prefix_restored':0,'A_source_preserved':True}
        self.rows.append(row);self.export()
        if kind in ('correction','full_hint') and second_config is None and not config.quick_key:
            self.assertTrue(any(matches),'late edit must leave real early complete batches equal')
            self.assertTrue(any(not item for item in matches),'late edit must actually change model inputs')
            self.assertTrue(matches[0],'edit must lie beyond actual lookahead')
        self.assertEqual(actual,expected,'matching complete calls should hit only with the Stage2 policy')
        self.assertEqual(len(trace_b['stage2']),sum(not value for value in expected))
        self.current_hints = self.hints;self.current_initial = self.initial

    def export(self):
        destination = os.environ.get('CORRIDORKEY_E005_MATRIX')
        if destination:
            path = Path(destination)
            path = path.with_name(path.stem+'-'+self._testMethodName+path.suffix)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(self.rows,indent=2))

    def test_late_corrections_and_full_hints_real_complete_calls(self):
        with cpu_runtime_context():
            for kind in ('correction','full_hint'):
                with self.subTest(kind=kind):self.compare(kind,self.config)

    def test_modes_independent_flags_initial_hint_and_regrouping(self):
        with cpu_runtime_context():
            for overrides in ({'low_vram':True},{'model_scale':.5},
                              {'quick_key':True,'quick_approach':'hold'},
                              {'quick_key':True,'quick_approach':'causal'}):
                with self.subTest(overrides=overrides):self.compare('correction',replace(self.config,**overrides))
            for policies in ((False,False),(False,True),(True,False)):
                with self.subTest(policies=policies):self.compare('full_hint',self.config,policies=policies)
            self.compare('initial_hint',self.config)
            self.compare('correction',self.config,second_config=replace(self.config,stage2_batch=1))

    def test_current_admission_late_mutation_quota_and_cancel(self):
        from corridorkey.control import CancellationToken, CancelledError
        from corridorkey.corrections import load_corrections
        from corridorkey.recovery_store import CacheInvalid
        with cpu_runtime_context():
            a,b = self.edits('correction')
            assert a is not None and b is not None
            workspace = self.root/'integrity'
            first = self.run_job(a,self.config,workspace);preserved = self.hashes(first['jobs'][0])
            data = load_corrections(b).to_dict()
            asset = b.parent/data['entries'][0]['asset']['path'];before = asset.read_bytes()
            asset.write_bytes(before+b'corrupt')
            models = len(self.seen)
            with self.assertRaises((ValueError,CacheInvalid)):self.run_job(b,self.config,workspace)
            self.assertEqual(len(self.seen),models);asset.write_bytes(before)
            hint = self.hints/'10.png';original_hint = hint.read_bytes()
            hint.write_bytes(b'not a decodable image')
            with self.assertRaises((ValueError,RuntimeError,CacheInvalid)):self.run_job(b,self.config,workspace)
            hint.write_bytes(original_hint)
            self.current_hints = self.root/'missing-hints'
            with self.assertRaises(FileNotFoundError):self.run_job(b,self.config,workspace)
            self.current_hints = self.hints
            original_initial = self.initial.read_bytes();self.initial.write_bytes(b'undecodable initial hint')
            with self.assertRaises((ValueError,RuntimeError,CacheInvalid)):self.run_job(b,self.config,workspace)
            self.initial.write_bytes(original_initial)
            for target in ('hint','correction'):
                mutated = False
                content = b.read_bytes()
                late = b.parent/f'late-{target}.json';late.write_bytes(content)
                def progress(event, target=target, content=content, late=late):
                    nonlocal mutated
                    if event.stage=='output_frames' and not mutated:
                        mutated = True
                        if target=='hint':hint.write_bytes(original_hint+b'changed')
                        else:
                            value = json.loads(content);value['revision'] += 1;late.write_text(json.dumps(value))
                with self.assertRaisesRegex(CacheInvalid,'dependencies changed'):
                    self.run_job(late,self.config,workspace,progress=progress)
                self.assertTrue(mutated)
                hint.write_bytes(original_hint);b.write_bytes(content)
            cancel_path = b.parent/'cancel-current.json';cancel_path.write_bytes(b.read_bytes())
            token = CancellationToken()
            def cancel(event):
                if event.stage=='recovery_publish':token.cancel('current incremental publication')
            with self.assertRaises(CancelledError):
                self.run_job(cancel_path,self.config,workspace,progress=cancel,cancellation=token)
            self.assertEqual(self.hashes(first['jobs'][0]),preserved)
            self.run_job(b,self.config,self.root/'zero',budget=0)
            self.assertTrue(self.seen[-1][0]['stage2'])
            self.assertEqual(self.calls[-1],[],'zero budget must retain no optional prediction identity')
