"""Exact temporal events replay raw outputs; current constraints rebuild carry."""
import json
import os
import shutil
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey.correction_editor import CorrectionEditor
from corridorkey.correction_rasters import CorrectionRasters
from corridorkey.durable_hann import DurableHannWindows
from corridorkey.durable_rvm import DurableRvmChunks
from corridorkey.recovery import run_recoverable
from corridorkey.recovery_identity import canonical
from tests import test_mixed_finishing_cache as cache
from tests import test_mixed_finishing_recovery as proof
from tests import test_stage2_incremental_recovery as stage2
from tests.test_durable_stage2 import cpu_runtime_context


class TemporalIncrementalRecoveryTests(unittest.TestCase):
    root: Path
    source: Path
    hints: Path
    initial: Path
    seen: list
    rows: list
    setUp_base = cache.MixedFinishingCacheTests.setUp
    scene = stage2.Stage2IncrementalRecoveryTests.scene
    request = stage2.Stage2IncrementalRecoveryTests.request
    model = stage2.Stage2IncrementalRecoveryTests.model
    factory = cache.MixedFinishingCacheTests.factory
    hashes = staticmethod(cache.MixedFinishingCacheTests.hashes)

    def setUp(self: Any):
        stage2.Stage2IncrementalRecoveryTests.setUp(self)
        self.events = []
        self.maxDiff = None

    def edits(self: Any, kind):
        if kind=='full_hint':return None,None
        editor = CorrectionEditor(self.request(self.config,self.root/'editor'),self.root)
        mask = np.zeros((64,64),np.uint8);mask[8:43,9:34] = 255
        editor.commit_brush(mask,kind='matte',frame=9,strength=.25,direction='both',
                            span=[0 if kind=='carry' else 9,12])
        a = editor.save();entry = editor.document.to_dict()['entries'][0]
        editor.edit(entry['id'],{'strength':.85})
        return a,editor.save()

    def run_job(self: Any, path, config, workspace, *, flags=(True,True,True), reuse=True,
                budget=32_000_000, progress=None, cancellation=None):
        from corridorkey import runtime
        rows = {'rvm':[],'hann':[],'states':[],'refinements':[]};raw = []
        def instrument(adapter,name,stack):
            identity,get,put = adapter.identity,adapter.get,adapter.put
            def identify(owner,*args,**kwargs):
                result = identity(owner,*args,**kwargs);value = result.to_dict()
                rows[name].append({'identity':value,'signature':canonical({k:v for k,v in value.items() if k!='parent_key'}),
                                   'actual_inputs':proof.frozen({'args':args,'kwargs':kwargs})})
                return result
            def lookup(owner,key):
                result = get(owner,key);rows[name][-1]['hit'] = result is not None
                if result is not None:
                    rows[name][-1]['raw_output'] = proof.frozen(result)
                    if name=='hann':raw.append((result['aux'],proof.frozen(result['aux'])))
                return result
            def publish(owner,key,*values):
                value = values[0] if name=='hann' else {'features':values[0],'state':values[1]}
                consumed = {'baton':value['baton'],'aux':value['aux'][:, :,0:1]} if name=='hann' else value
                rows[name][-1]['raw_output'] = proof.frozen(consumed)
                if name=='hann':raw.append((value['aux'],proof.frozen(value['aux'])))
                return put(owner,key,*values)
            stack.enter_context(patch.object(adapter,'identity',identify))
            stack.enter_context(patch.object(adapter,'get',lookup))
            stack.enter_context(patch.object(adapter,'put',publish))
        state_digest = runtime.replay_state_digest
        def state(**values):
            result = state_digest(**values)
            rows['states'].append({'digest':result,'values':proof.frozen({k:v for k,v in values.items() if k!='check'})})
            return result
        apply = CorrectionRasters.apply
        def current(owner,alpha,frame,**kwargs):
            before = proof.frozen(alpha);result = apply(owner,alpha,frame,**kwargs)
            if kwargs.get('temporal'):
                rows['refinements'].append({'frame':frame,'before':before,'after':proof.frozen(result)})
            return result
        with ExitStack() as stack:
            instrument(DurableRvmChunks,'rvm',stack);instrument(DurableHannWindows,'hann',stack)
            stack.enter_context(patch.object(runtime,'replay_state_digest',state))
            stack.enter_context(patch.object(CorrectionRasters,'apply',current))
            result = run_recoverable([self.request(replace(config,corrections_path=path),self.root/'output')],
                workspace=workspace,max_cache_bytes=budget,assets=self.assets,environment=self.environment,
                implementation_root=self.implementation,session_factory=self.factory,
                durable_foundations=flags[0],durable_rvm=flags[1],durable_hann=flags[2],
                durable_stage2=True,durable_writer_prefix=True,reuse_stage2_finishing_edits=True,
                reuse_upstream_finishing_edits=reuse,progress=progress,cancellation=cancellation)
        self.assertEqual(result['status'],'complete',result.get('error'))
        for value,before in raw:self.assertEqual(proof.frozen(value),before,'current constraints must not mutate cached raw aux')
        for value,before in self.seen[-1][1]:self.assertEqual(proof.frozen(value),before)
        rows['model'] = self.seen[-1][0]
        self.events.append(rows)
        return result

    def switch_hint(self):
        other = self.root/f'hints-B-{len(self.rows)}';shutil.copytree(self.hints,other)
        self.assertTrue(cv2.imwrite(str(other/'10.png'),np.full((64,64),61000,np.uint16)))
        self.current_hints = other

    def compare(self: Any, kind, config, *, uncached=False, flags=(True,True,True), reuse=True):
        a,b = self.edits(kind);workspace = self.root/f'{kind}-{len(self.rows)}'
        first = self.run_job(a,config,workspace,flags=flags,reuse=reuse,budget=0 if uncached else 32_000_000)
        left = self.events[-1];old = first['jobs'][0];preserved = self.hashes(old)
        if kind=='full_hint':self.switch_hint()
        second = self.run_job(b,config,self.root/f'uncached-B-{len(self.rows)}' if uncached else workspace,
                              flags=flags,reuse=reuse,budget=0 if uncached else 32_000_000)
        right = self.events[-1];new = second['jobs'][0]
        clean = self.run_job(b,config,self.root/f'clean-{len(self.rows)}',flags=flags,reuse=False,budget=0)
        self.assertEqual(self.hashes(new),self.hashes(clean['jobs'][0]))
        changes = sum(int(np.count_nonzero(cv2.imread(str(x),-1)!=cv2.imread(str(y),-1)))
                      for x,y in zip(sorted(Path(new['outputs']['alpha']).glob('*.exr')),
                                     sorted(Path(old['outputs']['alpha']).glob('*.exr'))))
        self.assertGreater(changes,0)
        for key in ('key','output_dir'):self.assertNotEqual(old[key],new[key])
        self.assertNotEqual(first['journal'],second['journal'])
        self.assertEqual(new['origin_record']['diagnostics']['durable_writer_prefix']['restored_prefix_frames'],0)
        self.assertEqual(self.hashes(old),preserved)
        self.assertEqual({p.name:p.read_bytes() for p in self.source.iterdir()},self.originals)
        matches = {};expected = {}
        for name in ('rvm','hann'):
            prior = {row['signature'] for row in left[name]}
            matches[name] = [row['signature'] in prior for row in right[name]]
            expected[name] = [value and reuse and not uncached for value in matches[name]]
        if flags[1]:
            self.assertTrue(all(matches['rvm']),'unchanged actual source/state partitions must match')
            self.assertTrue(any(row['identity']['incoming_state_sha256'] is not None for row in right['rvm']))
            self.assertTrue(any(row['raw_output']['state']['nonzero'] > 0 for row in right['rvm']))
        if flags[2] and not config.quick_key:
            self.assertTrue(matches['hann'][0],'late anchor must be outside real first window')
            self.assertTrue(any(not value for value in matches['hann']))
        if kind=='carry' and flags[2] and not config.quick_key:
            self.assertEqual(left['hann'][0]['raw_output'],right['hann'][0]['raw_output'])
            self.assertNotEqual(left['states'][1]['values']['carry'],right['states'][1]['values']['carry'])
            self.assertFalse(matches['hann'][1],'current corrected outgoing aux must change next event')
            self.assertNotEqual(left['refinements'][0]['after'],right['refinements'][0]['after'])
        for name in ('rvm','hann'):
            prior_raw = {item['signature']:item['raw_output'] for item in left[name]}
            for item in right[name]:
                if item['signature'] in prior_raw:self.assertEqual(item['raw_output'],prior_raw[item['signature']])
        stats = new['origin_record']['diagnostics']
        quick = stats.get('quick_key') or {}
        plan = {key:quick.get(key) for key in ('anchors','extra_neural_frames','promoted_anchors',
                 'promotion_reasons','fallback_frames','effective_cuts','approach','native_context_policy')}
        row = {'quick_plan':plan,'kind':kind,'uncached':uncached,'flags':flags,'reuse':reuse,
               'settings':{'low':config.low_vram,'scale':config.model_scale,'quick':config.quick_key,
                           'approach':config.quick_approach,'adaptive':config.quick_adaptive},
               'A':left,'B':right,'matches':matches,'expected_hits':expected,'clean_B_exact':True,
               'alpha_different_pixels':changes,'fresh_full_output':True,'prefix_restored':0,
               'raw_aux_immutable':True,'A_source_preserved':True}
        self.rows.append(row);self.export()
        for name in ('rvm','hann'):
            self.assertEqual([item['hit'] for item in right[name]],expected[name],name+' exact event qualification')
        self.current_hints = self.hints

    def export(self):
        destination = os.environ.get('CORRIDORKEY_E006_MATRIX')
        if destination:
            path = Path(destination);path = path.with_name(path.stem+'-'+self._testMethodName+path.suffix)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(self.rows,indent=2))

    def test_uncached_current_aux_carry_witness(self: Any):
        with cpu_runtime_context():
            for kind in ('correction','full_hint','carry'):
                with self.subTest(kind=kind):self.compare(kind,self.config,uncached=True)

    def test_temporal_qualification_late_edits_and_current_aux(self: Any):
        with cpu_runtime_context():
            for kind in ('correction','full_hint','carry'):
                with self.subTest(kind=kind):self.compare(kind,self.config)

    def test_modes_independent_durable_stages_flags_and_real_adaptive_fallback(self: Any):
        from corridorkey import draft
        from corridorkey.motion import Propagation
        offline,past = draft.offline_frames,draft.past_frames
        def motion(*args,**kwargs):
            alpha = next(value for value in args if isinstance(value,np.ndarray) and value.ndim==2).copy()
            return Propagation(alpha,np.zeros_like(alpha),np.ones_like(alpha,dtype=bool),
                               {'flow_evaluations':4,'motion_seconds':0.,'uncertain_fraction':1.,'motion_fraction':1.})
        def offline_current(*args,**kwargs):return offline(*args,interpolate=motion,**kwargs)
        def past_current(*args,**kwargs):return past(*args,propagate=motion,**kwargs)
        with cpu_runtime_context(),patch.object(draft,'offline_frames',offline_current),patch.object(draft,'past_frames',past_current):
            for kind,overrides in (('carry',{'low_vram':True}),('carry',{'model_scale':.5}),
                                  ('correction',{'quick_key':True,'quick_approach':'hold'}),
                                  ('full_hint',{'quick_key':True,'quick_adaptive':True}),
                                  ('carry',{'quick_key':True,'quick_approach':'causal','quick_adaptive':True})):
                with self.subTest(kind=kind,overrides=overrides):self.compare(kind,replace(self.config,**overrides))
            for flags,reuse in (((True,True,True),False),((True,False,False),True),
                                ((False,True,False),True),((False,False,True),True)):
                with self.subTest(flags=flags,reuse=reuse):self.compare('correction',self.config,flags=flags,reuse=reuse)

    def test_current_integrity_late_publication_quota_and_cancel(self: Any):
        from corridorkey.control import CancellationToken, CancelledError
        from corridorkey.corrections import load_corrections
        from corridorkey.recovery_store import CacheInvalid
        with cpu_runtime_context():
            a,b = self.edits('carry');assert a is not None and b is not None
            workspace = self.root/'current-integrity'
            first = self.run_job(a,self.config,workspace);preserved = self.hashes(first['jobs'][0])
            data = load_corrections(b).to_dict();asset = b.parent/data['entries'][0]['asset']['path']
            original = asset.read_bytes();asset.write_bytes(original+b'corrupt')
            before_models = len(self.seen)
            with self.assertRaises((ValueError,CacheInvalid)):self.run_job(b,self.config,workspace)
            self.assertEqual(len(self.seen),before_models);asset.write_bytes(original)
            hint = self.hints/'10.png';before_hint = hint.read_bytes();hint.write_bytes(b'not an image')
            with self.assertRaises((ValueError,RuntimeError,CacheInvalid)):self.run_job(b,self.config,workspace)
            hint.write_bytes(before_hint)
            self.current_hints = self.root/'missing-hints'
            with self.assertRaises(FileNotFoundError):self.run_job(b,self.config,workspace)
            self.current_hints = self.hints
            before_initial = self.initial.read_bytes();self.initial.write_bytes(b'not an initial image')
            with self.assertRaises((ValueError,RuntimeError,CacheInvalid)):self.run_job(b,self.config,workspace)
            self.initial.write_bytes(before_initial)
            for target in ('hint','correction'):
                late = b.parent/f'late-{target}.json';content = b.read_bytes();late.write_bytes(content)
                mutated = False
                def progress(event, target=target, late=late, content=content):
                    nonlocal mutated
                    if event.stage=='output_frames' and not mutated:
                        mutated = True
                        if target=='hint':hint.write_bytes(before_hint+b'late mutation')
                        else:
                            value = json.loads(content);value['revision'] += 1;late.write_text(json.dumps(value))
                with self.assertRaisesRegex(CacheInvalid,'dependencies changed'):
                    self.run_job(late,self.config,workspace,progress=progress)
                self.assertTrue(mutated);hint.write_bytes(before_hint)
            cancel_path = b.parent/'cancel.json';cancel_path.write_bytes(b.read_bytes())
            token = CancellationToken()
            def cancel(event):
                if event.stage=='recovery_publish':token.cancel('current temporal publication')
            with self.assertRaises(CancelledError):
                self.run_job(cancel_path,self.config,workspace,progress=cancel,cancellation=token)
            self.assertEqual(self.hashes(first['jobs'][0]),preserved)
            self.run_job(b,self.config,self.root/'zero',budget=0)
            for name in ('rvm','hann'):self.assertFalse(any(item['hit'] for item in self.events[-1][name]))
            self.assertTrue(self.seen[-1][0]['rvm']);self.assertTrue(self.seen[-1][0]['stage1'])

    def test_parent_dispatch_builds_only_enabled_domain(self: Any):
        from corridorkey import recovery_identity
        legacy,temporal = recovery_identity.upstream_finishing_context,recovery_identity.temporal_finishing_context
        with cpu_runtime_context():
            for index,(flags,reuse) in enumerate((((True,False,False),True),((False,True,False),True),
                                                ((False,False,True),True),((True,True,True),False))):
                with patch.object(recovery_identity,'upstream_finishing_context',wraps=legacy) as iid, \
                     patch.object(recovery_identity,'temporal_finishing_context',wraps=temporal) as state:
                    self.run_job(None,self.config,self.root/f'dispatch-{index}',flags=flags,reuse=reuse)
                self.assertEqual(iid.call_count,int(reuse and flags[0]))
                self.assertEqual(state.call_count,int(reuse and (flags[1] or flags[2])))
