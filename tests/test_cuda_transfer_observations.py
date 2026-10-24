"""CUDA admission/callback plumbing fixtures; not actual GPU timing evidence."""
import itertools
import unittest
from unittest.mock import Mock, call, patch

import torch

from corridorkey import runtime
from corridorkey.diagnostics import RunDiagnostics


class CudaObservationTests(unittest.TestCase):
    def test_runtime_owned_waits_have_no_extra_sync_calls_and_never_hold_collector_lock(self):
        diagnostics = RunDiagnostics(profile_stages=True, clock=itertools.count().__next__)
        calls = []
        def synchronize(index):
            assert diagnostics._stage_lock is not None
            self.assertTrue(diagnostics._stage_lock.acquire(blocking=False))
            diagnostics._stage_lock.release()
            calls.append(index)
        with patch.object(runtime.torch.cuda, 'synchronize', synchronize), patch.object(runtime.torch.cuda, 'reset_peak_memory_stats'):
            runtime._begin_resource_profile(diagnostics, 1)
            with diagnostics.stage('compute'):
                pass
        self.assertEqual(calls, [1, 1])
        rows = {row['name']: row for row in diagnostics.stages}
        self.assertEqual(rows['cuda_sync_before_scope']['seconds'], 1)
        self.assertEqual(rows['cuda_sync_after_scope']['seconds'], 1)
        self.assertEqual(rows['compute']['seconds'], 3)
        self.assertEqual(rows['compute']['method'], 'synchronized_wall')
        self.assertEqual(rows['cuda_sync_before_scope']['method'], 'host_wall')

    def test_transfer_admission_rejects_cpu_off_generic_and_mismatched_device(self):
        display = torch.zeros(2, 3, 64, 64, dtype=torch.bfloat16)
        diagnostic = RunDiagnostics(profile_stages=True)
        with patch.object(runtime.torch.cuda, 'is_available', return_value=True), \
             patch.object(runtime.torch.cuda, 'synchronize'), patch.object(runtime.torch.cuda, 'reset_peak_memory_stats'):
            self.assertFalse(runtime._profile_rvm_rgb_upload(diagnostic, display, torch.device('cuda:1')))
            runtime._begin_resource_profile(diagnostic, 1)
            self.assertTrue(runtime._profile_rvm_rgb_upload(diagnostic, display, torch.device('cuda:1')))
            self.assertFalse(runtime._profile_rvm_rgb_upload(diagnostic, display, torch.device('cuda:0')))
            self.assertFalse(runtime._profile_rvm_rgb_upload(diagnostic, display, torch.device('cuda')))
            self.assertFalse(runtime._profile_rvm_rgb_upload(diagnostic, display, torch.device('cpu')))
            self.assertFalse(runtime._profile_rvm_rgb_upload(RunDiagnostics(), display, torch.device('cuda:1')))
            diagnostic.synchronize = Mock()
            self.assertFalse(runtime._profile_rvm_rgb_upload(diagnostic, display, torch.device('cuda:1')))

    def test_generic_callback_and_replacement_preserve_old_clock_semantics(self):
        clock = iter([0, 1, 2, 3, 4, 5])
        diagnostic = RunDiagnostics(profile_stages=True, clock=lambda: next(clock))
        callback = Mock()
        diagnostic.synchronize = callback
        with diagnostic.stage('generic'):
            pass
        with diagnostic.stage('host', gpu=False):
            pass
        self.assertEqual([r['seconds'] for r in diagnostic.stages], [1, 1])
        callback.assert_has_calls([call(), call()])
        self.assertEqual(callback.call_count, 2)
        self.assertNotIn('cuda_stage_observations', diagnostic.snapshot())
        replacement = RunDiagnostics(profile_stages=True, clock=iter(range(10)).__next__)
        replacement._configure_cuda_synchronizer(Mock(), 1)
        replacement.synchronize = Mock()
        with replacement.stage('replaced'):
            pass
        self.assertEqual([(r['name'], r['seconds']) for r in replacement.stages], [('replaced', 1)])

    def test_captured_device_is_not_later_default_and_unsupported_inputs_have_no_metric(self):
        diagnostic = RunDiagnostics(profile_stages=True)
        cpu = torch.zeros(2, 3, 64, 64, dtype=torch.bfloat16)
        meta = torch.empty(2, 3, 64, 64, device='meta')
        with patch.object(runtime.torch.cuda, 'synchronize') as sync, \
             patch.object(runtime.torch.cuda, 'reset_peak_memory_stats'), \
             patch.object(runtime.torch.cuda, 'is_available', return_value=True), \
             patch.object(runtime.torch.cuda, 'current_device', side_effect=AssertionError('must use captured index')):
            runtime._begin_resource_profile(diagnostic, 1)
            self.assertTrue(runtime._profile_rvm_rgb_upload(diagnostic, cpu, torch.device('cuda:1')))
            self.assertFalse(runtime._profile_rvm_rgb_upload(diagnostic, meta, torch.device('cuda:1')))
            self.assertFalse(runtime._profile_rvm_rgb_upload(diagnostic, object(), torch.device('cuda:1')))
            with diagnostic.stage('device-bound'):
                pass
            sync.assert_has_calls([call(1), call(1)])
            self.assertEqual(sync.call_count, 2)
        with patch.object(runtime.torch.cuda, 'is_available', return_value=False):
            self.assertFalse(runtime._profile_rvm_rgb_upload(diagnostic, cpu, torch.device('cuda:1')))

    def test_wait_error_and_body_error_keep_original_status_and_call_population(self):
        for side_effect, rows, calls in (([RuntimeError('pre')], [('cuda_sync_before_scope', 'failed')], 1),
                ([None, RuntimeError('post')], [('cuda_sync_before_scope', 'complete'),
                    ('cuda_sync_after_scope', 'failed'), ('compute', 'failed')], 2)):
            diagnostic = RunDiagnostics(profile_stages=True, clock=itertools.count().__next__)
            with patch.object(runtime.torch.cuda, 'synchronize', side_effect=side_effect) as synchronize, \
                 patch.object(runtime.torch.cuda, 'reset_peak_memory_stats'):
                runtime._begin_resource_profile(diagnostic, 0)
                with self.assertRaises(RuntimeError), diagnostic.stage('compute'):
                    pass
                self.assertEqual(synchronize.call_count, calls)
                self.assertEqual([(r['name'], r['status']) for r in diagnostic.stages], rows)
        diagnostic = RunDiagnostics(profile_stages=True, clock=itertools.count().__next__)
        with patch.object(runtime.torch.cuda, 'synchronize') as synchronize, patch.object(runtime.torch.cuda, 'reset_peak_memory_stats'):
            runtime._begin_resource_profile(diagnostic, 0)
            with self.assertRaisesRegex(RuntimeError, 'body primary'), diagnostic.stage('compute'):
                raise RuntimeError('body primary')
            self.assertEqual(synchronize.call_count, 1, 'original body failure never executes post barrier')
            self.assertEqual([(r['name'], r['status']) for r in diagnostic.stages],
                [('cuda_sync_before_scope', 'complete'), ('compute', 'failed')])

    def test_runtime_binding_creates_no_self_cycle_and_cpu_off_paths_do_not_wait(self):
        import weakref
        diagnostic = RunDiagnostics(profile_stages=True)
        with patch.object(runtime.torch.cuda, 'reset_peak_memory_stats'):
            runtime._begin_resource_profile(diagnostic, 0)
        reference = weakref.ref(diagnostic)
        del diagnostic
        self.assertIsNone(reference(), 'global torch function partial must not retain diagnostics')
        quiet = RunDiagnostics(clock=Mock(return_value=0))
        quiet.synchronize = Mock(side_effect=AssertionError('off must not synchronize'))
        with quiet.stage('off'):
            pass
        quiet.clock.assert_called_once()
        self.assertEqual(quiet.stages, [])

    def test_cpu_recurrent_miss_hit_pixels_and_states_emit_no_cuda_copy_measurement(self):
        import numpy as np

        from corridorkey.recovery_store import BundleStore
        from tests import test_rvm_prepass
        fixture = test_rvm_prepass.RvmPrepassTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        diag = RunDiagnostics(profile_stages=True)
        with BundleStore(fixture.root / 'cache', 100000) as store, \
             patch.object(test_rvm_prepass, 'RunDiagnostics', return_value=diag):
            first, _ = fixture.run_prepass(fixture.adapter(store))
            self.assertGreater(len(fixture.calls), 0)
            fixture.calls.clear()
            replay, _ = fixture.run_prepass(fixture.adapter(store))
        self.assertEqual(fixture.calls, [])
        np.testing.assert_array_equal(replay, first)
        self.assertEqual(diag.stages, [], 'CPU execution is not a CUDA transfer or synchronization observation')

    def test_dashboard_maps_new_lanes_without_additive_cost_claim(self):
        from corridorkey.feasibility import project
        from tests.test_feasibility import inference_fixture
        diag = RunDiagnostics(profile_stages=True, clock=itertools.count().__next__)
        with patch.object(runtime.torch.cuda, 'synchronize'), patch.object(runtime.torch.cuda, 'reset_peak_memory_stats'):
            runtime._begin_resource_profile(diag, 1)
            # Synthetic callback plumbing only; no CPU transfer is represented here.
            with diag.stage('rvm_rgb_upload_and_float'):
                pass
        report = inference_fixture()
        report['runs'][0]['jobs'][0]['diagnostics'].update(diag.snapshot())
        projected = project(report)
        lanes = projected['samples'][0]['jobs'][0]['stage_lanes']
        self.assertEqual([r['name'] for r in lanes['transfer']], ['rvm_rgb_upload_and_float'])
        self.assertEqual({r['name'] for r in lanes['synchronization']}, {'cuda_sync_before_scope', 'cuda_sync_after_scope'})
        self.assertTrue(any('never summed' in warning for warning in projected['warnings']))

    def test_nested_upload_profile_uses_one_normal_pair_and_wait_observation_adds_none(self):
        diag = RunDiagnostics(profile_stages=True, clock=itertools.count().__next__)
        with patch.object(runtime.torch.cuda, 'synchronize') as synchronize, patch.object(runtime.torch.cuda, 'reset_peak_memory_stats'):
            runtime._begin_resource_profile(diag, 1)
            # Scope/callback fixture only: no tensor is transferred on this CPU host.
            with diag.stage('rvm_prepass'), diag.stage('rvm_rgb_upload_and_float'):
                pass
        self.assertEqual(synchronize.call_count, 4, 'one pair per enabled GPU scope, no wait-only extra pair')
        rows = {row['name']: row for row in diag.stages}
        self.assertEqual(rows['cuda_sync_before_scope']['count'], 2)
        self.assertEqual(rows['cuda_sync_after_scope']['count'], 2)
        self.assertEqual(rows['rvm_rgb_upload_and_float']['count'], 1)

    def test_upload_expression_preserves_inline_order_arguments_and_temporary_lifetimes(self):
        import ast
        import inspect
        source = ast.parse(inspect.getsource(runtime.precompute_rvm_to_memmap))
        expected = ast.dump(ast.parse('display.unsqueeze(0).to(device, non_blocking=True).float()', mode='eval').body)
        expressions = [ast.dump(node.value) for node in ast.walk(source) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'rgb' for target in node.targets)]
        self.assertEqual(expressions.count(expected), 2, 'eligible/fallback must retain original complete inline expression')
