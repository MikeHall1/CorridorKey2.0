"""Bounded optional stage telemetry; controlled component fixtures, not long video."""
import unittest
from typing import Any
from unittest.mock import Mock

from corridorkey.diagnostics import RunDiagnostics
from corridorkey.feasibility import project
from corridorkey.worker_protocol import encode_frame


class StageRetentionTests(unittest.TestCase):
    def recorded(self, count):
        now = iter(range(2 * count + 1))
        diagnostics = RunDiagnostics(profile_stages=True, clock=lambda: next(now))
        for _ in range(count):
            with diagnostics.stage('stage2_inference', gpu=False):
                pass
        return diagnostics

    def test_ten_thousand_scopes_remain_projectable_with_exact_aggregate_counts(self):
        from tests.test_feasibility import inference_fixture
        diagnostics = self.recorded(10000)
        snapshot: Any = diagnostics.snapshot()
        report = inference_fixture()
        report['runs'][0]['jobs'][0]['diagnostics'].update(snapshot)
        projected = project(report)
        job = projected['samples'][0]['jobs'][0]
        self.assertLessEqual(len(snapshot['stages']), 64)
        self.assertEqual(snapshot['stage_observations']['observations'], 10000)
        row = snapshot['stages'][0]
        self.assertEqual((row['count'], row['seconds'], row['min_seconds'], row['max_seconds']), (10000, 10000, 1, 1))
        self.assertEqual(job['stages'][0]['count'], 10000)

    def test_large_controlled_scope_count_fits_actual_terminal_wire(self):
        diagnostics = self.recorded(60000)
        frame = encode_frame({'kind': 'result', 'payload': {'record': {'diagnostics': diagnostics.snapshot()}}})
        self.assertLess(len(frame), 65536)

    def test_sync_failure_nested_and_disabled_legacy_contract(self):
        clock = iter(range(20))
        diagnostics = RunDiagnostics(profile_stages=True, clock=lambda: next(clock))
        sync = diagnostics.synchronize = Mock()
        with diagnostics.stage('outer'), diagnostics.stage('inner', gpu=False):
            pass
        with self.assertRaisesRegex(RuntimeError, 'primary'), diagnostics.stage('inner', gpu=False):
            raise RuntimeError('primary')
        self.assertEqual(sync.call_count, 2)
        self.assertEqual([(r['name'], r['status'], r['seconds']) for r in diagnostics.stages],
                         [('inner', 'complete', 1), ('outer', 'complete', 3), ('inner', 'failed', 1)])
        quiet = RunDiagnostics(clock=Mock(return_value=0))
        quiet.synchronize = Mock(side_effect=AssertionError('no sync'))
        with quiet.stage('unused'):
            pass
        self.assertEqual(quiet.stages, [])
        self.assertNotIn('stage_observations', quiet.snapshot())
        quiet.clock.assert_called_once()

    def test_fixed_name_cardinality_and_all_method_status_overflows_are_truthful(self):
        import itertools

        from corridorkey.diagnostics import (
            MAX_STAGE_GROUPS,
            MAX_STAGE_NAME_CHARS,
            STAGE_OVERFLOW_NAME,
        )
        from tests.test_feasibility import inference_fixture
        diagnostics = RunDiagnostics(profile_stages=True, clock=itertools.count().__next__)
        diagnostics.synchronize = Mock()
        for index in range(MAX_STAGE_GROUPS - 4):
            with diagnostics.stage(f'known-{index}', gpu=False):
                pass
        for index in range(5000):
            gpu, failed = bool(index % 2), bool(index % 4 < 2)
            try:
                with diagnostics.stage('x' * (MAX_STAGE_NAME_CHARS + 1) if index == 0 else f'unknown-{index}', gpu=gpu):
                    if failed:
                        raise RuntimeError('original stage failure')
            except RuntimeError as exc:
                self.assertEqual(str(exc), 'original stage failure')
        snapshot: Any = diagnostics.snapshot()
        self.assertEqual(len(snapshot['stages']), MAX_STAGE_GROUPS)
        self.assertEqual(snapshot['stage_observations']['observations'], 5060)
        self.assertEqual(snapshot['stage_observations']['overflow_observations'], 5000)
        overflow = [row for row in snapshot['stages'] if row['overflow']]
        self.assertEqual({(r['method'], r['status']) for r in overflow},
                         {(m, s) for m in ('host_wall', 'synchronized_wall') for s in ('complete', 'failed')})
        self.assertTrue(all(r['name'] == STAGE_OVERFLOW_NAME and r['count'] == 1250 for r in overflow))
        self.assertTrue(all(len(r['name']) <= MAX_STAGE_NAME_CHARS for r in snapshot['stages']))
        self.assertEqual(sum(r['seconds'] for r in snapshot['stages']), 5060)
        report = inference_fixture()
        report['runs'][0]['jobs'][0]['diagnostics'].update(snapshot)
        job = project(report)['samples'][0]['jobs'][0]
        self.assertTrue(all(r['name'] != STAGE_OVERFLOW_NAME for rows in job['stage_lanes'].values()
                            if isinstance(rows, list) for r in rows))

    def test_same_name_method_status_groups_preserve_exact_extrema_and_mutable_rows(self):
        clock = iter([0, 1, 3, 4, 9, 10, 13, 14, 18])
        diagnostics = RunDiagnostics(profile_stages=True, clock=lambda: next(clock))
        diagnostics.synchronize = Mock()
        for gpu in (False, False, True):
            with diagnostics.stage('compute', gpu=gpu):
                pass
        with self.assertRaises(RuntimeError), diagnostics.stage('compute', gpu=False):
            raise RuntimeError('primary')
        host, sync, failed = diagnostics.stages
        self.assertEqual((host['count'], host['seconds'], host['min_seconds'], host['max_seconds']), (2, 7, 2, 5))
        self.assertEqual((sync['method'], sync['seconds'], failed['status'], failed['seconds']),
                         ('synchronized_wall', 3, 'failed', 4))
        # Existing CPU-only workflow fixture annotates these mutable rows after runtime returns.
        for row in diagnostics.stages:
            row['method'] = 'host_wall_explicit_cpu_fixture'
        snapshot: Any = diagnostics.snapshot()
        self.assertTrue(all(r['method'] == 'host_wall_explicit_cpu_fixture' for r in snapshot['stages']))

    def test_sync_errors_keep_original_failure_and_method_status_evidence(self):
        import itertools
        diagnostics = RunDiagnostics(profile_stages=True, clock=itertools.count().__next__)
        diagnostics.synchronize = Mock(side_effect=[None, RuntimeError('sync failed')])
        with self.assertRaisesRegex(RuntimeError, 'sync failed'), diagnostics.stage('compute'):
            pass
        self.assertEqual((diagnostics.stages[0]['method'], diagnostics.stages[0]['status']),
                         ('synchronized_wall', 'failed'))
        diagnostics.synchronize = Mock(side_effect=RuntimeError('pre-sync failed'))
        with self.assertRaisesRegex(RuntimeError, 'pre-sync failed'), diagnostics.stage('not-entered'):
            self.fail('must not enter')
        snapshot: Any = diagnostics.snapshot()
        self.assertEqual(snapshot['stage_observations']['observations'], 1)

    def test_projection_refuses_malformed_aggregates_and_retains_legacy_rows(self):
        import copy

        from tests.test_feasibility import inference_fixture
        original: Any = self.recorded(2).snapshot()
        for change in ({'count': False}, {'min_seconds': -1}, {'max_seconds': 5}, {'overflow': True}):
            report = inference_fixture()
            value = copy.deepcopy(original)
            value['stages'][0].update(change)
            report['runs'][0]['jobs'][0]['diagnostics'].update(value)
            with self.assertRaises(ValueError):
                project(report)
        for change in ({'observations': 3}, {'overflow_observations': 1}, {'format': 'unknown'}, {'group_limit': True}):
            report = inference_fixture()
            value = copy.deepcopy(original)
            value['stage_observations'].update(change)
            report['runs'][0]['jobs'][0]['diagnostics'].update(value)
            with self.assertRaises(ValueError):
                project(report)
        legacy = project(inference_fixture())['samples'][0]['jobs'][0]
        self.assertNotIn('stage_observations', legacy)
        self.assertNotIn('count', legacy['stages'][0])

    def test_aggregate_moments_duplicate_groups_and_named_capacity_are_enforced(self):
        import copy

        from tests.test_feasibility import inference_fixture
        original: Any = self.recorded(2).snapshot()
        cases = []
        total = copy.deepcopy(original)
        total['stages'][0]['seconds'] = 100
        cases.append(total)
        duplicate = copy.deepcopy(original)
        duplicate['stages'].append(dict(duplicate['stages'][0]))
        duplicate['stage_observations']['observations'] = 4
        cases.append(duplicate)
        named = copy.deepcopy(original)
        named['stages'] = [{**named['stages'][0], 'name': f'named-{i}'} for i in range(61)]
        named['stage_observations']['observations'] = 122
        cases.append(named)
        for value in cases:
            report = inference_fixture()
            report['runs'][0]['jobs'][0]['diagnostics'].update(value)
            with self.subTest(value=value['stage_observations']), self.assertRaises(ValueError):
                project(report)

    def test_aggregate_extrema_occurrence_nonfinite_bounds_and_valid_rounding(self):
        import copy

        from tests.test_feasibility import inference_fixture
        original: Any = self.recorded(2).snapshot()
        def check_row(row, valid):
            value = copy.deepcopy(original)
            value['stages'][0].update(row)
            value['stage_observations']['observations'] = row['count']
            report = inference_fixture()
            report['runs'][0]['jobs'][0]['diagnostics'].update(value)
            if valid:
                project(report)
            else:
                with self.assertRaises(ValueError):
                    project(report)
        for row in ({'count': 3, 'seconds': 3., 'min_seconds': 1., 'max_seconds': 2.},
                    {'count': 3, 'seconds': 6., 'min_seconds': 1., 'max_seconds': 2.},
                    {'count': 1, 'seconds': 2., 'min_seconds': 1., 'max_seconds': 2.},
                    {'count': 3, 'seconds': 1e308, 'min_seconds': 1e308, 'max_seconds': 1e308},
                    {'count': 10**400, 'seconds': 1., 'min_seconds': 0., 'max_seconds': 1.},
                    {'count': 2, 'seconds': 3e-300, 'min_seconds': 1e-300, 'max_seconds': 1e-300}):
            check_row(row, False)
        for row in ({'count': 3, 'seconds': .1 + .2 + .1, 'min_seconds': .1, 'max_seconds': .2},
                    {'count': 10000, 'seconds': sum([.1] * 10000), 'min_seconds': .1, 'max_seconds': .1},
                    {'count': 1000000, 'seconds': sum([.1] * 1000000), 'min_seconds': .1, 'max_seconds': .1},
                    {'count': 1, 'seconds': 1e-300, 'min_seconds': 1e-300, 'max_seconds': 1e-300},
                    {'count': 2, 'seconds': 2e-300, 'min_seconds': 1e-300, 'max_seconds': 1e-300}):
            check_row(row, True)
