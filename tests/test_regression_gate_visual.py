"""Bound declarations over actual analytic EXR exports are not human authentication."""
import copy
import hashlib
import json
import unittest
from pathlib import Path

from corridorkey import regression_gate as gate
from tests import test_regression_gate_numerical as numerical


class VisualGateTests(unittest.TestCase):
    def setUp(self):
        fixture = numerical.NumericalGateTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.report, self.policy = fixture.root, fixture.report, fixture.policy
        self.report_path, self.policy_path, self.evidence_path = fixture.report_path, fixture.policy_path, fixture.evidence_path
        self.declaration_path, self.visual_path = self.root / 'declaration.json', self.root / 'visual.json'
        self.policy.update(schema_version=4, visual_review_required=True,
            visual={'mode': 'bound_declaration_required', 'selections': [{'pair': 0, 'job': 0, 'source_frames': [0]}]},
            accepted_tradeoffs=['Fixture-only declared tradeoff; no real inspection claim'])
        self.write_policy()
        self.write_numerical()
        self.declaration = {'schema_version': 1, 'kind': 'regression_visual_declaration',
            'reviewer': 'Analytic test fixture, not a real reviewer', 'outcome': 'accept',
            'policy_sha256': self.policy_sha, 'numerical_evidence_sha256': self.numerical_sha,
            'selections': copy.deepcopy(self.policy['visual']['selections']),
            'accepted_tradeoffs': list(self.policy['accepted_tradeoffs']),
            'observations': 'Generated acceptance declaration for contract tests only.'}
        self.write_declaration()

    def write_policy(self):
        self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()

    def write_numerical(self):
        self.evidence_path.write_text(json.dumps(gate.capture_numerical(self.report_path, self.policy_path, self.policy_sha)))
        self.numerical_sha = hashlib.sha256(self.evidence_path.read_bytes()).hexdigest()

    def write_declaration(self):
        self.declaration_path.write_text(json.dumps(self.declaration))
        self.declaration_sha = hashlib.sha256(self.declaration_path.read_bytes()).hexdigest()

    def capture(self):
        from corridorkey.regression_visual import capture
        return capture(self.report_path, self.policy_path, self.policy_sha,
            self.evidence_path, self.numerical_sha, self.declaration_path, self.declaration_sha)

    def evaluate(self, evidence):
        self.visual_path.write_text(json.dumps(evidence))
        return gate.evaluate(self.report_path, self.policy_path, self.policy_sha,
            numerical_evidence_path=self.evidence_path, numerical_evidence_sha256=self.numerical_sha,
            visual_evidence_path=self.visual_path,
            visual_evidence_sha256=hashlib.sha256(self.visual_path.read_bytes()).hexdigest())

    def test_bound_accept_is_not_authenticated_human_review(self):
        evidence = self.capture()
        result = self.evaluate(evidence)
        self.assertEqual(result['status'], 'pass', result)
        row = next(c for c in result['checks'] if c['name'] == 'visual_review')
        self.assertEqual(row['human_inspection_status'], 'not_verified')
        self.assertIn('not authenticated', row['trust_scope'])
        self.assertEqual(evidence['bindings'][0]['source_frames'], [0])
        self.assertEqual(set(evidence['bindings'][0]['sides']), {'baseline', 'candidate'})

    def test_verified_human_requirement_never_passes_a_declaration(self):
        self.policy['visual']['mode'] = 'verified_human_review_required'
        self.write_policy()
        self.write_numerical()
        self.declaration.update(policy_sha256=self.policy_sha, numerical_evidence_sha256=self.numerical_sha)
        self.write_declaration()
        result = self.evaluate(self.capture())
        self.assertEqual(result['status'], 'not_verified')

    def refresh(self):
        self.policy['context_sha256'] = gate.digest(gate.context(self.report))
        self.report_path.write_text(json.dumps(self.report))
        self.write_policy()
        self.write_numerical()
        self.declaration.update(policy_sha256=self.policy_sha, numerical_evidence_sha256=self.numerical_sha,
            selections=copy.deepcopy(self.policy['visual']['selections']), accepted_tradeoffs=self.policy['accepted_tradeoffs'])
        self.write_declaration()

    def test_reject_unable_missing_and_legacy_review_never_become_pass(self):
        for outcome, expected in (('reject', 'fail'), ('unable_to_assess', 'not_verified')):
            self.declaration['outcome'] = outcome
            self.write_declaration()
            self.assertEqual(self.evaluate(self.capture())['status'], expected)
        missing = gate.evaluate(self.report_path, self.policy_path, self.policy_sha,
            numerical_evidence_path=self.evidence_path, numerical_evidence_sha256=self.numerical_sha)
        self.assertEqual(missing['status'], 'not_verified')
        self.policy['schema_version'] = 2
        del self.policy['visual']
        self.write_policy()
        self.write_numerical()
        legacy = gate.evaluate(self.report_path, self.policy_path, self.policy_sha,
            numerical_evidence_path=self.evidence_path, numerical_evidence_sha256=self.numerical_sha)
        self.assertEqual(legacy['status'], 'not_verified')

    def test_capture_never_fills_missing_or_foreign_reviewer_claims(self):
        original = copy.deepcopy(self.declaration)
        for field in original:
            self.declaration = {k: v for k, v in original.items() if k != field}
            self.write_declaration()
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.capture()
        changes = ({'reviewer': ''}, {'reviewer': 'x' * 257}, {'outcome': 'PASS'},
            {'policy_sha256': 'a' * 64}, {'numerical_evidence_sha256': 'b' * 64},
            {'selections': []}, {'accepted_tradeoffs': []}, {'observations': 'x' * 4097},
            {'selections': [{'pair': 0, 'job': 0, 'source_frames': [False]}]})
        for change in changes:
            self.declaration = {**original, **change}
            self.write_declaration()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.capture()

    def test_exact_absolute_frame_subset_binds_both_native_sides(self):
        for side in gate.SIDES:
            self.report['provenance'][side]['normalized_batch']['jobs'][0]['settings']['start_frame'] = 7
        for pair in self.report['pairs']:
            for sample in pair['samples'].values():
                record = sample['runs'][0]['jobs'][0]
                record['requested_settings']['start_frame'] = 7
                record['diagnostics']['effective_settings']['start_frame'] = 7
                record['diagnostics']['source']['start_frame'] = 7
                sample['warmup'][0]['requested_settings']['start_frame'] = 7
                sample['warmup'][0]['diagnostics']['effective_settings']['start_frame'] = 7
        self.policy['visual']['selections'][0]['source_frames'] = [7]
        self.refresh()
        captured = self.capture()
        self.assertEqual(self.evaluate(captured)['status'], 'pass')
        for side in gate.SIDES:
            frames = captured['bindings'][0]['sides'][side]['frames']
            self.assertEqual(list(frames), ['7'])
            self.assertEqual(set(frames['7']), {'alpha', 'fg'})
        self.policy['visual']['selections'][0]['source_frames'] = [0]
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'outside actual pinned'):
            self.capture()

    def test_policy_selections_reject_overlap_foreign_jobs_and_excess(self):
        original = copy.deepcopy(self.policy)
        invalid = ([], [{'pair': 0, 'job': 0, 'source_frames': [0, 0]}],
            [{'pair': True, 'job': 0, 'source_frames': [0]}],
            [{'pair': 0, 'job': 0, 'source_frames': [0]}] * 2,
            [{'pair': 0, 'job': 0, 'source_frames': list(range(121))}])
        for rows in invalid:
            self.policy = copy.deepcopy(original)
            self.policy['visual']['selections'] = rows
            self.write_policy()
            with self.assertRaises(ValueError):
                gate._policy(self.policy_path, self.policy_sha)
        for row in ({'pair': 3, 'job': 0, 'source_frames': [0]},
                    {'pair': 0, 'job': 1, 'source_frames': [0]}):
            self.policy = copy.deepcopy(original)
            self.policy['visual']['selections'] = [row]
            self.refresh()
            with self.assertRaisesRegex(ValueError, 'unmeasured pair/job'):
                self.capture()

    def test_v4_requires_numerical_and_independent_optional_resource_contract(self):
        original = copy.deepcopy(self.policy)
        for change in ({'visual_review_required': False}, {'required_evidence': []},
                       {'required_evidence': ['numerical_parity', 'observed_memory']}):
            self.policy = {**original, **change}
            self.write_policy()
            with self.assertRaises(ValueError):
                gate._policy(self.policy_path, self.policy_sha)
        self.policy = copy.deepcopy(original)
        del self.policy['numerical']
        self.write_policy()
        with self.assertRaises(ValueError):
            gate._policy(self.policy_path, self.policy_sha)

    def test_receipt_reconstruction_rejects_binding_and_native_byte_changes(self):
        original = self.capture()
        tampered = copy.deepcopy(original)
        tampered['bindings'][0]['sides']['candidate']['job_id'] = 'foreign'
        self.assertEqual(self.evaluate(tampered)['status'], 'not_verified')
        self.declaration['reviewer'] = 'Changed declaration'
        self.write_declaration()
        self.assertEqual(self.evaluate(original)['status'], 'not_verified')
        self.declaration['reviewer'] = 'Analytic test fixture, not a real reviewer'
        self.write_declaration()
        # Changed EXR bytes invalidate even an unchanged numerical value.
        export = next(iter(original['bindings'][0]['sides']['candidate']['frames']['0'].values()))['path']
        Path(export).write_bytes(Path(export).read_bytes() + b'changed')
        self.assertEqual(self.evaluate(original)['status'], 'not_verified')

    def test_visual_accept_never_overrides_numerical_or_timing_failure(self):
        import cv2
        import numpy as np
        job = self.report['pairs'][0]['samples']['candidate']['runs'][0]['jobs'][0]
        export = Path(job['diagnostics']['outputs']['alpha']) / 'alpha_000000.exr'
        self.assertTrue(cv2.imwrite(str(export), np.full((2, 2), .8, np.float32)))
        self.refresh()
        result = self.evaluate(self.capture())
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(next(c for c in result['checks'] if c['name'] == 'visual_review')['status'], 'pass')

    def test_strict_json_bounds_and_aliases_reject_before_capture(self):
        import os

        from corridorkey import regression_visual as visual
        original = self.declaration_path.read_bytes()
        for raw in (b'{"reviewer":"one","reviewer":"two"}', b'{"value":NaN}',
                    b'{"x":' + b'[' * 65 + b'0' + b']' * 65 + b'}',
                    b' ' * (visual.MAX_DECLARATION_BYTES + 1)):
            self.declaration_path.write_bytes(raw)
            self.declaration_sha = hashlib.sha256(raw).hexdigest()
            with self.assertRaises(ValueError):
                self.capture()
        self.declaration_path.write_bytes(original)
        self.declaration_sha = hashlib.sha256(original).hexdigest()
        alias = self.root / 'declaration-link.json'
        os.link(self.declaration_path, alias)
        with self.assertRaises(ValueError):
            self.capture()
        alias.unlink()
        self.assertEqual(self.declaration_path.read_bytes(), original)

    def test_cancellation_and_post_snapshot_mutation_fail_closed(self):
        from unittest.mock import patch

        from corridorkey.control import CancelledError
        from corridorkey.regression_visual import capture
        args = (self.report_path, self.policy_path, self.policy_sha, self.evidence_path,
                self.numerical_sha, self.declaration_path, self.declaration_sha)
        def cancel():
            raise CancelledError('stop visual capture')
        with self.assertRaises(CancelledError):
            capture(*args, check=cancel)
        original = gate.capture_numerical
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            self.declaration_path.write_bytes(self.declaration_path.read_bytes() + b' ')
            return result
        with patch.object(gate, 'capture_numerical', mutate), self.assertRaisesRegex(ValueError, 'changed'):
            capture(*args)

    def test_cli_capture_evaluate_and_original_input_protection(self):
        import contextlib
        import io
        from unittest.mock import patch

        from corridorkey import regression_visual as visual
        captured = self.root / 'captured-visual.json'
        base = ['--report', str(self.report_path), '--policy', str(self.policy_path), '--policy-sha256', self.policy_sha,
                '--numerical-evidence', str(self.evidence_path), '--numerical-evidence-sha256', self.numerical_sha]
        args = [*base, '--capture-visual', '--visual-declaration', str(self.declaration_path),
                '--visual-declaration-sha256', self.declaration_sha]
        originals = {p: p.read_bytes() for p in (self.report_path, self.policy_path, self.evidence_path, self.declaration_path)}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main([*args, '--output', str(captured)]), 0)
            result_path = self.root / 'evaluated-visual.json'
            self.assertEqual(gate.main([*base, '--visual-evidence', str(captured), '--visual-evidence-sha256',
                hashlib.sha256(captured.read_bytes()).hexdigest(), '--output', str(result_path)]), 0)
        self.assertEqual(json.loads(result_path.read_bytes())['status'], 'pass')
        for path in originals:
            with (patch.object(visual, 'capture', side_effect=AssertionError('must guard before capture')),
                  self.assertRaises(ValueError)):
                gate.main([*args, '--output', str(path)])
        for path, raw in originals.items():
            self.assertEqual(path.read_bytes(), raw)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            gate.main([*base, '--capture-visual', '--output', str(self.root / 'missing.json')])

    def test_late_hardlink_and_parent_redirect_cannot_publish_over_media(self):
        import os
        from unittest.mock import patch

        from corridorkey import regression_visual as visual
        output = self.root / 'late.json'
        original = self.declaration_path.read_bytes()
        capture = visual.capture
        args = ['--report', str(self.report_path), '--policy', str(self.policy_path), '--policy-sha256', self.policy_sha,
                '--capture-visual', '--numerical-evidence', str(self.evidence_path), '--numerical-evidence-sha256', self.numerical_sha,
                '--visual-declaration', str(self.declaration_path), '--visual-declaration-sha256', self.declaration_sha,
                '--output', str(output)]
        def alias(*a, **kw):
            result = capture(*a, **kw)
            os.link(self.declaration_path, output)
            return result
        with patch.object(visual, 'capture', alias), self.assertRaises(ValueError):
            gate.main(args)
        output.unlink()
        self.assertEqual(self.declaration_path.read_bytes(), original)
        parent = self.root / 'future'
        parent.mkdir()
        output = parent / 'visual.json'
        args[-1] = str(output)
        def redirect(*a, **kw):
            result = capture(*a, **kw)
            parent.rmdir()
            parent.symlink_to(self.declaration_path.parent, target_is_directory=True)
            return result
        with patch.object(visual, 'capture', redirect), self.assertRaises(ValueError):
            gate.main(args)
        parent.unlink()
        self.assertEqual(self.declaration_path.read_bytes(), original)
        self.assertFalse((self.root / 'visual.json').exists())

    def test_visual_publication_rechecks_declaration_after_evaluation(self):
        from unittest.mock import patch

        from corridorkey import regression_visual as visual
        self.visual_path.write_text(json.dumps(self.capture()))
        output = self.root / 'must-not-publish.json'
        original = visual.admit
        def change(*args, **kwargs):
            row, inputs = original(*args, **kwargs)
            self.declaration_path.write_bytes(self.declaration_path.read_bytes() + b' ')
            return row, inputs
        with patch.object(visual, 'admit', change), self.assertRaisesRegex(ValueError, 'changed'):
            gate.main(['--report', str(self.report_path), '--policy', str(self.policy_path), '--policy-sha256', self.policy_sha,
                '--numerical-evidence', str(self.evidence_path), '--numerical-evidence-sha256', self.numerical_sha,
                '--visual-evidence', str(self.visual_path), '--visual-evidence-sha256', hashlib.sha256(self.visual_path.read_bytes()).hexdigest(),
                '--output', str(output)])
        self.assertFalse(output.exists())

    def test_optional_resource_phase_remains_independent_and_timing_noise_does_not_hide_reject(self):
        from unittest.mock import patch
        self.policy['resources'] = {'constraints': [{'job': 0, 'metric': 'peak_allocated_vram_bytes', 'maximum_bytes': 100}],
            'limits': {'max_jobs': 24, 'max_frames': 100, 'max_wall_seconds': 60,
                       'max_output_bytes': 1024**2, 'read_budget': 1024**3, 'max_files': 100000}}
        self.policy['required_evidence'].append('observed_memory')
        self.refresh()
        self.visual_path.write_text(json.dumps(self.capture()))
        with patch('corridorkey.regression_resources.admit', return_value=(
                {'name': 'observed_memory', 'status': 'fail'}, [], str(self.root))) as resource:
            result = gate.evaluate(self.report_path, self.policy_path, self.policy_sha,
                numerical_evidence_path=self.evidence_path, numerical_evidence_sha256=self.numerical_sha,
                visual_evidence_path=self.visual_path, visual_evidence_sha256=hashlib.sha256(self.visual_path.read_bytes()).hexdigest(),
                resource_evidence_path=self.root / 'fixture-resource.json', resource_evidence_sha256='a' * 64)
        self.assertEqual(result['status'], 'fail')
        resource.assert_called_once()
        self.assertEqual(next(c for c in result['checks'] if c['name'] == 'visual_review')['status'], 'pass')
        del self.policy['resources']
        self.policy['required_evidence'].remove('observed_memory')
        self.report['pairs'][0]['samples']['candidate']['runs'][0]['elapsed_seconds'] *= 5
        self.declaration['outcome'] = 'reject'
        self.refresh()
        result = self.evaluate(self.capture())
        self.assertEqual(next(c for c in result['checks'] if c['name'] == 'elapsed_seconds')['status'], 'inconclusive')
        self.assertEqual(result['status'], 'fail')

    def test_timing_regression_and_colour_association_identity_remain_effective(self):
        for pair in self.report['pairs']:
            pair['samples']['candidate']['runs'][0]['elapsed_seconds'] *= 5
        self.refresh()
        captured = self.capture()
        result = self.evaluate(captured)
        self.assertEqual(result['status'], 'fail')
        # Even permitted straight foreground association must not silently rebind an earlier declaration.
        for pair in self.report['pairs']:
            for sample in pair['samples'].values():
                sample['runs'][0]['jobs'][0]['diagnostics']['output_contracts']['fg']['alpha_association'] = 'straight'
        self.report_path.write_text(json.dumps(self.report))
        self.assertEqual(self.evaluate(captured)['status'], 'not_verified')
