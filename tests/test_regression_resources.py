"""Portable resource producer/receipt checks, not CUDA or memory-fit measurements."""
import unittest
from contextlib import contextmanager
from unittest.mock import patch


class ProfileDeviceTests(unittest.TestCase):
    def test_profile_queries_bind_captured_selected_device_not_later_default(self):
        from corridorkey import runtime
        from corridorkey.diagnostics import RunDiagnostics
        diagnostics = RunDiagnostics(profile_stages=True)
        with (patch.object(runtime.torch.cuda, 'reset_peak_memory_stats') as reset,
              patch.object(runtime.torch.cuda, 'synchronize') as synchronize,
              patch.object(runtime.torch.cuda, 'max_memory_allocated', return_value=100) as allocated,
              patch.object(runtime.torch.cuda, 'max_memory_reserved', return_value=200) as reserved,
              patch.object(runtime.torch.cuda, 'get_device_name', return_value='mock selected GPU1') as name):
            runtime._begin_resource_profile(diagnostics, 1)
            assert diagnostics.synchronize is not None
            diagnostics.synchronize()
            resources = runtime._finish_resource_profile(1)
        reset.assert_called_once_with(1)
        synchronize.assert_called_once_with(1)
        allocated.assert_called_once_with(1)
        reserved.assert_called_once_with(1)
        name.assert_called_once_with(1)
        self.assertEqual(resources['device_index'], 1)
        self.assertEqual(resources['reset_device_index'], 1)
        self.assertEqual(resources['peak_allocated_vram_bytes'], 100)
        self.assertEqual(resources['peak_reserved_vram_bytes'], 200)


class ResourcePolicyTests(unittest.TestCase):
    def test_version3_resource_policy_admission_is_explicit_and_additive(self):
        import hashlib
        import json

        from corridorkey.regression_gate import _policy
        from tests.test_regression_gate import RegressionGateTests
        fixture = RegressionGateTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        policy = fixture.policy
        policy.update(schema_version=3, required_evidence=['observed_memory'], resources={
            'constraints': [{'job': 0, 'metric': 'peak_allocated_vram_bytes', 'maximum_bytes': 100}],
            'limits': {'max_jobs': 24, 'max_frames': 100, 'max_wall_seconds': 60,
                       'max_output_bytes': 1024**2, 'read_budget': 1024**3, 'max_files': 100000}})
        raw = json.dumps(policy).encode()
        fixture.policy_path.write_bytes(raw)
        admitted, _ = _policy(fixture.policy_path, hashlib.sha256(raw).hexdigest())
        self.assertEqual(admitted['schema_version'], 3)
        self.assertNotIn('numerical', admitted)

    def test_resource_phase_accepts_v3_and_v4_with_resources_only(self):
        import tempfile
        from pathlib import Path

        from corridorkey import regression_resources as resources
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            pair, policy = root / 'pair.json', root / 'policy.json'
            pair.write_text('{}'); policy.write_text('{}')
            for version in (3, 4):
                with (self.subTest(version=version), patch('corridorkey.regression_gate._policy', return_value=(
                        {'schema_version': version, 'resources': {}}, b'{}')),
                        patch.object(resources, 'read_report', side_effect=RuntimeError('admitted; next bounded report read')),
                        self.assertRaisesRegex(RuntimeError, 'next bounded report read')):
                    resources._prepare(pair, policy, 'fixture', None)
            for value in ({'schema_version': 2, 'resources': {}}, {'schema_version': 4}):
                with patch('corridorkey.regression_gate._policy', return_value=(value, b'{}')), self.assertRaises(ValueError):
                    resources._prepare(pair, policy, 'fixture', None)


class ResourceArithmeticTests(unittest.TestCase):
    def test_allocator_max_boundary_and_regression_not_rss(self):
        from corridorkey.regression_resources import summarize, validate_policy
        rows = [{'phase': 'measured', 'job': 0, 'resources': {'peak_allocated_vram_bytes': n}}
                for n in (80, 90, 100, 95, 90, 98)]
        constraint = {'job': 0, 'metric': 'peak_allocated_vram_bytes', 'maximum_bytes': 100}
        self.assertEqual(summarize(rows, [constraint], 3)[0]['status'], 'pass')
        constraint['maximum_bytes'] = 99
        self.assertEqual(summarize(rows, [constraint], 3)[0]['status'], 'fail')
        with self.assertRaises(ValueError):
            summarize(rows[:-1], [constraint], 3)
        with self.assertRaises(ValueError):
            validate_policy({'constraints': [{'job': 0, 'metric': 'rss', 'maximum_bytes': 100}], 'limits': {}})


class RepositoryBudgetTests(unittest.TestCase):
    def test_optional_bound_preserves_digest_exact_boundary_and_cancellation(self):
        import tempfile
        from pathlib import Path

        from corridorkey import benchmark
        from corridorkey.control import CancelledError
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'tracked.py').write_bytes(b'abc')
            def command(*args):
                if args[1] == 'ls-files':
                    return b'tracked.py\0'
                return b'fixture'
            with patch.object(benchmark, 'ROOT', root), patch.object(benchmark, 'command', side_effect=command):
                legacy = benchmark.repository_identity()
                self.assertEqual(benchmark.repository_identity(max_bytes=3), legacy)
                with self.assertRaisesRegex(ValueError, 'byte limit'):
                    benchmark.repository_identity(max_bytes=2)
                def cancel():
                    raise CancelledError('fixture')
                with self.assertRaises(CancelledError):
                    benchmark.repository_identity(max_bytes=3, check=cancel)
                checks = [0]
                def grow():
                    checks[0] += 1
                    if checks[0] == 3:
                        (root / 'tracked.py').write_bytes(b'abcd')
                with self.assertRaisesRegex(ValueError, 'grew or changed'):
                    benchmark.repository_identity(max_bytes=3, check=grow)



# Integration cases below pin package files; run only after the coordinator's source freeze.
class ResourcePhaseTests(unittest.TestCase):
    def setUp(self):
        import copy
        import hashlib
        import json
        from pathlib import Path

        from corridorkey import benchmark
        from corridorkey.calibration import _model_files
        from corridorkey.config import InferenceConfig
        from corridorkey.identities import source_identity
        from corridorkey.regression_gate import context, digest
        from corridorkey.runtime_paths import RuntimePaths
        from corridorkey.session import JobRequest
        from corridorkey.worker_protocol import job_to_wire
        from tests.test_calibration import calibration_fixture
        from tests.test_calibration_task import FakeClient
        from tests.test_regression_gate import RegressionGateTests
        fixture = RegressionGateTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root.resolve()
        original, options = calibration_fixture(self.root)
        paths = RuntimePaths.from_package(benchmark.ROOT, asset_root=options['asset_selection'].root)
        self.model_paths = {k: str(getattr(paths, k)) for k in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint', 'cradio_dir')}
        model_pins, _ = _model_files(options['asset_selection'], None)
        model_hashes = {p.path: hashlib.sha256(Path(p.path).read_bytes()).hexdigest() for p in model_pins}
        self.report, self.policy = fixture.report, fixture.policy
        self.report_path, self.policy_path = self.root / 'pair.json', self.root / 'policy.json'
        self.parent = self.root / 'attempts'
        self.parent.mkdir()
        class Client(FakeClient):
            model_paths = self.model_paths
            def submit_preview(self, job):
                return self.send('preview', {'job': job_to_wire(job)})
            def cancel(self, target, reason):
                return self.send('cancel', {'target': target})
            def send(self, action, payload):
                request = super().send(action, payload)
                if action == 'cancel':
                    return request
                record = self.event['payload']['record']
                record.update(job_id=f'analytic-{id(self)}-{self.jobs}',
                              hints={'first_frame': None, 'full': None}, asset_paths=self.model_paths)
                diagnostics = record['diagnostics']
                count = diagnostics['frames_submitted']
                diagnostics['source']['output_hw'] = [8, 12]
                diagnostics['execution'] = {'backend': 'pytorch', 'requested_device': 'cuda',
                    'gpu': 'analytic fixture; NOT hardware measurement', 'device_index': 1, 'cuda_version': 'fixture',
                    'autocast_dtype': 'bfloat16', 'completed_prediction_frames': count,
                    'stage2_frame_evaluations': count, 'stage2_calls': 1,
                    'temporal_strategy': 'full_rate_with_chronological_rvm_prepass'}
                diagnostics['profiling_enabled'] = payload['job']['profile_stages']
                diagnostics['stages'] = [{'method': 'synchronized_wall', 'status': 'complete', 'seconds': 1}]
                from corridorkey.regression_resources import RESET_SCOPE, RESOURCE_SCOPE
                diagnostics['resources'] = {'peak_allocated_vram_bytes': 100, 'peak_reserved_vram_bytes': 200,
                    'gpu': diagnostics['execution']['gpu'], 'cuda_version': 'fixture', 'device_index': 1,
                    'reset_device_index': 1, 'scope': RESOURCE_SCOPE, 'reset_scope': RESET_SCOPE}
                return request
        self.Client = Client
        FakeClient.instances = []; FakeClient.live = 0
        originals = {}
        for side, budget in (('baseline', 0), ('candidate', 1)):
            cfg = InferenceConfig(output_format='exr', outputs={'alpha'}, start_frame=1, num_frames=2,
                                  foundation_cache_mb=budget, temp_dir=self.root / 'original-temp')
            job = JobRequest.create(original.input_path, self.root / f'original-{side}', cfg)
            client = Client()
            client.send('preview', {'job': job_to_wire(job)})
            record = copy.deepcopy(client.event['payload']['record']); client.close()
            record['model_reused'] = True
            record['diagnostics']['profiling_enabled'] = False
            record['diagnostics']['resources'] = None
            originals[side] = record
            self.report['provenance'][side] = {'normalized_batch': {'schema_version': 1, 'kind': 'inference_batch',
                'weights': self.model_paths['weights'], 'jobs': [{'input': job.input_path, 'output_dir': str(self.root / 'frozen-pair-staging' / side / '0'),
                    'settings': json.loads(job.config_json), 'writer': json.loads(job.writer_json),
                    'hint_first_frame': None, 'hint_video': None}]}, 'frozen_manifest': {'model_state': 'warm_session'}}
        for pair in self.report['pairs']:
            for side in ('baseline', 'candidate'):
                sample = pair['samples'][side]
                record = copy.deepcopy(originals[side]); record['job_id'] = f'original-{pair["index"]}-{side}'
                actual_root = self.root / 'inference-artifacts' / f'{pair["index"]}-{side}'
                sample['artifact_dir'] = str(actual_root)
                record['output_dir'] = str(actual_root / '0' / '0')
                record['diagnostics']['outputs'] = {kind: str(Path(record['output_dir']) / kind)
                                                     for kind in record['diagnostics']['outputs']}
                sample['runs'][0]['jobs'] = [record]
                warm = copy.deepcopy(record); warm.update(model_reused=False, job_id=f'warm-{pair["index"]}-{side}')
                warm['output_dir'] = str(actual_root / 'warmup' / '0')
                warm['diagnostics']['outputs'] = {kind: str(Path(warm['output_dir']) / kind) for kind in warm['diagnostics']['outputs']}
                sample['warmup'] = [warm]
                sample['weights'] = model_hashes
                sample['assets'] = {original.input_path: source_identity(original.input_path)}
                sample['environment']['hardware'].update(gpu=record['diagnostics']['execution']['gpu'],
                    device_index=1, cuda_version='fixture', driver='fixture driver')
        self.policy.update(schema_version=3, context_sha256=digest(context(self.report)), required_evidence=['observed_memory'],
            resources={'constraints': [{'job': 0, 'metric': 'peak_allocated_vram_bytes', 'maximum_bytes': 100}],
                'limits': {'max_jobs': 24, 'max_frames': 48, 'max_wall_seconds': 60,
                           'max_output_bytes': 1024**2, 'read_budget': 1024**3, 'max_files': 100000}})
        self.report_path.write_text(json.dumps(self.report)); self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        self.signature = {'repository': self.report['pairs'][0]['samples']['baseline']['repository'], 'fixture': True}
        self.live = self.enterContext(patch('corridorkey.regression_resources._live', return_value=self.signature))

    @contextmanager
    def temp_only_provenance(self):
        from pathlib import Path

        from corridorkey import regression_resources as resources
        actual = Path.glob
        package = Path(resources.__file__).parent
        def selected(path, pattern, *args, **kwargs):
            return iter(()) if path == package and pattern == '*.py' else actual(path, pattern, *args, **kwargs)
        with (patch.object(resources.benchmark, 'command', return_value=b''),
              patch.object(Path, 'glob', selected)):
            yield

    def test_verified_source_hashes_serialize_without_a_second_content_read(self):
        from pathlib import Path

        from corridorkey import regression_resources as resources
        from corridorkey.identities import source_identity
        source = self.report['pairs'][0]['samples']['baseline']['runs'][0]['jobs'][0]['input']
        expected = source_identity(source)
        actual = resources.read_file
        source_reads = []
        def read(pin, **kwargs):
            value = actual(pin, **kwargs)
            if Path(pin.path).parent == Path(source):
                source_reads.append(pin.path)
            return value
        with (self.temp_only_provenance(), patch.object(resources, 'read_file', side_effect=read),
              patch.object(resources.benchmark, 'asset_identity', side_effect=AssertionError('unbounded source reread'))):
            prepared = resources._prepare(self.report_path, self.policy_path, self.policy_sha, None)
        self.assertEqual(len(source_reads), len(expected['files']))
        self.assertEqual(len(set(source_reads)), len(source_reads))
        self.assertEqual(prepared[-1]['sources'][source]['files'], source_reads)

    def test_source_cancel_growth_and_missing_file_are_bounded(self):
        from pathlib import Path

        from corridorkey import regression_resources as resources
        from corridorkey.control import CancellationToken, CancelledError
        source = Path(self.report['pairs'][0]['samples']['baseline']['runs'][0]['jobs'][0]['input'])
        actual = resources.read_file
        token = CancellationToken()
        completed = []
        def cancelled(pin, **kwargs):
            value = actual(pin, **kwargs)
            if Path(pin.path).parent == source:
                completed.append(pin.path)
                token.cancel('after first verified source frame')
            return value
        with self.temp_only_provenance(), patch.object(resources, 'read_file', side_effect=cancelled), self.assertRaises(CancelledError):
            resources._prepare(self.report_path, self.policy_path, self.policy_sha, token.check)
        self.assertEqual(len(completed), 1)
        path = source / '0.png'
        raw = path.read_bytes()
        reads = []
        def grown(pin, **kwargs):
            value = actual(pin, **kwargs)
            if pin.path == str(path):
                reads.append(pin.size)
                path.write_bytes(raw + b'x' * 2048)
            return value
        with (self.temp_only_provenance(), patch.object(resources, 'read_file', side_effect=grown),
              patch.object(resources.benchmark, 'asset_identity', side_effect=AssertionError('unbounded grown read')),
              self.assertRaisesRegex(ValueError, 'changed')):
            resources._prepare(self.report_path, self.policy_path, self.policy_sha, None)
        self.assertEqual(reads, [len(raw)])
        path.unlink()
        with self.temp_only_provenance(), self.assertRaisesRegex(ValueError, 'content changed'):
            resources._prepare(self.report_path, self.policy_path, self.policy_sha, None)

    def test_membership_additions_and_order_use_bounded_metadata_only(self):
        from pathlib import Path

        from corridorkey import regression_resources as resources
        with self.temp_only_provenance():
            prepared = resources._prepare(self.report_path, self.policy_path, self.policy_sha, None)
        membership = prepared[-1]
        resources._membership_fence(membership, 100000, None)
        source = Path(next(iter(membership['sources'])))
        extra = source / '10.png'
        extra.write_bytes(b'new frame, never decoded')
        with self.assertRaisesRegex(ValueError, 'membership'):
            resources._membership_fence(membership, 100000, None)
        extra.unlink()
        extra = Path(self.model_paths['cradio_dir']) / 'new.json'
        extra.write_bytes(b'new model member, never deserialized')
        with self.assertRaisesRegex(ValueError, 'Model membership'):
            resources._membership_fence(membership, 100000, None)

    def test_correction_dotdot_keys_compare_admitted_asset_hash_before_worker(self):
        import hashlib
        import json

        from corridorkey import regression_resources as resources
        from corridorkey.regression_gate import context, digest
        snapshots = self.root / 'snapshots'; snapshots.mkdir()
        assets = self.root / 'assets'; assets.mkdir()
        asset = assets / 'mask.png'; asset.write_bytes(b'original admitted hash fixture')
        document = snapshots / 'corrections.json'; document.write_bytes(b'{}')
        correction = {'path': str(document), 'manifest_sha256': hashlib.sha256(document.read_bytes()).hexdigest(),
                      'assets': [{'path': str(snapshots / '..' / 'assets' / 'mask.png'),
                                  'sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}]}
        for side in ('baseline', 'candidate'):
            self.report['provenance'][side]['normalized_batch']['jobs'][0]['settings']['corrections_path'] = str(document)
        for pair in self.report['pairs']:
            for sample in pair['samples'].values():
                sample['corrections'] = [correction]
                for record in [*sample['runs'][0]['jobs'], *sample['warmup']]:
                    record['requested_settings']['corrections_path'] = str(document)
                    record['diagnostics']['effective_settings']['corrections_path'] = str(document)
        self.policy['context_sha256'] = digest(context(self.report))
        self.report_path.write_text(json.dumps(self.report)); self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        with self.temp_only_provenance():
            prepared = resources._prepare(self.report_path, self.policy_path, self.policy_sha, None)
        self.assertIn(str(asset), [p.path for p in prepared[4]])
        asset.write_bytes(b'changed hash fixture')
        with self.temp_only_provenance(), self.assertRaisesRegex(ValueError, 'content differs'):
            resources._prepare(self.report_path, self.policy_path, self.policy_sha, None)

    def test_cancelled_artifacts_are_retained_without_content_hashing(self):
        from pathlib import Path

        from corridorkey import regression_resources as resources
        from corridorkey.control import CancellationToken
        token = CancellationToken()
        class Cancel(self.Client):
            def next_event(self, timeout):
                event = super().next_event(timeout)
                token.cancel('stop before artifact verification')
                return event
        actual = resources.read_file
        def no_artifact_hash(pin, **kwargs):
            self.assertFalse(Path(pin.path).is_relative_to(self.parent), 'failed artifacts must not be hashed')
            return actual(pin, **kwargs)
        with self.temp_only_provenance(), patch.object(resources, 'read_file', side_effect=no_artifact_hash):
            attempt = resources.run_phase(self.report_path, self.policy_path, self.policy_sha,
                                          self.parent, worker_factory=Cancel, cancellation=token)
        receipt = attempt.to_dict()
        self.assertEqual(receipt['status'], 'cancelled')
        self.assertEqual(receipt['artifact_verification'], 'not_verified')
        self.assertEqual(receipt['artifacts'], [])
        self.assertTrue(Path(receipt['trials'][0]['record']['path']).exists())

    def test_original_producer_relocation_validates_without_mutating_frozen_template(self):
        import copy

        from corridorkey.batch import jobs_from_data
        from corridorkey.regression_resources import _validate_original
        for side in ('baseline', 'candidate'):
            template = jobs_from_data(self.report['provenance'][side]['normalized_batch'], self.report_path)[0][0]
            before = template
            for pair in self.report['pairs']:
                original = pair['samples'][side]['runs'][0]['jobs'][0]
                self.assertNotEqual(template.output_dir, original['output_dir'])
                _validate_original(original, template)
                self.assertEqual(template, before)
                for defect in ('source', 'settings', 'writer', 'hints', 'profile'):
                    bad = copy.deepcopy(original)
                    if defect == 'source':
                        bad['input'] = str(self.root / 'wrong-source')
                    elif defect == 'settings':
                        bad['requested_settings']['stage2_batch'] += 1
                    elif defect == 'writer':
                        bad['writer_settings']['threads'] += 1
                    elif defect == 'hints':
                        bad['hints']['first_frame'] = str(self.root / 'wrong-hint.png')
                    else:
                        bad['diagnostics']['profiling_enabled'] = True
                    with self.assertRaises(ValueError):
                        _validate_original(bad, template)

    def test_separate_lifecycle_fixture_never_qualifies_and_does_not_change_pair_timings(self):
        import hashlib

        from corridorkey.regression_gate import evaluate
        from corridorkey.regression_resources import run_phase
        original = self.report_path.read_bytes()
        attempt = run_phase(self.report_path, self.policy_path, self.policy_sha, self.parent, worker_factory=self.Client)
        receipt = attempt.to_dict()
        self.assertEqual(receipt['status'], 'complete', receipt['error'])
        self.assertEqual(len(receipt['trials']), 12)
        self.assertEqual([r['phase'] for r in receipt['trials'][:2]], ['warmup', 'measured'])
        self.assertEqual(receipt['producer'], 'fixture_provider')
        self.assertTrue(receipt['cleanup_verified'])
        self.assertEqual(self.report_path.read_bytes(), original)
        gate = evaluate(self.report_path, self.policy_path, self.policy_sha,
            resource_evidence_path=attempt.report_path, resource_evidence_sha256=hashlib.sha256(attempt.payload_json).hexdigest())
        self.assertEqual(gate['status'], 'not_verified')

    def test_cpu_record_and_wrong_profile_device_cannot_be_resource_observations(self):
        import copy

        # Use a real immutable request; provider counters are explicit structural fixtures only.
        from corridorkey.batch import jobs_from_data
        from corridorkey.regression_resources import _record
        job = jobs_from_data(self.report['provenance']['baseline']['normalized_batch'], self.report_path)[0][0]
        from dataclasses import replace
        job = replace(job, profile_stages=True)
        client = self.Client()
        try:
            client.submit_preview(job)
            record = client.event['payload']['record']
            original = copy.deepcopy(record)
            self.assertEqual(_record(record, job, original, False)['peak_allocated_vram_bytes'], 100)
            record['diagnostics']['resources']['reset_device_index'] = 0
            with self.assertRaisesRegex(ValueError, 'reset/device'):
                _record(record, job, original, False)
            record = copy.deepcopy(original)
            record['diagnostics']['execution']['neural_inference'] = False
            with self.assertRaises(ValueError):
                _record(record, job, original, False)
        finally:
            client.close()

    def test_inner_verification_callbacks_do_not_rescan_all_pins(self):
        from dataclasses import replace
        from pathlib import Path

        from corridorkey import regression_resources as resources
        from corridorkey.batch import jobs_from_data
        from corridorkey.project_collection import admit_file, read_file
        pins = []
        for index in range(4):
            path = self.root / f'verification-{index}.bin'
            path.write_bytes(bytes([index]))
            pin = admit_file(path)
            sha, _ = read_file(pin)
            pins.append(replace(pin, sha256=sha))
        jobs = {side: jobs_from_data(self.report['provenance'][side]['normalized_batch'], self.report_path)[0]
                for side in ('baseline', 'candidate')}
        roots = dict.fromkeys(jobs, str(Path(self.model_paths['foundation_stats']).parent.parent))
        prepared = (self.policy, self.report, jobs, roots, pins, self.signature, 100, {'sources': {}, 'models': []})
        input_names = {pin.path for pin in pins}
        nested_pin_scans = []
        actual = resources.read_file
        with (patch.object(resources, '_prepare', return_value=prepared),
              patch.object(resources, 'assert_pin', wraps=resources.assert_pin) as assertion):
            def streamed(pin, *, check=None, **kwargs):
                if pin.path in input_names:
                    self.assertIsNotNone(check)
                    assert check is not None
                    before = assertion.call_count
                    check()
                    nested_pin_scans.append(assertion.call_count - before)
                return actual(pin, check=check, **kwargs)
            with patch.object(resources, 'read_file', side_effect=streamed):
                attempt = resources.run_phase(self.report_path, self.policy_path, self.policy_sha,
                                              self.parent, worker_factory=self.Client)
            self.assertGreaterEqual(assertion.call_count, 3 * len(pins))
        self.assertEqual(attempt.to_dict()['status'], 'complete', attempt.to_dict()['error'])
        self.assertEqual(len(nested_pin_scans), len(pins))
        self.assertEqual(nested_pin_scans, [0] * len(pins))

    def test_full_phase_boundary_still_rejects_source_model_and_code_drift(self):
        from dataclasses import replace
        from pathlib import Path

        from corridorkey import regression_resources as resources
        from corridorkey.batch import jobs_from_data
        from corridorkey.project_collection import admit_file, read_file
        jobs = {side: jobs_from_data(self.report['provenance'][side]['normalized_batch'], self.report_path)[0]
                for side in ('baseline', 'candidate')}
        roots = dict.fromkeys(jobs, str(Path(self.model_paths['foundation_stats']).parent.parent))
        for role in ('source', 'model', 'code'):
            with self.subTest(role=role):
                path = self.root / f'{role}-pin.bin'
                path.write_bytes(b'pinned')
                pin = admit_file(path)
                sha, _ = read_file(pin)
                prepared = (self.policy, self.report, jobs, roots, [replace(pin, sha256=sha)], self.signature, 100, {'sources': {}, 'models': []})
                def changed(*args, captured=path, **kwargs):
                    captured.write_bytes(b'edited')
                    return self.signature
                with (patch.object(resources, '_prepare', return_value=prepared),
                      patch.object(resources, '_live', side_effect=changed)):
                    attempt = resources.run_phase(self.report_path, self.policy_path, self.policy_sha,
                                                  self.parent, worker_factory=self.Client)
                self.assertEqual(attempt.to_dict()['status'], 'not_verified')
                self.assertIn('changed', attempt.to_dict()['error'])

    def test_structural_production_receipt_binds_actual_records_and_limit_boundaries(self):
        import hashlib
        import json

        from corridorkey.regression_gate import evaluate
        from corridorkey.regression_resources import run_phase
        # Structural GPU-record simulation only. No CUDA query or real measurement occurs.
        with patch('corridorkey.regression_resources.WorkerClient', self.Client):
            attempt = run_phase(self.report_path, self.policy_path, self.policy_sha, self.parent)
        self.assertEqual(attempt.to_dict()['status'], 'complete', attempt.to_dict()['error'])
        def screen():
            return evaluate(self.report_path, self.policy_path, self.policy_sha,
                resource_evidence_path=attempt.report_path,
                resource_evidence_sha256=hashlib.sha256(attempt.report_path.read_bytes()).hexdigest())
        result = screen()
        self.assertEqual(result['status'], 'pass', result)
        self.assertEqual(result['checks'][-1]['observations'][0]['observed_max_bytes'], 100)
        # Trial deletion is honest malformed evidence, not a substitute for all repetitions.
        raw = attempt.report_path.read_bytes()
        receipt = json.loads(raw); receipt['trials'].pop()
        attempt.report_path.write_text(json.dumps(receipt))
        self.assertEqual(screen()['status'], 'not_verified')
        attempt.report_path.write_bytes(raw)
        self.policy['resources']['constraints'][0]['maximum_bytes'] = 99
        self.policy_path.write_text(json.dumps(self.policy))
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        with patch('corridorkey.regression_resources.WorkerClient', self.Client):
            attempt = run_phase(self.report_path, self.policy_path, self.policy_sha, self.parent)
        result = screen()
        self.assertEqual(result['status'], 'fail', result)

    def test_cancellation_and_unsafe_output_parent_fail_closed(self):
        from pathlib import Path

        from corridorkey.control import CancellationToken
        from corridorkey.regression_resources import run_phase
        from tests.test_calibration_task import FakeClient
        before = len(FakeClient.instances)
        source = Path(self.report['pairs'][0]['samples']['baseline']['runs'][0]['jobs'][0]['input'])
        with self.assertRaises(ValueError):
            run_phase(self.report_path, self.policy_path, self.policy_sha, source, worker_factory=self.Client)
        self.assertEqual(len(FakeClient.instances), before)
        token = CancellationToken()
        class Cancel(self.Client):
            def next_event(self, timeout):
                event = super().next_event(timeout)
                token.cancel('fixture after first completed result')
                return event
        attempt = run_phase(self.report_path, self.policy_path, self.policy_sha, self.parent,
                            worker_factory=Cancel, cancellation=token)
        receipt = attempt.to_dict()
        self.assertEqual(receipt['status'], 'cancelled')
        self.assertTrue(receipt['cleanup_verified'])
        self.assertEqual(len(receipt['trials']), 1)

    def test_memory_report_paths_and_late_alias_fences_preserve_sources(self):
        import hashlib
        import os
        from pathlib import Path

        from corridorkey import regression_numerical
        from corridorkey.regression_gate import main
        from corridorkey.regression_resources import run_phase
        with patch('corridorkey.regression_resources.WorkerClient', self.Client):
            attempt = run_phase(self.report_path, self.policy_path, self.policy_sha, self.parent)
        args = ['--report', str(self.report_path), '--policy', str(self.policy_path), '--policy-sha256', self.policy_sha,
                '--resource-evidence', str(attempt.report_path), '--resource-evidence-sha256', hashlib.sha256(attempt.payload_json).hexdigest()]
        for destination in (self.policy_path, self.report_path, attempt.report_path, attempt.report_path.parent / 'new.json'):
            with self.subTest(destination=destination), self.assertRaises((ValueError, FileExistsError)):
                main([*args, '--output', str(destination)])
        artifact = Path(attempt.to_dict()['trials'][0]['record']['path'])
        output = self.root / 'late.json'
        original = artifact.read_bytes()
        fence = regression_numerical.fence
        def late(inputs, **kwargs):
            os.link(artifact, output)
            fence(inputs, **kwargs)
        try:
            with patch.object(regression_numerical, 'fence', side_effect=late), self.assertRaises(ValueError):
                main([*args, '--output', str(output)])
            self.assertEqual(artifact.read_bytes(), original)
        finally:
            output.unlink(missing_ok=True)


class SequenceIdentityTests(unittest.TestCase):
    def test_verified_natural_order_is_exact_legacy_wire_and_detached(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path

        from corridorkey.identities import sequence_identity, source_identity
        from corridorkey.regression_resources import _files
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for name in ('10.png', '2.png', '1.png'):
                (root / name).write_bytes(name.encode())
            frames = [{'name': name, 'sha256': hashlib.sha256(name.encode()).hexdigest()}
                      for name in ('1.png', '2.png', '10.png')]
            expected = {'path': str(root), 'files': frames,
                        'sha256': hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()}
            self.assertEqual(source_identity(root), expected)
            self.assertEqual([Path(p.path).name for p in _files(root, 10, None)], ['1.png', '2.png', '10.png'])
            detached = sequence_identity(root, frames)
            frames[0]['sha256'] = 'changed caller data'
            self.assertNotEqual(detached['files'], frames)
