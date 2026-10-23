import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'cv2', 'torch'))
if HAS_STACK or TYPE_CHECKING:
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    import numpy as np

    from corridorkey.config import InferenceConfig
    from corridorkey.recovery import run_recoverable
    from corridorkey.recovery_store import CacheInvalid
    from corridorkey.session import InferenceSession, JobRequest
    from corridorkey.video_io import ExrSequenceWriter


@unittest.skipUnless(HAS_STACK, 'requires CPU export adapters; never actual neural inference')
class RecoveryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'synthetic identity input; renderer does not perform neural inference')
        code = self.root / 'implementation/corridorkey'
        code.mkdir(parents=True)
        (code / 'runtime.py').write_text('# CPU test fixture implementation identity')
        self.assets = {}
        for name in ('weights', 'foundation_stats', 'rvm_ckpt', 'moge_checkpoint'):
            path = self.root / name
            path.write_bytes(name.encode())
            self.assets[name] = path
        directory = self.root / 'cradio'
        directory.mkdir()
        (directory / 'config.json').write_text('{}')
        self.assets['cradio_dir'] = directory
        self.environment = {'backend': 'cpu-test-fixture', 'precision': 'fixture-float32',
                            'dependencies': {'fixture': '1'}, 'runtime': {'synthetic': True}}
        self.workspace = self.root / 'workspace'
        self.jobs = [JobRequest.create(self.source, self.root / name,
                                      InferenceConfig(outputs={'alpha'}, output_format='exr')) for name in ('first', 'second')]
        self.loads = 0
        self.renders = []
        self.fail_render = False
        self.change_source = False

    def factory(self, **kwargs):
        def load():
            self.loads += 1
            return object()

        def render(model, job, config, writer_config, diagnostics):
            self.renders.append(job.output_dir)
            directory = Path(job.output_dir) / 'synthetic_alpha_exr'
            writer = ExrSequenceWriter(str(directory), 24., (2, 1), channels=1, dtype='float', frame_name='alpha')
            try:
                writer.write_many(np.tile(np.array([0., 0.5], dtype=np.float32).reshape(1, 1, 2, 1), (2, 1, 1, 1)))
            finally:
                writer.release()
            diagnostics.outputs['alpha'] = str(directory)
            diagnostics.output_contracts['alpha'] = {'format': 'exr', 'channels': 1}
            diagnostics.frames_submitted = 2
            diagnostics.execution = {'backend': 'cpu-test-fixture', 'neural_inference': False}
            if self.change_source:
                self.source.write_bytes(b'changed during fixture render')
            if self.fail_render:
                raise RuntimeError('fixture interrupted after writes')

        return InferenceSession(loader=load, runner=render, disposer=lambda model: None, **kwargs)

    def run_jobs(self, *, jobs=None, resume=False, budget=100000):
        return run_recoverable(jobs or self.jobs, workspace=self.workspace, max_cache_bytes=budget,
                               assets=self.assets, environment=self.environment,
                               implementation_root=self.root / 'implementation', resume=resume,
                               session_factory=self.factory)

    def test_second_identical_job_restores_bytes_and_resume_never_loads_model(self):
        report = self.run_jobs()
        self.assertEqual([job['mode'] for job in report['jobs']], ['render', 'verified_cache_restore'])
        self.assertEqual(self.loads, 1)
        self.assertEqual(len(self.renders), 1)
        first, second = report['jobs']
        self.assertEqual((Path(first['outputs']['alpha']) / 'alpha_000000.exr').read_bytes(),
                         (Path(second['outputs']['alpha']) / 'alpha_000000.exr').read_bytes())
        self.assertFalse(second['new_session_run'])
        resumed = self.run_jobs(resume=True)
        self.assertEqual([job['mode'] for job in resumed['jobs']], ['verified_existing', 'verified_existing'])
        self.assertEqual(self.loads, 1)
        self.assertEqual(len(self.renders), 1)
        self.assertTrue(Path(resumed['report_path']).is_file())

    def test_corrupt_output_is_restored_to_fresh_sibling_and_user_files_survive(self):
        report = self.run_jobs()
        old = Path(report['jobs'][1]['outputs']['alpha'])
        (old / 'alpha_000000.exr').write_bytes(b'corrupt')
        (old / 'user-footage.png').write_bytes(b'preserve me')
        resumed = self.run_jobs(resume=True)
        self.assertEqual(resumed['jobs'][1]['mode'], 'verified_cache_restore')
        self.assertNotEqual(resumed['jobs'][1]['output_dir'], report['jobs'][1]['output_dir'])
        self.assertEqual((old / 'user-footage.png').read_bytes(), b'preserve me')
        self.assertEqual((old / 'alpha_000000.exr').read_bytes(), b'corrupt')
        self.assertEqual(len(self.renders), 1)

    def test_failed_job_is_unsealed_and_resume_restarts_whole_without_overwrite(self):
        self.fail_render = True
        with self.assertRaisesRegex(RuntimeError, 'fixture interrupted'):
            self.run_jobs(jobs=self.jobs[:1])
        original = self.root / 'first/synthetic_alpha_exr/alpha_000000.exr'
        before = original.read_bytes()
        self.assertEqual(list((self.workspace / 'entries').iterdir()), [])
        self.fail_render = False
        report = self.run_jobs(jobs=self.jobs[:1], resume=True)
        self.assertEqual(report['jobs'][0]['mode'], 'render')
        self.assertNotEqual(report['jobs'][0]['output_dir'], str(self.root / 'first'))
        self.assertEqual(original.read_bytes(), before)
        state = json.loads(Path(report['journal']).read_text())['payload']['jobs'][0]
        self.assertEqual([attempt['status'] for attempt in state], ['failed', 'complete'])

    def test_changed_source_invalidates_and_budget_zero_keeps_journal_only(self):
        report = self.run_jobs(budget=0)
        self.assertEqual([job['mode'] for job in report['jobs']], ['render', 'render'])
        self.assertEqual(len(self.renders), 2)
        self.assertTrue(all(job['cache_note'] for job in report['jobs']))
        self.source.write_bytes(b'new source content')
        after = self.run_jobs(resume=True, budget=0)
        self.assertNotEqual(report['jobs'][0]['key'], after['jobs'][0]['key'])
        self.assertEqual(len(self.renders), 4)
        self.assertTrue((self.root / 'first').is_dir())

    def test_dependencies_changed_during_render_never_publish_cache(self):
        self.change_source = True
        with self.assertRaisesRegex(CacheInvalid, 'dependencies changed during render'):
            self.run_jobs(jobs=self.jobs[:1])
        self.assertEqual(list((self.workspace / 'entries').iterdir()), [])
        self.assertTrue((self.root / 'first/synthetic_alpha_exr/alpha_000000.exr').exists())

    def test_metadata_status_cli_is_read_only_and_needs_no_cuda(self):
        self.run_jobs()
        journals = {p: p.read_bytes() for p in (self.workspace / 'journals').iterdir()}
        result = subprocess.run([sys.executable, '-m', 'corridorkey.recovery', 'status', '--workspace', str(self.workspace)],
                                check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('NOT VERIFIED', result.stdout)
        self.assertEqual(journals, {p: p.read_bytes() for p in journals})

    def test_workspace_must_not_overlap_inputs_or_outputs(self):
        nested = replace(self.jobs[0], output_dir=str(self.workspace / 'render'))
        with self.assertRaisesRegex(ValueError, 'separate'):
            self.run_jobs(jobs=[nested])

    def test_actual_hard_process_exit_recovers_journal_and_preserves_partial_output(self):
        from corridorkey.diagnostics import json_value

        payload = {'jobs': [json_value(self.jobs[0])], 'assets': {name: str(path) for name, path in self.assets.items()},
                   'environment': self.environment, 'implementation_root': str(self.root / 'implementation'),
                   'workspace': str(self.workspace)}
        code = '''import json,os,sys
from pathlib import Path
from corridorkey.recovery import run_recoverable
from corridorkey.session import InferenceSession,JobRequest
p=json.loads(sys.argv[1])
def crash(model,job,config,writer,diagnostics):
    (Path(job.output_dir)/'partial-user-visible.bin').write_bytes(b'preserve partial fixture')
    os._exit(74)
def factory(**kwargs):
    return InferenceSession(loader=lambda:object(),runner=crash,disposer=lambda model:None,**kwargs)
run_recoverable([JobRequest(**v) for v in p['jobs']],workspace=p['workspace'],max_cache_bytes=100000,
    assets=p['assets'],environment=p['environment'],implementation_root=p['implementation_root'],session_factory=factory)
'''
        child = subprocess.run([sys.executable, '-c', code, json.dumps(payload)], check=False,
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 74, child.stderr)
        old = self.root / 'first/partial-user-visible.bin'
        report = self.run_jobs(jobs=self.jobs[:1], resume=True)
        self.assertEqual(old.read_bytes(), b'preserve partial fixture')
        self.assertNotEqual(report['jobs'][0]['output_dir'], str(self.root / 'first'))
        state = json.loads(Path(report['journal']).read_text())['payload']['jobs'][0]
        self.assertEqual([attempt['status'] for attempt in state], ['interrupted', 'complete'])
        self.assertEqual(report['jobs'][0]['mode'], 'render')

    def test_cancellation_before_cache_publish_preserves_outputs_and_never_seals_partial(self):
        from corridorkey.control import CancellationToken, CancelledError

        token = CancellationToken()

        def progress(event):
            if event.stage == 'recovery_publish':
                token.cancel('fixture cancel before publication')

        with self.assertRaisesRegex(CancelledError, 'fixture cancel'):
            run_recoverable(self.jobs[:1], workspace=self.workspace, max_cache_bytes=100000,
                            assets=self.assets, environment=self.environment, implementation_root=self.root / 'implementation',
                            session_factory=self.factory, cancellation=token, progress=progress)
        self.assertEqual(list((self.workspace / 'entries').iterdir()), [])
        self.assertTrue((self.root / 'first/synthetic_alpha_exr/alpha_000000.exr').is_file())
        report = self.run_jobs(jobs=self.jobs[:1], resume=True)
        self.assertEqual(report['jobs'][0]['mode'], 'render')
        state = json.loads(Path(report['journal']).read_text())['payload']['jobs'][0]
        self.assertEqual([attempt['status'] for attempt in state], ['cancelled', 'complete'])

    def test_session_loaded_asset_paths_must_match_fingerprinted_assets(self):
        original = self.factory

        def mismatched_factory(**kwargs):
            session = original(**kwargs)
            session._asset_paths = {name: str(path) for name, path in self.assets.items()}
            session._asset_paths['weights'] = str(self.source)
            return session

        with self.assertRaisesRegex(CacheInvalid, 'session-loaded asset paths'):
            run_recoverable(self.jobs[:1], workspace=self.workspace, max_cache_bytes=100000,
                            assets=self.assets, environment=self.environment, implementation_root=self.root / 'implementation',
                            session_factory=mismatched_factory)
        self.assertEqual(list((self.workspace / 'entries').iterdir()), [])
        self.assertTrue((self.root / 'first/synthetic_alpha_exr/alpha_000000.exr').exists())

    def test_changed_model_assets_between_jobs_require_a_new_session(self):
        def progress(event):
            if event.stage == 'recovery_identity' and event.completed == 1:
                self.assets['weights'].write_bytes(b'new weights while old model stays loaded')

        with self.assertRaisesRegex(CacheInvalid, 'changed during batch'):
            run_recoverable(self.jobs, workspace=self.workspace, max_cache_bytes=100000,
                            assets=self.assets, environment=self.environment, implementation_root=self.root / 'implementation',
                            session_factory=self.factory, progress=progress)
        self.assertEqual(len(self.renders), 1)
        resumed = self.run_jobs(resume=True)
        self.assertEqual(resumed['status'], 'complete')
        self.assertEqual(resumed['jobs'][0]['mode'], 'render')
        self.assertEqual(resumed['jobs'][1]['mode'], 'verified_cache_restore')
        self.assertEqual(len(self.renders), 2)

    def test_cli_rejects_invalid_budget_before_assets_backend_or_workspace_creation(self):
        result = subprocess.run([sys.executable, '-m', 'corridorkey.recovery', 'run', '--batch', str(self.root / 'missing.json'),
                                 '--workspace', str(self.workspace), '--max-cache-bytes', '-1'],
                                check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('max-cache-bytes must be nonnegative', result.stderr)
        self.assertFalse(self.workspace.exists())

    def test_failed_job_reuses_only_sealed_iid_frames_in_fresh_session(self):
        import torch

        from corridorkey.foundation_cache import FoundationWindowCache

        evaluated = []
        interrupted = True
        original_factory = self.factory

        def factory(**kwargs):
            session = original_factory(**kwargs)
            export = session._runner

            def render(model, job, config, writer, diagnostics):
                adapter = diagnostics.durable_foundations
                assert adapter is not None
                cache = FoundationWindowCache(0, persistent=adapter)

                def extract(display):
                    evaluated.extend(display[:, 0, 0, 0].tolist())
                    return {'moge': display[:, :1].float().clone()}

                for frame in range(2):
                    display = torch.full((1, 3, 1, 1), frame / 2, dtype=torch.bfloat16)
                    result = cache.window(display, frame, device='cpu', channels={'moge': 1},
                                          grid_hw=(1, 1), extract=extract)
                    self.assertEqual(float(result['moge'].item()), frame / 2)
                    if interrupted:
                        raise RuntimeError('fixture interrupted after sealed first frame')
                export(model, job, config, writer, diagnostics)

            session._runner = render
            return session

        def invoke(*, resume=False):
            return run_recoverable(
                self.jobs[:1], workspace=self.workspace, max_cache_bytes=100000, assets=self.assets,
                environment=self.environment, implementation_root=self.root / 'implementation',
                session_factory=factory, durable_foundations=True, resume=resume)
        with self.assertRaisesRegex(RuntimeError, 'sealed first frame'):
            invoke()
        failed = json.loads(next((self.workspace / 'reports').glob('*.json')).read_text())
        self.assertEqual(failed['failed_attempt']['diagnostics']['durable_foundations']['stored_frames'], 1)
        self.assertEqual(failed['jobs'], [])
        interrupted = False
        report = invoke(resume=True)
        self.assertEqual(report['jobs'][0]['mode'], 'render')
        counts = report['jobs'][0]['origin_record']['diagnostics']['durable_foundations']
        self.assertEqual((counts['hits'], counts['stored_frames']), (1, 1))
        self.assertEqual(evaluated, [0., 0.5])
        self.assertEqual(self.loads, 2)
        self.assertNotEqual(report['jobs'][0]['output_dir'], self.jobs[0].output_dir)

    def test_cancelled_rvm_prepass_restores_recurrent_state_in_fresh_session(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch

        import torch

        from corridorkey import runtime
        from corridorkey.control import CancellationToken, CancelledError

        original_factory = self.factory
        evaluated = []
        incoming = []

        class RVM(torch.nn.Module):
            embed_dim = 1

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward(self, rgb, state=None):
                incoming.append(None if state is None else state.clone())
                evaluated.append(len(rgb[0]))
                state = torch.zeros(1, 2, 1) if state is None else state
                state = state + rgb.mean() + 1
                return state[:, 1:].reshape(1, 1, 1, 1, 1), state

        def factory(**kwargs):
            session = original_factory(**kwargs)
            export = session._runner

            def render(model, job, config, writer, diagnostics):
                adapter = diagnostics.durable_rvm
                assert adapter is not None and adapter.bound
                foundations = SimpleNamespace(rvm=RVM(), rvm_mean=torch.zeros(1), rvm_std=torch.ones(1))

                def source(*args, **kwargs):
                    for frame in range(2):
                        yield frame, None, torch.full((1, 3, 64, 64), frame / 2)

                with (patch.object(runtime, 'iter_frame_dir_chunks', side_effect=source),
                      patch.object(torch, 'autocast', side_effect=lambda *a, **k: nullcontext()),
                      patch.object(torch.cuda, 'empty_cache')):
                    runtime.precompute_rvm_to_memmap(
                        SimpleNamespace(foundations=foundations), 'fixture',
                        {'num_frames': 2, 'shape_hw': (64, 64)}, Path(job.output_dir) / 'rvm.f16',
                        chunk=1, reader_kind='frame_dir', control=diagnostics.control,
                        durable_rvm=adapter, diagnostics=diagnostics)
                np.testing.assert_array_equal(np.fromfile(Path(job.output_dir) / 'rvm.f16', dtype=np.float16), [1., 2.5])
                export(model, job, config, writer, diagnostics)

            session._runner = render
            return session

        def invoke(*, resume=False, cancellation=None, progress=None):
            return run_recoverable(
                self.jobs[:1], workspace=self.workspace, max_cache_bytes=100000, assets=self.assets,
                environment=self.environment, implementation_root=self.root / 'implementation',
                session_factory=factory, durable_rvm=True, resume=resume, cancellation=cancellation, progress=progress)

        token = CancellationToken()

        def progress(event):
            if event.stage == 'rvm_prepass' and event.completed == 1:
                token.cancel('fixture prepass interruption')

        with self.assertRaisesRegex(CancelledError, 'prepass interruption'):
            invoke(cancellation=token, progress=progress)
        failed = json.loads(next((self.workspace / 'reports').glob('*.json')).read_text())
        self.assertEqual(failed['failed_attempt']['diagnostics']['durable_rvm']['stored_chunks'], 1)
        report = invoke(resume=True)
        stats = report['jobs'][0]['origin_record']['diagnostics']['durable_rvm']
        self.assertEqual((stats['chunk_hits'], stats['frames_reused'], stats['stored_chunks']), (1, 1, 1))
        self.assertEqual(evaluated, [1, 1])
        self.assertEqual(self.loads, 2)
        self.assertIsNone(incoming[0])
        torch.testing.assert_close(incoming[1], torch.ones(1, 2, 1), rtol=0, atol=0)
        self.assertNotEqual(report['jobs'][0]['output_dir'], self.jobs[0].output_dir)

    def test_failed_hann_event_job_replays_verified_prefix_in_new_session(self):
        import torch

        from corridorkey.durable_hann import replay_state_digest

        original_factory = self.factory
        interrupted = True
        computed = []

        def factory(**kwargs):
            session = original_factory(**kwargs)
            export = session._runner

            def render(model, job, config, writer, diagnostics):
                adapter = diagnostics.durable_hann
                assert adapter is not None and adapter.bound
                incoming = replay_state_digest(carry={}, accumulators={}, weights={}, pending_frames={}, previous_segment=None)
                for frame in range(2):
                    inputs = {'rgb': torch.full((1, 1, 3, 64, 64), frame / 2),
                              'hint': torch.zeros(1, 1, 1, 64, 64), 'hint_quality': torch.zeros(1, 1),
                              'moge': torch.zeros(1, 1, 1, 1, 1), 'cradio': None, 'rvm': None}
                    identity = adapter.identity(schedule={'windows': [(0, 1), (1, 2)]},
                                                incoming_replay_sha256=incoming, index=frame, start=frame, stop=frame + 1,
                                                next_start=frame + 1, segment=(0, 2), carry=False, d_16=1, inputs=inputs)
                    event = adapter.get(identity)
                    if event is None:
                        computed.append(frame)
                        event = {'baton': torch.full((1, 1, 1, 4, 4), frame / 2)}
                        adapter.put(identity, event)
                    self.assertEqual(float(event['baton'].mean()), frame / 2)
                    if interrupted:
                        raise RuntimeError('fixture stops after sealed Hann event')
                export(model, job, config, writer, diagnostics)

            session._runner = render
            return session

        def invoke(*, resume=False):
            return run_recoverable(
                self.jobs[:1], workspace=self.workspace, max_cache_bytes=100000, assets=self.assets,
                environment=self.environment, implementation_root=self.root / 'implementation',
                session_factory=factory, durable_hann=True, resume=resume)

        with self.assertRaisesRegex(RuntimeError, 'sealed Hann event'):
            invoke()
        interrupted = False
        report = invoke(resume=True)
        stats = report['jobs'][0]['origin_record']['diagnostics']['durable_hann']
        self.assertEqual((stats['window_hits'], stats['stored_windows']), (1, 1))
        self.assertEqual(computed, [0, 1])
        self.assertEqual(self.loads, 2)
        self.assertNotEqual(report['jobs'][0]['output_dir'], self.jobs[0].output_dir)
