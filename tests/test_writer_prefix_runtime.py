"""Real CPU EXR recovery keeps complete inference and skips verified writes only."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2', 'safetensors'))


@unittest.skipUnless(STACK, 'Requires explicit CPU numeric fixture, not pretrained inference')
class WriterPrefixRuntimeTests(unittest.TestCase):
    def setUp(self):
        from tests.test_durable_stage2 import DurableStage2Tests
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.helper = DurableStage2Tests()
        self.helper.root = self.root

    def adapter(self, store, *, resume=False):
        from corridorkey.durable_writer_prefix import DurableWriterPrefix
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
            'assets': {}, 'environment': {'runtime': {'synthetic': True}}}))
        adapter = DurableWriterPrefix(store, parent, validate_context=lambda: None, resume=resume)
        adapter.bind(None)
        return adapter

    def run_fixture(self, name, adapter=None, **kwargs):
        from corridorkey.diagnostics import RunDiagnostics
        diagnostics = RunDiagnostics()
        diagnostics.durable_writer_prefix = adapter
        return self.helper.run_runtime(self.root / name, None, diagnostics=diagnostics, **kwargs)

    def output_bytes(self, diagnostics):
        return {role: {p.name: p.read_bytes() for p in Path(folder).glob('*.exr')}
                for role, folder in diagnostics.outputs.items()}

    def test_mid_batch_seal_interruption_resume_and_all_restored_replay(self):
        from corridorkey.control import CancelledError
        from corridorkey.recovery_store import BundleStore
        for low, quick in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(low=low, quick=quick), BundleStore(
                    self.root / f'cache{low}{quick}', 8_000_000, structural_admission=True) as store:
                first = self.adapter(store)
                checkpoint = first.checkpoint

                def interrupt_after_first_frame(frame_ids, *, completed, checkpoint=checkpoint):
                    self.assertTrue(checkpoint(frame_ids[:1], completed=completed))
                    raise CancelledError('fixture interruption between sealed frame groups')

                with patch.object(first, 'checkpoint', side_effect=interrupt_after_first_frame), self.assertRaises(CancelledError):
                    self.run_fixture(f'interrupted{low}{quick}', first, low=low, quick=quick, corrections=True)
                old = self.root / f'interrupted{low}{quick}' / 'output'
                old_files = {p.relative_to(old): p.read_bytes() for p in old.rglob('*') if p.is_file()}
                self.assertEqual(first.snapshot()['stored_frames'], 1)
                resumed = self.adapter(store, resume=True)
                model, actual = self.run_fixture(f'resumed{low}{quick}', resumed, low=low, quick=quick, corrections=True)
                oracle_model, oracle = self.run_fixture(f'oracle{low}{quick}', low=low, quick=quick, corrections=True)
                self.assertEqual(self.output_bytes(actual), self.output_bytes(oracle))
                self.assertEqual(model.stage2.calls, oracle_model.stage2.calls)
                self.assertEqual(actual.frames_submitted, 9)
                assert actual.writer_prefix_runtime is not None
                self.assertEqual(actual.writer_prefix_runtime['native_new_written_frames'], 8)
                self.assertEqual(actual.writer_prefix_runtime['newly_submitted_frames'], 8)
                self.assertEqual(actual.writer_prefix_runtime['native_new_written_files'], 16)
                self.assertEqual(actual.writer_prefix_runtime['native_completion_status'], 'complete')
                self.assertEqual({p.relative_to(old): p.read_bytes() for p in old.rglob('*') if p.is_file()}, old_files)
                all_restored = self.adapter(store, resume=True)
                model, complete = self.run_fixture(f'all{low}{quick}', all_restored, low=low, quick=quick, corrections=True)
                self.assertEqual(all_restored.snapshot()['restored_prefix_frames'], 9)
                assert complete.writer_prefix_runtime is not None
                self.assertEqual(complete.writer_prefix_runtime['native_new_written_files'], 0)
                self.assertEqual(complete.frames_submitted, 9)
                self.assertEqual(model.stage2.calls, oracle_model.stage2.calls)
                self.assertEqual(self.output_bytes(complete), self.output_bytes(oracle))

    def test_zero_budget_continues_and_default_never_calls_barrier(self):
        from corridorkey.recovery_store import BundleStore
        with BundleStore(self.root / 'zero', 0, structural_admission=True) as store:
            adapter = self.adapter(store)
            _, result = self.run_fixture('zero-run', adapter)
            self.assertTrue(adapter.snapshot()['checkpoint_disabled'])
            assert result.writer_prefix_runtime is not None
            self.assertEqual(result.writer_prefix_runtime['native_new_written_frames'], 9)
            self.assertEqual(result.frames_submitted, 9)
        with patch('corridorkey.video_io.ExrSequenceWriter.flush_barrier', side_effect=AssertionError('default must not flush')):
            _, oracle = self.run_fixture('default')
        self.assertEqual(self.output_bytes(result), self.output_bytes(oracle))

    def test_copied_output_mutation_during_render_refuses_completion(self):
        from corridorkey.recovery_store import BundleStore, CacheInvalid
        from corridorkey.video_io import ExrSequenceWriter
        with BundleStore(self.root / 'mutation', 8_000_000, structural_admission=True) as store:
            self.run_fixture('seed', self.adapter(store))
            resumed = self.adapter(store, resume=True)
            release = ExrSequenceWriter.release
            changed = False

            def mutate_after_release(writer):
                nonlocal changed
                release(writer)
                if not changed:
                    path = next(Path(writer.path).glob('*.exr'))
                    data = path.read_bytes()
                    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
                    changed = True

            with patch.object(ExrSequenceWriter, 'release', mutate_after_release), self.assertRaises(CacheInvalid):
                self.run_fixture('mutated', resumed)
            self.assertTrue(changed)

    def test_non_exr_refused_before_model_or_output_creation(self):
        from corridorkey.config import InferenceConfig
        from corridorkey.recovery_store import BundleStore
        from corridorkey.session import InferenceSession, JobRequest
        source = self.root / 'source.mov'
        source.write_bytes(b'preflight-only fixture')
        destination = self.root / 'unsupported'
        job = JobRequest.create(source, destination, InferenceConfig(output_format='video'))
        loader, runner = Mock(), Mock()
        with BundleStore(self.root / 'preflight-cache', 0) as store, InferenceSession(
                loader=loader, runner=runner, disposer=lambda model: None) as session:
            with self.assertRaisesRegex(ValueError, 'requires EXR'):
                session.run(job, durable_writer_prefix=self.adapter(store))
            loader.assert_not_called()
            runner.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertEqual(source.read_bytes(), b'preflight-only fixture')

    def test_prepare_failure_preserves_error_and_cleans_temporary_owner(self):
        from corridorkey.recovery_store import BundleStore, CacheInvalid
        from corridorkey.video_io import ExrSequenceWriter
        with BundleStore(self.root / 'bad-prepare-cache', 0) as store:
            adapter = self.adapter(store)
            error = CacheInvalid('fixture preparation refused')
            with patch.object(adapter, 'prepare', side_effect=error), \
                 patch.object(ExrSequenceWriter, 'release') as release, \
                 self.assertRaises(CacheInvalid) as caught:
                self.run_fixture('bad-prepare', adapter)
            self.assertIs(caught.exception, error)
            release.assert_not_called()
            self.assertEqual(list((self.root / 'bad-prepare' / 'temp').iterdir()), [])
