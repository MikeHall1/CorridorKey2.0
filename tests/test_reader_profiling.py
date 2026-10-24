"""Optional host reader measurements preserve actual source pixels and lifecycle."""
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import cv2
import numpy as np
import torch

from corridorkey import video_io
from corridorkey.diagnostics import RunDiagnostics


class ReaderProfilingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        for frame in range(4):
            self.assertTrue(cv2.imwrite(str(self.source / f'{frame:03}.png'), np.full((64, 64, 3), 40 + frame, np.uint8)))

    def meta(self):
        return video_io.probe_frame_dir_meta(str(self.source), num_frames=-1, shape_quantum=64, fps=24,
                                             exr_linear=False, model_scale=1.)

    def test_actual_reader_reports_native_decode_and_sibling_cpu_preparation_cache_misses_only(self):
        meta = self.meta()
        diag = RunDiagnostics(profile_stages=True)
        diag.synchronize = Mock(side_effect=AssertionError('reader threads must never synchronize GPU'))
        profiled = {**meta, '_host_observer': diag}
        reference = video_io.SequentialFrameDirWindowReader(str(self.source), meta, include_review=False)
        reader = video_io.SequentialFrameDirWindowReader(str(self.source), profiled, include_review=False)
        self.addCleanup(reference.close); self.addCleanup(reader.close)
        for start, stop in ((0, 2), (1, 3), (1, 3)):
            actual, expected = reader.window(start, stop), reference.window(start, stop)
            for a, b in zip(actual, expected, strict=True):
                if a is None:
                    self.assertIsNone(b)
                else:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        rows = {r['name']: r for r in diag.stages}
        self.assertEqual(rows['source_decode']['count'], 3)
        self.assertEqual(rows['source_normalize']['count'], 3)
        self.assertEqual(rows['source_colour']['count'], 3)
        self.assertEqual(rows['source_model_prepare']['count'], 3)
        self.assertTrue(all(r['method'] == 'host_wall' for r in rows.values()))
        diag.synchronize.assert_not_called()

    def test_snapshot_cannot_publish_partial_concurrent_aggregate_update(self):
        diag = RunDiagnostics(profile_stages=True)
        diag._record_stage('compute', 1., 'complete', 'host_wall')
        entered, release, copied = threading.Event(), threading.Event(), threading.Event()
        class PausedRow(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if key == 'seconds':
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError('test owner failed to release update')
        row = PausedRow(diag.stages[0])
        diag.stages[0] = row
        assert diag._stage_groups is not None
        diag._stage_groups[('compute', 'host_wall', 'complete')] = row
        snapshots: list[Any] = []
        def copy():
            snapshots.append(diag.snapshot())
            copied.set()
        writer = threading.Thread(target=lambda: diag._record_stage('compute', 1., 'complete', 'host_wall'))
        reader = threading.Thread(target=copy)
        writer.start()
        self.assertTrue(entered.wait(3))
        try:
            reader.start()
            self.assertFalse(copied.wait(.05), 'snapshot observed partial seconds/count update')
        finally:
            release.set(); writer.join(3); reader.join(3)
        self.assertFalse(writer.is_alive()); self.assertFalse(reader.is_alive())
        self.assertEqual(snapshots[0]['stages'][0]['count'], 2)
        self.assertEqual(snapshots[0]['stage_observations']['observations'], 2)

    def test_storage_colour_alpha_spatial_and_captured_bytes_are_exact_with_observer(self):
        cases = [('png8', np.full((64, 64, 3), 123, np.uint8), False, 'ignore', 'legacy'),
                 ('png16', np.full((64, 64, 3), 35000, np.uint16), False, 'ignore', 'legacy'),
                 ('exr', np.tile(np.array([-.25, .5, 2.], np.float32), (64, 64, 1)), True, 'ignore', 'legacy'),
                 ('bt2020', np.tile(np.array([-.25, .5, 2.], np.float32), (64, 64, 1)), True, 'ignore', 'linear_bt2020_d65'),
                 ('rgba', np.tile(np.array([-.125, .25, 1., .5], np.float32), (64, 64, 1)), True, 'premultiplied', 'legacy')]
        for label, image, linear, alpha, colour in cases:
            source = self.root / label
            source.mkdir()
            path = source / ('000.exr' if linear else '000.png')
            self.assertTrue(cv2.imwrite(str(path), image))
            for scale in (1., .5):
                with self.subTest(storage=label, scale=scale):
                    meta = video_io.probe_frame_dir_meta(str(source), num_frames=-1, shape_quantum=64,
                        fps=24, exr_linear=linear, model_scale=scale, alpha_mode=alpha, source_colour_profile=colour)
                    diag = RunDiagnostics(profile_stages=True)
                    for need_model in (False, True):
                        expected = video_io.preprocess_frame_dir_path(str(path), meta, need_model=need_model)
                        actual = video_io.preprocess_frame_dir_path(str(path), {**meta, '_host_observer': diag}, need_model=need_model)
                        for a, b in zip(actual, expected, strict=True):
                            if a is None:
                                self.assertIsNone(b)
                            else:
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
                    expected = video_io.read_image_rgb(path, alpha_mode=alpha)
                    actual = video_io.read_image_rgb(path, alpha_mode=alpha, encoded_bytes=path.read_bytes(), host_observer=diag)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(next(r for r in diag.stages if r['name'] == 'source_decode')['count'], 3)

    def test_actual_video_offset_sparse_discard_and_chunk_counts_exclude_hints(self):
        path = self.root / 'source.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*'MJPG'), 24, (64, 64))
        self.assertTrue(writer.isOpened())
        try:
            for frame in range(7):
                writer.write(np.full((64, 64, 3), frame * 25, np.uint8))
        finally:
            writer.release()
        meta = video_io.probe_video_meta(str(path), num_frames=4, start_frame=2, shape_quantum=64,
            native_resolution=True, start_mode='begin', seed=0, model_scale=1.,
            linear_brightness=0., linear_contrast=1., linear_contrast_pivot=.18)
        diag = RunDiagnostics(profile_stages=True)
        profiled = {**meta, '_host_observer': diag}
        reference = video_io.SequentialVideoWindowReader(str(path), meta)
        reader = video_io.SequentialVideoWindowReader(str(path), profiled)
        try:
            for left, right in zip(reader.window(2, 4), reference.window(2, 4), strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            # Four reads, but first two discarded source frames have no CPU preparation.
            self.assertEqual({r['name']: r['count'] for r in diag.stages}, {'source_decode': 4, 'source_preprocess': 2})
        finally:
            reader.close(); reference.close()
        before = diag.snapshot()
        hints = video_io.SequentialHintVideoWindowReader(str(path), profiled)
        try:
            hints.window(0, 2)
        finally:
            hints.close()
        self.assertEqual(diag.snapshot(), before)
        reference_chunks = list(video_io.iter_video_frame_chunks(str(path), meta, chunk_size=2, need_model=False))
        chunks = list(video_io.iter_video_frame_chunks(str(path), profiled, chunk_size=2, need_model=False))
        for a, b in zip(chunks, reference_chunks, strict=True):
            self.assertEqual(a[:2], b[:2])
            torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        self.assertEqual(next(r for r in diag.stages if r['name'] == 'source_decode')['count'], 8)

    def test_native_failure_and_default_off_do_not_synchronize_or_create_stage_contexts(self):
        diag = RunDiagnostics(profile_stages=True)
        with self.assertRaises(FileNotFoundError):
            video_io.read_image_rgb(self.root / 'missing.png', host_observer=diag)
        self.assertEqual([(r['name'], r['status']) for r in diag.stages], [('source_decode', 'complete')])
        from unittest.mock import patch
        with patch.object(RunDiagnostics, 'stage', side_effect=AssertionError('off path must not allocate a stage context')):
            meta = self.meta()
            reader = video_io.SequentialFrameDirWindowReader(str(self.source), meta)
            try:
                reader.window(0, 2)
            finally:
                reader.close()

    def test_full_shared_runtime_profile_parity_prepasses_quick_and_metadata_retirement(self):
        import weakref
        from unittest.mock import patch

        from corridorkey import runtime
        from tests import test_quick_runtime
        helper = test_quick_runtime.QuickRuntimeTests()
        original_reader = runtime.SequentialFrameDirWindowReader
        observers = []
        def constructor(source, meta, *, include_review=True):
            if '_host_observer' in meta:
                observers.append(weakref.ref(meta['_host_observer']))
            return original_reader(source, meta, include_review=include_review)
        for index, (low, quick, scale) in enumerate(((False, False, 1.), (True, False, 1.),
                                                     (False, True, 1.), (True, True, .5))):
            with self.subTest(low=low, quick=quick, scale=scale):
                roots = [self.root / f'run-{index}-{arm}' for arm in ('off', 'on')]
                for root in roots:
                    root.mkdir()
                settings = {'start_frame': 2, 'num_frames': 5, 'model_scale': scale}
                _, reference = helper.run_job(roots[0], low_vram=low, enabled=quick, fallback='neural', flashing=quick,
                    use_rvm=True, settings_overrides=settings)
                diag = RunDiagnostics(profile_stages=True)
                model: Any
                with patch.object(runtime, 'SequentialFrameDirWindowReader', constructor):
                    model, actual = helper.run_job(roots[1], low_vram=low, enabled=quick, fallback='neural', flashing=quick,
                        use_rvm=True, settings_overrides=settings, diagnostics=diag)
                self.assertIs(actual, diag)
                self.assertGreater(len(model.foundations.rvm.calls), 0)
                self.assertGreater(next(r for r in diag.stages if r['name'] == 'source_decode')['count'], 0)
                self.assertTrue(all(r['method'] == 'host_wall' for r in diag.stages))
                assert isinstance(diag.source, dict)
                self.assertNotIn('_host_observer', diag.source)
                for kind, directory in diag.outputs.items():
                    for a, b in zip(sorted(Path(directory).glob('*.exr')),
                                    sorted(Path(reference.outputs[kind]).glob('*.exr')), strict=True):
                        np.testing.assert_array_equal(cv2.imread(str(a), -1), cv2.imread(str(b), -1))
                self.assertTrue(all(ref() is diag for ref in observers))
                retired = weakref.ref(diag)
                del diag, actual
                self.assertIsNone(retired(), 'owned retired readers/meta must not retain diagnostics after runtime returns')
                observers.clear()

    def test_concurrent_scope_population_and_snapshots_preserve_complete_consistent_counts(self):
        from concurrent.futures import ThreadPoolExecutor
        diag = RunDiagnostics(profile_stages=True)
        diag.synchronize = Mock(side_effect=AssertionError('host readers cannot sync GPU'))
        def work():
            for _ in range(2000):
                with diag.stage('source_decode', gpu=False):
                    pass
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(work) for _ in range(4)]
            for _ in range(100):
                snapshot: Any = diag.snapshot()
                self.assertEqual(sum(r['count'] for r in snapshot['stages']), snapshot['stage_observations']['observations'])
            for future in futures:
                future.result()
        snapshot: Any = diag.snapshot()
        self.assertEqual(snapshot['stage_observations']['observations'], 8000)
        self.assertEqual(snapshot['stages'][0]['count'], 8000)
        diag.synchronize.assert_not_called()

    def test_source_witness_and_native_work_run_outside_collector_lock(self):
        from unittest.mock import patch

        from corridorkey.control import JobControl
        from corridorkey.source_witness import SourceWitness
        meta = self.meta()
        meta['exr_linear'] = True
        diag = RunDiagnostics(profile_stages=True)
        witness = SourceWitness(meta, JobControl())
        meta.update(_host_observer=diag, _source_witness=witness)
        original = cv2.imread
        def decode(*args, **kwargs):
            assert diag._stage_lock is not None
            self.assertTrue(diag._stage_lock.acquire(blocking=False), 'native call must run outside collector lock')
            diag._stage_lock.release()
            return original(*args, **kwargs)
        observed = witness.observe_path
        def observe(*args, **kwargs):
            self.assertEqual({r['name'] for r in diag.stages}, {'source_decode', 'source_normalize', 'source_colour'})
            assert diag._stage_lock is not None
            self.assertTrue(diag._stage_lock.acquire(blocking=False), 'witness hashing must run outside collector lock')
            diag._stage_lock.release()
            return observed(*args, **kwargs)
        with patch.object(cv2, 'imread', decode), patch.object(witness, 'observe_path', observe):
            video_io.preprocess_frame_dir_path(meta['frame_paths'][0], meta, need_model=True)
        self.assertEqual(len(witness.digests), 1)
        self.assertEqual(len(diag.stages), 4)

    def test_cancelled_active_prepass_retires_reader_threads_and_freezes_observations(self):
        from corridorkey.control import CancellationToken, CancelledError, JobControl
        from tests import test_quick_runtime
        helper = test_quick_runtime.QuickRuntimeTests()
        original_threads = {t.ident for t in threading.enumerate()}
        for profile in (False, True):
            root = self.root / f'cancel-{profile}'
            root.mkdir()
            token = CancellationToken()
            def progress(event, token=token):
                if event.stage == 'rvm_prepass' and event.completed > 0:
                    token.cancel('controlled prepass cancellation')
            diag = RunDiagnostics(profile_stages=profile)
            diag.control = JobControl(token, progress)
            with self.assertRaisesRegex(CancelledError, 'controlled prepass cancellation'):
                helper.run_job(root, use_rvm=True, diagnostics=diag)
            self.assertEqual(list((root / 'temp').iterdir()), [])
            if profile:
                self.assertGreater(next(r for r in diag.stages if r['name'] == 'source_decode')['count'], 0)
            self.assertEqual(diag.snapshot(), diag.snapshot())
        self.assertFalse([t.name for t in threading.enumerate() if t.ident not in original_threads
                          and t.name.startswith('ThreadPoolExecutor')])

    def test_png_integer_decode_payload_retires_before_float_rgb_conversion(self):
        import weakref
        from unittest.mock import patch
        original_read, original_colour = cv2.imread, cv2.cvtColor
        for dtype in (np.uint8, np.uint16):
            for channels in (3, 4):
                for profile in (False, True):
                    with self.subTest(dtype=dtype, channels=channels, profile=profile):
                        image = np.full((8, 12, channels), 40 if dtype == np.uint8 else 10000, dtype)
                        path = self.root / f'lifetime-{dtype.__name__}-{channels}.png'
                        self.assertTrue(cv2.imwrite(str(path), image))
                        decoded_refs, alive_at_colour = [], []
                        def read(*args, decoded_refs=decoded_refs, **kwargs):
                            decoded = original_read(*args, **kwargs)
                            decoded_refs.append(weakref.ref(decoded))
                            return decoded
                        def colour(*args, decoded_refs=decoded_refs, alive_at_colour=alive_at_colour, **kwargs):
                            if args[1] == cv2.COLOR_BGR2RGB:
                                alive_at_colour.append(decoded_refs[-1]() is not None)
                            return original_colour(*args, **kwargs)
                        observer = RunDiagnostics(profile_stages=True) if profile else None
                        with patch.object(cv2, 'imread', read), patch.object(cv2, 'cvtColor', colour):
                            actual = video_io.read_image_rgb(path, host_observer=observer)
                        expected = (image.astype(np.float32) / np.iinfo(dtype).max)[..., :3][..., ::-1]
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(alive_at_colour, [False], 'raw integer array outlives original inline normalization')
                        self.assertIsNone(decoded_refs[0]())

    def test_source_float_views_match_original_padding_and_resize_lifetime(self):
        import weakref
        from unittest.mock import patch
        source = self.root / 'padded-lifetime'
        source.mkdir()
        path = source / '000.png'
        self.assertTrue(cv2.imwrite(str(path), np.full((32, 48, 3), 100, np.uint8)))
        original_views, original_encode = video_io._read_frame_dir_views, video_io.cu.linear_to_asinh
        for scale in (1., .5):
            meta = video_io.probe_frame_dir_meta(str(source), num_frames=-1, shape_quantum=64, fps=24,
                                                exr_linear=False, model_scale=scale)
            for profile in (False, True):
                with self.subTest(scale=scale, profile=profile):
                    refs, alive = [], []
                    def views(*args, refs=refs, **kwargs):
                        values = original_views(*args, **kwargs)
                        refs.extend(weakref.ref(value) for value in values)
                        return values
                    def encode(*args, refs=refs, alive=alive, **kwargs):
                        alive.append(tuple(ref() is not None for ref in refs))
                        return original_encode(*args, **kwargs)
                    selected = {**meta, '_host_observer': RunDiagnostics(profile_stages=True)} if profile else meta
                    with patch.object(video_io, '_read_frame_dir_views', views), patch.object(video_io.cu, 'linear_to_asinh', encode):
                        result = video_io.preprocess_frame_dir_path(str(path), selected, need_model=True)
                    self.assertEqual(tuple(result[0].shape), (3, 64, 64))
                    # Original spatial path retains unused display until return,
                    # but rebinds/resizes linear before encoding. Padding releases both.
                    self.assertEqual(alive, [(scale < 1., False)])
