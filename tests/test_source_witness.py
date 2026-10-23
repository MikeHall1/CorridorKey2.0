import hashlib
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from corridorkey.config import InferenceConfig
from corridorkey.control import JobControl
from corridorkey.ofx_cache import seal
from corridorkey.source_witness import SourceWitness, load_witness
from tests import test_quick_runtime


class SourceWitnessTests(unittest.TestCase):
    run_job = test_quick_runtime.QuickRuntimeTests.run_job

    def test_exact_bits_repeated_reads_missing_frames_and_source_change(self):
        meta = {'native_hw': (2, 3), 'num_frames': 2, 'start_frame': 7,
                'native_fps': 24., 'frame_paths': ['one', 'two']}
        witness = SourceWitness(meta, JobControl())
        pixels = np.arange(18, dtype=np.float32).reshape(2, 3, 3) - 4
        witness.observe_path('one', pixels)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'all selected'):
                witness.finish(Path(folder))
            threads = [threading.Thread(target=witness.observe_path, args=('one', pixels)) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
                self.assertFalse(thread.is_alive())
            witness.observe_path('two', pixels[:, ::-1])
            changed = pixels.copy()
            changed[0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'changed between processing reads'):
                witness.observe_path('one', changed)
            record = witness.finish(Path(folder))
            raw = Path(record['path']).read_bytes()
            self.assertEqual(raw[:36], struct.pack('<8sIIIqd', b'CKSRC001', 3, 2, 2, 7, 24.))
            expected = hashlib.sha256(b'CorridorKey/source/linear-rgb-f32/v1\0' + struct.pack('<III', 0, 3, 2) + pixels.astype('<f4').tobytes()).digest()
            self.assertEqual(raw[36:68], expected)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record['sha256'])
            with self.assertRaisesRegex(RuntimeError, 'finalized'):
                witness.observe_path('one', pixels)

    def test_actual_runtime_both_memory_and_full_sparse_paths_seal_v2(self):
        for low_vram in (False, True):
            for sparse in (False, True):
                with self.subTest(low_vram=low_vram, sparse=sparse), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    _, diagnostics = self.run_job(folder, low_vram=low_vram, enabled=sparse, settings_overrides={
                        'frame_dir_linear': True, 'source_witness': True, 'start_frame': 2, 'num_frames': 4})
                    snapshot = json.loads(json.dumps(diagnostics.snapshot()))
                    metadata = snapshot['source_witness']
                    self.assertEqual((metadata['start_frame'], metadata['count']), (2, 4))
                    self.assertNotIn('_source_witness', snapshot['source'])
                    raw = Path(metadata['path']).read_bytes()
                    for index in range(4):
                        # Fixture source pixels are known independently of observer internals.
                        rgb = np.full((64, 64, 3), (80 + (index + 2) * 4) / 255, np.float32)
                        expected = hashlib.sha256(b'CorridorKey/source/linear-rgb-f32/v1\0' + struct.pack('<III', index, 64, 64) + rgb.astype('<f4').tobytes()).digest()
                        self.assertEqual(raw[36 + index * 32:68 + index * 32], expected)
                    record = {'schema_version': 1, 'status': 'complete', 'input': str(root / 'source'),
                              'output_dir': str(root / 'output'), 'requested_settings': {
                                  'source_witness': True, 'frame_dir_linear': True, 'frame_dir_alpha': 'ignore'},
                              'diagnostics': diagnostics.snapshot()}
                    report = root / 'record.json'
                    report.write_text(json.dumps(record))
                    cache = seal(report, root)
                    data = cache.read_bytes()
                    self.assertEqual(data[:8], b'CKOFX002')
                    for index in range(4):
                        self.assertEqual(data[132 + index * 64:164 + index * 64], raw[36 + index * 32:68 + index * 32])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), metadata['sha256'])
                    source_before = {path.name: path.read_bytes() for path in (root / 'source').iterdir()}
                    Path(metadata['path']).write_bytes(raw[:-1])
                    with self.assertRaisesRegex(ValueError, 'incomplete, changed'):
                        seal(report, root)
                    self.assertEqual(source_before, {path.name: path.read_bytes() for path in (root / 'source').iterdir()})

    def test_witness_is_explicit_and_rejects_unsupported_or_missing_provenance(self):
        self.assertFalse(InferenceConfig().source_witness)
        for settings in ({'source_witness': True}, {'source_witness': True, 'frame_dir_linear': True},
                         {'source_witness': 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                config = InferenceConfig()
                for name, value in settings.items():
                    setattr(config, name, value)
                config.validate()
        record = {'requested_settings': {'source_witness': True}, 'diagnostics': {}}
        with self.assertRaisesRegex(ValueError, 'completed producer'):
            load_witness(record, width=3, height=2, count=2, start=7, fps=24.)
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'actual channel layout'):
            self.run_job(folder, source_rgba=True, settings_overrides={
                'frame_dir_linear': True, 'source_witness': True})

    def test_sealing_rejects_nonlegacy_profile_even_with_valid_sidecar(self):
        meta = {'native_hw': (2, 3), 'num_frames': 1, 'start_frame': 0,
                'native_fps': 24., 'frame_paths': ['frame']}
        witness = SourceWitness(meta, JobControl())
        witness.observe_path('frame', np.ones((2, 3, 3), np.float32))
        with tempfile.TemporaryDirectory() as folder:
            metadata = witness.finish(Path(folder))
            settings = {'source_witness': True, 'frame_dir_linear': True, 'frame_dir_alpha': 'ignore'}
            record = {'requested_settings': settings, 'output_dir': folder,
                      'diagnostics': {'source_witness': metadata}}
            args = {'width': 3, 'height': 2, 'count': 1, 'start': 0, 'fps': 24.}
            self.assertIsNotNone(load_witness(record, **args))
            for profile in ('linear_bt709_d65', 'linear_bt2020_d65'):
                settings['source_colour_profile'] = profile
                with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, 'completed producer'):
                    load_witness(record, **args)
            settings['source_colour_profile'] = 'legacy'
            self.assertIsNotNone(load_witness(record, **args))
