"""Shared native output/correction/sparse paths with replayed Stage1 events."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import test_correction_runtime
import test_quick_runtime

HAS_STACK = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2', 'safetensors'))


@unittest.skipUnless(HAS_STACK, 'requires CPU output adapters; no pretrained neural inference')
class HannRuntimeTests(unittest.TestCase):
    run_job = test_quick_runtime.QuickRuntimeTests.run_job
    project = test_correction_runtime.CorrectionRuntimeTests.project

    def test_full_sparse_causal_cuts_corrections_and_fallback_replay_in_both_memory_paths(self):
        import cv2
        import numpy as np

        from corridorkey.diagnostics import RunDiagnostics
        from corridorkey.durable_hann import DurableHannWindows
        from corridorkey.recovery_identity import ReplayIdentity, canonical
        from corridorkey.recovery_store import BundleStore

        parent = ReplayIdentity(canonical({'schema_version': 1, 'kind': 'whole_job_exr_replay',
                                          'assets': {}, 'environment': {'runtime': {'synthetic': True}}}))
        for low_vram in (False, True):
            for mode in ('full', 'offline', 'hold', 'causal', 'corrections', 'full_corrections', 'fallback'):
                with self.subTest(low_vram=low_vram, mode=mode), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    with BundleStore(root / 'cache', 3000000) as store:
                        reference = {}
                        for repeat in range(2):
                            adapter = DurableHannWindows(store, parent, validate_context=lambda: None)
                            adapter.bind(None)
                            diagnostics = RunDiagnostics()
                            diagnostics.durable_hann = adapter
                            model, diagnostics = self.run_job(
                                root, low_vram=low_vram, enabled=not mode.startswith('full'), diagnostics=diagnostics,
                                approach=mode if mode in ('hold', 'causal') else 'offline',
                                cuts=() if mode.startswith('full') else (4,), context_dependent=True,
                                corrections=self.project if 'corrections' in mode else None,
                                fallback='neural' if mode == 'fallback' else 'stop', flashing=mode == 'fallback')
                            outputs = {f'{name}/{file.name}': cv2.imread(str(file), -1)
                                       for name, directory in diagnostics.outputs.items()
                                       for file in sorted(Path(directory).glob('*.exr'))}
                            self.assertEqual(len(outputs), 36)
                            self.assertTrue(all(value is not None for value in outputs.values()))
                            seen = vars(model.stage1)['seen']
                            assert isinstance(seen, list)
                            assert isinstance(diagnostics.execution, dict)
                            if repeat == 0:
                                reference = outputs
                                self.assertGreater(len(seen), 0)
                            else:
                                self.assertEqual(len(seen), 0)
                                self.assertGreater(adapter.hits, 0)
                                self.assertEqual(adapter.misses, 0)
                                self.assertEqual(diagnostics.execution['stage1_frame_evaluations'], 0)
                                for name, actual in outputs.items():
                                    np.testing.assert_array_equal(actual, reference[name])
