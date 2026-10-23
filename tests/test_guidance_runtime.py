"""Accepted schema4 clean-plate trimaps in the shared CPU toy runtime.

Actual native PNG/EXR and full/Quick memory paths; deterministic substitutes
prove final-only integration, not pretrained inference or footage quality.
"""
import tempfile
import unittest
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import patch

import cv2
import numpy as np
import torch

from corridorkey.config import InferenceConfig
from corridorkey.correction_editor import CorrectionEditor
from corridorkey.corrections import load_corrections
from corridorkey.session import JobRequest
from tests import test_quick_runtime


class _StageObservations(Protocol):
    seen_hints: list[torch.Tensor]
    seen: list[tuple[int, float, torch.Tensor]]


class GuidanceRuntimeTests(unittest.TestCase):
    run_job = test_quick_runtime.QuickRuntimeTests.run_job

    def project(self, source, root, *, attach=True):
        first = source / 'frame0.png'
        reference = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
        image = reference.copy()
        image[20:44, 20:44] = [20, 20, 220]
        self.assertTrue(cv2.imwrite(str(first), image))
        plate = root / 'plate.png'
        sample = root / 'samples.png'
        self.assertTrue(cv2.imwrite(str(plate), reference))
        mask = np.full((64, 64), 255, np.uint8)
        mask[16:48, 16:48] = 0
        self.assertTrue(cv2.imwrite(str(sample), mask))
        self.originals = {path: path.read_bytes() for path in [*source.iterdir(), plate, sample]}
        if not attach:
            return None
        editor = CorrectionEditor(JobRequest.create(source, root / 'editor-output', InferenceConfig(output_format='exr')), root)
        editor.prepare_plate(frame=0, reference_path=plate, reference_profile='srgb', samples_path=sample)
        summary = editor.plate_summary()
        assert summary is not None
        self.assertTrue(summary['usable'])
        editor.plate_view('guidance')
        editor.accept_plate()
        path = editor.save()
        data = load_corrections(path, source=source).to_dict()
        self.assertEqual(data['schema_version'], 4)
        self.assertEqual(data['entries'][0]['kind'], 'trimap')
        self.assertEqual(data['guidance'][0]['kind'], 'clean_plate')
        self.assertEqual(data['guidance'][0]['observation']['diagnostics']['actual_inference'], 'NOT PERFORMED')
        return path

    def check_originals(self):
        self.assertEqual({path: path.read_bytes() for path in self.originals}, self.originals)

    def test_accepted_trimaps_final_alpha_only_full_quick_both_memory_paths(self):
        for quick in (False, True):
            for low_vram in (False, True):
                with self.subTest(quick=quick, low_vram=low_vram), tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as accepted_dir:
                    baseline, baseline_diagnostics = self.run_job(baseline_dir, enabled=quick, low_vram=low_vram,
                                                                 corrections=lambda source, root: self.project(source, root, attach=False))
                    self.check_originals()
                    inputs = []

                    def observe(module, args, inputs=inputs):
                        if type(module).__name__ in ('Stage1', 'Stage2'):
                            inputs.append((type(module).__name__, args[0].shape[2], args[1].shape[2] if type(module).__name__ == 'Stage1' else None))

                    hook = torch.nn.modules.module.register_module_forward_pre_hook(observe)
                    try:
                        model, diagnostics = self.run_job(accepted_dir, enabled=quick, low_vram=low_vram, corrections=self.project)
                    finally:
                        hook.remove()
                    self.check_originals()
                    self.assertTrue(inputs)
                    self.assertTrue(all(rgb_channels == 3 for _, rgb_channels, _ in inputs))
                    self.assertTrue(all(hint_channels == 1 for name, _, hint_channels in inputs if name == 'Stage1'))
                    accepted_stage = cast(_StageObservations, model.stage1)
                    baseline_stage = cast(_StageObservations, baseline.stage1)
                    self.assertEqual(len(accepted_stage.seen_hints), len(baseline_stage.seen_hints))
                    for accepted, original in zip(accepted_stage.seen_hints, baseline_stage.seen_hints):
                        torch.testing.assert_close(accepted, original, rtol=0, atol=0)
                    for accepted, original in zip(accepted_stage.seen, baseline_stage.seen):
                        self.assertEqual(accepted[:2], original[:2])
                        torch.testing.assert_close(accepted[2], original[2], rtol=0, atol=0)
                    assert diagnostics.corrections is not None
                    self.assertEqual(diagnostics.corrections['hint_frame_evaluations_conditioned'], 0)
                    self.assertEqual(diagnostics.corrections['aux_frame_evaluations_refined'], 0)
                    self.assertEqual(diagnostics.corrections['output_frames_refined'], 1)
                    assert diagnostics.execution is not None and baseline_diagnostics.execution is not None
                    for field in ('stage1_frame_evaluations', 'stage2_frame_evaluations'):
                        self.assertEqual(diagnostics.execution[field], baseline_diagnostics.execution[field])
                    files = sorted(Path(diagnostics.outputs['alpha']).glob('*.exr'))
                    self.assertEqual(len(files), 9)
                    alpha = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
                    np.testing.assert_array_equal(alpha[4:14, 4:14], 0.)
                    np.testing.assert_array_equal(alpha[24:40, 24:40], 1.)
                    for frame in range(1, 9):
                        np.testing.assert_allclose(cv2.imread(str(files[frame]), cv2.IMREAD_UNCHANGED), .4, atol=1e-6)

    def test_source_decode_drift_refuses_before_model_forward_or_writer_creation(self):
        for changed in ({'frame_dir_linear': True}, {'frame_dir_linear': True, 'source_colour_profile': 'linear_bt2020_d65'}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                calls = []
                hook = torch.nn.modules.module.register_module_forward_pre_hook(lambda module, args, calls=calls: calls.append(type(module).__name__))
                self.addCleanup(hook.remove)
                with patch('corridorkey.runtime.ExrSequenceWriter') as writer, self.assertRaisesRegex(ValueError, 'guidance source decode changed'):
                    self.run_job(folder, enabled=False, corrections=self.project, settings_overrides=changed)
                hook.remove()
                writer.assert_not_called()
                self.assertEqual(calls, [])
                self.check_originals()
