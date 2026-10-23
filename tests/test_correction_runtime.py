import importlib.util
from pathlib import Path
import tempfile
import unittest

import test_quick_runtime

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


@unittest.skipUnless(HAS_STACK, "requires optional CPU numeric stack")
class CorrectionRuntimeTests(unittest.TestCase):
    run_job = test_quick_runtime.QuickRuntimeTests.run_job

    def project(self, source, root):
        import cv2
        import numpy as np
        from corridorkey.corrections import CorrectionDocument, save_corrections
        from corridorkey.identities import source_identity, file_sha256
        entries = []
        left, right = np.zeros((64, 64), np.uint8), np.zeros((64, 64), np.uint8)
        left[:, :32], right[:, 32:] = 255, 255
        for frame, kind, mask, strength in ((1, "keep", left, 1), (2, "remove", right, 1),
                                              (3, "matte", np.full((64, 64), 204, np.uint8), 0.5)):
            path = root / f"{kind}.png"
            cv2.imwrite(str(path), mask)
            entries.append({"id": kind, "revision": 1, "kind": kind, "frame": frame,
                            "asset": {"path": path.name, "sha256": file_sha256(path)},
                            "strength": strength, "direction": "both", "range": [frame, frame + 1]})
        entries.append({"id": "reset", "revision": 1, "kind": "reset", "frame": 4})
        data = {"schema_version": 1, "revision": 1,
                "source": {"sha256": source_identity(source)["sha256"], "start_frame": 0,
                           "frame_count": 9, "fps": 24, "width": 64, "height": 64},
                "shots": [[0, 9]], "entries": entries}
        path = root / "corrections.json"
        save_corrections(path, CorrectionDocument.parse(data, root))
        return path

    def test_full_rate_and_quick_both_memory_paths_apply_authored_alpha_once(self):
        import cv2
        import numpy as np
        for enabled in (False, True):
            for low_vram in (False, True):
                with self.subTest(enabled=enabled, low_vram=low_vram), tempfile.TemporaryDirectory() as folder:
                    _, d = self.run_job(folder, enabled=enabled, low_vram=low_vram, corrections=self.project)
                    paths = sorted(Path(d.outputs["alpha"]).glob("*.exr"))
                    self.assertEqual(len(paths), 9)
                    keep, remove, matte = [cv2.imread(str(paths[t]), -1) for t in (1, 2, 3)]
                    np.testing.assert_allclose(keep[:, :32], 1)
                    np.testing.assert_allclose(keep[:, 32:], 0.4)
                    np.testing.assert_allclose(remove[:, :32], 0.4)
                    np.testing.assert_array_equal(remove[:, 32:], 0)
                    np.testing.assert_allclose(matte, 0.6, atol=1e-6)
                    self.assertEqual(d.corrections["output_frames_refined"], 3)
                    self.assertGreater(d.corrections["hint_frame_evaluations_conditioned"], 0)
                    if enabled:
                        self.assertEqual(d.quick_key["planned_anchors"], [0, 1, 2, 3, 4, 8])

    def test_reset_starts_fresh_hint_carry_and_actual_rvm_state(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            model, d = self.run_job(folder, enabled=False, use_rvm=True, corrections=self.project)
            self.assertEqual(model.foundations.rvm.calls, [(4, None), (4, None), (1, 4)])
            self.assertTrue(torch.all(model.stage1.seen[1][2] == -1))
            self.assertTrue(torch.allclose(model.stage1.seen_hints[0][0, 3], torch.full((1, 64, 64), 0.8, dtype=torch.bfloat16)))
            self.assertGreater(d.corrections["aux_frame_evaluations_refined"], 0)

    def test_interval_one_preserves_the_corrected_full_rate_path(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _, baseline = self.run_job(first, enabled=False, corrections=self.project)
            _, full = self.run_job(second, interval=1, corrections=self.project)
            for name in baseline.outputs:
                for left, right in zip(sorted(Path(baseline.outputs[name]).glob("*.exr")),
                                       sorted(Path(full.outputs[name]).glob("*.exr"))):
                    np.testing.assert_array_equal(cv2.imread(str(left), -1), cv2.imread(str(right), -1))

    def test_forward_static_range_does_not_feed_model_hints(self):
        from corridorkey.corrections import load_corrections, CorrectionDocument, save_corrections
        import cv2
        import numpy as np
        def project(source, root):
            path = self.project(source, root)
            data = load_corrections(path).to_dict()
            data["entries"] = [data["entries"][0]]
            data["entries"][0].update(direction="forward", range=[1, 4])
            save_corrections(path, CorrectionDocument.parse(data, root))
            return path
        with tempfile.TemporaryDirectory() as folder:
            _, d = self.run_job(folder, enabled=False, corrections=project)
            paths = sorted(Path(d.outputs["alpha"]).glob("*.exr"))
            self.assertEqual(d.corrections["hint_frame_evaluations_conditioned"], 0)
            self.assertEqual(d.corrections["aux_frame_evaluations_refined"], 0)
            for frame in (1, 2, 3):
                np.testing.assert_array_equal(cv2.imread(str(paths[frame]), -1)[:, :32], 1)
            np.testing.assert_allclose(cv2.imread(str(paths[4]), -1), 0.4)

    def test_project_geometry_and_new_cut_conflicts_fail_before_neural_stages(self):
        from corridorkey.corrections import load_corrections, CorrectionDocument, save_corrections
        from unittest.mock import patch
        from corridorkey import runtime
        def bad_geometry(source, root):
            path = self.project(source, root)
            data = load_corrections(path).to_dict()
            data["source"]["width"] = 63
            save_corrections(path, CorrectionDocument.parse(data, root))
            return path
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime, "_phase1_setup") as phase:
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.run_job(folder, corrections=bad_geometry)
            phase.assert_not_called()
        def crosses_cut(source, root):
            path = self.project(source, root)
            data = load_corrections(path).to_dict()
            data["entries"] = [data["entries"][0]]
            data["entries"][0].update(direction="forward", range=[1, 8])
            save_corrections(path, CorrectionDocument.parse(data, root))
            return path
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime, "_phase1_setup") as phase:
            with self.assertRaisesRegex(ValueError, "crosses an effective cut"):
                self.run_job(folder, corrections=crosses_cut, scene_cut=True, auto_cuts=True)
            phase.assert_not_called()

    def test_correction_cancellation_closes_owned_temporary_maps(self):
        from corridorkey.control import JobControl, CancellationToken, CancelledError
        from corridorkey.diagnostics import RunDiagnostics
        token = CancellationToken()
        def progress(event):
            if event.stage == "stage1_windows" and event.completed == 1:
                token.cancel("cancel corrections")
        diagnostics = RunDiagnostics()
        diagnostics.control = JobControl(token, progress)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(CancelledError, "cancel corrections"):
                self.run_job(folder, corrections=self.project, diagnostics=diagnostics)
            self.assertEqual(list(Path(folder, "temp").iterdir()), [])

    def test_directional_constraints_do_not_enter_quick_key_propagation_state(self):
        from corridorkey.corrections import load_corrections, CorrectionDocument, save_corrections
        import cv2
        import numpy as np
        def project(source, root):
            path = self.project(source, root)
            data = load_corrections(path).to_dict()
            data["entries"] = [data["entries"][0]]
            data["entries"][0].update(direction="forward", range=[1, 2])
            save_corrections(path, CorrectionDocument.parse(data, root))
            return path
        for approach in ("offline", "hold", "causal"):
            with self.subTest(approach=approach), tempfile.TemporaryDirectory() as folder:
                _, d = self.run_job(folder, corrections=project, approach=approach)
                paths = sorted(Path(d.outputs["alpha"]).glob("*.exr"))
                np.testing.assert_array_equal(cv2.imread(str(paths[1]), -1)[:, :32], 1)
                for frame in (0, 2, 3, 4, 5, 6, 7, 8):
                    np.testing.assert_allclose(cv2.imread(str(paths[frame]), -1), 0.4, atol=1e-6)
                self.assertEqual(d.corrections["propagation_anchor_evaluations_refined"], 0)

    def test_mixed_constraint_order_and_soft_strength_survive_draft_anchor_export(self):
        from corridorkey.corrections import load_corrections, CorrectionDocument, save_corrections
        import cv2
        import numpy as np
        def project(source, root):
            path = self.project(source, root)
            data = load_corrections(path).to_dict()
            first = data["entries"][0]
            first.update(frame=3, direction="forward", range=[3, 4])
            matte = data["entries"][2]
            data["entries"] = [first, matte]
            save_corrections(path, CorrectionDocument.parse(data, root))
            return path
        with tempfile.TemporaryDirectory() as folder:
            _, d = self.run_job(folder, corrections=project, fallback="approximate")
            file = sorted(Path(d.outputs["alpha"]).glob("*.exr"))[3]
            alpha = cv2.imread(str(file), -1)
            np.testing.assert_allclose(alpha[:, :32], 0.9, atol=1e-6)
            np.testing.assert_allclose(alpha[:, 32:], 0.6, atol=1e-6)
