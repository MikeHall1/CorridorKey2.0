"""Exact logical scratch planning and allocation fence regressions."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from corridorkey.temp_budget import (
    MAX_BYTES,
    MapSpec,
    TempMapBudget,
    extent,
    plan_temp_maps,
)


class TempBudgetTests(unittest.TestCase):
    def model(self, *, rvm=True, cradio=3):
        return NS(foundations=NS(rvm=NS(embed_dim=7) if rvm else None),
                  stage1=NS(moge_dim=5, cradio_dim=cradio, d_16=11))

    def test_exact_recipes_and_conditional_dimensions(self):
        meta = {"num_frames": 9, "shape_hw": (128, 192), "output_hw": (71, 83)}
        specs = plan_temp_maps(self.model(), meta, low_vram=True, anchor_count=3)
        self.assertEqual({s.name: s.size for s in specs}, {
            "rvm_feat.f16": 9 * 7 * 2 * 3 * 2,
            "foundation_moge.f32": 9 * 5 * 2 * 3 * 4,
            "foundation_cradio.f32": 9 * 3 * 2 * 3 * 4,
            "baton.f16": 9 * 11 * 8 * 12 * 2,
            "anchor_alpha.f32": 3 * 71 * 83 * 4})
        tiny = NS(foundations=NS(rvm=None))
        self.assertEqual(plan_temp_maps(tiny, meta, low_vram=False), ())
        self.assertNotIn("foundation_cradio.f32", {s.name for s in plan_temp_maps(
            self.model(rvm=False, cradio=0), meta, low_vram=True)})

    def test_checked_arithmetic_and_limit_boundaries(self):
        for shape in ((0,), (-1,), (True,), (MAX_BYTES, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                extent(shape, 2)
        spec = MapSpec("map", (4, 5), 2)
        for limit in (None, 40, MAX_BYTES):
            TempMapBudget((spec,), limit).admit()
        budget = TempMapBudget((spec,), 39)
        with self.assertRaisesRegex(ValueError, "40 bytes; limit is 39"):
            budget.admit()
        self.assertEqual(budget.report["status"], "rejected")
        TempMapBudget((), 0).admit()
        for invalid in (False, -1, 1.5, MAX_BYTES + 1):
            with self.assertRaises(ValueError):
                TempMapBudget((), invalid)

    def test_fence_before_factory_and_failed_creation_keeps_reservation(self):
        import numpy as np

        from corridorkey.resources import ResourceScope
        budget = TempMapBudget((MapSpec("map", (4,), 2),), 8)
        budget.admit()
        with tempfile.TemporaryDirectory() as folder:
            budget.root = Path(folder)
            factory = Mock(side_effect=OSError("disk full"))
            with ResourceScope() as scope:
                for path, shape, dtype, mode in ((Path(folder)/"other", (4,), np.float16, "w+"),
                    (Path(folder)/"map", (5,), np.float16, "w+"),
                    (Path(folder)/"map", (4,), np.float32, "w+"),
                    (Path(folder)/"map", (4,), np.float16, "r"),
                    (Path(folder)/"otherdir"/"map", (4,), np.float16, "w+")):
                    with self.assertRaises(RuntimeError):
                        budget.mapping(scope, factory, path, dtype=dtype, mode=mode, shape=shape)
                factory.assert_not_called()
                with self.assertRaises(OSError):
                    budget.mapping(scope, factory, Path(folder)/"map", dtype=np.float16, mode="w+", shape=(4,))
            self.assertEqual(budget.report["reserved_bytes"], 8)
            self.assertEqual(budget.report["created_bytes"], 0)
            with self.assertRaises(OSError):
                budget.cleanup(lambda: (_ for _ in ()).throw(OSError("cleanup denied")))
            self.assertEqual(budget.report["reserved_bytes"], 8)
            budget.cleanup(lambda: None)
            self.assertEqual(budget.report["reserved_bytes"], 0)

    def test_real_mapping_close_readonly_and_duplicate_are_counted_once(self):
        import numpy as np

        from corridorkey.resources import ResourceScope
        budget = TempMapBudget((MapSpec("map", (4,), 2),), 8)
        budget.admit()
        with tempfile.TemporaryDirectory() as folder:
            budget.root = Path(folder)
            path = Path(folder)/"map"
            with ResourceScope() as scope:
                mapping = budget.mapping(scope, np.memmap, path, dtype=np.float16, mode="w+", shape=(4,))
                mapping[:] = 1
            self.assertEqual(path.stat().st_size, 8)
            self.assertEqual(budget.report["reserved_bytes"], 8)
            with ResourceScope() as scope:
                read = scope.mapping(np.memmap, path, dtype=np.float16, mode="r", shape=(4,))
                self.assertEqual(read.tolist(), [1] * 4)
                with self.assertRaises(RuntimeError):
                    budget.mapping(scope, np.memmap, path, dtype=np.float16, mode="w+", shape=(4,))
            self.assertEqual(budget.report["created_bytes"], 8)
            path.unlink()
            budget.cleanup(lambda: None)
            self.assertEqual(budget.report["peak_reserved_bytes"], 8)
