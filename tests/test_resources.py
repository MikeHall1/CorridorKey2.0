import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext
from types import SimpleNamespace

from corridorkey.resources import ResourceScope

HAS_STACK = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "cv2"))


class OwnershipTests(unittest.TestCase):
    def test_reverse_cleanup_preserves_primary_and_attempts_every_resource(self):
        events = []
        def broken():
            events.append("broken")
            raise OSError("flush failed")
        with self.assertRaisesRegex(ValueError, "processing") as caught:
            with ResourceScope() as resources:
                resources.own(None, lambda: events.append("first"))
                resources.own(None, broken)
                raise ValueError("processing")
        self.assertEqual(events, ["broken", "first"])
        self.assertIn("flush failed", caught.exception.__notes__[0])


@unittest.skipUnless(HAS_STACK, "requires optional CPU numeric stack")
class MappingTests(unittest.TestCase):
    def test_rvm_chronology_resets_even_inside_prefetched_chunks(self):
        import numpy as np
        import torch
        from corridorkey import runtime
        calls = []
        class RVM(torch.nn.Module):
            embed_dim = 1
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
            def forward(self, rgb, state=None):
                calls.append((rgb.shape[1], state))
                start = state or 0
                values = torch.arange(start + 1, start + rgb.shape[1] + 1, dtype=torch.float32)
                return values.reshape(1, -1, 1, 1, 1), start + rgb.shape[1]
        foundations = SimpleNamespace(rvm=RVM(), rvm_mean=torch.zeros(1), rvm_std=torch.ones(1))
        def source(*args, **kwargs):
            for first, count in ((0, 4), (4, 3)):
                yield first, None, torch.zeros(count, 3, 64, 64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "rvm.f16")
            with (patch.object(runtime, "iter_frame_dir_chunks", side_effect=source),
                  patch.object(torch.cuda, "empty_cache"),
                  patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext())):
                runtime.precompute_rvm_to_memmap(SimpleNamespace(foundations=foundations), "source",
                                                 {"num_frames": 7, "shape_hw": (64, 64)}, path,
                                                 reader_kind="frame_dir", chunk=4, reset_frames=(0, 3, 5))
            np.testing.assert_array_equal(np.fromfile(path, dtype=np.float16), [1, 2, 3, 1, 2, 1, 2])
            self.assertEqual(calls, [(3, None), (1, None), (1, 1), (2, None)])

    def test_mapping_flushes_and_closes_on_exception(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "map.f32")
            with self.assertRaises(ValueError):
                with ResourceScope() as resources:
                    mapping = resources.mapping(np.memmap, path, dtype="float32", mode="w+", shape=(2,))
                    mapping[:] = [1, 2]
                    raise ValueError("stop")
            self.assertTrue(mapping._mmap.closed)
            np.testing.assert_array_equal(np.fromfile(path, dtype="float32"), [1, 2])
            path.unlink()

    def test_cancelled_prepasses_close_maps_and_prefetched_source(self):
        import numpy as np
        import torch
        from corridorkey import runtime
        from corridorkey.control import JobControl, CancellationToken, CancelledError

        class RVM(torch.nn.Module):
            embed_dim = 1
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
            def forward(self, rgb, state=None):
                return torch.zeros(1, rgb.shape[1], 1, 2, 2), None

        class Foundations(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rvm = RVM()
                self.rvm_mean = torch.zeros(1)
                self.rvm_std = torch.ones(1)
            def forward(self, **kwargs):
                return {"moge": torch.zeros(1, 1, 2, 2)}

        model = SimpleNamespace(foundations=Foundations(), stage1=SimpleNamespace(moge_dim=1, cradio_dim=0))
        for prepass in (runtime.precompute_rvm_to_memmap, runtime.precompute_foundations_to_memmap):
            with self.subTest(prepass=prepass.__name__), tempfile.TemporaryDirectory() as folder:
                token = CancellationToken()
                closed, mappings = [], []
                def source(*args, **kwargs):
                    try:
                        for t in range(3):
                            yield t, None, torch.zeros(1, 3, 128, 128)
                    finally:
                        closed.append(True)
                original_memmap = np.memmap
                def tracked(*args, **kwargs):
                    mapping = original_memmap(*args, **kwargs)
                    mappings.append(mapping)
                    return mapping
                def progress(event):
                    if event.completed == 1:
                        token.cancel("cancel prepass")
                with (patch.object(runtime, "iter_frame_dir_chunks", side_effect=source),
                      patch.object(np, "memmap", side_effect=tracked),
                      patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext())):
                    with self.assertRaisesRegex(CancelledError, "cancel prepass"):
                        prepass(model, "fixture", {"num_frames": 3, "shape_hw": (128, 128)},
                                Path(folder, "cache"), chunk=1, reader_kind="frame_dir",
                                control=JobControl(token, progress))
                self.assertEqual(closed, [True])
                self.assertTrue(mappings)
                self.assertTrue(all(mapping._mmap.closed for mapping in mappings))

    def test_prepass_rejects_truncated_or_nonchronological_chunks(self):
        import torch
        from corridorkey.runtime import _checked_chunks
        for starts in ([0, 2], [0, 0], [0]):
            with self.subTest(starts=starts):
                chunks = ((t, None, torch.zeros(1, 3, 2, 2)) for t in starts)
                with self.assertRaises(RuntimeError):
                    list(_checked_chunks(chunks, 3))
