import unittest
import weakref
from unittest.mock import Mock

from corridorkey.cache import ByteBudgetCache


class CacheBudgetTests(unittest.TestCase):
    def test_lru_eviction_releases_old_payload_before_replacement_allocation(self):
        class Payload:
            pass
        cache = ByteBudgetCache(8)
        first = Payload()
        ref = weakref.ref(first)
        cache.store(0, 4, lambda: first)
        del first
        cache.store(1, 4, Payload)

        def replace():
            self.assertIsNone(ref(), "evicted payload must not be retained during allocation")
            self.assertEqual(cache.bytes_used, 4)
            return Payload()

        cache.store(2, 4, replace)
        self.assertIsNone(cache.get(0))
        self.assertIsNotNone(cache.get(1))
        self.assertIsNotNone(cache.get(2))
        self.assertEqual(cache.snapshot()["peak_payload_bytes"], 8)
        self.assertEqual(cache.snapshot()["evictions"], 1)

    def test_recently_used_frame_survives_and_obsolete_frames_are_released(self):
        cache = ByteBudgetCache(8)
        for i in (0, 1):
            cache.store(i, 4, lambda i=i: bytes([i]) * 4)
        self.assertEqual(cache.get(0), bytes(4))
        cache.store(2, 4, lambda: bytes([2]) * 4)
        self.assertIsNone(cache.get(1))
        cache.discard_before(2)
        self.assertEqual(cache.snapshot()["entries"], 1)
        self.assertEqual(cache.bytes_used, 4)
        cache.clear()
        self.assertEqual(cache.bytes_used, 0)
        self.assertEqual(cache.snapshot()["entries"], 0)

    def test_oversized_frame_is_not_allocated(self):
        factory = Mock()
        for budget in (0, 3):
            cache = ByteBudgetCache(budget)
            self.assertFalse(cache.store(0, 4, factory))
            self.assertEqual(cache.snapshot()["oversized_frames"], 1)
            self.assertEqual(cache.bytes_used, 0)
        factory.assert_not_called()

    def test_failed_copy_is_not_published(self):
        cache = ByteBudgetCache(4)
        cache.store(0, 4, lambda: b"abcd")
        with self.assertRaisesRegex(OSError, "copy failed"):
            cache.store(1, 4, Mock(side_effect=OSError("copy failed")))
        self.assertIsNone(cache.get(1))
        self.assertEqual(cache.bytes_used, 0)

    def test_invalid_budget_and_size(self):
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                ByteBudgetCache(value)
        cache = ByteBudgetCache(1)
        for value in (-1, 0, 1.5, True):
            with self.assertRaises(ValueError):
                cache.store(0, value, lambda: None)
