"""Single-owner bounded payload cache. No model, device or filesystem imports."""
from collections import OrderedDict


class ByteBudgetCache:
    """LRU admission with eviction before allocation via a caller-owned factory.

    The budget covers retained payload bytes, not Python objects or caller-held
    references. Callers must not retain borrowed values after eviction. Each
    factory must return an independently owned payload of the declared size.
    """
    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("cache max_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self.bytes_used = 0
        self.peak_bytes = 0
        self.hits = self.misses = self.evictions = self.oversized = 0

    def get(self, key):
        if key not in self._entries:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return self._entries[key][1]

    def store(self, key, size, factory):
        if type(size) is not int or size <= 0:
            raise ValueError("payload size must be a positive integer")
        if size > self.max_bytes:
            self.oversized += 1
            return False
        if key in self._entries:
            self.bytes_used -= self._entries.pop(key)[0]
        while self.bytes_used + size > self.max_bytes:
            entry = self._entries.popitem(last=False)
            old_size = entry[1][0]
            del entry  # Release the evicted payload before allocating its replacement.
            self.bytes_used -= old_size
            self.evictions += 1
        # No new entry is published when allocation/copy fails.
        value = factory()
        self._entries[key] = (size, value)
        self.bytes_used += size
        self.peak_bytes = max(self.peak_bytes, self.bytes_used)
        return True

    def discard_before(self, frame):
        for key in list(self._entries):
            if key < frame:
                self.bytes_used -= self._entries.pop(key)[0]

    def clear(self):
        self._entries.clear()
        self.bytes_used = 0

    def snapshot(self):
        requests = self.hits + self.misses
        return {"budget_bytes": self.max_bytes, "resident_bytes": self.bytes_used,
                "peak_payload_bytes": self.peak_bytes, "entries": len(self._entries),
                "hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / requests if requests else None,
                "evictions": self.evictions, "oversized_frames": self.oversized}
