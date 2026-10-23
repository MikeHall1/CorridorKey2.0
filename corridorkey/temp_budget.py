"""Exact admission for job-local writable map payloads, without numeric imports.

This is a logical file-byte ceiling, not a RAM, VRAM, output, physical disk
reservation or whole-process resource limit. Files remain charged until cleanup.
"""
from dataclasses import dataclass
from pathlib import Path

MAX_BYTES = (1 << 63) - 1


def extent(shape, itemsize):
    total = itemsize
    if type(total) is not int or total <= 0:
        raise ValueError("map item size must be a positive integer")
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("map dimensions must be positive integers")
        total *= dimension
        if total > MAX_BYTES:
            raise ValueError("map extent exceeds supported file size")
    return total


@dataclass(frozen=True)
class MapSpec:
    name: str
    shape: tuple[int, ...]
    itemsize: int

    @property
    def size(self):
        return extent(self.shape, self.itemsize)


def plan_temp_maps(model, meta, *, low_vram, anchor_count=None):
    """Use precisely the conditional dimensions consumed by runtime map sites."""
    frames = int(meta["num_frames"])
    height, width = meta["shape_hw"]
    maps = []
    rvm = model.foundations.rvm
    if rvm is not None:
        maps.append(MapSpec("rvm_feat.f16", (frames, rvm.embed_dim, height // 64, width // 64), 2))
    if low_vram:
        moge = int(getattr(model.stage1, "moge_dim", 0) or 0)
        cradio = int(getattr(model.stage1, "cradio_dim", 0) or 0)
        maps.append(MapSpec("foundation_moge.f32", (frames, moge, height // 64, width // 64), 4))
        if cradio > 0:
            maps.append(MapSpec("foundation_cradio.f32", (frames, cradio, height // 64, width // 64), 4))
        maps.append(MapSpec("baton.f16", (frames, model.stage1.d_16, height // 16, width // 16), 2))
    if anchor_count is not None:
        output_h, output_w = meta["output_hw"]
        maps.append(MapSpec("anchor_alpha.f32", (anchor_count, output_h, output_w), 4))
    return tuple(maps)


class TempMapBudget:
    def __init__(self, specs, limit):
        if limit is not None and (type(limit) is not int or not 0 <= limit <= MAX_BYTES):
            raise ValueError("temp map budget must be None or an integer in 0..2^63-1")
        self.specs = {spec.name: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("duplicate temporary map plan")
        required = sum(spec.size for spec in specs)
        if required > MAX_BYTES:
            raise ValueError("aggregate map extent exceeds supported file size")
        self.root: Path | None = None
        self.claimed = set()
        self.report = {
            "required_bytes": required, "limit_bytes": limit,
            "reserved_bytes": 0, "peak_reserved_bytes": 0, "created_bytes": 0,
            "status": "planned", "cleanup": "not_started",
            "maps": [{"name": spec.name, "shape": list(spec.shape), "bytes": spec.size,
                      "status": "planned"} for spec in specs],
            "scope": "aggregate logical writable inference map payloads; excludes RAM, VRAM, outputs, durable caches, filesystem overhead and physical disk reservation",
        }

    def admit(self):
        limit = self.report["limit_bytes"]
        if limit is not None and self.report["required_bytes"] > limit:
            self.report["status"] = "rejected"
            raise ValueError(f"Managed inference scratch requires {self.report['required_bytes']} bytes; limit is {limit} bytes")
        self.report["status"] = "admitted"

    def mapping(self, resources, factory, path, *, dtype, mode, shape):
        # Only write sites call this fence. Never infer a write from a readonly reopen.
        name = Path(path).name
        if self.root is None or Path(path).parent != self.root:
            raise RuntimeError("Writable temporary map is outside the owned job directory")
        spec = self.specs.get(name)
        itemsize = dtype().itemsize
        if (self.report["status"] != "admitted" or mode != "w+" or spec is None
                or tuple(shape) != spec.shape or itemsize != spec.itemsize or name in self.claimed):
            raise RuntimeError("Writable temporary map does not match admitted plan")
        self.claimed.add(name)
        self.report["reserved_bytes"] += spec.size
        self.report["peak_reserved_bytes"] = self.report["reserved_bytes"]
        row = next(row for row in self.report["maps"] if row["name"] == name)
        row["status"] = "creation_started"
        # A failed factory may leave a partial file. Keep its claim until cleanup.
        result = resources.mapping(factory, path, dtype=dtype, mode=mode, shape=shape)
        row["status"] = "created"
        self.report["created_bytes"] += spec.size
        return result

    def cleanup(self, callback):
        self.report["cleanup"] = "started"
        try:
            callback()
        except BaseException:
            self.report["cleanup"] = "failed; reserved bytes may remain"
            raise
        self.report["cleanup"] = "complete"
        self.report["reserved_bytes"] = 0
