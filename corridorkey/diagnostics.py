"""Job records and optional synchronized stage measurements; no model imports."""
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, overload

if TYPE_CHECKING:
    from .corrections import CorrectionDocument
    from .durable_foundations import DurableFeatureFrames
    from .durable_hann import DurableHannWindows
    from .durable_rvm import DurableRvmChunks
    from .durable_stage2 import DurableStage2Batches
    from .durable_writer_prefix import DurableWriterPrefix
    from .output_telemetry import OutputTelemetry

from .control import JobControl

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def finite_json_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('JSON numbers must be finite, including exponent overflow')
    return value


@overload
def json_value(value: dict[Any, Any]) -> dict[str, JsonValue]: ...


@overload
def json_value(value: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any]) -> list[JsonValue]: ...


@overload
def json_value(value: object) -> JsonValue: ...


def json_value(value: object) -> JsonValue:
    """Copy supported record values into finite JSON data; reject lossy inputs.

    Homogeneous numeric/string sets retain natural ordering. Other sets use
    canonical JSON text, so heterogeneous and nested values need no comparison.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key is not None and not isinstance(key, (str, bool, int, float, Path)):
                raise TypeError(f"Unsupported JSON record key: {type(key).__name__}")
            if isinstance(key, float) and not math.isfinite(key):
                raise ValueError("JSON record keys must be finite")
            name = str(key)
            if name in result:
                raise ValueError(f"JSON record key collision: {name!r}")
            result[name] = json_value(item)
        return result
    if isinstance(value, (set, frozenset)):
        items = [json_value(v) for v in value]
        numbers = [v for v in items if isinstance(v, (int, float))]
        if len(numbers) == len(items):
            return [v for v in sorted(numbers)]
        strings = [v for v in items if isinstance(v, str)]
        if len(strings) == len(items):
            return [v for v in sorted(strings)]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, allow_nan=False,
                                                      separators=(",", ":")))
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON record values must be finite")
        return value
    raise TypeError(f"Unsupported JSON record value: {type(value).__name__}")


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".record-", suffix=".json", delete=False) as stream:
            name = stream.name
            json.dump(json_value(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


class RunDiagnostics:
    def __init__(self, *, profile_stages=False, clock=time.perf_counter):
        self.profile_stages = profile_stages
        self.clock = clock
        self.started = clock()
        self.synchronize = None
        self.stages = []
        self.frames_submitted = 0
        self.first_frame_ready_seconds = None
        self.effective: JsonValue = None
        self.source = None
        self.source_witness: dict[str, JsonValue] | None = None
        self.execution = None
        self.outputs = {}
        self.output_contracts = {}
        self.resources = None
        self.resource_policy: dict[str, Any] | None = None
        self.foundation_cache = None
        self.durable_foundations: DurableFeatureFrames | None = None
        self.durable_hann: DurableHannWindows | None = None
        self.durable_rvm: DurableRvmChunks | None = None
        self.durable_stage2: DurableStage2Batches | None = None
        self.durable_writer_prefix: DurableWriterPrefix | None = None
        self.writer_prefix_runtime: dict[str, Any] | None = None
        self.output_telemetry: OutputTelemetry | None = None
        self.memory_policy = None
        self.warnings = []
        self.control = JobControl()
        self.quick_key = None
        self.correction_document: CorrectionDocument | None = None
        self.corrections: dict[str, Any] | None = None
        self.correction_preflight_seconds: float | None = None
        self.alpha_refinement = None

    @contextmanager
    def stage(self, name, *, gpu=True):
        if not self.profile_stages:
            yield
            return
        synchronize = self.synchronize if gpu else None
        if synchronize:
            synchronize()
        started = self.clock()
        succeeded = False
        try:
            yield
            if synchronize:
                synchronize()
            succeeded = True
        finally:
            self.stages.append({"name": name, "seconds": self.clock() - started,
                                "status": "complete" if succeeded else "failed",
                                "method": "synchronized_wall" if synchronize else "host_wall"})

    def frame_ready(self):
        if self.first_frame_ready_seconds is None:
            self.first_frame_ready_seconds = self.clock() - self.started

    def snapshot(self):
        return json_value({"effective_settings": self.effective, "source": self.source,
                           "last_checkpoint": self.control.last_checkpoint,
                           "source_witness": self.source_witness,
                           "execution": self.execution,
                           "quick_key": self.quick_key,
                           "corrections": self.corrections,
                           "correction_preflight_seconds": self.correction_preflight_seconds,
                           "alpha_refinement": self.alpha_refinement,
                           "outputs": self.outputs, "output_contracts": self.output_contracts,
                           "frames_submitted": self.frames_submitted,
                           "first_rendered_batch_seconds": self.first_frame_ready_seconds,
                           "stages": self.stages, "profiling_enabled": self.profile_stages,
                           "resources": self.resources, "warnings": self.warnings,
                           "resource_policy": self.resource_policy,
                           "foundation_cache": self.foundation_cache, "memory_policy": self.memory_policy,
                           "durable_foundations": self.durable_foundations.snapshot() if self.durable_foundations is not None else None,
                           "durable_hann": self.durable_hann.snapshot() if self.durable_hann is not None else None,
                           "durable_rvm": self.durable_rvm.snapshot() if self.durable_rvm is not None else None,
                           "durable_stage2": self.durable_stage2.snapshot() if self.durable_stage2 is not None else None,
                           "durable_writer_prefix": ({**self.durable_writer_prefix.snapshot(), **(self.writer_prefix_runtime or {})}
                                                     if self.durable_writer_prefix is not None else None),
                           "output_telemetry": self.output_telemetry.snapshot() if self.output_telemetry is not None else None,
                           "model_only_fps": None, "quality": None,
                           "look_ahead_seconds": None,
                           "timing_note": "Stage scopes may be nested; do not sum them. First batch is CPU-rendered, not encoded playback."})
