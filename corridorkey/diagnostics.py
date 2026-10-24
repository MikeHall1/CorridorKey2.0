"""Job records and optional synchronized stage measurements; no model imports."""
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable
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

MAX_STAGE_GROUPS = 64
MAX_STAGE_NAME_CHARS = 128
STAGE_OVERFLOW_NAME = "<stage-name-overflow>"
STAGE_AGGREGATION_SCOPE = "Per (name, method, status) inclusive scope totals; not event order or additive end-to-end time."

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
        self.synchronize: Callable[[], None] | None = None
        self._cuda_synchronize: Callable[[], None] | None = None
        self._cuda_synchronize_device: int | None = None
        self.stages = []
        self._stage_lock = threading.Lock() if profile_stages else None
        self._stage_groups = {} if profile_stages else None
        self._stage_named_groups = 0
        self._stage_observations = 0
        self._stage_overflow_observations = 0
        # Execution control, like control/durable stores; not a measured observation.
        self.preview_frame: int | None = None
        self.preview_selection: dict[str, Any] | None = None
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

    def _configure_cuda_synchronizer(self, synchronize: Callable[[], None], device_index: int) -> None:
        if type(device_index) is not int or device_index < 0 or not callable(synchronize):
            raise ValueError("CUDA stage synchronization requires a callable and explicit device index")
        self.synchronize = synchronize
        self._cuda_synchronize = synchronize
        self._cuda_synchronize_device = device_index

    def _matches_cuda_synchronizer(self, device_index: int | None) -> bool:
        return (self.profile_stages and self._cuda_synchronize is not None
                and self.synchronize is self._cuda_synchronize and self._cuda_synchronize_device == device_index)

    def _observe_cuda_wait(self, synchronize: Callable[[], None], name: str) -> None:
        started = self.clock()
        succeeded = False
        try:
            synchronize()
            succeeded = True
        finally:
            self._record_stage(name, self.clock() - started, "complete" if succeeded else "failed", "host_wall")

    @contextmanager
    def stage(self, name, *, gpu=True):
        if not self.profile_stages:
            yield
            return
        synchronize = self.synchronize if gpu else None
        observe_waits = synchronize is not None and self._matches_cuda_synchronizer(self._cuda_synchronize_device)
        if synchronize:
            if observe_waits:
                self._observe_cuda_wait(synchronize, "cuda_sync_before_scope")
            else:
                synchronize()
        started = self.clock()
        succeeded = False
        try:
            yield
            if synchronize:
                if observe_waits:
                    self._observe_cuda_wait(synchronize, "cuda_sync_after_scope")
                else:
                    synchronize()
            succeeded = True
        finally:
            self._record_stage(name, self.clock() - started, "complete" if succeeded else "failed",
                               "synchronized_wall" if synchronize else "host_wall")

    def _record_stage(self, name, seconds, status, method):
        assert self._stage_lock is not None
        with self._stage_lock:
            assert self._stage_groups is not None
            self._stage_observations += 1
            named = isinstance(name, str) and len(name) <= MAX_STAGE_NAME_CHARS and name != STAGE_OVERFLOW_NAME
            key = (name, method, status) if named else (STAGE_OVERFLOW_NAME, method, status)
            # Reserve four slots so unknown names never erase method/status evidence.
            if not named or key not in self._stage_groups and self._stage_named_groups >= MAX_STAGE_GROUPS - 4:
                key = (STAGE_OVERFLOW_NAME, method, status)
                self._stage_overflow_observations += 1
                named = False
            row = self._stage_groups.get(key)
            if row is None:
                row = {"name": key[0], "seconds": seconds, "status": status, "method": method,
                       "count": 1, "min_seconds": seconds, "max_seconds": seconds, "overflow": not named}
                self._stage_groups[key] = row
                self.stages.append(row)
                self._stage_named_groups += int(named)
            else:
                row["seconds"] += seconds
                row["count"] += 1
                row["min_seconds"] = min(row["min_seconds"], seconds)
                row["max_seconds"] = max(row["max_seconds"], seconds)


    def frame_ready(self):
        if self.first_frame_ready_seconds is None:
            self.first_frame_ready_seconds = self.clock() - self.started

    def snapshot(self):
        if self._stage_lock is None:
            stages, stage_metadata = self.stages, {}
        else:
            with self._stage_lock:
                stages = [dict(row) for row in self.stages]
                stage_metadata = {"stage_observations": {"schema_version": 1, "format": "aggregated_scopes_v1",
                    "group_limit": MAX_STAGE_GROUPS, "name_limit": MAX_STAGE_NAME_CHARS,
                    "observations": self._stage_observations, "overflow_observations": self._stage_overflow_observations,
                    "scope": STAGE_AGGREGATION_SCOPE}}
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
                           "preview_selection": self.preview_selection,
                           "first_rendered_batch_seconds": self.first_frame_ready_seconds,
                           "stages": stages, "profiling_enabled": self.profile_stages,
                           **stage_metadata,
                           **({"cuda_stage_observations": {"device_index": self._cuda_synchronize_device,
                               "runtime_callback_active": self.synchronize is self._cuda_synchronize,
                               "scope": "Host waits around existing device-bound stage callbacks only; overlap stage totals. "
                                        "RVM RGB upload includes float promotion; not pure DMA or all transfers."}}
                              if self.profile_stages and self._cuda_synchronize_device is not None else {}),
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
