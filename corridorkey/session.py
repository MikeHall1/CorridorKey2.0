"""Single-owner persistent inference session shared by CLI, launcher and batch."""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .durable_foundations import DurableFeatureFrames
    from .durable_hann import DurableHannWindows
    from .durable_rvm import DurableRvmChunks
    from .durable_stage2 import DurableStage2Batches
    from .durable_writer_prefix import DurableWriterPrefix
    from .memory_guard import ProcessTreeRssGuard

from .config import InferenceConfig, WriterConfig
from .control import CancelledError, JobControl
from .diagnostics import RunDiagnostics, json_value
from .lifecycle import cleanup_all


def _hint_path(path):
    if path is None or str(path).strip().lower() in {"", "none", "null", "off", "false", "-"}:
        return None
    return str(Path(path).resolve())


@dataclass(frozen=True)
class JobRequest:
    input_path: str
    output_dir: str
    config_json: str
    writer_json: str
    initial_hint_path: str | None = None
    hint_video_path: str | None = None
    profile_stages: bool = False

    @classmethod
    def create(cls, input_path, output_dir, config=None, writer=None, *,
               initial_hint_path=None, hint_video_path=None, profile_stages=False):
        config = config if config is not None else InferenceConfig()
        writer = writer if writer is not None else WriterConfig()
        config.validate()
        writer.validate()
        if config.corrections_path is not None:
            config = replace(config, corrections_path=str(Path(config.corrections_path).resolve()))
        if type(profile_stages) is not bool:
            raise ValueError("profile_stages must be a boolean")
        return cls(str(Path(input_path).resolve()), str(Path(output_dir).resolve()),
                   json.dumps(json_value(config), sort_keys=True, allow_nan=False),
                   json.dumps(json_value(writer), sort_keys=True, allow_nan=False),
                   _hint_path(initial_hint_path), _hint_path(hint_video_path), profile_stages)

    def configurations(self):
        settings = json.loads(self.config_json)
        settings["outputs"] = set(settings["outputs"])
        settings["temp_dir"] = Path(settings["temp_dir"])
        config, writer = InferenceConfig(**settings), WriterConfig(**json.loads(self.writer_json))
        config.validate()
        writer.validate()
        return config, writer

    def validate_source(self, config):
        source = Path(self.input_path)
        if not source.exists():
            raise FileNotFoundError(self.input_path)
        if config.source_colour_profile != 'legacy' and not source.is_dir():
            raise ValueError('Declared source primaries require a scene-linear frame directory; video is unsupported')
        if config.source_witness and not source.is_dir():
            raise ValueError('source_witness requires a scene-linear RGB frame directory, not video.')
        if config.frame_dir_alpha != "ignore" and not source.is_dir():
            raise ValueError("premultiplied input requires a frame directory; video alpha decoding is unsupported")
        for hint in (self.initial_hint_path, self.hint_video_path):
            if hint is not None and not Path(hint).exists():
                raise FileNotFoundError(hint)
        if source.is_dir():
            if config.start_mode != "begin":
                raise ValueError("frame-directory inputs currently support start_mode=begin only")
            if (config.linear_brightness, config.linear_contrast, config.linear_contrast_pivot) != (1.0, 1.0, 0.18):
                raise ValueError("linear brightness/contrast adjustments are currently supported for video inputs only")
        if Path(self.output_dir).exists() and not Path(self.output_dir).is_dir():
            raise ValueError("output_dir must be a directory")


@dataclass(frozen=True)
class JobResult:
    """Immutable serialized snapshot; to_dict returns a fresh mutable copy."""
    record_json: str

    def to_dict(self):
        return json.loads(self.record_json)


class InferenceSession:
    def __init__(self, package_root=None, weights=None, *, loader=None, runner=None, disposer=None, asset_root=None):
        self.package_root = Path(package_root or Path(__file__).resolve().parents[1]).resolve()
        self.asset_root = Path(asset_root).resolve() if asset_root is not None else None
        self.weights = str(weights) if weights is not None else None
        self._loader = loader or self._load
        self._runner = runner or self._run
        self._disposer = disposer or self._dispose
        self._model = None
        self._asset_paths = None
        self._load_seconds = None
        self._lock = threading.Lock()
        self._closed = False
        self.last_result = None

    def _load(self):
        import torch

        from .runtime import RuntimePaths, load_model
        if not torch.cuda.is_available():
            raise RuntimeError("CorridorKey v2 runtime requires a CUDA GPU.")
        paths = RuntimePaths.from_package(self.package_root, self.weights, asset_root=self.asset_root)
        self._asset_paths = {name: str(getattr(paths, name)) for name in
                             ("weights", "foundation_stats", "rvm_ckpt", "moge_checkpoint", "cradio_dir")}
        started = time.perf_counter()
        try:
            model = load_model(paths, device="cuda")
            torch.cuda.synchronize()
        finally:
            self._load_seconds = time.perf_counter() - started
        return model

    @staticmethod
    def _run(model, job, config, writer, diagnostics):
        from .runtime import run_inference
        return run_inference(model, job.input_path, job.output_dir, config, writer,
                             initial_hint_path=job.initial_hint_path, hint_video_path=job.hint_video_path,
                             diagnostics=diagnostics)

    @staticmethod
    def _dispose(model):
        import torch
        model.cpu()
        torch.cuda.empty_cache()

    def _discard(self):
        if self._model is not None:
            model, self._model = self._model, None
            self._disposer(model)

    def run(self, job: JobRequest, *, cancellation=None, progress=None,
            durable_foundations: DurableFeatureFrames | None = None,
            durable_rvm: DurableRvmChunks | None = None,
            durable_hann: DurableHannWindows | None = None,
            durable_stage2: DurableStage2Batches | None = None,
            durable_writer_prefix: DurableWriterPrefix | None = None,
            resource_guard: ProcessTreeRssGuard | None = None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("session is busy; use a separate session for concurrent jobs")
        started = time.perf_counter()
        diagnostics = RunDiagnostics(profile_stages=job.profile_stages)
        diagnostics.control = JobControl(cancellation, progress)
        diagnostics.durable_foundations = durable_foundations
        diagnostics.durable_rvm = durable_rvm
        diagnostics.durable_hann = durable_hann
        diagnostics.durable_stage2 = durable_stage2
        diagnostics.durable_writer_prefix = durable_writer_prefix
        reused = self._model is not None
        model_touched = False
        guard = resource_guard
        record = {"schema_version": 1, "job_id": uuid.uuid4().hex, "status": "failed",
                  "input": job.input_path, "output_dir": job.output_dir,
                  "hints": {"first_frame": job.initial_hint_path, "full": job.hint_video_path},
                  "requested_settings": None,
                  "writer_settings": None,
                  "weights": self.weights, "model_reused": reused,
                  "model_initialization_seconds": 0.0, "model_load_seconds": 0.0 if reused else None,
                  "processing_seconds": None, "end_to_end_fps": None,
                  "provenance_note": "Paths only; asset hashes and repository provenance require a benchmark report."}
        try:
            if self._closed:
                raise RuntimeError("session is closed")
            diagnostics.control.checkpoint("validate", 0, 1)
            config, writer = job.configurations()
            if durable_writer_prefix is not None and config.output_format != 'exr':
                raise ValueError('durable writer-prefix recovery requires EXR output')
            record["requested_settings"] = json_value(config)
            record["writer_settings"] = json_value(writer)
            if config.process_tree_rss_stop_bytes is not None:
                from .memory_guard import ProcessTreeRssGuard

                if guard is None:
                    guard = ProcessTreeRssGuard(config.process_tree_rss_stop_bytes,
                                               cancel_check=diagnostics.control.cancellation.check)
                elif guard.limit_bytes != config.process_tree_rss_stop_bytes:
                    raise ValueError('supplied resource guard differs from the requested RSS threshold')
                diagnostics.control.check_hook = guard.check
                guard.check(force=True)
            elif guard is not None:
                raise ValueError('a supplied resource guard requires a configured RSS stop threshold')
            job.validate_source(config)
            if config.corrections_path is not None:
                from .corrections import load_corrections
                diagnostics.control.checkpoint("correction_preflight", 0, 1)
                preflight_started = time.perf_counter()
                try:
                    diagnostics.correction_document = load_corrections(config.corrections_path, source=job.input_path,
                                                                      check=diagnostics.control.check)
                finally:
                    diagnostics.correction_preflight_seconds = time.perf_counter() - preflight_started
                diagnostics.corrections = {"identity": diagnostics.correction_document.identity,
                                           "revision": diagnostics.correction_document.to_dict()["revision"],
                                           "document_path": str(config.corrections_path),
                                           "source": diagnostics.correction_document.to_dict()["source"],
                                           "entries": diagnostics.correction_document.to_dict()["entries"]}
                diagnostics.control.checkpoint("correction_preflight", 1, 1)
            from .output_paths import validate_output_paths

            validate_output_paths(job.input_path, job.output_dir, config,
                                  initial_hint_path=job.initial_hint_path,
                                  hint_video_path=job.hint_video_path,
                                  correction_document=diagnostics.correction_document,
                                  check=diagnostics.control.check)
            diagnostics.control.checkpoint("validate", 1, 1)
            # Check an encoder before allocating a model. EXR needs no FFmpeg.
            if writer.ffmpeg_bin and config.output_format == "video":
                import shutil
                diagnostics.control.checkpoint("encoder_preflight", 0, 1)
                if shutil.which(writer.ffmpeg_bin) is None:
                    raise FileNotFoundError(f"ffmpeg binary not found: {writer.ffmpeg_bin}; install ffmpeg or select EXR output")
                diagnostics.control.checkpoint("encoder_preflight", 1, 1)
            if guard is not None:
                guard.check(force=True)
            if self._model is None:
                diagnostics.control.checkpoint("model_load", 0, 1)
                model_touched = True
                self._load_seconds = None
                load_started = time.perf_counter()
                try:
                    self._model = self._loader()
                finally:
                    record["model_initialization_seconds"] = time.perf_counter() - load_started
                    record["model_load_seconds"] = self._load_seconds
                diagnostics.control.checkpoint("model_load", 1, 1)
            if guard is not None:
                guard.check(force=True)
            processing_started = time.perf_counter()
            model_touched = True
            # The rendered-batch latency excludes model load and preflight.
            diagnostics.started = processing_started
            try:
                diagnostics.control.checkpoint("processing", 0, None)
                if durable_foundations is not None:
                    durable_foundations.bind(self._asset_paths)
                if durable_rvm is not None:
                    durable_rvm.bind(self._asset_paths)
                if durable_hann is not None:
                    durable_hann.bind(self._asset_paths)
                if durable_stage2 is not None:
                    durable_stage2.bind(self._asset_paths)
                if durable_writer_prefix is not None:
                    durable_writer_prefix.bind(self._asset_paths)
                self._runner(self._model, job, config, writer, diagnostics)
                diagnostics.control.check()
                if guard is not None:
                    guard.check(force=True)
            finally:
                record["processing_seconds"] = time.perf_counter() - processing_started
            record["status"] = "complete"
            if diagnostics.frames_submitted:
                record["end_to_end_fps"] = diagnostics.frames_submitted / record["processing_seconds"]
        except BaseException as exc:
            record["status"] = "cancelled" if isinstance(exc, (KeyboardInterrupt, CancelledError)) else "failed"
            record["failure_checkpoint"] = json_value(diagnostics.control.last_checkpoint)
            error_info: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
            record["error"] = error_info
            # Failed jobs never leave a potentially contaminated model for reuse.
            if model_touched:
                cleanup_all([self._discard], primary_error=exc)
            error_info["notes"] = getattr(exc, "__notes__", [])
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            try:
                if guard is not None:
                    diagnostics.resource_policy = guard.snapshot()
                record["diagnostics"] = diagnostics.snapshot()
                record["asset_paths"] = self._asset_paths
                result = JobResult(json.dumps(json_value(record), sort_keys=True, allow_nan=False))
                self.last_result = result
            finally:
                self._lock.release()
        return result

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close a busy session")
        try:
            self._closed = True
            self._discard()
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        cleanup_all([self.close], primary_error=exc)
