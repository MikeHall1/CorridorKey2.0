from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import color_utils as cu
from .config import InferenceConfig
from .runtime_paths import RuntimePaths
from .lifecycle import cleanup_all
from .resources import ResourceScope, scoped_resources
from .temp_budget import TempMapBudget, plan_temp_maps
from .spatial_draft import NativeGuidance, resize_coverage
from .diagnostics import RunDiagnostics, json_value
from .output_telemetry import OutputTelemetry
from .foundation_cache import FoundationWindowCache
from .durable_hann import replay_state_digest
from .control import JobControl
from .quick_key import plan_anchors, AnchorPlan
from .model import CorridorKeyV3
from .video_io import (
    ExrSequenceWriter,
    center_pad_to_shape,
    DraftSourceReader,
    ThreadedVideoWriter,
    WriterConfig,
    apply_linear_adjust,
    frame_paths,
    has_linear_adjust,
    iter_frame_dir_chunks,
    iter_video_frame_chunks,
    load_initial_hint_frame,
    make_checker,
    make_hint_reader,
    none_path,
    probe_frame_dir_meta,
    probe_video_meta,
    raw_clip_name,
    taper_weights,
    window_starts,
    SequentialFrameDirWindowReader,
    SequentialVideoWindowReader,
)


def _writable_map(budget, resources, factory, path, **kwargs):
    if budget is None:  # Standalone helper callers retain their existing API.
        return resources.mapping(factory, path, **kwargs)
    return budget.mapping(resources, factory, path, **kwargs)


# Peak GPU memory in GiB by megapixels of model input, measured at each Hann
# chunk size. Keys are low_vram False/True.
_VRAM_PEAK_GB = {
    False: ((8, 24, 48, 80), (
        (0.98, (6.19, 7.52, 9.77, 12.89)),
        (2.21, (7.50, 11.33, 18.66, 24.89)),
        (3.93, (9.06, 14.78, 23.92, 33.19)),
        (8.36, (13.94, 25.96, 45.15, 70.45)),
    )),
    True: ((1, 2, 4, 8, 24, 48, 80), (
        (0.98, (2.71, 2.78, 2.94, 3.25, 4.54, 6.57, 9.50)),
        (2.21, (2.80, 2.97, 3.35, 4.12, 7.69, 13.20, 20.56)),
        (3.93, (2.92, 3.23, 3.85, 5.09, 10.09, 17.57, 27.75)),
        (8.36, (3.25, 3.92, 5.28, 8.38, 20.78, 39.66, 64.65)),
    )),
}
_VRAM_MARGIN_SCALE = 1.1
_VRAM_MARGIN_GB = 0.3


def _interp(x: float, points) -> float:
    """Piecewise-linear through (x, y) points, extrapolating past the ends."""
    points = list(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1 or (x1, y1) == points[-1]:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[0][1]


def estimate_peak_vram_gb(megapixels: float, frames: int, low_vram: bool) -> float:
    """Estimated peak GPU memory for a Hann window of `frames` frames."""
    chunks, rows = _VRAM_PEAK_GB[low_vram]
    frames = max(frames, chunks[0])
    per_chunk = [
        max(_interp(megapixels, [(mp, peaks[i]) for mp, peaks in rows]), rows[0][1][i])
        for i in range(len(chunks))
    ]
    return _interp(frames, zip(chunks, per_chunk))


def select_vram_profile(shape_hw, num_frames: int, budget_gb: float):
    """Largest Hann window that fits `budget_gb`, as (chunk, stride, low_vram, estimate).

    Bigger windows give better temporal consistency, and low-VRAM mode is
    exact, so a larger low-VRAM window is preferred over a smaller normal one.
    Falls back to a single-frame window when nothing is estimated to fit.
    """
    megapixels = shape_hw[0] * shape_hw[1] / 1e6
    for chunk in reversed(_VRAM_PEAK_GB[True][0]):
        for low_vram in (False, True):
            if chunk not in _VRAM_PEAK_GB[low_vram][0]:
                continue
            estimate = estimate_peak_vram_gb(megapixels, min(chunk, num_frames), low_vram)
            if estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB <= budget_gb:
                return chunk, max(1, chunk // 2), low_vram, estimate
    return 1, 1, True, estimate


def apply_auto_vram(cfg: InferenceConfig, shape_hw, num_frames: int):
    free_bytes, _ = torch.cuda.mem_get_info()
    # Memory this process already holds (the loaded model) is part of the budget.
    reserved_bytes = torch.cuda.memory_reserved()
    available_bytes = free_bytes + reserved_bytes
    cap_bytes = cfg.auto_vram_budget_mb * 2**20 if cfg.auto_vram_budget_mb is not None else None
    budget_bytes = min(available_bytes, cap_bytes) if cap_bytes is not None else available_bytes
    if cap_bytes is not None and torch.cuda.memory_allocated() > cap_bytes:
        raise MemoryError("current process allocations exceed auto_vram_budget_mb; increase the selection budget or close/reload the session")
    budget_gb = budget_bytes / 2**30
    chunk, stride, low_vram, estimate = select_vram_profile(shape_hw, num_frames, budget_gb)
    if cap_bytes is not None and estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB > budget_gb:
        raise MemoryError(f"no single-frame profile fits the available/capped {budget_gb:.2f} GiB budget under the inherited VRAM estimate; free GPU memory, increase the cap where possible, or use a smaller input")
    cfg.hann_chunk, cfg.hann_stride, cfg.low_vram = chunk, stride, low_vram
    print(f"  Auto VRAM: {budget_gb:.1f} GB available at {shape_hw[1]}x{shape_hw[0]} -> "
          f"hann_chunk={chunk}, hann_stride={stride}, low_vram={low_vram} "
          f"(estimated peak {estimate:.1f} GB)")
    if estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB > budget_gb:
        print("  [WARNING] Even a single-frame window may not fit this GPU at this "
              "resolution. Inference may run out of memory.")
    return {"device_free_bytes": free_bytes, "process_reserved_bytes": reserved_bytes,
            "available_bytes": available_bytes, "user_cap_bytes": cap_bytes,
            "selection_budget_bytes": budget_bytes, "estimated_peak_bytes": int(estimate * 2**30),
            "note": "Heuristic profile-selection budget, not a hard allocator or whole-process cap."}



def prepare_runtime_environment(paths: RuntimePaths) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    moge_root = paths.package_root / "third_party/MoGe"
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))


def _resolve_package_path(paths: RuntimePaths, value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = paths.package_root / p
    return str(p.resolve())


def load_runtime_checkpoint(path: Path) -> dict:
    """Read a runtime checkpoint from .safetensors (preferred) or legacy .pt."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework="pt") as f:
            header = f.metadata() or {}
        return {
            "format": header.get("format"),
            "model_config": json.loads(header.get("model_config", "{}")),
            "metadata": json.loads(header.get("metadata", "{}")),
            "state_dict": load_file(str(path), device="cpu"),
        }
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_model(paths: RuntimePaths, device: str = "cuda") -> CorridorKeyV3:
    paths.validate()
    prepare_runtime_environment(paths)
    ckpt = load_runtime_checkpoint(paths.weights)
    if ckpt.get("format") != "corridorkey_v2_runtime_checkpoint":
        raise RuntimeError(f"{paths.weights} is not a CorridorKey v2 runtime checkpoint")
    foundation_keys = [key for key in ckpt["state_dict"] if key.startswith("foundations.")]
    if foundation_keys:
        raise RuntimeError(
            "runtime checkpoint must omit frozen foundations; use the separately "
            f"selected model and stats assets instead: {foundation_keys[:8]}")
    cfg = dict(ckpt["model_config"])
    cfg["cradio_repo"] = str(paths.cradio_dir)
    cfg["moge_checkpoint"] = str(paths.moge_checkpoint)
    cfg["rvm_ckpt"] = str(paths.rvm_ckpt)
    cfg["foundation_stats_path"] = str(paths.foundation_stats)
    cfg["gradient_checkpoint"] = False
    cfg["device"] = device
    cfg["stage1_recurrent_64_sites"] = ()

    print(f"Loading CorridorKey v2 runtime weights: {paths.weights}")
    model = CorridorKeyV3(**cfg)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    train_missing = [k for k in missing if not k.startswith("foundations.")]
    train_unexpected = [k for k in unexpected if not k.startswith("foundations.")]
    if train_missing or train_unexpected:
        raise RuntimeError(
            "runtime checkpoint did not match model: "
            f"missing={train_missing[:8]} unexpected={train_unexpected[:8]}")
    print(f"  loaded weights; omitted frozen foundation tensors={len(missing)}")
    model.eval()
    del ckpt
    return model


def configure_low_vram_inference(model, cfg: InferenceConfig):
    model.stage1.low_vram_enc_frame_chunk = max(0, int(cfg.low_vram_enc_frame_chunk))
    for module in model.modules():
        name = module.__class__.__name__
        if name in ("SwiGLU", "FusionLinSwiglu"):
            module.inference_chunk_tokens = max(0, int(cfg.low_vram_mlp_chunk_tokens))
        if name == "SwinAttention3D":
            module.inference_window_batch = max(0, int(cfg.low_vram_swin_window_batch))
        if name == "CSwinAttention3D":
            module.inference_strip_batch = max(0, int(cfg.low_vram_swin_window_batch))
        if name in ("FullSpatialAttention2D", "FullAttention3D"):
            module.inference_query_chunk_tokens = max(
                0, int(cfg.low_vram_full_attn_query_chunk_tokens))


def configure_inference_memory(model, cfg: InferenceConfig):
    """Apply every job's policy, clearing overrides left by a previous clip."""
    if not cfg.low_vram:
        cfg = replace(cfg, low_vram_enc_frame_chunk=0, low_vram_mlp_chunk_tokens=0,
                      low_vram_swin_window_batch=0, low_vram_full_attn_query_chunk_tokens=0)
    configure_low_vram_inference(model, cfg)


def _build_foundation_inputs(rgb_bt: torch.Tensor):
    B, T, _, H, W = rgb_bt.shape
    flat = rgb_bt.reshape(B * T, 3, H, W)
    rgb_radio = F.interpolate(flat, size=(H // 4, W // 4), mode="area")
    H_moge, W_moge = (H * 7) // 32, (W * 7) // 32
    rgb_moge = F.interpolate(flat, size=(H_moge, W_moge), mode="area")
    return (
        rgb_radio.reshape(B, T, 3, H // 4, W // 4),
        rgb_moge.reshape(B, T, 3, H_moge, W_moge),
    )


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def _checked_chunks(chunks, total):
    """Reject missing/overlapping frames before publishing a feature prepass."""
    next_frame = 0
    try:
        for item in chunks:
            start, _, display = item
            count = int(display.shape[0])
            if start != next_frame or count <= 0 or start + count > total:
                raise RuntimeError(f"Misaligned source chunk: expected {next_frame}, got {start}+{count}/{total}")
            next_frame += count
            yield item
        if next_frame != total:
            raise RuntimeError(f"Incomplete source prepass: decoded {next_frame}/{total} frames")
    finally:
        chunks.close()


def _split_chunks(chunks, reset_frames):
    try:
        for t0, model, display in chunks:
            stop = t0 + len(display)
            boundaries = [t0] + sorted(t for t in reset_frames if t0 < t < stop) + [stop]
            for first, last in zip(boundaries, boundaries[1:]):
                yield first, None if model is None else model[first - t0:last - t0], display[first - t0:last - t0]
    finally:
        chunks.close()


@torch.no_grad()
@scoped_resources
def precompute_rvm_to_memmap(model, source, meta, rvm_path: Path,
                             *, chunk: int, reader_kind: str, control=None, reset_frames=(),
                             durable_rvm=None, diagnostics=None, temp_budget=None, _resources=None):
    control = control if control is not None else JobControl()
    control.checkpoint("rvm_prepass", 0, int(meta["num_frames"]))
    assert model.foundations.rvm is not None
    device = next(model.foundations.rvm.parameters()).device
    T_all = int(meta["num_frames"])
    if durable_rvm is not None:
        reset_frames = tuple(reset_frames)
        if (len(set(reset_frames)) != len(reset_frames) or
                any(type(frame) is not int or not 0 <= frame < T_all for frame in reset_frames)):
            raise ValueError("durable RVM reset frames must be unique native indices inside the selected timeline")
    if diagnostics is not None:
        if diagnostics.execution is None:
            diagnostics.execution = {}
        diagnostics.execution["rvm_frame_evaluations"] = 0
    H, W = meta["shape_hw"]
    H_e, W_e = H // 4, W // 4
    D = model.foundations.rvm.embed_dim
    h_e, w_e = H_e // 16, W_e // 16
    print(f"    RVM continuous prepass: T={T_all}, chunk={chunk}, "
          f"input={W_e}x{H_e}, grid={w_e}x{h_e}")
    out = _writable_map(temp_budget, _resources, np.memmap, rvm_path, dtype=np.float16, mode="w+",
                    shape=(T_all, D, h_e, w_e))
    mean = model.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
    std = model.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)
    if reader_kind == "video":
        chunks = iter_video_frame_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    else:
        chunks = iter_frame_dir_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    state = None
    shot_start = 0
    chunks = _resources.own(_split_chunks(_checked_chunks(chunks, T_all), reset_frames))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            control.check()
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            if t0 in reset_frames:
                state = None
                shot_start = t0
            Tc = display.shape[0]
            identity = (durable_rvm.identity(t0, shot_start, display, state, channels=D, grid_hw=(h_e, w_e))
                        if durable_rvm is not None else None)
            cached = durable_rvm.get(identity) if durable_rvm is not None else None
            if cached is not None:
                state = cached["state"].to(device)
                out[t0:t0 + Tc] = cached["features"].numpy()
                del cached
            else:
                rgb = display.unsqueeze(0).to(device, non_blocking=True).float()
                rgb_rvm = F.interpolate(
                    rgb.reshape(Tc, 3, H, W), size=(H_e, W_e),
                    mode="bilinear", align_corners=False,
                ).reshape(1, Tc, 3, H_e, W_e).clamp(0.0, 1.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats, state = model.foundations.rvm(rgb_rvm, state=state)
                if diagnostics is not None:
                    diagnostics.execution["rvm_frame_evaluations"] += Tc
                feats_cpu = (feats.float().cpu() - mean) / std
                normalized = feats_cpu[0].numpy().astype(np.float16)
                out[t0:t0 + Tc] = normalized
                if durable_rvm is not None:
                    durable_rvm.put(identity, torch.from_numpy(normalized), state)
                del rgb, rgb_rvm, feats, feats_cpu, normalized
            del display
            control.checkpoint("rvm_prepass", t0 + Tc, T_all)
    out.flush()
    del out
    torch.cuda.empty_cache()


@torch.no_grad()
@scoped_resources
def precompute_foundations_to_memmap(model, source, meta, prefix: Path,
                                     *, chunk: int, reader_kind: str, control=None,
                                     durable_foundations=None, diagnostics=None, temp_budget=None, _resources=None):
    control = control if control is not None else JobControl()
    control.checkpoint("foundation_prepass", 0, int(meta["num_frames"]))
    device = next(model.foundations.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    h64, w64 = H // 64, W // 64
    C_cradio = int(getattr(model.stage1, "cradio_dim", 0) or 0)
    C_moge = int(getattr(model.stage1, "moge_dim", 0) or 0)
    if C_moge <= 0:
        raise RuntimeError("MoGe is required for this model")
    assert _resources is not None
    feature_cache = FoundationWindowCache(0, persistent=durable_foundations)
    _resources.own(feature_cache)

    def extract(selected):
        count = selected.shape[0]
        rgb = selected.unsqueeze(0).to(device, non_blocking=True)
        rgb_radio, rgb_moge = _build_foundation_inputs(rgb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = model.foundations(
                rgb_cradio=rgb_radio.reshape(count, 3, *rgb_radio.shape[-2:]),
                rgb_moge=rgb_moge.reshape(count, 3, *rgb_moge.shape[-2:]))
        return {name: features[name].float() for name, width in
                (("moge", C_moge), ("cradio", C_cradio)) if width > 0}
    paths = {
        "moge": str(prefix) + "_moge.f32",
        "moge_shape": (T_all, C_moge, h64, w64),
    }
    moge_out = _writable_map(temp_budget, _resources, np.memmap, paths["moge"], dtype=np.float32, mode="w+",
                         shape=paths["moge_shape"])
    cradio_out = None
    if C_cradio > 0:
        paths["cradio"] = str(prefix) + "_cradio.f32"
        paths["cradio_shape"] = (T_all, C_cradio, h64, w64)
        cradio_out = _writable_map(temp_budget, _resources, np.memmap, paths["cradio"], dtype=np.float32, mode="w+",
                               shape=paths["cradio_shape"])
    print(f"    frozen VFM prepass: T={T_all}, grid={w64}x{h64}, chunk={chunk}")
    if reader_kind == "video":
        chunks = iter_video_frame_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    else:
        chunks = iter_frame_dir_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    chunks = _resources.own(_checked_chunks(chunks, T_all))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            control.check()
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            ff = feature_cache.window(display, t0, device=device,
                                      channels={"moge": C_moge, "cradio": C_cradio},
                                      grid_hw=(h64, w64), extract=extract)
            moge_out[t0:t0 + Tc] = ff["moge"].float().cpu().numpy()
            if cradio_out is not None:
                cradio_out[t0:t0 + Tc] = ff["cradio"].float().cpu().numpy()
            del ff, display
            control.checkpoint("foundation_prepass", t0 + Tc, T_all)
    moge_out.flush()
    del moge_out
    if cradio_out is not None:
        cradio_out.flush()
        del cradio_out
    torch.cuda.empty_cache()
    if diagnostics is not None:
        diagnostics.foundation_cache = {**feature_cache.snapshot(),
                                       "mode": "low_vram_disk_prepass",
                                       "durable_enabled": durable_foundations is not None}
    return paths


def _phase1_setup(model, *, stage1_on_gpu: bool = True):
    model.foundations.cuda()
    # Low-VRAM mode precomputes foundation features first, so stage 1 waits on CPU.
    if stage1_on_gpu:
        model.stage1.cuda()
    else:
        model.stage1.cpu()
    model.stage2.cpu()
    torch.cuda.empty_cache()


def _phase2_setup(model):
    model.foundations.cpu()
    model.stage1.cpu()
    torch.cuda.empty_cache()
    model.stage2.cuda()
    torch.cuda.synchronize()


@torch.no_grad()
@scoped_resources
def hann_stage1_stream(model, source, meta, *, reader_cls, reader_kind: str,
                       chunk: int, stride: int, baton_path: Path,
                       initial_hint, initial_hint_quality: float,
                       full_hint_video_path: str | None,
                       full_hint_quality: float,
                       rvm_path: Path | None,
                       foundation_paths,
                       carry_hint_from_aux: bool,
                       carry_hint_quality: float,
                       on_baton_frames, foundation_cache_bytes=0, diagnostics=None,
                       anchor_plan=None, context_radius=2, causal_context=False,
                       correction_rasters=None, segments=None, temp_budget=None, _resources=None):
    diagnostics = diagnostics if diagnostics is not None else RunDiagnostics()
    feature_cache = FoundationWindowCache(foundation_cache_bytes, persistent=diagnostics.durable_foundations) if foundation_paths is None else None
    device = next(model.stage1.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    d_16 = model.stage1.d_16
    h16, w16 = H // 16, W // 16
    targets = set(anchor_plan.anchors) if anchor_plan is not None else None
    if anchor_plan is None:
        windows = [(first + t, first + min(t + chunk, last - first))
                   for first, last in (segments or ((0, T_all),))
                   for t in window_starts(last - first, chunk, stride)]
    else:
        windows = sorted({(context.start, context.stop)
                          for t in anchor_plan.anchors
                          for context in (anchor_plan.context(t, context_radius, past_only=causal_context),)})
        if any(stop - start > chunk for start, stop in windows):
            raise ValueError("Quick Key context exceeds hann_chunk; increase the chunk or reduce quick_context_radius")
    starts = [first for first, _ in windows]
    durable_hann = diagnostics.durable_hann
    if durable_hann is not None and len(getattr(model.stage1, "recurrent_64", ())) > 0:
        raise ValueError("durable Hann replay requires disabled Stage1 recurrent sites")
    schedule = {"windows": windows, "chunk": chunk, "stride": stride,
                "targets": sorted(targets) if targets is not None else None,
                "causal_context": causal_context, "streaming": on_baton_frames is not None,
                "context_radius": context_radius, "segments": segments}
    overlap = max(0, chunk - stride)
    baton_fp16 = None
    if on_baton_frames is None:
        baton_fp16 = _writable_map(temp_budget, _resources, np.memmap, baton_path, dtype=np.float16, mode="w+",
                               shape=(T_all, d_16, h16, w16))
    live_acc = {}
    live_w = {}
    print(f"    Hann stage1: chunk={chunk}, stride={stride}, "
          f"windows={len(starts)}, baton_grid={w16}x{h16}")

    rvm_map = None
    if rvm_path is not None:
        D = model.foundations.rvm.embed_dim
        rvm_map = _resources.mapping(np.memmap, rvm_path, dtype=np.float16, mode="r",
                            shape=(T_all, D, H // 64, W // 64))
    cradio_map = moge_map = None
    if foundation_paths is not None:
        moge_map = _resources.mapping(np.memmap, foundation_paths["moge"], dtype=np.float32,
                             mode="r", shape=tuple(foundation_paths["moge_shape"]))
        if model.stage1.cradio_dim > 0:
            cradio_map = _resources.mapping(np.memmap, foundation_paths["cradio"], dtype=np.float32,
                                   mode="r", shape=tuple(foundation_paths["cradio_shape"]))

    reader = reader_cls(str(source), meta)
    try:
        hint_reader = make_hint_reader(full_hint_video_path, meta)
    except BaseException as exc:
        cleanup_all([reader.close], primary_error=exc)
        raise
    carry_cache = {}
    stage2_frame_cache = {}

    def flush_before(t_limit: int):
        ready = []
        for k in sorted(k for k in live_acc if k < t_limit):
            frame = (live_acc.pop(k) / max(live_w.pop(k), 1e-6)).astype(np.float16)
            if on_baton_frames is None:
                baton_fp16[k] = frame
            else:
                model_frame = stage2_frame_cache.pop(k)
                ready.append((k, frame, model_frame))
        if ready:
            on_baton_frames(ready)

    try:
        previous_segment = None
        for i, (t0, t1) in enumerate(windows):
            diagnostics.control.checkpoint("stage1_windows", i, len(starts))
            if anchor_plan is not None:
                segment = anchor_plan.segment(t0)
            elif segments is not None:
                segment = next(s for s in segments if s[0] <= t0 < s[1])
            else:
                segment = (0, T_all)
            if anchor_plan is not None or segments is not None:
                if segment != previous_segment:
                    carry_cache.clear()
                previous_segment = segment
            next_start = starts[i + 1] if i + 1 < len(starts) else T_all
            Tc = t1 - t0
            model_chunk, display_chunk, review_chunk = reader.window(t0, t1)
            hint_cpu = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16)
            hint_quality_cpu = torch.zeros(1, Tc, dtype=torch.bfloat16)
            hint_present = torch.zeros(1, Tc, dtype=torch.bool)
            if hint_reader is not None:
                hint_cpu[0] = hint_reader.window(t0, t1)
                hint_quality_cpu.fill_(float(full_hint_quality))
                hint_present.fill_(True)
            for j, t_abs in enumerate(range(t0, t1)):
                if t_abs in carry_cache:
                    hint_cpu[0, j] = carry_cache[t_abs]
                    hint_quality_cpu[0, j] = float(carry_hint_quality)
                    hint_present[0, j] = True
            if initial_hint is not None and t0 == 0:
                hint_cpu[0, 0] = initial_hint.to(dtype=hint_cpu.dtype)
                hint_quality_cpu[0, 0] = float(initial_hint_quality)
                hint_present[0, 0] = True
            if correction_rasters is not None:
                for j, frame in enumerate(range(t0, t1)):
                    if not correction_rasters.has_condition(frame):
                        continue
                    base = (_crop_native_torch(hint_cpu[0, j:j + 1].float(), meta)[0, 0].numpy()
                            if hint_present[0, j] else None)
                    if base is not None and meta.get('spatial_draft'):
                        base = resize_coverage(base, meta['output_hw'])
                    hint, quality = correction_rasters.condition(frame, base)
                    if hint is not None:
                        if meta.get('spatial_draft'):
                            hint = resize_coverage(hint, meta['model_native_hw'], down=True)
                        padded, _ = center_pad_to_shape(hint, H, W)
                        hint_cpu[0, j, 0] = torch.from_numpy(padded).to(hint_cpu.dtype)
                        hint_present[0, j] = True
                        if quality is not None:
                            hint_quality_cpu[0, j] = float(initial_hint_quality) * quality
                        if diagnostics.corrections is not None:
                            diagnostics.corrections["hint_frame_evaluations_conditioned"] += 1
            if not bool(hint_present.any().item()):
                hint_quality_cpu.fill_(-1.0)

            if on_baton_frames is not None:
                for j, t_abs in enumerate(range(t0, t1)):
                    if (targets is None or t_abs in targets) and (not causal_context or t_abs == t1 - 1):
                        stage2_frame_cache[t_abs] = model_chunk[j]

            rgb_c = model_chunk.unsqueeze(0).to(device, non_blocking=True)
            hint_c = hint_cpu.to(device, non_blocking=True)
            hint_quality_c = hint_quality_cpu.to(device, non_blocking=True)
            ff = None
            if foundation_paths is None:
                def extract_foundations(display):
                    diagnostics.control.check()
                    n = display.shape[0]
                    rgb_found = display.unsqueeze(0).to(device, non_blocking=True)
                    rgb_radio, rgb_moge = _build_foundation_inputs(rgb_found)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        raw = model.foundations(
                            rgb_cradio=rgb_radio.reshape(n, 3, *rgb_radio.shape[-2:]),
                            rgb_moge=rgb_moge.reshape(n, 3, *rgb_moge.shape[-2:]),
                        )
                    required = ("moge", "cradio") if model.stage1.cradio_dim > 0 else ("moge",)
                    if any(name not in raw for name in required):
                        raise RuntimeError(f"foundation extractor did not return required features: {required}")
                    return {name: raw[name] for name in required}

                with diagnostics.stage("iid_foundations"):
                    ff = feature_cache.window(
                        display_chunk, t0, device=device,
                        channels={"moge": model.stage1.moge_dim, "cradio": model.stage1.cradio_dim},
                        grid_hw=(H // 64, W // 64), extract=extract_foundations, retain_from=next_start)
                cradio_feat = ff["cradio"].reshape(1, Tc, -1, H // 64, W // 64) if "cradio" in ff else None
                moge_feat = ff["moge"].reshape(1, Tc, -1, H // 64, W // 64)
            else:
                moge_feat = torch.from_numpy(np.asarray(moge_map[t0:t1]).copy())
                moge_feat = moge_feat.unsqueeze(0).to(device, non_blocking=True).float()
                cradio_feat = None
                if cradio_map is not None:
                    cradio_feat = torch.from_numpy(np.asarray(cradio_map[t0:t1]).copy())
                    cradio_feat = cradio_feat.unsqueeze(0).to(device, non_blocking=True).float()
            rvm_feat = None
            if rvm_map is not None:
                rvm_feat = torch.from_numpy(np.asarray(rvm_map[t0:t1]).copy())
                rvm_feat = rvm_feat.unsqueeze(0).to(device, non_blocking=True).float()
            identity = None
            s1 = None
            if durable_hann is not None:
                identity = durable_hann.identity(
                    schedule=schedule,
                    incoming_replay_sha256=replay_state_digest(
                        carry=carry_cache, accumulators=live_acc, weights=live_w,
                        pending_frames=stage2_frame_cache, previous_segment=previous_segment,
                        check=diagnostics.control.check),
                    index=i, start=t0, stop=t1, next_start=next_start, segment=segment,
                    carry=carry_hint_from_aux, d_16=d_16,
                    inputs={"rgb": rgb_c, "hint": hint_c, "hint_quality": hint_quality_c,
                            "moge": moge_feat, "cradio": cradio_feat, "rvm": rvm_feat})
                s1 = durable_hann.get(identity)
            if s1 is None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s1 = model.stage1(
                        rgb_c, hint_c, moge_feat, cradio_feat=cradio_feat,
                        rvm_feat=rvm_feat, t_stride=1.0,
                        hint_quality=hint_quality_c,
                    )
                if diagnostics.execution is not None:
                    diagnostics.execution["stage1_frame_evaluations"] = diagnostics.execution.get("stage1_frame_evaluations", 0) + Tc
                    diagnostics.execution["stage1_windows_evaluated"] = diagnostics.execution.get("stage1_windows_evaluated", 0) + 1
                if durable_hann is not None:
                    durable_hann.put(identity, s1)
            else:
                s1 = {name: value.to(device) for name, value in s1.items()}
            baton_gpu = s1["baton"][0].float()
            if carry_hint_from_aux:
                aux_alpha = s1["aux"][0, :, 0:1].float()
                aux_full = F.interpolate(
                    aux_alpha, size=(H, W), mode="bilinear",
                    align_corners=False).clamp(0.0, 1.0)
                if correction_rasters is not None:
                    top, left = meta["pad"][2], meta["pad"][0]
                    height, width = meta["model_native_hw"]
                    for j, frame in enumerate(range(t0, t1)):
                        if correction_rasters.has_effect(frame, temporal=True):
                            native = aux_full[j, 0, top:top + height, left:left + width].float().cpu().numpy()
                            if meta.get('spatial_draft'):
                                native = resize_coverage(native, meta['output_hw'])
                            refined = correction_rasters.apply(native, frame, temporal=True)
                            if meta.get('spatial_draft'):
                                refined = resize_coverage(refined, meta['model_native_hw'], down=True)
                            aux_full[j, 0, top:top + height, left:left + width] = torch.from_numpy(refined).to(device)
                            if diagnostics.corrections is not None:
                                diagnostics.corrections["aux_frame_evaluations_refined"] += 1
                aux_cpu = aux_full.to("cpu", dtype=torch.bfloat16)
                for j, t_abs in enumerate(range(t0, t1)):
                    carry_cache[t_abs] = aux_cpu[j].clone()
                for old_t in [k for k in carry_cache if k < next_start]:
                    del carry_cache[old_t]
                del aux_alpha, aux_full, aux_cpu
            w = taper_weights(Tc, overlap, is_first=(i == 0),
                              is_last=(i == len(starts) - 1),
                              device="cpu", dtype=torch.float32)
            weighted = (baton_gpu * w.to(device).view(Tc, 1, 1, 1)).to("cpu").numpy()
            w_np = w.numpy()
            for j, t_abs in enumerate(range(t0, t1)):
                if (targets is not None and t_abs not in targets) or (causal_context and t_abs != t1 - 1):
                    continue
                if t_abs not in live_acc:
                    live_acc[t_abs] = weighted[j].copy()
                    live_w[t_abs] = float(w_np[j])
                else:
                    live_acc[t_abs] += weighted[j]
                    live_w[t_abs] += float(w_np[j])
            del s1, baton_gpu, weighted, ff, cradio_feat, moge_feat, rvm_feat
            del rgb_c, hint_c, hint_quality_c, hint_cpu, hint_quality_cpu
            del model_chunk, display_chunk, review_chunk
            flush_before(next_start)
            if feature_cache is not None:
                feature_cache.cache.discard_before(next_start)
            diagnostics.control.checkpoint("stage1_windows", i + 1, len(starts))
            print(f"      window {i + 1}/{len(starts)} [{t0}:{t1}]", flush=True)
        torch.cuda.synchronize()
        flush_before(T_all + 1)
    finally:
        if feature_cache is not None:
            feature_cache.close()
            diagnostics.foundation_cache = feature_cache.snapshot()
        cleanup_all([reader.close] + ([hint_reader.close] if hint_reader else []),
                    primary_error=sys.exception())

    if baton_fp16 is not None:
        baton_fp16.flush()
    del baton_fp16, live_acc, live_w, rvm_map, cradio_map, moge_map
    torch.cuda.empty_cache()


def _crop_native_torch(x: torch.Tensor, meta: dict) -> torch.Tensor:
    h, w = meta["model_native_hw"]
    pad_l, _, pad_t, _ = meta["pad"]
    return x[..., pad_t:pad_t + h, pad_l:pad_l + w]


def _resize_output_torch(x: torch.Tensor, meta: dict) -> torch.Tensor:
    if tuple(meta['model_native_hw']) == tuple(meta['output_hw']) or tuple(x.shape[-2:]) == tuple(meta['output_hw']):
        return x
    return _require_finite(F.interpolate(x.float(), size=meta['output_hw'], mode='bilinear', align_corners=False),
                           'native spatial-draft resampling')


def _native_alpha_override(native_alpha, logits, meta):
    if tuple(native_alpha.shape) != (logits.shape[0], 1, *meta['output_hw']):
        raise ValueError('Native alpha override must match prediction batch and output geometry.')
    return _require_finite(native_alpha, 'native alpha override').clamp(0, 1)


def _linear_adjust_inverse_torch(linear: torch.Tensor, meta: dict) -> torch.Tensor:
    gain = float(meta.get("linear_brightness", 1.0))
    contrast = float(meta.get("linear_contrast", 1.0))
    if abs(gain - 1.0) <= 1e-6 and abs(contrast - 1.0) <= 1e-6:
        return linear
    pivot = float(meta.get("linear_contrast_pivot", 0.18))
    return (linear / gain - pivot) / contrast + pivot


def _despill_average_srgb_torch(fg: torch.Tensor, strength: float = 0.5, *,
                               screen="green", preserve_luma=False, alpha=None, edges_only=False) -> torch.Tensor:
    if strength <= 0.0:
        return fg
    work = fg[:, [0, 2, 1]] if screen == "blue" else fg
    r = work[:, 0:1]
    g = work[:, 1:2]
    b = work[:, 2:3]
    channel_sum = _require_finite(r + b, "despill channel sum")
    spill = (g - channel_sum * 0.5).clamp(min=0.0)
    full = torch.cat([r + spill * 0.5, g - spill, b + spill * 0.5], dim=1)
    if screen == "blue":
        full = full[:, [0, 2, 1]]
    if preserve_luma:
        weights = fg.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        full = full + ((fg - full) * weights).sum(dim=1, keepdim=True)
    if edges_only:
        if alpha is None:
            raise ValueError("edge-weighted despill requires coverage")
        weight = strength * 4 * alpha.clamp(0, 1) * (1 - alpha.clamp(0, 1))
        return fg * (1 - weight) + full * weight
    if strength >= 1.0:
        return full
    return fg * (1.0 - strength) + full * strength


def _despill_configured(fg, alpha, meta):
    result = _despill_average_srgb_torch(
        fg, float(meta.get("despill_strength", 0.5)), screen=meta.get("despill_screen", "green"),
        preserve_luma=meta.get("despill_preserve_luma", False), alpha=alpha,
        edges_only=meta.get("despill_edges_only", False))
    return _require_finite(result, "despilled foreground")


def _require_finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values; processing rejected")
    return value


def _srgb_to_bgr_uint16_cpu(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0.0, 1.0)
    x = x[:, [2, 1, 0]].permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return (x * 65535.0).astype(np.uint16)


def _srgb_to_bgra_uint16_cpu(rgb: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    rgb = rgb.clamp(0.0, 1.0)
    alpha = alpha.clamp(0.0, 1.0)
    bgra = torch.cat([rgb[:, [2, 1, 0]], alpha], dim=1)
    bgra = bgra.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return (bgra * 65535.0).astype(np.uint16)


def _chw_to_bgr_f32_cpu(x: torch.Tensor) -> np.ndarray:
    return (x[:, [2, 1, 0]].permute(0, 2, 3, 1)
            .contiguous().cpu().numpy().astype(np.float32, copy=False))


def _chw_to_bgra_f32_cpu(rgb: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    bgra = torch.cat([rgb[:, [2, 1, 0]], alpha], dim=1)
    return (bgra.permute(0, 2, 3, 1)
            .contiguous().cpu().numpy().astype(np.float32, copy=False))


def _render_predictions_exr_torch(meta: dict, outputs: set[str],
                                  checker: torch.Tensor | None,
                                  logits: torch.Tensor, *, native_alpha=None) -> dict[str, np.ndarray]:
    """Render a stage-2 batch as unclamped scene-linear float32 NHWC arrays.

    HDR contract: `fg`, `cutout`, and `checker` color stays in decoded linear
    light with no [0, 1] clamp and no quantization, so values above 1.0 (and
    negative values the model may emit) survive into the EXR files. Alpha is a
    matte and stays clamped to [0, 1].
    """
    _require_finite(logits, "model output")
    alpha_model = logits[:, 0:1].float()
    pa_model = (_crop_native_torch(alpha_model, meta) if meta.get('spatial_draft') else
                _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0))
    despill_strength = float(meta.get("despill_strength", 0.5))

    pa = (_native_alpha_override(native_alpha, logits, meta) if native_alpha is not None
          else _resize_output_torch(pa_model, meta).clamp(0.0, 1.0))
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        rendered["alpha"] = (pa.permute(0, 2, 3, 1)
                             .contiguous().cpu().numpy()
                             .astype(np.float32, copy=False))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    _require_finite(pf_linear, "decoded scene-linear foreground")
    pf_linear = _resize_output_torch(pf_linear, meta)
    if "fg" in outputs:
        rendered["fg"] = _chw_to_bgr_f32_cpu(pf_linear)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_configured(pf_linear, pa, meta)
        cutout = pf_linear_despilled * pa if meta.get("cutout_alpha_mode") == "premultiplied" else pf_linear_despilled
        rendered["cutout"] = _chw_to_bgra_f32_cpu(cutout, pa)
    if "checker" in outputs:
        assert checker is not None
        checker_b = checker.to(device=pf_linear.device,
                               dtype=pf_linear.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        pf_despill = _despill_configured(pf_linear, pa, meta)
        comp_lin = pf_despill * pa + bg_lin * (1.0 - pa)
        rendered["checker"] = _chw_to_bgr_f32_cpu(comp_lin)
    return rendered


def _render_predictions_torch(meta: dict, outputs: set[str],
                              checker: torch.Tensor | None,
                              logits: torch.Tensor, *, native_alpha=None) -> dict[str, np.ndarray]:
    """Render a stage-2 batch on GPU, returning uint16 BGR/BGRA NHWC arrays.

    This removes the previous per-frame numpy HDR decode, sRGB encode,
    composition, channel shuffle, and uint8 conversion hot path.
    """
    _require_finite(logits, "model output")
    alpha_model = logits[:, 0:1].float()
    pa_model = (_crop_native_torch(alpha_model, meta) if meta.get('spatial_draft') else
                _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0))
    despill_strength = float(meta.get("despill_strength", 0.5))

    pa = (_native_alpha_override(native_alpha, logits, meta) if native_alpha is not None
          else _resize_output_torch(pa_model, meta).clamp(0.0, 1.0))
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        rendered["alpha"] = _srgb_to_bgr_uint16_cpu(pa.expand(-1, 3, -1, -1))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    _require_finite(pf_linear, "decoded scene-linear foreground")
    if meta.get('spatial_draft'):
        pf_linear = _resize_output_torch(pf_linear, meta)
    pf_linear_pos = pf_linear.clamp(min=0.0)
    pf = _resize_output_torch(cu.linear_to_srgb(pf_linear_pos), meta).clamp(0.0, 1.0)
    if "fg" in outputs:
        rendered["fg"] = _srgb_to_bgr_uint16_cpu(pf)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_configured(pf_linear_pos, pa, meta)
        if meta.get("cutout_linear", False):
            pf_linear_despilled_resized = _resize_output_torch(pf_linear_despilled, meta).clamp(0.0, 1.0)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_linear_despilled_resized, pa)
        elif meta.get("despill_space") == "linear":
            pf_despill = cu.linear_to_srgb(pf_linear_despilled.clamp(min=0)).clamp(0, 1)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_despill, pa)
        else:
            pf_despill = _despill_configured(pf, pa, meta)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_despill, pa)
    if "checker" in outputs:
        assert checker is not None
        checker_b = checker.to(device=pf.device, dtype=pf.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        alpha = pa
        foreground = (_despill_configured(pf_linear_pos, pa, meta) if meta.get("despill_space") == "linear"
                      else cu.srgb_to_linear(_despill_configured(pf, pa, meta)))
        comp_despill_lin = foreground * alpha + bg_lin * (1.0 - alpha)
        comp_despill = cu.linear_to_srgb(
            comp_despill_lin.clamp(min=0.0)).clamp(0.0, 1.0)
        rendered["checker"] = _srgb_to_bgr_uint16_cpu(comp_despill)
    return rendered


def _render_source_draft(frame, meta, outputs, checker, *, use_exr):
    """Shared colour/export helpers, with current source foreground and draft alpha."""
    linear = torch.from_numpy(frame.rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
    alpha = torch.from_numpy(frame.alpha.copy()).unsqueeze(0).unsqueeze(0)
    _require_finite(linear, "draft source")
    _require_finite(alpha, "draft coverage")
    result = {}
    strength = float(meta.get("despill_strength", 0.5))
    if "alpha" in outputs:
        result["alpha"] = (alpha.permute(0, 2, 3, 1).numpy() if use_exr
                           else _srgb_to_bgr_uint16_cpu(alpha.expand(-1, 3, -1, -1)))
    if use_exr:
        if "fg" in outputs:
            result["fg"] = _chw_to_bgr_f32_cpu(linear)
        if "cutout" in outputs or "checker" in outputs:
            despilled = _despill_configured(linear, alpha, meta)
            if "cutout" in outputs:
                cutout = despilled * alpha if meta.get("cutout_alpha_mode") == "premultiplied" else despilled
                result["cutout"] = _chw_to_bgra_f32_cpu(cutout, alpha)
            if "checker" in outputs:
                bg = cu.srgb_to_linear(checker.unsqueeze(0))
                result["checker"] = _chw_to_bgr_f32_cpu(despilled * alpha + bg * (1 - alpha))
    else:
        display = cu.linear_to_srgb(linear.clamp(min=0)).clamp(0, 1)
        if "fg" in outputs:
            result["fg"] = _srgb_to_bgr_uint16_cpu(display)
        if "cutout" in outputs:
            linear_operation = meta.get("cutout_linear") or meta.get("despill_space") == "linear"
            rgb = _despill_configured(linear.clamp(min=0) if linear_operation else display, alpha, meta)
            if linear_operation and not meta.get("cutout_linear"):
                rgb = cu.linear_to_srgb(rgb.clamp(min=0)).clamp(0, 1)
            result["cutout"] = _srgb_to_bgra_uint16_cpu(rgb, alpha)
        if "checker" in outputs:
            foreground = (_despill_configured(linear.clamp(min=0), alpha, meta) if meta.get("despill_space") == "linear"
                          else cu.srgb_to_linear(_despill_configured(display, alpha, meta)))
            bg = cu.srgb_to_linear(checker.unsqueeze(0))
            result["checker"] = _srgb_to_bgr_uint16_cpu(cu.linear_to_srgb((foreground * alpha + bg * (1 - alpha)).clamp(min=0)))
    return result


def run_inference(model, input_path: str, output_dir: str, cfg: InferenceConfig,
                  writer_cfg: WriterConfig, *, initial_hint_path: str | None,
                  hint_video_path: str | None, diagnostics: RunDiagnostics | None = None):
    diagnostics = diagnostics if diagnostics is not None else RunDiagnostics()
    diagnostics.control.checkpoint("probe", 0, 1)
    cfg.validate()
    writer_cfg.validate()
    # Automatic memory selection must not mutate a reusable caller's settings.
    cfg = replace(cfg, outputs=set(cfg.outputs), temp_dir=Path(cfg.temp_dir))
    input_path = str(input_path)
    output_dir = str(output_dir)
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    if os.path.isdir(input_path):
        if cfg.start_mode != "begin":
            raise ValueError("frame-directory inputs currently support start_mode=begin only; select a directory containing the desired range")
        if (cfg.linear_brightness, cfg.linear_contrast, cfg.linear_contrast_pivot) != (1.0, 1.0, 0.18):
            raise ValueError("linear brightness/contrast adjustments are currently supported for video inputs only")
    if cfg.source_colour_profile != 'legacy' and not os.path.isdir(input_path):
        raise ValueError('Declared source primaries require a scene-linear frame directory; video is unsupported')
    if diagnostics.durable_writer_prefix is not None and cfg.output_format != 'exr':
        raise ValueError('Durable writer-prefix recovery requires all outputs to be EXR')
    from .output_paths import output_layout, validate_output_paths

    correction_document = diagnostics.correction_document
    if cfg.corrections_path is not None and correction_document is None:
        from .corrections import load_corrections

        correction_document = load_corrections(cfg.corrections_path, source=input_path,
                                              check=diagnostics.control.check)
        diagnostics.correction_document = correction_document

    def protect_inputs():
        validate_output_paths(input_path, output_dir, cfg,
                              initial_hint_path=initial_hint_path, hint_video_path=hint_video_path,
                              correction_document=correction_document, check=diagnostics.control.check)

    protect_inputs()
    planned_layout = output_layout(input_path, cfg)
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    is_dir = os.path.isdir(input_path)
    if cfg.source_witness and not is_dir:
        raise ValueError('source_witness requires a scene-linear RGB frame directory, not video.')
    if not is_dir and cfg.frame_dir_alpha != "ignore":
        raise ValueError("premultiplied source decoding requires a scene-linear frame directory")
    reader_kind = "frame_dir" if is_dir else "video"
    reader_cls = partial(SequentialFrameDirWindowReader if is_dir else SequentialVideoWindowReader,
                         include_review=False)
    if is_dir:
        meta = probe_frame_dir_meta(
            input_path, num_frames=cfg.num_frames, start_frame=cfg.start_frame,
            shape_quantum=cfg.shape_quantum, fps=cfg.frame_dir_fps,
            exr_linear=cfg.frame_dir_linear, model_scale=cfg.model_scale, alpha_mode=cfg.frame_dir_alpha,
            source_colour_profile=cfg.source_colour_profile)
        if (not cfg.frame_dir_linear
                and meta["frame_paths"]
                and meta["frame_paths"][0].lower().endswith(".exr")):
            print("  [WARNING] EXR input frames are being treated as "
                  "display-coded and clipped to [0, 1]. For scene-linear HDR "
                  "EXR plates, pass --frame_dir_linear.")
    else:
        meta = probe_video_meta(
            input_path, num_frames=cfg.num_frames, start_frame=cfg.start_frame,
            shape_quantum=cfg.shape_quantum, native_resolution=True,
            start_mode=cfg.start_mode, seed=cfg.seed,
            model_scale=cfg.model_scale,
            linear_brightness=cfg.linear_brightness,
            linear_contrast=cfg.linear_contrast,
            linear_contrast_pivot=cfg.linear_contrast_pivot)
    spatial = bool(meta.get('spatial_draft'))
    if spatial:
        diagnostics.warnings.append('Spatial draft: reduced model input, native-size resampled exports; quality/speed not validated')
        if not meta['spatial_draft']['model_grid_reduced']:
            diagnostics.warnings.append('Spatial draft padding retains the native padded model grid; no grid-size saving')
    diagnostics.control.checkpoint("probe", 1, 1)
    T_all = int(meta["num_frames"])
    if T_all <= 0:
        raise ValueError("source contains no readable frames")
    witness = None
    if cfg.source_witness:
        from .source_witness import SourceWitness

        witness = SourceWitness(meta, diagnostics.control)
        meta['_source_witness'] = witness
    H, W = meta["shape_hw"]
    correction_document = diagnostics.correction_document
    correction_rasters = None
    correction_segments = None
    correction_cuts, correction_resets = [], []
    if cfg.corrections_path is not None:
        from .corrections import load_corrections
        from .correction_rasters import CorrectionRasters
        if correction_document is None:
            correction_document = load_corrections(cfg.corrections_path, source=input_path,
                                                  check=diagnostics.control.check)
            diagnostics.correction_document = correction_document
        correction_rasters = CorrectionRasters(correction_document, meta, check=diagnostics.control.check)
        authored = correction_document.to_dict()
        correction_cuts = [shot[0] for shot in authored["shots"] if shot[0] > 0]
        correction_resets = list(correction_document.resets)
        boundaries = sorted({0, T_all} | set(correction_cuts) | set(correction_resets))
        correction_segments = tuple(zip(boundaries, boundaries[1:]))
        from .correction_reporting import correction_report
        diagnostics.corrections = correction_report(correction_document, cfg.corrections_path, runtime_fields={
                                   "hint_frame_evaluations_conditioned": 0, "aux_frame_evaluations_refined": 0,
                                   "output_frames_refined": 0, "refinement_dtype": "float32",
                                   "propagation_anchor_evaluations_refined": 0,
                                   "directional_range_mode": "tracked_final_constraints_and_authored_static_ranges" if authored.get("tracks") else "static_raster_direct_constraints",
                                   "model_conditioning": "both_direction_authored_hints_and_auxiliary_carry_only",
                                   "output_cache_reuse": "none; every job recomputes"})
    if cfg.auto_vram:
        diagnostics.memory_policy = apply_auto_vram(cfg, (H, W), T_all)
        if cfg.low_vram and cfg.foundation_cache_mb:
            diagnostics.warnings.append("automatic selection chose low_vram: existing disk foundation prepass replaces optional RAM overlap cache")
            cfg.foundation_cache_mb = 0
    if cfg.low_vram:
        diagnostics.foundation_cache = {"mode": "low_vram_disk_prepass", "ram_budget_bytes": 0,
                                        "foundation_frames_evaluated": None, "hit_rate": None}
    H_out, W_out = meta["output_hw"]
    analysis = None
    effective_cuts = sorted(set(cfg.quick_cut_frames) | set(correction_cuts))
    if cfg.quick_key and cfg.quick_interval > 1 and cfg.quick_auto_cuts:
        from .shot_analysis import analyze_shots, ShotSettings
        protect_inputs()
        trace_path = Path(output_dir) / f"{raw_clip_name(input_path)}_shot_analysis.jsonl"
        analysis = {"status": "running", "frames_analyzed": 0, "trace_path": str(trace_path)}
        diagnostics.quick_key = {"requested": True, "active": True, "analysis": analysis}
        try:
            with ResourceScope() as resources, diagnostics.stage("shot_analysis", gpu=False):
                trace = resources.own(trace_path.open("w", encoding="utf-8"))
                source = resources.own(DraftSourceReader(input_path, meta, is_dir=is_dir))
                def emit(record):
                    analysis["frames_analyzed"] = record["frame"] + 1
                    trace.write(json.dumps(record, allow_nan=False) + "\n")
                analysis.update(analyze_shots(T_all, source.read,
                                             settings=ShotSettings(max_side=cfg.quick_analysis_max_side,
                                                                   cut_threshold=cfg.quick_cut_threshold),
                                             control=diagnostics.control, emit=emit))
                analysis["source_frames_decoded"] = source.frames_decoded
            analysis["status"] = "complete"
        except BaseException:
            analysis["status"] = "interrupted"
            raise
        effective_cuts = sorted(set(effective_cuts) | set(analysis["detected_cuts"]))
    if correction_document is not None:
        from .correction_tracking_context import tracking_cuts, tracking_decode
        correction_document.validate_track_context(
            cuts=tracking_cuts(cfg), decode=tracking_decode(cfg, input_path))
        correction_document.validate_guidance_context(decode=tracking_decode(cfg, input_path))
        for track in correction_document.to_dict().get('tracks', []):
            if any(track['range'][0] < cut < track['range'][1] for cut in effective_cuts):
                raise ValueError('Tracked correction crosses an effective cut; regenerate with another anchor/range.')
        for entry in correction_document.to_dict()["entries"]:
            if entry["kind"] != "reset" and any(entry["range"][0] < cut < entry["range"][1] for cut in effective_cuts):
                raise ValueError("correction raster range crosses an effective cut; revise shot/range metadata or the heuristic cut policy")
    quick_plan = (plan_anchors(T_all, cfg.quick_interval, cuts=effective_cuts,
                              resets=sorted(set(cfg.quick_reset_frames) | set(correction_resets)), max_gap=cfg.quick_max_gap,
                              corrections=correction_document.frames if correction_document is not None else (),
                              include_final=cfg.quick_approach != "causal")
                  if cfg.quick_key and cfg.quick_interval > 1 else None)
    if quick_plan is not None:
        if max(len(quick_plan.context(t, cfg.quick_context_radius, past_only=cfg.quick_approach == "causal"))
               for t in quick_plan.anchors) > cfg.hann_chunk:
            raise ValueError("Quick Key context exceeds hann_chunk; increase it or reduce quick_context_radius")
        diagnostics.quick_key = {"requested": True, "active": True, "interval": cfg.quick_interval,
                                 "context_radius": cfg.quick_context_radius,
                                 "context_strategy": ("past_native_context_owned_anchor" if cfg.quick_approach == "causal"
                                                      else "native_contiguous_windows_with_overlapping_baton_blend"),
                                 "fallback_policy": cfg.quick_fallback, "foundation_passes": []}
        diagnostics.quick_key.update({"analysis": analysis, "supplied_cuts": list(cfg.quick_cut_frames),
                                      "effective_cuts": effective_cuts, "approach": cfg.quick_approach,
                                      "native_context_policy": "past_only_owned_anchor" if cfg.quick_approach == "causal" else "symmetric_context_with_overlap_blend",
                                      "whole_clip_precompute": True})
        diagnostics.warnings.append("Quick Key SOURCE-RGB DRAFT: reconstructed foreground is not used; whole-clip precompute is offline; reliability is uncalibrated")
    elif cfg.quick_key:
        diagnostics.quick_key = {"requested": True, "active": False, "interval": 1,
                                 "reason": "original full-rate inference path"}
    meta["cutout_linear"] = cfg.cutout_linear
    meta["cutout_alpha_mode"] = cfg.cutout_alpha_mode
    meta["despill_strength"] = cfg.despill_strength
    meta.update(despill_screen=cfg.despill_screen, despill_space=cfg.despill_space,
                despill_preserve_luma=cfg.despill_preserve_luma, despill_edges_only=cfg.despill_edges_only)
    refine_active = cfg.alpha_refine_radius > 0 and cfg.alpha_refine_strength > 0
    finish_active = cfg.alpha_expand_radius != 0 or cfg.alpha_feather_radius > 0
    edge_finishing: dict[str, Any] = {
        'active': finish_active, 'expand_radius': cfg.alpha_expand_radius,
        'feather_radius': cfg.alpha_feather_radius, 'units': 'native output pixels',
        'order': 'spatial refinement, expansion/contraction, feather, trimap/authored constraints, render/despill/association',
        'scope': 'final output only; never propagation anchors or carried hints',
        'frames_evaluated': 0, 'frames_changed': 0, 'seconds': 0.0}
    diagnostics.alpha_refinement = {"active": refine_active, "radius": cfg.alpha_refine_radius,
                                    "strength": cfg.alpha_refine_strength, "sigma": cfg.alpha_refine_sigma,
                                    "frames_evaluated": 0, "frames_changed": 0, "seconds": 0.0,
                                    "guidance": "current source SDR proxy", "hard_endpoints_locked": True,
                                    "quality_improvement": "unvalidated", "before_authored_constraints": True,
                                    "edge_finishing": edge_finishing}
    fps = float(meta["native_fps"])
    diagnostics.effective = json_value(cfg)
    diagnostics.source = {k: json_value(v) for k, v in meta.items() if k != "frame_paths" and not k.startswith('_')}
    diagnostics.execution = {
        "backend": "pytorch", "requested_device": "cuda", "autocast_dtype": "bfloat16",
        "rvm_cache_dtype": "float16", "foundation_cache_dtype": "float32" if cfg.low_vram or cfg.foundation_cache_mb else None,
        "foundation_extraction_policy": "disk_prepass" if cfg.low_vram else ("missing_iid_frames" if cfg.foundation_cache_mb else "full_windows"),
        "baton_dtype": "float16", "cached_playback": False,
        "source_colour": "scene_linear" if is_dir and cfg.frame_dir_linear else "display_coded_assumed_srgb",
        "model_rgb": "signed_asinh_scene_linear", "alpha": "coverage",
        "temporal_strategy": "full_rate_with_chronological_rvm_prepass",
        "stage1_frame_evaluations": 0, "stage1_windows_evaluated": 0,
        "stage2_frame_evaluations": 0, "stage2_calls": 0, "completed_prediction_frames": 0,
        "rvm_frame_evaluations": 0,
        "nonfinite_policy": "fail; no NaN/Inf repair",
        "rgb_primaries": ('BT.709/D65 working basis; model training primaries unverified' if cfg.source_colour_profile != 'legacy'
                          else "assumed sRGB/BT.709; no automatic ICC/OCIO/primaries transform"),
        **({'source_colour_contract': meta['source_colour_contract']} if cfg.source_colour_profile != 'legacy' else {}),
    }
    if quick_plan is not None:
        diagnostics.execution["temporal_strategy"] = "sparse_native_context_with_full_rate_reset_aware_rvm"
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        diagnostics.execution.update({"gpu": props.name, "device_index": device_index,
                                      "vram_bytes": props.total_memory, "cuda_version": torch.version.cuda})
    if diagnostics.profile_stages:
        diagnostics.synchronize = torch.cuda.synchronize
        torch.cuda.reset_peak_memory_stats()
    clip_name = raw_clip_name(input_path)
    # Foreground/alpha-only jobs never use a checker image.
    checker_t = torch.from_numpy(make_checker(H_out, W_out)) if "checker" in cfg.outputs else None
    writers = {}
    prefix = diagnostics.durable_writer_prefix
    restored_prefix = 0
    prefix_runtime: dict[str, Any] | None = None
    confirmed_cursors = {}
    output_telemetry = None

    def record_native_cursor(name, cursor):
        if prefix_runtime is not None:
            confirmed_cursors[name] = cursor
            prefix_runtime['native_new_written_frames'] = min(confirmed_cursors.values()) - restored_prefix
            prefix_runtime['native_new_written_files'] = sum(value - restored_prefix for value in confirmed_cursors.values())
            prefix_runtime['native_completion_status'] = 'complete' if min(confirmed_cursors.values()) == T_all else 'partial'

    def release_writer(name, writer):
        writer.release()
        if prefix is not None:
            record_native_cursor(name, writer.frame_idx)

    suffix = {name: layout['frame_name'] for name, layout in planned_layout['outputs'].items()}
    try:
        temp_budget = TempMapBudget(plan_temp_maps(
            model, meta, low_vram=cfg.low_vram,
            anchor_count=len(quick_plan.anchors) if quick_plan is not None else None),
            cfg.temp_map_budget_bytes)
        diagnostics.execution["temp_maps"] = temp_budget.report
        temp_budget.admit()
        job_temp = tempfile.TemporaryDirectory(prefix="job-", dir=cfg.temp_dir)
    except BaseException:
        if correction_rasters is not None:
            cleanup_all([correction_rasters.close], primary_error=sys.exception())
        raise
    job_dir = Path(job_temp.name)
    temp_budget.root = job_dir
    job_resources = ResourceScope()
    if correction_rasters is not None:
        job_resources.own(correction_rasters)
    guidance_source = None
    try:
        trimap_active = correction_rasters is not None and correction_rasters.has_trimaps
        if (spatial and refine_active) or (trimap_active and quick_plan is None):
            guidance_source = job_resources.own(NativeGuidance(
                lambda: DraftSourceReader(input_path, meta, is_dir=is_dir), check=diagnostics.control.check))
            if trimap_active:
                guidance_source.stats['scope'] = 'additional native source decoding for final trimap guidance and optional spatial refinement'
                diagnostics.execution['native_trimap_guidance'] = guidance_source.stats
            else:
                diagnostics.execution['spatial_guidance'] = guidance_source.stats
        if trimap_active:
            from .trimap_refinement import refinement_contract
            diagnostics.corrections['trimap_refinement'] = refinement_contract()
        use_exr = cfg.output_format == "exr"
        if cfg.output_format not in ("video", "exr"):
            raise ValueError(f"unsupported output format: {cfg.output_format}")
        layouts = {}
        for name in sorted(cfg.outputs):
            if name not in suffix:
                raise ValueError(f"unsupported output choice: {name}")
            channels = (4 if name == "cutout" else (1 if name == "alpha" else 3)) if use_exr else (4 if name == "cutout" else 3)
            relative = planned_layout['outputs'][name]['path']
            layouts[name] = {"directory": relative, "frame_name": suffix[name]}
            diagnostics.outputs[name] = os.path.join(output_dir, relative)
            diagnostics.output_contracts[name] = {
                "format": cfg.output_format, "channels": channels,
                **({'spatial_draft': meta['spatial_draft']} if spatial else {}),
                "alpha_association": cfg.cutout_alpha_mode if name == "cutout" else None,
                "dtype": cfg.exr_dtype if use_exr else "uint16 encoder input",
                "codec": None if use_exr else ("prores_ks" if name == "cutout" else writer_cfg.codec),
                "pix_fmt": None if use_exr else ("yuva444p10le" if name == "cutout" else writer_cfg.pix_fmt),
                "threads": None if use_exr else max(1, writer_cfg.threads),
                "rgb_representation": None if name == "alpha" else ("scene_linear" if use_exr or (name == "cutout" and cfg.cutout_linear) else "display_coded"),
                "alpha_representation": "coverage (no transfer function)",
                "rgb_primaries": None if name == "alpha" else ('BT.709/D65 working output' if cfg.source_colour_profile != 'legacy'
                                                              else "assumed sRGB/BT.709; no automatic colour management"),
                **({'source_colour_contract': meta['source_colour_contract']}
                   if name != 'alpha' and cfg.source_colour_profile != 'legacy' else {}),
                "alpha_video_tag_policy": "legacy RGB video tags; treat pixel values as data coverage, bypass RGB transfer" if name == "alpha" and not use_exr else None,
                "despill_screen": cfg.despill_screen if name in {"cutout", "checker"} else None,
                "despill_space": ("scene_linear" if use_exr or cfg.despill_space == "linear" or (name == "cutout" and cfg.cutout_linear)
                                  else "display_coded_srgb") if name in {"cutout", "checker"} else None,
                "despill_preserve_linear_y": cfg.despill_preserve_luma if name in {"cutout", "checker"} else False,
                "half_out_of_range": "finite saturation to +/-65504" if use_exr and cfg.exr_dtype == "half" else None,
                "metadata_scope": "job record/settings; host interpretation and embedded EXR chromaticities unvalidated",
                "foreground_mode": "source_rgb_draft" if quick_plan is not None else "model_reconstructed",
            }

        protect_inputs()
        if prefix is not None:
            restored_prefix = prefix.prepare(output_dir, frame_count=T_all, native_hw=(H_out, W_out), fps=fps,
                outputs={name: {**layout, 'contract': diagnostics.output_contracts[name]} for name, layout in layouts.items()},
                resume=prefix.resume_requested)
            prefix_runtime = {'newly_submitted_frames': 0, 'native_new_written_frames': 0,
                'native_new_written_files': 0, 'native_completion_status': 'not_verified',
                'native_completion_scope': 'Confirmed native writes only; partial/unverified counts are lower bounds. Copied prefix is excluded.'}
            diagnostics.writer_prefix_runtime = prefix_runtime
            confirmed_cursors = dict.fromkeys(layouts, restored_prefix)
        if diagnostics.profile_stages:
            output_telemetry = OutputTelemetry(
                sorted(cfg.outputs), T_all, native_completion=use_exr,
                origin=diagnostics.started, clock=diagnostics.clock, restored_prefix_frames=restored_prefix)
            diagnostics.output_telemetry = output_telemetry
        for name in sorted(cfg.outputs):
            contract = diagnostics.output_contracts[name]
            channels = contract['channels']
            if use_exr:
                make_exr = (partial(ExrSequenceWriter,
                                    on_frame_written=partial(output_telemetry.written, name),
                                    clock=diagnostics.clock)
                            if output_telemetry is not None else ExrSequenceWriter)
                writers[name] = make_exr(
                    diagnostics.outputs[name], fps, (W_out, H_out), channels=channels,
                    dtype=cfg.exr_dtype, frame_name=suffix[name],
                    check=diagnostics.control.check, timeout_seconds=writer_cfg.timeout_seconds,
                    **({'start_index': restored_prefix} if prefix is not None else {}))
            else:
                color_trc = "linear" if (name == "cutout" and cfg.cutout_linear) else "iec61966-2-1"
                writers[name] = ThreadedVideoWriter(
                    diagnostics.outputs[name], fps, (W_out, H_out), writer_cfg, channels=channels, color_trc=color_trc,
                    check=diagnostics.control.check)

        def submit_outputs(rendered, frame_ids, frame_count):
            if set(rendered) != cfg.outputs or any(len(frames) != frame_count for frames in rendered.values()):
                raise RuntimeError('renderer returned an incomplete output batch')
            skip = sum(index < restored_prefix for index in frame_ids) if prefix is not None else 0
            if skip == frame_count:
                return
            submitted_ids = frame_ids[skip:]
            for name, frames in rendered.items():
                writers[name].write_many(frames[skip:] if skip else frames)
                if output_telemetry is not None:
                    output_telemetry.submitted(name, submitted_ids)
            if prefix_runtime is not None:
                prefix_runtime['newly_submitted_frames'] += frame_count - skip
            if prefix is not None and not prefix.snapshot()['checkpoint_disabled']:
                completed = {}
                for name, writer in writers.items():
                    completed[name] = writer.flush_barrier()
                    record_native_cursor(name, completed[name])
                if prefix.checkpoint(submitted_ids, completed=completed):
                    state = prefix.snapshot()
                    diagnostics.control.checkpoint('writer_prefix_frames', restored_prefix + state['stored_frames'], T_all)

        # Write settings log file in the output folder
        settings_path = os.path.join(output_dir, f"{clip_name}_settings.txt")
        protect_inputs()
        try:
            with open(settings_path, "w") as sf:
                sf.write(f"CorridorKey v2 Inference Settings Log\n")
                sf.write(f"====================================\n")
                sf.write(f"Input: {input_path}\n")
                sf.write(f"Output Directory: {output_dir}\n")
                sf.write(f"Resolution: {W_out}x{H_out}\n")
                sf.write(f"FPS: {fps}\n")
                sf.write(f"Outputs: {', '.join(sorted(cfg.outputs))}\n")
                sf.write(f"Output Format: {cfg.output_format}\n")
                if cfg.output_format == "exr":
                    sf.write(f"EXR Pixel Type: {cfg.exr_dtype}\n")
                sf.write(f"Hint First Frame: {initial_hint_path}\n")
                sf.write(f"Hint Video: {hint_video_path}\n")
                sf.write(f"Hint Quality (Trust): {cfg.hint_quality}\n")
                sf.write(f"Carry Hint: {cfg.carry_hint}\n")
                sf.write(f"Carry Hint Quality: {cfg.carry_hint_quality}\n")
                sf.write(f"Despill Strength: {cfg.despill_strength}\n")
                sf.write(f"Cutout Linear: {cfg.cutout_linear}\n")
                sf.write(f"Frame Dir Linear: {cfg.frame_dir_linear}\n")
                sf.write(f"Source Colour Profile: {cfg.source_colour_profile}\n")
                sf.write(f"Hann Chunk: {cfg.hann_chunk}\n")
                sf.write(f"Hann Stride: {cfg.hann_stride}\n")
                sf.write(f"Low VRAM: {cfg.low_vram}\n")
                sf.write(f"Model Scale: {cfg.model_scale}\n")
                sf.write(f"Num Frames: {cfg.num_frames}\n")
                sf.write(f"Start Mode: {cfg.start_mode}\n")
                sf.write(f"Seed: {cfg.seed}\n")
                sf.write(f"Quick Key: {json.dumps(diagnostics.quick_key)}\n")
                sf.write(f"Corrections: {cfg.corrections_path}; identity={correction_document.identity if correction_document else None}\n")
                sf.write(f"Colour/association contracts: {json.dumps(diagnostics.output_contracts)}\n")
                sf.write(f"Input alpha policy: {cfg.frame_dir_alpha}; source linear: {cfg.frame_dir_linear}\n")
                sf.write(f"Alpha refinement: {json.dumps(diagnostics.alpha_refinement)}\n")
                sf.write(f"Linear Brightness: {cfg.linear_brightness}\n")
                sf.write(f"Linear Contrast: {cfg.linear_contrast}\n")
                sf.write(f"Linear Contrast Pivot: {cfg.linear_contrast_pivot}\n")
        except Exception as e:
            print(f"[WARNING] Failed to write settings file: {e}")

        diagnostics.control.checkpoint("prepare", 0, 1)
        initial_hint = (
            load_initial_hint_frame(initial_hint_path, meta)
            if none_path(initial_hint_path) is not None else None
        )
        model.eval()
        configure_inference_memory(model, cfg)
        with diagnostics.stage("phase1_setup"):
            _phase1_setup(model, stage1_on_gpu=not cfg.low_vram)
        rvm_path = job_dir / "rvm_feat.f16"
        baton_path = job_dir / "baton.f16"
        foundation_prefix = job_dir / "foundation"
        if model.foundations.rvm is not None:
            with diagnostics.stage("rvm_prepass"):
                precompute_rvm_to_memmap(
                    model, input_path, meta, rvm_path,
                    chunk=1 if cfg.quick_approach == "causal" else max(1, min(int(cfg.hann_chunk), 8)),
                    reader_kind=reader_kind, control=diagnostics.control,
                    durable_rvm=diagnostics.durable_rvm, diagnostics=diagnostics, temp_budget=temp_budget,
                    reset_frames=(tuple(start for start, _ in quick_plan.segments) if quick_plan else
                                  tuple(start for start, _ in correction_segments) if correction_segments else ()))
            rvm_feat_path = rvm_path
        else:
            rvm_feat_path = None
        foundation_paths = None
        if cfg.low_vram:
            with diagnostics.stage("foundation_prepass"):
                foundation_paths = precompute_foundations_to_memmap(
                    model, input_path, meta, foundation_prefix,
                    chunk=max(1, int(cfg.low_vram_foundation_chunk)),
                    reader_kind=reader_kind, control=diagnostics.control,
                    durable_foundations=diagnostics.durable_foundations,
                    diagnostics=diagnostics, temp_budget=temp_budget)
            model.foundations.cpu()
            model.stage2.cpu()
            model.stage1.cuda()
            torch.cuda.empty_cache()
        else:
            model.stage2.cuda()
            model.stage2.eval()
            torch.cuda.empty_cache()

        stage2_batch = max(1, int(cfg.stage2_batch))
        anchor_rows = {frame: row for row, frame in enumerate(quick_plan.anchors)} if quick_plan else {}
        anchor_alpha = (_writable_map(temp_budget, job_resources, np.memmap, job_dir / "anchor_alpha.f32", mode="w+",
                                            dtype=np.float32, shape=(len(anchor_rows), H_out, W_out))
                        if quick_plan is not None else None)
        anchors_written = set()
        fallback_raw_alpha = {}
        neural_refined_frames = set()

        def finish_native(alpha):
            from .refinement import finish_alpha
            record = edge_finishing
            started = time.perf_counter()
            try:
                result = finish_alpha(alpha, expansion=cfg.alpha_expand_radius,
                                      feather=cfg.alpha_feather_radius, check=diagnostics.control.check)
                record['frames_evaluated'] += 1
                changed = False
                for y in range(0, alpha.shape[0], 256):
                    for x in range(0, alpha.shape[1], 256):
                        diagnostics.control.check()
                        if not np.array_equal(alpha[y:y + 256, x:x + 256], result[y:y + 256, x:x + 256]):
                            changed = True
                            break
                    if changed:
                        break
                record['frames_changed'] += int(changed)
                return result
            finally:
                record['seconds'] += time.perf_counter() - started

        def refine_native(alpha, guidance):
            from .refinement import refine_alpha
            started = time.perf_counter()
            try:
                result = refine_alpha(alpha, guidance, radius=cfg.alpha_refine_radius,
                                      strength=cfg.alpha_refine_strength, sigma=cfg.alpha_refine_sigma,
                                      check=diagnostics.control.check)
                diagnostics.alpha_refinement["frames_evaluated"] += 1
                diagnostics.alpha_refinement["frames_changed"] += int(not np.array_equal(alpha, result))
                return result
            finally:
                diagnostics.alpha_refinement["seconds"] += time.perf_counter() - started
        draft_trace = None
        if quick_plan is not None:
            protect_inputs()
            trace_path = Path(output_dir) / f"{clip_name}_quick_key_frames.jsonl"
            draft_trace = job_resources.own(trace_path.open("w", encoding="utf-8"))
            diagnostics.quick_key["frame_diagnostics_path"] = str(trace_path)

        def write_prediction_batches(items, alpha_sink=None):
            assert diagnostics.execution is not None
            device = next(model.stage2.parameters()).device
            for offset in range(0, len(items), stage2_batch):
                diagnostics.control.checkpoint("stage2_frames" if quick_plan else "output_frames",
                                               diagnostics.execution["completed_prediction_frames"] if quick_plan else diagnostics.frames_submitted,
                                               None if quick_plan else T_all)
                sub = items[offset:offset + stage2_batch]
                rgb_t = torch.stack([item[2] for item in sub], dim=0)
                rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=True)
                baton_np = np.stack([item[1] for item in sub], axis=0)
                baton_t = torch.from_numpy(baton_np).unsqueeze(0)
                baton_t = baton_t.to(device, non_blocking=True).float()
                prediction_store = diagnostics.durable_stage2
                prediction_identity = None
                logits = None
                if prediction_store is not None:
                    prediction_identity = prediction_store.identity(
                        frames=tuple(item[0] for item in sub),
                        policy="quick_fallback" if alpha_sink is not None else "quick_anchor" if quick_plan else "full_rate",
                        rgb=rgb_t, baton=baton_t)
                    if prediction_identity is not None:
                        with diagnostics.stage("stage2_cache_lookup", gpu=False):
                            logits = prediction_store.get(prediction_identity)
                        if logits is not None:
                            logits = logits.to(device=device)
                if logits is None:
                    with diagnostics.stage("stage2_inference"):
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model.stage2(rgb_t, baton_t)
                            diagnostics.execution["stage2_frame_evaluations"] += len(sub)
                            diagnostics.execution["stage2_calls"] += 1
                            if isinstance(logits, tuple):
                                logits = logits[0]
                    if tuple(logits.shape) != (1, len(sub), 4, H, W):
                        raise RuntimeError("Stage 2 returned an incomplete prediction batch")
                    _require_finite(logits, "Stage 2 output")
                    if prediction_identity is not None:
                        assert prediction_store is not None
                        with diagnostics.stage("stage2_cache_store", gpu=False):
                            prediction_store.put(prediction_identity, logits)
                diagnostics.execution["completed_prediction_frames"] += len(sub)
                native_alpha = None
                if spatial and (refine_active or (quick_plan is None and finish_active) or (quick_plan is None and correction_rasters is not None
                                                 and any(correction_rasters.has_effect(item[0]) for item in sub))):
                    native_alpha = _resize_output_torch(_crop_native_torch(logits[0, :, 0:1].float(), meta), meta).clamp(0, 1)
                    for index, item in enumerate(sub):
                        alpha = native_alpha[index, 0].cpu().numpy()
                        guide = None
                        if refine_active:
                            assert guidance_source is not None
                            guide = cu.linear_to_vfm_sdr(guidance_source.read(item[0]))
                            alpha = refine_native(alpha, guide)
                            neural_refined_frames.add(item[0])
                        if finish_active and quick_plan is None:
                            alpha = finish_native(alpha)
                        if quick_plan is None and correction_rasters is not None and correction_rasters.has_effect(item[0]):
                            trimap_guide = None
                            if correction_rasters.requires_guidance(item[0]):
                                assert guidance_source is not None
                                if guide is None:
                                    guide = cu.linear_to_vfm_sdr(guidance_source.read(item[0]))
                                trimap_guide = guide
                            alpha = correction_rasters.apply(alpha, item[0], guidance=trimap_guide)
                            diagnostics.corrections['output_frames_refined'] += 1
                        native_alpha[index, 0] = torch.from_numpy(alpha).to(device)
                if refine_active and not spatial:
                    logits = logits.float().clone()
                    native_rgb = _crop_native_torch(rgb_t[0].float(), meta)
                    linear_guide = _linear_adjust_inverse_torch(cu.asinh_to_linear(native_rgb), meta)
                    _require_finite(linear_guide, "alpha refinement guidance")
                    guidance = cu.linear_to_vfm_sdr_torch(linear_guide).permute(0, 2, 3, 1).cpu().numpy()
                    top, left = meta["pad"][2], meta["pad"][0]
                    height, width = meta["model_native_hw"]
                    for index, item in enumerate(sub):
                        native = logits[0, index, 0, top:top + height, left:left + width].clamp(0, 1).cpu().numpy()
                        refined = refine_native(native, guidance[index])
                        logits[0, index, 0, top:top + height, left:left + width] = torch.from_numpy(refined).to(device)
                        neural_refined_frames.add(item[0])
                if finish_active and not spatial and quick_plan is None:
                    native_alpha = _resize_output_torch(_crop_native_torch(logits[0, :, 0:1].float(), meta), meta).clamp(0, 1)
                    for index in range(len(sub)):
                        native_alpha[index, 0] = torch.from_numpy(finish_native(native_alpha[index, 0].cpu().numpy())).to(device)
                if not spatial and quick_plan is None and correction_rasters is not None and any(correction_rasters.has_effect(item[0]) for item in sub):
                    logits = logits.float().clone()
                    top, left = meta["pad"][2], meta["pad"][0]
                    height, width = meta["model_native_hw"]
                    for index, item in enumerate(sub):
                        frame = item[0]
                        if correction_rasters.has_effect(frame):
                            native = (native_alpha[index, 0] if native_alpha is not None else
                                      logits[0, index, 0, top:top + height, left:left + width]).cpu().numpy()
                            trimap_guide = None
                            if correction_rasters.requires_guidance(frame):
                                assert guidance_source is not None
                                trimap_guide = cu.linear_to_vfm_sdr(guidance_source.read(frame))
                            refined = correction_rasters.apply(native, frame, guidance=trimap_guide)
                            if native_alpha is not None:
                                native_alpha[index, 0] = torch.from_numpy(refined).to(device)
                            else:
                                logits[0, index, 0, top:top + height, left:left + width] = torch.from_numpy(refined).to(device)
                            diagnostics.corrections["output_frames_refined"] += 1
                if quick_plan is not None:
                    alphas = (native_alpha if native_alpha is not None else
                              _resize_output_torch(_crop_native_torch(logits[0, :, 0:1].float(), meta), meta).clamp(0, 1))
                    alphas = alphas[:, 0].cpu().numpy()
                    if not np.isfinite(alphas).all():
                        raise ValueError("Stage 2 returned non-finite draft anchor alpha")
                    for item, alpha in zip(sub, alphas):
                        if alpha_sink is not None:
                            alpha_sink(item[0], alpha.copy())
                        else:
                            anchor_alpha[anchor_rows[item[0]]] = alpha
                            anchors_written.add(item[0])
                    del logits, baton_t, rgb_t, baton_np, alphas, native_alpha
                    continue
                render_fn = (_render_predictions_exr_torch if use_exr
                             else _render_predictions_torch)
                with diagnostics.stage("render"):
                    rendered = (render_fn(meta, cfg.outputs, checker_t, logits[0], native_alpha=native_alpha)
                                if native_alpha is not None else render_fn(meta, cfg.outputs, checker_t, logits[0]))
                diagnostics.frame_ready()
                frame_ids = tuple(item[0] for item in sub) if output_telemetry is not None or prefix is not None else ()
                if output_telemetry is not None:
                    output_telemetry.render_ready(frame_ids)
                with diagnostics.stage("writer_submit"):
                    submit_outputs(rendered, frame_ids, len(sub))
                diagnostics.frames_submitted += len(sub)
                diagnostics.control.checkpoint("output_frames", diagnostics.frames_submitted, T_all)
                del logits, baton_t, rgb_t, baton_np, rendered, native_alpha

        with diagnostics.stage("stage1_stream" if cfg.low_vram else "stage1_stream_including_stage2_render_submit"):
            hann_stage1_stream(
                model, input_path, meta, reader_cls=reader_cls,
                reader_kind=reader_kind, chunk=cfg.hann_chunk, stride=cfg.hann_stride,
                baton_path=baton_path, initial_hint=initial_hint,
                initial_hint_quality=cfg.hint_quality,
                full_hint_video_path=hint_video_path,
                full_hint_quality=cfg.hint_quality,
                rvm_path=rvm_feat_path, foundation_paths=foundation_paths,
                carry_hint_from_aux=cfg.carry_hint,
                carry_hint_quality=cfg.carry_hint_quality,
                on_baton_frames=None if cfg.low_vram else write_prediction_batches,
                foundation_cache_bytes=cfg.foundation_cache_mb * 2**20, diagnostics=diagnostics,
                anchor_plan=quick_plan, context_radius=cfg.quick_context_radius,
                causal_context=cfg.quick_approach == "causal", correction_rasters=correction_rasters,
                segments=correction_segments, temp_budget=temp_budget)

        if quick_plan is not None and diagnostics.foundation_cache is not None:
            diagnostics.quick_key["foundation_passes"].append(dict(diagnostics.foundation_cache))

        if cfg.low_vram:
            _phase2_setup(model)
            with ResourceScope() as resources:
                baton = resources.mapping(np.memmap,
                    baton_path, dtype=np.float16, mode="r",
                    shape=(T_all, model.stage1.d_16, H // 16, W // 16))
                reader = resources.own(reader_cls(input_path, meta))
                target_frames = list(quick_plan.anchors) if quick_plan else list(range(T_all))
                for offset in range(0, len(target_frames), stage2_batch):
                    diagnostics.control.check()
                    subset = target_frames[offset:offset + stage2_batch]
                    items = []
                    if quick_plan is None:
                        model_chunk, _display_chunk, _review_chunk = reader.window(subset[0], subset[-1] + 1)
                        items = [(t, np.asarray(baton[t]).copy(), model_chunk[t - subset[0]]) for t in subset]
                    else:
                        for t in subset:
                            model_chunk, _display_chunk, _review_chunk = reader.window(t, t + 1)
                            items.append((t, np.asarray(baton[t]).copy(), model_chunk[0]))
                    write_prediction_batches(items)
                    del model_chunk, _display_chunk, _review_chunk, items
        if quick_plan is not None:
            from .draft import offline_frames, past_frames
            from .motion import MotionSettings
            if anchors_written != set(quick_plan.anchors):
                raise RuntimeError("Quick Key did not predict all scheduled anchors")
            original_foundation_cache = diagnostics.foundation_cache

            def neural_fallback(frame):
                # The entire native-context window is evaluated, never a sparse
                # time-compressed neural input. Reuse chronological RVM features.
                diagnostics.control.check()
                if cfg.low_vram:
                    model.stage2.cpu()
                    model.stage1.cuda()
                else:
                    _phase1_setup(model)
                single = AnchorPlan(T_all, cfg.quick_interval, quick_plan.segments, (frame,), ())
                items = []
                hann_stage1_stream(
                    model, input_path, meta, reader_cls=reader_cls, reader_kind=reader_kind,
                    chunk=cfg.hann_chunk, stride=cfg.hann_stride, baton_path=baton_path,
                    initial_hint=initial_hint, initial_hint_quality=cfg.hint_quality,
                    full_hint_video_path=hint_video_path, full_hint_quality=cfg.hint_quality,
                    rvm_path=rvm_feat_path, foundation_paths=foundation_paths,
                    carry_hint_from_aux=cfg.carry_hint, carry_hint_quality=cfg.carry_hint_quality,
                    on_baton_frames=items.extend, foundation_cache_bytes=cfg.foundation_cache_mb * 2**20,
                    diagnostics=diagnostics, anchor_plan=single, context_radius=cfg.quick_context_radius,
                    causal_context=cfg.quick_approach == "causal", correction_rasters=correction_rasters,
                    segments=correction_segments, temp_budget=temp_budget)
                if len(items) != 1 or items[0][0] != frame:
                    raise RuntimeError("Quick Key neural fallback did not produce the requested frame")
                if not cfg.low_vram and diagnostics.foundation_cache is not None:
                    diagnostics.quick_key["foundation_passes"].append(dict(diagnostics.foundation_cache))
                _phase2_setup(model)
                result = {}
                write_prediction_batches(items, alpha_sink=result.__setitem__)
                if correction_rasters is not None:
                    fallback_raw_alpha[frame] = result[frame].copy()
                if correction_rasters is not None and correction_rasters.has_effect(frame, temporal=True):
                    diagnostics.corrections["propagation_anchor_evaluations_refined"] += 1
                    return correction_rasters.apply(result[frame], frame, temporal=True)
                return result[frame]

            with ResourceScope() as readers:
                current = readers.own(DraftSourceReader(input_path, meta, is_dir=is_dir))
                anchor_source = (readers.own(DraftSourceReader(input_path, meta, is_dir=is_dir))
                                 if cfg.quick_approach == "offline" else None)
                settings = MotionSettings(max_side=cfg.quick_flow_max_side,
                                          min_reliability=cfg.quick_min_reliability,
                                          max_disagreement=cfg.quick_max_disagreement)
                with diagnostics.stage("quick_key_motion_render_submit", gpu=False):
                    controls = dict(settings=settings, fallback=cfg.quick_fallback,
                                    neural_fallback=neural_fallback, control=diagnostics.control,
                                    stats=diagnostics.quick_key, adaptive=cfg.quick_adaptive,
                                    motion_threshold=cfg.quick_motion_threshold)
                    def read_alpha(t):
                        raw = anchor_alpha[anchor_rows[t]]
                        if correction_rasters is not None and correction_rasters.has_effect(t, temporal=True):
                            diagnostics.corrections["propagation_anchor_evaluations_refined"] += 1
                            return correction_rasters.apply(raw, t, temporal=True)
                        return raw
                    frames = (offline_frames(quick_plan, current.read, anchor_source.read, read_alpha, **controls)
                              if cfg.quick_approach == "offline" else
                              past_frames(quick_plan, current.read, read_alpha, approach=cfg.quick_approach, **controls))
                    for frame in frames:
                        raw = fallback_raw_alpha.pop(frame.index, None)
                        if refine_active and frame.index not in neural_refined_frames:
                            frame.alpha = refine_native(frame.alpha, cu.linear_to_vfm_sdr(frame.rgb))
                            frame.diagnostics["alpha_refinement"] = True
                        if finish_active:
                            # Match known-frame raw-alpha constraint precedence without
                            # feeding artist finishing into propagation or fallback state.
                            if correction_rasters is not None and correction_rasters.has_effect(frame.index):
                                if frame.index in anchor_rows:
                                    assert anchor_alpha is not None
                                    raw = anchor_alpha[anchor_rows[frame.index]]
                                if raw is not None:
                                    frame.alpha = raw
                            frame.alpha = finish_native(frame.alpha)
                            frame.diagnostics['edge_finishing'] = True
                        if correction_rasters is not None and correction_rasters.has_effect(frame.index):
                            # Known neural frames get their complete ordered
                            # output constraints exactly once, from raw alpha.
                            # Directional-only masks never enter flow state.
                            if frame.index in anchor_rows:
                                assert anchor_alpha is not None
                                raw = anchor_alpha[anchor_rows[frame.index]]
                            frame.alpha = correction_rasters.apply(
                                raw if raw is not None and not finish_active else frame.alpha, frame.index,
                                guidance=(cu.linear_to_vfm_sdr(frame.rgb)
                                          if correction_rasters.requires_guidance(frame.index) else None))
                            assert diagnostics.corrections is not None
                            diagnostics.corrections["output_frames_refined"] += 1
                            frame.diagnostics["authored_constraints"] = True
                        rendered = _render_source_draft(frame, meta, cfg.outputs, checker_t, use_exr=use_exr)
                        if set(rendered) != cfg.outputs or any(len(batch) != 1 for batch in rendered.values()):
                            raise RuntimeError("draft renderer returned an incomplete output frame")
                        diagnostics.frame_ready()
                        if output_telemetry is not None:
                            output_telemetry.render_ready((frame.index,))
                        draft_trace.write(json.dumps({"frame": frame.index,
                                                      "event": "rendered_before_writer_submission",
                                                      **frame.diagnostics}, allow_nan=False) + "\n")
                        submit_outputs(rendered, (frame.index,), 1)
                        diagnostics.frames_submitted += 1
                assert diagnostics.quick_key is not None
                diagnostics.quick_key["source_frames_decoded"] = current.frames_decoded
                diagnostics.quick_key["anchor_source_frames_decoded"] = anchor_source.frames_decoded if anchor_source else 0
            diagnostics.foundation_cache = original_foundation_cache
        if prefix_runtime is not None and prefix_runtime['newly_submitted_frames'] + restored_prefix != T_all:
            raise RuntimeError('Writer-prefix restored/new coverage differs from the selected timeline')
        if diagnostics.frames_submitted != T_all:
            raise RuntimeError(f"Incomplete render: submitted {diagnostics.frames_submitted}/{T_all} frames")
    finally:
        if diagnostics.source is not None and "decoded_source_dtypes_observed" in meta:
            diagnostics.source["decoded_source_dtypes_observed"] = sorted(set(meta["decoded_source_dtypes_observed"]))
        with diagnostics.stage("writer_finalize_and_cleanup", gpu=False):
            try:
                cleanup_all(
                    [partial(release_writer, name, writer) for name, writer in writers.items()]
                    + [lambda: job_resources.__exit__(None, None, None), lambda: temp_budget.cleanup(job_temp.cleanup), torch.cuda.empty_cache],
                    primary_error=sys.exception())
            finally:
                if output_telemetry is not None:
                    for name, writer in writers.items():
                        observation_error = getattr(writer, "observation_error", None)
                        if observation_error is not None:
                            output_telemetry.observation_failed(f"{name}: {observation_error}")
                    output_telemetry.finish()
    if prefix is not None:
        prefix.verify_restored_prefix()
    if witness is not None:
        diagnostics.source_witness = witness.finish(Path(output_dir))
    diagnostics.control.checkpoint("complete", T_all, T_all)
    diagnostics.outputs = {name: writer.path for name, writer in writers.items()}
    if diagnostics.profile_stages:
        diagnostics.resources = {
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
            "gpu": torch.cuda.get_device_name(), "cuda_version": torch.version.cuda,
            "scope": "processing after model load; process allocation peaks, not whole-device usage"}
    print(f"  wrote {len(writers)} output(s) for {clip_name} "
          f"({T_all} frames @ {W_out}x{H_out})")
