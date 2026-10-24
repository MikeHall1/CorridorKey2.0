#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
from corridorkey.config import InferenceConfig, WriterConfig
from corridorkey.session import InferenceSession, JobRequest
from corridorkey.diagnostics import write_json_atomic


def _parse_outputs(values):
    if not values:
        return {"alpha", "fg"}
    out = set()
    aliases = {
        "raw_fg": "fg",
        "foreground": "fg",
    }
    for value in values:
        for part in str(value).split(","):
            part = part.strip().lower()
            if part:
                part = aliases.get(part, part)
                out.add(part)
    allowed = {"alpha", "fg", "checker", "cutout"}
    bad = sorted(out - allowed)
    if bad:
        raise ValueError(f"unsupported --outputs value(s): {bad}")
    return out


def main(argv=None, *, session=None):
    ap = argparse.ArgumentParser(
        description="CorridorKey v2 native-resolution raw-video inference")
    ap.add_argument("--input", required=True,
                    help="Input video file or directory of image frames.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default=None,
                    help="Runtime checkpoint (.safetensors or legacy .pt). "
                         "Defaults to weights/corridorkey_v2.safetensors.")
    ap.add_argument("--hint_first_frame", default=None,
                    help="Optional matte image/dir; first frame only.")
    ap.add_argument("--hint_video", default=None,
                    help="Optional full-frame hint video or frame directory.")
    ap.add_argument("--hint_quality", type=float, default=0.95)
    ap.add_argument("--carry_hint_quality", type=float, default=0.95)
    ap.add_argument("--no_carry_hint", action="store_true")
    ap.add_argument("--hann_chunk", type=int, default=None,
                    help="Frames per stage-1 window. Default: chosen "
                         "automatically from free GPU memory.")
    ap.add_argument("--hann_stride", type=int, default=None,
                    help="Frames between stage-1 windows. Default: half of "
                         "--hann_chunk.")
    ap.add_argument("--native_resolution", action="store_true", default=True,
                    help="Accepted for clarity; native resolution is always used.")
    ap.add_argument("--low_vram", action="store_true",
                    help="Staged low-memory inference. Passing this or "
                         "--hann_chunk turns off automatic selection.")
    ap.add_argument("--foundation_cache_mb", type=int, default=0,
                    help="Opt-in float32 RAM reuse of overlapping C-RADIO/MoGe features; payload budget in MiB. Default: off.")
    ap.add_argument("--temp_map_budget_bytes", type=int, default=None,
                    help="Aggregate managed inference scratch-map payload ceiling in bytes; 0 permits no maps. Default: unlimited. Excludes outputs, caches, filesystem overhead and RAM/VRAM.")
    ap.add_argument("--process_tree_rss_stop_bytes", type=int, default=None,
                    help="Opt-in sampled resident-memory stop threshold for this process and tracked descendants in bytes. Cooperative checks can overshoot; not a hard RAM cap. Default: off.")
    ap.add_argument("--auto_vram_budget_mb", type=int, default=None,
                    help="Cap the budget used by automatic VRAM profile selection (MiB); not an allocator limit.")
    ap.add_argument("--temp_dir",
                    default=str(Path(tempfile.gettempdir()) / "corridorkey_v2"))
    ap.add_argument("--outputs", nargs="*", default=None,
                    help=("Any of: alpha fg/raw_fg checker cutout. "
                          "Default: alpha fg."))
    ap.add_argument("--num_frames", type=int, default=-1)
    ap.add_argument("--start_frame", type=int, default=None,
                    help="Explicit zero-based source in frame; requires start_mode=begin. num_frames limits the remaining range.")
    ap.add_argument("--start_mode", choices=("begin", "middle", "random_middle"),
                    default="begin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_scale", type=float, default=1.0,
                    help="Model scale in (0, 1]; default 1 preserves native processing. "
                         "Below 1 explicitly selects approximate spatial draft with native-sized exports; quality is unverified.")
    ap.add_argument("--frame_dir_fps", type=float, default=24.0)
    ap.add_argument("--frame_dir_linear", action="store_true",
                    help="Treat frame-directory inputs as scene-linear RGB.")
    ap.add_argument('--source_colour_profile', choices=('legacy', 'linear_bt709_d65', 'linear_bt2020_d65'),
                    default='legacy', help='Explicit scene-linear sequence primaries; output uses the assumed BT.709 working basis. Requires --frame_dir_linear; strict source witness unsupported.')
    ap.add_argument("--source_witness", action="store_true",
                    help="Record exact consumed source pixels for strict bound caches; scene-linear RGB frame directories/EXR only, max16384 frames.")
    ap.add_argument("--frame_dir_alpha", choices=("ignore", "premultiplied"), default="ignore",
                    help="Explicit premultiplied decoding requires scene-linear RGBA frames; default preserves legacy ignored-alpha RGB.")
    ap.add_argument("--cutout_alpha_mode", choices=("straight", "premultiplied"), default="straight",
                    help="Premultiplied output is supported for scene-linear EXR cutout only.")
    ap.add_argument("--cutout_linear", action="store_true",
                    help="Export cutout as straight linear RGB instead of straight sRGB.")
    ap.add_argument("--despill_strength", type=float, default=0.5,
                    help="Despill strength from 0.0 (none) to 1.0 (full). Default: 0.5.")
    ap.add_argument("--despill_screen", choices=("green", "blue"), default="green")
    ap.add_argument("--despill_space", choices=("legacy", "linear"), default="legacy")
    ap.add_argument("--despill_preserve_luma", action="store_true", help="Preserve weighted linear Y; requires linear despill space.")
    ap.add_argument("--despill_edges_only", action="store_true", help="Weight despill by 4*alpha*(1-alpha).")
    ap.add_argument("--alpha_refine_radius", type=int, default=0, help="Opt-in native spatial alpha filter radius [0,4]; 0 is off. Not a quality guarantee.")
    ap.add_argument("--alpha_refine_strength", type=float, default=1.0)
    ap.add_argument("--alpha_refine_sigma", type=float, default=0.1)
    ap.add_argument("--shot_profiles", type=Path, default=None, help="Apply a saved source/base-recipe-bound shot profile document; explicit settings must match.")
    ap.add_argument("--alpha_expand_radius", type=int, default=0, help="Native coverage square expansion (+) / contraction (-) pixels [-32,32]; before authored constraints, 0 off.")
    ap.add_argument("--alpha_feather_radius", type=int, default=0, help="Native coverage Gaussian feather support radius [0,32] pixels; after expansion, before authored constraints, 0 off.")
    ap.add_argument("--linear_brightness", type=float, default=1.0)
    ap.add_argument("--linear_contrast", type=float, default=1.0)
    ap.add_argument("--linear_contrast_pivot", type=float, default=0.18)
    ap.add_argument("--output_format", choices=("video", "exr"),
                    default="video",
                    help=("video: ffmpeg-encoded preview videos (display-"
                          "referred, clipped to [0,1]). exr: unclamped scene-"
                          "linear OpenEXR image sequences (full HDR float)."))
    ap.add_argument("--exr_dtype", choices=("half", "float"), default="half",
                    help="EXR pixel type for --output_format exr. Default: half.")
    ap.add_argument("--ffmpeg_bin", default="ffmpeg")
    ap.add_argument("--ffmpeg_codec", default="libx264")
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--pix_fmt", default="yuv444p10le",
                    help="Pixel format for x264 preview outputs. Default is "
                         "10-bit; pass yuv444p for 8-bit compatibility.")
    ap.add_argument("--bitrate", default=None)
    ap.add_argument("--ffmpeg_threads", type=int, default=2)
    ap.add_argument("--writer_timeout_seconds", type=float, default=60.0,
                    help="Maximum writer queue/finalization wait before failure (default 60 seconds).")
    ap.add_argument("--stage2_batch", type=int, default=1,
                    help="Number of /1 output frames to run/render per stage-2 batch.")
    ap.add_argument("--low_vram_foundation_chunk", type=int, default=4)
    ap.add_argument("--low_vram_mlp_chunk_tokens", type=int, default=65536)
    ap.add_argument("--low_vram_enc_frame_chunk", type=int, default=1)
    ap.add_argument("--low_vram_swin_window_batch", type=int, default=16)
    ap.add_argument("--low_vram_full_attn_query_chunk_tokens", type=int,
                    default=2048)
    ap.add_argument("--diagnostics", default=None,
                    help="Write an atomic JSON job record, including failures, to this path.")
    ap.add_argument("--profile_stages", action="store_true",
                    help="Synchronize GPU stage scopes for diagnostic timings; affects throughput.")
    ap.add_argument("--quick_key", action="store_true", help="Explicit offline source-RGB draft mode; not full-quality keying.")
    ap.add_argument("--quick_interval", type=int, default=5, help="Anchor distance in native frames; 1 retains full-rate inference.")
    ap.add_argument("--quick_context_radius", type=int, default=2, help="Native context frames on each side of anchors.")
    ap.add_argument("--quick_flow_max_side", type=int, default=640)
    ap.add_argument("--quick_fallback", choices=("stop", "approximate", "neural"), default="stop")
    ap.add_argument("--quick_min_reliability", type=float, default=0.2)
    ap.add_argument("--quick_max_disagreement", type=float, default=0.25)
    ap.add_argument("--quick_cut_frames", type=int, nargs="*", default=[], help="Selected-clip frame indices starting new shots.")
    ap.add_argument("--quick_reset_frames", type=int, nargs="*", default=[])
    ap.add_argument("--quick_auto_cuts", action="store_true", help="Opt-in heuristic shot analysis; can miss cuts or flag flashes.")
    ap.add_argument("--quick_cut_threshold", type=float, default=0.35)
    ap.add_argument("--quick_analysis_max_side", type=int, default=128)
    ap.add_argument("--quick_adaptive", action="store_true", help="Promote additional neural fallbacks to anchors; requires --quick_fallback neural.")
    ap.add_argument("--quick_max_gap", type=int, default=None)
    ap.add_argument("--quick_motion_threshold", type=float, default=0.05, help="Adaptive 95th-percentile anchor displacement / image diagonal.")
    ap.add_argument("--quick_approach", choices=("offline", "hold", "causal"), default="offline",
                    help="Offline bidirectional / hold-last-matte / past-only motion. Whole-clip precompute remains offline.")
    ap.add_argument("--corrections", default=None, help="Versioned local authored correction document; content/geometry/timebase are validated.")
    args = ap.parse_args(argv)

    # With no explicit memory settings, pick them per clip from free GPU memory.
    auto_vram = args.hann_chunk is None and args.hann_stride is None and not args.low_vram
    hann_chunk = args.hann_chunk if args.hann_chunk is not None else 80
    hann_stride = args.hann_stride if args.hann_stride is not None else max(1, hann_chunk // 2)
    infer_cfg = InferenceConfig(
        hann_chunk=hann_chunk,
        hann_stride=hann_stride,
        auto_vram=auto_vram,
        foundation_cache_mb=args.foundation_cache_mb,
        temp_map_budget_bytes=args.temp_map_budget_bytes,
        process_tree_rss_stop_bytes=args.process_tree_rss_stop_bytes,
        auto_vram_budget_mb=args.auto_vram_budget_mb,
        carry_hint=not args.no_carry_hint,
        hint_quality=args.hint_quality,
        carry_hint_quality=args.carry_hint_quality,
        model_scale=args.model_scale,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        start_mode=args.start_mode,
        seed=args.seed,
        low_vram=args.low_vram,
        low_vram_foundation_chunk=args.low_vram_foundation_chunk,
        low_vram_mlp_chunk_tokens=args.low_vram_mlp_chunk_tokens,
        low_vram_enc_frame_chunk=args.low_vram_enc_frame_chunk,
        low_vram_swin_window_batch=args.low_vram_swin_window_batch,
        low_vram_full_attn_query_chunk_tokens=(
            args.low_vram_full_attn_query_chunk_tokens),
        stage2_batch=args.stage2_batch,
        outputs=_parse_outputs(args.outputs),
        temp_dir=Path(args.temp_dir),
        linear_brightness=args.linear_brightness,
        linear_contrast=args.linear_contrast,
        linear_contrast_pivot=args.linear_contrast_pivot,
        frame_dir_fps=args.frame_dir_fps,
        frame_dir_linear=args.frame_dir_linear,
        source_colour_profile=args.source_colour_profile,
        source_witness=args.source_witness,
        frame_dir_alpha=args.frame_dir_alpha, cutout_alpha_mode=args.cutout_alpha_mode,
        cutout_linear=args.cutout_linear,
        despill_strength=args.despill_strength,
        despill_screen=args.despill_screen, despill_space=args.despill_space,
        despill_preserve_luma=args.despill_preserve_luma, despill_edges_only=args.despill_edges_only,
        alpha_refine_radius=args.alpha_refine_radius, alpha_refine_strength=args.alpha_refine_strength,
        alpha_refine_sigma=args.alpha_refine_sigma,
        alpha_expand_radius=args.alpha_expand_radius, alpha_feather_radius=args.alpha_feather_radius,
        output_format=args.output_format,
        exr_dtype=args.exr_dtype,
        quick_key=args.quick_key, quick_interval=args.quick_interval,
        quick_context_radius=args.quick_context_radius, quick_flow_max_side=args.quick_flow_max_side,
        quick_fallback=args.quick_fallback, quick_min_reliability=args.quick_min_reliability,
        quick_max_disagreement=args.quick_max_disagreement,
        quick_cut_frames=args.quick_cut_frames, quick_reset_frames=args.quick_reset_frames,
        quick_auto_cuts=args.quick_auto_cuts, quick_cut_threshold=args.quick_cut_threshold,
        quick_analysis_max_side=args.quick_analysis_max_side,
        quick_adaptive=args.quick_adaptive, quick_max_gap=args.quick_max_gap,
        quick_motion_threshold=args.quick_motion_threshold,
        quick_approach=args.quick_approach,
        corrections_path=args.corrections,
    )
    try:
        infer_cfg.validate()
    except ValueError as exc:
        ap.error(str(exc))
    writer_cfg = WriterConfig(
        ffmpeg_bin=args.ffmpeg_bin,
        codec=args.ffmpeg_codec,
        crf=args.crf,
        preset=args.preset,
        pix_fmt=args.pix_fmt,
        bitrate=args.bitrate,
        threads=args.ffmpeg_threads,
        timeout_seconds=args.writer_timeout_seconds,
    )
    try:
        job = JobRequest.create(args.input, args.output_dir, infer_cfg, writer_cfg,
                                initial_hint_path=args.hint_first_frame,
                                hint_video_path=args.hint_video, profile_stages=args.profile_stages)
    except ValueError as exc:
        ap.error(str(exc))
    if args.shot_profiles is not None:
        from corridorkey.shot_profiles import apply_profiles, load_profiles
        try:
            _, profile_plan = load_profiles(args.shot_profiles)
            job = apply_profiles(job, profile_plan)
        except (ValueError, OSError) as exc:
            ap.error(str(exc))
    if session is not None:
        if args.weights is not None and str(args.weights) != session.weights:
            ap.error("a persistent session cannot change weights; create a new session")
        return _run_job(session, job, args.diagnostics,
                        protected_paths=() if args.shot_profiles is None else (args.shot_profiles,))
    with InferenceSession(Path(__file__).resolve().parent, args.weights) as owned_session:
        return _run_job(owned_session, job, args.diagnostics,
                        protected_paths=() if args.shot_profiles is None else (args.shot_profiles,))


def _run_job(session, job, diagnostics_path, *, protected_paths=()):
    from corridorkey.output_paths import (
        validate_batch_output_paths,
        validate_report_destination,
    )
    protected_paths = tuple(protected_paths)
    if protected_paths:
        validate_batch_output_paths((job,), weights=session.weights,
            package_root=session.package_root, asset_root=session.asset_root, protected_paths=protected_paths)
    def persist():
        validate_report_destination(diagnostics_path, (job,), weights=session.weights,
                                    package_root=session.package_root, asset_root=session.asset_root, protected_paths=protected_paths)
        write_json_atomic(diagnostics_path, session.last_result.to_dict())
    if diagnostics_path is not None:
        validate_report_destination(diagnostics_path, (job,), weights=session.weights,
                                    package_root=session.package_root, asset_root=session.asset_root, protected_paths=protected_paths)
    previous = session.last_result
    try:
        return session.run(job)
    finally:
        if (diagnostics_path is not None and session.last_result is not None
                and session.last_result is not previous):
            # Report failure must not replace an inference/cancellation exception.
            import sys
            from corridorkey.lifecycle import cleanup_all
            cleanup_all([persist], primary_error=sys.exception())

if __name__ == "__main__":
    main()
