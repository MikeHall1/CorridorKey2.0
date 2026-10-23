"""Shared job settings; importable without PyTorch or GUI dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import tempfile


@dataclass
class InferenceConfig:
    hann_chunk: int = 80
    hann_stride: int = 40
    carry_hint: bool = True
    hint_quality: float = 0.95
    carry_hint_quality: float = 0.95
    shape_quantum: int = 128
    model_scale: float = 1.0
    num_frames: int = -1
    start_mode: str = "begin"
    start_frame: int | None = None
    seed: int = 0
    low_vram: bool = False
    # Pick hann_chunk/hann_stride/low_vram per clip from free GPU memory.
    auto_vram: bool = False
    auto_vram_budget_mb: int | None = None
    foundation_cache_mb: int = 0
    temp_map_budget_bytes: int | None = None
    process_tree_rss_stop_bytes: int | None = None
    low_vram_foundation_chunk: int = 4
    low_vram_mlp_chunk_tokens: int = 65536
    low_vram_enc_frame_chunk: int = 1
    low_vram_swin_window_batch: int = 16
    low_vram_full_attn_query_chunk_tokens: int = 2048
    stage2_batch: int = 1
    outputs: set[str] = field(default_factory=lambda: {"alpha", "fg"})
    temp_dir: Path = Path(tempfile.gettempdir()) / "corridorkey_v2"
    linear_brightness: float = 1.0
    linear_contrast: float = 1.0
    linear_contrast_pivot: float = 0.18
    frame_dir_fps: float = 24.0
    frame_dir_linear: bool = False
    source_colour_profile: str = 'legacy'
    source_witness: bool = False
    cutout_linear: bool = False
    despill_strength: float = 0.5
    output_format: str = "video"
    exr_dtype: str = "half"
    quick_key: bool = False
    quick_interval: int = 5
    quick_context_radius: int = 2
    quick_flow_max_side: int = 640
    quick_fallback: str = "stop"
    quick_min_reliability: float = 0.2
    quick_max_disagreement: float = 0.25
    quick_cut_frames: list[int] = field(default_factory=list)
    quick_reset_frames: list[int] = field(default_factory=list)
    quick_auto_cuts: bool = False
    quick_cut_threshold: float = 0.35
    quick_analysis_max_side: int = 128
    quick_adaptive: bool = False
    quick_max_gap: int | None = None
    quick_motion_threshold: float = 0.05
    quick_approach: str = "offline"
    corrections_path: str | Path | None = None
    frame_dir_alpha: str = "ignore"
    cutout_alpha_mode: str = "straight"
    alpha_refine_radius: int = 0
    alpha_refine_strength: float = 1.0
    alpha_refine_sigma: float = 0.1
    despill_screen: str = "green"
    despill_space: str = "legacy"
    despill_preserve_luma: bool = False
    despill_edges_only: bool = False

    def validate(self):
        if type(self.alpha_refine_radius) is not int or not 0 <= self.alpha_refine_radius <= 4:
            raise ValueError("alpha_refine_radius must be an integer in [0,4]")
        for name in ("alpha_refine_strength", "alpha_refine_sigma"):
            value = getattr(self, name)
            minimum = 1e-6 if name.endswith("sigma") else 0
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= 1:
                raise ValueError(f"invalid {name}")
        if self.despill_screen not in ("green", "blue") or self.despill_space not in ("legacy", "linear"):
            raise ValueError("despill_screen must be green/blue and despill_space legacy/linear")
        if type(self.despill_preserve_luma) is not bool or type(self.despill_edges_only) is not bool:
            raise ValueError("despill luminance/edge controls must be booleans")
        if self.despill_preserve_luma and self.despill_space != "linear":
            raise ValueError("despill_preserve_luma requires explicit linear despill_space")
        if self.frame_dir_alpha not in ("ignore", "premultiplied"):
            raise ValueError("frame_dir_alpha must be ignore or premultiplied")
        if self.frame_dir_alpha == "premultiplied" and not self.frame_dir_linear:
            raise ValueError("premultiplied frame input requires explicit frame_dir_linear=True")
        if self.source_colour_profile not in ('legacy', 'linear_bt709_d65', 'linear_bt2020_d65'):
            raise ValueError('source_colour_profile must be legacy, linear_bt709_d65 or linear_bt2020_d65')
        if self.source_colour_profile != 'legacy' and not self.frame_dir_linear:
            raise ValueError('Declared source primaries require explicit scene-linear frame-directory input')
        if self.source_colour_profile != 'legacy' and self.source_witness:
            raise ValueError('Declared source primaries with strict source_witness/OFX binding are not supported')
        if self.cutout_alpha_mode not in ("straight", "premultiplied"):
            raise ValueError("cutout_alpha_mode must be straight or premultiplied")
        if self.cutout_alpha_mode == "premultiplied" and (self.output_format != "exr" or not isinstance(self.outputs, (set, frozenset)) or "cutout" not in self.outputs):
            raise ValueError("premultiplied output currently requires EXR cutout; video association remains straight")
        if self.corrections_path is not None and (not isinstance(self.corrections_path, (str, Path)) or not str(self.corrections_path).strip()):
            raise ValueError("corrections_path must be a local file path or null")
        for name in ("carry_hint", "low_vram", "auto_vram", "frame_dir_linear", "source_witness", "cutout_linear", "quick_key", "quick_auto_cuts", "quick_adaptive"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.source_witness and (not self.frame_dir_linear or self.frame_dir_alpha != 'ignore' or self.output_format != 'exr'):
            raise ValueError('source_witness requires explicit scene-linear RGB frame input and EXR output.')
        for name, minimum in (("quick_interval", 1), ("quick_context_radius", 0), ("quick_flow_max_side", 16), ("quick_analysis_max_side", 16)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.quick_fallback not in ("stop", "approximate", "neural"):
            raise ValueError("quick_fallback must be stop, approximate or neural")
        if self.quick_approach not in ("offline", "hold", "causal"):
            raise ValueError("quick_approach must be offline, hold or causal")
        if self.quick_approach != "offline" and (not self.quick_key or self.quick_interval == 1):
            raise ValueError("hold/causal approaches require sparse Quick Key; interval 1 retains the original full-rate path")
        if self.quick_adaptive and self.quick_approach == "hold":
            raise ValueError("hold-last-matte does not support adaptive flow anchors")
        for name in ("quick_min_reliability", "quick_max_disagreement", "quick_cut_threshold", "quick_motion_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0,1]")
        if self.quick_auto_cuts and (not self.quick_key or self.quick_interval == 1):
            raise ValueError("quick_auto_cuts requires sparse Quick Key; interval 1 retains the original full-rate path")
        if self.quick_max_gap is not None:
            if type(self.quick_max_gap) is not int or self.quick_max_gap < 1:
                raise ValueError("quick_max_gap must be a positive integer or null")
            if not self.quick_key or self.quick_interval == 1:
                raise ValueError("quick_max_gap requires sparse Quick Key")
        if self.quick_adaptive and (not self.quick_key or self.quick_interval == 1 or self.quick_fallback != "neural"):
            raise ValueError("quick_adaptive requires sparse Quick Key and quick_fallback=neural")
        for name in ("quick_cut_frames", "quick_reset_frames"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(type(t) is not int or t < 0 for t in values):
                raise ValueError(f"{name} must contain nonnegative integer source-frame indices")
            if values and (not self.quick_key or self.quick_interval == 1):
                raise ValueError(f"{name} requires sparse Quick Key; interval 1 retains the original full-rate path")
        for name in ("model_scale", "hint_quality", "carry_hint_quality", "despill_strength",
                     "linear_brightness", "linear_contrast", "linear_contrast_pivot", "frame_dir_fps"):
            if type(getattr(self, name)) not in (int, float):
                raise ValueError(f"{name} must be a number")
        if not isinstance(self.outputs, (set, frozenset)):
            raise ValueError("outputs must be a set of output names")
        if not isinstance(self.temp_dir, (str, Path)):
            raise ValueError("temp_dir must be a filesystem path")
        if type(self.foundation_cache_mb) is not int or self.foundation_cache_mb < 0:
            raise ValueError("foundation_cache_mb must be a nonnegative integer (MiB); 0 disables reuse")
        if self.temp_map_budget_bytes is not None and (
                type(self.temp_map_budget_bytes) is not int or not 0 <= self.temp_map_budget_bytes < 2**63):
            raise ValueError("temp_map_budget_bytes must be an integer in [0, 9223372036854775807] or null (unlimited)")
        if self.process_tree_rss_stop_bytes is not None and (
                type(self.process_tree_rss_stop_bytes) is not int or not 0 <= self.process_tree_rss_stop_bytes < 2**63):
            raise ValueError("process_tree_rss_stop_bytes must be an integer in [0, 9223372036854775807] or null (off)")
        if self.low_vram and self.foundation_cache_mb:
            raise ValueError("low_vram already caches foundations on disk; set foundation_cache_mb=0")
        if self.auto_vram_budget_mb is not None:
            if type(self.auto_vram_budget_mb) is not int or self.auto_vram_budget_mb <= 0:
                raise ValueError("auto_vram_budget_mb must be a positive integer (MiB)")
            if not self.auto_vram:
                raise ValueError("auto_vram_budget_mb requires auto_vram=True; omit explicit Hann/low-VRAM CLI flags")
        for name in ("hann_chunk", "hann_stride", "stage2_batch",
                     "low_vram_foundation_chunk"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hann_stride > self.hann_chunk:
            raise ValueError("hann_stride must be <= hann_chunk")
        for name in ("low_vram_mlp_chunk_tokens", "low_vram_enc_frame_chunk",
                     "low_vram_swin_window_batch",
                     "low_vram_full_attn_query_chunk_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer (0 disables chunking)")
        if type(self.shape_quantum) is not int or self.shape_quantum < 64 or self.shape_quantum % 64:
            raise ValueError("shape_quantum must be a positive multiple of 64")
        if not 0 < self.model_scale <= 1 or not math.isfinite(self.model_scale):
            raise ValueError("model_scale must be finite and in (0, 1]; values below 1 explicitly select approximate spatial draft")
        for name in ("hint_quality", "carry_hint_quality", "despill_strength"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("linear_brightness", "linear_contrast"):
            value = getattr(self, name)
            if not math.isfinite(value) or value == 0:
                raise ValueError(f"{name} must be finite and nonzero so output adjustments can be inverted")
        if not math.isfinite(self.frame_dir_fps) or self.frame_dir_fps <= 0:
            raise ValueError("frame_dir_fps must be finite and > 0")
        if not math.isfinite(self.linear_contrast_pivot):
            raise ValueError("linear_contrast_pivot must be finite")
        if type(self.num_frames) is not int:
            raise ValueError("num_frames must be an integer (<= 0 means all)")
        if not isinstance(self.start_mode, str) or self.start_mode not in {"begin", "middle", "random_middle"}:
            raise ValueError("unsupported start_mode")
        if self.start_frame is not None:
            if type(self.start_frame) is not int or self.start_frame < 0:
                raise ValueError("start_frame must be null or a nonnegative integer")
            if self.start_mode != "begin":
                raise ValueError("explicit start_frame requires start_mode=begin")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not self.outputs or self.outputs - {"alpha", "fg", "checker", "cutout"}:
            raise ValueError("outputs must be a nonempty subset of alpha, fg, checker, cutout")
        if not isinstance(self.output_format, str) or self.output_format not in {"video", "exr"}:
            raise ValueError("output_format must be video or exr")
        if not isinstance(self.exr_dtype, str) or self.exr_dtype not in {"half", "float"}:
            raise ValueError("exr_dtype must be half or float")

@dataclass
class WriterConfig:
    ffmpeg_bin: str = "ffmpeg"
    codec: str = "libx264"
    crf: int = 12
    preset: str = "medium"
    pix_fmt: str = "yuv444p10le"
    bitrate: str | None = None
    threads: int = 2
    timeout_seconds: float = 60.0

    def validate(self):
        if type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("writer timeout_seconds must be finite and > 0")
        if type(self.crf) is not int or type(self.threads) is not int:
            raise ValueError("writer crf and threads must be integers")
        for name in ("ffmpeg_bin", "codec", "preset", "pix_fmt"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"writer {name} must be a nonempty string")
        if self.bitrate is not None and not isinstance(self.bitrate, str):
            raise ValueError("writer bitrate must be a string or null")
