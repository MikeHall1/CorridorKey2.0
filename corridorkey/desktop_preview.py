"""CPU source/export viewing; display conversion never changes render data."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np
from numpy.typing import NDArray

from .scheduling import frame_paths


@dataclass(frozen=True)
class PreviewFrame:
    pixels: NDArray[np.uint8]
    count: int
    width: int
    height: int
    fps: float | None
    note: str


def read_frame(path: str, index: int, *, alpha_mode: str = 'ignore') -> tuple[NDArray[np.float32], int, float | None]:
    from .video_io import read_image_rgb

    if type(index) is not int or index < 0:
        raise ValueError('Preview frame must be nonnegative.')
    source = Path(path)
    if source.is_dir() or source.suffix.lower() in {'.png', '.jpg', '.jpeg', '.exr'}:
        paths = frame_paths(source) if source.is_dir() else [str(source)]
        if not paths:
            raise ValueError('No supported images found in the selected directory.')
        if index >= len(paths):
            raise ValueError(f'Preview frame {index} exceeds {len(paths)} available frames.')
        return np.asarray(read_image_rgb(paths[index], alpha_mode=alpha_mode), dtype=np.float32), len(paths), None
    if alpha_mode != 'ignore':
        raise ValueError('Video premultiplied-alpha preview is unsupported.')
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open preview source: {source}')
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count <= 0 or index >= count:
            raise ValueError('Video frame count unavailable or requested frame out of range.')
        if index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError('Video decoder cannot seek to this frame.')
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError('Video frame decode failed.')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb, count, fps if np.isfinite(fps) and fps > 0 else None
    finally:
        capture.release()


def preview(path: str, index: int, *, linear: bool = False, coverage: bool = False,
            alpha_mode: str = 'ignore', alpha_path: str | None = None,
            background: tuple[float, float, float] | None = None, max_side: int = 1280,
            expected_count: int | None = None, expected_hw: tuple[int, int] | None = None,
            source_colour_profile: str = 'legacy') -> PreviewFrame:
    from .color_utils import linear_to_srgb, srgb_to_linear

    if type(max_side) is not int or max_side < 1:
        raise ValueError('Preview maximum side must be positive.')
    if not coverage:
        from .colour_contract import describe_source_colour
        describe_source_colour(source_colour_profile, scene_linear=linear)
        source = Path(path)
        if source_colour_profile != 'legacy' and not (source.is_dir() or source.suffix.lower() in {'.png', '.jpg', '.jpeg', '.exr'}):
            raise ValueError('Explicit source colour profiles require scene-linear image sequences, not video.')
    rgb, count, fps = read_frame(path, index, alpha_mode=alpha_mode)
    if not coverage and source_colour_profile != 'legacy':
        from .colour_contract import to_working_rgb
        rgb = to_working_rgb(rgb, source_colour_profile)
    height, width = rgb.shape[:2]
    if expected_count is not None and count != expected_count:
        raise ValueError('Current export frame count differs from the completed render record.')
    if expected_hw is not None and (height, width) != expected_hw:
        raise ValueError('Current export geometry differs from the completed render record.')
    if coverage:
        alpha = rgb[:, :, :1]
        if np.any((alpha < 0) | (alpha > 1)):
            raise ValueError('Export alpha must be coverage in [0,1].')
        rgb = np.repeat(alpha, 3, axis=2)
    elif alpha_path is not None:
        alpha_rgb, alpha_count, _ = read_frame(alpha_path, index)
        if alpha_count != count or alpha_rgb.shape[:2] != (height, width):
            raise ValueError('Foreground and alpha exports must have matching geometry/frame count.')
        alpha = alpha_rgb[:, :, :1]
        if np.any((alpha < 0) | (alpha > 1)):
            raise ValueError('Export alpha must be coverage in [0,1].')
        if background is None:
            y, x = np.indices((height, width))
            checker = np.where(((x // 32 + y // 32) % 2)[:, :, None], 0.65, 0.35).astype(np.float32)
            backdrop = np.repeat(checker, 3, axis=2)
        else:
            if len(background) != 3 or not all(np.isfinite(v) and 0 <= v <= 1 for v in background):
                raise ValueError('Background must be three finite sRGB values in [0,1].')
            backdrop = np.broadcast_to(np.array(background, dtype=np.float32), rgb.shape).copy()
        # Explicitly composite straight foreground and coverage in linear light.
        foreground = rgb if linear else srgb_to_linear(rgb)
        rgb = linear_to_srgb(foreground * alpha + srgb_to_linear(backdrop) * (1 - alpha))
    elif linear:
        rgb = linear_to_srgb(rgb)
    rgb = np.asarray(rgb, dtype=np.float32)
    if not np.isfinite(rgb).all():
        raise ValueError('Preview contains nonfinite values.')
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    pixels = np.ascontiguousarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    note = ('Coverage display' if coverage else 'sRGB display; HDR/negative values clip only in this viewer')
    if not coverage and source_colour_profile != 'legacy':
        note += f'; source {source_colour_profile} → working BT.709 D65'
    if fps is not None:
        note += '; OpenCV video seek, VFR alignment unverified'
    return PreviewFrame(pixels, count, width, height, fps, note)


@dataclass(frozen=True)
class PreviewPair:
    a: PreviewFrame
    b: PreviewFrame


def preview_pair(a: dict, b: dict) -> PreviewPair:
    """Publish both thumbnails together or fail the entire comparison."""
    first, second = preview(**a), preview(**b)
    if (first.width, first.height) != (second.width, second.height):
        raise ValueError('Comparison frames have different native geometry.')
    return PreviewPair(first, second)
