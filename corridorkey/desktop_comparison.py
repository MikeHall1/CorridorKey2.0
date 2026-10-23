"""Source-coordinate mapping for completed renders; no decoding or model work."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .session import JobRequest


@dataclass(frozen=True)
class RenderView:
    source: str
    start: int
    count: int
    height: int
    width: int
    fps: float
    outputs: tuple[tuple[str, str], ...]
    linear: bool
    source_linear: bool
    source_alpha: str
    source_colour_profile: str = 'legacy'

    @property
    def end(self) -> int:
        return self.start + self.count

    def options(self, mode: str, source_index: int, background=None) -> dict[str, Any]:
        if type(source_index) is not int or not self.start <= source_index < self.end:
            raise ValueError(f'Source frame must be in [{self.start}, {self.end}).')
        if mode == 'Source':
            return {'path': self.source, 'index': source_index, 'linear': self.source_linear,
                    'alpha_mode': self.source_alpha, 'source_colour_profile': self.source_colour_profile, 'expected_hw': (self.height, self.width)}
        kind = 'alpha' if mode == 'Alpha' else 'fg'
        if mode not in {'Alpha', 'Foreground', 'Checker composite', 'Solid composite'}:
            raise ValueError('Unknown comparison view.')
        outputs = dict(self.outputs)
        if kind not in outputs:
            raise ValueError(f'This render did not export {kind}.')
        options: dict[str, Any] = {'path': outputs[kind], 'index': source_index - self.start,
                                  'coverage': kind == 'alpha', 'linear': self.linear,
                                  'expected_count': self.count, 'expected_hw': (self.height, self.width)}
        if 'composite' in mode:
            if 'alpha' not in outputs:
                raise ValueError('Compositing requires both alpha and foreground exports.')
            if mode == 'Solid composite' and background is None:
                raise ValueError('Choose a background colour first.')
            options.update(alpha_path=outputs['alpha'], background=background if mode == 'Solid composite' else None)
        return options


def completed_view(job: JobRequest, record: dict[str, Any]) -> RenderView:
    """Capture actual observed selection, never guess a middle/random offset."""
    if record.get('status') != 'complete' or record.get('input') != job.input_path:
        raise ValueError('Comparison requires the matching completed render record.')
    diagnostics = record.get('diagnostics', {})
    source = diagnostics.get('source')
    if source is None:
        raise ValueError('This render has no recorded source range; render again with current diagnostics.')
    if not isinstance(source, dict):
        raise TypeError('Recorded source metadata must be an object.')
    start, count = source.get('start_frame'), source.get('num_frames')
    hw, fps = source.get('native_hw'), source.get('native_fps')
    if (type(start) is not int or start < 0 or type(count) is not int or count < 1
            or diagnostics.get('frames_submitted') != count
            or not isinstance(hw, (list, tuple)) or len(hw) != 2
            or any(type(v) is not int or v < 1 for v in hw)
            or isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(fps) or fps <= 0):
        raise ValueError('Completed render has invalid or incomplete source range/geometry/timing.')
    outputs = diagnostics.get('outputs')
    if not isinstance(outputs, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in outputs.items()):
        raise ValueError('Completed render has invalid output paths.')
    config, _ = job.configurations()
    return RenderView(job.input_path, start, count, hw[0], hw[1], float(fps), tuple(outputs.items()),
                      config.output_format == 'exr', config.frame_dir_linear if Path(job.input_path).is_dir() else False,
                      config.frame_dir_alpha, config.source_colour_profile)


def common_range(a: RenderView, b: RenderView) -> tuple[int, int]:
    if a.source != b.source:
        raise ValueError('Synchronized comparison requires the same source path.')
    if (a.width, a.height) != (b.width, b.height) or not math.isclose(a.fps, b.fps, rel_tol=1e-4, abs_tol=.001):
        raise ValueError('Synchronized comparison requires matching native geometry and frame rate.')
    start, end = max(a.start, b.start), min(a.end, b.end)
    if start >= end:
        raise ValueError('These render source ranges do not overlap.')
    return start, end
