"""Pure completed-record admission shared by desktop and cache I/O."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MAX_FRAME_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 1_000_000


@dataclass(frozen=True)
class CacheSpec:
    width: int
    height: int
    count: int
    start: int
    fps: float
    source: str
    draft: bool


def cache_spec(record: dict[str, Any]) -> CacheSpec:
    if type(record.get('schema_version')) is not int or record['schema_version'] != 1:
        raise ValueError('Expected shared completed record schema_version=1.')
    if record.get('status') != 'complete':
        raise ValueError('Select a completed engine render.')
    diagnostics = record.get('diagnostics')
    if not isinstance(diagnostics, dict):
        raise TypeError('Completed diagnostics must be an object.')
    source, outputs, contracts = (diagnostics.get(name) for name in ('source', 'outputs', 'output_contracts'))
    if not all(isinstance(value, dict) for value in (source, outputs, contracts)):
        raise TypeError('Completed source/output metadata must be objects.')
    assert isinstance(source, dict) and isinstance(outputs, dict) and isinstance(contracts, dict)
    for kind, channels in [('fg', 3), ('alpha', 1)]:
        contract = contracts.get(kind)
        if (not isinstance(outputs.get(kind), str) or not outputs[kind] or not isinstance(contract, dict)
                or contract.get('format') != 'exr' or type(contract.get('channels')) is not int
                or contract['channels'] != channels):
            raise ValueError('Cache requires completed EXR foreground3 and alpha1 exports.')
    if contracts['fg'].get('rgb_representation') != 'scene_linear':
        raise ValueError('Cache requires scene-linear foreground RGB.')
    hw = source.get('native_hw')
    if not isinstance(hw, (list, tuple)) or len(hw) != 2:
        raise ValueError('Completed native geometry must have height and width.')
    height, width = hw
    start, count, fps = source.get('start_frame'), diagnostics.get('frames_submitted'), source.get('native_fps')
    if type(width) is not int or type(height) is not int or type(count) is not int or type(start) is not int:
        raise TypeError('Cache geometry/count/start must be integers.')
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        raise TypeError('Cache FPS must be numeric and not boolean.')
    if (min(width, height, count) < 1 or count > MAX_FRAMES
            or width * height * 16 > MAX_FRAME_BYTES or type(start) is not int or not 0 <= start < 2**63
            or type(source.get('num_frames')) is not int or source['num_frames'] != count):
        raise ValueError('Unsupported cache range/geometry/frame size/timing.')
    try:
        fps = float(fps)
    except OverflowError as exc:
        raise ValueError('Cache FPS exceeds finite numeric range.') from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Cache FPS must be finite and positive.')
    source_path = record.get('input')
    if not isinstance(source_path, str) or not source_path:
        source_path = 'Source path not recorded'
    if len(source_path) > 4096:
        raise ValueError('Recorded source path exceeds display/admission limit.')
    return CacheSpec(width, height, count, start, fps, source_path,
                     contracts['fg'].get('foreground_mode') == 'source_rgb_draft' or
                     (isinstance(contracts['fg'].get('spatial_draft'), dict) and
                      contracts['fg']['spatial_draft'].get('enabled') is True))
