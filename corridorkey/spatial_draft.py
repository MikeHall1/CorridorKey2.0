"""Explicit spatial-draft geometry, coverage domains and owned native guidance.

The full-resolution branch bypasses these resamplers. Reduced model work does
not imply smaller native exports, proportional RAM use, speed or quality parity.
"""
from __future__ import annotations

from typing import Any


def describe_spatial_draft(meta, *, shape_quantum):
    if float(meta.get('model_scale', 1.0)) == 1.0:
        return False
    native_h, native_w = meta['native_hw']
    reference = [((size + shape_quantum - 1) // shape_quantum) * shape_quantum for size in (native_h, native_w)]
    return {
        'enabled': True, 'version': 'spatial-draft-v1', 'requested_scale': float(meta['model_scale']),
        'model_native_hw': list(meta['model_native_hw']), 'shape_hw': list(meta['shape_hw']),
        'output_hw': list(meta['output_hw']), 'native_reference_shape_hw': reference,
        'model_grid_reduced': meta['shape_hw'][0] * meta['shape_hw'][1] < reference[0] * reference[1],
        'input_rgb_resampling': 'scene-linear OpenCV INTER_LANCZOS4 before model encoding and SDR proxy',
        'output_alpha_resampling': 'torch bilinear align_corners=False then coverage clamp',
        'output_rgb_resampling': 'scene-linear torch bilinear align_corners=False before transfer/despill/composition',
        'conditioning_resampling': 'OpenCV INTER_LINEAR model coverage to native constraints; INTER_AREA back to model',
        'refinement_domain': 'native output pixels; independently owned native source guidance',
        'buffer_scope': 'native resize/refinement/output RAM excluded from managed temporary-map payload ceiling',
        'quality': 'explicit approximation; real-model quality and speed not verified',
    }


def resize_coverage(alpha, shape_hw, *, down=False):
    """Coverage only; no RGB transfer, with explicit bounded interpolation."""
    import cv2
    import numpy as np

    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim != 2 or not np.isfinite(alpha).all():
        raise ValueError('Spatial draft coverage must be finite HW float32.')
    if tuple(alpha.shape) == tuple(shape_hw):
        return alpha
    height, width = shape_hw
    result = cv2.resize(np.clip(alpha, 0, 1), (width, height),
                        interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR)
    return np.clip(result, 0, 1).astype(np.float32, copy=False)


class NativeGuidance:
    """One live forward-only source owner, reopened for backward/repeated reads."""
    def __init__(self, factory, *, check):
        self.factory, self.check, self.reader = factory, check, None
        self.stats: dict[str, Any] = {'frames_requested': 0, 'source_frames_decoded': 0, 'reader_restarts': 0,
                      'status': 'idle', 'scope': 'additional native source decoding for spatial-draft refinement'}
        self.opened = False

    def read(self, frame):
        self.check()
        if self.reader is not None and frame < self.reader.next_frame:
            old, self.reader = self.reader, None
            old.close()
        if self.reader is None:
            self.reader = self.factory()
            self.stats['reader_restarts'] += int(self.opened)
            self.opened = True
        reader = self.reader
        before = reader.frames_decoded
        self.stats['frames_requested'] += 1
        self.stats['status'] = 'reading'
        try:
            result = reader.read(frame)
            self.check()
            self.stats['status'] = 'ready'
            return result.rgb
        except BaseException:
            self.stats['status'] = 'failed'
            raise
        finally:
            self.stats['source_frames_decoded'] += reader.frames_decoded - before

    def close(self):
        reader, self.reader = self.reader, None
        try:
            if reader is not None:
                reader.close()
        finally:
            self.stats['status'] = 'closed'
