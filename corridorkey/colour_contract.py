"""Explicit linear RGB primary conversion; the engine's BT.709 basis is assumed.

Primary chromaticities and D65 white are from ITU-R BT.2020-2 Table 3 and
BT.709-6 section 1.3:
https://www.itu.int/rec/R-REC-BT.2020-2-201510-I/en
https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
The matrix is inv(normalized BT.709 RGB->XYZ) @ normalized BT.2020 RGB->XYZ,
using R/G/B xy (.708,.292)/(.170,.797)/(.131,.046) and
(.640,.330)/(.300,.600)/(.150,.060), with shared D65 (.3127,.3290).
This is a linear primary conversion, not either recommendation's OETF,
ICC/OCIO handling, HDR mastering, or verification of model training primaries.
"""
from __future__ import annotations

from typing import Any

PROFILES = ('legacy', 'linear_bt709_d65', 'linear_bt2020_d65')
TRANSFORM_VERSION = 'linear-primary-d65-v1'
BT2020_TO_BT709 = (
    (1.6604910021084347, -0.5876411387885492, -0.07284986331988483),
    (-0.12455047452159074, 1.1328998971259594, -0.008349422604369515),
    (-0.018150763354905206, -0.10057889800800739, 1.118729661362913),
)
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
STANDARDS = (
    'https://www.itu.int/rec/R-REC-BT.2020-2-201510-I/en',
    'https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en',
)


def _validate_profile(profile):
    if type(profile) is not str or profile not in PROFILES:
        raise ValueError('source colour profile must be legacy, linear_bt709_d65 or linear_bt2020_d65')


def describe_source_colour(profile, *, scene_linear: bool) -> dict:
    """Finite JSON metadata only; importing/calling this never imports NumPy/Torch."""
    _validate_profile(profile)
    if type(scene_linear) is not bool:
        raise ValueError('scene_linear must be a boolean')
    if profile != 'legacy' and not scene_linear:
        raise ValueError('explicit source primary profiles require scene-linear image sequences')
    return {
        'schema_version': 1,
        'profile': profile,
        'source_transfer': 'scene_linear' if scene_linear else 'display_coded_assumed_srgb',
        'source_primaries': {'legacy': 'assumed_bt709_d65', 'linear_bt709_d65': 'bt709_d65',
                             'linear_bt2020_d65': 'bt2020_d65'}[profile],
        'working_primaries': 'bt709_d65',
        'working_basis': 'engine_assumption_training_primaries_unverified',
        'transform_version': TRANSFORM_VERSION,
        'matrix': [list(row) for row in (BT2020_TO_BT709 if profile == 'linear_bt2020_d65' else IDENTITY)],
        'matrix_dtype': 'float32',
        'conversion_applied': profile == 'linear_bt2020_d65',
        'clipping': 'none_before_proxy/display',
        'alpha': 'coverage_not_transformed',
        'standards': list(STANDARDS),
    }


def to_working_rgb(rgb: Any, profile: str) -> Any:
    """Convert straight float32 HWC RGB without modifying input or clamping gamut.

    Legacy and explicit BT.709 return the same array, preserving signed-zero bits.
    Coverage must be handled separately; RGBA arrays are refused.
    """
    _validate_profile(profile)
    import numpy as np

    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.float32 or rgb.ndim != 3
            or rgb.shape[-1] != 3 or min(rgb.shape[:2]) <= 0):
        raise ValueError('primary conversion requires a nonempty float32 HWC RGB array')
    if not np.isfinite(rgb).all():
        raise ValueError('source RGB contains nonfinite values')
    if profile != 'linear_bt2020_d65':
        return rgb
    matrix = np.asarray(BT2020_TO_BT709, dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        converted = rgb @ matrix.T
    if not np.isfinite(converted).all():
        raise ValueError('linear primary conversion overflowed float32 RGB')
    return converted
