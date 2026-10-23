"""Optional native CPU alpha adjustment; smoothing is not an accuracy claim."""
import math
import numpy as np


def refine_alpha(alpha, guidance, *, radius=0, strength=1.0, sigma=0.1, check=None):
    """Joint RGB/spatial weighted averaging of partially covered pixels only.

    Exactly-zero/one centres are held fixed. Guidance is an explicit display
    proxy. No temporal state or learned confidence is used. Radius 0 is a no-op.
    """
    if type(radius) is not int or not 0 <= radius <= 4:
        raise ValueError("alpha refine radius must be an integer in [0,4]")
    for name, value, minimum in (("strength", strength, 0), ("sigma", sigma, 1e-6)):
        if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= 1:
            raise ValueError(f"invalid alpha refine {name}")
    alpha = np.asarray(alpha, dtype=np.float32)
    guidance = np.asarray(guidance, dtype=np.float32)
    if (alpha.ndim != 2 or min(alpha.shape) < 1 or guidance.shape != (*alpha.shape, 3)
            or not np.isfinite(alpha).all() or not np.isfinite(guidance).all()
            or np.any((alpha < 0) | (alpha > 1)) or np.any((guidance < 0) | (guidance > 1))):
        raise ValueError("refinement requires finite coverage and matching display RGB in [0,1]")
    if check is not None:
        check()
    if radius == 0 or strength == 0:
        return alpha.copy()
    h, w = alpha.shape
    padded_alpha = np.pad(alpha, radius, mode="edge")
    padded_rgb = np.pad(guidance, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    numerator, denominator = np.zeros_like(alpha), np.zeros_like(alpha)
    spatial_sigma = max(0.5, radius / 2)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if check is not None:
                check()
            rgb = padded_rgb[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            distance = np.sum((rgb - guidance) ** 2, axis=-1)
            weight = np.exp(-distance / (2 * sigma * sigma)) * math.exp(-(dx * dx + dy * dy) / (2 * spatial_sigma**2))
            neighbour = padded_alpha[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            numerator += weight * neighbour
            denominator += weight
    filtered = numerator / np.maximum(denominator, 1e-12)
    blend = alpha * (1 - strength) + filtered * strength
    return np.where((alpha > 0) & (alpha < 1), np.clip(blend, 0, 1), alpha).astype(np.float32)
