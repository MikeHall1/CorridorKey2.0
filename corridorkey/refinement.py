"""Optional native CPU alpha adjustment; smoothing is not an accuracy claim."""
import math
import numpy as np


def finish_alpha(alpha, *, expansion=0, feather=0, region=None, check=None):
    """Native coverage: signed square max/min radius, then finite Gaussian feather.

    Replicate image borders. 256-pixel tiles with exact combined halos bound
    filter scratch independently of image dimensions; only the result is full size.
    RGB and temporal state are deliberately outside this operation.
    """
    if type(expansion) is not int or not -32 <= expansion <= 32:
        raise ValueError('alpha expansion must be an integer in [-32,32]')
    if type(feather) is not int or not 0 <= feather <= 32:
        raise ValueError('alpha feather must be an integer in [0,32]')
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim != 2 or min(alpha.shape) < 1:
        raise ValueError('edge finishing requires nonempty 2D coverage')
    if region is not None:
        region = np.asarray(region, dtype=np.float32)
        if region.shape != alpha.shape:
            raise ValueError('edge finishing region must match native coverage')
    if check is not None:
        check()
    import cv2
    result = np.empty_like(alpha)
    halo = abs(expansion) + feather
    kernel = np.ones((2 * abs(expansion) + 1,) * 2, dtype=np.uint8) if expansion else None
    for y in range(0, alpha.shape[0], 256):
        for x in range(0, alpha.shape[1], 256):
            if check is not None:
                check()
            y0, x0 = max(0, y - halo), max(0, x - halo)
            y1, x1 = min(alpha.shape[0], y + 256 + halo), min(alpha.shape[1], x + 256 + halo)
            tile = alpha[y0:y1, x0:x1]
            if not np.isfinite(tile).all() or np.any((tile < 0) | (tile > 1)):
                raise ValueError('edge finishing requires finite coverage in [0,1]')
            if expansion:
                assert kernel is not None
                operation = cv2.dilate if expansion > 0 else cv2.erode
                tile = operation(tile, kernel, borderType=cv2.BORDER_REPLICATE)
            if feather:
                tile = cv2.GaussianBlur(tile, (2 * feather + 1,) * 2, max(.5, feather / 2),
                                        borderType=cv2.BORDER_REPLICATE)
                tile = np.clip(tile, 0, 1)
            h, w = min(256, alpha.shape[0] - y), min(256, alpha.shape[1] - x)
            finished = tile[y - y0:y - y0 + h, x - x0:x - x0 + w]
            if region is not None:
                weight = region[y:y + h, x:x + w]
                if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
                    raise ValueError('edge finishing region requires finite coverage in [0,1]')
                original = alpha[y:y + h, x:x + w]
                blend = original * (1 - weight) + finished * weight
                finished = np.where(weight == 0, original, np.where(weight == 1, finished, blend))
            result[y:y + h, x:x + w] = finished
    if check is not None:
        check()
    return result


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
