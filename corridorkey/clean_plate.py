"""Optional one-frame clean-plate guidance, never a learned matte or acceptance.

Inputs must already be straight scene-linear float32 RGB in the engine's assumed
BT.709/D65 working basis. Callers own explicit shared transfer/primary decoding.
Translation registration, exposure fitting and class thresholds are heuristics.
No input, correction, render, project, asset or model is modified or published.
Native OpenCV operations are not forcibly interruptible; checks bracket them.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

KERNEL_VERSION = 'clean-plate-guidance-v1'
INPUT_CONTRACT = 'straight_scene_linear_bt709_d65_float32'
MAX_PIXELS = 4 * 1024**2
MAX_WORKING_BYTES = 768 * 1024**2


def _integer(value, name, low=0, high=2**31 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an exact integer in [{low}, {high}].')


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in [{low}, {high}].')


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be lowercase SHA256 text.')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PlateMapping:
    """Absolute source selection and absolute reference-index correspondence.

    reference_count is the total available reference count from index zero.
    sequence maps source_start to reference_start + frame_offset; single keeps
    that same reference for every selected source frame. No cycling/resampling.
    Sequence timing compatibility must be established by the caller, not FPS
    inferred from filenames or averaged VFR decoder metadata.
    """
    mode: Literal['single', 'sequence']
    source_start: int
    source_count: int
    reference_start: int
    reference_count: int
    frame_offset: int = 0

    def validate(self):
        if self.mode not in {'single', 'sequence'}:
            raise ValueError('Plate mapping must be single or sequence.')
        _integer(self.source_start, 'source_start')
        _integer(self.source_count, 'source_count', 1, 300)
        _integer(self.reference_start, 'reference_start')
        _integer(self.reference_count, 'reference_count', 1, 100000)
        _integer(self.frame_offset, 'frame_offset', -(2**31 - 1))
        if self.source_start + self.source_count > 2**31:
            raise ValueError('Selected source range exceeds supported absolute frame indices.')
        first = self.reference_start + self.frame_offset
        last = first + (self.source_count - 1 if self.mode == 'sequence' else 0)
        if not 0 <= first <= last < self.reference_count:
            raise ValueError('Plate mapping does not cover the complete selected source range.')

    def reference_frame(self, source_frame: int) -> int:
        self.validate()
        _integer(source_frame, 'source_frame')
        offset = source_frame - self.source_start
        if not 0 <= offset < self.source_count:
            raise ValueError('Source frame is outside the declared plate mapping.')
        return self.reference_start + self.frame_offset + (offset if self.mode == 'sequence' else 0)


@dataclass(frozen=True)
class CleanPlateProvenance:
    source_identity: str
    reference_identity: str
    source_decode_identity: str
    reference_decode_identity: str
    source_frame: int
    reference_frame: int
    mapping: PlateMapping
    input_contract: str  # Required explicit declaration, never inferred.

    def validate(self):
        for name in ('source_identity', 'reference_identity', 'source_decode_identity', 'reference_decode_identity'):
            _sha(getattr(self, name), name)
        if not isinstance(self.mapping, PlateMapping):
            raise TypeError('Plate mapping must be a PlateMapping instance.')
        if self.input_contract != INPUT_CONTRACT:
            raise ValueError('Clean-plate inputs require explicit straight scene-linear float32 working BT.709/D65 declaration.')
        _integer(self.reference_frame, 'reference_frame')
        if self.mapping.reference_frame(self.source_frame) != self.reference_frame:
            raise ValueError('Provided reference frame differs from explicit source/reference mapping.')


@dataclass(frozen=True)
class CleanPlateSettings:
    alignment: Literal['declared', 'estimate_translation']
    translation_xy: tuple[float, float] = (0., 0.)
    compensate_exposure: bool = False
    min_gain: float = .25
    max_gain: float = 4.
    max_translation_pixels: float = 32.  # Per-axis absolute native-pixel limit.
    proxy_max_side: int = 1024
    min_texture: float = .003
    min_phase_response: float = .15
    min_sample_pixels: int = 32
    min_valid_fraction: float = .7
    sample_error_threshold: float = .15
    min_sample_match_fraction: float = .8
    background_threshold: float = .04
    foreground_threshold: float = .35
    shadow_chroma_threshold: float = .04
    foreground_chroma_threshold: float = .12
    screen_radius: int = 1
    seed_erosion: int = 1
    max_pixels: int = MAX_PIXELS
    max_working_bytes: int = MAX_WORKING_BYTES

    def validate(self):
        if self.alignment not in {'declared', 'estimate_translation'}:
            raise ValueError('Alignment must be declared or estimate_translation.')
        if (type(self.translation_xy) is not tuple or len(self.translation_xy) != 2):
            raise ValueError('Translation must be a two-element tuple of native-pixel offsets.')
        _number(self.max_translation_pixels, 'max_translation_pixels', 0., 256.)
        for value in self.translation_xy:
            _number(value, 'translation', -self.max_translation_pixels, self.max_translation_pixels)
        if self.alignment == 'estimate_translation' and self.translation_xy != (0., 0.):
            raise ValueError('Estimated alignment cannot silently combine a declared translation.')
        if type(self.compensate_exposure) is not bool:
            raise ValueError('compensate_exposure must be a boolean.')
        _number(self.min_gain, 'min_gain', .01, 100.)
        _number(self.max_gain, 'max_gain', .01, 100.)
        if self.min_gain >= self.max_gain:
            raise ValueError('Exposure gain bounds must be ordered.')
        for name in ('min_texture', 'sample_error_threshold', 'background_threshold', 'foreground_threshold',
                     'shadow_chroma_threshold', 'foreground_chroma_threshold'):
            _number(getattr(self, name), name, 1e-6, 100.)
        for name in ('min_phase_response', 'min_valid_fraction', 'min_sample_match_fraction'):
            _number(getattr(self, name), name, 0., 1.)
        if self.background_threshold >= self.foreground_threshold or self.shadow_chroma_threshold >= self.foreground_chroma_threshold:
            raise ValueError('Guidance thresholds must leave an explicit unknown band.')
        _integer(self.proxy_max_side, 'proxy_max_side', 16, 1024)
        _integer(self.min_sample_pixels, 'min_sample_pixels', 4, MAX_PIXELS)
        _integer(self.screen_radius, 'screen_radius', 0, 31)
        _integer(self.seed_erosion, 'seed_erosion', 0, 8)
        _integer(self.max_pixels, 'max_pixels', 1, MAX_PIXELS)
        _integer(self.max_working_bytes, 'max_working_bytes', 1, MAX_WORKING_BYTES)


@dataclass(frozen=True)
class CleanPlateCandidate:
    height: int
    width: int
    usable: bool
    trimap_bytes: bytes = field(repr=False)
    screen_field_bytes: bytes = field(repr=False)
    residual_bytes: bytes = field(repr=False)
    validity_bytes: bytes = field(repr=False)
    provenance_json: str
    diagnostics_json: str
    identity: str

    def labels(self) -> NDArray[np.uint8]:
        return np.frombuffer(self.trimap_bytes, np.uint8).reshape(self.height, self.width)

    def screen_field(self) -> NDArray[np.float32]:
        return np.frombuffer(self.screen_field_bytes, '<f4').reshape(self.height, self.width, 3)

    def residual(self) -> NDArray[np.float32]:
        return np.frombuffer(self.residual_bytes, '<f4').reshape(self.height, self.width)

    def validity(self) -> NDArray[np.uint8]:
        return np.frombuffer(self.validity_bytes, np.uint8).reshape(self.height, self.width)

    def guidance_bytes(self) -> bytes:
        if not self.usable:
            raise ValueError('Unusable clean-plate candidate cannot be accepted as guidance.')
        return self.trimap_bytes

    def report(self) -> dict:
        return {'identity': self.identity, 'provenance': json.loads(self.provenance_json),
                'diagnostics': json.loads(self.diagnostics_json)}


def assist_clean_plate(source_rgb: NDArray[np.float32], reference_rgb: NDArray[np.float32],
                       background_samples: NDArray, *, settings: CleanPlateSettings,
                       provenance: CleanPlateProvenance,
                       check: Callable[[], None] | None = None) -> CleanPlateCandidate:
    check = check or (lambda: None)
    check()
    if not isinstance(settings, CleanPlateSettings) or not isinstance(provenance, CleanPlateProvenance):
        raise TypeError('Settings and provenance must be validated dataclass instances.')
    settings.validate()
    provenance.validate()
    for image in (source_rgb, reference_rgb):
        if (not isinstance(image, np.ndarray) or image.dtype != np.float32 or image.ndim != 3
                or image.shape[-1] != 3 or min(image.shape[:2]) < 4):
            raise ValueError('Clean-plate RGB must be native float32 HWC three-channel arrays, at least 4x4.')
    if source_rgb.shape != reference_rgb.shape:
        raise ValueError('Source and reference must have matching native geometry; no implicit resizing/alignment.')
    height, width = source_rgb.shape[:2]
    if height * width > settings.max_pixels:
        raise ValueError('Clean-plate native pixel budget exceeded.')
    estimated_bytes = height * width * 320 + min(height, settings.proxy_max_side) * min(width, settings.proxy_max_side) * 24
    if estimated_bytes > settings.max_working_bytes:
        raise ValueError('Clean-plate estimated working-byte budget exceeded; native library allocations remain outside this estimate.')
    if (not isinstance(background_samples, np.ndarray) or background_samples.shape != (height, width)
            or background_samples.dtype not in (np.bool_, np.uint8)):
        raise ValueError('Background samples must be a native binary bool/uint8 mask.')
    if np.any((background_samples != 0) & (background_samples != 1)):
        raise ValueError('Background samples must contain exact binary 0/1 values, not coverage or trimap classes.')
    if not np.isfinite(source_rgb).all() or not np.isfinite(reference_rgb).all():
        raise ValueError('Clean-plate RGB contains nonfinite values.')
    check()
    source, reference = source_rgb.copy(), reference_rgb.copy()
    samples = background_samples.astype(np.uint8, copy=True)
    context = {'identity_scope': 'Hashes witness the actual supplied RGB/mask bytes plus caller-declared source/reference/decode identities; file/model authenticity is not verified.',
               'kernel_version': KERNEL_VERSION, 'opencv_version': cv2.__version__, 'settings': asdict(settings),
               'binding': asdict(provenance),
               'input_hashes': {'source_rgb_f32le': _digest(source.astype('<f4', copy=False).tobytes()),
                                'reference_rgb_f32le': _digest(reference.astype('<f4', copy=False).tobytes()),
                                'background_samples_u8': _digest(samples.tobytes())}}
    diagnostics = {'usable': False, 'status': 'unusable', 'estimated_working_bytes': estimated_bytes,
                   'memory_scope': 'Admission estimate and explicit pixel/proxy limits; not a process-wide hard RAM limit.',
                   'alignment': {'requested': settings.alignment, 'translation_reference_to_source_xy': None,
                                 'phase_response': None, 'scope': 'Translation/photometric heuristics, not calibrated confidence or geometric proof.'},
                   'exposure': {'requested': settings.compensate_exposure, 'gain': None},
                   'warnings': ['Similar-coloured foreground can be indistinguishable from the empty plate and may be classified as background.',
                                'Rough guidance requires visual inspection and deliberate acceptance; it is not an alpha matte or new neural conditioning.'],
                   'actual_inference': 'NOT PERFORMED', 'quality_benefit': 'NOT VERIFIED'}
    check()

    def finish(usable, labels=None, field=None, residual=None, valid=None, reason=None):
        check()
        labels = np.full((height, width), 128, np.uint8) if labels is None else labels
        field = np.zeros((height, width, 3), np.float32) if field is None else field
        residual = np.zeros((height, width), np.float32) if residual is None else residual
        valid = np.zeros((height, width), np.uint8) if valid is None else valid
        diagnostics.update({'usable': usable, 'status': 'candidate' if usable else 'unusable', 'reason': reason,
                            'class_counts': {name: int(np.count_nonzero(labels == value)) for name, value in
                                             (('background', 0), ('unknown', 128), ('foreground', 255))}})
        blobs = (labels.tobytes(), field.astype('<f4', copy=False).tobytes(),
                 residual.astype('<f4', copy=False).tobytes(), valid.astype(np.uint8, copy=False).tobytes())
        context['output_hashes'] = {name: _digest(blob) for name, blob in zip(('trimap', 'screen_field', 'residual', 'validity'), blobs)}
        context_json, diagnostics_json = _canonical(context), _canonical(diagnostics)
        identity = _digest(context_json.encode('utf-8'))
        check()
        return CleanPlateCandidate(height, width, usable, *blobs, context_json, diagnostics_json, identity)

    if int(np.count_nonzero(samples)) < settings.min_sample_pixels:
        return finish(False, reason='Insufficient artist-declared background sample support.')
    luminance = np.array([.2126, .7152, .0722], np.float32)
    dx, dy = (float(v) for v in settings.translation_xy)
    if settings.alignment == 'estimate_translation':
        scale = min(1., settings.proxy_max_side / max(height, width))
        proxy_hw = (max(4, round(width * scale)), max(4, round(height * scale)))
        ref_gray, src_gray = reference @ luminance, source @ luminance
        if scale < 1:
            ref_gray = cv2.resize(ref_gray, proxy_hw, interpolation=cv2.INTER_AREA)
            src_gray = cv2.resize(src_gray, proxy_hw, interpolation=cv2.INTER_AREA)
        with np.errstate(over="ignore", invalid="ignore"):
            ref_texture, src_texture = float(ref_gray.std()), float(src_gray.std())
        diagnostics['alignment'].update({'proxy_hw': [ref_gray.shape[0], ref_gray.shape[1]],
                                         'reference_texture_std': ref_texture if math.isfinite(ref_texture) else None,
                                         'source_texture_std': src_texture if math.isfinite(src_texture) else None,
                                         'estimator': 'opencv-phase-correlation-full-proxy-hanning',
                                         'estimator_limits': 'The full proxy is used to avoid mask-edge registration bias. Moving foreground, repeated patterns and illumination changes can bias its result; artist samples validate the candidate separately.'})
        if not all(math.isfinite(v) for v in (ref_texture, src_texture)):
            return finish(False, reason='Translation texture analysis overflowed float32; no alignment accepted.')
        if min(ref_texture, src_texture) < settings.min_texture:
            return finish(False, reason='Insufficient texture for translation estimation; explicitly declare alignment or choose a suitable reference/sample.')
        check()
        window = cv2.createHanningWindow((ref_gray.shape[1], ref_gray.shape[0]), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(ref_gray, src_gray, window)
        dx, dy = shift[0] * width / ref_gray.shape[1], shift[1] * height / ref_gray.shape[0]
        diagnostics['alignment']['phase_response'] = float(response) if math.isfinite(response) else None
        if not all(math.isfinite(v) for v in (dx, dy, response)) or response < settings.min_phase_response:
            return finish(False, reason='Translation response is unusable; no fallback alignment accepted.')
    diagnostics['alignment']['translation_reference_to_source_xy'] = [dx, dy]
    if max(abs(dx), abs(dy)) > settings.max_translation_pixels:
        return finish(False, reason='Translation exceeds the requested native-pixel bound.')
    check()
    transform = np.array([[1., 0., dx], [0., 1., dy]], np.float32)
    aligned = cv2.warpAffine(reference, transform, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0., 0., 0.))
    coverage = cv2.warpAffine(np.ones((height, width), np.float32), transform, (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0., 0., 0.))
    valid = coverage >= 1 - 1e-6
    diagnostics['alignment']['valid_fraction'] = float(valid.mean())
    if valid.mean() < settings.min_valid_fraction:
        return finish(False, reason='Insufficient aligned reference coverage; uncovered pixels cannot guide the result.')
    sample_valid = samples.astype(bool) & valid
    ref_luma, src_luma = aligned @ luminance, source @ luminance
    sample_valid &= (ref_luma > 1e-5) & (src_luma > 1e-5)
    sample_count = int(np.count_nonzero(sample_valid))
    diagnostics['exposure']['sample_count'] = sample_count
    if sample_count < settings.min_sample_pixels:
        return finish(False, reason='Insufficient aligned positive-luminance background samples for checking exposure/alignment.')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        gain = float(np.median(src_luma[sample_valid] / ref_luma[sample_valid])) if settings.compensate_exposure else 1.
    diagnostics['exposure']['gain'] = gain if math.isfinite(gain) else None
    if not math.isfinite(gain) or not settings.min_gain <= gain <= settings.max_gain:
        return finish(False, reason='Exposure gain exceeds the requested bound; it was not clamped.')
    with np.errstate(over='ignore', invalid='ignore'):
        aligned = aligned * np.float32(gain)
    if not np.isfinite(aligned).all():
        return finish(False, reason='Exposure-adjusted reference overflowed float32.')
    check()
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        sample_difference = np.linalg.norm(source - aligned, axis=2)
        sample_scale = np.linalg.norm(aligned, axis=2)
        sample_error = sample_difference / np.maximum(sample_scale, np.float32(.01))
    if not all(np.isfinite(value).all() for value in (sample_difference, sample_scale, sample_error)):
        return finish(False, reason='Background sample comparison overflowed float32; no repair or accepted guidance.')
    match_fraction = float(np.mean(sample_error[sample_valid] <= settings.sample_error_threshold))
    diagnostics['alignment'].update({'sample_relative_error_median': float(np.median(sample_error[sample_valid])),
                                     'sample_match_fraction': match_fraction,
                                     'sample_validation_limits': 'Mismatch may reflect misalignment, occlusion, exposure/colour differences or inaccurate sample declarations; this does not identify a unique cause.'})
    if match_fraction < settings.min_sample_match_fraction:
        return finish(False, reason='Declared/estimated alignment and exposure are inconsistent with the background samples; no guidance may be accepted.')
    if settings.screen_radius:
        size = 2 * settings.screen_radius + 1
        denominator = cv2.GaussianBlur(valid.astype(np.float32), (size, size), 0)
        numerator = cv2.GaussianBlur(aligned * valid[..., None], (size, size), 0)
        field = numerator / np.maximum(denominator[..., None], np.float32(1e-6))
        valid &= denominator >= .999
    else:
        field = aligned.copy()
    check()
    diagnostics['alignment']['screen_field_valid_fraction'] = float(valid.mean())
    if valid.mean() < settings.min_valid_fraction:
        return finish(False, reason='Insufficient screen-field support after alignment/filtering; no guidance may be accepted.')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        residual_difference = np.linalg.norm(source - field, axis=2)
        residual_scale = np.linalg.norm(field, axis=2)
        residual = residual_difference / np.maximum(residual_scale, np.float32(.01))
        positive_source, positive_field = np.maximum(source, 0), np.maximum(field, 0)
        source_sum = positive_source.sum(axis=2, keepdims=True)
        field_sum = positive_field.sum(axis=2, keepdims=True)
        source_chroma = positive_source / np.maximum(source_sum, np.float32(1e-6))
        field_chroma = positive_field / np.maximum(field_sum, np.float32(1e-6))
    chroma_error = np.linalg.norm(source_chroma - field_chroma, axis=2)
    if not all(np.isfinite(value).all() for value in (residual_difference, residual_scale, residual, field, source_sum, field_sum, chroma_error)):
        return finish(False, reason='Analysis overflowed float32; no repair or accepted guidance.')
    shadow_or_similar = valid & (chroma_error <= settings.shadow_chroma_threshold) & (residual > settings.background_threshold)
    bg = valid & (residual <= settings.background_threshold)
    fg = valid & (residual >= settings.foreground_threshold) & (chroma_error >= settings.foreground_chroma_threshold) & ~shadow_or_similar
    if settings.seed_erosion:
        kernel = np.ones((2 * settings.seed_erosion + 1, 2 * settings.seed_erosion + 1), np.uint8)
        bg = cv2.erode(bg.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=(0.,)).astype(bool)
        fg = cv2.erode(fg.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=(0.,)).astype(bool)
    if not bg.any() and not fg.any():
        return finish(False, reason='No conservative foreground/background seeds remain; inspect and adjust sampling/thresholds before acceptance.')
    labels = np.full((height, width), 128, np.uint8)
    labels[bg], labels[fg] = 0, 255
    diagnostics.update({'shadow_or_similar_unknown_pixels': int(np.count_nonzero(shadow_or_similar)),
                        'screen_field': 'Exposure-adjusted aligned reference, optionally normalized Gaussian-smoothed in native working linear RGB.',
                        'residual': 'L2 working-linear RGB difference divided by reference L2 magnitude with floor 0.01; heuristic suspicion, not confidence.',
                        'chroma_analysis': 'Negative working RGB is clipped to zero only in the chromaticity heuristic; original inputs and the RGB screen field are not clipped.',
                        'trimap_contract': {'background': 0, 'unknown': 128, 'foreground': 255,
                                            'integration': 'Existing final-only correction/trimap refinement after deliberate acceptance; no new neural channels.'}})
    check()
    return finish(True, labels, field, residual.astype(np.float32), valid.astype(np.uint8))


def screen_sample_reference(source_rgb, points, *, sample_radius=3, influence_radius=256,
                            max_pixels=MAX_PIXELS, check=None):
    """Build an explicitly synthetic screen from native source patch medians.

    Inverse-square interpolation is an uncalibrated spatial colour heuristic.
    Row tiles bound temporary work independently of sample count; no neural work
    or geometric registration is performed. Unsupported pixels remain unknown.
    """
    check = check or (lambda: None)
    check()
    _integer(max_pixels, 'max_pixels', 1, MAX_PIXELS)
    _integer(sample_radius, 'sample_radius', 1, 31)
    _integer(influence_radius, 'influence_radius', 1, 32768)
    if (not isinstance(source_rgb, np.ndarray) or source_rgb.dtype != np.float32
            or source_rgb.ndim != 3 or source_rgb.shape[-1] != 3
            or min(source_rgb.shape[:2]) < 4 or max(source_rgb.shape[:2]) > 32768
            or source_rgb.shape[0] * source_rgb.shape[1] > max_pixels):
        raise ValueError('Screen samples require bounded native float32 RGB geometry.')
    if not np.isfinite(source_rgb).all():
        raise ValueError('Screen samples require finite working-linear RGB.')
    if not isinstance(points, list) or not 1 <= len(points) <= 32:
        raise ValueError('Choose between1 and32 native screen sample points.')
    height, width = source_rgb.shape[:2]
    selected, colours = [], []
    samples = np.zeros((height, width), np.uint8)
    for point in points:
        check()
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('Each screen sample must be a native [x,y] pair.')
        x, y = point
        _integer(x, 'sample x', 0, width - 1)
        _integer(y, 'sample y', 0, height - 1)
        if point in selected:
            raise ValueError('Screen sample points must be unique.')
        selected.append(point.copy())
        patch = (slice(max(0, y - sample_radius), min(height, y + sample_radius + 1)),
                 slice(max(0, x - sample_radius), min(width, x + sample_radius + 1)))
        colours.append(np.median(source_rgb[patch].reshape(-1, 3), axis=0))
        samples[patch] = 1
    field = np.empty_like(source_rgb)
    support = np.empty((height, width), np.bool_)
    xx = np.arange(width, dtype=np.float64)[None, :]
    for start in range(0, height, 128):
        check()
        stop = min(start + 128, height)
        yy = np.arange(start, stop, dtype=np.float64)[:, None]
        numerator = np.zeros((stop - start, width, 3), np.float64)
        denominator = np.zeros((stop - start, width), np.float64)
        nearest = np.full((stop - start, width), np.inf, np.float64)
        for (x, y), colour in zip(selected, colours, strict=True):
            check()
            distance = (xx - x)**2 + (yy - y)**2
            np.minimum(nearest, distance, out=nearest)
            weight = 1. / np.maximum(distance, .25)
            numerator += weight[..., None] * colour
            denominator += weight
        field[start:stop] = (numerator / denominator[..., None]).astype(np.float32)
        support[start:stop] = nearest <= influence_radius**2
    check()
    return field, samples, support, {
        'kernel_version': 'screen-colour-idw-v1', 'points': selected,
        'sample_radius': sample_radius, 'influence_radius': influence_radius,
        'sample_colours_linear_bt709': [colour.tolist() for colour in colours],
        'interpolation': 'native inverse-square distance with0.25 square-pixel floor;128-row tiles',
        'outside_influence': 'unknown trimap classes; not a processing-region gate',
        'scope': 'Synthetic screen from artist-declared background samples, not an empty plate or calibrated foreground model.',
    }


def restrict_screen_candidate(candidate, support, sampling):
    """Bind synthetic-reference provenance and remove unsupported guidance seeds."""
    if not isinstance(candidate, CleanPlateCandidate):
        raise TypeError('Screen guidance requires a CleanPlateCandidate instance.')
    if (not isinstance(support, np.ndarray) or support.dtype != np.bool_
            or support.shape != (candidate.height, candidate.width)):
        raise ValueError('Screen influence support must be a native boolean mask.')
    if (not isinstance(sampling, dict) or sampling.get('kernel_version') != 'screen-colour-idw-v1'
            or len(_canonical(sampling).encode()) > 16 * 1024):
        raise ValueError('Screen sampling provenance must be a bounded supported record.')
    labels = candidate.labels().copy()
    labels[~support] = 128
    valid = candidate.validity().copy()
    valid[~support] = 0
    context, diagnostics = json.loads(candidate.provenance_json), json.loads(candidate.diagnostics_json)
    context['screen_sampling'] = sampling
    labels_raw, valid_raw = labels.tobytes(), valid.tobytes()
    context['output_hashes'].update(trimap=_digest(labels_raw), validity=_digest(valid_raw))
    usable = candidate.usable and bool(np.any(labels != 128))
    diagnostics.update(usable=usable, status='candidate' if usable else 'unusable',
                       reason=diagnostics['reason'] if candidate.usable is False else
                              (None if usable else 'No guidance seeds remain inside the declared screen influence.'),
                       screen_field='Explicit synthetic spatial screen from working-linear source patch medians; not an empty plate.',
                       screen_sampling=sampling,
                       class_counts={name: int(np.count_nonzero(labels == value)) for name, value in
                                     (('background', 0), ('unknown', 128), ('foreground', 255))})
    diagnostics['warnings'][0] = 'Screen-coloured foreground and contaminated artist samples can be mistaken for background.'
    diagnostics['warnings'].append('This synthetic reference is not an empty plate; sample contamination, spatial extrapolation and lighting changes require visual inspection. No plate alignment or background-change proof is available.')
    diagnostics['warnings'].append('Outside sample reach remains unknown; accepted trimaps still use existing full-frame unknown-class refinement, not a neural or output-region gate.')
    context_json = _canonical(context)
    return CleanPlateCandidate(candidate.height, candidate.width, usable, labels_raw, candidate.screen_field_bytes,
                               candidate.residual_bytes, valid_raw, context_json, _canonical(diagnostics),
                               _digest(context_json.encode()))
