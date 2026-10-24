"""Optional coarse click guidance, independent of Qt and the matting engine.

Import is stdlib-only. The synchronous session belongs in an owned child, never
on the UI thread. One native operation runs at a time; clear/bind invalidate
logical results while an existing native call retires. Native calls are only
cooperatively cancellable; the external process owner handles forced teardown.
An exclusive resource lease is caller authority, not a GPU allocation guarantee.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import re
import stat
import struct
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .control import JobControl
from .identities import verify_file_observation


class GuidanceUnavailable(RuntimeError):
    """Missing explicit local provider capability; no automatic fallback."""


class StaleGuidance(RuntimeError):
    """An obsolete operation retired without publishing its result."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Source/frame identity must be a lowercase SHA256.')


@dataclass(frozen=True)
class Click:
    x: int
    y: int
    positive: bool

    def validate(self, width: int, height: int) -> None:
        if type(self.x) is not int or type(self.y) is not int or not 0 <= self.x < width or not 0 <= self.y < height or type(self.positive) is not bool:
            raise ValueError('Clicks require native integer XY coordinates and a boolean positive/negative label.')


@dataclass(frozen=True)
class GuidanceFrame:
    source_sha256: str
    source_frame: int
    decode_identity: str
    width: int
    height: int
    rgb_bytes: bytes

    def validate(self, max_pixels: int) -> None:
        _hash(self.source_sha256)
        if type(self.source_frame) is not int or self.source_frame < 0:
            raise ValueError('Guidance frame must be an absolute nonnegative source index.')
        if not isinstance(self.decode_identity, str) or not 1 <= len(self.decode_identity) <= 1024:
            raise ValueError('Record the explicit assumed-sRGB uint8 display proxy conversion in decode_identity.')
        if (type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) < 2
                or self.width * self.height > max_pixels):
            raise ValueError('Guidance native dimensions exceed the explicit pixel budget.')
        if type(self.rgb_bytes) is not bytes or len(self.rgb_bytes) != self.width * self.height * 3:
            raise ValueError('Guidance requires immutable native HWC uint8 assumed-sRGB bytes.')

    @property
    def identity(self) -> str:
        return _sha(_canonical({'source': self.source_sha256, 'frame': self.source_frame,
                    'decode': self.decode_identity, 'width': self.width, 'height': self.height,
                    'rgb_sha256': _sha(self.rgb_bytes)}).encode())


@dataclass(frozen=True)
class GuidanceRequest:
    frame_identity: str
    generation: int
    clicks: tuple[Click, ...]


@dataclass(frozen=True)
class ExclusiveResourceLease:
    device: str
    owner_identity: str
    is_exclusive: Callable[[], bool]

    def check(self, device: str) -> None:
        if (not self.owner_identity or self.device != device or not self.is_exclusive()):
            raise GuidanceUnavailable('An active exclusive resource lease for the requested device is required; retire the matting owner first.')


@dataclass(frozen=True)
class ProviderPrediction:
    mask_bytes: bytes
    mask_format: str
    logits_bytes: bytes | None
    provenance_json: str
    fixture: bool = False


class GuidanceProvider(Protocol):
    device: str

    def set_frame(self, frame: GuidanceFrame, control: JobControl) -> int:
        """Bind one image, returning retained embedding bytes (not peak VRAM)."""
        ...

    def predict(self, clicks: tuple[Click, ...], previous_logits: bytes | None,
                control: JobControl) -> ProviderPrediction: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class GuidanceResult:
    frame_identity: str
    source_sha256: str
    source_frame: int
    decode_identity: str
    generation: int
    width: int
    height: int
    mask_bytes: bytes
    mask_format: str
    provenance_json: str
    requires_acceptance: bool = True

    @property
    def identity(self) -> str:
        return _sha(self.provenance_json.encode())


class GuidanceSession:
    """One bound frame/embedding/result and bounded iterative logits.

    Changing frame or clearing selection invalidates all outstanding generations.
    Reset is deferred to the operation owner, never concurrent with prediction.
    The external owner must retain its lease until close() completes. Budgets
    bound input and retained payloads; native transient allocations are external.
    """
    def __init__(self, provider: GuidanceProvider, lease: ExclusiveResourceLease, *,
                 max_pixels: int = 4 * 1024**2, max_embedding_bytes: int = 256 * 1024**2,
                 max_result_bytes: int = 32 * 1024**2, max_logits_bytes: int = 256 * 256 * 4):
        for value in (max_pixels, max_embedding_bytes, max_result_bytes, max_logits_bytes):
            if type(value) is not int or value < 1:
                raise ValueError('Guidance budgets must be positive exact integers.')
        lease.check(provider.device)
        self.provider, self.lease = provider, lease
        self.max_pixels, self.max_embedding_bytes = max_pixels, max_embedding_bytes
        self.max_result_bytes, self.max_logits_bytes = max_result_bytes, max_logits_bytes
        self._state, self._operation = threading.Lock(), threading.Lock()
        self._frame: GuidanceFrame | None = None
        self._provider_frame: str | None = None
        self._previous_logits: bytes | None = None
        self._previous_clicks: tuple[Click, ...] = ()
        self.last_result: GuidanceResult | None = None
        self.generation = 0
        self.closed = False
        self.cleanup_error: str | None = None

    def bind(self, frame: GuidanceFrame) -> int:
        if not isinstance(frame, GuidanceFrame):
            raise TypeError('Bind a GuidanceFrame instance, not a class.')
        frame.validate(self.max_pixels)
        with self._state:
            if self.closed:
                raise RuntimeError('Guidance session is closed.')
            self._frame = frame
            self._clear_locked()
            return self.generation

    def _clear_locked(self) -> None:
        self.generation += 1
        self._previous_logits = None
        self._previous_clicks = ()
        self.last_result = None

    def clear(self) -> int:
        with self._state:
            if self.closed:
                raise RuntimeError('Guidance session is closed.')
            self._clear_locked()
            return self.generation

    def request(self, clicks: tuple[Click, ...]) -> GuidanceRequest:
        with self._state:
            if self._frame is None or self.closed:
                raise RuntimeError('Bind a source frame before selecting a subject.')
            self.generation += 1  # A newer prompt request fences an older native result.
            self.last_result = None
            return GuidanceRequest(self._frame.identity, self.generation, clicks)

    def predict(self, request: GuidanceRequest, *, control: JobControl | None = None) -> GuidanceResult:
        if not isinstance(request, GuidanceRequest):
            raise TypeError('Predict requires a GuidanceRequest instance, not a class.')
        control = control if control is not None else JobControl()
        control.check()
        if not self._operation.acquire(blocking=False):
            raise RuntimeError('Guidance provider is busy; supersede logically and wait for its owned native operation.')
        try:
            with self._state:
                frame = self._frame
                if self.cleanup_error is not None:
                    raise GuidanceUnavailable('Previous provider cleanup is uncertain; retire its owned process before another selection.')
                if (self.closed or frame is None or type(request.generation) is not int
                        or request.generation != self.generation or request.frame_identity != frame.identity):
                    raise StaleGuidance('Guidance request no longer matches the source/frame/generation.')
                if type(request.clicks) is not tuple or not 1 <= len(request.clicks) <= 256:
                    raise ValueError('Provide one to 256 immutable positive/negative clicks; clear is a separate operation.')
                for click in request.clicks:
                    if not isinstance(click, Click):
                        raise TypeError('Click prompts must be Click instances.')
                    click.validate(frame.width, frame.height)
                if not any(c.positive for c in request.clicks):
                    raise ValueError('Select a subject with at least one positive click before adding negative clicks.')
                previous = (self._previous_logits if request.clicks[:len(self._previous_clicks)] == self._previous_clicks else None)
            self.lease.check(self.provider.device)
            if self._provider_frame != frame.identity:
                self.provider.reset()
                self._provider_frame = None
                control.check()
                self.lease.check(self.provider.device)
                embedding_bytes = self.provider.set_frame(frame, control)
                if type(embedding_bytes) is not int or not 0 <= embedding_bytes <= self.max_embedding_bytes:
                    raise ValueError('Provider retained embedding exceeds the explicit byte budget.')
                self._provider_frame = frame.identity
            control.check()
            self.lease.check(self.provider.device)
            output = self.provider.predict(request.clicks, previous, control)
            control.check()
            self.lease.check(self.provider.device)
            if not isinstance(output, ProviderPrediction):
                raise TypeError('Provider must return a ProviderPrediction instance.')
            size = frame.width * frame.height
            if output.mask_format not in ('binary_u8', 'coverage_f32') or type(output.mask_bytes) is not bytes:
                raise ValueError('Provider must return immutable binary/coverage coarse-mask bytes.')
            expected = size if output.mask_format == 'binary_u8' else size * 4
            if len(output.mask_bytes) != expected or expected > self.max_result_bytes:
                raise ValueError('Provider coarse-mask geometry/payload exceeds bounds.')
            if output.mask_format == 'binary_u8':
                if any(v not in (0, 255) for v in output.mask_bytes):
                    raise ValueError('Binary coarse guidance requires exactly 0/255 bytes.')
            elif any(not math.isfinite(v[0]) or not 0 <= v[0] <= 1 for v in struct.iter_unpack('<f', output.mask_bytes)):
                raise ValueError('Coarse coverage must be finite float32 in [0,1].')
            if (output.logits_bytes is not None and (type(output.logits_bytes) is not bytes
                    or len(output.logits_bytes) > self.max_logits_bytes or len(output.logits_bytes) % 4
                    or any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', output.logits_bytes)))):
                raise ValueError('Provider iterative logits exceed finite float32 byte bounds.')
            if type(output.fixture) is not bool or not isinstance(output.provenance_json, str) or len(output.provenance_json.encode()) > 64 * 1024:
                raise ValueError('Provider provenance must be bounded JSON and explicitly identify fixtures.')
            provenance = {'schema_version': 1, 'role': 'coarse subject guidance; not alpha matting',
                'evidence': 'provider-interface fixture; no SAM2 inference executed' if output.fixture else 'provider execution; visual quality not verified',
                'frame_identity': frame.identity, 'source_sha256': frame.source_sha256,
                'source_frame': frame.source_frame, 'decode_identity': frame.decode_identity,
                'generation': request.generation, 'clicks': [asdict(c) for c in request.clicks],
                'mask_format': output.mask_format, 'mask_sha256': _sha(output.mask_bytes),
                'previous_logits_sha256': None if previous is None else _sha(previous),
                'provider': json.loads(output.provenance_json), 'device': self.provider.device}
            serialized = _canonical(provenance)
            if len(serialized.encode()) > 128 * 1024:
                raise ValueError('Guidance provenance exceeds its explicit byte budget.')
            result = GuidanceResult(frame.identity, frame.source_sha256, frame.source_frame, frame.decode_identity,
                request.generation, frame.width, frame.height, output.mask_bytes, output.mask_format, serialized)
            with self._state:
                if self.closed or request.generation != self.generation or self._frame is not frame:
                    raise StaleGuidance('Obsolete provider result retired after clear/rebind; no selection published.')
                self._previous_logits, self._previous_clicks = output.logits_bytes, request.clicks
                self.last_result = result
            return result
        except BaseException as original:
            # This is the operation owner: safe to reset after native retirement.
            try:
                self.provider.reset()
            except Exception as cleanup:
                logging.getLogger(__name__).exception('Guidance provider cleanup remains unverified')
                self.cleanup_error = f'{type(cleanup).__name__}: {cleanup}'[:2048]
                original.add_note('Guidance provider cleanup NOT VERIFIED: ' + self.cleanup_error)
            self._provider_frame = None
            with self._state:
                self._previous_logits = None
                self._previous_clicks = ()
            raise
        finally:
            self._operation.release()

    def close(self) -> None:
        if not self._operation.acquire(blocking=False):
            raise RuntimeError('Wait for the owned guidance operation to retire before close.')
        try:
            with self._state:
                self.closed = True
                self._frame = None
                self._clear_locked()
            self.provider.close()
            self._provider_frame = None
        finally:
            self._operation.release()


class LocalSAM2Provider:
    """Lazy optional SAM2.1 adapter, explicit local files/device/precision only.

    Real SAM2 execution is NOT VERIFIED here. This adapter uses the official
    build_sam2/SAM2ImagePredictor APIs; no hub/model download helper is called.
    Hydra context is created before importing SAM2 and never clears a caller's
    initialized context. Postprocessing/compilation are explicitly disabled.
    CUDA float32 requires strict inherited matmul/cuDNN settings; this adapter
    refuses TF32/relaxed policy rather than mutating global engine settings.
    Initial regular single-link file identities remain pinned across imports,
    bounded hashing and load fences; native loading is not a filesystem lock.
    """
    def __init__(self, config: Path, checkpoint: Path, *, device: str,
                 lease: ExclusiveResourceLease, precision: str = 'float32'):
        if not re.fullmatch(r'cpu|mps|cuda(?::\d+)?', device) or precision not in ('float32', 'bfloat16'):
            raise ValueError('Select explicit cpu/mps/cuda[:index] and float32 or bfloat16 precision.')
        if device == 'mps' and precision != 'float32':
            raise GuidanceUnavailable('This optional adapter has no verified MPS bfloat16 path; choose explicit float32 or another device.')
        self.config, self.checkpoint = Path(config).absolute(), Path(checkpoint).absolute()
        self.device, self.precision, self.lease = device, precision, lease
        self.predictor: Any = None
        self._torch: Any = None
        self._shape: tuple[int, int] | None = None
        self._provenance = ''
        lease.check(device)

    def _precision_policy(self, torch) -> None:
        if self.device.startswith('cuda') and self.precision == 'float32':
            try:
                strict = (torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
                          and torch.get_float32_matmul_precision() == 'highest')
            except (AttributeError, RuntimeError) as exc:
                raise GuidanceUnavailable('Requested CUDA float32 policy cannot be verified; use an isolated strict provider owner.') from exc
            if not strict:
                raise GuidanceUnavailable('Requested CUDA float32 refuses inherited TF32 or relaxed matmul policy. Configure an isolated strict provider owner; no global flags are changed and no precision fallback is performed.')

    @staticmethod
    def _asset_identity(path: Path, limit: int):
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise GuidanceUnavailable(f'Select an existing bounded local SAM2.1 asset: {path}. No download or installation is performed.') from exc
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or getattr(info, 'st_file_attributes', 0) & 0x400 or not 0 < info.st_size <= limit):
            raise GuidanceUnavailable(f'Select a bounded regular single-link local SAM2.1 asset without aliases: {path}. No download or installation is performed.')
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _contexts(self):
        self._precision_policy(self._torch)
        return (self._torch.inference_mode(), self._torch.autocast(self.device.split(':')[0], dtype=self._torch.bfloat16)
                if self.precision == 'bfloat16' else self._torch.autocast(self.device.split(':')[0], enabled=False))

    def _load(self, control: JobControl) -> None:
        self.lease.check(self.device)
        control.check()
        if self.predictor is not None:
            return
        assets = ((self.config, 4 * 1024**2), (self.checkpoint, 2 * 1024**3))
        identities = [self._asset_identity(path, limit) for path, limit in assets]
        def recheck_assets():
            if [self._asset_identity(path, limit) for path, limit in assets] != identities:
                raise ValueError('Local SAM2 asset changed from its initially admitted identity; no result published.')
        try:
            torch = importlib.import_module('torch')
            hydra = importlib.import_module('hydra')
            global_hydra = importlib.import_module('hydra.core.global_hydra').GlobalHydra
        except ImportError as exc:
            raise GuidanceUnavailable('Optional SAM2.1 dependencies are unavailable; install a compatible pinned provider in an isolated environment deliberately. Existing workflows need no provider.') from exc
        recheck_assets()  # Imports cannot replace/grow a previously admitted file.
        if global_hydra.instance().is_initialized():
            raise GuidanceUnavailable('SAM2 requires its isolated provider owner; an existing Hydra configuration context must remain untouched.')
        if self.device.startswith('cuda') and (not torch.cuda.is_available() or int(self.device.split(':')[1] if ':' in self.device else 0) >= torch.cuda.device_count()):
            raise GuidanceUnavailable('Requested CUDA segmentation device is unavailable; no CPU/backend fallback is performed.')
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise GuidanceUnavailable('Requested MPS segmentation device is unavailable; no backend fallback is performed.')
        self._precision_policy(torch)
        def identity(info):
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
        hashes = []
        for path, expected_identity in zip((self.config, self.checkpoint), identities, strict=True):
            path_before = path.lstat()
            if identity(path_before) != expected_identity:
                raise ValueError('Local SAM2 asset changed before identity hashing.')
            digest = hashlib.sha256()
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0)
            with os.fdopen(os.open(path, flags), 'rb') as stream:
                opened = os.fstat(stream.fileno())
                verify_file_observation(path, path_before, opened, opened)
                remaining = expected_identity[2]
                while remaining:
                    control.check()
                    self.lease.check(self.device)
                    chunk = stream.read(min(1024**2, remaining))
                    if not chunk:
                        raise ValueError('Local SAM2 asset truncated during identity hashing.')
                    remaining -= len(chunk)
                    digest.update(chunk)
                if stream.read(1) or identity(os.fstat(stream.fileno())) != identity(opened):
                    raise ValueError('Local SAM2 asset changed during identity hashing.')
                verify_file_observation(path, path_before, opened, os.fstat(stream.fileno()))
            hashes.append(digest.hexdigest())
        try:
            with hydra.initialize_config_dir(config_dir=str(self.config.parent), version_base='1.2'):
                builder = importlib.import_module('sam2.build_sam').build_sam2
                predictor_type = importlib.import_module('sam2.sam2_image_predictor').SAM2ImagePredictor
                self.lease.check(self.device)
                recheck_assets()  # Provider module imports are another mutation boundary.
                self._precision_policy(torch)
                model = builder(self.config.name, str(self.checkpoint), device=self.device,
                                apply_postprocessing=False, hydra_overrides_extra=['++model.compile_image_encoder=false'])
                self.lease.check(self.device)
                model = model.to(dtype=torch.float32)
                self.predictor = predictor_type(model, max_hole_area=0, max_sprinkle_area=0)
        except ImportError as exc:
            raise GuidanceUnavailable('Optional SAM2.1 package/configuration is unavailable; provide a deliberate local installation and matching checkpoint.') from exc
        try:
            recheck_assets()
        except (OSError, ValueError, GuidanceUnavailable):
            self.predictor = None
            raise
        self._torch = torch
        try:
            version = importlib.metadata.version('SAM-2')
        except importlib.metadata.PackageNotFoundError:
            version = 'unavailable; record local provider source revision externally'
        self._provenance = _canonical({'provider': 'local-sam2.1-image', 'package_version': version,
            'config_sha256': hashes[0], 'checkpoint_sha256': hashes[1], 'device': self.device,
            'precision': self.precision, 'postprocessing': False, 'compile_image_encoder': False,
            'torch_version': str(torch.__version__), 'matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'provider_source_revision': 'NOT VERIFIED; supply pinned local installation provenance externally',
            'input': 'native HWC uint8 assumed-sRGB proxy; explicit conversion identity in frame',
            'multimask_output': False, 'mask_threshold': 0.0})
        control.check()

    def set_frame(self, frame: GuidanceFrame, control: JobControl) -> int:
        frame.validate(4 * 1024**2)
        self._load(control)
        np = importlib.import_module('numpy')
        self.lease.check(self.device)
        image = np.frombuffer(frame.rgb_bytes, dtype=np.uint8).reshape(frame.height, frame.width, 3).copy()
        inference, autocast = self._contexts()
        with inference, autocast:
            self.predictor.set_image(image)
        self._shape = (frame.height, frame.width)
        features = self.predictor._features
        tensors = [features['image_embed'], *features['high_res_feats']]
        retained = sum(int(t.numel()) * int(t.element_size()) for t in tensors)
        control.check()
        return retained

    def predict(self, clicks: tuple[Click, ...], previous_logits: bytes | None,
                control: JobControl) -> ProviderPrediction:
        self.lease.check(self.device)
        control.check()
        if self.predictor is None or self._shape is None:
            raise RuntimeError('Set a source frame before SAM2 prediction.')
        np = importlib.import_module('numpy')
        mask_input = None if previous_logits is None else np.frombuffer(previous_logits, dtype='<f4').reshape(1, 256, 256).copy()
        inference, autocast = self._contexts()
        with inference, autocast:
            masks, _, logits = self.predictor.predict(
                point_coords=np.array([(c.x, c.y) for c in clicks], dtype=np.float32),
                point_labels=np.array([int(c.positive) for c in clicks], dtype=np.int32),
                mask_input=mask_input, multimask_output=False, return_logits=False, normalize_coords=True)
        control.check()
        if masks.shape != (1, *self._shape) or logits.shape != (1, 256, 256):
            raise ValueError('SAM2 returned unsupported candidate geometry; no implicit resizing or candidate substitution.')
        return ProviderPrediction((masks[0].astype(np.uint8) * 255).tobytes(), 'binary_u8',
                                  logits.astype('<f4').tobytes(), self._provenance)

    def reset(self) -> None:
        if self.predictor is not None:
            self.predictor.reset_predictor()
        self._shape = None

    def close(self) -> None:
        self.reset()
        self.predictor = None
        self._torch = None
