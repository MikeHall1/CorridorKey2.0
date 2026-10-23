"""Actual authored alpha operations; no fabricated trimap/model channels."""
from collections import OrderedDict
import hashlib
import math
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np


class CorrectionRasters:
    def __init__(self, document, meta, *, check=None):
        self.document = document
        data = document.to_dict()
        self._entries = data["entries"]
        self._cache = OrderedDict()
        self.check = check or (lambda: None)
        source = data["source"]
        self.shape = (source["height"], source["width"])
        if (self.shape != tuple(meta["native_hw"])
                or source["start_frame"] != meta.get("start_frame", 0)
                or source["frame_count"] != meta["num_frames"]
                or not math.isclose(source["fps"], meta["native_fps"], rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("correction source geometry/selected range/FPS does not match the current job")
        # Decode/validate early; retain no more than two float32 native masks.
        for entry in self._entries:
            if entry["kind"] != "reset":
                self._mask(entry)

    def _mask(self, entry):
        self.check()
        path = ((self.document.base_dir / entry["asset"]["path"])
                if entry["kind"] == "trimap" else self.document.asset_path(entry))
        key = (str(path), entry["asset"]["sha256"], entry["kind"],
               "strict-trimap-v1" if entry["kind"] == "trimap" else "coverage-v1")
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        # Evict before decoding/admitting a new native-size mask.
        if len(self._cache) >= 2:
            self._cache.popitem(last=False)
        if entry["kind"] == "trimap":
            from .trimap_refinement import load_trimap_asset
            mask = load_trimap_asset(path, expected_hw=self.shape,
                                     expected_sha256=entry["asset"]["sha256"], check=self.check).labels
            self._cache[key] = mask
            return mask
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["asset"]["sha256"]:
            raise ValueError(f"correction asset changed before decode: {entry['id']}")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"could not decode correction asset: {path}")
        if image.ndim == 3:
            if image.shape[2] != 4:
                raise ValueError("correction raster must be grayscale coverage or RGBA alpha, not RGB luminance")
            image = image[..., 3]
        if image.shape != self.shape:
            raise ValueError(f"correction raster shape {image.shape} must match native {self.shape}; implicit resizing is disabled")
        if image.dtype == np.uint8:
            mask = image.astype(np.float32) / 255
        elif image.dtype == np.uint16:
            mask = image.astype(np.float32) / 65535
        elif np.issubdtype(image.dtype, np.floating):
            mask = image.astype(np.float32)
        else:
            raise ValueError("unsupported correction raster precision")
        if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError("correction raster must contain finite coverage in [0,1]")
        mask.setflags(write=False)
        self._cache[key] = mask
        self.check()
        return mask

    def _active(self, frame, *, temporal=False):
        return [entry for entry in self._entries if entry["kind"] != "reset" and entry["strength"] > 0
                and entry["range"][0] <= frame < entry["range"][1]
                and (not temporal or (entry["direction"] == "both" and entry["kind"] != "trimap"))]

    def has_effect(self, frame, *, temporal=False):
        return bool(self._active(frame, temporal=temporal))

    def has_condition(self, frame):
        return any(e["kind"] not in ("reset", "trimap") and e["strength"] > 0 and e["direction"] == "both"
                   and e["frame"] == frame for e in self._entries)

    @property
    def has_trimaps(self):
        return any(e["kind"] == "trimap" and e["strength"] > 0 for e in self._entries)

    def requires_guidance(self, frame):
        return any(e["kind"] == "trimap" for e in self._active(frame))

    def _operation(self, alpha, entry, *, guidance=None):
        mask, strength = self._mask(entry), entry["strength"]
        if entry["kind"] == "trimap":
            if guidance is None:
                raise ValueError("final trimap refinement requires native RGB guidance")
            from .trimap_refinement import refine_trimap
            return refine_trimap(alpha, mask, guidance, strength=strength, check=self.check)
        if entry["kind"] == "matte":
            return alpha * (1 - strength) + mask * strength
        if entry["kind"] == "keep":
            return alpha + (1 - alpha) * mask * strength
        return alpha * (1 - mask * strength)

    def apply(self, alpha, frame, *, temporal=False, guidance=None):
        alpha = np.asarray(alpha, dtype=np.float32)
        if alpha.shape != self.shape or not np.isfinite(alpha).all():
            raise ValueError("correction input alpha must be finite and match native dimensions")
        result = np.clip(alpha, 0, 1).copy()
        for entry in self._active(frame, temporal=temporal):
            result = self._operation(result, entry, guidance=guidance)
        return np.clip(result, 0, 1).astype(np.float32, copy=False)

    def condition(self, frame, base_hint=None):
        """Both-direction authored hints only; partial regions need a known base.

        Forward/backward entries apply as direct static raster constraints only.
        No reverse neural pass or per-pixel hint-presence channel is invented.
        """
        hint = None if base_hint is None else np.asarray(base_hint, dtype=np.float32).copy()
        if hint is not None and (hint.shape != self.shape or not np.isfinite(hint).all()):
            raise ValueError("base hint must be finite native coverage")
        quality = None
        for entry in self._entries:
            if (entry["kind"] in ("reset", "trimap") or entry["direction"] != "both"
                    or entry["frame"] != frame or entry["strength"] == 0):
                continue
            if entry["kind"] == "matte":
                hint = self._mask(entry).copy() if hint is None else self._operation(hint, entry)
                quality = entry["strength"]
            elif hint is not None:
                hint = self._operation(hint, entry)
        return hint, quality

    def close(self):
        self._cache.clear()
