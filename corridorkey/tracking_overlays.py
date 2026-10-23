"""Optional spatial tracking diagnostics; never correction/render pixels.

Maps are quantized 8-bit area-average display proxies (maximum side 256), not
native-pixel masks. Reliability is an uncalibrated flow/photometric heuristic;
uncertainty can suggest correspondence failures/occlusion, never prove them.
An authored anchor's initialized reliability=1 is a convention, not a measured
success. No disagreement map is invented from scalar diagnostics. No flow,
neural operation, default IO or source reading is performed here.

Receipt IO is confined to an explicit existing private preview directory. The
caller owns its lifetime/aggregate budget. Failed writes clean only leaves this
operation exclusively created, preserving primary failures and cleanup notes.
Ordinary mutation is fenced; hostile concurrent filesystem owners are excluded.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .correction_tracking import TrackingCandidate
from .project_collection import (
    FilePin,
    admit_file,
    assert_pin,
    check_now,
    fingerprint,
    read_file,
    real_path,
    safe_name,
)

MAX_SIDE = 256
MAX_PIXELS = 4 * 1024 * 1024
MAX_MAP_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_PROVENANCE_BYTES = 32 * 1024
KINDS = ('reliability', 'uncertainty')


def _primitive(value, depth=0):
    if depth > 16:
        raise ValueError('Overlay metadata nesting exceeds 16')
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, str) and len(value.encode()) <= MAX_PROVENANCE_BYTES:
        return
    if isinstance(value, list) and len(value) <= 1024:
        for item in value:
            _primitive(item, depth + 1)
        return
    if isinstance(value, dict) and len(value) <= 256 and all(isinstance(key, str) for key in value):
        for item in value.values():
            _primitive(item, depth + 1)
        return
    raise TypeError('Overlay metadata requires finite JSON primitives, not dataclass classes/instances or sets')


def _canonical(value: Any, limit=MAX_RECEIPT_BYTES) -> bytes:
    _primitive(value)
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > limit:
        raise ValueError('Overlay metadata byte bound exceeded')
    return raw


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate overlay JSON field')
        result[key] = value
    return result


def _parse(raw: bytes, limit=MAX_RECEIPT_BYTES) -> dict:
    if not isinstance(raw, bytes):
        raise TypeError('Overlay metadata requires immutable bytes')
    if len(raw) > limit:
        raise ValueError('Overlay metadata byte bound exceeded')
    try:
        data = json.loads(raw, object_pairs_hook=_pairs)
    except RecursionError as exc:
        raise ValueError('Overlay metadata nesting exceeds decoder limits') from exc
    if not isinstance(data, dict):
        raise TypeError('Overlay metadata must be a JSON object')
    _canonical(data, limit)
    return data


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{label} must be a lowercase SHA256')


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} requires an exact integer in [{low},{high}]')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class OverlayBinding:
    source_identity: str
    document_identity: str
    candidate_identity: str
    decode_identity: str
    source_frame: int
    local_frame: int
    native_width: int
    native_height: int
    source_proxy_sha256: str

    def __post_init__(self):
        for name in ('source_identity', 'document_identity', 'candidate_identity', 'decode_identity', 'source_proxy_sha256'):
            _hash(getattr(self, name), name)
        _integer(self.local_frame, 0, 2**63 - 1, 'local_frame')
        _integer(self.source_frame, self.local_frame, 2**63 - 1, 'source_frame')
        for name in ('native_width', 'native_height'):
            _integer(getattr(self, name), 2, MAX_PIXELS, name)
        if self.native_width * self.native_height > MAX_PIXELS:
            raise ValueError('Overlay native pixel bound exceeded')


@dataclass(frozen=True)
class OverlayMap:
    kind: str
    width: int
    height: int
    payload: bytes

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError('Only reliability and uncertainty tracking layers are available')
        _integer(self.width, 1, MAX_SIDE, 'proxy width')
        _integer(self.height, 1, MAX_SIDE, 'proxy height')
        if not isinstance(self.payload, bytes):
            raise TypeError('Overlay payload requires immutable bytes')
        if len(self.payload) != self.width * self.height:
            raise ValueError('Overlay u8 payload size differs from its proxy geometry')


def _provenance(data, binding, width, height):
    fields = {'contract', 'candidate_identity', 'status', 'reliability_calibrated', 'sample_contract',
              'max_side', 'estimator', 'motion', 'diagnostics', 'anchor_frames', 'cuts', 'resets', 'limitation'}
    if (set(data) != fields or data['contract'] != 'tracking-overlays-v1' or data['reliability_calibrated'] is not False
            or data['candidate_identity'] != binding.candidate_identity
            or data['status'] not in ('authored_anchor', 'anchor_required', 'stopped', 'approximation', 'preview')
            or data['sample_contract'] != 'u8_0_255_area_average_display_proxy'):
        raise ValueError('Overlay provenance contract/binding differs')
    _integer(data['max_side'], 1, MAX_SIDE, 'provenance max_side')
    scale = min(1., data['max_side'] / max(binding.native_width, binding.native_height))
    if (width, height) != (max(1, round(binding.native_width * scale)), max(1, round(binding.native_height * scale))):
        raise ValueError('Overlay display proxy aspect/geometry differs from the bound native canvas')


@dataclass(frozen=True)
class OverlayBundle:
    binding: OverlayBinding
    maps: tuple[OverlayMap, ...]
    provenance_json: bytes

    def __post_init__(self):
        if not isinstance(self.binding, OverlayBinding):
            raise TypeError('Overlay binding requires a dataclass instance')
        if not isinstance(self.maps, tuple) or len(self.maps) != 2 or any(not isinstance(layer, OverlayMap) for layer in self.maps):
            raise TypeError('Overlay bundle requires an immutable pair of map instances')
        if tuple(layer.kind for layer in self.maps) != KINDS or len({(layer.width, layer.height) for layer in self.maps}) != 1:
            raise ValueError('Overlay maps require ordered matching reliability/uncertainty geometry')
        if self.maps[0].width > self.binding.native_width or self.maps[0].height > self.binding.native_height:
            raise ValueError('Diagnostic display proxies cannot enlarge the native canvas')
        data = _parse(self.provenance_json, MAX_PROVENANCE_BYTES)
        _provenance(data, self.binding, self.maps[0].width, self.maps[0].height)
        object.__setattr__(self, 'provenance_json', _canonical(data, MAX_PROVENANCE_BYTES))

    @property
    def identity(self):
        return _sha(_canonical({'binding': asdict(self.binding), 'maps': [{'kind': layer.kind, 'width': layer.width, 'height': layer.height, 'sha256': _sha(layer.payload)} for layer in self.maps],
                                'provenance': _parse(self.provenance_json)}))


@dataclass(frozen=True)
class OverlayReceipt:
    payload_json: bytes

    def __post_init__(self):
        data = _parse(self.payload_json)
        _validate_receipt(data)
        object.__setattr__(self, 'payload_json', _canonical(data))

    def to_dict(self):
        return _parse(self.payload_json)

    @property
    def binding(self):
        return OverlayBinding(**self.to_dict()['binding'])

    @property
    def identity(self):
        return self.to_dict()['bundle_identity']

    @property
    def bytes_charged(self):
        return sum(row['bytes'] for row in self.to_dict()['maps']) + len(self.payload_json)


@dataclass(frozen=True)
class OverlayRGBA:
    width: int
    height: int
    pixels: bytes
    legend_json: bytes

    def __post_init__(self):
        _integer(self.width, 1, MAX_SIDE, 'RGBA width')
        _integer(self.height, 1, MAX_SIDE, 'RGBA height')
        if not isinstance(self.pixels, bytes) or len(self.pixels) != self.width * self.height * 4:
            raise ValueError('RGBA payload differs from bounded immutable proxy geometry')
        object.__setattr__(self, 'legend_json', _canonical(_parse(self.legend_json, 4096), 4096))

    @property
    def legend(self):
        return _parse(self.legend_json)


def build_tracking_overlays(candidate: TrackingCandidate, *, document_identity: str, decode_identity: str,
                            max_side: int = MAX_SIDE, check=None) -> OverlayBundle | None:
    if not isinstance(candidate, TrackingCandidate):
        raise TypeError('Expected a TrackingCandidate instance')
    _hash(document_identity, 'document_identity')
    _hash(decode_identity, 'decode_identity')
    _integer(max_side, 1, MAX_SIDE, 'max_side')
    check_now(check)
    if candidate.status == 'authored_anchor':
        return None  # Initialized ones/zeros are declared-anchor metadata, not spatial flow measurements.
    if candidate.reliability_bytes is None and candidate.uncertain_bytes is None:
        return None
    if candidate.reliability_bytes is None or candidate.uncertain_bytes is None:
        raise ValueError('Tracking candidate has incomplete spatial diagnostics')
    _integer(candidate.width, 2, MAX_PIXELS, 'native width')
    _integer(candidate.height, 2, MAX_PIXELS, 'native height')
    pixels = candidate.width * candidate.height
    if pixels > MAX_PIXELS or not isinstance(candidate.reliability_bytes, bytes) or not isinstance(candidate.uncertain_bytes, bytes) or len(candidate.reliability_bytes) != pixels * 4 or len(candidate.uncertain_bytes) != pixels:
        raise ValueError('Tracking diagnostics exceed exact native geometry/byte bounds')
    provenance = _parse(candidate.provenance_json.encode(), MAX_PROVENANCE_BYTES)
    diagnostics = _parse(candidate.diagnostics_json.encode(), MAX_PROVENANCE_BYTES)
    track = provenance.get('track')
    if not isinstance(track, dict) or not isinstance(track.get('request'), dict) or not isinstance(provenance.get('source_proxy_sha256'), dict):
        raise TypeError('Tracking candidate provenance lacks bounded source/request bindings')
    request = track['request']
    _integer(request.get('source_start'), 0, 2**63 - 1, 'tracking source_start')
    _integer(request.get('start'), 0, 2**63 - 1, 'tracking range start')
    _integer(request.get('end'), request['start'] + 1, 2**63 - 1, 'tracking range end')
    if (not request['start'] <= candidate.local_frame < request['end'] or diagnostics.get('reliability_calibrated') is not False
            or request.get('source_identity') != provenance.get('source_identity')):
        raise ValueError('Tracking source/range/uncalibrated diagnostics contract differs')
    if (provenance.get('local_frame') != candidate.local_frame or provenance.get('source_frame') != candidate.source_frame
            or provenance.get('status') != candidate.status or request.get('source_decode_identity') != decode_identity
            or request.get('source_start', -1) + candidate.local_frame != candidate.source_frame):
        raise ValueError('Tracking candidate coordinates/decode provenance differ')
    source_hash = provenance.get('source_proxy_sha256', {}).get(str(candidate.local_frame))
    source_identity = provenance.get('source_identity')
    if not isinstance(source_identity, str):
        raise TypeError('Candidate source identity must be a SHA256 string')
    binding = OverlayBinding(source_identity, document_identity, candidate.identity, decode_identity,
                             candidate.source_frame, candidate.local_frame, candidate.width, candidate.height, source_hash)
    import cv2
    import numpy as np
    reliability = np.frombuffer(candidate.reliability_bytes, dtype='<f4').reshape(candidate.height, candidate.width)
    uncertain = np.frombuffer(candidate.uncertain_bytes, dtype=np.uint8).reshape(candidate.height, candidate.width)
    if not np.isfinite(reliability).all() or np.any((reliability < 0) | (reliability > 1)) or np.any(uncertain > 1):
        raise ValueError('Tracking diagnostic values must be finite unit reliability and binary uncertainty')
    scale = min(1., max_side / max(candidate.width, candidate.height))
    width, height = max(1, round(candidate.width * scale)), max(1, round(candidate.height * scale))
    layers = []
    for kind, values in zip(KINDS, (reliability, uncertain), strict=True):
        check_now(check)
        proxy = cv2.resize(values.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        payload = np.rint(np.clip(proxy, 0, 1) * 255).astype(np.uint8).tobytes()
        layers.append(OverlayMap(kind, width, height, payload))
    metadata = {'contract': 'tracking-overlays-v1', 'candidate_identity': candidate.identity,
                'status': candidate.status, 'reliability_calibrated': False,
                'sample_contract': 'u8_0_255_area_average_display_proxy', 'max_side': max_side,
                'estimator': provenance['track'].get('estimator'), 'motion': request.get('motion'),
                'diagnostics': diagnostics, 'anchor_frames': provenance.get('anchor_frames', []),
                'cuts': request.get('cuts', []), 'resets': request.get('resets', []),
                'limitation': 'Flow/photometric heuristic, not confidence or proof of occlusion; no spatial disagreement measurement available.'}
    check_now(check)
    return OverlayBundle(binding, tuple(layers), _canonical(metadata, MAX_PROVENANCE_BYTES))


def verify_binding(bundle: OverlayBundle, expected_binding: OverlayBinding) -> None:
    if not isinstance(bundle, OverlayBundle) or not isinstance(expected_binding, OverlayBinding):
        raise TypeError('Overlay binding checks require dataclass instances')
    if bundle.binding != expected_binding:
        raise ValueError('Stale overlay: source/frame/correction/decode/candidate binding differs')


def _validate_receipt(data):
    if set(data) != {'schema_version', 'type', 'binding', 'bundle_identity', 'maps', 'provenance'} or type(data['schema_version']) is not int or data['schema_version'] != 1 or data['type'] != 'tracking_overlay_receipt':
        raise ValueError('Unsupported overlay receipt schema/type/fields')
    if not isinstance(data['binding'], dict) or set(data['binding']) != set(OverlayBinding.__dataclass_fields__):
        raise ValueError('Overlay receipt binding fields differ')
    binding = OverlayBinding(**data['binding'])
    _hash(data['bundle_identity'], 'bundle_identity')
    rows = data['maps']
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError('Overlay receipt requires exactly two map leaves')
    seen = set()
    for kind, row in zip(KINDS, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {'kind', 'width', 'height', 'name', 'bytes', 'sha256'} or row['kind'] != kind:
            raise ValueError('Overlay receipt map fields/order differ')
        name = safe_name(row['name'])
        if name in seen or re.fullmatch(r'overlay-[0-9a-f]{32}-(reliability|uncertainty)\.u8', name) is None or not name.endswith('-' + kind + '.u8'):
            raise ValueError('Overlay receipt leaf name is unsafe or duplicated')
        seen.add(name)
        _integer(row['width'], 1, MAX_SIDE, 'map width')
        _integer(row['height'], 1, MAX_SIDE, 'map height')
        if type(row['bytes']) is not int or row['bytes'] != row['width'] * row['height']:
            raise ValueError('Overlay receipt map extent differs')
        _hash(row['sha256'], 'map sha256')
    if len({(row['width'], row['height']) for row in rows}) != 1 or rows[0]['width'] > binding.native_width or rows[0]['height'] > binding.native_height:
        raise ValueError('Overlay receipt proxy/native geometry differs')
    provenance = data['provenance']
    if not isinstance(provenance, dict):
        raise TypeError('Overlay receipt provenance must be an object')
    _canonical(provenance, MAX_PROVENANCE_BYTES)
    _provenance(provenance, binding, rows[0]['width'], rows[0]['height'])


def write_bundle(bundle: OverlayBundle, owned_preview_dir: Path, *, remaining_budget: int, check=None) -> OverlayReceipt:
    if not isinstance(bundle, OverlayBundle):
        raise TypeError('Expected OverlayBundle instance')
    _integer(remaining_budget, 0, MAX_MAP_BYTES, 'remaining map budget')
    root = real_path(owned_preview_dir, directory=True)
    token = uuid.uuid4().hex
    rows = [{'kind': layer.kind, 'width': layer.width, 'height': layer.height,
             'name': f'overlay-{token}-{layer.kind}.u8', 'bytes': len(layer.payload), 'sha256': _sha(layer.payload)} for layer in bundle.maps]
    receipt = OverlayReceipt(_canonical({'schema_version': 1, 'type': 'tracking_overlay_receipt', 'binding': asdict(bundle.binding),
                                         'bundle_identity': bundle.identity, 'maps': rows, 'provenance': _parse(bundle.provenance_json)}))
    if receipt.bytes_charged > remaining_budget:
        raise ValueError('Tracking overlay map/receipt byte budget exceeded')
    created = []
    verified_pins = []
    root_id = (root.stat().st_dev, root.stat().st_ino)
    def check_root():
        current = real_path(root, directory=True).stat()
        if (current.st_dev, current.st_ino) != root_id:
            raise ValueError('Owned preview directory changed during overlay IO')
    try:
        for row, layer in zip(rows, bundle.maps, strict=True):
            check_now(check)
            check_root()
            path = root / row['name']
            with path.open('xb') as stream:
                opened = os.fstat(stream.fileno())
                created.append((path, (opened.st_dev, opened.st_ino)))
                stream.write(layer.payload)
                stream.flush()
                pin = FilePin(str(path), fingerprint(os.fstat(stream.fileno())), row['sha256'])
            read_file(pin, check=check)
            verified_pins.append(pin)
        for pin in verified_pins:
            assert_pin(pin)
        check_root()
        check_now(check)
        return receipt
    except BaseException as primary:
        for path, owned_id in created:
            try:
                check_root()
                info = path.lstat()
                if (info.st_dev, info.st_ino) != owned_id:
                    raise ValueError('Created overlay leaf was replaced; cleanup refused')
                path.unlink()
            except (OSError, ValueError) as cleanup:
                primary.add_note(f'Private overlay cleanup failed: {path}: {cleanup}')
        raise


def read_bundle(receipt: OverlayReceipt | dict, owned_preview_dir: Path, *, expected_binding: OverlayBinding | None = None, check=None) -> OverlayBundle:
    if isinstance(receipt, dict):
        receipt = OverlayReceipt(_canonical(receipt))
    if not isinstance(receipt, OverlayReceipt):
        raise TypeError('Expected OverlayReceipt instance or primitive receipt object')
    data = receipt.to_dict()
    binding = receipt.binding
    if expected_binding is not None:
        if not isinstance(expected_binding, OverlayBinding):
            raise TypeError('Expected OverlayBinding instance')
        if binding != expected_binding:
            raise ValueError('Stale overlay receipt binding differs')
    root = real_path(owned_preview_dir, directory=True)
    root_id = (root.stat().st_dev, root.stat().st_ino)
    pins = [admit_file(root / row['name']) for row in data['maps']]
    if any(pin.size != row['bytes'] for pin, row in zip(pins, data['maps'], strict=True)):
        raise ValueError('Overlay file size differs from its bounded receipt')
    layers = []
    for pin, row in zip(pins, data['maps'], strict=True):
        digest, raw = read_file(pin, check=check, capture=True)
        if digest != row['sha256']:
            raise ValueError('Overlay file content identity differs')
        layers.append(OverlayMap(row['kind'], row['width'], row['height'], raw))
    for pin in pins:
        assert_pin(pin)
    current_root = real_path(root, directory=True).stat()
    if (current_root.st_dev, current_root.st_ino) != root_id:
        raise ValueError('Overlay preview directory changed during read')
    bundle = OverlayBundle(binding, tuple(layers), _canonical(data['provenance'], MAX_PROVENANCE_BYTES))
    if bundle.identity != receipt.identity:
        raise ValueError('Overlay bundle/receipt identity differs')
    check_now(check)
    return bundle


def render_layer(bundle: OverlayBundle, kind: str, *, opacity: float = .55) -> OverlayRGBA:
    if not isinstance(bundle, OverlayBundle):
        raise TypeError('Expected OverlayBundle instance')
    if kind not in KINDS:
        raise ValueError('Spatial disagreement/occlusion ground truth is unavailable; choose reliability or uncertainty')
    if type(opacity) not in (int, float) or not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('Overlay opacity must be finite in [0,1]')
    import numpy as np
    layer = next(layer for layer in bundle.maps if layer.kind == kind)
    values = np.frombuffer(layer.payload, np.uint8).reshape(layer.height, layer.width)
    rgba = np.zeros((layer.height, layer.width, 4), np.uint8)
    rgba[:, :, :3] = (255, 70, 30)
    intensity = 255 - values if kind == 'reliability' else values
    rgba[:, :, 3] = np.rint(intensity.astype(np.float32) * opacity).astype(np.uint8)
    status = _parse(bundle.provenance_json)['status']
    legend = {'kind': kind, 'display': 'Red opacity: low heuristic reliability' if kind == 'reliability' else 'Red opacity: fraction flagged uncertain within each proxy cell',
              'range': [0, 1], 'encoding': 'u8 quantized area-average display proxy; not native pixel coverage',
              'calibrated': False, 'status': status,
              'limitation': 'Not quality confidence, defect proof or verified occlusion. No disagreement map available.',
              'anchor_note': 'Authored-anchor reliability=1 and uncertainty=0 are initialized conventions, not measured flow success.'}
    return OverlayRGBA(layer.width, layer.height, rgba.tobytes(), _canonical(legend))
