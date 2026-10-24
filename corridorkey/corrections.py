"""Versioned local authored raster corrections and conservative dependencies."""
import hashlib
import json
import math
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path

from . import guidance_provenance
from .diagnostics import finite_json_float, write_json_atomic
from .identities import file_sha256, source_identity, verify_file_observation

REGION_TARGETS = {'refinement': 'edge_finishing', 'composite': 'despill', 'diagnostic': 'tracking_overlay'}

MAX_CORRECTION_BYTES = 4 * 1024 * 1024
MAX_TRACK_FRAMES = 300
MAX_TRACK_ANCHORS = 32
MAX_TRACK_PIXELS = 4 * 1024 * 1024
MAX_TRACK_BYTES = 512 * 1024 * 1024


def guidance_binding(document_data, entry_id, *, kind, resources, parameters, decode, observation):
    """Path-free generation binding; see guidance_provenance for its contract."""
    return guidance_provenance.guidance_binding(document_data, entry_id, kind=kind, resources=resources,
                                               parameters=parameters, decode=decode, observation=observation)


def attach_guidance(document, entry_id, *, guidance_id, kind, resources, parameters, decode, observation):
    """Attach provenance in one detached schema4 document revision."""
    return guidance_provenance.attach_guidance(document, entry_id, guidance_id=guidance_id, kind=kind,
                                              resources=resources, parameters=parameters, decode=decode, observation=observation)


def _assets(data):
    for entry in data['entries']:
        if 'asset' in entry:
            yield entry['asset']
    for track in data.get('tracks', []):
        for row in track['frames']:
            yield row['asset']
    for origin in data.get('guidance', []):
        for resource in origin['resources']:
            yield resource['asset']


def _decode_contract(decode):
    keys = {'contract', 'source_kind', 'frame_dir_linear', 'frame_dir_alpha', 'source_colour_profile',
            'linear_brightness', 'linear_contrast', 'linear_contrast_pivot'}
    if not isinstance(decode, dict) or set(decode) != keys:
        raise ValueError('tracking decode contract has unknown/missing fields')
    if (decode['contract'] != 'corridorkey-native-rgb-v1' or decode['source_kind'] not in ('frames', 'video')
            or type(decode['frame_dir_linear']) is not bool or decode['frame_dir_alpha'] not in ('ignore', 'premultiplied')
            or decode['source_colour_profile'] not in ('legacy', 'linear_bt709_d65', 'linear_bt2020_d65')):
        raise ValueError('unsupported tracking decode contract')
    if (decode['frame_dir_alpha'] == 'premultiplied' or decode['source_colour_profile'] != 'legacy') and not decode['frame_dir_linear']:
        raise ValueError('tracking alpha/colour declaration requires linear frames')
    if decode['source_kind'] == 'video' and (decode['frame_dir_alpha'] != 'ignore' or decode['source_colour_profile'] != 'legacy'):
        raise ValueError('tracking video colour/alpha conversion is unsupported')
    for name in ('linear_brightness', 'linear_contrast', 'linear_contrast_pivot'):
        value = decode[name]
        if type(value) not in (int, float) or not math.isfinite(value) or (name != 'linear_contrast_pivot' and value == 0):
            raise ValueError('tracking linear adjustments must be finite and brightness/contrast nonzero')


def _tracking_contract(tracking):
    keys = {'kernel_version', 'direction', 'unreliable_policy', 'max_uncertain_fraction', 'motion',
            'max_frames', 'max_pixels', 'max_anchor_bytes', 'max_anchors', 'estimator_identity'}
    if not isinstance(tracking, dict) or set(tracking) != keys:
        raise ValueError('tracking settings have unknown/missing fields')
    if (tracking['kernel_version'] != 'correction-tracking-v1' or tracking['direction'] not in ('forward', 'backward', 'both')
            or tracking['unreliable_policy'] not in ('stop', 'request_anchor', 'retain_approximation')
            or not isinstance(tracking['estimator_identity'], str) or not 1 <= len(tracking['estimator_identity']) <= 256):
        raise ValueError('unsupported tracking settings')
    for name, limit in (('max_frames', MAX_TRACK_FRAMES), ('max_pixels', MAX_TRACK_PIXELS),
                        ('max_anchor_bytes', 64 * 1024 * 1024), ('max_anchors', MAX_TRACK_ANCHORS)):
        _integer(tracking[name], name, 1)
        if tracking[name] > limit:
            raise ValueError('tracking budget exceeds supported persistence limit')
    fraction = tracking['max_uncertain_fraction']
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('tracking uncertain fraction must be finite in [0,1]')
    motion = tracking['motion']
    if not isinstance(motion, dict) or set(motion) != {'max_side', 'consistency_pixels', 'photometric_scale', 'min_reliability', 'max_disagreement'}:
        raise ValueError('tracking motion settings have unknown/missing fields')
    _integer(motion['max_side'], 'motion max_side', 16)
    if motion['max_side'] > 4096:
        raise ValueError('tracking motion proxy exceeds supported limit')
    for name in ('consistency_pixels', 'photometric_scale', 'min_reliability', 'max_disagreement'):
        value = motion[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('tracking motion settings must be finite positive numbers')
    if motion['min_reliability'] > 1 or motion['max_disagreement'] > 1:
        raise ValueError('tracking reliability/disagreement thresholds must be <=1')


def track_binding(document_data, anchor_ids, *, cuts, decode, tracking):
    """Content binding without paths, tracks, schema migration or accepting revision."""
    data = document_data
    if (not isinstance(anchor_ids, list) or not 1 <= len(anchor_ids) <= MAX_TRACK_ANCHORS
            or any(not isinstance(value, str) for value in anchor_ids) or len(set(anchor_ids)) != len(anchor_ids)):
        raise ValueError('tracking anchor IDs must be a bounded unique ordered list')
    entries = data['entries']
    selected = [entry for entry in entries if entry['id'] in anchor_ids]
    if [entry['id'] for entry in selected] != anchor_ids:
        raise ValueError('tracking anchors must exist in document composition order')
    positions = [index for index, entry in enumerate(entries) if entry['id'] in anchor_ids]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError('tracking anchors must occupy contiguous composition positions')
    first = selected[0]
    if first['kind'] not in ('matte', 'keep', 'remove', 'region') or any(
            (entry['kind'], entry['strength'], entry.get('purpose'), entry.get('target')) !=
            (first['kind'], first['strength'], first.get('purpose'), first.get('target')) for entry in selected):
        raise ValueError('tracking anchors require the same coverage operation, strength, purpose and target')
    if first['kind'] == 'region' and (data.get('schema_version') != 5 or not isinstance(first.get('purpose'), str)
                                    or first['purpose'] not in REGION_TARGETS
                                    or REGION_TARGETS[first['purpose']] != first.get('target')):
        raise ValueError('tracked regions require version5 supported purpose/target anchors')
    if len({entry['frame'] for entry in selected}) != len(selected):
        raise ValueError('tracking anchors must have distinct frames')
    count = data['source']['frame_count']
    if (not isinstance(cuts, list) or any(type(value) is not int or not 0 <= value <= count for value in cuts)
            or sorted(set(cuts)) != cuts):
        raise ValueError('tracking cuts must be sorted unique local integer boundaries')
    _decode_contract(decode)
    _tracking_contract(tracking)
    anchors = []
    for entry in selected:
        item = json.loads(canonical(entry))
        item['asset'].pop('path')
        anchors.append(item)
    return json.loads(canonical({'source': data['source'], 'shots': data['shots'],
                                'resets': sorted(entry['frame'] for entry in entries if entry['kind'] == 'reset'),
                                'cuts': cuts, 'decode': decode, 'tracking': tracking, 'anchors': anchors,
                                'entry_order': [entry['id'] for entry in entries]}))


def _parse_tracks(data):
    tracks = data['tracks']
    if not isinstance(tracks, list) or len(tracks) > MAX_TRACK_ANCHORS:
        raise ValueError('tracked corrections must be a bounded list')
    seen, spans, total = set(), {}, 0
    for track in tracks:
        if not isinstance(track, dict) or set(track) != {'id', 'owner_id', 'anchor_ids', 'range', 'direction', 'binding', 'binding_sha256', 'frames', 'observations'}:
            raise ValueError('tracked correction has unknown/missing fields')
        if not isinstance(track['id'], str) or not 1 <= len(track['id']) <= 256 or track['id'] in seen:
            raise ValueError('track IDs must be bounded unique strings')
        seen.add(track['id'])
        binding = track['binding']
        if not isinstance(binding, dict) or set(binding) != {'source', 'shots', 'resets', 'cuts', 'decode', 'tracking', 'anchors', 'entry_order'}:
            raise ValueError('tracking binding has unknown/missing fields')
        expected = track_binding(data, track['anchor_ids'], cuts=binding['cuts'], decode=binding['decode'], tracking=binding['tracking'])
        _hash(track['binding_sha256'])
        if binding != expected or hashlib.sha256(canonical(binding).encode()).hexdigest() != track['binding_sha256']:
            raise ValueError('stale or corrupt tracking binding; re-track authored corrections')
        span = track['range']
        if (not isinstance(span, list) or len(span) != 2 or any(type(value) is not int for value in span)
                or not 0 <= span[0] < span[1] <= data['source']['frame_count']):
            raise ValueError('tracked range must be a local nonempty half-open range')
        lo, hi = span
        settings = binding['tracking']
        if (track['owner_id'] != track['anchor_ids'][0] or track['direction'] != settings['direction']
                or not all(lo <= entry['frame'] < hi for entry in binding['anchors'])
                or hi - lo > settings['max_frames'] or len(binding['anchors']) > settings['max_anchors']
                or min(data['source']['height'], data['source']['width']) < 2
                or data['source']['height'] * data['source']['width'] > settings['max_pixels']
                or len(binding['anchors']) * data['source']['height'] * data['source']['width'] * 4 > settings['max_anchor_bytes']):
            raise ValueError('track owner, direction, anchors or resource limits disagree')
        anchor_frames = [entry['frame'] for entry in binding['anchors']]
        if (track['direction'] == 'forward' and lo != min(anchor_frames)) or (track['direction'] == 'backward' and hi != max(anchor_frames) + 1):
            raise ValueError('tracked direction and anchor range disagree')
        boundaries = set(binding['cuts']) | set(binding['resets']) | {shot[0] for shot in data['shots']}
        if any(lo < point < hi for point in boundaries):
            raise ValueError('tracking cannot cross declared cuts, resets or shots')
        for identifier in track['anchor_ids']:
            if any(lo < other_hi and other_lo < hi for other_lo, other_hi in spans.get(identifier, [])):
                raise ValueError('overlapping tracks cannot replace the same authored anchor')
            spans.setdefault(identifier, []).append((lo, hi))
        frames = track['frames']
        if not isinstance(frames, list) or len(frames) != hi - lo:
            raise ValueError('tracked masks must cover every accepted frame exactly once')
        total += len(frames)
        if total > MAX_TRACK_FRAMES:
            raise ValueError('document tracked frame asset count exceeds 300')
        for index, row in enumerate(frames, lo):
            if not isinstance(row, dict) or set(row) != {'frame', 'asset'} or type(row['frame']) is not int or row['frame'] != index:
                raise ValueError('tracked mask rows must be contiguous and ordered')
            _asset(row['asset'])
        observations = track['observations']
        if not isinstance(observations, list) or len(observations) != len(frames):
            raise ValueError('tracking observations must cover every accepted frame')
        for frame, observation in zip(frames, observations, strict=True):
            _observation(observation, frame['frame'], data['source'], anchor_frames, settings)


def _primitive(value, depth=0):
    if depth > 16:
        raise ValueError('tracking diagnostic nesting exceeds 16')
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, str) and len(value) <= 4096:
        return
    if isinstance(value, list) and len(value) <= 512:
        for item in value:
            _primitive(item, depth + 1)
        return
    if isinstance(value, dict) and len(value) <= 512:
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise ValueError('tracking diagnostic keys must be bounded strings')
            _primitive(item, depth + 1)
        return
    raise ValueError('tracking diagnostics require bounded finite primitive JSON')


def _observation(observation, frame, source, anchor_frames, settings):
    keys = {'frame', 'source_frame', 'status', 'candidate_identity', 'source_proxy_sha256', 'derived_mask_sha256', 'diagnostics'}
    if (not isinstance(observation, dict) or set(observation) != keys
            or type(observation['frame']) is not int or observation['frame'] != frame
            or type(observation['source_frame']) is not int or observation['source_frame'] != source['start_frame'] + frame):
        raise ValueError('tracking observation frame binding disagrees with accepted masks')
    status = observation['status']
    if (status not in ('authored_anchor', 'preview', 'approximation')
            or (frame in anchor_frames) != (status == 'authored_anchor')
            or (status == 'approximation' and settings['unreliable_policy'] != 'retain_approximation')):
        raise ValueError('tracking observation status requires an authored anchor or explicit approximation policy')
    _hash(observation['candidate_identity'])
    _hash(observation['derived_mask_sha256'])
    hashes = observation['source_proxy_sha256']
    if not isinstance(hashes, dict) or not 1 <= len(hashes) <= 3 or str(frame) not in hashes:
        raise ValueError('tracking source proxy hashes require the target and at most two anchors')
    for key, value in hashes.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdigit() or len(key) > 20:
            raise ValueError('tracking proxy hash keys must be canonical local frame strings')
        index = int(key)
        if key != str(index) or not 0 <= index < source['frame_count'] or (index != frame and index not in anchor_frames):
            raise ValueError('tracking proxy hash references an unrelated source frame')
        _hash(value)
    if not isinstance(observation['diagnostics'], dict):
        raise TypeError('tracking diagnostics require a primitive object')
    _primitive(observation['diagnostics'])
    if len(canonical(observation).encode()) > 64 * 1024:
        raise ValueError('tracking observation exceeds 64 KiB')


def _asset(asset):
    if not isinstance(asset, dict) or set(asset) != {'path', 'sha256'}:
        raise ValueError('correction asset requires local path and sha256')
    if not isinstance(asset['path'], str) or not asset['path'].strip() or '://' in asset['path']:
        raise ValueError('correction assets must be local filesystem paths')
    _hash(asset['sha256'])


def _derived_identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _derived_size(path, shape):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or getattr(info, 'st_file_attributes', 0) & 0x400
            or not 0 < info.st_size <= shape[0] * shape[1] * 8 + 1024**2):
        raise ValueError('tracked mask must be a bounded regular single-link file')
    return info


def load_tracked_asset(path, *, expected_hw, expected_sha256, expected_mask_sha256=None, check=None):
    """Admit encoded extent and EXR native float32 geometry before decoding."""
    check = check or (lambda: None)
    shape = tuple(expected_hw)
    if len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape) or shape[0] * shape[1] > MAX_TRACK_PIXELS:
        raise ValueError('tracked mask native geometry exceeds bounds')
    path = Path(path)
    check()
    before = _derived_size(path, shape)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    try:
        opened = os.fstat(fd)
        verify_file_observation(path, before, opened, opened)
        chunks, remaining, digest = [], before.st_size, hashlib.sha256()
        while remaining:
            check()
            chunk = os.read(fd, min(1024**2, remaining))
            if not chunk:
                raise ValueError('tracked mask truncated during read')
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1) or _derived_identity(os.fstat(fd)) != _derived_identity(opened):
            raise ValueError('tracked mask changed during read')
        verify_file_observation(path, before, opened, os.fstat(fd))
        raw = b''.join(chunks)
    finally:
        os.close(fd)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('tracked mask hash mismatch')
    from .trimap_refinement import _admit_header
    if _admit_header(raw, shape) != 'exr':
        raise ValueError('tracked masks require lossless native float32 EXR')
    # The shared bounded header admission permits HALF categorical assets too;
    # tracked soft coverage specifically requires a FLOAT Y channel.
    position = 8
    while raw[position] != 0:
        end = raw.index(b'\0', position)
        name, position = raw[position:end], end + 1
        position = raw.index(b'\0', position) + 1
        size = struct.unpack('<i', raw[position:position + 4])[0]
        position += 4
        if name == b'channels' and struct.unpack('<i', raw[position + 2:position + 6])[0] != 2:
            raise ValueError('tracked masks require float32 EXR, not half precision')
        position += size
    check()
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != shape or image.dtype != np.float32:
        raise ValueError('tracked mask must decode as exactly native float32 coverage')
    if not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
        raise ValueError('tracked mask must contain finite coverage in [0,1]')
    if expected_mask_sha256 is not None and hashlib.sha256(image.astype('<f4', copy=False).tobytes()).hexdigest() != expected_mask_sha256:
        raise ValueError('tracked decoded mask differs from accepted candidate provenance')
    image.setflags(write=False)
    check()
    return image


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate correction JSON field')
        result[key] = value
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("asset/source sha256 must be a lowercase 64-digit hex string")


@dataclass(frozen=True)
class CorrectionDocument:
    payload: str
    base_dir: Path

    @classmethod
    def parse(cls, data, base_dir: str | Path = "."):
        keys = {"schema_version", "revision", "source", "shots", "entries"}
        if isinstance(data, dict) and data.get('schema_version') in (3, 4, 5):
            keys.add('tracks')
        if isinstance(data, dict) and data.get('schema_version') in (4, 5):
            keys.add('guidance')
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("corrections require schema_version, revision, source, shots and entries only")
        if type(data["schema_version"]) is not int or data["schema_version"] not in (1, 2, 3, 4, 5):
            raise ValueError("unsupported correction schema version")
        _integer(data["revision"], "revision")
        source = data["source"]
        if not isinstance(source, dict) or set(source) != {"sha256", "start_frame", "frame_count", "fps", "width", "height"}:
            raise ValueError("correction source requires hash, selected start/count, fps and native dimensions")
        _hash(source["sha256"])
        for name in ("start_frame", "frame_count", "width", "height"):
            _integer(source[name], name, 0 if name == "start_frame" else 1)
        if type(source["fps"]) not in (int, float) or not math.isfinite(source["fps"]) or source["fps"] <= 0:
            raise ValueError("correction source fps must be finite and positive")
        shots = data["shots"]
        if not isinstance(shots, list) or not shots:
            raise ValueError("correction shots must partition the selected timeline")
        end = 0
        for shot in shots:
            if (not isinstance(shot, list) or len(shot) != 2 or any(type(x) is not int for x in shot)
                    or shot[0] != end or shot[1] <= shot[0] or shot[1] > source["frame_count"]):
                raise ValueError("correction shots must be contiguous half-open [start,end) ranges")
            end = shot[1]
        if end != source["frame_count"]:
            raise ValueError("correction shots must cover all selected frames")
        if not isinstance(data["entries"], list):
            raise ValueError("correction entries must be an ordered list")
        seen = set()
        for entry in data["entries"]:
            common = {"id", "revision", "frame", "kind"}
            reset = isinstance(entry, dict) and entry.get("kind") == "reset"
            if data['schema_version'] >= 4 and isinstance(entry, dict) and 'guidance_id' in entry and not reset:
                common.add('guidance_id')
            region = isinstance(entry, dict) and entry.get('kind') == 'region'
            if region:
                common.update(('purpose', 'target'))
            if not isinstance(entry, dict) or set(entry) != (common if reset else common | {"asset", "strength", "direction", "range"}):
                raise ValueError("unknown or missing correction fields")
            if not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in seen:
                raise ValueError("correction IDs must be unique nonempty strings")
            seen.add(entry["id"])
            _integer(entry["revision"], "entry revision")
            _integer(entry["frame"], "entry frame")
            kinds = {"matte", "keep", "remove", "reset"}
            if data["schema_version"] >= 2:
                kinds.add("trimap")
            if data['schema_version'] >= 5:
                kinds.add('region')
            if entry["frame"] >= source["frame_count"] or entry["kind"] not in kinds:
                raise ValueError("unsupported correction kind or frame outside selected timeline")
            if reset:
                continue
            if region and (not isinstance(entry['purpose'], str) or entry['purpose'] not in REGION_TARGETS
                           or REGION_TARGETS[entry['purpose']] != entry['target']
                           or entry['direction'] != 'both' or 'guidance_id' in entry):
                raise ValueError('regions require authored both-direction supported purpose/target coverage')
            _asset(entry['asset'])
            strength = entry["strength"]
            if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("correction strength must be finite in [0,1]")
            span = entry["range"]
            if (not isinstance(span, list) or len(span) != 2 or any(type(t) is not int for t in span)
                    or not 0 <= span[0] <= entry["frame"] < span[1] <= source["frame_count"]):
                raise ValueError("correction range must contain its authored frame")
            shot = next(s for s in shots if s[0] <= entry["frame"] < s[1])
            if span[0] < shot[0] or span[1] > shot[1]:
                raise ValueError("correction range cannot cross a declared shot")
            if (entry["direction"] not in ("forward", "backward", "both")
                    or (entry["direction"] == "forward" and span[0] != entry["frame"])
                    or (entry["direction"] == "backward" and span[1] != entry["frame"] + 1)):
                raise ValueError("correction direction and range disagree")
        if data['schema_version'] >= 3:
            _parse_tracks(data)
        if data['schema_version'] >= 4:
            from .guidance_provenance import parse_guidance

            parse_guidance(data)
        payload = canonical(data)
        if len(payload.encode()) > MAX_CORRECTION_BYTES:
            raise ValueError('correction document exceeds the 4MiB limit')
        return cls(payload, Path(base_dir).resolve())

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def identity(self):
        data = self.to_dict()
        for asset in _assets(data):
            asset.pop('path')
        return hashlib.sha256(canonical(data).encode()).hexdigest()

    def asset_bindings(self):
        """Detached authored/derived dependencies; no mutable document aliases."""
        return tuple(_assets(self.to_dict()))

    def asset_paths(self):
        return tuple((self.base_dir / asset['path']).resolve() for asset in self.asset_bindings())

    def validate_track_context(self, *, cuts, decode):
        """Refuse reuse under different live decode or reset/cut settings."""
        for track in self.to_dict().get('tracks', []):
            if track['binding']['cuts'] != cuts or track['binding']['decode'] != decode:
                raise ValueError('tracked correction context changed; re-track or explicitly remove tracks')

    def validate_guidance_context(self, *, decode):
        """Accepted generated rasters require the original live source decoder."""
        for origin in self.to_dict().get('guidance', []):
            if origin['decode']['source'] != decode:
                raise ValueError('guidance source decode changed; regenerate or explicitly detach guidance')

    @property
    def frames(self):
        return tuple(sorted({entry["frame"] for entry in self.to_dict()["entries"]
                             if entry["kind"] == "reset" or (entry['kind'] != 'region' and entry["strength"] > 0)}))

    @property
    def resets(self):
        return tuple(entry["frame"] for entry in self.to_dict()["entries"] if entry["kind"] == "reset")

    def asset_path(self, entry):
        return (self.base_dir / entry["asset"]["path"]).resolve()

    def admit_track_assets(self, *, check=None):
        """Preflight aggregate derived extents without reading/decoding masks."""
        data = self.to_dict()
        shape = (data['source']['height'], data['source']['width'])
        # Admit all derived extents before any derived hash/decode allocation.
        total = 0
        for track in data.get('tracks', []):
            for row in track['frames']:
                if check is not None:
                    check()
                total += _derived_size(self.base_dir / row['asset']['path'], shape).st_size
                if total > MAX_TRACK_BYTES:
                    raise ValueError('document tracked encoded bytes exceed 512 MiB')

    def verify_assets(self, *, check=None):
        from .guidance_provenance import verify_guidance_resources

        data = self.to_dict()
        shape = (data['source']['height'], data['source']['width'])
        self.admit_track_assets(check=check)
        verify_guidance_resources(data, self.base_dir, check=check)
        for entry in data["entries"]:
            if entry['kind'] == 'region':
                from .trimap_refinement import load_coverage_asset

                load_coverage_asset(self.base_dir / entry['asset']['path'], expected_hw=shape,
                                    expected_sha256=entry['asset']['sha256'], check=check)
                continue
            if entry["kind"] == "trimap":
                from .trimap_refinement import load_trimap_asset

                source = data["source"]
                # Preserve the lexical file name so admission can reject a
                # symlink before any content read or unbounded identity hash.
                load_trimap_asset(self.base_dir / entry["asset"]["path"],
                                  expected_hw=(source["height"], source["width"]),
                                  expected_sha256=entry["asset"]["sha256"], check=check)
                continue
            if "asset" in entry and file_sha256(self.asset_path(entry), check=check) != entry["asset"]["sha256"]:
                raise ValueError(f"correction asset hash mismatch: {entry['id']}")
        for track in data.get('tracks', []):
            for row, observation in zip(track['frames'], track['observations'], strict=True):
                load_tracked_asset(self.base_dir / row['asset']['path'], expected_hw=shape,
                                   expected_sha256=row['asset']['sha256'], expected_mask_sha256=observation['derived_mask_sha256'], check=check)


def load_corrections(path, *, source=None, check=None):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        raw = stream.read(MAX_CORRECTION_BYTES + 1)
    if len(raw) > MAX_CORRECTION_BYTES:
        raise ValueError('correction document exceeds the 4MiB limit')
    try:
        data = json.loads(raw, object_pairs_hook=_unique_pairs, parse_float=finite_json_float, parse_constant=finite_json_float)
    except RecursionError as exc:
        raise ValueError('correction JSON nesting exceeds decoder limits') from exc
    document = CorrectionDocument.parse(data, path.parent)
    document.verify_assets(check=check)
    if source is not None and source_identity(source, check=check)["sha256"] != document.to_dict()["source"]["sha256"]:
        raise ValueError("correction source content hash mismatch")
    return document


def save_corrections(path, document):
    # Relocation rewrites asset paths; hashes/identity remain content-based.
    path = Path(path).resolve()
    data = document.to_dict()
    for asset in _assets(data):
        asset['path'] = os.path.relpath((document.base_dir / asset['path']).resolve(), path.parent)
    if len((json.dumps(data, indent=2, allow_nan=False) + '\n').encode()) > MAX_CORRECTION_BYTES:
        raise ValueError('correction document exceeds the 4MiB limit')
    write_json_atomic(path, data)


def invalidated_ranges(before, after):
    """Every changed correction conservatively invalidates its whole authored shot."""
    old, new = before.to_dict(), after.to_dict()
    if old["source"] != new["source"] or old["shots"] != new["shots"]:
        return [(0, max(old["source"]["frame_count"], new["source"]["frame_count"]))]
    first = {e["id"]: e for e in old["entries"]}
    second = {e["id"]: e for e in new["entries"]}
    changed = {key for key in first.keys() | second.keys() if first.get(key) != second.get(key)}
    old_guidance = {origin['id']: origin for origin in old.get('guidance', [])}
    new_guidance = {origin['id']: origin for origin in new.get('guidance', [])}
    changed_guidance = {key for key in old_guidance.keys() | new_guidance.keys()
                        if old_guidance.get(key) != new_guidance.get(key)}
    changed.update(table[key]['entry_id'] for table in (old_guidance, new_guidance)
                   for key in changed_guidance if key in table)
    old_tracks = {track['id']: track for track in old.get('tracks', [])}
    new_tracks = {track['id']: track for track in new.get('tracks', [])}
    changed_tracks = {key for key in old_tracks.keys() | new_tracks.keys() if old_tracks.get(key) != new_tracks.get(key)}
    # Reordering changes composition even if entry bodies are unchanged.
    common = first.keys() & second.keys()
    if [e["id"] for e in old["entries"] if e["id"] in common] != [e["id"] for e in new["entries"] if e["id"] in common]:
        changed |= first.keys() | second.keys()
    ranges = {tuple(shot) for table in (first, second) for key in changed if key in table
              for shot in new['shots'] if shot[0] <= table[key]['frame'] < shot[1]}
    ranges.update(tuple(shot) for table in (old_tracks, new_tracks) for key in changed_tracks if key in table
                  for shot in new['shots'] if table[key]['range'][0] < shot[1] and shot[0] < table[key]['range'][1])
    return sorted(ranges)


class CorrectionHistory:
    """Immutable document snapshots; UI can persist selected revisions with save."""
    def __init__(self, document, *, max_snapshots: int | None = None):
        if max_snapshots is not None and (type(max_snapshots) is not int or max_snapshots < 2):
            raise ValueError('history max_snapshots must be null or an integer >=2')
        self.max_snapshots = max_snapshots
        self._documents, self._cursor = [document], 0

    @property
    def current(self):
        return self._documents[self._cursor]

    def apply(self, document):
        if document.to_dict()["revision"] <= max(d.to_dict()["revision"] for d in self._documents):
            raise ValueError("new correction revisions must increase monotonically")
        invalidated = invalidated_ranges(self.current, document)
        self._documents[self._cursor + 1:] = [document]
        self._cursor += 1
        if self.max_snapshots is not None and len(self._documents) > self.max_snapshots:
            dropped = len(self._documents) - self.max_snapshots
            del self._documents[:dropped]
            self._cursor -= dropped
        return invalidated

    def undo(self):
        if self._cursor == 0:
            raise ValueError("no correction revision to undo")
        before = self.current
        self._cursor -= 1
        return invalidated_ranges(before, self.current)

    def redo(self):
        if self._cursor + 1 == len(self._documents):
            raise ValueError("no correction revision to redo")
        before = self.current
        self._cursor += 1
        return invalidated_ranges(before, self.current)
