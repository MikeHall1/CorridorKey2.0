"""Versioned local authored raster corrections and conservative dependencies."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .diagnostics import finite_json_float, write_json_atomic
from .identities import file_sha256, source_identity

MAX_CORRECTION_BYTES = 4 * 1024 * 1024


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
        if not isinstance(data, dict) or set(data) != {"schema_version", "revision", "source", "shots", "entries"}:
            raise ValueError("corrections require schema_version, revision, source, shots and entries only")
        if type(data["schema_version"]) is not int or data["schema_version"] not in (1, 2):
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
            if not isinstance(entry, dict) or set(entry) != (common if reset else common | {"asset", "strength", "direction", "range"}):
                raise ValueError("unknown or missing correction fields")
            if not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in seen:
                raise ValueError("correction IDs must be unique nonempty strings")
            seen.add(entry["id"])
            _integer(entry["revision"], "entry revision")
            _integer(entry["frame"], "entry frame")
            kinds = {"matte", "keep", "remove", "reset"}
            if data["schema_version"] == 2:
                kinds.add("trimap")
            if entry["frame"] >= source["frame_count"] or entry["kind"] not in kinds:
                raise ValueError("unsupported correction kind or frame outside selected timeline")
            if reset:
                continue
            asset = entry["asset"]
            if not isinstance(asset, dict) or set(asset) != {"path", "sha256"}:
                raise ValueError("correction asset requires local path and sha256")
            if not isinstance(asset["path"], str) or not asset["path"].strip() or "://" in asset["path"]:
                raise ValueError("correction assets must be local filesystem paths")
            _hash(asset["sha256"])
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
        return cls(canonical(data), Path(base_dir).resolve())

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def identity(self):
        data = self.to_dict()
        for entry in data["entries"]:
            if "asset" in entry:
                entry["asset"].pop("path")
        return hashlib.sha256(canonical(data).encode()).hexdigest()

    @property
    def frames(self):
        return tuple(sorted({entry["frame"] for entry in self.to_dict()["entries"]
                             if entry["kind"] == "reset" or entry["strength"] > 0}))

    @property
    def resets(self):
        return tuple(entry["frame"] for entry in self.to_dict()["entries"] if entry["kind"] == "reset")

    def asset_path(self, entry):
        return (self.base_dir / entry["asset"]["path"]).resolve()

    def verify_assets(self, *, check=None):
        data = self.to_dict()
        for entry in data["entries"]:
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
    import os
    for entry in data["entries"]:
        if "asset" in entry:
            entry["asset"]["path"] = os.path.relpath(document.asset_path(entry), path.parent)
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
    # Reordering changes composition even if entry bodies are unchanged.
    common = first.keys() & second.keys()
    if [e["id"] for e in old["entries"] if e["id"] in common] != [e["id"] for e in new["entries"] if e["id"] in common]:
        changed |= first.keys() | second.keys()
    return sorted({tuple(shot) for table in (first, second) for key in changed if key in table
                   for shot in new["shots"] if shot[0] <= table[key]["frame"] < shot[1]})


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
