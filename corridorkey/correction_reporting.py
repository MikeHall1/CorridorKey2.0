"""Bounded correction provenance for records; the local manifest is authoritative.

Never copy per-frame tracking observations into worker terminal messages. A
four-MiB valid manifest does not fit inside a four-MiB record envelope. Keep
small legacy detail fields, explicitly count/omit large ones, and leave room
for the runtime's fixed counters and optional refinement contract.
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .corrections import CorrectionDocument

MAX_CORRECTION_REPORT_BYTES = 64 * 1024
RUNTIME_RESERVE_BYTES = 4 * 1024


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _bounded_field(value: Any, limit: int) -> Any:
    return value if len(_encoded(value)) <= limit else None


def _track_summary(track: dict) -> dict:
    original = {'id': track['id'], 'owner_id': track['owner_id'], 'range': track['range'],
                'direction': track['direction'], 'binding_sha256': track['binding_sha256'],
                'anchor_count': len(track['anchor_ids']), 'frame_count': len(track['frames']),
                'observation_count': len(track['observations']),
                'approximation_count': sum(row['status'] == 'approximation' for row in track['observations'])}
    result = {key: _bounded_field(value, 256 if key in ('id', 'owner_id') else 512) for key, value in original.items()}
    omitted = [key for key, value in result.items() if value is None and original[key] is not None]
    if omitted:
        result['omitted_fields'] = omitted
        result['summary_sha256'] = hashlib.sha256(_encoded(original)).hexdigest()
    return result


def correction_report(document: CorrectionDocument, document_path, *, runtime_fields: dict | None = None) -> dict:
    """Copy finite metadata into a <=60-KiB snapshot, reserving 4 KiB for runtime.

    Unusually large IDs, revision numbers or paths are explicitly omitted too;
    the complete document identity and optional locator hash retain provenance.
    This summarizes observed documents, not model execution or visual quality.
    """
    data = document.to_dict()
    path = str(document_path)
    result: dict = {'identity': document.identity, 'revision': _bounded_field(data['revision'], 1024),
                    'document_path': _bounded_field(path, 8192),
                    'metadata_scope': 'Bounded correction metadata; full authored entries and per-frame tracking provenance remain in the local manifest. Identity binds its content, not its path.',
                    'entries_count': len(data['entries']), 'shots_count': len(data['shots']),
                    'authored_frames_count': len(document.frames), 'reset_frames_count': len(document.resets),
                    'tracks_count': len(data.get('tracks', [])),
                    'guidance_count': len(data.get('guidance', [])),
                    'refinement_region_count': sum(entry['kind'] == 'region' for entry in data['entries']),
                    'guidance_counts_by_kind': {kind: sum(origin['kind'] == kind for origin in data.get('guidance', []))
                                                for kind in ('clean_plate', 'subject_clicks')},
                    'guidance_scope': 'Generation observations, resource hashes and bindings remain in the manifest; final guidance is not proof of provider execution or improved matting quality.',
                    'tracks': [_track_summary(track) for track in data.get('tracks', [])],
                    'omitted_fields': ['tracks[].frames', 'tracks[].observations', 'tracks[].binding'] if data.get('tracks') else []}
    if data.get('guidance'):
        result['omitted_fields'].append('guidance')
    for key, value in (('document_path', path), ('revision', data['revision'])):
        if result[key] is None:
            result['omitted_fields'].append(key)
            result[key + '_sha256'] = hashlib.sha256(_encoded(value)).hexdigest()
    if runtime_fields:
        if result.keys() & runtime_fields.keys():
            raise ValueError('Runtime correction fields cannot replace provenance metadata.')
        result.update(runtime_fields)
    limit = MAX_CORRECTION_REPORT_BYTES - RUNTIME_RESERVE_BYTES
    fields = (('source', data['source']), ('shots', data['shots']),
              ('authored_frames', list(document.frames)), ('reset_frames', list(document.resets)),
              ('entries', data['entries']))
    for key, value in fields:
        if len(_encoded(value)) <= limit and len(_encoded({**result, key: value})) <= limit - 256:
            result[key] = value
        else:
            result['omitted_fields'].append(key)
    if len(_encoded(result)) > limit:
        raise ValueError('Correction diagnostic metadata exceeds its explicit 64 KiB budget.')
    return result
